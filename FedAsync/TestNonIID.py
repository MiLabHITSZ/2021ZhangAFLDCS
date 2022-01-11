import os
import datetime
import threading
import copy

import tensorflow as tf

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Models.FC3 as FC3
import Models.CNN as CNN
import Models.VGG13 as VGG13
import AsyncServer

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

DATA_TYPE = "FashionMNIST"
MODEL_NAME = "FC3"

RESULT_FILE_NAME = "TestNonIID"

EPOCHS = 5000

CLIENT_NUMBER = 1000

USE_IID_CLIENTS = False

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
E = 5
R_RHO = 7

SCHEDULER_INTERVAL = 80
CHECK_IN_INTERVAL = 1000
CHECK_IN_NUM = 1000
CLIENT_STALENESS_SETTING = [2, 128, 63, 40]  # lower, upper, mu, sigma

ALPHA = []
for e in range(EPOCHS):
    ALPHA.append(0)

if MODEL_NAME == "FC3":
    global_network = FC3.FC3(DATA_TYPE)
elif MODEL_NAME == "CNN":
    global_network = CNN.CNN(DATA_TYPE)
elif MODEL_NAME == "VGG13":
    global_network = VGG13.VGG13(DATA_TYPE)
else:
    print("Unexpected dataset name!")
    global_network = FC3.FC3(DATA_TYPE)
init_weights = copy.deepcopy(global_network.get_init_weights())

if DATA_TYPE == "MNIST" or "FashionMNIST" or "CIFAR10" or "CIFAR100":
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)
else:
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)

    client_staleness_list = Tools.generate_normal_distribution_list(CLIENT_STALENESS_SETTING[0],
                                                                    CLIENT_STALENESS_SETTING[1],
                                                                    CLIENT_STALENESS_SETTING[2],
                                                                    CLIENT_STALENESS_SETTING[3], CLIENT_NUMBER)

    accuracy_lists = []
    loss_lists = []


    # alpha_list = [0.1, 0.5, 0.9,
    #               0.1, 0.3, 0.5, 0.7, 0.9]
    # s_setting_list = [("Constant", 0, 0), ("Constant", 0, 0), ("Constant", 0, 0),
    #                   ("Constant", 0, 0), ("Constant", 0, 0), ("Constant", 0, 0), ("Constant", 0, 0), ("Constant", 0, 0)]
    # protocol_list = ["FedAsync", "FedAsync", "FedAsync",
    #                  "FedAsync", "FedAsync", "FedAsync", "FedAsync", "FedAsync"]
    # curve_name_list = ["FedAsync+Const0.1, IID", "FedAsync+Const0.5, IID", "FedAsync+Const0.9, IID",
    #                    "FedAsync+Const0.1, Non-IID", "FedAsync+Const0.3, Non-IID", "FedAsync+Const0.5, Non-IID", "FedAsync+Const0.7, Non-IID", "FedAsync+Const0.9, Non-IID"]
    # rho_list = [0, 0, 0,
    #             0, 0, 0, 0, 0]
    alpha_list = [0.1, 0.1]
    s_setting_list = [("Constant", 0, 0), ("Constant", 0, 0)]
    protocol_list = ["FedAsync", "FedAsync"]
    curve_name_list = ["FedAsync+Const, IID", "FedAsync+Const, Non-IID"]
    rho_list = [0, 0]

    for i in range(len(alpha_list)):
        # if i < 3:
        if i % 2 == 0:
            USE_IID_CLIENTS = 0
        else:
            USE_IID_CLIENTS = 1

        client_list = Tools.generate_clients([0], c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE, BATCH_SIZE,
                                             USE_IID_CLIENTS, x, y)

        for e in range(EPOCHS):
            ALPHA[e] = alpha_list[i]

        global_network.set_weights(copy.deepcopy(init_weights))
        for client in client_list:
            client.set_client_weights(copy.deepcopy(init_weights))

        s_setting = s_setting_list[i]
        protocol = protocol_list[i]
        print(protocol, ":")

        async_server = AsyncServer.AsyncServer(protocol, 0, DATA_TYPE, MODEL_NAME, CLIENT_RATIO,
                                               client_list, [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E, rho_list[i], EPOCHS,
                                               ALPHA, SCHEDULER_INTERVAL, CHECK_IN_INTERVAL, CHECK_IN_NUM,
                                               global_network, init_weights, s_setting, client_staleness_list)

        async_server.run()
        print("")

        accuracy_list, loss_list = async_server.get_accuracy_and_loss_list()
        accuracy_lists.append(accuracy_list)
        loss_lists.append(loss_list)

        del client_list
        del async_server

        print("Thread count =", threading.activeCount())
        print(*threading.enumerate(), sep="\n")

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS, len(alpha_list), curve_name_list, accuracy_lists, loss_lists)
