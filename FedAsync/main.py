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

# DATA_TYPE = "MNIST"
# MODEL_NAME = "FC3"
DATA_TYPE = "FashionMNIST"
MODEL_NAME = "FC3"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "VGG13"

RESULT_FILE_NAME = "FashionMNIST"

EPOCHS = 3000

PRE_TRAIN_EPOCH = 0

CLIENT_NUMBER = 1000
USE_CLIENT_NUMBER = 1000

USE_IID_CLIENTS = 2

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
E = 5
R_RHO = 7
SCHEDULER_INTERVAL = 30
CHECK_IN_INTERVAL = 600  # 200
CHECK_IN_NUM = 200  # 100

CLIENT_STALENESS_SETTING = [2, 128, 63, 40]  # lower, upper, mu, sigma

# USE_GPU_ID = [0, 1, 2, 3, 4, 5]
USE_GPU_ID = [0]

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

    client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    client_staleness_list = Tools.generate_normal_distribution_list(CLIENT_STALENESS_SETTING[0],
                                                                    CLIENT_STALENESS_SETTING[1],
                                                                    CLIENT_STALENESS_SETTING[2],
                                                                    CLIENT_STALENESS_SETTING[3], USE_CLIENT_NUMBER)

    for pe in range(PRE_TRAIN_EPOCH):
        global_network.model_train_one_epoch(x, y, BATCH_SIZE)
        accuracy, loss = global_network.evaluate_network(x_test, y_test)
        print("Pre-train Epoch", pe, ":  accuracy =", accuracy, ", loss =", loss)
    pre_trained_global_weights = copy.deepcopy(global_network.get_init_weights())

    non_iid_client_index = []
    for i in range(USE_CLIENT_NUMBER):
        if i < (USE_CLIENT_NUMBER / 2):
            non_iid_client_index.append(i)
        else:
            if i % 100 < 150:
                non_iid_client_index.append(i)
    print(non_iid_client_index)

    accuracy_lists = []
    loss_lists = []

    clustering_result_lists = []

    alpha_list = [1, 0.1, 0.1, 0.1]
    s_setting_list = [("Constant", 0, 0),
                      ("Constant", 0, 0), ("Polynomial", 0.5, 0), ("Hinge", 0.1, 60)]
    protocol_list = ["ClusteredFedAsync",
                     "FedAsync", "FedAsync", "FedAsync"]
    curve_name_list = ["CAFL",
                       "FedAsync+Const", "FedAsync+Poly", "FedAsync+Hinge"]
    use_scheduling_strategy_list = [2, 0, 0, 0]
    rho_list = [0, 0, 0, 0]

    for i in range(len(alpha_list)):
        for e in range(EPOCHS):
            ALPHA[e] = alpha_list[i]

        global_network.set_weights(copy.deepcopy(pre_trained_global_weights))
        for client in client_list:
            client.set_client_weights(copy.deepcopy(pre_trained_global_weights))

        non_iid_client_list = []
        for n_i_c_i in non_iid_client_index:
            non_iid_client_list.append(client_list[n_i_c_i])

        s_setting = s_setting_list[i]
        protocol = protocol_list[i]
        use_scheduling_strategy = use_scheduling_strategy_list[i]
        print(protocol, ":")

        async_server = AsyncServer.AsyncServer(protocol, use_scheduling_strategy, DATA_TYPE, MODEL_NAME, CLIENT_RATIO,
                                               non_iid_client_list, [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E,
                                               rho_list[i], EPOCHS, ALPHA, SCHEDULER_INTERVAL, CHECK_IN_INTERVAL,
                                               CHECK_IN_NUM, global_network, init_weights, s_setting,
                                               client_staleness_list)

        async_server.run()
        print("")

        accuracy_list, loss_list = async_server.get_accuracy_and_loss_list()
        accuracy_lists.append(accuracy_list)
        loss_lists.append(loss_list)

        clustering_result_list = async_server.get_clustering_result_list()
        clustering_result_lists.append(clustering_result_list)

        del non_iid_client_list
        del async_server

        print("Thread count =", threading.activeCount())
        print(*threading.enumerate(), sep="\n")

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    for clustering_result_list in clustering_result_lists:
        for i in range(len(clustering_result_list)):
            print(i, ":", clustering_result_list[i])
        print("\n---------------------------------------------------------------------------------------------------\n")

    ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS, len(alpha_list), curve_name_list, accuracy_lists, loss_lists)
