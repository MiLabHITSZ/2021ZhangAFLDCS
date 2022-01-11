import os
import datetime
import random
import time
import threading
import copy
import pickle

import tensorflow as tf
import numpy as np

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.CNN as CNN
import Models.VGG13 as VGG13
import Models.VGG16 as VGG16
import platform

import DensityPeaks.ClusterWithDensityPeaks as ClusterWithDensityPeaks

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# DATA_TYPE = "MNIST"
# MODEL_NAME = "FC3"
DATA_TYPE = "FashionMNIST"
MODEL_NAME = "FC3"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "CNN"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "VGG13"

RESULT_FILE_NAME = "TestDPC_t"

if platform.system().lower() == 'windows':
    CLIENTS_WEIGHTS_PACKAGE_PATH = 'C:/Users/4444/PycharmProjects/Multi_server_federated_learning/clients_weights'
elif platform.system().lower() == 'linux':
    CLIENTS_WEIGHTS_PACKAGE_PATH = '/home/zrz/codes/Multi_server_federated_learning/clients_weights'
else:
    CLIENTS_WEIGHTS_PACKAGE_PATH = 'C:/Users/4444/PycharmProjects/Multi_server_federated_learning/clients_weights'

MODE = 1

T_STEP = 0.01

# NON_IID_SETTINGS_LIST = [1, 2, 4]
NON_IID_SETTINGS_LIST = [2]
# DATASET_LIST = ["MNIST", "FashionMNIST"]
DATASET_LIST = ["FashionMNIST"]
# DATASET_LIST = ["CIFAR10"]
DPC_SETTING_LIST = [("multidimensional_point", -2, -0.5, 3.2),
                    ("multidimensional_point", -4, -0.5, 3.2),
                    ("multidimensional_point", -6, -0.5, 3.2)]

NON_IID_SETTINGS_MAX = {1: 20, 2: 12, 4: 18}
T_MAX_DICT = {(1, "MNIST", ("multidimensional_point", -2, -0.5, 3.2)): 11.4,
              (1, "MNIST", ("multidimensional_point", -4, -0.5, 3.2)): 7.5,
              (1, "MNIST", ("multidimensional_point", -6, -0.5, 3.2)): 5.0,
              (1, "FashionMNIST", ("multidimensional_point", -2, -0.5, 3.2)): 18.4,
              (1, "FashionMNIST", ("multidimensional_point", -4, -0.5, 3.2)): 12.7,
              (1, "FashionMNIST", ("multidimensional_point", -6, -0.5, 3.2)): 8.8,
              (1, "CIFAR10", ("multidimensional_point", -2, -0.5, 3.2)): 13,  # 15.9,
              (1, "CIFAR10", ("multidimensional_point", -4, -0.5, 3.2)): 8.9,  # 10.5,
              (1, "CIFAR10", ("multidimensional_point", -6, -0.5, 3.2)): 6.85,  # 8.1,
              (2, "MNIST", ("multidimensional_point", -2, -0.5, 3.2)): 6.7,
              (2, "MNIST", ("multidimensional_point", -4, -0.5, 3.2)): 4.5,  # 4.6,
              (2, "MNIST", ("multidimensional_point", -6, -0.5, 3.2)): 3.1,
              (2, "FashionMNIST", ("multidimensional_point", -2, -0.5, 3.2)): 9.19,  # 10.3,
              (2, "FashionMNIST", ("multidimensional_point", -4, -0.5, 3.2)): 5.99,  # 6.2,
              (2, "FashionMNIST", ("multidimensional_point", -6, -0.5, 3.2)): 4.2,
              (2, "CIFAR10", ("multidimensional_point", -2, -0.5, 3.2)): 6.4,
              (2, "CIFAR10", ("multidimensional_point", -4, -0.5, 3.2)): 3.85,  # 4.2,
              (2, "CIFAR10", ("multidimensional_point", -6, -0.5, 3.2)): 2.85,  # 3.2,
              (4, "MNIST", ("multidimensional_point", -2, -0.5, 3.2)): 10.4,
              (4, "MNIST", ("multidimensional_point", -4, -0.5, 3.2)): 6.9,
              (4, "MNIST", ("multidimensional_point", -6, -0.5, 3.2)): 4.6,
              (4, "FashionMNIST", ("multidimensional_point", -2, -0.5, 3.2)): 16.7,
              (4, "FashionMNIST", ("multidimensional_point", -4, -0.5, 3.2)): 11.7,
              (4, "FashionMNIST", ("multidimensional_point", -6, -0.5, 3.2)): 8.2,
              (4, "CIFAR10", ("multidimensional_point", -2, -0.5, 3.2)): 10.5,  # 11.4,
              (4, "CIFAR10", ("multidimensional_point", -4, -0.5, 3.2)): 6.5,  # 6.8,
              (4, "CIFAR10", ("multidimensional_point", -6, -0.5, 3.2)): 4.4}

if DATA_TYPE == "CIFAR10":
    CLIENT_NUMBER = 200
    USE_CLIENT_NUMBER = 200
else:
    CLIENT_NUMBER = 1000
    USE_CLIENT_NUMBER = 1000
TYPE_SIZE = int(CLIENT_NUMBER / 10)

BATCH_SIZE = 50
E = 5

USE_IID_CLIENTS = 1

# USE_GPU_ID = [1, 2, 3, 4, 5]
# USE_GPU_ID = [0]
USE_GPU_ID = [3]

if MODEL_NAME == "FC3":
    global_network = FC3.FC3(DATA_TYPE)
elif MODEL_NAME == "CNN":
    global_network = CNN.CNN(DATA_TYPE)
elif MODEL_NAME == "VGG13":
    global_network = VGG13.VGG13(DATA_TYPE)
elif MODEL_NAME == "VGG16":
    global_network = VGG16.VGG16(DATA_TYPE)
else:
    print("Unexpected dataset name!")
    global_network = FC3.FC3(DATA_TYPE)

if DATA_TYPE == "MNIST" or "FashionMNIST" or "CIFAR10" or "CIFAR100":
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)
else:
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)


def get_clients_pre_weights(dataset_name, non_iid_mode):
    save_path = os.path.join(CLIENTS_WEIGHTS_PACKAGE_PATH,
                             dataset_name + '_UIC' + str(non_iid_mode) + '_client_pre_weights_dict.npy')
    if dataset_name == "CIFAR10":
        model_name = "VGG13"
        global_model = VGG13.VGG13(dataset_name)
        client_number = 200
        client_size = int(50000 / client_number)
    else:
        model_name = "FC3"
        global_model = FC3.FC3(dataset_name)
        client_number = 1000
        client_size = int(50000 / client_number)
    server_model_weights = copy.deepcopy(global_model.get_weights())
    if os.path.isfile(save_path):
        print("Load clients weights:", save_path)
        client_pre_weights_dict = np.load(save_path, allow_pickle=True).item()
    else:
        client_pre_weights_dict = {}
        c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(dataset_name)

        client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, dataset_name, model_name, client_number,
                                             client_size, BATCH_SIZE, non_iid_mode, x, y)
        print("Length of client list:", len(client_list))

        # 所有 client 进行预训练
        for c_i in range(len(client_list)):
            print(c_i, " ")
            client_list[c_i].set_client_weights(server_model_weights)
            client_list[c_i].client_train_one_epoch(BATCH_SIZE, E)
            client_pre_weights_dict[c_i] = copy.deepcopy(client_list[c_i].get_client_weights())
        print("")

        np.save(save_path, client_pre_weights_dict)

    return server_model_weights, client_pre_weights_dict


def get_label(node_id, non_iid_setting_name, dataset_name):
    if non_iid_setting_name == 2:
        if dataset_name == "CIFAR10":
            label = int(node_id % 10)
        else:
            label = int(node_id % 10)
    else:
        if dataset_name == "CIFAR10":
            label = int(node_id / 20)
        else:
            label = int(node_id / 100)
    return label


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    correct_ratio_lists = []

    server_weights = copy.deepcopy(global_network.get_weights())

    if MODE == 1:
        del global_network
        for non_iid_setting in NON_IID_SETTINGS_LIST:
            non_iid_setting_score_list = []
            for dataset in DATASET_LIST:
                del server_weights
                dataset_score_list = []
                server_weights, client_pre_weights_dict = get_clients_pre_weights(dataset, non_iid_setting)
                for dpc_setting in DPC_SETTING_LIST:
                    print("non_iid_setting:", non_iid_setting, "dataset:", dataset, "dpc_setting:", dpc_setting)
                    layer_score_list = []

                    # 初始化客户端和 true_labels_list
                    client_id_list = [i for i in range(len(client_pre_weights_dict))]
                    if dataset == "CIFAR10":
                        use_client_number = 200
                    else:
                        use_client_number = 1000
                    use_client_id_list = random.sample(client_id_list, use_client_number)
                    use_client_id_list.sort()
                    true_labels_list = []
                    use_client_pre_weights_dict = {}
                    for i in client_id_list:
                        if i in use_client_id_list:
                            use_client_pre_weights_dict[i] = client_pre_weights_dict[i]
                            true_labels_list.append(get_label(i, non_iid_setting, dataset))
                        else:
                            true_labels_list.append(-1)

                    # 初始化 ClusterWithDensityPeaks
                    T = 1.0
                    cluster_with_density_peaks = ClusterWithDensityPeaks.ClusterWithDensityPeaks(
                        use_client_pre_weights_dict, server_weights, dpc_setting[0], dpc_setting[1], dpc_setting[2], T)

                    while T <= T_MAX_DICT[(non_iid_setting, dataset, dpc_setting)]:
                        # 重新设置 T
                        cluster_with_density_peaks.set_t(T)
                        cluster_with_density_peaks.add_nodes({})

                        # 进行聚类
                        clusters = cluster_with_density_peaks.clustering()
                        cluster_result = cluster_with_density_peaks.get_clusters()

                        # 将 cluster_result 按 cluster_id 重新排序
                        sorted_cluster_result = {}
                        extra_key_count = 10
                        for key in cluster_result:
                            re_clustered = False
                            # 寻找可用的 cluster_id
                            for node_id in cluster_result[key]:
                                sort_cluster_id = get_label(cluster_result[key][0], non_iid_setting, dataset)
                                if sort_cluster_id not in sorted_cluster_result:
                                    sorted_cluster_result[sort_cluster_id] = copy.deepcopy(cluster_result[key])
                                    re_clustered = True
                                    break
                            # 若无可用的 cluster_id，则增加 cluster_id
                            if not re_clustered:
                                sorted_cluster_result[extra_key_count] = copy.deepcopy(cluster_result[key])
                                extra_key_count += 1

                        # 根据聚类结果填写 pred_labels_list
                        pred_labels_list = []
                        for client_id in client_id_list:
                            # 在聚类结果中找到 client_id 并放入 pred_labels_list
                            cluster_id = -2
                            for key in sorted_cluster_result:
                                if client_id in sorted_cluster_result[key]:
                                    cluster_id = key
                                    break
                            pred_labels_list.append(cluster_id)

                        # 计算得分
                        # print("true_labels_list:", true_labels_list)
                        # print("pred_labels_list:", pred_labels_list)
                        accuracy, precision, recall, f1 = Tools.calculate_accuracy_precision_recall_f1(true_labels_list,
                                                                                                       pred_labels_list)
                        print("T =", T, ":", accuracy, precision, recall, f1)

                        layer_score_list.append(f1)

                        T += T_STEP

                    # 为 score_list 的尾部填充 0
                    # while T < NON_IID_SETTINGS_MAX[non_iid_setting]:
                    #     layer_score_list.append(0)
                    #     T += T_STEP

                    print(layer_score_list)

                    del cluster_with_density_peaks

                    dataset_score_list.append(layer_score_list)

                non_iid_setting_score_list.append(dataset_score_list)
                # 保存 dataset_score_list
                save_path = os.path.join(CLIENTS_WEIGHTS_PACKAGE_PATH,
                                         "Non_IID" + str(non_iid_setting) + "_" + dataset + "_list")
                np.save(save_path, dataset_score_list)

                del client_pre_weights_dict

            # 输出并保存图片
            ResultManager.draw_gradual_bars(non_iid_setting_score_list, "test", NON_IID_SETTINGS_MAX[non_iid_setting])

    elif MODE == 2:
        T = 1.0
        server_weights, client_pre_weights_dict = get_clients_pre_weights(DATA_TYPE, USE_IID_CLIENTS)
        cluster_with_density_peaks = ClusterWithDensityPeaks.ClusterWithDensityPeaks(
            client_pre_weights_dict, server_weights, "multidimensional_point", -6, -0.5, T)
        t_clustering_result_list = []
        for i in range(2600):
            cluster_with_density_peaks.set_t(T)
            cluster_with_density_peaks.add_nodes({})
            clusters = cluster_with_density_peaks.clustering()
            clustering_result = cluster_with_density_peaks.show_clusters()
            t_clustering_result = [T]
            t_clustering_result.extend(clustering_result)
            t_clustering_result_list.append(t_clustering_result)
            print("clustering_result:", clustering_result)

            T += 0.1
            print("T =", T)

        for t_c in t_clustering_result_list:
            print(t_c)
        print("-----------------------------------------------------------------------------------")
        for t_c in t_clustering_result_list:
            if t_c[1: len(t_c)] == [TYPE_SIZE, TYPE_SIZE, TYPE_SIZE, TYPE_SIZE, TYPE_SIZE,
                                    TYPE_SIZE, TYPE_SIZE, TYPE_SIZE, TYPE_SIZE, TYPE_SIZE]:
                print(t_c)
    else:
        pass

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # if MODE == 1:
    #     ResultManager.handle_result(RESULT_FILE_NAME, CLIENT_NUMBER - INIT_CLIENT_NUMBER, len(setting_list),
    #                                 ["DPC-2", "DPC-4", "DPC-6", ], correct_ratio_lists, correct_ratio_lists)
