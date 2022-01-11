import random
from tqdm import tqdm
import copy
import time
import threading
from multiprocessing import Process, Queue

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import Models.CNN as CNN
import Datasets.MNIST as MNIST
import Datasets.CIFAR10 as CIFAR10
import DensityPeaks.ClusterWithDensityPeaks as ClusterWithDensityPeaks

import AsyncClient
import SchedulerThread
import UpdaterThread
import Time


class AsyncClientManager:
    def __init__(self, protocol, client_list, init_weights, q, b_s, e, rho, c_t, stop_event, client_staleness_list):
        self.protocol = protocol
        self.client_list = client_list
        self.init_weights = init_weights
        self.queue = q
        self.batch_size = b_s
        self.e = e
        self.rho = rho
        self.current_time = c_t
        self.stop_event = stop_event
        self.client_staleness_list = client_staleness_list

        self.pre_train_queue = Queue()
        self.client_pre_weights_dict = {}
        self.c_dict = {}
        self.cluster_client_dict = {}
        self.cluster_queue_size_dict = {}
        self.thread_lock = threading.Lock()
        self.check_in_thread_lock = threading.Lock()
        self.checked_in = False

        self.clustering_result_list = []

        # 初始化clients
        self.client_thread_list = []
        for i in range(len(self.client_list)):
            client_delay = self.client_staleness_list[i]
            self.client_thread_list.append(AsyncClient.AsyncClient(i, self.client_list[i],
                                                                   self.queue, self.pre_train_queue,
                                                                   self.batch_size, self.e, self.rho,
                                                                   self.stop_event, client_delay, False))

        self.checked_in_client_thread_list = []
        self.unchecked_in_client_thread_list = []
        self.checking_in_client_thread_id_list = []
        for i in range(len(self.client_thread_list)):
            # if i < 20:
            if i % 100 < 20:  # 140:
                self.checked_in_client_thread_list.append(self.client_thread_list[i])
            else:
                self.unchecked_in_client_thread_list.append(self.client_thread_list[i])

        # 启动checked in clients
        print("Start checked in clients:")
        for client_thread in self.checked_in_client_thread_list:
            client_thread.start()

        if self.protocol == "ClusteredFedAsync":
            # 让各个checked in client进行一次预训练以进行聚类
            self.pre_train_clients(self.checked_in_client_thread_list)

            self.cluster_with_density_peaks = None
            self.clusters = None

            print("Initial clustering:")
            self.receive_clients_pre_train_weights(len(self.checked_in_client_thread_list), True)
            # self.cluster_with_density_peaks = ClusterWithDensityPeaks.ClusterWithDensityPeaks(
            #     self.client_pre_weights_dict,
            #     self.init_weights,
            #     "multidimensional_point",
            #     4, 0.05632, 4.1)
            # self.clusters = self.cluster_with_density_peaks.clustering()
            # clustering_result = self.cluster_with_density_peaks.show_clusters()
            # t_clustering_result = [int(self.current_time.get_time())]
            # t_clustering_result.extend(clustering_result)
            # self.clustering_result_list.append(t_clustering_result)

            # 初始化c_dict和cluster_client_dict
            self.generate_c_dict_and_cluster_client_dict()

    def pre_train_clients(self, client_thread_list):
        print("Pre-train clients:")
        for s_client_thread in client_thread_list:
            # 预训练不使用正则化
            s_client_thread.set_rho(0)
            s_client_thread.set_e(5)
            s_client_thread.set_pre_training(True)

            # 将server的初始模型参数发给client
            s_client_thread.set_client_weight(self.init_weights)

            # 启动一次client线程
            s_client_thread.set_event()

        print("pre-train client_thread_list length:", len(client_thread_list))

    def receive_clients_pre_train_weights(self, receive_num, is_init_pre_train):
        new_nodes_dict = {}

        # 接各个client发回的c_id、模型参数和时间戳并存储
        for r in range(receive_num):
            while True:
                # 接收一个client发回的模型参数和时间戳
                if not self.pre_train_queue.empty():
                    (c_id, client_weights, time_stamp) = self.pre_train_queue.get()
                    self.client_pre_weights_dict[c_id] = client_weights
                    break
                else:
                    time.sleep(0.01)

            # 还原每个client的rho和pre_training
            received_client_thread = self.find_client_thread_by_c_id(c_id)
            received_client_thread.set_rho(self.rho)
            received_client_thread.set_pre_training(False)

            if not is_init_pre_train:
                self.thread_lock.acquire()
                # 更新checked_in_client_thread_list和checked_in_client_thread_list
                self.checked_in_client_thread_list.append(received_client_thread)
                self.unchecked_in_client_thread_list.remove(received_client_thread)
                print("[", len(self.checked_in_client_thread_list), len(self.unchecked_in_client_thread_list), "]")
                self.thread_lock.release()

            new_nodes_dict[c_id] = copy.deepcopy(self.client_pre_weights_dict[c_id])

            print("Received", r, "th pre-train:", c_id, "[", received_client_thread.get_delay(), "]", len(self.checked_in_client_thread_list))

        if self.protocol == "ClusteredFedAsync":
            if is_init_pre_train:
                self.cluster_with_density_peaks = ClusterWithDensityPeaks.ClusterWithDensityPeaks(
                    self.client_pre_weights_dict, self.init_weights, "multidimensional_point", -2, -0.5, 3.2)
            else:
                self.cluster_with_density_peaks.add_nodes(new_nodes_dict)
            self.clusters = self.cluster_with_density_peaks.clustering()
            self.cluster_with_density_peaks.set_t(3.2)
            clustering_result = self.cluster_with_density_peaks.show_clusters()
            t_clustering_result = [int(self.current_time.get_time())]
            t_clustering_result.extend(clustering_result)
            self.clustering_result_list.append(t_clustering_result)
            print("clustering_result:", clustering_result)

            # 更新c_dict和cluster_client_dict
            self.check_in_thread_lock.acquire()
            print("==update c_dict and cluster_client_dict: check_in_thread_lock, acquired")
            time.sleep(0.01)
            self.update_c_dict_and_cluster_client_dict()
            self.checked_in = True
            time.sleep(0.01)
            print("==check_in_thread_lock, released")
            self.check_in_thread_lock.release()

    def client_check_in(self, check_in_number):
        if len(self.unchecked_in_client_thread_list) > 0:
            self.thread_lock.acquire()
            if check_in_number >= len(self.unchecked_in_client_thread_list):
                print("| remain---------------------------------------------------------------------", check_in_number)
                check_in_number = len(self.unchecked_in_client_thread_list)
            check_in_clients = random.sample(self.unchecked_in_client_thread_list, check_in_number)

            # 去除已经在checking in的clients
            cc = 0
            while cc < len(check_in_clients):
                cc_id = int(check_in_clients[cc].get_client_id())
                if cc_id in self.checking_in_client_thread_id_list:
                    check_in_clients.remove(check_in_clients[cc])
                else:
                    cc += 1

            # 启动received_client_thread
            for c_i_client in check_in_clients:
                print("Start client", c_i_client.get_client_id())
                c_i_client.start()
                self.checking_in_client_thread_id_list.append(int(c_i_client.get_client_id()))
            # for c_i_client in check_in_clients:
            #     c_i_client.join()
            self.thread_lock.release()

            if self.protocol == "FedAsync":
                self.thread_lock.acquire()
                # 更新checked_in_client_thread_list和checked_in_client_thread_list
                for c_i_client in check_in_clients:
                    self.checked_in_client_thread_list.append(c_i_client)
                    self.unchecked_in_client_thread_list.remove(c_i_client)
                    print("[", len(self.checked_in_client_thread_list), len(self.unchecked_in_client_thread_list), "]")
                self.thread_lock.release()
            elif self.protocol == "ClusteredFedAsync":
                # 预训练check in clients
                self.pre_train_clients(check_in_clients)
            else:
                print("Unexpected protocol!!!!")
        else:
            check_in_number = 0

        return check_in_number

    def stop_all_clients(self):
        # 终止所有client线程
        self.stop_event.set()
        for client_threads in self.client_thread_list:
            client_threads.set_event()

    def set_client_thread_list(self, new_client_thread_list):
        self.thread_lock.acquire()
        self.client_thread_list = new_client_thread_list
        self.thread_lock.release()

    def get_client_thread_list(self):
        self.thread_lock.acquire()
        client_thread_list = self.client_thread_list
        self.thread_lock.release()
        return client_thread_list

    def find_client_thread_by_c_id(self, c_id):
        self.thread_lock.acquire()
        target_client_thread = None
        for client_thread in self.client_thread_list:
            if client_thread.get_client_id() == c_id:
                target_client_thread = client_thread
        self.thread_lock.release()
        return target_client_thread

    def set_checked_in_client_thread_list(self, new_checked_in_client_thread_list):
        self.thread_lock.acquire()
        self.checked_in_client_thread_list = new_checked_in_client_thread_list
        self.thread_lock.release()

    def get_checked_in_client_thread_list(self):
        self.thread_lock.acquire()
        checked_in_client_thread_list = self.checked_in_client_thread_list
        self.thread_lock.release()
        return checked_in_client_thread_list

    def get_unchecked_in_client_thread_list_len(self):
        self.thread_lock.acquire()
        unchecked_in_client_thread_list_len = len(self.unchecked_in_client_thread_list)
        self.thread_lock.release()
        return unchecked_in_client_thread_list_len

    def generate_c_dict_and_cluster_client_dict(self):
        self.c_clear()
        self.cluster_client_dict_clear()
        for cluster in self.clusters:
            cluster_id = cluster[0].get_cluster_id()
            self.c_add(cluster_id, 0)

            self.cluster_client_dict_add(cluster_id)
            for node in cluster:
                self.cluster_client_dict_append(cluster_id, node.get_node_id(), 0)
            # print("cluster_client_dict:", self.cluster_client_dict[cluster_id])

    def update_c_dict_and_cluster_client_dict(self):
        old_cluster_client_dict = self.copy_cluster_client_dict()
        self.c_clear()
        self.cluster_client_dict_clear()

        for cluster in self.clusters:
            cluster_id = cluster[0].get_cluster_id()
            self.c_add(cluster_id, 0)
            self.cluster_client_dict_add(cluster_id)

            for node in cluster:
                # 找出node已有的c值，若该node为首次出现，则c = 0
                node_id = node.get_node_id()
                node_c = 0
                for key in old_cluster_client_dict:
                    if node_id in old_cluster_client_dict[key]:
                        node_c = old_cluster_client_dict[key][node_id]
                        break

                self.c_update(cluster_id, (self.c_get(cluster_id) + node_c))
                self.cluster_client_dict_append(cluster_id, node.get_node_id(), node_c)

            # print("cluster_client_dict:", self.cluster_client_dict[cluster_id])

    def get_check_in_thread_lock(self):
        return self.check_in_thread_lock

    def set_checked_in(self, new_checked_in):
        self.thread_lock.acquire()
        self.checked_in = new_checked_in
        self.thread_lock.release()

    def get_checked_in(self):
        self.thread_lock.acquire()
        checked_in = self.checked_in
        self.thread_lock.release()
        return checked_in

    def get_clustering_result_list(self):
        return self.clustering_result_list

    def c_add(self, k, v):
        self.thread_lock.acquire()
        self.c_dict[k] = v
        self.thread_lock.release()

    def c_update(self, k, v):
        self.thread_lock.acquire()
        self.c_dict[k] = v
        self.thread_lock.release()

    def c_get(self, k):
        self.thread_lock.acquire()
        ck = self.c_dict[k]
        self.thread_lock.release()
        return ck

    def c_get_sum(self):
        return sum(self.c_dict.values())

    def c_clear(self):
        self.thread_lock.acquire()
        self.c_dict.clear()
        self.thread_lock.release()

    def copy_c(self):
        self.thread_lock.acquire()
        c_c_dict = copy.deepcopy(self.c_dict)
        self.thread_lock.release()
        return c_c_dict

    def cluster_client_dict_add(self, k):
        self.thread_lock.acquire()
        self.cluster_client_dict[k] = {}
        self.thread_lock.release()

    def cluster_client_dict_append(self, k, kk, v):
        self.thread_lock.acquire()
        self.cluster_client_dict[k][kk] = v
        self.thread_lock.release()

    def cluster_client_dict_update_one(self, k, kk):
        self.thread_lock.acquire()
        self.cluster_client_dict[k][kk] += 1
        self.thread_lock.release()

    def cluster_client_dict_get(self, k):
        self.thread_lock.acquire()
        client_dict = self.cluster_client_dict[k]
        self.thread_lock.release()
        return client_dict

    def cluster_client_dict_client_get(self, k, kk):
        self.thread_lock.acquire()
        c_n = self.cluster_client_dict[k][kk]
        self.thread_lock.release()
        return c_n

    def cluster_client_dict_find(self, kk):
        cluster_id = -1
        self.thread_lock.acquire()
        for key in self.cluster_client_dict:
            if kk in self.cluster_client_dict[key]:
                cluster_id = key
                break
        self.thread_lock.release()
        return cluster_id

    def cluster_client_dict_clear(self):
        self.thread_lock.acquire()
        self.cluster_client_dict.clear()
        self.thread_lock.release()

    def copy_cluster_client_dict(self):
        self.thread_lock.acquire()
        c_cluster_client_dict = copy.deepcopy(self.cluster_client_dict)
        self.thread_lock.release()
        return c_cluster_client_dict

    def get_cluster_client_dict_sum_len(self):
        self.thread_lock.acquire()
        sum_len = 0
        for k in self.cluster_client_dict:
            sum_len += len(self.cluster_client_dict[k])
        self.thread_lock.release()
        return sum_len

    def get_cluster_client_dict_len(self, k):
        self.thread_lock.acquire()
        client_dict_len = len(self.cluster_client_dict[k])
        self.thread_lock.release()
        return client_dict_len

    def set_cluster_queue_size_dict(self, new_set_cluster_queue_size_dict):
        self.thread_lock.acquire()
        self.cluster_queue_size_dict.clear()
        self.cluster_queue_size_dict = new_set_cluster_queue_size_dict
        self.thread_lock.release()

    def get_cluster_queue_size_dict(self):
        self.thread_lock.acquire()
        cluster_queue_size_dict = copy.deepcopy(self.cluster_queue_size_dict)
        self.thread_lock.release()
        return cluster_queue_size_dict
