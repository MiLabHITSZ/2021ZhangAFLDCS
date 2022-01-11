import threading
import time
import numpy
import random
import copy
from multiprocessing import Process, Queue

import tensorflow as tf
import utils.Tools as Tools


class UpdaterThread(threading.Thread):
    def __init__(self, protocol, queue, server_thread_lock, alpha, t, c_t, g_n, t_d, async_client_manager, s_e, s_s):
        threading.Thread.__init__(self)
        self.protocol = protocol
        self.queue = queue
        self.server_thread_lock = server_thread_lock
        self.alpha = alpha
        self.T = t
        self.current_time = c_t
        self.server_network = g_n
        self.test_data = t_d
        self.async_client_manager = async_client_manager
        self.stop_event = s_e
        self.s_setting = s_s

        self.cluster_queue_dict = {}

        self.check_in_thread_lock = self.async_client_manager.get_check_in_thread_lock()

        self.event = threading.Event()
        self.event.clear()

        self.sum_delay = 0

        self.accuracy_list = []
        self.loss_list = []

    def run(self):
        # 初始化各个cluster的参数容器
        c_dict = self.async_client_manager.copy_c()
        for key in c_dict:
            self.cluster_queue_dict[key] = Queue()

        last_accuracy = 0.1
        last_loss = 2.3

        for epoch in range(self.T):
            # self.check_in_thread_lock.acquire()
            while True:
                self.check_in_thread_lock.acquire()
                c_r = 0
                # 接收一个client发回的模型参数和时间戳
                if not self.queue.empty():
                    (c_id, client_weights, time_stamp) = self.queue.get()
                    self.sum_delay += (self.current_time.get_time() - time_stamp)
                    print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                          self.current_time.get_time(), "| queue size = ", self.queue.qsize())

                    if self.protocol == "ClusteredFedAsync":
                        # 若有新client checked in，则先更新cluster_queue_dict
                        if self.async_client_manager.get_checked_in():
                            print("update_cluster_queue_dict!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            self.update_cluster_queue_dict()
                            self.async_client_manager.set_checked_in(False)

                        # 将该client所属的cluster的c值加1，同时求出c/C，将其weights存入其所在cluster的参数容器
                        cluster_id = self.async_client_manager.cluster_client_dict_find(c_id)
                        print("Updater received: c_id = ", c_id, "cluster_id = ", cluster_id)
                        self.async_client_manager.c_update(cluster_id, (self.async_client_manager.c_get(cluster_id) + 1))
                        self.async_client_manager.cluster_client_dict_update_one(cluster_id, c_id)

                        c_r = float(self.async_client_manager.c_get(cluster_id) / self.async_client_manager.c_get_sum())

                        # 更新async_client_manager中的cluster_client_size_dict
                        cluster_client_size_dict = {}
                        for key in self.cluster_queue_dict:
                            cluster_client_size_dict[key] = self.cluster_queue_dict[key].qsize()
                        self.async_client_manager.set_cluster_queue_size_dict(cluster_client_size_dict)

                        self.cluster_queue_dict[cluster_id].put((c_id, client_weights, time_stamp))

                    self.event.set()
                else:
                    (c_id, client_weights, time_stamp) = (0, [], 0)

                if self.event.is_set():
                    if self.protocol == "ClusteredFedAsync":
                        # 每当server集齐所有cluster的client返回的更新，就进行合并，并更新全局模型
                        collected = True
                        for key in self.cluster_queue_dict:
                            if self.cluster_queue_dict[key].empty():
                                collected = False
                                break

                        if not collected:
                            print("Not Collected,", end=" ")
                            print("Len of cluster_queue_dict:", len(self.cluster_queue_dict))
                            # for key in self.cluster_queue_dict:
                            #     print(self.cluster_queue_dict[key].qsize(), end=" | ")
                            # print("----------------", end=" | ")

                        if collected:
                            cluster_weights_list = []
                            cluster_time_stamp_list = []
                            print("")
                            for key in self.cluster_queue_dict:
                                (c_id, cluster_weights, cluster_time_stamp) = self.cluster_queue_dict[key].get()

                                # 用每个cluster中当前client的数量与当前所有client数量的比值做为cluster_weights的权重
                                cluster_id = self.async_client_manager.cluster_client_dict_find(c_id)
                                cluster_len = self.async_client_manager.get_cluster_client_dict_len(cluster_id)
                                sum_len = self.async_client_manager.get_cluster_client_dict_sum_len()
                                # weight = float(cluster_len) / float(sum_len)
                                # weight = 0.1
                                # print(round(weight, 2), end=",")

                                cluster_weights_list.append(cluster_weights)
                                # cluster_weights_list.append(Tools.weight_nd_array_list(cluster_weights, weight))

                                cluster_time_stamp_list.append(cluster_time_stamp)
                            print("")
                            print("Collected,", len(cluster_weights_list))
                            for key in self.cluster_queue_dict:
                                print(self.cluster_queue_dict[key].qsize(), end=" | ")
                            print("----------------", end=" | ")
                            sum_weights = Tools.sum_nd_array_lists(cluster_weights_list)
                            merged_weights = Tools.avg_nd_array_list(sum_weights, len(cluster_weights_list))
                            # merged_weights = Tools.sum_nd_array_lists(cluster_weights_list)
                            merged_time_stamp = numpy.mean(cluster_time_stamp_list)
                            print("merged_time_stamp =", merged_time_stamp, "-", self.current_time.get_time(), "=",
                                  merged_time_stamp - self.current_time.get_time())

                            # # 更新async_client_manager中的cluster_client_size_dict
                            # cluster_client_size_dict = {}
                            # for key in self.cluster_queue_dict:
                            #     cluster_client_size_dict[key] = self.cluster_queue_dict[key].qsize()
                            # self.async_client_manager.set_cluster_queue_size_dict(cluster_client_size_dict)

                            self.server_thread_lock.acquire()
                            print("==update_server_weights: server_thread_lock, acquired")
                            time.sleep(0.01)
                            self.update_server_weights(merged_weights, merged_time_stamp, self.alpha[epoch],
                                                       self.s_setting, c_r)
                            time.sleep(0.01)
                            print("==server_thread_lock, released")
                            self.server_thread_lock.release()

                            # 计算并记录accuracy和loss
                            self.server_thread_lock.acquire()
                            print("==run_server_test: server_thread_lock, acquired")
                            time.sleep(0.01)
                            last_accuracy, last_loss = self.run_server_test(epoch)
                            time.sleep(0.01)
                            print("==server_thread_lock, released")
                            self.server_thread_lock.release()
                        else:
                            # 记录accuracy和loss
                            self.accuracy_list.append(last_accuracy)
                            self.loss_list.append(last_loss)
                            print('Epoch(t):', epoch, 'accuracy:', last_accuracy, 'loss:', float(last_loss), self.s_setting)
                        # self.server_thread_lock.acquire()
                        # self.run_server_test(epoch)
                        # self.server_thread_lock.release()
                    else:
                        # 使用接收的client发回的模型参数和时间戳对全局模型进行更新
                        self.server_thread_lock.acquire()
                        self.update_server_weights(client_weights, time_stamp, self.alpha[epoch], self.s_setting, c_r)
                        self.run_server_test(epoch)
                        self.server_thread_lock.release()

                    self.event.clear()
                    self.check_in_thread_lock.release()
                    time.sleep(0.01)
                    break
                else:
                    # self.event.wait()  # 等待标志位设定
                    self.check_in_thread_lock.release()
                    time.sleep(0.01)

            # self.check_in_thread_lock.release()
            self.current_time.time_add()
            time.sleep(0.01)

        # 关闭所有cluster_queue
        if self.protocol == "ClusteredFedAsync":
            c_dict = self.async_client_manager.copy_c()
            for key in c_dict:
                print("Queue", key, "has", self.cluster_queue_dict[key].qsize(), "left")
                if not self.cluster_queue_dict[key].empty():
                    for q in range(self.cluster_queue_dict[key].qsize()):
                        self.cluster_queue_dict[key].get()
                self.cluster_queue_dict[key].close()

        print("Average delay =", (self.sum_delay / self.T), "|", self.protocol)

        # 终止所有client线程
        # print("----------------------------------------------------------------------------stop")
        self.async_client_manager.stop_all_clients()
        # print("----------------------------------------------------------------------------stopped")
        # print("Thread count =", threading.activeCount())
        # print(*threading.enumerate(), sep="\n")

    def update_cluster_queue_dict(self):
        # 将原cluster_queue_dict转换为cluster_list_dict以存储其中的数据，然后将cluster_queue_dict清空
        c_sum = 0
        cluster_list_dict = {}
        for key in self.cluster_queue_dict:
            cluster_list_dict[key] = []
            # print(self.cluster_queue_dict[key].qsize())
            for i in range(1000):
                if not self.cluster_queue_dict[key].empty():
                    cluster_list_dict[key].append(self.cluster_queue_dict[key].get())
            # while True:
            #     if self.cluster_queue_dict[key].empty():
            #         print("--", self.cluster_queue_dict[key].empty())
            #         break
            #     else:
            #         cluster_tuple = self.cluster_queue_dict[key].get()
            #         cluster_list_dict[key].append(cluster_tuple)
            # print(self.cluster_queue_dict[key].empty(), self.cluster_queue_dict[key].qsize())
            c_sum += len(cluster_list_dict[key])

        # 为新增加的cluster初始化参数容器，并删除已经不存在的cluster的参数容器
        for k in self.cluster_queue_dict:
            for _ in range(self.cluster_queue_dict[k].qsize()):
                self.cluster_queue_dict[k].get()
            self.cluster_queue_dict[k].close()
        self.cluster_queue_dict.clear()
        c_dict = self.async_client_manager.copy_c()
        for key in c_dict:
            if key not in self.cluster_queue_dict.keys():
                self.cluster_queue_dict[key] = Queue()

        # 生成新的cluster_queue_dict
        for key in cluster_list_dict:
            for c_tuple in cluster_list_dict[key]:
                new_cluster_id = self.async_client_manager.cluster_client_dict_find(c_tuple[0])
                self.cluster_queue_dict[new_cluster_id].put(c_tuple)

        cc_sum = 0
        for key in self.cluster_queue_dict:
            cc_sum += self.cluster_queue_dict[key].qsize()

        print(c_sum, "=", cc_sum)

    def update_server_weights(self, client_weights, time_stamp, alpha, s_setting, c_r):
        (s_type, a, b) = (s_setting[0], s_setting[1], s_setting[2])
        if s_type == "Constant":
            s = 1
        elif s_type == "Polynomial":
            s = float(1 / ((self.current_time.get_time() - time_stamp + 1) ** a))
        elif s_type == "Hinge":
            if (self.current_time.get_time() - time_stamp) <= b:
                s = 1
            else:
                s = float(1 / ((a * (self.current_time.get_time() - time_stamp - b)) + 1))
        else:
            s = 1
            print("Error in s-type!!!!")

        if self.protocol == "ClusteredFedAsync":
            # k = 1
            # r = k * (1 - c_r)
            r = 1
        else:
            r = 1
        alpha = alpha * s * r
        server_weights = copy.deepcopy(self.server_network.get_weights())
        updated_weights = []
        for w in range(len(server_weights)):
            updated_weights.append(((1 - alpha) * server_weights[w]) + (alpha * client_weights[w]))

        self.server_network.set_weights(updated_weights)

    def run_server_test(self, epoch):
        with tf.device('/gpu:0'):
            accuracy, loss = self.server_network.evaluate_network(self.test_data[0], self.test_data[1])
        self.accuracy_list.append(accuracy)
        self.loss_list.append(loss)
        print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss:', float(loss), self.s_setting)
        return accuracy, loss

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list
