import threading
import time
import random
import numpy
import copy
import utils.Tools as Tools


class SchedulerThread(threading.Thread):
    def __init__(self, protocol, use_scheduling_strategy, server_thread_lock, s_i, c_i_i, async_client_manager, q,
                 c_ratio, c_t, g_n, t, s_s):
        threading.Thread.__init__(self)
        self.protocol = protocol
        self.use_scheduling_strategy = use_scheduling_strategy
        self.server_thread_lock = server_thread_lock
        self.schedule_interval = s_i
        self.check_in_interval = c_i_i
        self.async_client_manager = async_client_manager
        self.queue = q
        self.c_ratio = c_ratio
        self.current_t = c_t
        self.server_network = g_n
        self.T = t
        self.s_setting = s_s

    def run(self):
        last_s_time = -1
        # last_c_time = -1
        while self.current_t.get_time() < self.T:
            current_time = self.current_t.get_time()

            # 每隔一段时间进行一次schedule
            if current_time % self.schedule_interval == 0 and current_time != last_s_time:
                print("| current_time |", current_time % self.schedule_interval, "= 0", current_time, "!=", last_s_time)
                print("| queue.size |", self.queue.qsize(), "<= 2 *", self.schedule_interval)
                # 如果server已收到且未使用的client更新数小于schedule interval，则进行schedule
                if self.queue.qsize() <= self.schedule_interval * 2:
                    last_s_time = current_time
                    print("Begin client select")
                    selected_client_threads = self.client_select()
                    print("\nSchedulerThread select(", len(selected_client_threads), "clients):")
                    self.server_thread_lock.acquire()
                    server_weights = copy.deepcopy(self.server_network.get_weights())
                    self.server_thread_lock.release()
                    for s_client_thread in selected_client_threads:
                        print(s_client_thread.get_client_id(), end=" | ")
                        # 将server的模型参数和时间戳发给client
                        s_client_thread.set_client_weight(server_weights)
                        s_client_thread.set_time_stamp(current_time)

                        # 启动一次client线程
                        s_client_thread.set_event()
                    del server_weights
                    print("\n-----------------------------------------------------------------Schedule complete")
                else:
                    print("\n-----------------------------------------------------------------No Schedule")
                time.sleep(0.01)
            else:
                time.sleep(0.01)

    def client_select(self):
        current_checked_client_tl = self.async_client_manager.get_checked_in_client_thread_list()
        select_num = int(self.c_ratio * len(current_checked_client_tl))
        # if select_num < self.schedule_interval * 2:
        #     select_num = self.schedule_interval * 2
        if select_num < self.schedule_interval + 1:
            select_num = self.schedule_interval + 1

        print("Current clients:", len(current_checked_client_tl), ", select:", select_num)

        if self.protocol == "FedAsync":
            selected_client_threads = random.sample(current_checked_client_tl, select_num)

        elif self.protocol == "ClusteredFedAsync":
            if 0 < self.use_scheduling_strategy < 3 or self.use_scheduling_strategy == 4:
                selected_client_threads = []
                selected_client_id_list = []
                c_dict = self.async_client_manager.copy_c()
                cluster_queue_size_dict = self.async_client_manager.get_cluster_queue_size_dict()

                queue_size_list = []
                cluster_size_list = []
                cluster_client_dict = self.async_client_manager.copy_cluster_client_dict()
                print("len of c_dict:", len(c_dict))
                print("len of cluster_queue_size_dict:", len(cluster_queue_size_dict))
                if len(cluster_queue_size_dict) == 0:
                    # 若是第一次schedule，则均匀schedule
                    for key in cluster_client_dict:
                        queue_size_list.append(1)
                        cluster_size_list.append(self.async_client_manager.get_cluster_client_dict_len(key))
                else:
                    for key in cluster_queue_size_dict:
                        queue_size_list.append(cluster_queue_size_dict[key])
                        cluster_size_list.append(self.async_client_manager.get_cluster_client_dict_len(key))

                select_list = Tools.balance_select(queue_size_list, cluster_size_list, select_num,
                                                   self.use_scheduling_strategy)
                print("queue_size_list:", queue_size_list)
                print("cluster_size_list:", cluster_size_list)
                print("Select_list:", select_list, "| sum =", sum(select_list))
                for key in cluster_client_dict:
                    client_id_list = list(cluster_client_dict[key].keys())
                    c_selected_client_id_list = random.sample(client_id_list, select_list[key])
                    selected_client_id_list.extend(c_selected_client_id_list)

                for client_thread in current_checked_client_tl:
                    if client_thread.get_client_id() in selected_client_id_list:
                        selected_client_threads.append(client_thread)
            elif self.use_scheduling_strategy == 3:
                selected_client_threads = []
                selected_client_id_list = []
                c_dict = self.async_client_manager.copy_c()
                cluster_queue_size_dict = self.async_client_manager.get_cluster_queue_size_dict()

                print("len of c_dict:", len(c_dict))
                print("len of cluster_queue_size_dict:", len(cluster_queue_size_dict))
                if len(cluster_queue_size_dict) == 0:
                    avg_queue_size = 0
                    var_queue_size = 0
                else:
                    avg_queue_size = float(sum(cluster_queue_size_dict.values()) / len(cluster_queue_size_dict))
                    var_queue_size = numpy.var(list(cluster_queue_size_dict.values()))
                print("Avg_queue_size:", avg_queue_size)
                print("Var of cluster_queue_size_dict:", var_queue_size)

                # 计算 2 < queue_size < avg_queue_size的cluster的数量
                small_cluster_num = 0
                for k in cluster_queue_size_dict:
                    print("cluster_queue_size_dict[", k, "]:", cluster_queue_size_dict[k])
                    if 2 < cluster_queue_size_dict[k] < avg_queue_size:
                        small_cluster_num += 1

                for key in c_dict:
                    # 该cluster已有的更新越多，则越少选择该cluster中的client
                    if self.async_client_manager.c_get_sum() == 0:
                        c_c_ratio = float(
                            len(self.async_client_manager.cluster_client_dict_get(key)) / len(
                                current_checked_client_tl))
                    else:
                        if self.s_setting[1] == 0 and len(c_dict) == len(cluster_queue_size_dict):
                            c_c_ratio = float(cluster_queue_size_dict[key] / sum(cluster_queue_size_dict.values()))
                        else:
                            c_c_ratio = float(
                                self.async_client_manager.c_get(key) / self.async_client_manager.c_get_sum())
                    c_select_ratio = 1 - c_c_ratio
                    client_dict = self.async_client_manager.cluster_client_dict_get(key)
                    # c_select_num = int(c_select_ratio * select_num / len(c_dict)) + 1

                    # 分布极其不均匀
                    if var_queue_size >= 30 and len(c_dict) > 10:
                        # 最大化queue_size <= 2的cluster的任务分配
                        if cluster_queue_size_dict[key] <= 2:
                            c_select_num = select_num
                        else:
                            # queue_size过大的cluster不分配任务
                            if cluster_queue_size_dict[key] >= avg_queue_size:
                                c_select_num = 0
                            # 其他cluster正常分配任务
                            else:
                                c_select_num = int(c_select_ratio * select_num / small_cluster_num) + 1
                    # 分布较为均匀
                    else:
                        c_select_num = int(c_select_ratio * select_num / len(c_dict)) + 1

                    if c_select_num > len(client_dict):
                        print(c_select_num, ">", len(client_dict))
                        c_select_num = len(client_dict)
                    # c_selected_client_id_list = random.sample(client_dict, c_select_num)
                    c_selected_client_id_list = random.sample(client_dict.keys(), c_select_num)
                    selected_client_id_list.extend(c_selected_client_id_list)
                    print("| Cluster", key, ":", self.async_client_manager.c_get(key), "/",
                          self.async_client_manager.c_get_sum(),
                          round(c_c_ratio, 4), round(c_select_ratio, 4), c_select_num, c_selected_client_id_list)

                # 若选出的client数不足，则进行补充
                print("len(selected_client_id_list) < select_num: ", len(selected_client_id_list), "<", select_num)
                if len(selected_client_id_list) < select_num:
                    insufficient_num = int(select_num - len(selected_client_id_list))
                    for key in c_dict:
                        client_dict = self.async_client_manager.cluster_client_dict_get(key)
                        if len(client_dict) > insufficient_num:
                            print("insufficient_num:", insufficient_num)
                            while insufficient_num > 0:
                                for k in client_dict.keys():
                                    if k not in selected_client_id_list:
                                        selected_client_id_list.append(k)
                                        insufficient_num -= 1
                                    if insufficient_num == 0:
                                        break
                            print("insufficient_num::", insufficient_num)

                for key in c_dict:
                    print(self.async_client_manager.get_cluster_client_dict_len(key), end=",")
                print("")

                for client_thread in current_checked_client_tl:
                    if client_thread.get_client_id() in selected_client_id_list:
                        selected_client_threads.append(client_thread)
            else:
                selected_client_threads = random.sample(current_checked_client_tl, select_num)

        else:
            print("Unexpected protocol!!!!")
            selected_client_threads = random.sample(current_checked_client_tl, select_num)

        return selected_client_threads
