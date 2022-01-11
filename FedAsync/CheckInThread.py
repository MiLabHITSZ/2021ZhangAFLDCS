import threading
import time


class CheckInThread(threading.Thread):
    def __init__(self, protocol, check_in_interval, check_in_num, async_client_manager, c_t, t):
        threading.Thread.__init__(self)
        self.protocol = protocol
        self.check_in_interval = check_in_interval
        self.check_in_num = check_in_num
        self.async_client_manager = async_client_manager
        self.current_t = c_t
        self.T = t

    def run(self):
        last_c_time = -1
        c_count = 0
        waiting_c_n = 0  # 已经check in但是尚未返回pre-train结果的client数
        # while self.current_t.get_time() < self.T:
        #     current_time = self.current_t.get_time()
        #
        #     # 每隔一段时间就有新的client check in
        #     if self.async_client_manager.get_unchecked_in_client_thread_list_len() > 0:
        #         if self.protocol == "ClusteredFedAsync":
        #             print("CT:", current_time, " > ", self.check_in_interval, "*", c_count, " == ",
        #                   (self.check_in_interval * c_count), ", last_c_time =", last_c_time,
        #                   ", waiting =", waiting_c_n,
        #                   "unchecked_in_clients =", self.async_client_manager.get_unchecked_in_client_thread_list_len())
        #
        #         # if current_time % self.check_in_interval == 0 and current_time != last_c_time:
        #         if current_time >= (self.check_in_interval * c_count) and current_time != last_c_time and waiting_c_n == 0:
        #             last_c_time = current_time
        #             c_count += 1
        #             waiting_c_n = self.async_client_manager.client_check_in(self.check_in_num)
        #             print("\n--------------------------------------------------------The", c_count,
        #                   "th Check in complete")
        #
        #         # check in之后等待接受clients pre-train返回的数据
        #         if self.protocol == "ClusteredFedAsync" and waiting_c_n > 0:
        #             self.async_client_manager.receive_clients_pre_train_weights(1, False)
        #             waiting_c_n -= 1
        #         elif self.protocol == "FedAsync":
        #             waiting_c_n = 0
        #
        #     time.sleep(0.01)

        while self.current_t.get_time() < self.T:
            current_time = self.current_t.get_time()

            # 每隔一段时间就有新的client check in
            if self.async_client_manager.get_unchecked_in_client_thread_list_len() > 0 and self.check_in_num > 0:
                if self.protocol == "ClusteredFedAsync" and waiting_c_n > 0:
                    print("CT:", current_time, " > ", self.check_in_interval, "*", c_count, " == ",
                          (self.check_in_interval * c_count), ", last_c_time =", last_c_time,
                          ", waiting =", waiting_c_n,
                          "unchecked_in_clients =", self.async_client_manager.get_unchecked_in_client_thread_list_len())

                # if current_time % self.check_in_interval == 0 and current_time != last_c_time:
                if current_time >= (self.check_in_interval * c_count) and current_time != last_c_time:
                    last_c_time = current_time
                    c_count += 1
                    # self.async_client_manager.client_check_in(self.check_in_num)
                    waiting_c_n += self.async_client_manager.client_check_in(self.check_in_num)
                    print("\n--------------------------------------------------------The", c_count, "th Check in complete")

                # # check in之后等待接受clients pre-train返回的数据
                # if self.protocol == "ClusteredFedAsync":
                #     self.async_client_manager.receive_clients_pre_train_weights(1, False)

                # check in之后等待接受clients pre-train返回的数据
                if self.protocol == "ClusteredFedAsync" and waiting_c_n > 0:
                    self.async_client_manager.receive_clients_pre_train_weights(1, False)
                    waiting_c_n -= 1
                elif self.protocol == "FedAsync":
                    waiting_c_n = 0
                    # self.async_client_manager.clustering_result_list.append([self.async_client_manager.get_unchecked_in_client_thread_list_len()])

            time.sleep(0.01)
