import threading
import time
import copy
import os


class AsyncClient(threading.Thread):
    def __init__(self, c_id, client, queue, pre_train_queue, batch_size, e, rho, stop_event, delay, pre_training):
        threading.Thread.__init__(self)
        self.client_id = c_id
        self.client = client
        self.queue = queue
        self.pre_train_queue = pre_train_queue
        self.event = threading.Event()
        self.event.clear()
        self.batch_size = batch_size
        self.e = e
        self.train_e = e
        self.stop_event = stop_event
        self.delay = delay
        self.rho = rho

        self.time_stamp = 0
        self.client_thread_lock = threading.Lock()

        self.weights_buffer = []
        self.time_stamp_buffer = 0
        self.received_weights = False
        self.received_time_stamp = False
        self.event_is_set = False
        self.pre_training = pre_training

    def run(self):
        while not self.stop_event.is_set():
            if self.received_weights:
                self.client.set_client_weights(self.weights_buffer)
                self.received_weights = False
            if self.received_time_stamp:
                self.time_stamp = self.time_stamp_buffer
                self.received_time_stamp = False
            if self.event_is_set:
                # self.event.set()
                self.event_is_set = False

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.client_thread_lock.acquire()
                # 该client进行训练
                if self.rho == 0:
                    uwr = False
                else:
                    uwr = True
                self.client.client_train_one_epoch(self.batch_size, self.e, use_weight_regularization=uwr, rho=self.rho)

                # client传回server的信息具有延迟
                print("Client", self.client_id, "trained")
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                if self.pre_training:
                    self.pre_train_queue.put((self.client_id, self.client.get_client_weights(), self.time_stamp))
                    self.e = self.train_e
                else:
                    self.queue.put((self.client_id, self.client.get_client_weights(), self.time_stamp))

                self.event.clear()

                self.client_thread_lock.release()

            # 该client等待被选中
            else:
                self.event.wait()

    def set_client_id(self, new_id):
        self.client_thread_lock.acquire()
        self.client_id = new_id
        self.client_thread_lock.release()

    def get_client_id(self):
        # self.client_thread_lock.acquire()
        c_id = copy.deepcopy(self.client_id)
        # self.client_thread_lock.release()
        return c_id

    def set_rho(self, new_rho):
        self.client_thread_lock.acquire()
        self.rho = new_rho
        self.client_thread_lock.release()

    def get_rho(self):
        # self.client_thread_lock.acquire()
        rho = copy.deepcopy(self.rho)
        # self.client_thread_lock.release()
        return rho

    def set_client_weight(self, weights):
        # self.client_thread_lock.acquire()
        self.weights_buffer = weights
        self.received_weights = True
        # self.client.set_client_weights(weights)
        # self.client_thread_lock.release()

    def get_client_weight(self):
        # self.client_thread_lock.acquire()
        client_weights = copy.deepcopy(self.client.get_client_weights())
        # self.client_thread_lock.release()
        return client_weights

    def set_event(self):
        # self.client_thread_lock.acquire()
        self.event_is_set = True
        self.event.set()
        # self.client_thread_lock.release()

    def get_event(self):
        # self.client_thread_lock.acquire()
        event_is_set = self.event.is_set()
        # self.client_thread_lock.release()
        return event_is_set

    def set_time_stamp(self, current_time):
        # self.client_thread_lock.acquire()
        self.time_stamp_buffer = current_time
        self.received_time_stamp = True
        # self.time_stamp = current_time
        # self.client_thread_lock.release()

    def get_time_stamp(self):
        # self.client_thread_lock.acquire()
        t_s = copy.deepcopy(self.time_stamp)
        # self.client_thread_lock.release()
        return t_s

    def set_delay(self, new_delay):
        self.client_thread_lock.acquire()
        self.delay = new_delay
        self.client_thread_lock.release()

    def get_delay(self):
        # self.client_thread_lock.acquire()
        delay = copy.deepcopy(self.delay)
        # self.client_thread_lock.release()
        return delay

    def set_pre_training(self, new_pre_training):
        self.client_thread_lock.acquire()
        self.pre_training = new_pre_training
        self.client_thread_lock.release()

    def get_pre_training(self):
        return self.pre_training

    def set_e(self, new_e):
        self.e = new_e

    def get_e(self):
        return self.e
