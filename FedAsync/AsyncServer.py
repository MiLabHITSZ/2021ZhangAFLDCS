import random
from tqdm import tqdm
import copy
import time
import threading
import ctypes
import inspect
import asyncio
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
import AsyncClientManager
import CheckInThread
import SchedulerThread
import UpdaterThread
import Time


def _async_raise(tid, exc_type):
    """raises the exception, performs cleanup if needed"""
    try:
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exc_type):
            exc_type = type(exc_type)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exc_type))
        if res == 0:
            # pass
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    except Exception as err:
        print(err)


class AsyncServer:
    def __init__(self, protocol, use_scheduling_strategy, data_type, model_name, c_r, client_list, valid_data,
                 test_data, batch_size, e, r, t, alpha, schedule_interval, check_in_interval, check_in_num,
                 global_network, init_weights, s_setting, c_s_list):
        self.protocol = protocol
        self.use_scheduling_strategy = use_scheduling_strategy
        self.data_type = data_type
        self.model_name = model_name
        self.client_ratio = c_r
        self.client_list = client_list
        self.clients_num = len(client_list)
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.e = e
        self.rho = r
        self.T = t
        self.alpha = alpha
        self.schedule_interval = schedule_interval
        self.check_in_interval = check_in_interval
        self.check_in_num = check_in_num
        self.global_network = global_network
        self.init_weights = init_weights
        self.s_setting = s_setting
        self.client_staleness_list = c_s_list

        self.current_t = Time.Time(0)
        self.queue = Queue()
        self.stop_event = threading.Event()
        self.stop_event.clear()

        self.accuracy_and_loss_list = []
        self.clustering_result_list = []

        if self.data_type == "MNIST":
            self.dataset = MNIST
        elif self.data_type == "CIFAR10":
            self.dataset = CIFAR10
        else:
            self.dataset = MNIST

        if self.model_name == "FC3":
            self.server_network = FC3.FC3(self.data_type)
        elif self.model_name == "VGG13":
            self.server_network = VGG13.VGG13(self.data_type)
        elif self.model_name == "CNN":
            self.server_network = CNN.CNN(self.data_type)
        else:
            pass

        self.server_thread_lock = threading.Lock()

        self.async_client_manager = AsyncClientManager.AsyncClientManager(self.protocol, self.client_list, init_weights,
                                                                          self.queue, self.batch_size, self.e, self.rho,
                                                                          self.current_t, self.stop_event,
                                                                          self.client_staleness_list)

        self.scheduler_thread = SchedulerThread.SchedulerThread(self.protocol, self.use_scheduling_strategy,
                                                                self.server_thread_lock, self.schedule_interval,
                                                                self.check_in_interval, self.async_client_manager,
                                                                self.queue, self.client_ratio, self.current_t,
                                                                self.global_network, self.T, self.s_setting)
        self.updater_thread = UpdaterThread.UpdaterThread(self.protocol, self.queue, self.server_thread_lock,
                                                          self.alpha, self.T, self.current_t, self.global_network,
                                                          self.test_data, self.async_client_manager, self.stop_event,
                                                          self.s_setting)
        self.check_in_thread = CheckInThread.CheckInThread(self.protocol, self.check_in_interval, self.check_in_num,
                                                           self.async_client_manager, self.current_t, self.T)

    def run(self):
        print("Start server:")

        # 启动server中的两个线程
        self.scheduler_thread.start()
        self.updater_thread.start()
        self.check_in_thread.start()

        # client_thread_list = self.async_client_manager.get_client_thread_list()
        client_thread_list = self.async_client_manager.get_checked_in_client_thread_list()
        for client_thread in client_thread_list:
            client_thread.join()
        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")
        # self.check_in_thread.join()
        _async_raise(self.check_in_thread.ident, SystemExit)
        print("check_in_thread joined")

        print("Thread count =", threading.activeCount())
        print(*threading.enumerate(), sep="\n")

        if not self.queue.empty():
            print("\nUn-used client weights:", self.queue.qsize())
            for q in range(self.queue.qsize()):
                self.queue.get()
        self.queue.close()

        self.accuracy_and_loss_list = self.updater_thread.get_accuracy_and_loss_list()
        # self.scheduler_thread.handled = True
        self.clustering_result_list = self.async_client_manager.get_clustering_result_list()
        del self.scheduler_thread
        del self.updater_thread
        del self.async_client_manager
        del self.check_in_thread
        print("End!")

    def get_accuracy_and_loss_list(self):
        return self.accuracy_and_loss_list

    def get_clustering_result_list(self):
        return self.clustering_result_list
