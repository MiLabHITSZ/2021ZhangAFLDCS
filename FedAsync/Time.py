import random
from tqdm import tqdm
import time
from multiprocessing import Process, Queue

import utils.Tools as Tools
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import Models.CNN as CNN
import Datasets.MNIST as MNIST
import Datasets.CIFAR10 as CIFAR10

import threading


class Time:
    def __init__(self, init_time):
        self.current_time = init_time

        # self.c_dict = {}
        # self.cluster_client_dict = {}
        self.thread_lock = threading.Lock()

    def time_add(self):
        self.thread_lock.acquire()
        self.current_time += 1
        self.thread_lock.release()

    def set_time(self, new_time):
        self.thread_lock.acquire()
        self.current_time = new_time
        self.thread_lock.release()

    def get_time(self):
        self.thread_lock.acquire()
        c_time = self.current_time
        self.thread_lock.release()
        return c_time

    # def c_add(self, k, v):
    #     self.thread_lock.acquire()
    #     self.c_dict[k] = v
    #     self.thread_lock.release()
    #
    # def c_update(self, k, v):
    #     self.thread_lock.acquire()
    #     self.c_dict[k] = v
    #     self.thread_lock.release()
    #
    # def c_get(self, k):
    #     self.thread_lock.acquire()
    #     ck = self.c_dict[k]
    #     self.thread_lock.release()
    #     return ck
    #
    # def c_get_sum(self):
    #     return sum(self.c_dict.values())
    #
    # def cluster_client_dict_add(self, k):
    #     self.thread_lock.acquire()
    #     self.cluster_client_dict[k] = []
    #     self.thread_lock.release()
    #
    # def cluster_client_dict_update(self, k, v):
    #     self.thread_lock.acquire()
    #     self.cluster_client_dict[k].append(v)
    #     self.thread_lock.release()
    #
    # def cluster_client_dict_get(self, k):
    #     self.thread_lock.acquire()
    #     client_list = self.cluster_client_dict[k]
    #     self.thread_lock.release()
    #     return client_list
    #
    # def cluster_client_dict_find(self, v):
    #     cluster_id = -1
    #     self.thread_lock.acquire()
    #     for key in self.cluster_client_dict:
    #         if v in self.cluster_client_dict[key]:
    #             cluster_id = key
    #             break
    #     self.thread_lock.release()
    #     return cluster_id
