import os
import random

import numpy as np
import copy
import math
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

import utils.Tools as Tools
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import Models.CNN as CNN
import Datasets.MNIST as MNIST

# gpu_id = random.randint(0, 4)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


class Client:
    def __init__(self, use_gpu_id, data_type, model_name, client_data, batch_size):
        self.use_gpu_id = use_gpu_id
        self.data_type = data_type
        self.model_name = model_name
        self.client_data = client_data
        self.batch_size = batch_size

        # 生成一个神经网络模型
        if self.model_name == "FC3":
            self.network = FC3.FC3(self.data_type)
        elif self.model_name == "VGG13":
            self.network = VGG13.VGG13(self.data_type)
        elif self.model_name == "CNN":
            self.network = CNN.CNN(self.data_type)
        else:
            pass

    def client_train_one_epoch(self, b, e, use_weight_regularization=False, rho=0):
        with tf.device('/gpu:' + str(self.use_gpu_id)):
        # with tf.device('/gpu:1'):
            # 批处理数据
            if self.batch_size != b:
                self.re_batch(b)

            if use_weight_regularization:
                # 存储接受到的初始模型参数
                r_weights = copy.deepcopy(self.network.get_weights())
                # 进行训练
                for i in range(e):
                    self.network.model_train_one_epoch(self.client_data[0], self.client_data[1], self.batch_size,
                                                       r_weights, rho)
            else:
                # 进行训练
                for i in range(e):
                    self.network.model_train_one_epoch(self.client_data[0], self.client_data[1], self.batch_size)

    def re_batch(self, b):
        self.batch_size = b

    def get_client_weights(self):
        return self.network.get_weights()

    def set_client_weights(self, weights):
        self.network.set_weights(weights)

    def get_data(self):
        return self.client_data

    def set_data(self, client_data):
        self.client_data = client_data
