import random

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
# from tensorflow.python.keras.api._v2.keras import layers, Sequential, regularizers, optimizers, models
from tensorflow.keras import layers, Sequential, optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import tensorflow.keras.preprocessing.image as image

import utils.Tools as Tools
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Datasets.CIFAR100 as CIFAR100
import Datasets.ImageNette as ImageNette


class VGG16:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name == "MNIST":
            self.dataset = MNIST
            self.output_size = 10
        elif self.dataset_name == "FashionMNIST":
            self.dataset = FashionMNIST
            self.output_size = 10
        elif self.dataset_name == "CIFAR10":
            self.dataset = CIFAR10
            self.output_size = 10
        elif self.dataset_name == "CIFAR100":
            self.dataset = CIFAR100
            self.output_size = 100
        elif self.dataset_name == "ImageNette":
            self.dataset = ImageNette
            self.output_size = 10
        else:
            print("Unexpected dataset name!")
            self.dataset = MNIST
            self.output_size = 10

        self.vgg_layers = [
            # 64 个 3x3 卷积核, 输入输出同大小
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # Conv-Conv-Pooling 单元 2,输出通道提升至 128，高宽大小减半
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # Conv-Conv-Pooling 单元 3,输出通道提升至 256，高宽大小减半
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # Conv-Conv-Pooling 单元 4,输出通道提升至 512，高宽大小减半
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # Conv-Conv-Pooling 单元 5,输出通道提升至 512，高宽大小减半
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

            # 创建 3 层全连接层子网络
            layers.Flatten(),
            layers.Dense(4096, activation=tf.nn.relu),
            # layers.Dropout(0.5),
            layers.Dense(4096, activation=tf.nn.relu),
            # layers.Dropout(0.5),
            layers.Dense(self.output_size, activation='softmax')
        ]

        # 利用前面创建的层列表构建网络容器
        self.vgg_network = Sequential(self.vgg_layers)

        # build网络
        if self.dataset_name == "MNIST":
            self.vgg_network.build(input_shape=[None, 28, 28, 1])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.vgg_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        elif self.dataset_name == "CIFAR10":
            self.vgg_network.build(input_shape=[None, 32, 32, 3])
            # self.optimizer = optimizers.Adam(lr=1e-4)
            # self.optimizer = optimizers.Adam(lr=1e-5)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
            self.vgg_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        elif self.dataset_name == "CIFAR100":
            self.vgg_network.build(input_shape=[None, 32, 32, 3])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.vgg_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        elif self.dataset_name == "ImageNette":
            self.vgg_network.build(input_shape=[None, 224, 224, 3])
            # self.optimizer = optimizers.Adam(lr=1e-4)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5, nesterov=True)
            self.vgg_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        else:
            print("Unexpected dataset name!")
            self.vgg_network.build(input_shape=[None, 32, 32, 3])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.vgg_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.init_weights = self.vgg_network.get_weights()

    def model_train_one_epoch(self, x_data, y_data, batch_size, r_weights=None, rho=0):
        train_data = self.dataset.preprocess_data(x_data, y_data, batch_size)
        if r_weights is None:
            updated_weights = self.dataset.train_one_epoch(self.vgg_network, train_data, self.optimizer)
        else:
            updated_weights = self.dataset.train_one_epoch(self.vgg_network, train_data, self.optimizer, r_weights, rho)
        self.vgg_network.set_weights(updated_weights)

    def evaluate_network(self, x_data, y_data):
        correct_num, total_num = 0, 0
        loss_list = []
        evaluate_data = self.dataset.preprocess_data(x_data, y_data, 64)

        # 记录预测正确的数量，总样本数量
        for x, y in evaluate_data:  # 遍历所有训练集样本
            y = tf.squeeze(y, axis=1)

            out = self.vgg_network(x, training=False)
            # print("out", out)
            out = tf.reshape(out, [out.shape[0], self.output_size])
            # print("out", out)
            # 先经过 softmax，再 argmax
            prob = tf.nn.softmax(out, axis=1)
            # print("prob", prob)
            pred = tf.argmax(prob, axis=1)
            # print("pred", pred)
            pred = tf.cast(pred, dtype=tf.int32)
            # print(pred, "=", y)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct_num += int(tf.reduce_sum(correct))
            total_num += int(x.shape[0])

            # 计算交叉熵损失函数，标量
            y_one_hot = tf.one_hot(y, depth=self.output_size)
            loss = tf.losses.categorical_crossentropy(y_one_hot, out, from_logits=True)
            loss_list.append(float(tf.reduce_mean(loss)))

        # 计算准确率和loss
        accuracy = float(correct_num / total_num)
        loss = sum(loss_list) / len(loss_list)
        return accuracy, float(loss)

    def get_init_weights(self):
        return self.init_weights

    def set_weights(self, weights):
        self.vgg_network.set_weights(weights)

    def get_weights(self):
        return self.vgg_network.get_weights()
