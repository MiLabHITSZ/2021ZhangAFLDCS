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


class CNN:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.cnn_layers = [
            layers.Conv2D(32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=6, padding='same'),

            layers.Conv2D(64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=6, padding='same'),

            # 创建 2 层全连接层子网络
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(10, activation=tf.nn.softmax)
        ]

        # 利用前面创建的层列表构建网络容器
        self.cnn_network = Sequential(self.cnn_layers)

        # build网络
        if self.dataset_name == "MNIST":
            self.dataset = MNIST
            self.output_size = 10
            self.cnn_network.build(input_shape=[None, 28, 28, 1])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.cnn_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        elif self.dataset_name == "FashionMNIST":
            self.dataset = FashionMNIST
            self.output_size = 10
            self.cnn_network.build(input_shape=[None, 28, 28, 1])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.cnn_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        elif self.dataset_name == "CIFAR10":
            self.dataset = CIFAR10
            self.output_size = 10
            self.cnn_network.build(input_shape=[None, 32, 32, 3])
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.cnn_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        elif self.dataset_name == "CIFAR100":
            self.dataset = CIFAR100
            self.output_size = 100
            self.cnn_network.build(input_shape=[None, 32, 32, 3])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.cnn_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        else:
            print("Unexpected dataset name!")
            self.dataset = CIFAR10
            self.output_size = 10
            self.cnn_network.build(input_shape=[None, 32, 32, 3])
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.cnn_network.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.init_weights = self.cnn_network.get_weights()

    def model_train_one_epoch(self, x_data, y_data, batch_size, r_weights=None, rho=0):
        train_data = self.dataset.preprocess_data(x_data, y_data, batch_size)
        if r_weights is None:
            updated_weights = self.dataset.train_one_epoch(self.cnn_network, train_data, self.optimizer)
        else:
            updated_weights = self.dataset.train_one_epoch(self.cnn_network, train_data, self.optimizer, r_weights, rho)
        self.cnn_network.set_weights(updated_weights)

    def evaluate_network(self, x_data, y_data):
        if self.dataset_name == "MNIST":
            x_data, y_data = MNIST.preprocess(x_data, y_data)
        elif self.dataset_name == "CIFAR10":
            evaluate_data = CIFAR10.preprocess_data(x_data, y_data, 64)
            # 记录预测正确的数量，总样本数量
            correct_num, total_num = 0, 0
            loss_list = []
            for x, y in evaluate_data:  # 遍历所有训练集样本
                out = self.cnn_network(x, training=False)
                out = tf.reshape(out, [out.shape[0], 10])
                # 先经过 softmax，再 argmax
                prob = tf.nn.softmax(out, axis=1)
                pred = tf.argmax(prob, axis=1)
                pred = tf.cast(pred, dtype=tf.int32)

                pred = tf.reshape(pred, [pred.shape[0], 1])
                correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
                correct = tf.reduce_sum(correct)
                total_num += x.shape[0]
                correct_num += int(correct)

                # 计算交叉熵损失函数，标量
                y = tf.squeeze(y, axis=1)
                y_one_hot = tf.one_hot(y, depth=10)
                loss = tf.losses.categorical_crossentropy(y_one_hot, out, from_logits=True)
                loss_list.append(float(tf.reduce_mean(loss)))

            # 计算准确率和loss
            accuracy = correct_num / total_num
            loss = sum(loss_list) / len(loss_list)
            return accuracy, float(loss)
        else:
            print("Unexpected dataset name!")
            x_data, y_data = MNIST.preprocess(x_data, y_data)
        e_loss, e_acc = self.cnn_network.evaluate(x_data, y_data)
        return e_acc, e_loss

    def get_init_weights(self):
        return self.init_weights

    def set_weights(self, weights):
        self.cnn_network.set_weights(weights)

    def get_weights(self):
        return self.cnn_network.get_weights()
