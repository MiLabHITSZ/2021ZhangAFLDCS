import os
import numpy as np
import math
import copy
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

import utils.Tools as Tools


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.  # 标准化到 0~1
    x = tf.reshape(x, [-1, 28 * 28])  # 打平
    y = tf.cast(y, dtype=tf.int32)  # 转成整型张量
    y = tf.one_hot(y, depth=10)  # one-hot 编码
    return x, y


def preprocess_data(x, y, batch):
    preprocessed_data = tf.data.Dataset.from_tensor_slices((x, y))  # 构建 Dataset 对象
    preprocessed_data = preprocessed_data.shuffle(10000)  # 随机打散样本，不会打乱样本与标签映射关系
    preprocessed_data = preprocessed_data.batch(batch)  # 设置批训练 batch size
    preprocessed_data = preprocessed_data.map(preprocess)
    return preprocessed_data


def draw_pictures(mnist_pictures):
    for picture in mnist_pictures:
        plt.figure()
        plt.imshow(picture)
    plt.show()


def train_one_epoch(network, train_data, optimizer, received_weights=None, rho=0):
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:  # 构建梯度记录环境
            out = network(x, training=True)
            out = tf.reshape(out, [out.shape[0], -1])
            # 真实标签 one-hot 编码，[b] => [b, 10]
            # y_one_hot = tf.one_hot(y, depth=10)
            y_one_hot = y
            y_one_hot = tf.reshape(y_one_hot, [y_one_hot.shape[0], -1])
            # 计算交叉熵损失函数，标量
            loss = tf.losses.categorical_crossentropy(y_one_hot, out, from_logits=True)

            if received_weights is not None:
                # 计算正则项
                difference = tf.constant(0, dtype=tf.float32)
                for layer in range(len(received_weights)):
                    received_weights[layer] = tf.cast(received_weights[layer], dtype=tf.float32)
                    w_difference = tf.math.square(received_weights[layer] - network.trainable_variables[layer])
                    difference += tf.math.reduce_sum(w_difference)

                loss = tf.math.reduce_mean(loss) + (rho * difference)
            else:
                loss = tf.math.reduce_mean(loss)

        # 对所有参数求梯度
        grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

    return network.get_weights()


class MNIST:
    def __init__(self):
        (x, y), (x_test, y_test) = datasets.mnist.load_data()  # 加载 MNIST 数据集
        self.train_data = [x, y]
        self.pre_valid_data = [x_test[0: 50], y_test[0: 50]]
        self.valid_data = [x_test[100: 5000], y_test[100: 5000]]
        self.test_data = [x_test[5000: 10000], y_test[5000: 10000]]
        self.big_test_data = [x_test, y_test]

    def get_sorted_dataset(self):
        x, y = self.train_data[0], self.train_data[1]
        sorted_mnist_x, sorted_mnist_y = Tools.generate_sorted_dataset(x, y, 60000)
        return sorted_mnist_x, sorted_mnist_y

    def get_train_data(self):
        return self.train_data[0], self.train_data[1]

    def get_pre_valid_data(self):
        return self.pre_valid_data[0], self.pre_valid_data[1]

    def get_valid_data(self):
        return self.valid_data[0], self.valid_data[1]

    def get_test_data(self):
        return self.test_data[0], self.test_data[1]

    def get_big_test_data(self):
        return self.big_test_data[0], self.big_test_data[1]
