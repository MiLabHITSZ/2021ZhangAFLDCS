import os
import numpy as np
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


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)  # 类型转换
    return x, y


def preprocess_data(x, y, batch):
    # 构建训练集对象，随机打乱，预处理，批量化
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.shuffle(1000).map(preprocess, num_parallel_calls=8).batch(batch)  # 构建测试集对象，预处理，批量化
    return db


def draw_pictures(cifar10_pictures):
    for picture in cifar10_pictures:
        plt.figure()
        plt.imshow(picture)
    plt.show()


def train_one_epoch(network, train_data, optimizer, received_weights=None, rho=0.005):
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:  # 构建梯度记录环境
            out = network(x, training=True)
            out = tf.reshape(out, [out.shape[0], -1])

            # 真实标签 one-hot 编码，[b] => [b, 10]
            # if len(y.shape) > 1:
            #     y = tf.squeeze(y, axis=1)
            y = tf.squeeze(y, axis=1)
            y_one_hot = tf.one_hot(y, depth=10)
            # y_one_hot = tf.reshape(y_one_hot, [y_one_hot.shape[0], -1])
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


class CIFAR10:
    def __init__(self):
        (x, y), (x_test, y_test) = datasets.cifar10.load_data()  # 加载 CIFAR10 数据集
        self.train_data = [x, y]
        self.pre_valid_data = [x_test[0: 50], y_test[0: 50]]
        self.valid_data = [x_test[100: 5000], y_test[100: 5000]]
        self.test_data = [x_test[5000: 10000], y_test[5000: 10000]]
        self.big_test_data = [x_test, y_test]

        print(type(x), x.shape, "|-\t-\t-|", type(y), y.shape)
        print(type(x[0]), x[0].shape, "|-\t-\t-\t-|", type(y[0]), y[0].shape, y[0])
        print(type(x[0][0]), x[0][0].shape, "|-\t-\t-\t-|", type(y[0][0]), y[0][0].shape, y[0][0])
        print(type(x[0][0][0]), x[0][0][0].shape, x[0][0][0])
        print(type(x[0][0][0][0]), x[0][0][0][0].shape, x[0][0][0][0])
        print("====================================================================================")
        print(type(x_test), x_test.shape, "|-\t-\t-|", type(y_test), y_test.shape)
        print(type(x_test[0]), x_test[0].shape, "|-\t-\t-\t-|", type(y_test[0]), y_test[0].shape, y_test[0])
        print(type(x_test[0][0]), x_test[0][0].shape, "|-\t-\t-\t-|", type(y_test[0][0]), y_test[0][0].shape, y_test[0][0])
        print(type(x_test[0][0][0]), x_test[0][0][0].shape, x_test[0][0][0])
        print(type(x_test[0][0][0][0]), x_test[0][0][0][0].shape, x_test[0][0][0][0])

    def get_sorted_dataset(self):
        x, y = self.train_data[0], self.train_data[1]
        sorted_cifar10_x, sorted_cifar10_y = Tools.generate_sorted_dataset(x, y, 50000)
        return sorted_cifar10_x, sorted_cifar10_y

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
