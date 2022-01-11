import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
# from tensorflow.python.keras.api._v2.keras import layers, Sequential, regularizers, optimizers, models
from tensorflow.keras import layers, Sequential
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


class FC3:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.fc_network = Sequential([layers.Dense(200, activation='relu'),
                                      layers.Dense(200, activation='relu'),
                                      layers.Dense(10)])
        if dataset_name == "MNIST":
            self.dataset = MNIST
            self.output_size = 10
            self.fc_network.build(input_shape=[None, 28 * 28])
            # self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False),
            #                         loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            #                         metrics=['accuracy'])
            self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, decay=0.001, momentum=0.0, nesterov=False),
                                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, decay=0.001, nesterov=False)
        elif dataset_name == "FashionMNIST":
            self.dataset = FashionMNIST
            self.output_size = 10
            self.fc_network.build(input_shape=[None, 28 * 28])
            self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, decay=0.001, momentum=0.0, nesterov=False),
                                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, decay=0.001, nesterov=False)
            # self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False),
            #                         loss=tf.losses.CategoricalCrossentropy(from_logits=True),
            #                         metrics=['accuracy'])
            # self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)
        elif self.dataset_name == "CIFAR10":
            self.dataset = CIFAR10
            self.output_size = 10
            self.fc_network.build(input_shape=[None, 32, 32, 3])
            self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False),
                                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
        elif self.dataset_name == "CIFAR100":
            self.dataset = CIFAR100
            self.output_size = 100
            self.fc_network.build(input_shape=[None, 32, 32, 3])
            self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False),
                                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
        else:
            print("Unexpected dataset name!")
            self.dataset = MNIST
            self.output_size = 10
            self.fc_network.build(input_shape=[None, 28 * 28])
            self.fc_network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False),
                                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False)

        self.init_weights = self.fc_network.get_weights()

    def model_train_one_epoch(self, x_data, y_data, batch_size, r_weights=None, rho=0):
        train_data = self.dataset.preprocess_data(x_data, y_data, batch_size)
        if r_weights is None:
            updated_weights = self.dataset.train_one_epoch(self.fc_network, train_data, self.optimizer)
        else:
            updated_weights = self.dataset.train_one_epoch(self.fc_network, train_data, self.optimizer, r_weights, rho)
        self.fc_network.set_weights(updated_weights)

    def evaluate_network(self, x_data, y_data):
        x_data, y_data = MNIST.preprocess(x_data, y_data)
        e_loss, e_acc = self.fc_network.evaluate(x_data, y_data)
        return e_acc, e_loss

    def get_init_weights(self):
        return self.init_weights

    def set_weights(self, weights):
        self.fc_network.set_weights(weights)

    def get_weights(self):
        return self.fc_network.get_weights()
