import random
from tqdm import tqdm

import utils.Tools as Tools
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import Models.CNN as CNN
import Datasets.MNIST as MNIST
import Datasets.CIFAR10 as CIFAR10


class Server:
    def __init__(self, data_type, model_name, c_ratio, client_list, init_weight, valid_data, test_data, batch_size, e):
        self.data_type = data_type
        self.model_name = model_name
        self.client_ratio = c_ratio
        self.client_list = client_list
        self.clients_num = len(client_list)
        self.server_weights = init_weight
        self.valid_data = valid_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.e = e

        self.clients_weights_list = []
        self.current_accuracy = 0
        self.current_loss = 0

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

    def train_clients_one_epoch(self, s_id):
        # 清空clients_weights_list
        self.clients_weights_list = []

        # 随机在本group中按比例选择一些本轮会参加训练的client
        random_clients = random.sample(self.client_list, int(self.clients_num * self.client_ratio))

        # 用选出的client进行训练
        for r_clients in tqdm(random_clients, ascii=True):
            # Server将当前模型参数发给各个Client
            r_clients.set_client_weights(self.server_weights)

            # 各个Client进行训练
            r_clients.model_train_one_epoch(self.batch_size, self.e)

            # 各个Client返回其模型参数
            self.clients_weights_list.append(r_clients.get_client_weights())

        # 将本组所有Client的weights进行平均，得到一个平均weights以更新server_weights
        sum_client_weights_list = Tools.sum_nd_array_lists(self.clients_weights_list)
        averaged_client_weights_list = Tools.avg_nd_array_list(sum_client_weights_list, len(random_clients))
        self.server_weights = averaged_client_weights_list
        self.server_network.set_weights(self.server_weights)

        # 对本server的当前model进行测试
        # self.run_server_valid(s_id)
        accuracy, loss = self.run_server_test(s_id)
        return accuracy, loss

    def train_clients_one_epoch_with_iid_selection(self, s_id, iid_clients_list, accuracy_lists, loss_lists):
        # 清空clients_weights_list
        self.clients_weights_list = []

        for iid_clients in iid_clients_list:
            iid_clients_weights_list = []
            for iid_client in iid_clients:
                # Server将当前模型参数发给该类中的各个Client
                iid_client.set_client_weights(self.server_weights)

                # 该类中的各个Client进行训练
                iid_client.model_train_one_epoch(self.batch_size, self.e)

                # 该类中的各个Client返回其模型参数
                iid_clients_weights_list.append(iid_client.get_client_weights())

            # 该类中的各个Client的参数进行平均
            if len(iid_clients_weights_list) > 0:
                iid_sum_client_weights_list = Tools.sum_nd_array_lists(iid_clients_weights_list)
                iid_averaged_client_weights_list = Tools.avg_nd_array_list(iid_sum_client_weights_list,
                                                                           len(iid_clients_weights_list))
                self.clients_weights_list.append(iid_averaged_client_weights_list)

        # 各个类的clients的平均参数再进行平均以获得全局参数
        sum_client_weights_list = Tools.sum_nd_array_lists(self.clients_weights_list)
        averaged_client_weights_list = Tools.avg_nd_array_list(sum_client_weights_list, len(self.clients_weights_list))
        self.server_weights = averaged_client_weights_list

        # 进行测试并记录accuracy和loss
        self.server_network.set_weights(self.server_weights)
        accuracy, loss = self.run_server_test(s_id)
        accuracy_lists[s_id].append(accuracy)
        loss_lists[s_id].append(loss)

        return accuracy_lists, loss_lists

    def set_clients(self, new_client_list):
        self.client_list = new_client_list

    def set_e(self, new_e):
        self.e = new_e

    def set_client_ratio(self, new_client_ratio):
        self.client_ratio = new_client_ratio

    def get_server_weights(self):
        return self.server_weights

    def set_server_weights(self, weights):
        self.server_weights = weights

    def get_clients_weights(self):
        return self.clients_weights_list

    def get_server_accuracy(self):
        return self.current_accuracy

    def get_server_loss(self):
        return self.current_loss

    def run_server_valid(self, s_id):
        accuracy, loss = self.server_network.evaluate_network(self.valid_data[0], self.valid_data[1])
        self.current_accuracy = accuracy
        self.current_loss = loss
        print('Server_id(v):', s_id, 'accuracy:', accuracy, 'loss:', float(loss))
        return accuracy, loss

    def run_server_test(self, s_id):
        accuracy, loss = self.server_network.evaluate_network(self.test_data[0], self.test_data[1])
        self.current_accuracy = accuracy
        self.current_loss = loss
        print('Server_id(t):', s_id, 'accuracy:', accuracy, 'loss:', float(loss))
        return accuracy, loss

    def evaluate_client(self, c_id, evaluate_data):
        e_client = self.client_list[c_id]
        self.server_network.set_weights(e_client.get_client_weights())
        e_x, e_y = MNIST.preprocess(evaluate_data[0], evaluate_data[1])
        accuracy, loss = self.server_network.evaluate_network(e_x, e_y)

        self.server_network.set_weights(self.server_weights)
        return accuracy, loss
