from random import shuffle
import numpy as np

from Server import Server
import utils.Tools as Tools
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import Models.CNN as CNN
import Datasets.MNIST as MNIST
import Datasets.CIFAR10 as CIFAR10


def grouping(clients_number, group_number, last_best_group):
    new_groups = []
    client_ids = []
    for c_id in range(clients_number):
        client_ids.append(c_id)
    shuffle(client_ids)  # 重排序
    group_len = int(clients_number / group_number)
    # 若无须保留上一个周期最好的组，则直接全部打乱重分组
    if len(last_best_group) == 0:
        for s_id in range(0, clients_number, group_len):
            new_groups.append(client_ids[s_id: s_id + group_len])
    # 若须保留上一个周期最好的组，则除该组成员外的将其他成员打乱进行重分组
    else:
        for lbg_member in last_best_group:
            client_ids.remove(int(lbg_member))
        new_groups.append(last_best_group)
        shuffle(client_ids)  # 重排序
        for s_id in range(0, clients_number - len(last_best_group), group_len):
            new_groups.append(client_ids[s_id: s_id + group_len])
    return new_groups


class ServerManager:
    def __init__(self, data_type, model_name, epochs, batch_size, e, grouping_cycle, group_num,
                 client_ratio, best_server_weight, client_list, valid_data, test_data):
        self.data_type = data_type
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.e = e
        self.grouping_cycle = grouping_cycle
        self.group_num = group_num
        self.client_ratio = client_ratio
        self.best_server_weight = best_server_weight
        self.client_list = client_list
        self.clients_num = len(self.client_list)
        self.group_length = int(self.clients_num / self.group_num)
        self.valid_data = valid_data
        self.test_data = test_data

        self.groups = []
        self.server_list = []
        self.servers_weights_list = []
        self.history_accuracy = []
        self.history_loss = []
        self.history_set = []
        self.test_accuracy_list = []
        self.test_loss_list = []

        if self.data_type == "MNIST":
            self.dataset = MNIST
        elif self.data_type == "CIFAR10":
            self.dataset = CIFAR10
        else:
            self.dataset = MNIST

        if self.model_name == "FC3":
            self.global_network = FC3.FC3(self.data_type)
        elif self.model_name == "VGG13":
            self.global_network = VGG13.VGG13(self.data_type)
        elif self.model_name == "CNN":
            self.global_network = CNN.CNN(self.data_type)
        else:
            pass

        self.global_weights = self.global_network.get_weights()

        # 将Clients进行分组
        self.groups = grouping(self.clients_num, self.group_num, [])

        # 创建Servers
        for g_id in range(self.group_num):
            # 给该server分配clients
            server_client_list = []
            for c_id in self.groups[g_id]:
                server_client_list.append(self.client_list[c_id])

            new_server = Server(self.data_type, self.model_name, self.client_ratio, server_client_list,
                                self.global_weights, self.valid_data, self.test_data, self.batch_size, self.e)
            self.server_list.append(new_server)

    def train_servers(self):
        # 训练 Epoch 数
        for epoch in range(self.epochs):
            # 清空servers_weights_list
            self.servers_weights_list = []
            servers_accuracy_list = []
            servers_loss_list = []

            # 每个融合周期进行重新分组，并将新的全局参数发送给各个server
            if epoch % self.grouping_cycle == 0:
                print("Epoch =", epoch, " Regrouping!")
                self.groups = grouping(self.clients_num, self.group_num, [])
                self.distribute_clients_to_servers(self.groups)

                for s_id in range(self.group_num):
                    self.server_list[s_id].set_server_weights(self.global_weights)

            # 每个server进行本epoch的训练过程
            for s_id in range(self.group_num):
                print("Train group", s_id, ", with clients:", end='')
                print(self.groups[s_id])

                # 第s_id号服务器中的clients进行训练
                server_accuracy, server_loss = self.server_list[s_id].train_clients_one_epoch(s_id)
                servers_accuracy_list.append(server_accuracy)
                servers_loss_list.append(server_loss)

            # 每个周期将所有server的weights进行平均，得到一个平均weights以更新main_weights
            if (epoch + 1) % self.grouping_cycle == 0:
                print("Merging servers!")
                # 获取每个server的weights
                for s_id in range(self.group_num):
                    self.servers_weights_list.append(self.server_list[s_id].get_server_weights())

                # 进行平均并更新全局参数
                sum_servers_weights_list = Tools.sum_nd_array_lists(self.servers_weights_list)
                averaged_client_weights_list = Tools.avg_nd_array_list(sum_servers_weights_list, self.group_num)
                self.global_weights = averaged_client_weights_list
                self.global_network.set_weights(self.global_weights)

            # 每个epoch对全局模型进行测试
            if (epoch + 1) % self.grouping_cycle == 0:
                # self.run_global_valid(epoch)
                self.run_global_test(epoch)
            else:
                # self.history_accuracy.append(self.history_accuracy[len(self.history_accuracy) - 1])
                # self.history_loss.append(self.history_loss[len(self.history_loss) - 1])
                averaged_accuracy = np.mean(servers_accuracy_list)
                averaged_loss = np.mean(servers_loss_list)
                self.history_accuracy.append(averaged_accuracy)
                self.history_loss.append(averaged_loss)
                # best_accuracy = max(servers_accuracy_list)
                # best_loss = max(servers_loss_list)
                # self.history_accuracy.append(best_accuracy)
                # self.history_loss.append(best_loss)

    def distribute_clients_to_servers(self, groups):
        for s_id in range(self.group_num):
            # 给该server分配clients
            server_client_list = []
            for c_id in groups[s_id]:
                server_client_list.append(self.client_list[c_id])
            self.server_list[s_id].set_clients(server_client_list)

    def run_global_valid(self, epoch):
        accuracy, loss = self.global_network.evaluate_network(self.valid_data[0], self.valid_data[1])
        self.history_accuracy.append(accuracy)
        self.history_loss.append(loss)
        print('Epoch(valid-' + str(self.grouping_cycle) + '):', epoch, 'accuracy:', accuracy, 'loss:', float(loss))
        return accuracy, loss

    def run_global_test(self, epoch):
        accuracy, loss = self.global_network.evaluate_network(self.test_data[0], self.test_data[1])
        self.history_accuracy.append(accuracy)
        self.history_loss.append(loss)
        print('Epoch(test-' + str(self.grouping_cycle) + '):', epoch, 'accuracy:', accuracy, 'loss:', float(loss), "\n")
        return accuracy, loss

    def get_history(self):
        return self.history_accuracy, self.history_loss

    def get_test_data(self):
        return self.test_accuracy_list, self.test_loss_list
