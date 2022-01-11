import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import copy
import math

import DensityPeaks.Node as Node
from utils import Tools


class NodesManager:
    def __init__(self, nodes_type, nodes):
        self.nodes_type = nodes_type
        self.nodes = nodes
        self.nodes_num = len(self.nodes)
        self.nodes_rho = [-1 for _ in range(self.nodes_num)]
        self.nodes_delta = [-1 for _ in range(self.nodes_num)]
        self.nodes_gama = [-1 for _ in range(self.nodes_num)]
        self.distance_matrix = [[-1 for _ in range(self.nodes_num)] for _ in range(self.nodes_num)]
        self.distance_list = []
        self.cluster_centers = []
        self.clusters = []

        # 计算所有点之间的距离并存入distance_matrix中
        print("nodes_num =", self.nodes_num, ":", end=" ")
        for i in range(self.nodes_num):
            if i % 50 == 0:
                print(i, end=" ")
            for j in range(self.nodes_num):
                if i == j:
                    self.distance_matrix[i][j] = 0
                elif self.distance_matrix[i][j] == -1:
                    distance = self.calculate_nodes_distance(self.nodes[i], self.nodes[j])
                    self.distance_matrix[i][j] = distance
                    self.distance_matrix[j][i] = distance
        print()
        self.calculate_distance_list()

    def add_nodes(self, new_nodes):
        for node in new_nodes:
            self.nodes.append(node)
            self.nodes_num += 1
            self.nodes_rho.append(-1)
            self.nodes_delta.append(-1)
            self.nodes_gama.append(-1)
            self.distance_matrix.append([-1 for _ in range(self.nodes_num - 1)])
            for i in range(self.nodes_num):
                self.distance_matrix[i].append(-1)

            # 计算新加入的点与所有点之间的距离并存入distance_matrix中
            for i in range(self.nodes_num):
                if i == self.nodes_num - 1:
                    self.distance_matrix[i][i] = 0
                elif self.distance_matrix[i][self.nodes_num - 1] == -1:
                    distance = self.calculate_nodes_distance(self.nodes[i], self.nodes[self.nodes_num - 1])
                    self.distance_matrix[i][self.nodes_num - 1] = distance
                    self.distance_matrix[self.nodes_num - 1][i] = distance

            # 更新distance_list
            self.calculate_distance_list()

        for node in self.nodes:
            node.reset()
        self.cluster_centers = []
        self.clusters = []

    def set_distance_matrix(self, distance_matrix):
        self.distance_matrix = distance_matrix
        # 更新distance_list
        self.calculate_distance_list()

    def get_distance_matrix(self):
        return self.distance_matrix

    def calculate_distance_list(self):
        # 计算出distance_list
        self.distance_list = []
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if i < j:
                    self.distance_list.append(self.distance_matrix[i][j])
        self.distance_list.sort()

    def get_distance_list(self):
        return self.distance_list

    def reset(self):
        self.nodes_num = len(self.nodes)
        self.nodes_rho = [-1 for _ in range(self.nodes_num)]
        self.nodes_delta = [-1 for _ in range(self.nodes_num)]
        self.nodes_gama = [-1 for _ in range(self.nodes_num)]
        self.distance_matrix = [[-1 for _ in range(self.nodes_num)] for _ in range(self.nodes_num)]
        self.cluster_centers = []
        self.clusters = []

    def reset_nodes(self):
        for node in self.nodes:
            node.reset()

    def calculate_nodes_distance(self, node1, node2):
        if self.nodes_type == "2_dimensional_point":
            xy = node1.get_node_c() - node2.get_node_c()
            distance = math.hypot(xy[0], xy[1])
        elif self.nodes_type == "3_dimensional_point":
            xyz = node1.get_node_c() - node2.get_node_c()
            distance = math.hypot(xyz[0], xyz[1], xyz[2])
        elif self.nodes_type == "multidimensional_point":
            c1 = node1.get_node_c()
            c2 = node2.get_node_c()
            c = c1 - c2
            # if len(c.shape) == 1:
            #     d = np.linalg.norm(c, axis=0, keepdims=True)
            # elif len(c.shape) == 2:
            #     d1 = np.linalg.norm(c, axis=1, keepdims=True)
            #     d = np.linalg.norm(d1, axis=0, keepdims=True)
            # else:
            #     print("Unexpected coordinate!")
            #     d = 0
            distance = Tools.l2_regularization_of_array(c)
            # s = 0
            # for i in range(c1.shape[0]):
            #     for j in range(c1.shape[1]):
            #         s += (c1[i][j] - c2[i][j]) ** 2
            # distance = math.sqrt(s)
            # distance = float(d)
        elif self.nodes_type == "multidimensional_list_point":
            c1 = node1.get_node_c()
            c2 = node2.get_node_c()
            distance_list = []
            for i in range(len(c1)):
                c = c1[i] - c2[i]
                d = Tools.l2_regularization_of_array(c)
                distance_list.append(d)
            distance_array = np.array(distance_list)
            distance = Tools.l2_regularization_of_array(distance_array)
        elif self.nodes_type == "multidimensional_reduction_point":
            xyz = node1.get_node_c() - node2.get_node_c()
            td = xyz[0]
            for i in range(len(xyz)):
                if i > 0:
                    td = math.hypot(td, xyz[i])
            distance = td
            # print("distance =", distance)
        else:
            print("Error nodes type!!")
            distance = -1
        return distance

    def calculate_nodes_rho(self, dc, mode):
        # print("nodes_rho:", end="")
        for i in range(self.nodes_num):
            rho = 0
            for j in range(self.nodes_num):
                if mode == 1 and 0 < self.distance_matrix[i][j] < dc:
                    rho += 1
                if mode == 2 and i != j:
                    rho += math.exp(-((self.distance_matrix[i][j] / dc) ** 2))
            self.nodes[i].set_rho(rho)
            self.nodes_rho[i] = rho
            self.nodes_gama[i] = self.nodes_rho[i] * self.nodes_delta[i]
            # print(self.nodes[i].get_rho(), end=" ")
        # print("")

    def calculate_nodes_delta(self):
        # print("nodes_delta:", end="")
        for i in range(self.nodes_num):
            delta = float('inf')
            nearest_higher_rho_node = None
            for j in range(self.nodes_num):
                if i != j and self.nodes[i].get_rho() < self.nodes[j].get_rho() and self.distance_matrix[i][j] < delta:
                    delta = self.distance_matrix[i][j]
                    nearest_higher_rho_node = self.nodes[j]
            # 对于rho最大的点，其delta取其与距它最远的点的距离
            if delta == float('inf'):
                # print("\n           m", end="")
                delta = max(self.distance_matrix[i])
                nearest_higher_rho_node = None

            self.nodes[i].set_delta(delta)
            self.nodes[i].set_nearest_higher_rho_node(nearest_higher_rho_node)
            self.nodes_delta[i] = delta
            self.nodes_gama[i] = self.nodes_rho[i] * self.nodes_delta[i]
            # print(round(self.nodes[i].get_delta(), 2), end=" ")
        # print("")

    def calculate_average_rho(self):
        # print("nodes_gama:", end="")
        # for node in self.nodes:
        #     print(round(node.get_gama(), 2), end=" ")
        # print("")
        average_rho = np.mean(self.nodes_rho)
        return average_rho

    def calculate_average_delta(self):
        # print("nodes_delta:", end="")
        # for node in self.nodes:
        #     print(round(node.get_delta(), 2), end=" ")
        # print("")
        average_delta = np.mean(self.nodes_delta)
        return average_delta

    def find_cluster_center(self, threshold):
        average_rho = np.mean(self.nodes_rho)
        average_delta = np.mean(self.nodes_delta)
        average_gama = np.mean(self.nodes_gama)
        # print("average_rho:", average_rho)
        # print("average_delta:", average_delta)
        # print("average_gama:", average_gama)
        c = 0
        cluster_id_count = 0
        # print("cluster center:", end="")
        for node in self.nodes:
            # if node.get_gama() > (average_gama * threshold):
            if node.get_delta() > (average_delta * threshold):
                self.cluster_centers.append(node)
                self.clusters.append([node])
                node.set_cluster_id(cluster_id_count)
                cluster_id_count += 1
                # print(node.get_node_id(), "=", round(node.get_gama(), 2), end="  |  ")
            c += 1
        # print("")

    def clustering(self):
        # 所有的点都加入与它最近的、且密度大于它的点所在的聚类
        clustered_nodes = []
        for cc in self.cluster_centers:
            clustered_nodes.append(cc)
        while len(clustered_nodes) < self.nodes_num:
            for node in self.nodes:
                if (node not in clustered_nodes) and (node.get_nearest_higher_rho_node().get_cluster_id() != -1):
                    node.set_cluster_id(node.get_nearest_higher_rho_node().get_cluster_id())
                    clustered_nodes.append(node)
                    self.clusters[node.get_cluster_id()].append(node)
        # for node in self.nodes:
        #     print(node.get_cluster_id(), end=" ")
        # print("")
        cluster_list = [[] for _ in range(len(self.clusters))]
        for n in range(self.nodes_num):
            cluster_list[int(self.nodes[n].get_cluster_id())].append(self.nodes[n].get_node_id())
        for cl in range(len(cluster_list)):
            c_l = copy.deepcopy(cluster_list[cl])
            c_l.sort()

    def show_distance_matrix(self):
        print("------------------------------------------------------------------------------------------------------")
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                print('{str:r<{len}}'.format(str=str(round(self.distance_matrix[i][j], 4)), len=4), end="\t")
            print("")
        print("------------------------------------------------------------------------------------------------------")

    def show_nodes_distribution(self):
        x_list = []
        y_list = []
        for node in self.nodes:
            x_list.append(node.get_node_c()[0])
            y_list.append(node.get_node_c()[1])
        plt.plot(x_list, y_list, 'o', color='b')
        plt.show()

    def show_nodes_rho_and_delta(self):
        x_list = []
        y_list = []
        for node in self.nodes:
            x_list.append(node.get_rho())
            y_list.append(node.get_delta())
        plt.plot(x_list, y_list, '+', color='r')
        plt.show()

    def get_clusters(self):
        return self.clusters
