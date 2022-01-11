import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import copy
import math

import DensityPeaks.Node as Node
import DensityPeaks.NodesManager as NodesManager

import utils.Tools as Tools


class ClusterWithDensityPeaks:
    def __init__(self, init_nodes_dict, init_weights, nodes_type, use_weight_layer, dc, t):
        self.init_nodes_dict = init_nodes_dict
        self.init_weights = init_weights
        self.nodes_type = nodes_type
        self.use_weight_layer = use_weight_layer
        self.dc = dc
        self.t = t

        self.nodes = []
        self.add_nodes(self.init_nodes_dict)

        self.nodes_manager = NodesManager.NodesManager(self.nodes_type, self.nodes)

    def clustering(self):
        distance_matrix = self.nodes_manager.get_distance_matrix()
        self.nodes_manager.set_distance_matrix(distance_matrix)
        # self.nodes_manager.show_distance_matrix()
        # self.nodes_manager.show_nodes_distribution()
        if self.dc >= 0:
            self.nodes_manager.calculate_nodes_rho(self.dc, 1)
        else:
            n = len(self.nodes)
            dc_index = int((n * (n - 1) / 2) * (-self.dc))
            distance_list = self.nodes_manager.get_distance_list()
            dc = distance_list[dc_index]
            # print("dc =", dc, "dc_index =", dc_index)
            self.nodes_manager.calculate_nodes_rho(dc, 2)
        self.nodes_manager.calculate_nodes_delta()
        # self.show_clusters()
        self.nodes_manager.find_cluster_center(self.t)
        self.nodes_manager.clustering()
        return self.nodes_manager.get_clusters()

    def show_clusters(self):
        # result = [self.dc, self.t]
        result = []
        for c in range(len(self.nodes_manager.clusters)):
            # print("Cluster", c, ":", len(self.nodes_manager.clusters[c]), self.nodes_manager.cluster_centers[c].get_gama())
            result.append(len(self.nodes_manager.clusters[c]))

        # self.nodes_manager.show_nodes_rho_and_delta()
        # print(result)
        return result

    def get_clusters(self):
        clusters = {}
        for c in range(len(self.nodes_manager.clusters)):
            clusters[c] = []
            for c_node in self.nodes_manager.clusters[c]:
                clusters[c].append(c_node.get_node_id())
        return clusters

    def add_nodes(self, new_nodes_dict):
        new_nodes = []
        for key in new_nodes_dict:
            if self.nodes_type == "multidimensional_point":
                # print("layers:", type(new_nodes_dict[key]), len(new_nodes_dict[key]))
                # for i in range(len(new_nodes_dict[key])):
                #     print(i, ":", type(new_nodes_dict[key][i]), new_nodes_dict[key][i].shape)

                weights_difference = new_nodes_dict[key][self.use_weight_layer] - self.init_weights[self.use_weight_layer]
                new_nodes.append(Node.Node(self.nodes_type, key, weights_difference))
            elif self.nodes_type == "multidimensional_list_point":
                weights_difference_list = [new_nodes_dict[key][0] - self.init_weights[0],
                                           new_nodes_dict[key][2] - self.init_weights[2],
                                           new_nodes_dict[key][4] - self.init_weights[4]]
                new_nodes.append(Node.Node(self.nodes_type, key, weights_difference_list))
            elif self.nodes_type == "multidimensional_reduction_point":
                weights_difference_list = [new_nodes_dict[key][0] - self.init_weights[0],
                                           new_nodes_dict[key][2] - self.init_weights[2],
                                           new_nodes_dict[key][4] - self.init_weights[4]]
                node_c = []
                for weights_difference in weights_difference_list:
                    node_c.append(Tools.l2_regularization_of_array(weights_difference))
                # print(np.array(node_c))
                new_nodes.append(Node.Node(self.nodes_type, key, np.array(node_c)))
            else:
                print("Error nodes type!!")

        if len(self.nodes) > 0:
            self.nodes_manager.add_nodes(new_nodes)
        else:
            self.nodes.extend(new_nodes)

    def set_distance_matrix(self, new_distance_matrix):
        self.nodes_manager.set_distance_matrix(new_distance_matrix)

    def get_distance_matrix(self):
        return self.nodes_manager.get_distance_matrix()

    def set_dc(self, new_dc):
        self.dc = new_dc

    def set_t(self, new_t):
        self.t = new_t

    def reset(self):
        self.nodes_manager.reset()
        self.nodes_manager.reset_nodes()
