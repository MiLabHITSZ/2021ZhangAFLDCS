import tensorflow as tf


class Node:
    def __init__(self, node_type, node_id, node_coordinate):
        self.node_type = node_type
        self.node_id = node_id
        self.rho = 0
        self.delta = 0
        self.gamma = 0
        self.nearest_higher_rho_node = None
        self.node_c = node_coordinate

        self.cluster_id = -1

    def reset(self):
        self.rho = 0
        self.delta = 0
        self.gamma = 0
        self.nearest_higher_rho_node = None

        self.cluster_id = -1

    def set_node_id(self, new_node_id):
        self.node_id = new_node_id

    def get_node_id(self):
        return self.node_id

    def set_rho(self, new_rho):
        self.rho = new_rho
        self.gamma = self.rho * self.delta

    def get_rho(self):
        return self.rho

    def set_delta(self, new_delta):
        self.delta = new_delta
        self.gamma = self.rho * self.delta

    def get_delta(self):
        return self.delta

    def get_gama(self):
        return self.gamma

    def set_nearest_higher_rho_node(self, new_nearest_higher_rho_node):
        self.nearest_higher_rho_node = new_nearest_higher_rho_node

    def get_nearest_higher_rho_node(self):
        return self.nearest_higher_rho_node

    def set_node_c(self, new_node_c):
        self.node_c = new_node_c

    def get_node_c(self):
        return self.node_c

    def set_cluster_id(self, new_cluster_id):
        self.cluster_id = new_cluster_id

    def get_cluster_id(self):
        return self.cluster_id
