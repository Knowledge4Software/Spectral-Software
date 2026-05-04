from core.interfaces.representation import GraphRepresentation
import numpy as np


class AdjacencyMatrixGraph(GraphRepresentation):
    def __init__(self, size=0):
        self.matrix = np.zeros((size, size))

    def add_node(self, node_id, **attrs):
        pass

    def add_edge(self, source, target, **attrs):
        self.matrix[source][target] = 1

    def get_nodes(self):
        return range(len(self.matrix))

    def get_edges(self):
        return zip(*self.matrix.nonzero())

    def get_node_attributes(self, node_id):
        return {}

    def get_edge_attributes(self, source, target):
        return {"weight": self.matrix[source][target]}

    def to_adjacency_matrix(self):
        return self.matrix

    def copy(self):
        g = AdjacencyMatrixGraph()
        g.matrix = self.matrix.copy()
        return g