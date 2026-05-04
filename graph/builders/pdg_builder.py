from core.interfaces.graph_builder import GraphBuilder
from core.abstractions.graph import Graph
from graph.representations.adjacency_list import AdjacencyListGraph


class DFGBuilder(GraphBuilder):
    def build(self, code_unit) -> Graph:
        return Graph(AdjacencyListGraph())