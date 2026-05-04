from core.interfaces.graph_builder import GraphBuilder
from core.abstractions.graph import Graph
from graph.representations.adjacency_list import AdjacencyListGraph


class ASTGraphBuilder(GraphBuilder):
    def build(self, code_unit) -> Graph:
        g = Graph(AdjacencyListGraph())

        ast = code_unit.metadata.get("ast")
        if ast is None:
            return g

        for node in ast.body:
            g.add_node(id(node))

        return g