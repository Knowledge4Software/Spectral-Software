import networkx as nx
from .base import GraphBuilder
from spectral_code.parsing.tree_sitter_parser import TreeSitterParser


class ASTGraphBuilder(GraphBuilder):
    def __init__(self):
        self.parser = TreeSitterParser()

    def build(self, code: str, lang: str = "python") -> nx.Graph:
        tree = self.parser.parse(code, lang)
        root = tree.root_node

        graph = nx.DiGraph()
        self._add_node(graph, root)

        return graph

    def _add_node(self, graph: nx.DiGraph, node, parent_id=None):
        node_id = id(node)

        graph.add_node(
            node_id,
            label=node.type,  # Tree-sitter equivalent of ast node type
            start=node.start_point,
            end=node.end_point
        )

        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for child in node.children:
            self._add_node(graph, child, node_id)