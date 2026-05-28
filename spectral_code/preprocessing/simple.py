import networkx as nx
from .base import Preprocessor

class SimpleGraphPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def process(self, graph: nx.DiGraph) -> nx.DiGraph:
        if graph is None:
            return nx.create_empty_copy(graph)
            
        cleaned_graph = graph.copy()
        
        isolated_nodes = [node for node in cleaned_graph.nodes() if cleaned_graph.in_degree(node) == 0 and cleaned_graph.out_degree(node) == 0]
        cleaned_graph.remove_nodes_from(isolated_nodes)
        
        return cleaned_graph