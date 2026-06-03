import networkx as nx
from typing import Optional

class GEDSimilarity:
    """
    Computes Graph Edit Distance (GED) between two graphs.
    Since GED is NP-hard, this is primarily useful for smaller graphs
    to validate structural similarity.
    """

    def __init__(self, timeout: Optional[float] = None):
        """
        Args:
            timeout: Maximum time in seconds to compute GED.
        """
        self.timeout = timeout

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """
        Computes the Graph Edit Distance.
        Returns the raw edit distance (number of edits).
        """
        # Note: nx.graph_edit_distance is expensive.
        # For directed graphs use nx.graph_edit_distance(g1, g2)
        # It handles both directed and undirected.
        distance = nx.graph_edit_distance(g1, g2, timeout=self.timeout)
        return distance if distance is not None else float('inf')

    def compute_normalized(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """
        Normalizes GED to a [0, 1] similarity score.
        Using a more robust normalization: 1 / (1 + d/max_size)
        """
        dist = self.compute(g1, g2)
        if dist == float('inf'):
            return 0.0
            
        max_possible_edits = max(
            g1.number_of_nodes() + g1.number_of_edges(),
            g2.number_of_nodes() + g2.number_of_edges()
        )
        
        if max_possible_edits == 0:
            return 1.0
            
        # Using exponential decay or simple ratio
        # Let's use a ratio that reflects structural overlap better
        similarity = 1.0 - (dist / max_possible_edits)
        return max(0.0, similarity)
