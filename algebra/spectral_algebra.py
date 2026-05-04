from core.abstractions.graph import Graph
from core.interfaces.graph_operator import GraphOperator


class SpectralAlgebra:
    """
    High-level algebraic operations on graphs.
    """

    def apply(self, operator: GraphOperator, *graphs: Graph) -> Graph:
        return operator.apply(*graphs)