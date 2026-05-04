from abc import ABC, abstractmethod
from core.abstractions.graph import Graph


class GraphOperator(ABC):
    """Base class for all graph operators."""
    pass


class UnaryGraphOperator(GraphOperator):
    """Operator acting on a single graph."""

    @abstractmethod
    def apply(self, g: Graph) -> Graph:
        pass


class BinaryGraphOperator(GraphOperator):
    """Operator acting on two graphs."""

    @abstractmethod
    def apply(self, g1: Graph, g2: Graph) -> Graph:
        pass


class TernaryGraphOperator(GraphOperator):
    """Operator acting on three graphs."""

    @abstractmethod
    def apply(self, g1: Graph, g2: Graph, g3: Graph) -> Graph:
        pass


class VariadicGraphOperator(GraphOperator):
    """Operator acting on an arbitrary number of graphs."""

    @abstractmethod
    def apply(self, *graphs: Graph) -> Graph:
        pass