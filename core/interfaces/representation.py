from abc import ABC, abstractmethod
from typing import Any, Iterable, Tuple


class GraphRepresentation(ABC):
    """
    Defines how a graph is internally represented.
    This is independent from how the graph is constructed or used.
    """

    @abstractmethod
    def add_node(self, node_id: Any, **attrs) -> None:
        pass

    @abstractmethod
    def add_edge(self, source: Any, target: Any, **attrs) -> None:
        pass

    @abstractmethod
    def get_nodes(self) -> Iterable[Any]:
        pass

    @abstractmethod
    def get_edges(self) -> Iterable[Tuple[Any, Any]]:
        pass

    @abstractmethod
    def get_node_attributes(self, node_id: Any) -> dict:
        pass

    @abstractmethod
    def get_edge_attributes(self, source: Any, target: Any) -> dict:
        pass

    @abstractmethod
    def to_adjacency_matrix(self) -> Any:
        """
        Returns a matrix-like representation (NumPy, sparse, etc.)
        """
        pass

    @abstractmethod
    def copy(self) -> "GraphRepresentation":
        pass