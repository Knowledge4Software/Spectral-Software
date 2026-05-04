from abc import ABC
from typing import Any, Iterable, Tuple
from core.interfaces.representation import GraphRepresentation


class Graph(ABC):
    """
    Abstract graph object that wraps a GraphRepresentation.
    Provides a stable interface for all higher-level modules.
    """

    def __init__(self, representation: GraphRepresentation):
        self._repr = representation

    def nodes(self) -> Iterable[Any]:
        return self._repr.get_nodes()

    def edges(self) -> Iterable[Tuple[Any, Any]]:
        return self._repr.get_edges()

    def node_attributes(self, node_id: Any) -> dict:
        return self._repr.get_node_attributes(node_id)

    def edge_attributes(self, source: Any, target: Any) -> dict:
        return self._repr.get_edge_attributes(source, target)

    def add_node(self, node_id: Any, **attrs) -> None:
        self._repr.add_node(node_id, **attrs)

    def add_edge(self, source: Any, target: Any, **attrs) -> None:
        self._repr.add_edge(source, target, **attrs)

    def to_adjacency_matrix(self):
        return self._repr.to_adjacency_matrix()

    def copy(self) -> "Graph":
        return self.__class__(self._repr.copy())

    def number_of_nodes(self) -> int:
        return len(list(self.nodes()))

    def number_of_edges(self) -> int:
        return len(list(self.edges()))