from core.interfaces.representation import GraphRepresentation


class AdjacencyListGraph(GraphRepresentation):
    def __init__(self):
        self._adj = {}

    def add_node(self, node_id, **attrs):
        self._adj.setdefault(node_id, {})

    def add_edge(self, source, target, **attrs):
        self._adj.setdefault(source, {})
        self._adj[source][target] = attrs

    def get_nodes(self):
        return self._adj.keys()

    def get_edges(self):
        for s in self._adj:
            for t in self._adj[s]:
                yield (s, t)

    def get_node_attributes(self, node_id):
        return {}

    def get_edge_attributes(self, source, target):
        return self._adj[source][target]

    def to_adjacency_matrix(self):
        return None  # placeholder

    def copy(self):
        new = AdjacencyListGraph()
        new._adj = {k: v.copy() for k, v in self._adj.items()}
        return new