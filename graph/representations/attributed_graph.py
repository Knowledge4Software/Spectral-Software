from graph.representations.adjacency_list import AdjacencyListGraph


class AttributedGraph(AdjacencyListGraph):
    def __init__(self):
        super().__init__()
        self.node_attrs = {}

    def add_node(self, node_id, **attrs):
        super().add_node(node_id)
        self.node_attrs[node_id] = attrs

    def get_node_attributes(self, node_id):
        return self.node_attrs.get(node_id, {})