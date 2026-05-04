from core.interfaces.graph_operator import BinaryGraphOperator


class GraphAddition(BinaryGraphOperator):
    def apply(self, g1, g2):
        g = g1.copy()
        for n in g2.nodes():
            g.add_node(n)
        for u, v in g2.edges():
            g.add_edge(u, v)
        return g