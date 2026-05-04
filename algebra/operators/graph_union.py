from core.interfaces.graph_operator import BinaryGraphOperator


class GraphUnion(BinaryGraphOperator):
    def apply(self, g1, g2):
        return GraphAddition().apply(g1, g2)