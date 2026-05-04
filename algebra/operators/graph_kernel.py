from core.interfaces.graph_operator import BinaryGraphOperator


class GraphKernel(BinaryGraphOperator):
    def apply(self, g1, g2):
        return g1  # placeholder