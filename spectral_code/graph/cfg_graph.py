# spectral_code/graph/cfg_graph.py

import networkx as nx
from spectral_code.graph.base import GraphBuilder
from py2cfg import CFGBuilder


class CFGGraphBuilder(GraphBuilder):
    def __init__(self):
        super().__init__()
        self.builder = CFGBuilder()

    def build(self, code: str, lang: str = "python") -> nx.Graph:
        if lang != "python":
            raise NotImplementedError(
                "CFG builder currently only supports Python"
            )

        cfg = self.builder.build_from_src("code", code)

        G = nx.DiGraph()

        visited = set()

        def visit(block):
            if block is None or block.id in visited:
                return

            visited.add(block.id)

            label = "\n".join(
                str(stmt) for stmt in block.statements
            ) or f"Block {block.id}"

            G.add_node(
                block.id,
                label=label,
                block_type=type(block).__name__,
            )

            for exit_edge in block.exits:
                target = exit_edge.target
                if target:
                    G.add_edge(
                        block.id,
                        target.id,
                    )
                    visit(target)

        visit(cfg.entryblock)

        return G