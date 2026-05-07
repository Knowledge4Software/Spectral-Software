from spectral_code.utils.errors import unknown_component_error
from spectral_code.graph.ast_graph import ASTGraphBuilder
from spectral_code.graph.cfg_graph import CFGGraphBuilder

GRAPH_REGISTRY = {
    "ast": ASTGraphBuilder,
    "cfg": CFGGraphBuilder,
}


def create_graph_builder(name: str, **kwargs):
    if name not in GRAPH_REGISTRY:
        raise unknown_component_error("graph builder", name, GRAPH_REGISTRY)

    return GRAPH_REGISTRY[name](**kwargs)