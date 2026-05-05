from spectral_code.utils.errors import unknown_component_error
from spectral_code.graph.ast_graph import ASTGraphBuilder

GRAPH_REGISTRY = {
    "ast": ASTGraphBuilder,
}


def create_graph_builder(name: str, **kwargs):
    if name not in GRAPH_REGISTRY:
        raise unknown_component_error("graph builder", name, GRAPH_REGISTRY)

    return GRAPH_REGISTRY[name](**kwargs)