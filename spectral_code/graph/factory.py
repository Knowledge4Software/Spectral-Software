from spectral_code.utils.errors import unknown_component_error
from spectral_code.graph.joern_graph import JoernGraphBuilder

def create_graph_builder(name: str, **kwargs):
    # Mapping old names and new names directly to JoernGraphBuilder implementation
    valid_names = ["ast", "cfg", "ddg", "pdg", "cpg"]
    
    if name not in valid_names:
        raise unknown_component_error("graph builder", name, valid_names)

    # By passing repr_type, JoernGraphBuilder decides internally what to export
    return JoernGraphBuilder(repr_type=name, **kwargs)
