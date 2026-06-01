from spectral_code.utils.errors import unknown_component_error
from spectral_code.visualization.matplotlib_renderer import MatplotlibGraphVisualizer

VISUALIZER_REGISTRY = {
    "matplotlib": MatplotlibGraphVisualizer,
    None: lambda **kwargs: None,
}


def create_graph_visualizer(name: str | None, **kwargs):
    if name not in VISUALIZER_REGISTRY:
        raise unknown_component_error("visualizer", name, VISUALIZER_REGISTRY)

    cls = VISUALIZER_REGISTRY[name]
    return cls(**kwargs) if callable(cls) else cls
