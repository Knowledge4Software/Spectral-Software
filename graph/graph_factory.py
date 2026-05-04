from core.interfaces.graph_builder import GraphBuilder


class GraphFactory:
    def __init__(self):
        self._builders = {}

    def register(self, name: str, builder: type[GraphBuilder]):
        self._builders[name] = builder

    def get_builder(self, name: str) -> GraphBuilder:
        if name not in self._builders:
            raise ValueError(f"Unknown graph builder: {name}")
        return self._builders[name]()