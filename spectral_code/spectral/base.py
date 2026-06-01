import numpy as np
import networkx as nx

class SpectralAnalyzer:
    def analyze(self, graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

class BaseSpectralTransform:
    def transform(self, graph: nx.Graph) -> np.ndarray:
        raise NotImplementedError