import numpy as np
import networkx as nx
from .base import BaseSpectralTransform, SpectralAnalyzer
from .solvers.dense import DenseEigenSolver
from .solvers.base import EigenSolver

class NormalizedLaplacianTransform(BaseSpectralTransform):
    def __init__(self, solver=None):
        self.solver = solver or DenseEigenSolver()

    def transform(self, graph: nx.Graph) -> np.ndarray:
        if graph is None or graph.number_of_nodes() == 0:
            return np.array([], dtype=np.float64)
        
        # Ensure graph is undirected for valid symmetric laplacian
        if graph.is_directed():
            graph = graph.to_undirected()
            
        L = nx.normalized_laplacian_matrix(graph).toarray()
        
        eigenvalues, _ = self.solver.solve(L)
        return np.sort(eigenvalues)

class NormalizedLaplacianSpectrum(SpectralAnalyzer):
    def __init__(self, solver: EigenSolver):
        self.solver = solver

    def analyze(self, graph: nx.Graph):
        if graph is None or graph.number_of_nodes() == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        if graph.is_directed():
            graph = graph.to_undirected()

        L = nx.normalized_laplacian_matrix(graph).toarray()
        eigvals, eigvecs = self.solver.solve(L)
        return eigvals, eigvecs