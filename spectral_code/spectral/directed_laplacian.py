import numpy as np
import networkx as nx
import warnings
import scipy.sparse as sp
from .base import BaseSpectralTransform, SpectralAnalyzer
from .solvers.dense import DenseEigenSolver
from .solvers.base import EigenSolver

class DirectedLaplacianSpectrum(SpectralAnalyzer):
    def __init__(self, solver: EigenSolver):
        self.solver = solver

    def analyze(self, graph: nx.Graph):
        if graph is None or graph.number_of_nodes() == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        if not graph.is_directed():
            graph = graph.to_directed()

        try:
            # networkx directed_laplacian_matrix uses PageRank to find a stationary distribution
            # and produces a symmetric directed laplacian.
            # Catch warnings about small graphs in scipy sparse eig finding
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                L = nx.directed_laplacian_matrix(graph, alpha=0.95)
                # L returned by this function is a numpy array (since networkx 2.5+)
                if sp.issparse(L):
                    L = L.toarray()
            
            eigvals, eigvecs = self.solver.solve(L)
            return eigvals, eigvecs
        except Exception as e:
            # Fallback if convergence fails or graph is problematic
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
