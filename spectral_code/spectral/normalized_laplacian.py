from __future__ import annotations

import numpy as np
import networkx as nx

from spectral_code.spectral.base import SpectralAnalyzer
from spectral_code.spectral.solvers.base import EigenSolver


class NormalizedLaplacianSpectrum(SpectralAnalyzer):
    def __init__(self, solver: EigenSolver):
        self.solver = solver

    def analyze(self, graph: nx.Graph):
        # AST/CFG in this project are directed; for spectral comparison we symmetrize.
        G = graph.to_undirected() if graph.is_directed() else graph

        A = nx.to_numpy_array(G, dtype=float)
        if A.size == 0:
            return np.array([]), np.array([[]])

        degrees = A.sum(axis=1)
        inv_sqrt_deg = np.zeros_like(degrees)
        nonzero = degrees > 0
        inv_sqrt_deg[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

        D_inv_sqrt = np.diag(inv_sqrt_deg)
        I = np.eye(A.shape[0], dtype=float)

        # L = I - D^{-1/2} A D^{-1/2}
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        eigvals, eigvecs = self.solver.solve(L)
        return eigvals, eigvecs
