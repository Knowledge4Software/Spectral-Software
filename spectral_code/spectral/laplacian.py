import numpy as np
import networkx as nx

from spectral_code.spectral.base import SpectralAnalyzer
from spectral_code.spectral.solvers.base import EigenSolver


class LaplacianSpectrum(SpectralAnalyzer):
    def __init__(self, solver: EigenSolver):
        self.solver = solver

    def analyze(self, graph: nx.Graph):
        A = nx.to_numpy_array(graph)
        D = np.diag(A.sum(axis=1))
        L = D - A

        eigvals, eigvecs = self.solver.solve(L)

        return eigvals, eigvecs