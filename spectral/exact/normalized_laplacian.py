from core.interfaces.spectrum import SpectrumComputer
from core.abstractions.spectrum import Spectrum
import numpy as np


class NormalizedLaplacianSpectrum(SpectrumComputer):
    def compute(self, graph):
        A = graph.to_adjacency_matrix()
        if A is None:
            return Spectrum([])

        D = np.diag(A.sum(axis=1))
        D_inv = np.linalg.inv(D + np.eye(len(D)))
        L = np.eye(len(A)) - D_inv @ A

        vals, vecs = np.linalg.eig(L)
        return Spectrum(vals, vecs)