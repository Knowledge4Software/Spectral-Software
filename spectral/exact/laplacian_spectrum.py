from core.interfaces.spectrum import SpectrumComputer
from core.abstractions.spectrum import Spectrum
import numpy as np


class LaplacianSpectrum(SpectrumComputer):
    def compute(self, graph):
        A = graph.to_adjacency_matrix()
        if A is None:
            return Spectrum([])

        D = np.diag(A.sum(axis=1))
        L = D - A
        vals, vecs = np.linalg.eig(L)

        return Spectrum(vals, vecs)