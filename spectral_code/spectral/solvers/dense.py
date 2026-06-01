import numpy as np
from .base import EigenSolver


class DenseEigenSolver(EigenSolver):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, matrix):
        return np.linalg.eigh(matrix, **self.kwargs)