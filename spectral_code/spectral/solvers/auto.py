import numpy as np
from scipy.sparse import issparse

from .dense import DenseEigenSolver
from .sparse import SparseEigenSolver
from .base import EigenSolver


class AutoEigenSolver(EigenSolver):
    def __init__(self, threshold=300, k=10):
        self.threshold = threshold
        self.k = k

        self.dense = DenseEigenSolver()
        self.sparse = SparseEigenSolver(k=k)

    def solve(self, matrix):
        n = matrix.shape[0]

        if n < self.threshold:
            return self.dense.solve(matrix)
        else:
            return self.sparse.solve(matrix)