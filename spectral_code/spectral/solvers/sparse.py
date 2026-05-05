import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from .base import EigenSolver

class SparseEigenSolver(EigenSolver):
    def __init__(self, k=10, which="SM", **kwargs):
        if k is None or k <= 0:
            raise ValueError(f"k must be positive for eigsh, got {k}")

        self.k = k
        self.which = which
        self.kwargs = kwargs
        
    def solve(self, matrix):
        sparse_matrix = csr_matrix(matrix)

        eigvals, eigvecs = eigsh(
            sparse_matrix,
            k=self.k,
            which=self.which,
            **self.kwargs
        )

        return eigvals, eigvecs