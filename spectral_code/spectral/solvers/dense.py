import numpy as np
from .base import EigenSolver


class DenseEigenSolver(EigenSolver):
    def __init__(self, *, eigenvalues_only: bool = False, **kwargs):
        """Dense symmetric eigensolver.

        Spectrum extraction persists only eigenvalues.  Computing the full
        eigenvector matrix in that path is unnecessarily expensive in both
        memory and time, particularly for the large CPGs produced by C#.
        Keep the historical ``eigh`` behaviour by default for callers that
        genuinely need eigenvectors, but expose the exact eigenvalue-only
        LAPACK routine for extraction.
        """
        self.eigenvalues_only = eigenvalues_only
        self.kwargs = kwargs

    def solve(self, matrix):
        if self.eigenvalues_only:
            return np.linalg.eigvalsh(matrix, **self.kwargs), None
        return np.linalg.eigh(matrix, **self.kwargs)
