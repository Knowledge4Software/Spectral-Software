import numpy as np

from spectral_code.spectral.solvers.dense import DenseEigenSolver


def test_dense_eigenvalue_only_matches_full_symmetric_spectrum() -> None:
    matrix = np.array([[3.0, -1.0, 0.5], [-1.0, 2.0, 0.0], [0.5, 0.0, 1.0]])

    expected, _ = DenseEigenSolver().solve(matrix)
    actual, vectors = DenseEigenSolver(eigenvalues_only=True).solve(matrix)

    assert vectors is None
    np.testing.assert_allclose(actual, expected)
