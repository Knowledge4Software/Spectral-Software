import numpy as np

def normalize_matrix(M: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(M)
    return M if norm == 0 else M / norm