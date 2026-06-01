import numpy as np
from .base import BaseSimilarity

class PSSSimilarity(BaseSimilarity):
    """
    Implements the official adapted Program Spectral Similarity (PSS) for function-level graphs.
    Dynamically strips trailing zero-paddings and applies correct paper normalization.
    """
    
    def _strip_padding(self, ev: np.ndarray) -> np.ndarray:
        # Since Laplacian eigenvalues are sorted ascendingly, 
        # any sudden drop to zero or trailing zeros represent artificial padding.
        nonzero_indices = np.nonzero(ev)[0]
        if len(nonzero_indices) == 0:
            return np.array([])
        return ev[:nonzero_indices[-1] + 1]

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        # 1. Strip trailing zero-paddings to recover true graph spectrum size
        ev1_true = self._strip_padding(ev1)
        ev2_true = self._strip_padding(ev2)
        
        if len(ev1_true) == 0 or len(ev2_true) == 0:
            return 0.0
            
        # 2. According to the PSS paper: first normalize the entire spectrum
        v1_norm = self._l2_normalize(ev1_true)
        v2_norm = self._l2_normalize(ev2_true)
        
        # 3. Then compute distance only up to the minimum length (Equation 2 in paper)
        min_length = min(len(v1_norm), len(v2_norm))
        v1_truncated = v1_norm[:min_length]
        v2_truncated = v2_norm[:min_length]
        
        # 4. Euclidean Distance Calculation on truncated normalized vectors
        distance = np.linalg.norm(v1_truncated - v2_truncated)
        
        # 5. Official Normalization to [0, 1] for a single layer: (sqrt(2) - distance) / sqrt(2)
        similarity = (np.sqrt(2) - distance) / np.sqrt(2)
        
        return float(np.clip(similarity, 0.0, 1.0))