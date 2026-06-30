import os
import numpy as np
from .base import BaseSimilarity

class PSSSimilarity(BaseSimilarity):
    """
    Program Spectral Similarity with an exponential distance kernel.

    The spectrum alignment follows the original PSS idea, but the final
    distance-to-similarity mapping is calibrated like the Wasserstein metric:
    S = exp(-(d / gamma) ** power). The old RBF-style d^2 mapping compressed
    many clone and non-clone pairs into the 0.8-1.0 range because normalized
    spectral distances are usually below 1.
    """
    def __init__(
        self,
        gamma: float | None = None,
        distance_power: float | None = None,
    ):
        self.gamma = float(gamma if gamma is not None else os.getenv("PSS_GAMMA", "0.1"))
        self.distance_power = float(
            distance_power
            if distance_power is not None
            else os.getenv("PSS_DISTANCE_POWER", "1.0")
        )
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.distance_power <= 0:
            raise ValueError("distance_power must be positive.")
    
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
            
        # 2. According to "Program Spectral Similarity":
        # To compare graphs of different sizes (N1 != N2) fairly, we use 
        # linear interpolation to "resample" the smaller spectrum to the size of the larger one.
        # This compares the "Shape" of the connectivity distribution.
        
        len1, len2 = len(ev1_true), len(ev2_true)
        if len1 == len2:
            v1, v2 = ev1_true, ev2_true
        else:
            max_len = max(len1, len2)
            # Resample both to max_len
            v1 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len1), ev1_true)
            v2 = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len2), ev2_true)
        
        # 3. L2-Normalize the resampled vectors
        v1_n = self._l2_normalize(v1)
        v2_n = self._l2_normalize(v2)
        
        # 4. Euclidean distance, then exponential kernel mapping to [0, 1].
        distance = np.linalg.norm(v1_n - v2_n)
        similarity = np.exp(-((distance / self.gamma) ** self.distance_power))
        
        return float(np.clip(similarity, 0.0, 1.0))
