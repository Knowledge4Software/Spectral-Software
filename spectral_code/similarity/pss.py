import os
from collections import defaultdict
from collections.abc import Sequence

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

    @staticmethod
    def _resample_rows(matrix: np.ndarray, target_length: int) -> np.ndarray:
        """Linearly resample equally sized spectra without a Python row loop."""
        source_length = matrix.shape[1]
        if source_length == target_length:
            return matrix
        positions = np.linspace(0.0, source_length - 1.0, target_length)
        lower = np.floor(positions).astype(np.int64)
        upper = np.minimum(lower + 1, source_length - 1)
        weight = positions - lower
        return matrix[:, lower] * (1.0 - weight) + matrix[:, upper] * weight

    def compute_many(
        self,
        left_spectra: Sequence[np.ndarray],
        right_spectra: Sequence[np.ndarray],
        *,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Vectorized equivalent of :meth:`compute` for large pair tables.

        Pairs are grouped by their two true spectrum lengths, so every group
        uses exactly the same interpolation grid as the scalar PSS algorithm.
        This avoids millions of small NumPy allocations in full RQ1 runs.
        """
        if len(left_spectra) != len(right_spectra):
            raise ValueError("left_spectra and right_spectra must have equal length")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        left = [self._strip_padding(np.asarray(values, dtype=np.float64)) for values in left_spectra]
        right = [self._strip_padding(np.asarray(values, dtype=np.float64)) for values in right_spectra]
        result = np.zeros(len(left), dtype=np.float64)
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (left_values, right_values) in enumerate(zip(left, right)):
            if len(left_values) and len(right_values):
                groups[(len(left_values), len(right_values))].append(index)

        for (left_length, right_length), indices in groups.items():
            target_length = max(left_length, right_length)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                left_matrix = np.stack([left[index] for index in batch_indices])
                right_matrix = np.stack([right[index] for index in batch_indices])
                left_matrix = self._resample_rows(left_matrix, target_length)
                right_matrix = self._resample_rows(right_matrix, target_length)
                left_norm = np.linalg.norm(left_matrix, axis=1, keepdims=True)
                right_norm = np.linalg.norm(right_matrix, axis=1, keepdims=True)
                left_matrix = np.divide(
                    left_matrix, left_norm, out=np.zeros_like(left_matrix), where=left_norm != 0
                )
                right_matrix = np.divide(
                    right_matrix, right_norm, out=np.zeros_like(right_matrix), where=right_norm != 0
                )
                distance = np.linalg.norm(left_matrix - right_matrix, axis=1)
                result[np.asarray(batch_indices)] = np.clip(
                    np.exp(-((distance / self.gamma) ** self.distance_power)), 0.0, 1.0
                )
        return result
