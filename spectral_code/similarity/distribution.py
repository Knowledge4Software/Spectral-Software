import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde, wasserstein_distance

from .base import BaseSimilarity


def _as_finite_1d(values: np.ndarray) -> np.ndarray:
    """Return a flattened float array with NaN/Inf values removed."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


class WassersteinSimilarity(BaseSimilarity):
    """
    Earth Mover's Distance similarity over 1D eigenvalue spectra.

    The raw 1D Wasserstein distance is mapped to [0, 1] with an exponential
    kernel: S = exp(-D / gamma). Smaller gamma values penalize structural
    mismatches more aggressively.
    """
    def __init__(self, gamma: float = 0.1):
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.gamma = float(gamma)

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        ev1 = _as_finite_1d(ev1)
        ev2 = _as_finite_1d(ev2)
        if ev1.size == 0 or ev2.size == 0:
            return 0.0

        distance = wasserstein_distance(ev1, ev2)
        similarity = np.exp(-distance / self.gamma)
        return float(np.clip(similarity, 0.0, 1.0))


class JensenShannonSimilarity(BaseSimilarity):
    """
    KDE-based Jensen-Shannon divergence similarity for eigenvalue spectra.

    Each discrete spectrum is first converted into a continuous density with a
    Gaussian KDE on a shared grid. The sampled densities are normalized into
    probability vectors, Jensen-Shannon divergence is computed, and similarity
    is returned as S = 1 - JSD.
    """
    def __init__(
        self,
        grid_size: int = 512,
        bandwidth_method: str | float | None = None,
        range_min: float | None = None,
        range_max: float | None = None,
        epsilon: float = 1e-12,
    ):
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.grid_size = int(grid_size)
        self.bandwidth_method = bandwidth_method
        self.range_min = range_min
        self.range_max = range_max
        self.epsilon = float(epsilon)

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        ev1 = _as_finite_1d(ev1)
        ev2 = _as_finite_1d(ev2)
        if ev1.size == 0 or ev2.size == 0:
            return 0.0

        grid = self._shared_grid(ev1, ev2)
        p1 = self._kde_probability(ev1, grid)
        p2 = self._kde_probability(ev2, grid)

        # scipy returns the Jensen-Shannon distance, i.e. sqrt(divergence).
        # With base=2, the squared divergence is bounded in [0, 1].
        jsd = float(jensenshannon(p1, p2, base=2.0) ** 2)
        similarity = 1.0 - jsd
        return float(np.clip(similarity, 0.0, 1.0))

    def _shared_grid(self, ev1: np.ndarray, ev2: np.ndarray) -> np.ndarray:
        if self.range_min is not None and self.range_max is not None:
            low = float(self.range_min)
            high = float(self.range_max)
        else:
            combined = np.concatenate((ev1, ev2))
            low = float(np.min(combined)) if self.range_min is None else float(self.range_min)
            high = float(np.max(combined)) if self.range_max is None else float(self.range_max)

        if not low < high:
            pad = max(abs(low) * 0.1, 1.0)
            low -= pad
            high += pad
        else:
            pad = 0.05 * (high - low)
            low -= pad
            high += pad

        return np.linspace(low, high, self.grid_size)

    def _kde_probability(self, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        try:
            kde = gaussian_kde(values, bw_method=self.bandwidth_method)
            density = kde(grid)
        except np.linalg.LinAlgError:
            density = self._degenerate_gaussian_density(values, grid)
        except ValueError:
            density = self._degenerate_gaussian_density(values, grid)

        density = np.maximum(density, 0.0) + self.epsilon
        total = np.sum(density)
        if not np.isfinite(total) or total <= 0:
            return np.full(grid.shape, 1.0 / grid.size, dtype=np.float64)
        return density / total

    def _degenerate_gaussian_density(self, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        center = float(np.mean(values))
        grid_span = float(grid[-1] - grid[0])
        bandwidth = max(grid_span / self.grid_size, self.epsilon)
        z = (grid - center) / bandwidth
        return np.exp(-0.5 * z * z)


class FisherInformationSimilarity(BaseSimilarity):
    """
    Fisher-style information distance over spectrum mean and variance.

    The spectrum is summarized as a Gaussian-like distribution. Distance is:
    sqrt(2 * log((var1 + var2 + (mu1 - mu2)^2) / (2 * sigma1 * sigma2)))
    and similarity is S = exp(-gamma * distance).
    """
    def __init__(self, gamma: float | None = None, epsilon: float = 1e-12):
        self.gamma = float(gamma if gamma is not None else os.getenv("FISHER_GAMMA", "1.0"))
        self.epsilon = float(epsilon)
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        ev1 = _as_finite_1d(ev1)
        ev2 = _as_finite_1d(ev2)
        if ev1.size == 0 or ev2.size == 0:
            return 0.0

        mu1 = float(np.mean(ev1))
        mu2 = float(np.mean(ev2))
        var1 = max(float(np.var(ev1)), self.epsilon)
        var2 = max(float(np.var(ev2)), self.epsilon)
        sigma1 = np.sqrt(var1)
        sigma2 = np.sqrt(var2)

        ratio = (var1 + var2 + (mu1 - mu2) ** 2) / (2.0 * sigma1 * sigma2)
        ratio = max(float(ratio), 1.0)
        distance = np.sqrt(2.0 * np.log(ratio))
        similarity = np.exp(-self.gamma * distance)
        return float(np.clip(similarity, 0.0, 1.0))
