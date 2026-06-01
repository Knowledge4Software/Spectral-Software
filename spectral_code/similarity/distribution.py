import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from .base import BaseSimilarity

class WassersteinSimilarity(BaseSimilarity):
    """
    Computes similarity based on the Earth Mover's Distance (Wasserstein).
    Treats the eigenvalues as empirical distributions.
    """
    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        if len(ev1) == 0 or len(ev2) == 0:
            return 0.0
            
        # compute Earth Mover's Distance
        dist = wasserstein_distance(ev1, ev2)
        
        # Convert distance [0, +inf) to similarity [0, 1]
        # Eigenvalues in normalized laplacians are roughly in [0, 2]
        # so max distance is around 2.
        similarity = 1.0 / (1.0 + dist)
        return float(np.clip(similarity, 0.0, 1.0))

class JensenShannonSimilarity(BaseSimilarity):
    """
    Computes Jensen-Shannon divergence similarity using Histograms.
    Solves the size-variance problem by mapping both graphs to a fixed-size 50-bin histogram.
    """
    def __init__(self, bins=50, range_min=0.0, range_max=2.0):
        self.bins = bins
        self.range = (range_min, range_max)

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        if len(ev1) == 0 or len(ev2) == 0:
            return 0.0
            
        # Create normalized histograms (fixed size representations)
        h1, _ = np.histogram(ev1, bins=self.bins, range=self.range)
        h2, _ = np.histogram(ev2, bins=self.bins, range=self.range)
        
        # Add epsilon to prevent divide/log by zero
        h1_smooth = h1 + 1e-10
        h2_smooth = h2 + 1e-10
        
        # Convert to probability distributions
        p1 = h1_smooth / np.sum(h1_smooth)
        p2 = h2_smooth / np.sum(h2_smooth)
        
        # Scipy jensenshannon returns distance bounded in [0, 1]
        dist = jensenshannon(p1, p2)
        
        # Similarity = 1 - Distance
        return float(1.0 - dist)