import numpy as np
from .base import BaseSimilarity

class HeatKernelSimilarity(BaseSimilarity):
    """
    Implements the multi-scale Heat Kernel Trace similarity metric.
    Maps variable-length spectrums to size-invariant diffusion signatures.
    """
    def __init__(self, time_scales=None):
        # Default multi-scale times to capture micro and macro structural topologies
        self.time_scales = time_scales if time_scales is not None else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        if len(ev1) == 0 or len(ev2) == 0:
            return 0.0
            
        # Compute Heat Kernel Traces: Z(t) = (1/N) * Sum(exp(-lambda * t))
        # Divided by N to make the trace perfectly scale-invariant (density of heat)
        trace1 = np.array([np.sum(np.exp(-ev1 * t)) for t in self.time_scales]) / len(ev1)
        trace2 = np.array([np.sum(np.exp(-ev2 * t)) for t in self.time_scales]) / len(ev2)
        
        # Normalized Euclidean Distance Similarity Mapping
        distance = np.linalg.norm(trace1 - trace2)
        norm_factor = np.linalg.norm(trace1) + np.linalg.norm(trace2)
        
        if norm_factor == 0:
            return 0.0
            
        return float(1.0 - (distance / norm_factor))