import os
import numpy as np
from .base import BaseSimilarity

class HeatKernelSimilarity(BaseSimilarity):
    """
    Multi-scale Heat Kernel Trace similarity in log space.

    Heat traces are exponential by nature, so comparing raw traces compresses
    scores near 1. We compare log heat signatures without L2 normalization and
    map their distance with S = exp(-gamma * d).
    """
    def __init__(self, time_scales=None, gamma: float | None = None, epsilon: float = 1e-12):
        # Default multi-scale times to capture micro and macro structural topologies
        self.time_scales = time_scales if time_scales is not None else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.gamma = float(gamma if gamma is not None else os.getenv("HEAT_KERNEL_GAMMA", "1.0"))
        self.epsilon = float(epsilon)
        if self.gamma <= 0:
            raise ValueError("gamma must be positive.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")

    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        if len(ev1) == 0 or len(ev2) == 0:
            return 0.0
            
        # Heat Kernel Trace: Z(t) = mean(exp(-lambda * t)).
        trace1 = np.array([np.mean(np.exp(-ev1 * t)) for t in self.time_scales], dtype=np.float64)
        trace2 = np.array([np.mean(np.exp(-ev2 * t)) for t in self.time_scales], dtype=np.float64)

        log_trace1 = np.log(np.maximum(trace1, self.epsilon))
        log_trace2 = np.log(np.maximum(trace2, self.epsilon))
        distance = np.linalg.norm(log_trace1 - log_trace2)
        similarity = np.exp(-self.gamma * distance)

        return float(np.clip(similarity, 0.0, 1.0))
