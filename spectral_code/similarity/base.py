from abc import ABC, abstractmethod
import numpy as np

class BaseSimilarity(ABC):
    """
    Abstract Base Class for all graph spectral similarity metrics.
    Ensures a unified interface across different mathematical formulations.
    """
    
    @abstractmethod
    def compute(self, ev1: np.ndarray, ev2: np.ndarray) -> float:
        """
        Computes the similarity score between two eigenvalue arrays.
        Returns a float typically bounded between 0.0 and 1.0.
        """
        pass