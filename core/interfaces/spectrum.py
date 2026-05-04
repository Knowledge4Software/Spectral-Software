from abc import ABC, abstractmethod
from core.abstractions.graph import Graph
from core.abstractions.spectrum import Spectrum

class SpectrumComputer(ABC):
    @abstractmethod
    def compute(self, graph: Graph) -> Spectrum:
        pass