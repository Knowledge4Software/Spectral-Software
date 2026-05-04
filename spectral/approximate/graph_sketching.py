from core.interfaces.spectrum import SpectrumComputer
from core.abstractions.spectrum import Spectrum


class GraphSketching(SpectrumComputer):
    def compute(self, graph):
        return Spectrum([])