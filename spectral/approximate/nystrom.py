from core.interfaces.spectrum import SpectrumComputer
from core.abstractions.spectrum import Spectrum


class Nystorm(SpectrumComputer):
    def compute(self, graph):
        return Spectrum([])