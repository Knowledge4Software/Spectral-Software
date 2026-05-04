from core.interfaces.spectrum import SpectrumComputer
from core.abstractions.spectrum import Spectrum


class RandomizedSVD(SpectrumComputer):
    def compute(self, graph):
        return Spectrum([])