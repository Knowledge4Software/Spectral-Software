from core.interfaces.dataset import Dataset
from core.interfaces.graph_builder import GraphBuilder
from core.interfaces.spectrum import SpectrumComputer


class CloneDetectionPipeline:
    def __init__(
        self,
        dataset: Dataset,
        graph_builder: GraphBuilder,
        spectrum_computer: SpectrumComputer,
    ):
        self.dataset = dataset
        self.graph_builder = graph_builder
        self.spectrum_computer = spectrum_computer

    def run(self):
        results = []

        for code_unit in self.dataset.load():
            graph = self.graph_builder.build(code_unit)
            spectrum = self.spectrum_computer.compute(graph)
            results.append((code_unit, spectrum))

        return results