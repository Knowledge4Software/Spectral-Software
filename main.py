from dataset.datasets import InMemoryDataset
from graph.graph_factory import GraphFactory
from spectral.spectrum_factory import SpectrumFactory


def main():
    dataset = InMemoryDataset([])

    graph_factory = GraphFactory()
    spectrum_factory = SpectrumFactory()

    # placeholder flow
    for code_unit in dataset.load():
        pass


if __name__ == "__main__":
    main()