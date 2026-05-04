from core.interfaces.dataset import Dataset
from core.abstractions.code_unit import CodeUnit


class BenchmarkLoader(Dataset):
    def __init__(self, samples: list[str], language: str):
        self.samples = samples
        self.language = language

    def load(self) -> list[CodeUnit]:
        return [CodeUnit(source=s, language=self.language) for s in self.samples]