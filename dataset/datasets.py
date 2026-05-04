from core.interfaces.dataset import Dataset
from core.abstractions.code_unit import CodeUnit


class InMemoryDataset(Dataset):
    def __init__(self, code_units: list[CodeUnit]):
        self._data = code_units

    def load(self) -> list[CodeUnit]:
        return self._data