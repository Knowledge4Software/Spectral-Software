from abc import ABC, abstractmethod
from core.abstractions.code_unit import CodeUnit

class Dataset(ABC):
    @abstractmethod
    def load(self) -> list[CodeUnit]:
        pass