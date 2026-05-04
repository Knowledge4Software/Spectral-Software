from abc import ABC, abstractmethod
from core.abstractions.code_unit import CodeUnit

class Preprocessor(ABC):
    @abstractmethod
    def transform(self, code_unit: CodeUnit) -> CodeUnit:
        pass