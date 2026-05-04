from abc import ABC, abstractmethod
from core.abstractions.code_unit import CodeUnit

class Language(ABC):
    @abstractmethod
    def parse(self, raw_code: str) -> CodeUnit:
        pass

    @abstractmethod
    def preprocess(self, code_unit: CodeUnit) -> CodeUnit:
        pass