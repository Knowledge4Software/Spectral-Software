from abc import ABC, abstractmethod
from core.abstractions.code_unit import CodeUnit
from core.abstractions.graph import Graph

class GraphBuilder(ABC):
    @abstractmethod
    def build(self, code_unit: CodeUnit) -> Graph:
        pass