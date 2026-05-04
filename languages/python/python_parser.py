from core.abstractions.code_unit import CodeUnit
import ast


class PythonParser:
    def parse(self, source: str) -> CodeUnit:
        tree = ast.parse(source)
        return CodeUnit(source=source, language="python", metadata={"ast": tree})