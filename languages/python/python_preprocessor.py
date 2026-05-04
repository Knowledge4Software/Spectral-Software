from core.interfaces.preprocessor import Preprocessor
from core.abstractions.code_unit import CodeUnit


class PythonPreprocessor(Preprocessor):
    def transform(self, code_unit: CodeUnit) -> CodeUnit:
        # simple normalization placeholder
        code_unit.source = code_unit.source.strip()
        return code_unit