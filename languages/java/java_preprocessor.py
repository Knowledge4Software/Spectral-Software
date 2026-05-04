from core.interfaces.preprocessor import Preprocessor
from core.abstractions.code_unit import CodeUnit


class JavaPreprocessor(Preprocessor):
    def transform(self, code_unit: CodeUnit) -> CodeUnit:
        return code_unit