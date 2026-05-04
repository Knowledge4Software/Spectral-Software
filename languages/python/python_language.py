from languages.base_language import BaseLanguage
from .python_parser import PythonParser
from .python_preprocessor import PythonPreprocessor
from core.abstractions.code_unit import CodeUnit


class PythonLanguage(BaseLanguage):
    def __init__(self):
        self.parser = PythonParser()
        self.preprocessor = PythonPreprocessor()

    def parse(self, raw_code: str) -> CodeUnit:
        return self.parser.parse(raw_code)

    def preprocess(self, code_unit: CodeUnit) -> CodeUnit:
        return self.preprocessor.transform(code_unit)