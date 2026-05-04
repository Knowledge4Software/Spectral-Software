from languages.base_language import BaseLanguage
from .java_parser import JavaParser
from .java_preprocessor import JavaPreprocessor
from core.abstractions.code_unit import CodeUnit


class JavaLanguage(BaseLanguage):
    def __init__(self):
        self.parser = JavaParser()
        self.preprocessor = JavaPreprocessor()

    def parse(self, raw_code: str) -> CodeUnit:
        return self.parser.parse(raw_code)

    def preprocess(self, code_unit: CodeUnit) -> CodeUnit:
        return self.preprocessor.transform(code_unit)