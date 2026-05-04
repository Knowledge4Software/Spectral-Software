from core.interfaces.preprocessor import Preprocessor
from core.abstractions.code_unit import CodeUnit


class PreprocessingPipeline:
    def __init__(self, steps: list[Preprocessor]):
        self.steps = steps

    def run(self, code_unit: CodeUnit) -> CodeUnit:
        for step in self.steps:
            code_unit = step.transform(code_unit)
        return code_unit