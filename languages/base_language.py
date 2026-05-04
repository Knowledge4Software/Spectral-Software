from core.interfaces.language import Language
from core.abstractions.code_unit import CodeUnit


class BaseLanguage(Language):
    """
    Optional shared logic for all languages.
    """
    def preprocess(self, code_unit: CodeUnit) -> CodeUnit:
        # default: no-op
        return code_unit