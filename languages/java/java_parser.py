from core.abstractions.code_unit import CodeUnit


class JavaParser:
    def parse(self, source: str) -> CodeUnit:
        # placeholder (no real parser yet)
        return CodeUnit(source=source, language="java")