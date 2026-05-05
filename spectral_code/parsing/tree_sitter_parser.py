from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs


class TreeSitterParser:
    def __init__(self):
        self.parsers = {
            "python": Parser(Language(tspython.language())),
            "java": Parser(Language(tsjava.language())),
            "javascript": Parser(Language(tsjs.language())),
        }
        
    def parse(self, code: str, lang: str):
        if lang not in self.parsers:
            raise ValueError(f"Unsupported language: {lang}")

        return self.parsers[lang].parse(bytes(code, "utf8"))