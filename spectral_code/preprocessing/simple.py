import re
from .base import Preprocessor

class SimplePreprocessor(Preprocessor):
    def process(self, code: str) -> str:
        # remove comments (very naive)
        code = re.sub(r"#.*", "", code)
        return code.strip()