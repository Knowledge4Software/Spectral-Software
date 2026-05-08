import re
from spectral_code.preprocessing.base import Preprocessor


class SimplePreprocessor(Preprocessor):
    def process(self, code: str) -> str:
        # remove block comments, line comments and python comments
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
        code = re.sub(r"//.*?$", "", code, flags=re.M)
        code = re.sub(r"#.*?$", "", code, flags=re.M)

        cleaned_lines = []
        for line in code.splitlines():
            line = line.rstrip()
            if line.strip():
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
