import re
from spectral_code.preprocessing.base import Preprocessor


class SimplePreprocessor(Preprocessor):
    def process(self, code: str) -> str:
        # Preprocessing regex can incorrectly ruin URLs (// in strings).
        # We'll skip comment stripping because Joern's Java parser inherently ignores comments
        # and doesn't get messed up.
        
        cleaned_lines = []
        for line in code.splitlines():
            line = line.rstrip()
            if line.strip():
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
