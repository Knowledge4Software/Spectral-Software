from core.interfaces.dataset import Dataset
from core.abstractions.code_unit import CodeUnit
import os


class FileSystemLoader(Dataset):
    def __init__(self, root: str, extension: str, language: str):
        self.root = root
        self.extension = extension
        self.language = language

    def load(self) -> list[CodeUnit]:
        data = []

        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                if f.endswith(self.extension):
                    path = os.path.join(dirpath, f)
                    with open(path, "r", encoding="utf-8") as file:
                        data.append(
                            CodeUnit(source=file.read(), language=self.language)
                        )

        return data