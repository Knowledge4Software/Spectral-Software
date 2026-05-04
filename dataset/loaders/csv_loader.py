from core.interfaces.dataset import Dataset
from core.abstractions.code_unit import CodeUnit
import csv


class CSVLoader(Dataset):
    def __init__(self, path: str, code_column: str, language: str):
        self.path = path
        self.code_column = code_column
        self.language = language

    def load(self) -> list[CodeUnit]:
        data = []

        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(
                    CodeUnit(source=row[self.code_column], language=self.language)
                )

        return data