from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import pgdumplib
except ModuleNotFoundError:  # pragma: no cover - dependency guard
    pgdumplib = None

from spectral_code.evaluation.bcb_dataset import ClonePair
from spectral_code.utils.dataset_paths import semantic_dump_path


LANGUAGE_ALIASES = {
    "c": "c",
    "cs": "csharp",
    "c#": "csharp",
    "csharp": "csharp",
    "java": "java",
    "python": "python",
}


@dataclass(frozen=True)
class SemanticCodeSnippet:
    snippet_id: int
    dataset_name: str
    language: str
    pair_number: int
    fragment_number: int
    source_question_id: int | None
    source_answer_id: int | None
    source_file: str
    start_line: int
    end_line: int
    line_count: int
    code: str


@dataclass(frozen=True)
class SemanticCloneRow:
    clone_id: int
    dataset_name: str
    language: str
    pair_number: int
    is_clone: int
    clone_kind: str
    source_question_id: int | None
    source_file: str
    code1_id: int
    code2_id: int


class SemanticBenchmarkLoader:
    def __init__(self, language: str, seed: int = 42, dump_path: str | Path | None = None):
        self.requested_language = language
        self.language = self._normalize_language(language)
        self.seed = seed
        self.dump_path = Path(dump_path) if dump_path is not None else semantic_dump_path()
        if pgdumplib is None:
            raise ModuleNotFoundError(
                "Missing dependency 'pgdumplib'. Install it in the active environment with "
                "`python -m pip install -r requirements.txt` or "
                "`.venv\\Scripts\\python -m pip install pgdumplib`."
            )
        if not self.dump_path.exists():
            raise FileNotFoundError(f"Semantic clone benchmark dump not found: {self.dump_path}")

    @staticmethod
    def _normalize_language(language: str) -> str:
        key = language.strip().lower()
        if key not in LANGUAGE_ALIASES:
            raise ValueError(f"Unsupported semantic benchmark language: {language}")
        return LANGUAGE_ALIASES[key]

    @staticmethod
    def _optional_int(raw: str | None) -> int | None:
        if raw is None or raw == r"\N" or raw == "":
            return None
        return int(raw)

    @classmethod
    @lru_cache(maxsize=4)
    def _load_archive(cls, dump_path_str: str):
        return pgdumplib.load(dump_path_str)

    def _archive(self):
        return self._load_archive(str(self.dump_path))

    def _code_snippets_for_language(self) -> dict[int, SemanticCodeSnippet]:
        snippets: dict[int, SemanticCodeSnippet] = {}
        for row in self._archive().table_data("semantic_clone", "code_snippet"):
            if row[2] != self.language:
                continue
            snippet = SemanticCodeSnippet(
                snippet_id=int(row[0]),
                dataset_name=row[1],
                language=row[2],
                pair_number=int(row[3]),
                fragment_number=int(row[4]),
                source_question_id=self._optional_int(row[5]),
                source_answer_id=self._optional_int(row[6]),
                source_file=row[7],
                start_line=int(row[8]),
                end_line=int(row[9]),
                line_count=int(row[10]),
                code=row[11],
            )
            snippets[snippet.snippet_id] = snippet
        if not snippets:
            raise RuntimeError(f"No code snippets found in semantic dump for language: {self.requested_language}")
        return snippets

    def _clone_rows_for_language(self) -> list[SemanticCloneRow]:
        rows: list[SemanticCloneRow] = []
        for row in self._archive().table_data("semantic_clone", "clone_pair"):
            if row[2] != self.language:
                continue
            rows.append(
                SemanticCloneRow(
                    clone_id=int(row[0]),
                    dataset_name=row[1],
                    language=row[2],
                    pair_number=int(row[3]),
                    is_clone=int(row[4]),
                    clone_kind=row[5],
                    source_question_id=self._optional_int(row[6]),
                    source_file=row[7],
                    code1_id=int(row[8]),
                    code2_id=int(row[9]),
                )
            )
        if not rows:
            raise RuntimeError(f"No clone pairs found in semantic dump for language: {self.requested_language}")
        return rows

    def positive_pairs(self) -> list[ClonePair]:
        snippets = self._code_snippets_for_language()
        pairs: list[ClonePair] = []
        for row in self._clone_rows_for_language():
            if row.is_clone != 1:
                continue
            left = snippets.get(row.code1_id)
            right = snippets.get(row.code2_id)
            if left is None or right is None:
                continue
            pairs.append(
                ClonePair(
                    left_id=left.snippet_id,
                    right_id=right.snippet_id,
                    label=1,
                    clone_type=f"semantic_{self.language}",
                    left_code=left.code,
                    right_code=right.code,
                )
            )
        if not pairs:
            raise RuntimeError(f"No positive semantic pairs were extracted for language: {self.requested_language}")
        return pairs

    def standalone_snippets(self) -> list[tuple[int, str]]:
        snippets = self._code_snippets_for_language()
        return [(snippet.snippet_id, snippet.code) for snippet in sorted(snippets.values(), key=lambda item: item.snippet_id)]

    def negative_pairs(self, target_count: int | None = None) -> list[ClonePair]:
        raise RuntimeError(
            "Semantic Clone Benchmark does not provide non-clone pairs. "
            "Automatic negative sampling is disabled; add explicit label-0 pairs to the prepared data when needed."
        )

    def get_pairs(self, negative_ratio: float = 0.0) -> list[ClonePair]:
        positives = self.positive_pairs()
        if negative_ratio > 0:
            raise RuntimeError(
                "Automatic semantic non-clone generation is disabled. "
                "Use the clone-only Semantic Clone Benchmark data or add your own label-0 pairs later."
            )
        return positives
