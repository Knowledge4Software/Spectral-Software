from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ClonePair:
    left_id: int
    right_id: int
    label: int
    clone_type: str | None
    left_code: str
    right_code: str


def _strip_comments_and_whitespace(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"//.*?$", " ", code, flags=re.M)
    code = re.sub(r"#.*?$", " ", code, flags=re.M)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


def _normalize_identifiers_and_literals(code: str) -> str:
    """
    Weak lexical normalizer used only as a fallback when no official type labels exist.
    """
    code = _strip_comments_and_whitespace(code)

    code = re.sub(r'"(?:\\.|[^"\\])*"', " <STR> ", code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", " <CHR> ", code)
    code = re.sub(r"\b\d+(?:\.\d+)?\b", " <NUM> ", code)

    keywords = {
        "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
        "return", "new", "class", "public", "private", "protected", "static", "final",
        "void", "int", "long", "double", "float", "boolean", "char", "byte", "short",
        "true", "false", "null", "try", "catch", "finally", "throw", "throws", "extends",
        "implements", "import", "package", "this", "super", "var", "def", "lambda",
        "and", "or", "not", "in", "is", "with", "as", "from", "pass", "yield", "async",
        "await", "elif"
    }

    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        return token if token in keywords else "<ID>"

    code = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", repl, code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()


class BigCloneBenchLoader:
    def __init__(self, data_dir: str | Path, type_labels_path: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.code_map = self._load_code_map(self.data_dir / "data.jsonl")
        self.splits = {
            "train": self._load_split(self.data_dir / "train.txt"),
            "valid": self._load_split(self.data_dir / "valid.txt"),
            "test": self._load_split(self.data_dir / "test.txt"),
        }
        self.type_map = self._load_type_map(type_labels_path) if type_labels_path else {}

    @staticmethod
    def _load_code_map(path: Path) -> dict[int, str]:
        code_map: dict[int, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                code_map[int(obj["idx"])] = obj["func"]
        return code_map

    @staticmethod
    def _load_split(path: Path) -> list[tuple[int, int, int]]:
        rows: list[tuple[int, int, int]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                left, right, label = line.strip().split("\t")
                rows.append((int(left), int(right), int(label)))
        return rows

    @staticmethod
    def _canonical_type_name(raw: str) -> str:
        raw = raw.strip().lower().replace("-", "_").replace(" ", "")
        mapping = {
            "1": "type_1", "type1": "type_1", "t1": "type_1", "type_1": "type_1",
            "2": "type_2", "type2": "type_2", "t2": "type_2", "type_2": "type_2",
            "3": "type_3", "type3": "type_3", "t3": "type_3", "type_3": "type_3",
            "4": "type_4", "type4": "type_4", "t4": "type_4", "type_4": "type_4",
        }
        if raw not in mapping:
            raise ValueError(f"Unknown clone type label: {raw}")
        return mapping[raw]

    def _load_type_map(self, path_like: str | Path) -> dict[tuple[int, int], str]:
        """
        Optional file format:
          TSV:  left_id <tab> right_id <tab> type
        or JSONL:
          {"left": 1, "right": 2, "type": "type_1"}
        """
        path = Path(path_like)
        if not path.exists():
            return {}

        mapping: dict[tuple[int, int], str] = {}

        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    left = int(obj["left"])
                    right = int(obj["right"])
                    ctype = self._canonical_type_name(str(obj["type"]))
                    mapping[tuple(sorted((left, right)))] = ctype
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    left, right, ctype = line.strip().split("\t")
                    mapping[tuple(sorted((int(left), int(right))))] = self._canonical_type_name(ctype)

        return mapping

    def get_pairs(self, split: str) -> list[ClonePair]:
        pairs: list[ClonePair] = []
        for left, right, label in self.splits[split]:
            key = tuple(sorted((left, right)))
            pairs.append(
                ClonePair(
                    left_id=left,
                    right_id=right,
                    label=label,
                    clone_type=self.type_map.get(key),
                    left_code=self.code_map[left],
                    right_code=self.code_map[right],
                )
            )
        return pairs

    def infer_type(self, pair: ClonePair) -> str:
        """
        Fallback heuristic only. For paper-grade reporting, prefer official type labels.
        """
        left_raw = _strip_comments_and_whitespace(pair.left_code)
        right_raw = _strip_comments_and_whitespace(pair.right_code)
        if left_raw == right_raw:
            return "type_1"

        left_norm = _normalize_identifiers_and_literals(pair.left_code)
        right_norm = _normalize_identifiers_and_literals(pair.right_code)
        if left_norm == right_norm:
            return "type_2"

        l_tokens = set(left_norm.split())
        r_tokens = set(right_norm.split())
        jaccard = len(l_tokens & r_tokens) / max(1, len(l_tokens | r_tokens))

        return "type_3" if jaccard >= 0.45 else "type_4"

    def stratified_sample(self, split: str, per_type: int, seed: int = 42) -> dict[str, list[ClonePair]]:
        rng = random.Random(seed)
        buckets: dict[str, list[ClonePair]] = {
            "type_1": [],
            "type_2": [],
            "type_3": [],
            "type_4": [],
        }

        for pair in self.get_pairs(split):
            if pair.label != 1:
                continue

            ctype = pair.clone_type or self.infer_type(pair)
            buckets[ctype].append(pair)

        sampled: dict[str, list[ClonePair]] = {}
        for ctype, items in buckets.items():
            if not items:
                sampled[ctype] = []
                continue
            n = min(per_type, len(items))
            sampled[ctype] = rng.sample(items, n)

        return sampled
    
    def sample_pairs(self, split: str, n: int, seed: int = 42) -> list[ClonePair]:
        rng = random.Random(seed)
        pairs = self.get_pairs(split)

        if n >= len(pairs):
            return pairs

        return rng.sample(pairs, n)
