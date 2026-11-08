"""Portable clean-data loading for canonical mono/cross-language experiments.

The module deliberately separates parser-specific raw node strings from the
model-facing representation.  Every node receives a canonical type id and a
small, fixed-size hashed lexical sketch.  Raw strings never become vocabulary
indices, which prevents a Java-only vocabulary from defining the Python/C
feature space during zero-shot evaluation.
"""

from __future__ import annotations

from .canonical_types import NUM_CANONICAL_TYPES, get_canonical_id

import csv
import gzip
import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd




RELATIONS = ("ast", "next_token", "ddg")
LEXICAL_PAD_ID = 0
LEXICAL_UNKNOWN_ID = 1
DEFAULT_LEXICAL_BUCKETS = 4096
DEFAULT_LEXICAL_SLOTS = 4
INPUT_REPRESENTATION_VERSION = "java_callee_lexical_v2"


@dataclass(frozen=True)
class PairLanguageFilter:
    """Language constraints for both endpoints of a pair."""

    left: str | None = None
    right: str | None = None

    @classmethod
    def mono(cls, language: str) -> "PairLanguageFilter":
        normalized = normalize_language(language)
        return cls(normalized, normalized)

    @classmethod
    def cross(cls, left: str, right: str) -> "PairLanguageFilter":
        return cls(normalize_language(left), normalize_language(right))


@dataclass(frozen=True)
class CanonicalGraph:
    code_id: str
    language: str
    canonical_ids: np.ndarray
    lexical_ids: np.ndarray
    source_lexical_ids: np.ndarray
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    eigenvalues: dict[str, np.ndarray]

    @property
    def node_count(self) -> int:
        return int(len(self.canonical_ids))


def normalize_language(language: str | None) -> str:
    value = (language or "unknown").strip().lower()
    aliases = {
        "javasrc": "java",
        "py": "python",
        "pythonsrc": "python",
        "csrc": "c",
        "c++": "cpp",
        "cxx": "cpp",
    }
    return aliases.get(value, value)


def _is_gzip_file(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(2) == b"\x1f\x8b"


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if _is_gzip_file(path) else path.open("r", encoding="utf-8")


def _find_file(root: Path, *names: str) -> Path:
    """Find normal gzip/JSON files and Kaggle's occasional ``.tmp`` form.

    Kaggle can expose a ZIP member such as ``codes.jsonl.gz`` as
    ``codes.jsonl/codes.jsonl.gz.tmp``.  The content is still valid gzip, so
    accepting the suffix is safe; ``_open_text`` identifies compression from
    the file magic rather than the filename.
    """
    matches = []
    for name in names:
        variants = (name, f"{name}.tmp")
        for variant in variants:
            direct = root / variant
            if direct.is_file():
                matches.append(direct)
            matches.extend(path for path in root.rglob(variant) if path.is_file())
    if not matches:
        visible = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()][:30] if root.exists() else []
        raise FileNotFoundError(f"Missing {names} below {root}. Files seen: {visible}")
    return min(set(matches), key=lambda path: (len(path.parts), str(path)))


def verify_alignment(root: Path, specification: dict) -> Path:
    """Accept Kaggle gzip/plain/tmp layouts while verifying the same payload."""
    name = specification["file"]
    try:
        path = _find_file(root, name, name.removesuffix(".gz"))
    except FileNotFoundError as exc:
        raise ValueError("Required DDG/AST alignment sidecar is missing") from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() == specification["sha256"]:
        return path
    if specification.get("content_sha256"):
        digest = hashlib.sha256()
        opener = gzip.open if _is_gzip_file(path) else open
        with opener(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() == specification["content_sha256"]:
            return path
    raise ValueError("Required DDG/AST alignment sidecar has changed")


def _stable_bucket(value: str, bucket_count: int) -> int:
    if bucket_count <= 2:
        raise ValueError("lexical bucket count must be greater than two")
    digest = hashlib.blake2b(value.encode("utf-8", errors="replace"), digest_size=8, person=b"spectral").digest()
    return 2 + int.from_bytes(digest, "little") % (bucket_count - 2)


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def lexical_feature_ids(
    raw_label: str,
    *,
    bucket_count: int = DEFAULT_LEXICAL_BUCKETS,
    slots: int = DEFAULT_LEXICAL_SLOTS,
) -> np.ndarray:
    """Return a bounded hashed subtoken/character sketch for a node label.

    Pure line-number labels are intentionally ignored. Numeric, string,
    boolean, and null values are normalized before hashing. The output space
    is fixed across languages and never fitted on the training language.
    """
    text = html.unescape(str(raw_label or "")).strip()
    result = np.full(slots, LEXICAL_PAD_ID, dtype=np.int64)
    if not text or text.isdigit():
        return result
    text = _CAMEL_BOUNDARY.sub(" ", text)
    raw_tokens = [token.lower() for token in _TOKEN_SPLIT.split(text) if token]
    normalized = []
    for token in raw_tokens:
        if token.isdigit():
            token = "<num>"
        elif token in {"true", "false"}:
            token = "<bool>"
        elif token in {"null", "none", "nullptr", "nil"}:
            token = "<null>"
        normalized.append(token)
    # Keep the first few subtokens before character n-grams.  In particular,
    # ``Name,total`` must retain ``total`` even when ``Name`` has many
    # trigrams.  This is a bounded *fallback*, not a train-fitted vocabulary.
    features: list[str] = [f"sub:{token}" for token in normalized]
    for token in normalized[:2]:
        compact = f"^{token}$"
        features.extend(f"tri:{compact[index:index + 3]}" for index in range(max(0, len(compact) - 2)))
    for index, feature in enumerate(features[:slots]):
        result[index] = _stable_bucket(feature, bucket_count)
    return result


_SOURCE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|&&|\|\||[+*/%<>=!&|~-]")
_SOURCE_KEYWORDS = {
    "def": "func", "function": "func", "void": "func", "class": "class",
    "if": "if", "else": "else", "elif": "else", "switch": "switch", "case": "switch",
    "for": "loop", "while": "loop", "do": "loop", "foreach": "loop",
    "return": "return", "break": "break", "continue": "break",
    "try": "exception", "catch": "exception", "except": "exception", "throw": "exception",
    "import": "import", "from": "import", "package": "import",
    "true": "bool", "false": "bool", "none": "null", "null": "null",
}


def source_lexical_ids(
    source_code: str,
    *,
    bucket_count: int = DEFAULT_LEXICAL_BUCKETS,
    slots: int = 96,
) -> np.ndarray:
    """Return a bounded, language-neutral hashed sketch of source tokens.

    Keywords are normalized to functional concepts and identifiers are split
    into subtokens before hashing.  No train-fitted vocabulary and no raw AST
    node labels are used.  Uniform sampling retains information from the whole
    method instead of favouring Java boilerplate at the beginning.
    """
    output = np.full(slots, LEXICAL_PAD_ID, dtype=np.int64)
    text = html.unescape(str(source_code or ""))
    features = []
    for token in _SOURCE_TOKEN.findall(text):
        normalized = token.lower()
        if normalized.isdigit():
            features.append("src:num")
        elif normalized in _SOURCE_KEYWORDS:
            features.append(f"src:kw:{_SOURCE_KEYWORDS[normalized]}")
        elif normalized[0].isalpha() or normalized[0] == "_":
            split = [part.lower() for part in _CAMEL_BOUNDARY.sub(" ", token).replace("_", " ").split() if part]
            features.extend(f"src:id:{part}" for part in split[:3])
        else:
            features.append(f"src:op:{normalized}")
    if not features:
        return output
    if len(features) <= slots:
        chosen = features
    else:
        chosen = [features[round(index * (len(features) - 1) / (slots - 1))] for index in range(slots)]
    for index, feature in enumerate(chosen[:slots]):
        output[index] = _stable_bucket(feature, bucket_count)
    return output


def lexical_source(raw_type: str, raw_label: str, canonical_id: int, *, language: str | None = None) -> str:
    """Select a source-level lexical fallback without using numeric line IDs.

    Python fallback graphs serialize labels such as ``Name,total`` while many
    Joern graphs serialize only a line number.  In the former case the parser
    type prefix is removed; in the latter, unknown raw node text and recovered
    Java callee names supply the fallback. Known structural parser tags are deliberately
    not hashed, because their canonical category already represents them.
    """
    raw_type_text = html.unescape(str(raw_type or "")).strip()
    label = html.unescape(str(raw_label or "")).strip()
    if label:
        prefix = raw_type_text + ","
        if prefix and label.lower().startswith(prefix.lower()):
            label = label[len(prefix):].strip()
        if label and not label.isdigit() and label.lower() != raw_type_text.lower():
            return label
    # Java exports can put the callee (e.g. ``println``) in node_types while
    # node_labels contains only a line number. Canonical mapping deliberately
    # recognizes that as Call_Expr; retain the callee in the separate hashed
    # lexical channel as well. Exact/parser-pattern structural tags remain
    # excluded, so CALL/IDENTIFIER/etc. do not become lexical features.
    unknown = get_canonical_id("Canonical_Unknown")
    java_callee = (str(language or "").strip().lower() == "java"
                   and canonical_id == get_canonical_id("CALL")
                   and get_canonical_id(raw_type_text) == unknown)
    if (canonical_id == unknown or java_callee) and raw_type_text and not raw_type_text.isdigit():
        return raw_type_text
    return ""


def _edge_array(rows: Sequence, cols: Sequence, node_count: int) -> np.ndarray:
    pairs = [
        (int(source), int(target))
        for source, target in zip(rows, cols)
        if 0 <= int(source) < node_count and 0 <= int(target) < node_count
    ]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2) if pairs else np.empty((0, 2), dtype=np.int32)


def _project_ddg(ast: dict, ddg: dict, node_count: int) -> np.ndarray:
    ast_ids = [str(value) for value in ast.get("node_ids", [])[:node_count]]
    ast_index = {node_id: index for index, node_id in enumerate(ast_ids)}
    ddg_ids = [str(value) for value in ddg.get("ast_node_ids", ddg.get("node_ids", []))]
    pairs = []
    for raw_source, raw_target in zip(ddg.get("row", []), ddg.get("col", [])):
        source_position, target_position = int(raw_source), int(raw_target)
        if not (0 <= source_position < len(ddg_ids) and 0 <= target_position < len(ddg_ids)):
            continue
        source = ast_index.get(ddg_ids[source_position])
        target = ast_index.get(ddg_ids[target_position])
        if source is not None and target is not None:
            pairs.extend(((source, target), (target, source)))
    return np.asarray(sorted(set(pairs)), dtype=np.int32).reshape(-1, 2) if pairs else np.empty((0, 2), dtype=np.int32)


def canonical_graph_from_record(
    record: dict,
    *,
    language: str,
    max_nodes: int = 512,
    lexical_buckets: int = DEFAULT_LEXICAL_BUCKETS,
    lexical_slots: int = DEFAULT_LEXICAL_SLOTS,
    source_code: str = "",
    source_slots: int = 96,
    strip_node_types: bool = False,
    strip_topology: bool = False,
) -> CanonicalGraph | None:
    graphs = record.get("graphs", {})
    ast_layer = graphs.get("ast", {})
    ddg_layer = graphs.get("ddg", {})
    ast = ast_layer.get("adjacency", ast_layer) if isinstance(ast_layer, dict) else {}
    ddg = ddg_layer.get("adjacency", ddg_layer) if isinstance(ddg_layer, dict) else {}
    raw_types = list(ast.get("node_types", []))
    if not raw_types:
        return None
    node_count = min(len(raw_types), max_nodes)
    raw_labels = list(ast.get("node_labels", []))
    if len(raw_labels) < node_count:
        raw_labels.extend([""] * (node_count - len(raw_labels)))
    # Zero is reserved for padding in the embedding table. Canonical IDs are
    # serialized as 1..N while the mapping API itself remains the requested
    # stable zero-based 0..N-1 vocabulary.
    raw_type_ids = [get_canonical_id(str(raw), language=language) for raw in raw_types[:node_count]]
    canonical_ids = np.asarray([raw_id + 1 for raw_id in raw_type_ids], dtype=np.int64)
    if strip_node_types:
        # One shared id for every node: adjacency survives, categories do not.
        canonical_ids = np.ones_like(canonical_ids)
    lexical_ids = np.stack(
        [
            lexical_feature_ids(
                lexical_source(raw_types[index], raw_labels[index], raw_type_ids[index], language=language),
                bucket_count=lexical_buckets,
                slots=lexical_slots,
            )
            for index in range(node_count)
        ]
    )
    ast_edges = _edge_array(ast.get("row", []), ast.get("col", []), node_count)
    ast_edges = np.asarray(sorted(set(map(tuple, np.vstack((ast_edges, ast_edges[:, ::-1]))))), dtype=np.int32) if len(ast_edges) else ast_edges
    next_edges = np.asarray(
        [(source, target) for index in range(node_count - 1) for source, target in ((index, index + 1), (index + 1, index))],
        dtype=np.int32,
    ).reshape(-1, 2)
    ddg_edges = _project_ddg(ast, ddg, node_count)
    if strip_topology:
        # Input-only ablation: the encoder receives no observed graph edges.
        empty_edges = np.empty((0, 2), dtype=np.int32)
        ast_edges, next_edges, ddg_edges = empty_edges, empty_edges.copy(), empty_edges.copy()
    eigenvalues = {
        graph_type: np.asarray(layer.get("eigenvalues", []), dtype=np.float32)
        for graph_type, layer in graphs.items()
        if isinstance(layer, dict)
    }
    return CanonicalGraph(
        code_id=str(record.get("code_id", record.get("id", ""))),
        language=normalize_language(language),
        canonical_ids=canonical_ids,
        lexical_ids=lexical_ids,
        source_lexical_ids=source_lexical_ids(source_code, bucket_count=lexical_buckets, slots=source_slots),
        edges=(ast_edges, next_edges, ddg_edges),
        eigenvalues=eigenvalues,
    )


class CleanDataCorpus:
    """Reader for ``spectral_clean_data_v1`` exports with language filters."""

    def __init__(self, root: str | Path, *, default_language: str | None = None) -> None:
        self.root = Path(root)
        self.codes_path = _find_file(self.root, "codes.jsonl.gz", "codes.jsonl")
        self.pairs_path = _find_file(self.root, "pairs.csv.gz", "pairs.csv")
        self.graphs_path = _find_file(self.root, "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp")
        self.default_language = normalize_language(default_language) if default_language else None
        self.code_metadata = {}
        self.ddg_alignment = {}
        try:
            alignment_path = _find_file(self.root, "ddg_ast_alignment.jsonl.gz", "ddg_ast_alignment.jsonl")
        except FileNotFoundError:
            alignment_path = None
        if alignment_path is not None:
            with _open_text(alignment_path) as stream:
                for line in stream:
                    item = json.loads(line)
                    self.ddg_alignment[str(item["code_id"])] = item["ast_node_ids"]
        self.code_languages, self.code_texts = self._load_code_metadata()

    def _load_code_metadata(self) -> tuple[dict[str, str], dict[str, str]]:
        languages, code_texts = {}, {}
        with _open_text(self.codes_path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                code_id = str(record.get("code_id", record.get("id", record.get("idx", ""))))
                raw_language = record.get("language", record.get("lang", self.default_language))
                if raw_language is None:
                    raise ValueError(
                        f"Code {code_id} in {self.codes_path} has no language; provide default_language explicitly."
                    )
                languages[code_id] = normalize_language(str(raw_language))
                code_texts[code_id] = str(record.get("code", record.get("source", "")) or "")
                self.code_metadata[code_id] = {key: record.get(key) for key in
                                              ("problem_id", "source_code_id", "source_sha256")}
        return languages, code_texts

    def pairs(
        self,
        split: str,
        *,
        language_filter: PairLanguageFilter | None = None,
        maximum: int | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        frame = pd.read_csv(
            self.pairs_path,
            compression="gzip" if _is_gzip_file(self.pairs_path) else None,
            dtype={"left_id": str, "right_id": str, "label": np.int8},
        )
        frame = frame[frame["split"].astype(str).str.lower() == split.strip().lower()].copy()
        frame["left_language"] = frame["left_id"].map(self.code_languages)
        frame["right_language"] = frame["right_id"].map(self.code_languages)
        for side in ("left", "right"):
            frame[f"{side}_problem_id"] = frame[f"{side}_id"].map(
                lambda value: self.code_metadata.get(str(value), {}).get("problem_id"))
        if frame[["left_language", "right_language"]].isna().any().any():
            raise RuntimeError("At least one pair endpoint is missing from codes.jsonl(.gz).")
        if language_filter:
            if language_filter.left:
                frame = frame[frame["left_language"] == normalize_language(language_filter.left)]
            if language_filter.right:
                frame = frame[frame["right_language"] == normalize_language(language_filter.right)]
        if maximum is not None and len(frame) > maximum:
            pieces = []
            for _, group in frame.groupby("label", sort=True):
                size = max(1, round(maximum * len(group) / len(frame)))
                pieces.append(group.sample(n=min(size, len(group)), random_state=seed))
            frame = pd.concat(pieces).sample(frac=1, random_state=seed).head(maximum)
        return frame.reset_index(drop=True)

    def load_graphs(
        self,
        code_ids: Iterable[str],
        *,
        max_nodes: int = 512,
        lexical_buckets: int = DEFAULT_LEXICAL_BUCKETS,
        lexical_slots: int = DEFAULT_LEXICAL_SLOTS,
        source_slots: int = 96,
        strip_node_types: bool = False,
        strip_topology: bool = False,
    ) -> dict[str, CanonicalGraph]:
        required = {str(value) for value in code_ids}
        graphs = {}
        with _open_text(self.graphs_path) as stream:
            for line in stream:
                record = json.loads(line)
                code_id = str(record.get("code_id", record.get("id", "")))
                if code_id not in required:
                    continue
                if code_id in self.ddg_alignment:
                    record["graphs"]["ddg"]["adjacency"]["ast_node_ids"] = self.ddg_alignment[code_id]
                graph = canonical_graph_from_record(
                    record,
                    language=self.code_languages[code_id],
                    max_nodes=max_nodes,
                    lexical_buckets=lexical_buckets,
                    lexical_slots=lexical_slots,
                    source_code=self.code_texts.get(code_id, ""),
                    source_slots=source_slots,
                    strip_node_types=strip_node_types,
                    strip_topology=strip_topology,
                )
                if graph is not None:
                    graphs[code_id] = graph
        return graphs


def pair_code_ids(*frames: pd.DataFrame) -> set[str]:
    values: set[str] = set()
    for frame in frames:
        values.update(frame["left_id"].astype(str))
        values.update(frame["right_id"].astype(str))
    return values
