#!/usr/bin/env python3
"""Calculate CodeNet clone/non-clone hardness by language configuration.

The script reads the CodeNet ZIP without extracting or modifying it and writes
one CSV. H_pos is one minus the mean positive-pair similarity; H_neg is the
mean negative-pair similarity. Similarity is exact set Jaccard over canonical
token categories and adjacent category bigrams.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ImportError:  # Progress bars are optional.
    tqdm = None


PAIR_KINDS = ("clone", "hard_nonclone", "nonclone_diff_problem")
NEGATIVE_KINDS = ("hard_nonclone", "nonclone_diff_problem")
SIMILARITY_NAME = "canonical_structural_token_unigram_bigram_set_jaccard_v1"

CONFIGURATIONS = {
    "java": ("Single-language", "Java - Java"),
    "python": ("Single-language", "Python - Python"),
    "cpp": ("Single-language", "C++ - C++"),
    "csharp": ("Single-language", "C# - C#"),
    "python_java": ("Cross-language", "Python - Java"),
    "python_cpp": ("Cross-language", "Python - C++"),
    "python_csharp": ("Cross-language", "Python - C#"),
    "java_cpp": ("Cross-language", "Java - C++"),
    "java_csharp": ("Cross-language", "Java - C#"),
    "cpp_csharp": ("Cross-language", "C++ - C#"),
}

KEYWORD_GROUPS = {
    "KW_IF": ("if", "elif"),
    "KW_ELSE": ("else",),
    "KW_BRANCH": ("switch", "case", "default", "match", "when"),
    "KW_LOOP": ("for", "foreach", "while", "do"),
    "KW_RETURN": ("return",),
    "KW_BREAK": ("break",),
    "KW_CONTINUE": ("continue",),
    "KW_GOTO": ("goto",),
    "KW_TRY": ("try",),
    "KW_CATCH": ("catch", "except"),
    "KW_FINALLY": ("finally",),
    "KW_THROW": ("throw", "throws", "raise"),
    "KW_ASSERT": ("assert",),
    "KW_CLASS": ("class", "struct", "interface", "record"),
    "KW_ENUM": ("enum",),
    "KW_FUNCTION": ("def", "function"),
    "KW_LAMBDA": ("lambda",),
    "KW_NAMESPACE": ("namespace", "package"),
    "KW_IMPORT": ("import", "from", "using", "include"),
    "KW_NEW": ("new",),
    "KW_DELETE": ("delete",),
    "KW_INHERIT": ("extends", "implements"),
    "KW_ACCESS": ("public", "private", "protected", "internal"),
    "KW_MODIFIER": (
        "static", "final", "const", "readonly", "virtual", "override",
        "abstract", "volatile", "synchronized",
    ),
    "KW_ASYNC": ("async",),
    "KW_AWAIT": ("await",),
    "KW_YIELD": ("yield",),
    "KW_TYPE": (
        "void", "auto", "var", "let", "int", "integer", "long", "short",
        "byte", "sbyte", "uint", "ulong", "ushort", "float", "double",
        "decimal", "char", "string", "bool", "boolean", "object",
    ),
    "LIT_BOOL": ("true", "false"),
    "LIT_NULL": ("none", "null", "nullptr", "nil"),
    "OP_&&": ("and",),
    "OP_||": ("or",),
    "OP_!": ("not",),
    "OP_MEMBERSHIP": ("in",),
    "OP_IDENTITY": ("is",),
    "OP_CAST": ("as",),
    "OP_SIZEOF": ("sizeof",),
    "OP_TYPEOF": ("typeof", "instanceof"),
}
KEYWORD_CATEGORY = {
    keyword: category
    for category, keywords in KEYWORD_GROUPS.items()
    for keyword in keywords
}

OPERATORS = (
    ">>>=", "<<=", ">>=", "**=", "//=", "...", "??=", "=>", "->", "::",
    "++", "--", "&&", "||", "==", "!=", "<=", ">=", "+=", "-=", "*=",
    "/=", "%=", "&=", "|=", "^=", "<<", ">>", "**", "//", "??", "?.",
    "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~",
    "?", ".",
)
DELIMITERS = {
    "(": "DELIM_PAREN_OPEN",
    ")": "DELIM_PAREN_CLOSE",
    "[": "DELIM_BRACKET_OPEN",
    "]": "DELIM_BRACKET_CLOSE",
    "{": "DELIM_BLOCK_OPEN",
    "}": "DELIM_BLOCK_CLOSE",
    ":": "DELIM_COLON",
    ";": "DELIM_STATEMENT_END",
    ",": "DELIM_COMMA",
}

TOKEN_PATTERN = re.compile(
    r"(?P<BLOCK_COMMENT>/\*.*?\*/)"
    r"|(?P<LINE_COMMENT>//[^\n]*)"
    r"|(?P<HASH_COMMENT>\#[^\n]*)"
    r"|(?P<STRING>(?:[rubfRUBF]{0,2})(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'))"
    r"|(?P<NUMBER>\b(?:0[xX][0-9a-fA-F]+|0[bB][01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[uUlLfFdDmM]*\b)"
    r"|(?P<IDENT>[A-Za-z_$][A-Za-z0-9_$]*)"
    rf"|(?P<OP>{'|'.join(re.escape(operator) for operator in OPERATORS)})"
    r"|(?P<DELIM>[()\[\]{}:;,])"
    r"|(?P<OTHER>\S)",
    re.DOTALL | re.MULTILINE,
)

BASE_CATEGORIES = sorted(
    set(KEYWORD_CATEGORY.values())
    | {f"OP_{operator}" for operator in OPERATORS}
    | set(DELIMITERS.values())
    | {
        "BEGIN", "END", "IDENTIFIER", "LIT_NUMBER", "LIT_STRING",
        "PREPROCESSOR_IMPORT", "PREPROCESSOR_DEFINE",
        "PREPROCESSOR_CONDITIONAL", "PREPROCESSOR_OTHER", "OTHER",
    }
)
CATEGORY_TO_ID = {category: index for index, category in enumerate(BASE_CATEGORIES)}
CATEGORY_COUNT = len(BASE_CATEGORIES)
FEATURE_COUNT = CATEGORY_COUNT + CATEGORY_COUNT * CATEGORY_COUNT


def _preprocessor_category(text: str, language: str | None) -> str | None:
    if (language or "").strip().lower() not in {"c", "c++", "cpp", "c#", "csharp", "cs"}:
        return None
    directive = text[1:].lstrip().split(maxsplit=1)[0].lower() if text[1:].strip() else ""
    if directive in {"include", "using", "import"}:
        return "PREPROCESSOR_IMPORT"
    if directive in {"define", "undef", "pragma", "line"}:
        return "PREPROCESSOR_DEFINE"
    if directive in {"if", "ifdef", "ifndef", "elif", "else", "endif"}:
        return "PREPROCESSOR_CONDITIONAL"
    if directive and re.fullmatch(r"[A-Za-z_]+", directive):
        return "PREPROCESSOR_OTHER"
    return None


def canonical_token_categories(source: str, language: str | None = None) -> Iterator[int]:
    for match in TOKEN_PATTERN.finditer(source or ""):
        kind = match.lastgroup
        token = match.group(0)
        if kind in {"BLOCK_COMMENT", "LINE_COMMENT"}:
            continue
        if kind == "HASH_COMMENT":
            category = _preprocessor_category(token, language)
            if category is not None:
                yield CATEGORY_TO_ID[category]
        elif kind == "STRING":
            yield CATEGORY_TO_ID["LIT_STRING"]
        elif kind == "NUMBER":
            yield CATEGORY_TO_ID["LIT_NUMBER"]
        elif kind == "IDENT":
            yield CATEGORY_TO_ID[KEYWORD_CATEGORY.get(token.lower(), "IDENTIFIER")]
        elif kind == "OP":
            yield CATEGORY_TO_ID[f"OP_{token}"]
        elif kind == "DELIM":
            yield CATEGORY_TO_ID[DELIMITERS[token]]
        else:
            yield CATEGORY_TO_ID["OTHER"]


def structural_fingerprint(source: str, language: str | None = None) -> int:
    sequence = [CATEGORY_TO_ID["BEGIN"]]
    sequence.extend(canonical_token_categories(source, language))
    sequence.append(CATEGORY_TO_ID["END"])

    fingerprint = 0
    for token_id in sequence:
        fingerprint |= 1 << token_id
    for left, right in zip(sequence, sequence[1:]):
        fingerprint |= 1 << (CATEGORY_COUNT + left * CATEGORY_COUNT + right)
    return fingerprint


def fingerprint_jaccard(left: int, right: int) -> float:
    union_size = (left | right).bit_count()
    return (left & right).bit_count() / union_size if union_size else 1.0


def summarize(values: np.ndarray, semantic_role: str) -> dict[str, float | int | str | None]:
    """Shared summary helper used by calculate_clean_zip_hardness.py."""

    if values.size == 0:
        raise ValueError("Cannot summarize an empty similarity array")
    mean = float(np.mean(values, dtype=np.float64))
    return {
        "pair_count": int(values.size),
        "semantic_role": semantic_role,
        "mean_syntactic_similarity": mean,
        "positive_syntactic_distance": 1.0 - mean if semantic_role == "positive" else None,
        "negative_hardness": mean if semantic_role == "negative" else None,
        "std_syntactic_similarity": float(np.std(values, dtype=np.float64)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "similarity_ge_0_50_ratio": float(np.mean(values >= 0.50)),
        "similarity_ge_0_75_ratio": float(np.mean(values >= 0.75)),
        "similarity_ge_0_90_ratio": float(np.mean(values >= 0.90)),
    }


@dataclass
class ProgramIndex:
    submission_to_row: dict[str, int]
    fingerprints: list[int]


def _progress(items: Iterable, description: str, quiet: bool) -> Iterable:
    if quiet or tqdm is None:
        return items
    return tqdm(items, desc=description, unit="file")


def build_program_index(archive: zipfile.ZipFile, quiet: bool) -> ProgramIndex:
    program_files = sorted(
        name for name in archive.namelist()
        if name.startswith("programs__") and name.endswith(".parquet")
    )
    if not program_files:
        raise ValueError("No programs__*.parquet files found in the dataset ZIP")

    submission_to_row: dict[str, int] = {}
    fingerprints: list[int] = []
    for name in _progress(program_files, "Fingerprinting programs", quiet):
        parquet = pq.ParquetFile(io.BytesIO(archive.read(name)))
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group,
                columns=["submission_id", "language", "source_code"],
            )
            for submission_id, language, source in zip(
                table.column("submission_id").to_pylist(),
                table.column("language").to_pylist(),
                table.column("source_code").to_pylist(),
            ):
                submission_id = str(submission_id)
                if submission_id in submission_to_row:
                    raise ValueError(f"Duplicate submission_id: {submission_id}")
                submission_to_row[submission_id] = len(fingerprints)
                fingerprints.append(structural_fingerprint(source or "", str(language)))
    return ProgramIndex(submission_to_row, fingerprints)


def _pair_file_identity(name: str) -> tuple[str, str]:
    stem = name.removeprefix("pairs__").removesuffix(".parquet")
    try:
        configuration, pair_kind = stem.split("__", 1)
    except ValueError as error:
        raise ValueError(f"Unexpected pair filename: {name}") from error
    if pair_kind not in PAIR_KINDS:
        raise ValueError(f"Unknown pair kind in {name}: {pair_kind}")
    return configuration, pair_kind


def calculate_similarities(
    archive: zipfile.ZipFile,
    programs: ProgramIndex,
    quiet: bool,
) -> dict[tuple[str, str], np.ndarray]:
    pair_files = sorted(
        name for name in archive.namelist()
        if name.startswith("pairs__") and name.endswith(".parquet")
    )
    expected_buckets = {(configuration, kind) for configuration in CONFIGURATIONS for kind in PAIR_KINDS}
    found_buckets = {_pair_file_identity(name) for name in pair_files}
    if found_buckets != expected_buckets:
        raise ValueError(
            "Pair buckets do not match the expected CodeNet 4L layout: "
            f"missing={sorted(expected_buckets - found_buckets)}, "
            f"unexpected={sorted(found_buckets - expected_buckets)}"
        )

    results: dict[tuple[str, str], np.ndarray] = {}
    for name in _progress(pair_files, "Calculating pair similarities", quiet):
        table = pq.read_table(
            io.BytesIO(archive.read(name)),
            columns=["submission_id_a", "submission_id_b"],
        )
        endpoint_a = table.column("submission_id_a").to_pylist()
        endpoint_b = table.column("submission_id_b").to_pylist()
        try:
            rows_a = (programs.submission_to_row[str(value)] for value in endpoint_a)
            rows_b = (programs.submission_to_row[str(value)] for value in endpoint_b)
            similarities = np.fromiter(
                (
                    fingerprint_jaccard(programs.fingerprints[left], programs.fingerprints[right])
                    for left, right in zip(rows_a, rows_b)
                ),
                dtype=np.float64,
                count=len(endpoint_a),
            )
        except KeyError as error:
            raise ValueError(f"Pair endpoint is absent from program tables: {error.args[0]}") from error
        results[_pair_file_identity(name)] = similarities
    return results


def build_table(
    similarities: dict[tuple[str, str], np.ndarray],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for configuration, (scope, display_name) in CONFIGURATIONS.items():
        positive = similarities[(configuration, "clone")]
        negative = np.concatenate(
            [similarities[(configuration, kind)] for kind in NEGATIVE_KINDS]
        )
        rows.append(
            {
                "scope": scope,
                "language_configuration": display_name,
                "positive_pairs": int(positive.size),
                "h_pos": 1.0 - float(np.mean(positive, dtype=np.float64)),
                "negative_pairs": int(negative.size),
                "h_neg": float(np.mean(negative, dtype=np.float64)),
            }
        )
    return rows


def write_csv(rows: Sequence[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "language_configuration",
        "positive_pairs",
        "h_pos",
        "negative_pairs",
        "h_neg",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_default_zip(script_dir: Path) -> Path:
    repository_root = script_dir.parents[2]
    candidates = sorted((repository_root / "data" / "external" / "codenet").glob("*.zip"))
    if len(candidates) != 1:
        raise FileNotFoundError("Pass the CodeNet ZIP path with --dataset-zip")
    return candidates[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    dataset_zip = args.dataset_zip or find_default_zip(script_dir)
    output_csv = args.output_csv or (script_dir / "hardness_by_language.csv")
    if not dataset_zip.is_file():
        raise FileNotFoundError(dataset_zip)

    with zipfile.ZipFile(dataset_zip, "r") as archive:
        if archive.testzip() is not None:
            raise zipfile.BadZipFile("ZIP CRC validation failed")
        programs = build_program_index(archive, args.quiet)
        similarities = calculate_similarities(archive, programs, args.quiet)

    write_csv(build_table(similarities), output_csv)
    print(output_csv.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
