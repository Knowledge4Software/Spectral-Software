"""Prepare the V3 benchmark releases for the shared graph/spectral pipeline.

The release archive is deliberately read in place: only the Parquet members
needed for a selected benchmark are read, not the 2.7 GB archive extracted.
Prepared records use numeric IDs because source IDs such as ``atcoder:123``
cannot safely become Windows file names during Joern extraction.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from spectral_code.evaluation.clean_data_export import (
    SPLITS,
    _read_splits,
    create_clean_data_zip,
    export_graph_spectra_from_sources,
    validate_clean_data_files,
    write_clean_pairs,
)
from spectral_code.utils.artifact_cleanup import cleanup_finalized_pipeline_artifacts
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for
from spectral_code.preprocessing.language_support import normalize_source_language


SPLIT_FILES = {"train": "train", "validation": "valid", "test": "test"}
GRAPH_TYPES = ["ast", "cfg", "ddg", "cpg"]
PAIR_PROVENANCE_FIELDS = (
    "pair_id",
    "source_clone_pair_id",
    "dataset",
    "configuration_id",
    "pair_kind",
    "pair_type",
    "negative_kind",
    "negative_type",
    "clone_type",
    "transformation",
    "transformation_type",
    "generation_method",
    "construction_method",
    "operator",
    "mutation_operator",
    "mutation_family",
    "mutation_engine",
    "mutation_site_policy",
    "mutation_reference",
    "mutation_applied",
    "mutated_side",
    "parent_code_id",
    "mutant_id",
    "behavioral_validation_performed",
    "compilation_checked",
    "execution_checked",
    "hardness_similarity_name",
    "source_clone_similarity",
    "mutation_pair_similarity",
    "pair_similarity",
    "injection_operator",
    "provenance",
    "attack_type",
    "edit_type",
    "perturbation_type",
    "is_mutation",
    "is_injection",
    "mutation_based",
    "injection_based",
)


@dataclass(frozen=True)
class V3Spec:
    key: str
    title: str
    code_member: str | None
    pair_root: str
    left_column: str
    right_column: str
    code_id_column: str
    code_column: str
    language_column: str
    split_code_members: bool = False
    java_compilation_units: bool = False


SPECS = {
    "atcoder_v3": V3Spec(
        "atcoder_v3", "ATCoder V3 problem-disjoint", None,
        "ATCoder/splits_v3_problem_disjoint",
        "function_id_1", "function_id_2", "function_id", "code", "language",
        split_code_members=True, java_compilation_units=True,
    ),
    "gptclonebench_v3": V3Spec(
        "gptclonebench_v3", "GPTCloneBench V3 author-group-safe",
        "GPTCloneBench/binary_classification/codes.parquet",
        "GPTCloneBench/binary_classification/authors_reverse_group_safe/splits_v3",
        "code_id_1", "code_id_2", "code_id", "code", "language",
    ),
    "semanticclonebench_v3": V3Spec(
        "semanticclonebench_v3", "SemanticCloneBench V3 group-disjoint",
        "SemanticCloneBench/binary_classification/codes.parquet",
        "SemanticCloneBench/binary_classification/splits_v3",
        "code1_id", "code2_id", "code_id", "code", "language",
    ),
    "codexglue_v3": V3Spec(
        "codexglue_v3", "CodeXGLUE official V3", 
        "CodeXGLUE/functions.parquet",
        "CodeXGLUE/official_splits_v3",
        "code_id_1", "code_id_2", "code_id", "code", "language",
    ),
}


def default_archive_path() -> Path:
    refreshed = DATA_ROOT / "base datasets.zip"
    return refreshed if refreshed.is_file() else DATA_ROOT / "DataSets.zip"


def default_prepared_dir(key: str) -> Path:
    return DATA_ROOT / "v3_prepared" / key


def _normalise_language(raw: object) -> str:
    return normalize_source_language(raw)


_JAVA_UNIT_RE = re.compile(
    r"(?m)^\s*(?:package\s+[\w.]+\s*;|import\s+[\w.*]+\s*;|"
    r"(?:public\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+\w+)"
)
_CSHARP_UNIT_RE = re.compile(
    r"(?m)^\s*(?:using\s+[\w.]+\s*;|namespace\s+[\w.]+|"
    r"(?:public\s+)?(?:abstract\s+|sealed\s+|static\s+)?(?:class|interface|struct|enum|record)\s+\w+)"
)


def _is_java_compilation_unit(code: str) -> bool:
    """Avoid wrapping full Java files while retaining method-snippet support."""
    return bool(_JAVA_UNIT_RE.search(code))


def _is_csharp_compilation_unit(code: str) -> bool:
    return bool(_CSHARP_UNIT_RE.search(code))


def _resolve_archive_member(archive: zipfile.ZipFile, member: str) -> str:
    """Resolve both the compact refreshed ZIP and the older nested release."""
    candidates = (member, f"unified-code-clone-benchmarks/{member}")
    names = set(archive.namelist())
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise FileNotFoundError(
        f"Archive member is missing: {member} (also tried the legacy nested path)."
    )


def _read_parquet(archive: zipfile.ZipFile, member: str) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - local dependency guard
        raise ModuleNotFoundError("V3 preparation requires pyarrow; install requirements.txt.") from exc
    try:
        payload = archive.read(_resolve_archive_member(archive, member))
    except KeyError as exc:  # pragma: no cover - guarded by resolver
        raise FileNotFoundError(f"Archive member is missing: {member}") from exc
    return pq.read_table(io.BytesIO(payload)).to_pylist()


def _pair_endpoint_columns(
    spec: V3Spec,
    pairs_by_split: dict[str, list[dict]],
) -> tuple[str, str]:
    sample = next((row for rows in pairs_by_split.values() for row in rows), None)
    if sample is None:
        raise RuntimeError(f"{spec.title} contains no pair rows.")
    if spec.left_column in sample and spec.right_column in sample:
        return spec.left_column, spec.right_column
    if "code_id_1" in sample and "code_id_2" in sample:
        return "code_id_1", "code_id_2"
    raise RuntimeError(
        f"Cannot identify pair endpoint columns for {spec.title}; columns={sorted(sample)}"
    )


def _add_inline_pair_codes(
    source_by_id: dict[str, dict],
    pairs_by_split: dict[str, list[dict]],
    spec: V3Spec,
    left_column: str,
    right_column: str,
) -> None:
    """Add refreshed mutation endpoints whose code is embedded in pair rows."""
    for rows in pairs_by_split.values():
        for pair in rows:
            for side, id_column in ((1, left_column), (2, right_column)):
                source_id = str(pair[id_column])
                code = next(
                    (
                        pair.get(candidate)
                        for candidate in (f"code_{side}", f"code{side}")
                        if pair.get(candidate) is not None
                    ),
                    None,
                )
                if code is None:
                    continue
                language = next(
                    (
                        pair.get(candidate)
                        for candidate in (f"language_{side}", f"language_{'a' if side == 1 else 'b'}", "language")
                        if pair.get(candidate) is not None
                    ),
                    None,
                )
                if language is None:
                    raise RuntimeError(f"Inline code {source_id!r} has no language.")
                existing = source_by_id.get(source_id)
                if existing is not None and str(existing[spec.code_column]) != str(code):
                    raise RuntimeError(f"Conflicting inline code records for source ID {source_id!r}.")
                source_by_id[source_id] = {
                    spec.code_id_column: source_id,
                    spec.code_column: str(code),
                    spec.language_column: language,
                }


def _read_pairs(archive: zipfile.ZipFile, spec: V3Spec) -> dict[str, list[dict]]:
    return {
        target: _read_parquet(archive, f"{spec.pair_root}/pairs_{source}.parquet")
        for source, target in SPLIT_FILES.items()
    }


def _read_codes(archive: zipfile.ZipFile, spec: V3Spec) -> list[dict]:
    if spec.split_code_members:
        records: list[dict] = []
        for source in SPLIT_FILES:
            records.extend(_read_parquet(archive, f"{spec.pair_root}/functions_{source}.parquet"))
        return records
    assert spec.code_member is not None
    return _read_parquet(archive, spec.code_member)


def _stable_choice_index(value: object, size: int) -> int:
    """Return a process-independent choice index for reproducible sampling."""
    if size <= 0:
        raise ValueError("size must be positive")
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _prepare_gptclonebench_non_self_pairs(
    pairs_by_split: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict]:
    """Remove identity pairs and restore class balance without changing groups.

    The supplied GPTCloneBench author-group-safe split contains, for every
    author unit, two Type-1 identity positives, one non-identity Type-4
    positive, and three constructed negatives.  Removing only the identity
    rows would therefore change every split from 1:1 to 1:3.  We retain the
    original group-safe assignment and deterministically keep one of the three
    negatives for each author unit.  This gives one non-self positive and one
    negative per unit without moving an endpoint, author unit, or source
    content between train, validation, and test.
    """
    filtered: dict[str, list[dict]] = {}
    audit: dict[str, object] = {
        "policy": (
            "Removed all code_id_1 == code_id_2 rows from the supplied "
            "author-group-safe split and retained one deterministic negative "
            "per anchor author unit to restore 1:1 class balance."
        ),
        "selection": "SHA-256(anchor_author_unit_id) modulo sorted negative candidates",
        "splits": {},
    }

    leakage_fields = (
        ("code_id_1", "code_id_2"),
        ("anchor_author_unit_id", "partner_author_unit_id"),
        ("source_pool_id_1", "source_pool_id_2"),
        ("source_original_content_sha256_1", "source_original_content_sha256_2"),
    )
    split_values: dict[str, dict[str, set[str]]] = {}

    for split, rows in pairs_by_split.items():
        self_rows = [
            row for row in rows
            if str(row["code_id_1"]) == str(row["code_id_2"])
        ]
        positives = [
            row for row in rows
            if int(row["label"]) == 1
            and str(row["code_id_1"]) != str(row["code_id_2"])
        ]
        negatives_by_anchor: dict[str, list[dict]] = {}
        for row in rows:
            if int(row["label"]) != 0:
                continue
            anchor = str(row.get("anchor_author_unit_id", ""))
            if not anchor:
                raise RuntimeError(
                    f"GPTCloneBench {split} negative lacks anchor_author_unit_id."
                )
            negatives_by_anchor.setdefault(anchor, []).append(row)

        positives_by_anchor = Counter(
            str(row.get("anchor_author_unit_id", "")) for row in positives
        )
        if not positives_by_anchor or set(positives_by_anchor.values()) != {1}:
            raise RuntimeError(
                f"GPTCloneBench {split} must contain exactly one non-self positive per anchor."
            )
        if set(negatives_by_anchor) != set(positives_by_anchor):
            raise RuntimeError(
                f"GPTCloneBench {split} positive/negative anchor sets do not match."
            )

        selected_negative_ids: set[str] = set()
        for anchor, candidates in negatives_by_anchor.items():
            ordered = sorted(
                candidates,
                key=lambda row: (str(row.get("pair_kind", "")), str(row.get("pair_id", ""))),
            )
            selected = ordered[_stable_choice_index(anchor, len(ordered))]
            selected_negative_ids.add(str(selected.get("pair_id", "")))
        if "" in selected_negative_ids or len(selected_negative_ids) != len(negatives_by_anchor):
            raise RuntimeError(f"GPTCloneBench {split} negative pair IDs are absent or repeated.")

        kept = [
            row for row in rows
            if (
                int(row["label"]) == 1
                and str(row["code_id_1"]) != str(row["code_id_2"])
            )
            or (
                int(row["label"]) == 0
                and str(row.get("pair_id", "")) in selected_negative_ids
            )
        ]
        positive_count = sum(int(row["label"]) == 1 for row in kept)
        negative_count = len(kept) - positive_count
        if positive_count != negative_count:
            raise RuntimeError(
                f"GPTCloneBench {split} is not balanced after self-pair removal: "
                f"{positive_count} positive, {negative_count} negative."
            )
        if any(str(row["code_id_1"]) == str(row["code_id_2"]) for row in kept):
            raise RuntimeError(f"GPTCloneBench {split} still contains a self-pair.")

        filtered[split] = kept
        split_values[split] = {}
        for columns in leakage_fields:
            if not all(column in rows[0] for column in columns):
                continue
            key = "/".join(columns)
            split_values[split][key] = {
                str(row[column])
                for row in kept
                for column in columns
                if row.get(column) is not None
            }
        audit["splits"][split] = {
            "input_pairs": len(rows),
            "removed_self_pairs": len(self_rows),
            "removed_excess_negatives": sum(int(row["label"]) == 0 for row in rows) - negative_count,
            "output_pairs": len(kept),
            "clone": positive_count,
            "non_clone": negative_count,
        }

    split_names = list(filtered)
    for index, left_split in enumerate(split_names):
        for right_split in split_names[index + 1:]:
            for field, left_values in split_values[left_split].items():
                overlap = left_values & split_values[right_split].get(field, set())
                if overlap:
                    raise RuntimeError(
                        f"GPTCloneBench leakage after filtering: {len(overlap):,} values "
                        f"from {field} occur in both {left_split} and {right_split}."
                    )
    audit["leakage_check"] = (
        "No code ID, author unit, source pool, or original-content hash crosses splits."
    )
    return filtered, audit


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as dst:
        for record in records:
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_pair_provenance(
    path: Path,
    pairs_by_split: dict[str, list[dict]],
) -> dict[str, object]:
    """Preserve only explicit pair-construction fields needed downstream."""
    fields = sorted(
        {
            field
            for rows in pairs_by_split.values()
            for row in rows
            for field in PAIR_PROVENANCE_FIELDS
            if field in row
        }
    )
    counts = {}
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as dst:
        for split in SPLITS:
            rows = pairs_by_split.get(split, [])
            counts[split] = len(rows)
            for row_index, row in enumerate(rows):
                record = {"split": split, "row_index": row_index}
                record.update({field: row.get(field) for field in fields if field in row})
                dst.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
                    + "\n"
                )
    return {"file": path.name, "fields": fields, "rows_by_split": counts}


def _read_pair_provenance(
    prepared_dir: Path,
    splits: dict[str, list[tuple[int, int, int]]],
) -> dict[str, list[dict[str, object]]] | None:
    path = prepared_dir / "pair_provenance.jsonl.gz"
    if not path.is_file():
        return None
    result = {split: [] for split in splits}
    with gzip.open(path, "rt", encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            split = str(record.pop("split"))
            row_index = int(record.pop("row_index"))
            if split not in result or row_index != len(result[split]):
                raise RuntimeError(
                    f"Pair provenance is not aligned with prepared split {split!r} at row {row_index}."
                )
            result[split].append(record)
    for split, rows in splits.items():
        if len(result[split]) != len(rows):
            raise RuntimeError(
                f"Pair provenance for {split} has {len(result[split]):,} rows; expected {len(rows):,}."
            )
    return result


def _existing_atcoder_ids(source_ids: set[str]) -> dict[str, int] | None:
    """Reuse our already-validated ATCoder graph IDs when the source sets match."""
    path = DATA_ROOT / "atcoder" / "code_id_map.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as src:
        rows = csv.DictReader(src)
        mapping = {str(row["source_function_id"]): int(row["code_id"]) for row in rows}
    return mapping if source_ids <= set(mapping) else None


def prepare_v3_benchmark(
    key: str,
    archive_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
    reuse_existing_atcoder_ids: bool = True,
) -> dict:
    """Write graph-pipeline inputs for one V3 benchmark without changing labels."""
    try:
        spec = SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown V3 benchmark {key!r}; choose from {sorted(SPECS)}") from exc
    archive_path = Path(archive_path or default_archive_path()).resolve()
    output_dir = Path(output_dir or default_prepared_dir(key)).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"V3 archive not found: {archive_path}")
    if output_dir.exists():
        if not overwrite and (output_dir / "metadata.json").is_file():
            return json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        if not overwrite:
            raise FileExistsError(f"Prepared directory exists: {output_dir}; use overwrite=True.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        pairs_by_split = _read_pairs(archive, spec)
        source_codes = _read_codes(archive, spec)

    left_column, right_column = _pair_endpoint_columns(spec, pairs_by_split)
    refreshed_mutation_release = any(
        "mutation_applied" in row or str(row.get("pair_kind", "")) == "nonclone_mutation"
        for rows in pairs_by_split.values()
        for row in rows[:1]
    )
    pair_filter_audit = None
    if spec.key == "gptclonebench_v3" and not refreshed_mutation_release:
        pairs_by_split, pair_filter_audit = _prepare_gptclonebench_non_self_pairs(
            pairs_by_split
        )

    referenced_ids = {
        str(pair[column])
        for rows in pairs_by_split.values()
        for pair in rows
        for column in (left_column, right_column)
    }
    source_by_id: dict[str, dict] = {}
    for row in source_codes:
        source_id = str(row[spec.code_id_column])
        if source_id in referenced_ids:
            if source_id in source_by_id and source_by_id[source_id].get(spec.code_column) != row.get(spec.code_column):
                raise RuntimeError(f"Conflicting code records for source ID {source_id!r}.")
            source_by_id[source_id] = row
    _add_inline_pair_codes(
        source_by_id,
        pairs_by_split,
        spec,
        left_column,
        right_column,
    )
    missing = referenced_ids - set(source_by_id)
    if missing:
        raise RuntimeError(f"{len(missing):,} pair-referenced code records are absent from {spec.title}.")

    reusable_ids = (
        _existing_atcoder_ids(set(source_by_id))
        if spec.key == "atcoder_v3" and reuse_existing_atcoder_ids
        else None
    )
    ordered_ids = sorted(source_by_id, key=(lambda source_id: reusable_ids[source_id]) if reusable_ids else str)
    numeric_id = reusable_ids or {source_id: index for index, source_id in enumerate(ordered_ids, start=1)}
    records: list[dict] = []
    languages = Counter()
    for source_id in ordered_ids:
        source = source_by_id[source_id]
        language = _normalise_language(source[spec.language_column])
        if language not in {"java", "python", "c", "cpp", "csharp"}:
            raise RuntimeError(f"Unsupported language {language!r} in {spec.title}.")
        record = {
            "idx": numeric_id[source_id],
            "func": str(source[spec.code_column]),
            "lang": language,
            "source_code_id": source_id,
        }
        if language == "java" and (spec.java_compilation_units or _is_java_compilation_unit(record["func"])):
            record["source_mode"] = "compilation_unit"
        elif language == "csharp" and _is_csharp_compilation_unit(record["func"]):
            record["source_mode"] = "compilation_unit"
        records.append(record)
        languages[language] += 1
    _write_jsonl(output_dir / "data.jsonl", records)
    for language in sorted(languages):
        language_dir = output_dir / language
        language_dir.mkdir()
        _write_jsonl(language_dir / "data.jsonl", (row for row in records if row["lang"] == language))

    with (output_dir / "code_id_map.csv").open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["code_id", "source_code_id", "language"])
        writer.writeheader()
        for record in records:
            writer.writerow({"code_id": record["idx"], "source_code_id": record["source_code_id"], "language": record["lang"]})

    pair_counts: dict[str, dict[str, int]] = {}
    # A release can retain directed/repeated rows *within* one official split.
    # Preserve those rows exactly.  Only the same unordered pair crossing two
    # different splits indicates a split-integrity issue.
    pair_split: dict[tuple[int, int], str] = {}
    for split, rows in pairs_by_split.items():
        counts = Counter()
        with (output_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as dst:
            for pair in rows:
                left = numeric_id[str(pair[left_column])]
                right = numeric_id[str(pair[right_column])]
                label = int(pair["label"])
                if label not in {0, 1}:
                    raise RuntimeError(f"Invalid binary label {label!r} in {spec.title}.")
                pair_key = (min(left, right), max(left, right))
                previous_split = pair_split.get(pair_key)
                if previous_split is not None and previous_split != split:
                    raise RuntimeError(
                        f"Pair overlap across V3 splits for {spec.title}: {pair_key} "
                        f"in {previous_split} and {split}."
                    )
                pair_split[pair_key] = split
                dst.write(f"{left}\t{right}\t{label}\n")
                counts["clone" if label else "non_clone"] += 1
        pair_counts[split] = {"pairs": len(rows), "clone": counts["clone"], "non_clone": counts["non_clone"]}

    split_policy = "Provided V3 train/validation/test split; no resampling or relabelling."
    if pair_filter_audit is not None:
        split_policy = (
            "Supplied author-group-safe train/validation/test assignment retained; "
            "identity positives removed and constructed negatives deterministically "
            "downsampled within each split to restore 1:1 class balance."
        )
    elif refreshed_mutation_release and spec.key in {"gptclonebench_v3", "semanticclonebench_v3"}:
        split_policy = (
            "Provided source-group-safe V3 assignment retained exactly; refreshed rows "
            "contain one mutation-derived negative per retained non-self clone."
        )
    pair_provenance = _write_pair_provenance(
        output_dir / "pair_provenance.jsonl.gz", pairs_by_split
    )
    metadata = {
        "format": "v3_graph_pipeline_input_v1",
        "dataset": spec.title,
        "dataset_key": spec.key,
        "archive_path": str(archive_path),
        "split_policy": split_policy,
        "source_id_mapping": "code_id_map.csv",
        "code_count": len(records),
        "codes_by_language": dict(sorted(languages.items())),
        "pairs": pair_counts,
        "canonical_mapping": "Applied dynamically from the SPECTRA-Siam Kaggle notebook to raw AST types; raw graph records remain preserved.",
        "graph_types_to_extract": GRAPH_TYPES,
        "reused_existing_atcoder_graph_ids": bool(reusable_ids),
        "pair_provenance": pair_provenance,
        "release_layout": "mutation_refresh" if refreshed_mutation_release else "legacy_v3",
    }
    if pair_filter_audit is not None:
        metadata["pair_filter_audit"] = pair_filter_audit
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _write_clean_codes_with_metadata(clean_dir: Path, prepared_dir: Path) -> dict[int, str]:
    codes: dict[int, str] = {}
    temporary = clean_dir / "codes.jsonl.gz.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as dst, (prepared_dir / "data.jsonl").open("r", encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            code_id, code = int(record["idx"]), str(record["func"])
            codes[code_id] = code
            clean_record = {
                "code_id": str(code_id), "source_code_id": record["source_code_id"],
                "code": code, "language": record["lang"],
                # CodeNet records the physical source-line count used during
                # sampling (which intentionally ignores a terminal newline).
                # Preserve that value so the portable artifact reports the
                # same inclusive 20..50 constraint that selected the pairs.
                "line_count": int(record.get("source_line_count", code.count("\n") + 1)),
                "char_count": len(code),
            }
            for field in (
                "problem_id", "status", "source_sha256", "is_mutant",
                "parent_submission_id", "mutation_family", "mutation_operator",
            ):
                if field in record:
                    clean_record[field] = record[field]
            dst.write(json.dumps(clean_record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(clean_dir / "codes.jsonl.gz")
    return codes


def _coverage_by_graph_type(graph_spectra_path: Path, graph_types: list[str]) -> dict[str, dict[str, int | float]]:
    coverage = {kind: {"records": 0, "missing_spectra": 0, "missing_fraction": 0.0} for kind in graph_types}
    with gzip.open(graph_spectra_path, "rt", encoding="utf-8") as src:
        for line in src:
            graphs = json.loads(line).get("graphs", {})
            for kind in graph_types:
                item = graphs.get(kind, {})
                coverage[kind]["records"] += 1
                if not item.get("eigenvalues"):
                    coverage[kind]["missing_spectra"] += 1
    for item in coverage.values():
        item["missing_fraction"] = item["missing_spectra"] / max(int(item["records"]), 1)
    return coverage


def _clean_codes_match_prepared(clean_dir: Path, prepared_dir: Path) -> bool:
    """Return whether a prior clean export represents the same code corpus."""
    clean_codes_path = clean_dir / "codes.jsonl.gz"
    prepared_codes_path = prepared_dir / "data.jsonl"
    if not clean_codes_path.is_file() or not prepared_codes_path.is_file():
        return False
    clean_codes: dict[int, tuple[str, str, str]] = {}
    with gzip.open(clean_codes_path, "rt", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            clean_codes[int(row["code_id"])] = (
                str(row.get("source_code_id", "")),
                str(row.get("language", "")),
                str(row.get("code", "")),
            )
    prepared_codes: dict[int, tuple[str, str, str]] = {}
    with prepared_codes_path.open("r", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            prepared_codes[int(row["idx"])] = (
                str(row.get("source_code_id", "")),
                str(row.get("lang", "")),
                str(row.get("func", "")),
            )
    return clean_codes == prepared_codes


def _reuse_clean_graph_records(
    source: Path,
    output_path: Path,
    needed_ids: set[int],
    graph_types: list[str],
) -> dict[str, int | str]:
    """Reuse a validated prior graph export when only pair rows changed."""
    temporary = output_path.with_name(output_path.name + ".tmp")
    written: set[int] = set()
    layers = missing = 0
    with gzip.open(source, "rt", encoding="utf-8") as src, gzip.open(
        temporary, "wt", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            record = json.loads(line)
            code_id = int(record["code_id"])
            if code_id not in needed_ids:
                continue
            graphs = record.get("graphs", {})
            absent = [kind for kind in graph_types if kind not in graphs]
            if absent:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Reusable graph record {code_id} lacks layers: {absent}"
                )
            for kind in graph_types:
                layers += 1
                if not graphs[kind].get("eigenvalues"):
                    missing += 1
            dst.write(json.dumps(record, separators=(",", ":")) + "\n")
            written.add(code_id)
    absent_ids = needed_ids - written
    if absent_ids:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Reusable graph export is missing {len(absent_ids):,} required code IDs."
        )
    temporary.replace(output_path)
    return {
        "methods": len(written),
        "graph_layers": layers,
        "missing_feature_layers": missing,
        "reused_from": str(source),
    }


def _reuse_existing_atcoder_graph_records(output_path: Path, needed_ids: set[int], graph_types: list[str]) -> dict[str, int]:
    """Filter the prior portable ATCoder export for the V3 re-split.

    The original extraction artefacts were intentionally cleaned after their
    first Kaggle export, but the final sparse graphs and spectra remain in the
    portable ``clean_data`` artifact.  Reusing those exact records is safer
    than parsing the same sources again.
    """
    source = output_root_for("atcoder") / "clean_data" / "graph_spectra.jsonl.gz"
    if not source.is_file():
        raise FileNotFoundError(f"Reusable ATCoder graph export is missing: {source}")
    temporary = output_path.with_name(output_path.name + ".tmp")
    written: set[int] = set()
    layers = missing = 0
    with gzip.open(source, "rt", encoding="utf-8") as src, gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as dst:
        for line in src:
            record = json.loads(line)
            code_id = int(record["code_id"])
            if code_id not in needed_ids:
                continue
            graphs = record.get("graphs", {})
            absent = [kind for kind in graph_types if kind not in graphs]
            if absent:
                raise RuntimeError(f"Reusable ATCoder graph {code_id} lacks layers: {absent}")
            for kind in graph_types:
                feature = graphs[kind]
                layers += 1
                if not feature.get("eigenvalues"):
                    missing += 1
            dst.write(json.dumps({"code_id": str(code_id), "graphs": {kind: graphs[kind] for kind in graph_types}}, separators=(",", ":")) + "\n")
            written.add(code_id)
    absent_ids = needed_ids - written
    if absent_ids:
        repair_roots = [output_root_for("atcoder_v3") / "repair" / language for language in ("java", "python")]
        manifests_ready = all((root / "clean_graphs" / "graph_shards_manifest.json").is_file() and (root / "spectral_features" / "spectral_features_manifest.json").is_file() for root in repair_roots)
        if not manifests_ready:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"{len(absent_ids):,} V3 ATCoder codes have no reusable graph record. "
                "Run the V3 pipeline with --repair-missing-atcoder-v3, then rerun stage 05."
            )
        repair_path = output_path.with_name("atcoder_v3_repair_graphs.jsonl.gz")
        repair_summary = export_graph_spectra_from_sources(repair_roots, repair_path, absent_ids, graph_types, precision=8)
        with gzip.open(temporary, "at", encoding="utf-8", newline="\n") as dst, gzip.open(repair_path, "rt", encoding="utf-8") as src:
            shutil.copyfileobj(src, dst)
        repair_path.unlink(missing_ok=True)
        written.update(absent_ids)
        layers += int(repair_summary["graph_layers"])
        missing += int(repair_summary["missing_feature_layers"])
    temporary.replace(output_path)
    return {"methods": len(written), "graph_layers": layers, "missing_feature_layers": missing, "reused_from": str(source)}


def prepare_atcoder_v3_repair_subset(prepared_dir: str | Path | None = None) -> dict[str, int]:
    """Write the tiny set of V3 codes absent from the older ATCoder export."""
    prepared_dir = Path(prepared_dir or default_prepared_dir("atcoder_v3")).resolve()
    splits, needed_ids = _read_splits(prepared_dir)
    del splits
    source = output_root_for("atcoder") / "clean_data" / "graph_spectra.jsonl.gz"
    present: set[int] = set()
    with gzip.open(source, "rt", encoding="utf-8") as src:
        for line in src:
            present.add(int(json.loads(line)["code_id"]))
    missing = needed_ids - present
    repair_dir = prepared_dir / "repair"
    if repair_dir.exists():
        shutil.rmtree(repair_dir)
    records_by_language: dict[str, list[dict]] = {"java": [], "python": []}
    with (prepared_dir / "data.jsonl").open("r", encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            if int(record["idx"]) in missing:
                records_by_language[record["lang"]].append(record)
    if sum(map(len, records_by_language.values())) != len(missing):
        raise RuntimeError("Could not locate every missing ATCoder V3 code in prepared data.")
    for language, records in records_by_language.items():
        target = repair_dir / language
        target.mkdir(parents=True, exist_ok=True)
        _write_jsonl(target / "data.jsonl", records)
    report = {language: len(records) for language, records in records_by_language.items()}
    (repair_dir / "metadata.json").write_text(json.dumps({"missing_code_count": len(missing), "by_language": report, "code_ids": sorted(missing)}, indent=2), encoding="utf-8")
    return report


def export_v3_clean_dataset(
    key: str,
    prepared_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    graph_types: list[str] | None = None,
    create_zip: bool = True,
    cleanup_intermediates: bool = True,
    max_missing_feature_fraction: float = 0.01,
    reuse_existing_clean_graphs: bool = True,
) -> dict:
    """Merge per-language graph pipelines into one portable Kaggle clean-data set."""
    if key not in SPECS and key != "codenet_4l":
        raise ValueError(
            f"Unknown prepared benchmark {key!r}; choose from {sorted((*SPECS, 'codenet_4l'))}"
        )
    prepared_dir = Path(prepared_dir or default_prepared_dir(key)).resolve()
    output_root = Path(output_root or output_root_for(key)).resolve()
    prepared_metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    graph_types = list(graph_types or GRAPH_TYPES)
    clean_dir = output_root / "clean_data"
    reusable_graph_export: Path | None = None
    if (
        reuse_existing_clean_graphs
        and (clean_dir / "graph_spectra.jsonl.gz").is_file()
        and _clean_codes_match_prepared(clean_dir, prepared_dir)
    ):
        reusable_graph_export = output_root / ".graph_spectra_pair_rebuild.jsonl.gz"
        shutil.copy2(clean_dir / "graph_spectra.jsonl.gz", reusable_graph_export)
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True)
    splits, needed_ids = _read_splits(prepared_dir)
    codes = _write_clean_codes_with_metadata(clean_dir, prepared_dir)
    if needed_ids - set(codes):
        raise RuntimeError("A pair references a missing prepared V3 code record.")
    pair_provenance = _read_pair_provenance(prepared_dir, splits)
    pair_counts = write_clean_pairs(clean_dir, splits, pair_metadata=pair_provenance)
    languages = sorted(prepared_metadata["codes_by_language"])
    # ATCoder V3 is a leakage-safe re-split of the exact code corpus already
    # parsed into ``outputs/atcoder``.  Its preserved numeric IDs let us reuse
    # those graphs rather than wasting another multi-hour Joern extraction.
    reuse_atcoder = key == "atcoder_v3" and prepared_metadata.get("reused_existing_atcoder_graph_ids")
    graph_base = output_root_for("atcoder") if reuse_atcoder else output_root
    graph_roots = [graph_base / language for language in languages]
    if reusable_graph_export is not None:
        graph_summary = _reuse_clean_graph_records(
            reusable_graph_export,
            clean_dir / "graph_spectra.jsonl.gz",
            needed_ids,
            graph_types,
        )
    elif reuse_atcoder:
        graph_summary = _reuse_existing_atcoder_graph_records(
            clean_dir / "graph_spectra.jsonl.gz", needed_ids, graph_types
        )
    else:
        graph_summary = export_graph_spectra_from_sources(
            graph_roots,
            clean_dir / "graph_spectra.jsonl.gz",
            needed_ids,
            graph_types,
            precision=8,
        )
    layers, missing = int(graph_summary["graph_layers"]), int(graph_summary["missing_feature_layers"])
    coverage = _coverage_by_graph_type(clean_dir / "graph_spectra.jsonl.gz", graph_types)
    # AST is the universal representation used by the canonical SPECTRA model.
    # Some Joern frontends (notably C#) do not expose a native CFG/DDG for every
    # syntactically valid fragment.  Keep those layers and report their coverage,
    # but never reject a publishable AST-complete benchmark for that limitation.
    ast_coverage = coverage.get("ast", {"missing_fraction": 0.0, "missing_spectra": 0, "records": 0})
    if float(ast_coverage["missing_fraction"]) > max_missing_feature_fraction:
        raise RuntimeError(
            "AST spectral coverage too low: "
            f"{ast_coverage['missing_spectra']:,}/{ast_coverage['records']:,} missing."
        )
    metadata = {
        "format": "spectral_clean_data_v1", "dataset": prepared_metadata["dataset"],
        "dataset_key": key, "prepared_data": str(prepared_dir),
        "graph_source_roots": {language: str(graph_base / language) for language in languages},
        "graph_types": graph_types, "float_precision": 8,
        "canonical_mapping": prepared_metadata["canonical_mapping"],
        "counts": {"codes": len(codes), "pairs": {**pair_counts, "clone": sum(label == 1 for rows in splits.values() for _, _, label in rows), "non_clone": sum(label == 0 for rows in splits.values() for _, _, label in rows), "total": sum(pair_counts.values())}, "graph_spectra": graph_summary},
        "spectral_coverage_by_graph_type": coverage,
        "required_complete_graph_type": "ast",
        "source_preparation": prepared_metadata,
        "pair_provenance_fields": sorted(
            {
                field
                for rows in (pair_provenance or {}).values()
                for record in rows
                for field in record
            }
        ),
    }
    # Surface the pair-construction policy in the portable bundle as well as
    # retaining the complete preparation metadata above.  This is important
    # for GPTCloneBench, whose V3 pair set removes identity positives and
    # deterministically downsamples negatives without changing author-safe
    # split membership.
    for audit_key in ("split_policy", "pair_filter_audit"):
        if audit_key in prepared_metadata:
            metadata[audit_key] = prepared_metadata[audit_key]
    if cleanup_intermediates and not reuse_atcoder and reusable_graph_export is None:
        metadata["cleanup_after_export"] = {language: cleanup_finalized_pipeline_artifacts(root, compute_size=True) for language, root in zip(languages, graph_roots)}
    (clean_dir / "README.md").write_text(
        f"# {prepared_metadata['dataset']} clean graph data\n\n"
        "Portable source-code, V3 pairs, sparse graph layers, and eigenvalue spectra. "
        "Language is retained per source record; canonical AST categories are applied by the training notebook.\n",
        encoding="utf-8",
    )
    (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_clean_data_files(clean_dir)
    if create_zip:
        zip_path = output_root / f"{key}_clean_data.zip"
        create_clean_data_zip(clean_dir, zip_path)
        metadata["zip"] = str(zip_path)
        (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if reusable_graph_export is not None:
        reusable_graph_export.unlink(missing_ok=True)
    return metadata
