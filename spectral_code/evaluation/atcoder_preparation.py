"""Prepare and export the normalized ATCoder release for the graph pipeline.

The upstream release stores code and labels in four Parquet tables inside the
unified benchmark archive.  This module creates the same ``data.jsonl`` and
split-file interface consumed by the existing XGLUE pipeline, while retaining
language and source identifiers needed for the final cross-language export.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from spectral_code.evaluation.clean_data_export import (
    CLEAN_DATA_REQUIRED_FILES,
    _read_splits,
    create_clean_data_zip,
    export_graph_spectra_from_sources,
    validate_clean_data_files,
    write_clean_pairs,
)
from spectral_code.utils.artifact_cleanup import cleanup_finalized_pipeline_artifacts
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for


ATCODER_ROOT = "ATCoder"
SOURCE_SPLIT_TO_EXPORT_SPLIT = {"train": "train", "dev": "valid", "test": "test"}
COMMON_GRAPH_TYPES = ["ast", "cfg", "ddg", "cpg"]


def default_archive_path() -> Path:
    return DATA_ROOT / "archive.zip"


def default_prepared_dir() -> Path:
    return DATA_ROOT / "atcoder"


def _read_parquet_from_zip(archive: zipfile.ZipFile, member: str):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
        raise ModuleNotFoundError(
            "ATCoder preparation needs pyarrow. Install project requirements first: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return pq.read_table(io.BytesIO(archive.read(member)))


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_atcoder_dataset(
    archive_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    include_invalid_generated_negatives: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert ATCoder Parquet tables into language-specific graph inputs.

    Function IDs in the release are strings (for example ``atcoder:665463``),
    while the graph pipeline uses integer file IDs.  ``code_id_map.csv`` is the
    reversible mapping between the two representations.
    """
    archive_path = Path(archive_path or default_archive_path()).resolve()
    output_dir = Path(output_dir or default_prepared_dir()).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"ATCoder archive not found: {archive_path}")
    if output_dir.exists():
        if not overwrite:
            required = [output_dir / "data.jsonl", output_dir / "train.txt"]
            if all(path.exists() for path in required):
                print(f"[*] Reusing prepared ATCoder data: {output_dir}")
                return json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            raise FileExistsError(f"Prepared ATCoder directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "java").mkdir()
    (output_dir / "python").mkdir()

    with zipfile.ZipFile(archive_path) as archive:
        functions = _read_parquet_from_zip(archive, f"{ATCODER_ROOT}/functions.parquet").to_pylist()
        function_metadata = _read_parquet_from_zip(
            archive, f"{ATCODER_ROOT}/function_metadata.parquet"
        ).to_pylist()
        pairs = _read_parquet_from_zip(archive, f"{ATCODER_ROOT}/pairs.parquet").to_pydict()
        pair_metadata = _read_parquet_from_zip(
            archive, f"{ATCODER_ROOT}/pair_metadata.parquet"
        ).to_pydict()

    metadata_by_source_id = {row["function_id"]: row for row in function_metadata}
    if len(metadata_by_source_id) != len(functions):
        raise RuntimeError("ATCoder functions and function metadata do not have matching primary keys.")

    code_rows: list[dict[str, object]] = []
    source_to_code_id: dict[str, int] = {}
    for code_id, row in enumerate(functions, start=1):
        source_id = row["function_id"]
        source_meta = metadata_by_source_id.get(source_id)
        if source_meta is None:
            raise RuntimeError(f"Missing function metadata for {source_id!r}.")
        language = str(source_meta["language"]).strip().lower()
        if language not in {"java", "python"}:
            raise RuntimeError(f"Unsupported ATCoder language: {source_meta['language']!r}")
        source_to_code_id[source_id] = code_id
        code_rows.append(
            {
                "idx": code_id,
                "func": row["code"],
                "lang": language,
                # AtCoder Java records are complete submissions, not method bodies.
                # The generic XGLUE path wraps Java method snippets, which would
                # make these compilation units invalid nested source files.
                "source_mode": "compilation_unit" if language == "java" else None,
                "source_function_id": source_id,
                "problem_id": source_meta["problem_id"],
                "source_split": source_meta["split"],
            }
        )

    _write_jsonl(output_dir / "data.jsonl", code_rows)
    _write_jsonl(output_dir / "java" / "data.jsonl", (row for row in code_rows if row["lang"] == "java"))
    _write_jsonl(output_dir / "python" / "data.jsonl", (row for row in code_rows if row["lang"] == "python"))

    with (output_dir / "code_id_map.csv").open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["code_id", "source_function_id", "language", "problem_id", "source_split"])
        writer.writeheader()
        for row in code_rows:
            writer.writerow(
                {
                    "code_id": row["idx"],
                    "source_function_id": row["source_function_id"],
                    "language": row["lang"],
                    "problem_id": row["problem_id"],
                    "source_split": row["source_split"],
                }
            )

    if pairs["pair_id"] != pair_metadata["pair_id"]:
        raise RuntimeError("ATCoder pair and pair-metadata rows are not aligned by pair_id.")
    split_handles = {
        split: (output_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n")
        for split in SOURCE_SPLIT_TO_EXPORT_SPLIT.values()
    }
    retained = Counter()
    source_counts = Counter()
    dropped_invalid = Counter()
    try:
        for index, source_pair_id in enumerate(pairs["pair_id"]):
            source_split = str(pair_metadata["split"][index])
            try:
                export_split = SOURCE_SPLIT_TO_EXPORT_SPLIT[source_split]
            except KeyError as exc:
                raise RuntimeError(f"Unexpected ATCoder split: {source_split!r}") from exc
            label = int(pair_metadata["label"][index])
            source_counts[(export_split, label)] += 1
            recommended = bool(pair_metadata["recommended_for_conservative_use"][index])
            if not include_invalid_generated_negatives and not recommended:
                dropped_invalid[(export_split, label)] += 1
                continue
            try:
                left_id = source_to_code_id[pairs["function_id_1"][index]]
                right_id = source_to_code_id[pairs["function_id_2"][index]]
            except KeyError as exc:
                raise RuntimeError(f"Pair {source_pair_id!r} references an unknown function.") from exc
            split_handles[export_split].write(f"{left_id}\t{right_id}\t{label}\n")
            retained[(export_split, label)] += 1
    finally:
        for handle in split_handles.values():
            handle.close()

    metadata = {
        "dataset": "ATCoder Java-Python official generated dataset",
        "archive_path": str(archive_path),
        "archive_member_root": ATCODER_ROOT,
        "graph_input_format": "spectral_pipeline_jsonl_v1",
        "function_count": len(code_rows),
        "functions_by_language": dict(Counter(str(row["lang"]) for row in code_rows)),
        "source_pair_count": len(pairs["pair_id"]),
        "include_invalid_generated_negatives": include_invalid_generated_negatives,
        "source_split_mapping": SOURCE_SPLIT_TO_EXPORT_SPLIT,
        "source_pair_counts": {f"{split}_{label}": count for (split, label), count in sorted(source_counts.items())},
        "retained_pair_counts": {f"{split}_{label}": count for (split, label), count in sorted(retained.items())},
        "dropped_invalid_generated_negative_counts": {
            f"{split}_{label}": count for (split, label), count in sorted(dropped_invalid.items())
        },
        "retained_pairs_total": sum(retained.values()),
        "code_id_map": "code_id_map.csv",
        "policy": (
            "Invalid generated negatives are excluded by default. Set "
            "ATCODER_INCLUDE_INVALID_GENERATED_NEGATIVES=1 to preserve all official rows."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _write_clean_codes_with_atcoder_metadata(output_dir: Path, prepared_dir: Path) -> dict[int, str]:
    codes: dict[int, str] = {}
    output_path = output_dir / "codes.jsonl.gz"
    with gzip.open(output_path, "wt", encoding="utf-8", newline="\n") as dst, (prepared_dir / "data.jsonl").open(
        "r", encoding="utf-8"
    ) as src:
        for line in src:
            row = json.loads(line)
            code_id = int(row["idx"])
            code = str(row["func"])
            codes[code_id] = code
            dst.write(
                json.dumps(
                    {
                        "code_id": str(code_id),
                        "code": code,
                        "language": row["lang"],
                        "source_function_id": row["source_function_id"],
                        "problem_id": row["problem_id"],
                        "source_split": row["source_split"],
                        "line_count": code.count("\n") + 1,
                        "char_count": len(code),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return codes


def export_atcoder_clean_dataset(
    prepared_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    graph_types: list[str] | None = None,
    create_zip: bool = True,
    cleanup_intermediates: bool = True,
    max_missing_feature_fraction: float = 0.01,
) -> dict[str, object]:
    """Merge the independently extracted Java/Python graphs into one upload."""
    prepared_dir = Path(prepared_dir or default_prepared_dir()).resolve()
    output_root = Path(output_root or output_root_for("atcoder")).resolve()
    graph_types = list(graph_types or COMMON_GRAPH_TYPES)
    clean_dir = output_root / "clean_data"
    java_root = output_root / "java"
    python_root = output_root / "python"
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=False)

    splits, needed_ids = _read_splits(prepared_dir)
    codes = _write_clean_codes_with_atcoder_metadata(clean_dir, prepared_dir)
    missing_codes = needed_ids - set(codes)
    if missing_codes:
        raise RuntimeError(f"{len(missing_codes):,} pair-referenced ATCoder functions are missing from data.jsonl.")
    pair_counts = write_clean_pairs(clean_dir, splits)
    graph_summary = export_graph_spectra_from_sources(
        [java_root, python_root],
        clean_dir / "graph_spectra.jsonl.gz",
        needed_ids,
        graph_types,
        precision=8,
    )
    total_layers = int(graph_summary["graph_layers"])
    missing_layers = int(graph_summary["missing_feature_layers"])
    if total_layers and missing_layers / total_layers > max_missing_feature_fraction:
        raise RuntimeError(
            "ATCoder graph/spectral coverage is too incomplete for publication: "
            f"{missing_layers:,}/{total_layers:,} missing feature layers "
            f"({missing_layers / total_layers:.2%}). Fix extraction before exporting."
        )
    source_metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "format": "spectral_clean_data_v1",
        "dataset": "ATCoder",
        "prepared_data_dir": str(prepared_dir),
        "graph_source_roots": {"java": str(java_root), "python": str(python_root)},
        "graph_types": graph_types,
        "float_precision": 8,
        "max_missing_feature_fraction": max_missing_feature_fraction,
        "source_split_mapping": SOURCE_SPLIT_TO_EXPORT_SPLIT,
        "counts": {
            "codes": len(codes),
            "pairs": {
                **pair_counts,
                "clone": sum(label == 1 for rows in splits.values() for _, _, label in rows),
                "non_clone": sum(label == 0 for rows in splits.values() for _, _, label in rows),
                "total": sum(pair_counts.values()),
            },
            "graph_spectra": graph_summary,
        },
        "source_preparation": source_metadata,
    }
    (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (clean_dir / "README.md").write_text(
        "# ATCoder clean graph dataset\n\n"
        "Cross-language Java/Python clone classification data. `codes.jsonl.gz` contains source language and "
        "the reversible original function ID; `pairs.csv.gz` uses `train`, `valid`, and `test` splits. "
        "Graphs and eigenvalues were extracted separately with the matching Joern frontend, then merged.\n",
        encoding="utf-8",
    )
    if cleanup_intermediates:
        metadata["cleanup_after_export"] = {
            "java": cleanup_finalized_pipeline_artifacts(java_root, compute_size=True),
            "python": cleanup_finalized_pipeline_artifacts(python_root, compute_size=True),
        }
    if create_zip:
        metadata["zip"] = str(output_root / "atcoder_clean_data.zip")
    (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_clean_data_files(clean_dir)
    if create_zip:
        create_clean_data_zip(clean_dir, Path(metadata["zip"]))
    return metadata
