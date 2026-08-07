"""Prepare the V3 benchmark releases for the shared graph/spectral pipeline.

The release archive is deliberately read in place: only the Parquet members
needed for a selected benchmark are read, not the 2.7 GB archive extracted.
Prepared records use numeric IDs because source IDs such as ``atcoder:123``
cannot safely become Windows file names during Joern extraction.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from spectral_code.evaluation.clean_data_export import (
    _read_splits,
    create_clean_data_zip,
    export_graph_spectra_from_sources,
    validate_clean_data_files,
    write_clean_pairs,
)
from spectral_code.utils.artifact_cleanup import cleanup_finalized_pipeline_artifacts
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for


SPLIT_FILES = {"train": "train", "validation": "valid", "test": "test"}
GRAPH_TYPES = ["ast", "cfg", "ddg", "cpg"]
LANGUAGE_ALIASES = {"c#": "csharp", "cs": "csharp", "csharp": "csharp", "py": "python"}


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
        "unified-code-clone-benchmarks/ATCoder/splits_v3_problem_disjoint",
        "function_id_1", "function_id_2", "function_id", "code", "language",
        split_code_members=True, java_compilation_units=True,
    ),
    "gptclonebench_v3": V3Spec(
        "gptclonebench_v3", "GPTCloneBench V3 author-group-safe",
        "unified-code-clone-benchmarks/GPTCloneBench/binary_classification/codes.parquet",
        "unified-code-clone-benchmarks/GPTCloneBench/binary_classification/authors_reverse_group_safe/splits_v3",
        "code_id_1", "code_id_2", "code_id", "code", "language",
    ),
    "semanticclonebench_v3": V3Spec(
        "semanticclonebench_v3", "SemanticCloneBench V3 group-disjoint",
        "unified-code-clone-benchmarks/SemanticCloneBench/binary_classification/codes.parquet",
        "unified-code-clone-benchmarks/SemanticCloneBench/binary_classification/splits_v3",
        "code1_id", "code2_id", "code_id", "code", "language",
    ),
    "codexglue_v3": V3Spec(
        "codexglue_v3", "CodeXGLUE official V3", 
        "unified-code-clone-benchmarks/CodeXGLUE/functions.parquet",
        "unified-code-clone-benchmarks/CodeXGLUE/official_splits_v3",
        "code_id_1", "code_id_2", "code_id", "code", "language",
    ),
}


def default_archive_path() -> Path:
    return DATA_ROOT / "DataSets.zip"


def default_prepared_dir(key: str) -> Path:
    return DATA_ROOT / "v3_prepared" / key


def _normalise_language(raw: object) -> str:
    language = str(raw or "").strip().lower()
    return LANGUAGE_ALIASES.get(language, language)


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


def _read_parquet(archive: zipfile.ZipFile, member: str) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - local dependency guard
        raise ModuleNotFoundError("V3 preparation requires pyarrow; install requirements.txt.") from exc
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise FileNotFoundError(f"Archive member is missing: {member}") from exc
    return pq.read_table(io.BytesIO(payload)).to_pylist()


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


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as dst:
        for record in records:
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


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

    referenced_ids = {
        str(pair[column])
        for rows in pairs_by_split.values()
        for pair in rows
        for column in (spec.left_column, spec.right_column)
    }
    source_by_id: dict[str, dict] = {}
    for row in source_codes:
        source_id = str(row[spec.code_id_column])
        if source_id in referenced_ids:
            if source_id in source_by_id and source_by_id[source_id].get(spec.code_column) != row.get(spec.code_column):
                raise RuntimeError(f"Conflicting code records for source ID {source_id!r}.")
            source_by_id[source_id] = row
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
        if language not in {"java", "python", "c", "csharp"}:
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
                left = numeric_id[str(pair[spec.left_column])]
                right = numeric_id[str(pair[spec.right_column])]
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

    metadata = {
        "format": "v3_graph_pipeline_input_v1",
        "dataset": spec.title,
        "dataset_key": spec.key,
        "archive_path": str(archive_path),
        "split_policy": "Provided V3 train/validation/test split; no resampling or relabelling.",
        "source_id_mapping": "code_id_map.csv",
        "code_count": len(records),
        "codes_by_language": dict(sorted(languages.items())),
        "pairs": pair_counts,
        "canonical_mapping": "Applied dynamically from the SPECTRA-Siam Kaggle notebook to raw AST types; raw graph records remain preserved.",
        "graph_types_to_extract": GRAPH_TYPES,
        "reused_existing_atcoder_graph_ids": bool(reusable_ids),
    }
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
            dst.write(json.dumps({
                "code_id": str(code_id), "source_code_id": record["source_code_id"],
                "code": code, "language": record["lang"],
                "line_count": code.count("\n") + 1, "char_count": len(code),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
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
) -> dict:
    """Merge per-language graph pipelines into one portable Kaggle clean-data set."""
    if key not in SPECS:
        raise ValueError(f"Unknown V3 benchmark {key!r}; choose from {sorted(SPECS)}")
    prepared_dir = Path(prepared_dir or default_prepared_dir(key)).resolve()
    output_root = Path(output_root or output_root_for(key)).resolve()
    prepared_metadata = json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))
    graph_types = list(graph_types or GRAPH_TYPES)
    clean_dir = output_root / "clean_data"
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True)
    splits, needed_ids = _read_splits(prepared_dir)
    codes = _write_clean_codes_with_metadata(clean_dir, prepared_dir)
    if needed_ids - set(codes):
        raise RuntimeError("A pair references a missing prepared V3 code record.")
    pair_counts = write_clean_pairs(clean_dir, splits)
    languages = sorted(prepared_metadata["codes_by_language"])
    # ATCoder V3 is a leakage-safe re-split of the exact code corpus already
    # parsed into ``outputs/atcoder``.  Its preserved numeric IDs let us reuse
    # those graphs rather than wasting another multi-hour Joern extraction.
    reuse_atcoder = key == "atcoder_v3" and prepared_metadata.get("reused_existing_atcoder_graph_ids")
    graph_base = output_root_for("atcoder") if reuse_atcoder else output_root
    graph_roots = [graph_base / language for language in languages]
    graph_summary = (
        _reuse_existing_atcoder_graph_records(clean_dir / "graph_spectra.jsonl.gz", needed_ids, graph_types)
        if reuse_atcoder
        else export_graph_spectra_from_sources(graph_roots, clean_dir / "graph_spectra.jsonl.gz", needed_ids, graph_types, precision=8)
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
    }
    if cleanup_intermediates and not reuse_atcoder:
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
    return metadata
