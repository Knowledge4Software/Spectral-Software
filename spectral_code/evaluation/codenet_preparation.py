"""Prepare the pre-split Project CodeNet 4L release for graph extraction.

The compact release contains one Parquet file per configuration, pair kind,
and official pair-level train/validation/test split.  This module preserves
those splits and can either retain the full 400,000 rows or take a deterministic
uniform subset across the 40 configuration/pair-kind buckets.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import io
import json
import shutil
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

from spectral_code.evaluation.v3_benchmark_preparation import (
    GRAPH_TYPES,
    export_v3_clean_dataset,
)
from spectral_code.preprocessing.language_support import normalize_source_language
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for


CONFIGURATIONS = (
    "python", "java", "cpp", "csharp",
    "python_java", "python_cpp", "python_csharp",
    "java_cpp", "java_csharp", "cpp_csharp",
)
PAIR_KINDS = ("clone", "nonclone_diff_problem", "hard_nonclone", "nonclone_mutation")
LANGUAGES = ("python", "java", "cpp", "csharp")
SPLITS = ("train", "valid", "test")
ARCHIVE_SPLITS = {"train": "train", "valid": "validation", "test": "test"}
SAMPLE_SEED = 20260815
DEFAULT_SAMPLE_SIZE: int | None = None

PAIR_PROVENANCE_FIELDS = (
    "pair_id", "source_clone_pair_id", "configuration_id", "pair_kind",
    "is_hard", "scope", "same_problem", "same_language", "evidence_type",
    "counterexample_available", "hard_negative_orientation",
    "mutation_family", "mutation_operator", "mutation_engine",
    "mutation_site_policy", "mutation_reference", "mutated_side",
    "parent_submission_id", "mutant_id", "behavioral_validation_performed",
    "hardness_similarity_name", "source_clone_similarity",
    "mutation_pair_similarity", "clone_similarity_score",
    "sampling_policy", "sampling_master_seed", "sampling_bucket_seed",
    "sample_rank", "split_row_index", "language_scope_normalized",
)


def default_archive_path() -> Path:
    return DATA_ROOT / "codenet dataset.zip"


def default_prepared_dir() -> Path:
    return DATA_ROOT / "codenet_4l_prepared"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet(archive: zipfile.ZipFile, member: str, columns: list[str] | None = None):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("CodeNet preparation requires pyarrow.") from exc
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise FileNotFoundError(f"CodeNet archive member is missing: {member}") from exc
    return pq.read_table(io.BytesIO(payload), columns=columns)


def _selected(values: list[str] | tuple[str, ...] | None, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    result = tuple(values or allowed)
    invalid = sorted(set(result) - set(allowed))
    if not result or invalid:
        raise ValueError(f"Invalid {label}: {invalid}; choose from {allowed}")
    return result


def _pair_endpoints(row: dict) -> tuple[str, str]:
    if "program_id_a" in row:
        return str(row["program_id_a"]), str(row["program_id_b"])
    return str(row["submission_id_a"]), str(row["submission_id_b"])


def _pair_provenance(row: dict) -> dict[str, object]:
    return {field: row.get(field) for field in PAIR_PROVENANCE_FIELDS if field in row}


def _json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"


def _allocate_integer(total: int, keys: tuple[str, ...], weights: tuple[float, ...]) -> dict[str, int]:
    """Allocate an exact integer total with deterministic largest remainders."""
    if total < 0 or not keys or len(keys) != len(weights) or sum(weights) <= 0:
        raise ValueError("Invalid integer-allocation request.")
    raw = [total * weight / sum(weights) for weight in weights]
    values = [int(value) for value in raw]
    remainder = total - sum(values)
    order = sorted(range(len(keys)), key=lambda index: (-(raw[index] - values[index]), index))
    for index in order[:remainder]:
        values[index] += 1
    return dict(zip(keys, values))


def _sample_targets(
    sample_size: int,
    configurations: tuple[str, ...],
    pair_kinds: tuple[str, ...],
) -> dict[tuple[str, str, str], int]:
    """Return exact quotas while keeping all selected bucket totals uniform."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    buckets = tuple((configuration, pair_kind) for configuration in configurations for pair_kind in pair_kinds)
    per_bucket = _allocate_integer(
        sample_size,
        tuple(f"{configuration}/{pair_kind}" for configuration, pair_kind in buckets),
        tuple(1.0 for _ in buckets),
    )
    targets: dict[tuple[str, str, str], int] = {}
    for bucket_index, (configuration, pair_kind) in enumerate(buckets):
        bucket_total = per_bucket[f"{configuration}/{pair_kind}"]
        raw = [bucket_total * weight for weight in (0.70, 0.15, 0.15)]
        values = [int(value) for value in raw]
        remainder = bucket_total - sum(values)
        # Alternate validation/test tie-breaking so their aggregate counts stay
        # equal even when an individual bucket cannot split an odd remainder.
        tie_order = (0, 1, 2) if bucket_index % 2 == 0 else (0, 2, 1)
        priority = {index: rank for rank, index in enumerate(tie_order)}
        order = sorted(range(3), key=lambda index: (-(raw[index] - values[index]), priority[index]))
        for index in order[:remainder]:
            values[index] += 1
        for split, count in zip(SPLITS, values):
            targets[(configuration, pair_kind, split)] = count
    if sum(targets.values()) != sample_size:
        raise RuntimeError("Internal CodeNet sampling quotas do not sum to sample_size.")
    return targets


def _capacity_aware_sample_targets(
    sample_size: int,
    configurations: tuple[str, ...],
    pair_kinds: tuple[str, ...],
    capacities: dict[tuple[str, str, str], int],
) -> dict[tuple[str, str, str], int]:
    """Rebalance split quotas while retaining equal configuration/kind totals.

    A min-cost flow first fills the ordinary per-bucket 70/15/15 targets and
    uses extra capacity only where a source-length filter makes those exact
    targets impossible. Global split totals and every configuration/kind total
    remain identical to the unfiltered uniform design.
    """
    preferred = _sample_targets(sample_size, configurations, pair_kinds)
    allocated: dict[tuple[str, str, str], int] = {}

    for pair_kind in pair_kinds:
        source, sink = "source", "sink"
        adjacency: dict[str, list[list]] = {}
        tracked: dict[tuple[str, str], list[list]] = {}

        def add_edge(left: str, right: str, capacity: int, cost: int) -> list:
            adjacency.setdefault(left, [])
            adjacency.setdefault(right, [])
            forward = [right, len(adjacency[right]), int(capacity), int(cost), int(capacity)]
            reverse = [left, len(adjacency[left]), 0, -int(cost), 0]
            adjacency[left].append(forward)
            adjacency[right].append(reverse)
            return forward

        configuration_totals = {
            configuration: sum(preferred[(configuration, pair_kind, split)] for split in SPLITS)
            for configuration in configurations
        }
        split_totals = {
            split: sum(preferred[(configuration, pair_kind, split)] for configuration in configurations)
            for split in SPLITS
        }
        required_total = sum(configuration_totals.values())
        for configuration in configurations:
            config_node = f"configuration:{configuration}"
            add_edge(source, config_node, configuration_totals[configuration], 0)
            for split in SPLITS:
                split_node = f"split:{split}"
                capacity = capacities[(configuration, pair_kind, split)]
                preferred_capacity = min(preferred[(configuration, pair_kind, split)], capacity)
                tracked[(configuration, split)] = [
                    add_edge(config_node, split_node, preferred_capacity, 0)
                ]
                overflow = capacity - preferred_capacity
                if overflow:
                    tracked[(configuration, split)].append(add_edge(config_node, split_node, overflow, 1))
        for split in SPLITS:
            add_edge(f"split:{split}", sink, split_totals[split], 0)

        flow = 0
        nodes = tuple(adjacency)
        while flow < required_total:
            distance = {node: float("inf") for node in nodes}
            previous: dict[str, tuple[str, int]] = {}
            distance[source] = 0
            for _ in range(len(nodes) - 1):
                changed = False
                for left in nodes:
                    if distance[left] == float("inf"):
                        continue
                    for edge_index, edge in enumerate(adjacency[left]):
                        right, _, remaining, cost, _ = edge
                        candidate = distance[left] + cost
                        if remaining > 0 and candidate < distance[right]:
                            distance[right] = candidate
                            previous[right] = (left, edge_index)
                            changed = True
                if not changed:
                    break
            if sink not in previous:
                break
            amount = required_total - flow
            cursor = sink
            while cursor != source:
                left, edge_index = previous[cursor]
                amount = min(amount, adjacency[left][edge_index][2])
                cursor = left
            cursor = sink
            while cursor != source:
                left, edge_index = previous[cursor]
                edge = adjacency[left][edge_index]
                reverse_index = edge[1]
                edge[2] -= amount
                adjacency[cursor][reverse_index][2] += amount
                cursor = left
            flow += amount

        if flow != required_total:
            raise RuntimeError(
                f"Source-line filter cannot supply {required_total:,} {pair_kind} pairs while preserving "
                "equal language-configuration totals and the global 70/15/15 split."
            )
        for configuration in configurations:
            for split in SPLITS:
                allocated[(configuration, pair_kind, split)] = sum(
                    edge[4] - edge[2] for edge in tracked[(configuration, split)]
                )

    if sum(allocated.values()) != sample_size:
        raise RuntimeError("Capacity-aware CodeNet quotas do not sum to sample_size.")
    return allocated


def _sample_rank(row: dict, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{row['pair_id']}".encode("utf-8")).digest()


def _source_line_count(source: object) -> int:
    """Count physical source lines, ignoring a terminal newline-only sentinel."""
    return max(1, len(str(source).splitlines()))


def prepare_codenet_dataset(
    archive_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    configurations: list[str] | tuple[str, ...] | None = None,
    pair_kinds: list[str] | tuple[str, ...] | None = None,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = SAMPLE_SEED,
    min_program_lines: int | None = None,
    max_program_lines: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Create graph-pipeline JSONL while preserving the release's pair splits.

    ``sample_size=None`` retains every row from the selected buckets.  A
    positive sample size takes a deterministic uniform subset across selected
    configuration/pair-kind buckets, preserving the release's 70/15/15 ratio.
    Optional inclusive source-line bounds are applied to both endpoints before
    deterministic pair sampling.
    """
    archive_path = Path(archive_path or default_archive_path()).resolve()
    output_dir = Path(output_dir or default_prepared_dir()).resolve()
    configurations = _selected(configurations, CONFIGURATIONS, "configurations")
    pair_kinds = _selected(pair_kinds, PAIR_KINDS, "pair kinds")
    if min_program_lines is not None and min_program_lines < 1:
        raise ValueError("min_program_lines must be at least 1 when supplied.")
    if max_program_lines is not None and max_program_lines < 1:
        raise ValueError("max_program_lines must be at least 1 when supplied.")
    if (
        min_program_lines is not None
        and max_program_lines is not None
        and min_program_lines > max_program_lines
    ):
        raise ValueError("min_program_lines must not exceed max_program_lines.")
    line_filter = (
        None
        if min_program_lines is None and max_program_lines is None
        else {
            "minimum": min_program_lines,
            "maximum": max_program_lines,
            "inclusive": True,
            "applies_to": "both_pair_endpoints",
        }
    )
    if not archive_path.is_file():
        raise FileNotFoundError(f"CodeNet archive not found: {archive_path}")
    archive_sha256 = _file_sha256(archive_path)
    sample_targets = _sample_targets(sample_size, configurations, pair_kinds) if sample_size is not None else None
    if output_dir.exists():
        if not overwrite and (output_dir / "metadata.json").is_file():
            existing = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            if existing.get("source_archive_sha256") != archive_sha256:
                raise RuntimeError(
                    f"Prepared CodeNet data was built from another archive; use overwrite=True: {output_dir}"
                )
            expected_mode = "full" if sample_size is None else "uniform_subset"
            if (
                existing.get("sampling_mode") != expected_mode
                or existing.get("requested_sample_size") != sample_size
                or existing.get("configurations") != list(configurations)
                or existing.get("pair_kinds") != list(pair_kinds)
                or existing.get("program_line_filter") != line_filter
            ):
                raise RuntimeError(
                    "Prepared CodeNet options do not match this request; use overwrite=True: "
                    f"{output_dir}"
                )
            return existing
        if not overwrite:
            raise FileExistsError(f"Prepared directory exists: {output_dir}; use overwrite=True.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    database_path = output_dir / ".codenet_prepare.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE code_ids (source_id TEXT PRIMARY KEY, numeric_id INTEGER)")
    connection.execute(
        "CREATE TABLE pairs (split TEXT, left_source TEXT, right_source TEXT, "
        "label INTEGER, provenance TEXT)"
    )

    bucket_counts: dict[str, dict[str, int]] = {}
    pair_counts = {split: Counter() for split in SPLITS}
    pair_kind_counts = {split: Counter() for split in SPLITS}
    needed_languages: set[str] = set()
    actual_sample_targets: dict[tuple[str, str, str], int] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            eligible_program_ids: set[str] | None = None
            eligible_programs_by_language: Counter = Counter()
            if line_filter is not None:
                eligible_program_ids = set()
                for language in LANGUAGES:
                    member = f"programs__{language}.parquet"
                    table = _read_parquet(archive, member, columns=["program_id", "language", "source_code"])
                    for row in table.to_pylist():
                        line_count = _source_line_count(row["source_code"])
                        if (
                            (min_program_lines is None or line_count >= min_program_lines)
                            and (max_program_lines is None or line_count <= max_program_lines)
                        ):
                            eligible_program_ids.add(str(row["program_id"]))
                            eligible_programs_by_language[normalize_source_language(row["language"])] += 1

                if sample_size is not None:
                    capacities: dict[tuple[str, str, str], int] = {}
                    for configuration in configurations:
                        for pair_kind in pair_kinds:
                            for split in SPLITS:
                                archive_split = ARCHIVE_SPLITS[split]
                                member = f"pairs__{configuration}__{pair_kind}__{archive_split}.parquet"
                                endpoints = _read_parquet(
                                    archive,
                                    member,
                                    columns=["program_id_a", "program_id_b"],
                                )
                                capacities[(configuration, pair_kind, split)] = sum(
                                    str(left) in eligible_program_ids and str(right) in eligible_program_ids
                                    for left, right in zip(
                                        endpoints["program_id_a"].to_pylist(),
                                        endpoints["program_id_b"].to_pylist(),
                                    )
                                )
                    sample_targets = _capacity_aware_sample_targets(
                        sample_size,
                        configurations,
                        pair_kinds,
                        capacities,
                    )

            for configuration in configurations:
                for pair_kind in pair_kinds:
                    for split in SPLITS:
                        archive_split = ARCHIVE_SPLITS[split]
                        member = f"pairs__{configuration}__{pair_kind}__{archive_split}.parquet"
                        table = _read_parquet(archive, member)
                        source_rows = table.to_pylist()
                        input_count = len(source_rows)
                        if eligible_program_ids is not None:
                            source_rows = [
                                row
                                for row in source_rows
                                if all(endpoint in eligible_program_ids for endpoint in _pair_endpoints(row))
                            ]
                        target = (
                            len(source_rows)
                            if sample_targets is None
                            else sample_targets.get((configuration, pair_kind, split), 0)
                        )
                        if len(source_rows) < target:
                            raise RuntimeError(
                                f"CodeNet stratum {configuration}/{pair_kind}/{split} has "
                                f"{len(source_rows):,} eligible rows after the source-line filter; "
                                f"needs {target:,}."
                            )
                        chosen = (
                            source_rows
                            if target == len(source_rows)
                            else heapq.nsmallest(target, source_rows, key=lambda row: _sample_rank(row, sample_seed))
                        )
                        inserts = []
                        endpoint_rows = []
                        expected_label = 1 if pair_kind == "clone" else 0
                        for row in chosen:
                            row_configuration = str(row.get("configuration_id", ""))
                            row_pair_kind = str(row.get("pair_kind", ""))
                            row_split = str(row.get("split", ""))
                            label = int(row["label"])
                            if row_configuration != configuration or row_pair_kind != pair_kind:
                                raise RuntimeError(f"Pair metadata does not match archive member {member}")
                            if row_split != archive_split or label != expected_label:
                                raise RuntimeError(f"Split/label metadata does not match archive member {member}")
                            left, right = _pair_endpoints(row)
                            needed_languages.update((
                                normalize_source_language(row["language_a"]),
                                normalize_source_language(row["language_b"]),
                            ))
                            inserts.append((
                                split,
                                left,
                                right,
                                label,
                                json.dumps(_pair_provenance(row), separators=(",", ":"), default=str),
                            ))
                            endpoint_rows.extend(((left,), (right,)))
                            pair_counts[split]["pairs"] += 1
                            pair_counts[split]["clone" if label else "non_clone"] += 1
                            pair_kind_counts[split][pair_kind] += 1
                        connection.executemany("INSERT INTO pairs VALUES (?, ?, ?, ?, ?)", inserts)
                        connection.executemany("INSERT OR IGNORE INTO code_ids(source_id) VALUES (?)", endpoint_rows)
                        actual_sample_targets[(configuration, pair_kind, split)] = len(chosen)
                        bucket_counts[member] = {
                            "input": input_count,
                            "eligible_after_line_filter": len(source_rows),
                            "target": target,
                            "retained": len(chosen),
                        }
                        connection.commit()

            source_ids = [row[0] for row in connection.execute("SELECT source_id FROM code_ids ORDER BY source_id")]
            numeric_ids = {source_id: index for index, source_id in enumerate(source_ids, start=1)}
            connection.executemany(
                "UPDATE code_ids SET numeric_id=? WHERE source_id=?",
                ((numeric_id, source_id) for source_id, numeric_id in numeric_ids.items()),
            )
            connection.commit()

            language_paths = {language: output_dir / language for language in LANGUAGES}
            for path in language_paths.values():
                path.mkdir()
            language_files = {
                language: (path / "data.jsonl").open("w", encoding="utf-8", newline="\n")
                for language, path in language_paths.items()
            }
            found: set[str] = set()
            languages = Counter()
            try:
                with (output_dir / "data.jsonl").open("w", encoding="utf-8", newline="\n") as all_codes, (
                    output_dir / "code_id_map.csv"
                ).open("w", encoding="utf-8", newline="") as map_file:
                    mapping = csv.DictWriter(map_file, fieldnames=["code_id", "source_code_id", "language"])
                    mapping.writeheader()
                    program_members = [f"programs__{language}.parquet" for language in sorted(needed_languages)]
                    for member in program_members:
                        columns = [
                            "program_id", "submission_id", "parent_submission_id", "problem_id",
                            "language", "source_code", "source_sha256", "is_mutant", "source_origin",
                            "mutation_family", "mutation_operator", "mutation_engine",
                            "mutation_site_policy", "behavioral_validation_performed", "mutation_reference",
                        ]
                        table = _read_parquet(archive, member, columns=columns)
                        for batch in table.to_batches(max_chunksize=25_000):
                            for row in batch.to_pylist():
                                source_id = str(row["program_id"])
                                numeric_id = numeric_ids.get(source_id)
                                if numeric_id is None:
                                    continue
                                if source_id in found:
                                    raise RuntimeError(f"Duplicate CodeNet program ID: {source_id}")
                                language = normalize_source_language(row["language"])
                                if language not in LANGUAGES:
                                    raise RuntimeError(f"Unsupported CodeNet language: {language!r}")
                                record = {
                                    "idx": numeric_id,
                                    "func": str(row["source_code"]),
                                    "lang": language,
                                    "source_code_id": source_id,
                                    "problem_id": str(row["problem_id"]),
                                    "source_sha256": str(row["source_sha256"]),
                                    "is_mutant": bool(row["is_mutant"]),
                                    "source_line_count": _source_line_count(row["source_code"]),
                                }
                                if language in {"java", "csharp"}:
                                    record["source_mode"] = "compilation_unit"
                                if row["is_mutant"]:
                                    record.update({
                                        "status": "mutation_derived",
                                        "parent_submission_id": str(row["parent_submission_id"]),
                                        "mutation_family": row.get("mutation_family"),
                                        "mutation_operator": row.get("mutation_operator"),
                                        "mutation_engine": row.get("mutation_engine"),
                                        "mutation_site_policy": row.get("mutation_site_policy"),
                                        "behavioral_validation_performed": row.get("behavioral_validation_performed"),
                                        "mutation_reference": row.get("mutation_reference"),
                                    })
                                else:
                                    record["status"] = row.get("source_origin")
                                line = _json_line(record)
                                all_codes.write(line)
                                language_files[language].write(line)
                                mapping.writerow({"code_id": numeric_id, "source_code_id": source_id, "language": language})
                                found.add(source_id)
                                languages[language] += 1
            finally:
                for file in language_files.values():
                    file.close()

        missing = set(numeric_ids) - found
        if missing:
            raise RuntimeError(f"{len(missing):,} pair-referenced CodeNet programs are absent from program tables.")

        split_files = {
            split: (output_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n")
            for split in SPLITS
        }
        with gzip.open(output_dir / "pair_provenance.jsonl.gz", "wt", encoding="utf-8", newline="\n") as provenance_file:
            row_indexes = Counter()
            query = (
                "SELECT p.split, a.numeric_id, b.numeric_id, p.label, p.provenance "
                "FROM pairs p JOIN code_ids a ON a.source_id=p.left_source "
                "JOIN code_ids b ON b.source_id=p.right_source ORDER BY p.rowid"
            )
            for split, left, right, label, provenance in connection.execute(query):
                split_files[split].write(f"{left}\t{right}\t{label}\n")
                record = {"split": split, "row_index": row_indexes[split], **json.loads(provenance)}
                provenance_file.write(_json_line(record))
                row_indexes[split] += 1
        for file in split_files.values():
            file.close()

        retained_pair_count = sum(counter["pairs"] for counter in pair_counts.values())
        metadata = {
            "format": "codenet_4l_graph_pipeline_input_v1",
            "dataset": (
                "Project CodeNet 4L complete pre-split release"
                if sample_size is None
                else f"Project CodeNet 4L uniform {sample_size:,}-pair subset"
            ),
            "dataset_key": "codenet_4l",
            "archive_path": str(archive_path),
            "source_archive_sha256": archive_sha256,
            "archive_layout": "40_bucket_10k_presplit_v1",
            "split_policy": (
                "Preserved source release 70/15/15 pair-level random split. "
                "Pair IDs are disjoint, but program endpoints and problems may overlap across splits."
            ),
            "endpoint_disjoint": False,
            "problem_disjoint": False,
            "configurations": list(configurations),
            "pair_kinds": list(pair_kinds),
            "sampling_policy": (
                "Full source buckets without resampling"
                if sample_size is None
                else (
                    "Deterministic SHA-256 subset with equal configuration/pair-kind totals, global 70/15/15 "
                    "quotas, and minimum-deviation capacity-aware per-configuration split rebalancing"
                    if line_filter is not None
                    else "Deterministic SHA-256 subset with equal configuration/pair-kind buckets and 70/15/15 quotas"
                )
            ),
            "sampling_mode": "full" if sample_size is None else "uniform_subset",
            "requested_sample_size": sample_size,
            "sample_size": retained_pair_count,
            "sample_seed": sample_seed,
            "program_line_filter": line_filter,
            "eligible_programs_by_language_before_pair_sampling": (
                dict(sorted(eligible_programs_by_language.items())) if line_filter is not None else None
            ),
            "sample_targets": {
                f"{configuration}/{pair_kind}/{split}": count
                for (configuration, pair_kind, split), count in actual_sample_targets.items()
            },
            "source_id_mapping": "code_id_map.csv",
            "code_count": len(found),
            "codes_by_language": dict(sorted(languages.items())),
            "pairs": {
                split: {
                    **dict(pair_counts[split]),
                    "by_pair_kind": dict(pair_kind_counts[split]),
                }
                for split in SPLITS
            },
            "bucket_counts": bucket_counts,
            "dropped_cross_split_pairs": 0,
            "canonical_mapping": "Applied dynamically from the SPECTRA-Siam notebook; raw graph records remain preserved.",
            "graph_types_to_extract": GRAPH_TYPES,
            "pair_provenance": {
                "file": "pair_provenance.jsonl.gz",
                "fields": list(PAIR_PROVENANCE_FIELDS),
                "rows_by_split": dict(row_indexes),
            },
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        Path(str(database_path) + "-wal").unlink(missing_ok=True)
        Path(str(database_path) + "-shm").unlink(missing_ok=True)


def export_codenet_clean_dataset(
    prepared_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    **kwargs,
) -> dict:
    # The Kaggle scope notebooks require every selected pair to survive the
    # graph join.  A partially parsed endpoint must therefore stop publishing
    # instead of being tolerated by the generic one-percent safety margin.
    kwargs.setdefault("max_missing_feature_fraction", 0.0)
    return export_v3_clean_dataset(
        "codenet_4l",
        prepared_dir or default_prepared_dir(),
        output_root or output_root_for("codenet_4l"),
        **kwargs,
    )
