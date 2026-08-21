"""Incremental graph construction for every clone endpoint in CodeNet 4L.

The ordinary dataset runner processes one complete language at a time and uses
dataset-local numeric IDs.  This module adds two properties needed for the full
clone corpus:

* prior portable graph records are reused by stable ``source_code_id`` and
  verified with ``source_sha256``;
* missing programs are processed in durable batches, so a completed batch is
  never sent to Joern again after an interruption.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from spectral_code.evaluation.clean_data_export import (
    create_clean_data_zip,
    export_graph_spectra_from_sources,
    validate_clean_data_files,
    write_clean_pairs,
)
from spectral_code.evaluation.codenet_preparation import (
    CONFIGURATIONS,
    GRAPH_TYPES,
    LANGUAGES,
    default_archive_path,
    prepare_codenet_dataset,
)
from spectral_code.evaluation.v3_benchmark_preparation import (
    _coverage_by_graph_type,
    _read_pair_provenance,
    _read_splits,
    _write_clean_codes_with_metadata,
)
from spectral_code.preprocessing.language_support import joern_language
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for


CACHE_FORMAT = "codenet_clone_graph_cache_v1"
DEFAULT_GRAPH_TYPES = tuple(GRAPH_TYPES)
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_EIGENVALUE_LIMIT = max(1, int(os.getenv("PACKAGE_EIGENVALUE_LIMIT", "128")))


@dataclass(frozen=True)
class CloneGraphPaths:
    archive: Path
    prepared: Path
    output: Path
    shared_cache: Path | None = None
    zip_filename: str = "codenet_4l_all_clones_clean_data.zip"

    @property
    def cache_dir(self) -> Path:
        return self.shared_cache or self.output / "graph_record_cache"

    @property
    def cache_shards(self) -> Path:
        return self.cache_dir / "shards"

    @property
    def cache_index(self) -> Path:
        return self.cache_dir / "index.sqlite3"

    @property
    def work_dir(self) -> Path:
        return self.output / "_batch_work"

    @property
    def clean_dir(self) -> Path:
        return self.output / "clean_data"

    @property
    def zip_path(self) -> Path:
        return self.output / self.zip_filename


def default_paths() -> CloneGraphPaths:
    return CloneGraphPaths(
        archive=default_archive_path(),
        prepared=DATA_ROOT / "codenet_4l_all_clones_prepared",
        output=output_root_for("codenet_4l_all_clones"),
    )


def default_reuse_source() -> Path | None:
    candidates = (
        output_root_for("codenet_4l_nonclone_12k") / "clean_data",
        output_root_for("codenet_4l_nonclone_12k") / "codenet_4l_clean_data.zip",
        output_root_for("codenet_4l_nonclone_12k").parent / "kaggle_datasets" / "codenet_4l_clean_data.zip",
    )
    return next((path for path in candidates if path.exists()), None)


def prepare_clone_pairs(
    paths: CloneGraphPaths,
    *,
    sample_size: int | None,
    overwrite: bool = False,
) -> dict:
    """Prepare a uniform clone-pair subset, or every pair when size is None."""
    return prepare_codenet_dataset(
        paths.archive,
        paths.prepared,
        configurations=list(CONFIGURATIONS),
        pair_kinds=["clone"],
        sample_size=sample_size,
        overwrite=overwrite,
    )


def prepare_all_clone_pairs(paths: CloneGraphPaths, *, overwrite: bool = False) -> dict:
    """Prepare all 100k clone pairs and every unique endpoint, without sampling."""
    return prepare_clone_pairs(paths, sample_size=None, overwrite=overwrite)


def _read_prepared_targets(prepared: Path) -> dict[str, dict[str, object]]:
    targets: dict[str, dict[str, object]] = {}
    with (prepared / "data.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row["source_code_id"])
            if source_id in targets:
                raise RuntimeError(f"Duplicate prepared source_code_id: {source_id}")
            targets[source_id] = {
                "code_id": int(row["idx"]),
                "language": str(row["lang"]),
                "source_sha256": str(row.get("source_sha256", "")),
            }
    return targets


def _connect_cache(paths: CloneGraphPaths, graph_types: tuple[str, ...], archive_sha256: str) -> sqlite3.Connection:
    paths.cache_shards.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(paths.cache_index)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS records ("
        "source_code_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, shard TEXT NOT NULL)"
    )
    expected = {
        "format": CACHE_FORMAT,
        "archive_sha256": archive_sha256,
        "graph_types": json.dumps(list(graph_types), separators=(",", ":")),
    }
    actual = dict(connection.execute("SELECT key, value FROM meta"))
    mismatches = {key: (actual.get(key), value) for key, value in expected.items() if key in actual and actual[key] != value}
    if mismatches:
        connection.close()
        raise RuntimeError(
            "Existing clone graph cache is incompatible with this run: "
            f"{mismatches}. Use another --output-dir; do not delete validated cache accidentally."
        )
    connection.executemany("INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", expected.items())
    connection.commit()
    _recover_unindexed_shards(connection, paths.cache_shards, graph_types)
    return connection


def _valid_graph_record(record: dict, graph_types: tuple[str, ...]) -> bool:
    graphs = record.get("graphs")
    if not isinstance(graphs, dict):
        return False
    return all(
        isinstance(graphs.get(kind), dict)
        and isinstance(graphs[kind].get("adjacency"), dict)
        for kind in graph_types
    )


def _record_supports_spectral_limit(
    record: dict,
    graph_types: tuple[str, ...],
    limit: int = DEFAULT_EIGENVALUE_LIMIT,
) -> bool:
    """Return false only for successful sparse spectra built with a smaller K.

    Exact spectra may legitimately be shorter than the consumer window when a
    graph has fewer nodes. Failed/missing spectra are handled by the ordinary
    coverage audit and must not be confused with an obsolete sparse K.
    """
    graphs = record.get("graphs", {})
    for kind in graph_types:
        layer = graphs.get(kind, {})
        status = str(layer.get("spectral_status", layer.get("status", ""))).lower()
        if "sparse" not in status or "fail" in status:
            continue
        values = layer.get("eigenvalues")
        try:
            value_count = len(values) if values is not None else 0
        except TypeError:
            value_count = 0
        if value_count < limit:
            return False
    return True


def _recover_unindexed_shards(
    connection: sqlite3.Connection,
    shard_dir: Path,
    graph_types: tuple[str, ...],
) -> None:
    indexed = {str(row[0]) for row in connection.execute("SELECT DISTINCT shard FROM records")}
    for shard_path in sorted(shard_dir.glob("*.jsonl.gz")):
        relative = shard_path.name
        if relative in indexed:
            continue
        recovered: list[tuple[str, str, str]] = []
        with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if not _valid_graph_record(record, graph_types):
                    raise RuntimeError(f"Invalid cache record in orphan shard: {shard_path}")
                recovered.append((
                    str(record["source_code_id"]),
                    str(record.get("source_sha256", "")),
                    relative,
                ))
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO records(source_code_id, source_sha256, shard) VALUES (?, ?, ?)",
                recovered,
            )


def _cache_ids(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT source_code_id FROM records")}


@contextmanager
def _open_clean_bundle(path: Path):
    """Yield text streams for codes and graphs from a clean directory or ZIP."""
    if path.is_dir():
        codes = path / "codes.jsonl.gz"
        graphs = path / "graph_spectra.jsonl.gz"
        if not codes.is_file() or not graphs.is_file():
            raise FileNotFoundError(f"Reusable clean-data files are incomplete below {path}")
        with gzip.open(codes, "rt", encoding="utf-8") as code_stream, gzip.open(
            graphs, "rt", encoding="utf-8"
        ) as graph_stream:
            yield code_stream, graph_stream
        return

    if path.suffix.lower() != ".zip" or not path.is_file():
        raise FileNotFoundError(f"Reusable clean-data directory/ZIP not found: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        code_names = [name for name in names if name.endswith("/codes.jsonl.gz") or name == "codes.jsonl.gz"]
        graph_names = [name for name in names if name.endswith("/graph_spectra.jsonl.gz") or name == "graph_spectra.jsonl.gz"]
        if len(code_names) != 1 or len(graph_names) != 1:
            raise RuntimeError(f"Expected one codes and graph_spectra file in {path}; found {code_names}, {graph_names}")
        with archive.open(code_names[0]) as raw_codes, archive.open(graph_names[0]) as raw_graphs:
            with gzip.open(raw_codes, "rt", encoding="utf-8") as code_stream, gzip.open(
                raw_graphs, "rt", encoding="utf-8"
            ) as graph_stream:
                yield code_stream, graph_stream


def _reuse_source_signature(path: Path) -> str:
    parts = [str(path.resolve())]
    tracked = (
        (path / "codes.jsonl.gz", path / "graph_spectra.jsonl.gz")
        if path.is_dir()
        else (path,)
    )
    for item in tracked:
        stat = item.stat()
        parts.extend((str(stat.st_size), str(stat.st_mtime_ns)))
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _write_cache_shard(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    records: Iterable[dict],
    *,
    shard_name: str,
    graph_types: tuple[str, ...],
) -> int:
    final_path = paths.cache_shards / shard_name
    temporary = final_path.with_name(final_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    rows: list[tuple[str, str, str]] = []
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for record in records:
            if not _valid_graph_record(record, graph_types):
                raise RuntimeError(f"Incomplete graph layers while writing cache shard {shard_name}")
            source_id = str(record["source_code_id"])
            source_sha256 = str(record.get("source_sha256", ""))
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows.append((source_id, source_sha256, shard_name))
    if not rows:
        temporary.unlink(missing_ok=True)
        return 0
    temporary.replace(final_path)
    with connection:
        connection.executemany(
            "INSERT OR REPLACE INTO records(source_code_id, source_sha256, shard) VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def seed_cache_from_clean_bundle(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
    source: Path,
    graph_types: tuple[str, ...],
    *,
    shard_size: int = 5_000,
) -> int:
    """Copy compatible old records into the cache, remapped by source ID."""
    reuse_marker = f"reuse_complete_{_reuse_source_signature(source)}"
    if connection.execute("SELECT 1 FROM meta WHERE key=?", (reuse_marker,)).fetchone():
        return 0
    present = _cache_ids(connection)
    old_codes: dict[str, tuple[str, str]] = {}
    total = 0
    with _open_clean_bundle(source) as (code_stream, graph_stream):
        for line in code_stream:
            row = json.loads(line)
            source_id = str(row.get("source_code_id", ""))
            if not source_id or source_id not in targets or source_id in present:
                continue
            source_sha256 = str(row.get("source_sha256", ""))
            target_sha256 = str(targets[source_id].get("source_sha256", ""))
            if source_sha256 and target_sha256 and source_sha256 != target_sha256:
                continue
            old_codes[str(row["code_id"])] = (source_id, target_sha256 or source_sha256)

        shard_records: list[dict] = []

        def flush_records(records: list[dict]) -> int:
            digest = hashlib.sha256()
            for record in records:
                digest.update(str(record["source_code_id"]).encode("utf-8"))
                digest.update(b"\n")
            name = f"reused_{digest.hexdigest()[:16]}.jsonl.gz"
            return _write_cache_shard(
                paths, connection, records, shard_name=name, graph_types=graph_types
            )

        for line in graph_stream:
            row = json.loads(line)
            mapping = old_codes.get(str(row.get("code_id", "")))
            if (
                mapping is None
                or not _valid_graph_record(row, graph_types)
                or not _record_supports_spectral_limit(row, graph_types)
            ):
                continue
            source_id, source_sha256 = mapping
            shard_records.append({
                "source_code_id": source_id,
                "source_sha256": source_sha256,
                "graphs": {kind: row["graphs"][kind] for kind in graph_types},
            })
            if len(shard_records) >= shard_size:
                total += flush_records(shard_records)
                present.update(record["source_code_id"] for record in shard_records)
                shard_records = []
        if shard_records:
            total += flush_records(shard_records)
    with connection:
        connection.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, '1')", (reuse_marker,))
    return total


def merge_cache_from_cache(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
    source_cache_dir: Path,
    graph_types: tuple[str, ...],
    *,
    shard_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """Safely import target records from another compatible graph-record cache.

    This is intentionally record-level rather than a SQLite-file copy: two
    machines can build disjoint language subsets independently, then merge
    their validated shards without replacing either machine's cache index.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    source_cache_dir = source_cache_dir.resolve()
    source_index = source_cache_dir / "index.sqlite3"
    source_shards = source_cache_dir / "shards"
    if not source_index.is_file() or not source_shards.is_dir():
        raise FileNotFoundError(
            "Source graph cache must contain index.sqlite3 and shards: "
            f"{source_cache_dir}"
        )
    source = sqlite3.connect(f"file:{source_index.as_posix()}?mode=ro", uri=True)
    try:
        expected = dict(connection.execute("SELECT key, value FROM meta"))
        actual = dict(source.execute("SELECT key, value FROM meta"))
        required_keys = ("format", "archive_sha256", "graph_types")
        mismatches = {
            key: (actual.get(key), expected.get(key))
            for key in required_keys
            if actual.get(key) != expected.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "Source graph cache is incompatible with the destination cache: "
                f"{mismatches}"
            )
        destination_ids = _cache_ids(connection)
        wanted: dict[str, str] = {}
        by_shard: dict[str, set[str]] = {}
        for source_id, source_sha256, shard in source.execute(
            "SELECT source_code_id, source_sha256, shard FROM records"
        ):
            source_id = str(source_id)
            target = targets.get(source_id)
            if target is None or source_id in destination_ids:
                continue
            target_sha256 = str(target.get("source_sha256", ""))
            cached_sha256 = str(source_sha256)
            if target_sha256 and cached_sha256 and target_sha256 != cached_sha256:
                continue
            wanted[source_id] = cached_sha256
            by_shard.setdefault(str(shard), set()).add(source_id)

        imported = 0
        ordinal = 0
        buffer: list[dict] = []
        cache_tag = hashlib.sha256(str(source_cache_dir).encode("utf-8")).hexdigest()[:16]

        def flush() -> None:
            nonlocal imported, ordinal, buffer
            if not buffer:
                return
            while (paths.cache_shards / f"merged_{cache_tag}_{ordinal:04d}.jsonl.gz").exists():
                ordinal += 1
            written = _write_cache_shard(
                paths,
                connection,
                buffer,
                shard_name=f"merged_{cache_tag}_{ordinal:04d}.jsonl.gz",
                graph_types=graph_types,
            )
            imported += written
            ordinal += 1
            buffer = []

        for shard_name, source_ids in sorted(by_shard.items()):
            shard_path = source_shards / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Indexed source graph-cache shard is missing: {shard_path}")
            with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    source_id = str(record.get("source_code_id", ""))
                    if source_id not in source_ids or source_id not in wanted:
                        continue
                    if not _valid_graph_record(record, graph_types):
                        raise RuntimeError(f"Invalid graph record in source cache shard: {shard_path}")
                    if not _record_supports_spectral_limit(record, graph_types):
                        # Do not contaminate a 128-value cache with a completed
                        # sparse record produced by an earlier K=64 run. The
                        # target remains missing and is rebuilt normally.
                        wanted.pop(source_id, None)
                        continue
                    buffer.append(record)
                    if len(buffer) >= shard_size:
                        flush()
        flush()
        expected_import = len(wanted)
        if imported != expected_import:
            # The check below catches corrupt indices/shards before a partial
            # cache is allowed to masquerade as a completed import.
            imported_ids = _cache_ids(connection) - destination_ids
            missing = set(wanted) - imported_ids
            if missing:
                raise RuntimeError(
                    f"Imported {imported:,}/{expected_import:,} requested cache records; "
                    f"{len(missing):,} indexed source records were not found in their shards"
                )
        return imported
    finally:
        source.close()


def _safe_remove_batch_work(path: Path, work_root: Path) -> None:
    resolved = path.resolve()
    root = work_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside the clone batch-work directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _batch_key(language: str, records: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in records:
        digest.update(str(row["source_code_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("source_sha256", "")).encode("ascii"))
        digest.update(b"\n")
    return f"{language}_{digest.hexdigest()[:16]}"


def _run_pipeline_script(project_root: Path, script: str, env: dict[str, str]) -> None:
    command = [sys.executable, str(project_root / script)]
    print("[*]", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, env=env, check=True)


def _manifest_has_methods(path: Path, expected_methods: int) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        int(manifest.get("total_methods", -1)) == expected_methods
        and int(manifest.get("total_base_layers_cleaned", 0)) > 0
    )


def _spectral_manifest_complete(
    path: Path,
    expected_methods: int,
    graph_types: tuple[str, ...],
) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    requested_top_k = int(os.getenv("SPECTRAL_APPROX_TOPK", "128"))
    return (
        int(manifest.get("total_methods", -1)) == expected_methods
        and tuple(manifest.get("graph_types", ())) == graph_types
        and int(manifest.get("approx_top_k", 0)) >= requested_top_k
        and all(Path(shard).is_file() for shard in manifest.get("shards", ()))
    )


def _raw_stage_complete(output_dir: Path, expected_methods: int) -> bool:
    timing_path = output_dir / "timing_stats.json"
    raw_dir = output_dir / "dataset_features"
    if not timing_path.is_file() or not raw_dir.is_dir():
        return False
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    basic_complete = (
        int(timing.get("total_methods", -1)) == expected_methods
        and "total_raw_extraction_time" in timing
        and sum(1 for _ in raw_dir.glob("*.json")) == expected_methods
    )
    if not basic_complete:
        return False
    if bool(timing.get("source_fallback_only_pipeline01", False)):
        return True
    # A direct Java frontend can return a CPG but omit the overlays consumed by
    # joern-export. Such a stage has JSON placeholders for every method, yet no
    # usable base graph. Do not resume it as if Stage 01 had succeeded.
    return all(int(timing.get(f"dot_mapped_{kind}", 0)) > 0 for kind in ("ast", "cfg", "ddg"))


def _batch_records(prepared: Path, language: str, missing: set[str], batch_size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    with (prepared / language / "data.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row["source_code_id"]) not in missing:
                continue
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def recover_completed_batch_work(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    graph_types: tuple[str, ...],
    work_roots: Iterable[Path],
) -> dict[str, int]:
    """Adopt stopped batches whose cleaned graph manifest is already complete.

    Stage 03 is resumable by spectral shard. Work directories stopped before a
    complete graph manifest are intentionally ignored because their raw Joern
    output cannot be certified as a complete batch.
    """
    project_root = Path(__file__).resolve().parents[2]
    present = _cache_ids(connection)
    report = {"recovered": 0, "already_cached": 0, "incomplete_skipped": 0}
    for work_root in work_roots:
        if not work_root.is_dir():
            continue
        for batch_root in sorted(path for path in work_root.iterdir() if path.is_dir()):
            data_path = batch_root / "input" / "data.jsonl"
            output_dir = batch_root / "output"
            graph_manifest = output_dir / "clean_graphs" / "graph_shards_manifest.json"
            if not data_path.is_file():
                continue
            # ``str.splitlines`` treats Unicode U+2028/U+2029 characters inside
            # a source-code string as record boundaries. JSONL records are
            # delimited only by physical newlines, so iterate the text stream.
            with data_path.open("r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            uncached = [row for row in records if str(row["source_code_id"]) not in present]
            if not uncached:
                report["already_cached"] += len(records)
                continue
            if not _manifest_has_methods(graph_manifest, len(records)):
                report["incomplete_skipped"] += len(uncached)
                continue

            spectral_manifest = output_dir / "spectral_features" / "spectral_features_manifest.json"
            if not _spectral_manifest_complete(spectral_manifest, len(records), graph_types):
                env = os.environ.copy()
                env.update({
                    "PYTHONPATH": str(project_root) + os.pathsep + env.get("PYTHONPATH", ""),
                    "OUTPUT_DIR": str(output_dir),
                    "SPECTRAL_GRAPH_TYPES": ",".join(graph_types),
                    "SPECTRAL_FORCE_REBUILD": "0",
                })
                print(f"[*] Recovering spectral shards for stopped batch {batch_root.name}.")
                _run_pipeline_script(project_root, "pipelines/03_extract_spectral_features.py", env)

            ids = {int(row["idx"]) for row in uncached}
            source_by_id = {int(row["idx"]): row for row in uncached}
            exported = batch_root / "recovered_export.jsonl.gz"
            export_graph_spectra_from_sources(
                [output_dir], exported, ids, list(graph_types), precision=8
            )

            def recovered_records() -> Iterator[dict]:
                with gzip.open(exported, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        graph_record = json.loads(line)
                        row = source_by_id[int(graph_record["code_id"])]
                        yield {
                            "source_code_id": str(row["source_code_id"]),
                            "source_sha256": str(row.get("source_sha256", "")),
                            "graphs": graph_record["graphs"],
                        }

            written = _write_cache_shard(
                paths,
                connection,
                recovered_records(),
                shard_name=f"recovered_{batch_root.name}.jsonl.gz",
                graph_types=graph_types,
            )
            exported.unlink(missing_ok=True)
            report["recovered"] += written
            present.update(str(row["source_code_id"]) for row in uncached)
            print(f"[+] Recovered {written:,} records from stopped batch {batch_root.name}.")
    return report


def build_missing_graph_batches(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
    graph_types: tuple[str, ...],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    languages: tuple[str, ...] = LANGUAGES,
    max_batches: int | None = None,
    keep_batch_work: bool = False,
    fast_source_graphs: bool = True,
) -> dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    project_root = Path(__file__).resolve().parents[2]
    present = _cache_ids(connection)
    missing = set(targets) - present
    report = {language: 0 for language in languages}
    completed_batches = 0
    paths.work_dir.mkdir(parents=True, exist_ok=True)

    for language in languages:
        language_missing = {
            source_id for source_id in missing if targets[source_id]["language"] == language
        }
        if not language_missing:
            print(f"[=] {language}: all target programs are already cached.")
            continue
        print(f"[*] {language}: {len(language_missing):,} programs remain.")
        for records in _batch_records(paths.prepared, language, language_missing, batch_size):
            if max_batches is not None and completed_batches >= max_batches:
                return report
            key = _batch_key(language, records)
            batch_root = paths.work_dir / key
            input_dir = batch_root / "input"
            output_dir = batch_root / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            data_path = input_dir / "data.jsonl"
            with data_path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

            base_layers = [kind for kind in graph_types if kind != "cpg"]
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(project_root) + os.pathsep + env.get("PYTHONPATH", ""),
                "BCB_DATA_FILE": str(data_path),
                "BCB_DATA_DIR": str(input_dir),
                "OUTPUT_DIR": str(output_dir),
                "JOERN_LANGUAGE": joern_language(language),
                "JOERN_USE_DIRECT_FRONTEND": env.get(
                    "JOERN_USE_DIRECT_FRONTEND", "0"
                ),
                "PIPELINE_GRAPH_TYPES": ",".join(base_layers),
                "PIPELINE_BASE_LAYERS": ",".join(base_layers),
                "SPECTRAL_GRAPH_TYPES": ",".join(graph_types),
                "JOERN_PARSE_CHUNK_SIZE": env.get("JOERN_PARSE_CHUNK_SIZE", "500"),
                "JOERN_EXPORT_WORKERS": env.get("JOERN_EXPORT_WORKERS", "2"),
                "GRAPH_SHARD_SIZE": env.get("GRAPH_SHARD_SIZE", "500"),
                "SPECTRAL_APPROX_TOPK": env.get("SPECTRAL_APPROX_TOPK", "128"),
                "SPECTRAL_SPARSE_SOLVER": env.get("SPECTRAL_SPARSE_SOLVER", "shift_invert"),
                "SPECTRAL_SHARD_WORKERS": env.get("SPECTRAL_SHARD_WORKERS", "4"),
                "SPECTRAL_WORKERS": env.get("SPECTRAL_WORKERS", "1"),
                "SPECTRAL_BLAS_THREADS": env.get("SPECTRAL_BLAS_THREADS", "1"),
                "BCB_MAX_METHOD_LINES": "0",
                "BCB_MAX_METHOD_CHARS": "0",
                "BCB_MAX_LONGEST_LINE": "0",
                # Spectral extraction is sharded and can safely continue from
                # feature shards committed before Ctrl+C.
                "SPECTRAL_FORCE_REBUILD": "0",
                "PIPELINE_SOURCE_FALLBACK_ONLY": (
                    "1" if fast_source_graphs and language in {"python", "csharp"} else "0"
                ),
            })
            print(f"[*] Batch {key}: graphing {len(records):,} {language} programs.")
            graph_manifest = output_dir / "clean_graphs" / "graph_shards_manifest.json"
            spectral_manifest = output_dir / "spectral_features" / "spectral_features_manifest.json"
            graph_ready = _manifest_has_methods(graph_manifest, len(records))
            spectral_ready = _spectral_manifest_complete(
                spectral_manifest, len(records), graph_types
            )
            if graph_ready:
                print("[=] Reusing completed Stage 01+02 artifacts for interrupted batch.")
            else:
                if _raw_stage_complete(output_dir, len(records)):
                    print("[=] Reusing completed Stage 01 artifacts for interrupted batch.")
                else:
                    _run_pipeline_script(project_root, "pipelines/01_extract_dataset.py", env)
                _run_pipeline_script(project_root, "pipelines/02_build_graph_db.py", env)
            if spectral_ready:
                print("[=] Reusing completed Stage 03 artifacts for interrupted batch.")
            else:
                _run_pipeline_script(project_root, "pipelines/03_extract_spectral_features.py", env)

            ids = {int(row["idx"]) for row in records}
            source_by_id = {int(row["idx"]): row for row in records}
            exported = batch_root / "exported.jsonl.gz"
            export_graph_spectra_from_sources(
                [output_dir], exported, ids, list(graph_types), precision=8
            )

            def cache_records() -> Iterator[dict]:
                with gzip.open(exported, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        graph_record = json.loads(line)
                        row = source_by_id[int(graph_record["code_id"])]
                        yield {
                            "source_code_id": str(row["source_code_id"]),
                            "source_sha256": str(row.get("source_sha256", "")),
                            "graphs": graph_record["graphs"],
                        }

            written = _write_cache_shard(
                paths,
                connection,
                cache_records(),
                shard_name=f"built_{key}.jsonl.gz",
                graph_types=graph_types,
            )
            if written != len(records):
                raise RuntimeError(f"Batch {key} cached {written:,}/{len(records):,} graph records")
            report[language] += written
            completed_batches += 1
            missing.difference_update(str(row["source_code_id"]) for row in records)
            if not keep_batch_work:
                _safe_remove_batch_work(batch_root, paths.work_dir)
            print(f"[+] Batch {key} cached; {len(missing):,} total programs remain.")
    return report


def cache_audit(
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
) -> dict[str, object]:
    cached = {
        str(source_id): str(source_sha256)
        for source_id, source_sha256 in connection.execute(
            "SELECT source_code_id, source_sha256 FROM records"
        )
    }
    matching = {
        source_id
        for source_id, target in targets.items()
        if source_id in cached
        and (
            not str(target.get("source_sha256", ""))
            or not cached[source_id]
            or str(target["source_sha256"]) == cached[source_id]
        )
    }
    missing = set(targets) - matching
    by_language = {
        language: sum(targets[source_id]["language"] == language for source_id in missing)
        for language in LANGUAGES
    }
    return {
        "target_codes": len(targets),
        "cached_matching_codes": len(matching),
        "remaining_codes": len(missing),
        "remaining_by_language": by_language,
    }


def _iter_cached_target_records(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
    graph_types: tuple[str, ...],
) -> Iterator[dict]:
    shard_names = [str(row[0]) for row in connection.execute("SELECT DISTINCT shard FROM records ORDER BY shard")]
    written: set[str] = set()
    for shard_name in shard_names:
        shard_path = paths.cache_shards / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Indexed graph-cache shard is missing: {shard_path}")
        with gzip.open(shard_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                source_id = str(row["source_code_id"])
                target = targets.get(source_id)
                if target is None or source_id in written:
                    continue
                target_sha256 = str(target.get("source_sha256", ""))
                cached_sha256 = str(row.get("source_sha256", ""))
                if target_sha256 and cached_sha256 and target_sha256 != cached_sha256:
                    continue
                if not _valid_graph_record(row, graph_types):
                    continue
                if not _record_supports_spectral_limit(row, graph_types):
                    raise RuntimeError(
                        f"Cached record {source_id} contains a successful sparse spectrum "
                        f"shorter than {DEFAULT_EIGENVALUE_LIMIT}; rerun its graph batch with "
                        f"SPECTRAL_APPROX_TOPK={DEFAULT_EIGENVALUE_LIMIT}."
                    )
                written.add(source_id)
                compact_graphs = {}
                for kind in graph_types:
                    layer = row["graphs"][kind]
                    adjacency = layer.get("adjacency", {})
                    compact_adjacency = {
                        key: adjacency[key]
                        for key in (
                            "num_nodes",
                            "num_edges",
                            "node_ids",
                            "node_types",
                            "row",
                            "col",
                        )
                        if key in adjacency
                    }
                    if kind == "ast" and "node_labels" in adjacency:
                        compact_adjacency["node_labels"] = adjacency["node_labels"]
                    eigenvalues = layer.get("eigenvalues", [])
                    try:
                        eigenvalues = list(eigenvalues)[:DEFAULT_EIGENVALUE_LIMIT]
                    except TypeError:
                        eigenvalues = []
                    compact_graphs[kind] = {
                        "adjacency": compact_adjacency,
                        "eigenvalues": eigenvalues,
                        "spectral_status": layer.get("spectral_status", "missing"),
                    }
                yield {
                    "code_id": str(target["code_id"]),
                    "graphs": compact_graphs,
                }
    missing = set(targets) - written
    if missing:
        raise RuntimeError(f"Cannot package: {len(missing):,} clone endpoint graph records are still missing")


def package_codenet_pair_graphs(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
    graph_types: tuple[str, ...],
    *,
    dataset_name: str,
    dataset_key: str,
    readme_summary: str,
    create_zip: bool = True,
) -> dict:
    audit = cache_audit(connection, targets)
    if audit["remaining_codes"]:
        raise RuntimeError(
            f"Cannot package while {audit['remaining_codes']:,} endpoint graph records remain: "
            f"{audit['remaining_by_language']}"
        )
    paths.clean_dir.mkdir(parents=True, exist_ok=True)
    codes = _write_clean_codes_with_metadata(paths.clean_dir, paths.prepared)
    splits, needed_ids = _read_splits(paths.prepared)
    if needed_ids != set(codes):
        raise RuntimeError("Prepared pairs and code corpus are inconsistent")
    provenance = _read_pair_provenance(paths.prepared, splits)
    pair_counts = write_clean_pairs(paths.clean_dir, splits, pair_metadata=provenance)

    graph_path = paths.clean_dir / "graph_spectra.jsonl.gz"
    temporary = graph_path.with_name(graph_path.name + ".tmp")
    record_count = 0
    missing_features = 0
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for record in _iter_cached_target_records(paths, connection, targets, graph_types):
            for kind in graph_types:
                if not record["graphs"][kind].get("eigenvalues"):
                    missing_features += 1
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            record_count += 1
    if record_count != len(targets):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Packaged {record_count:,}/{len(targets):,} endpoint graphs")
    temporary.replace(graph_path)

    prepared_metadata = json.loads((paths.prepared / "metadata.json").read_text(encoding="utf-8"))
    pair_total = sum(len(rows) for rows in splits.values())
    clone_pairs = sum(label == 1 for rows in splits.values() for _, _, label in rows)
    non_clone_pairs = pair_total - clone_pairs
    coverage = _coverage_by_graph_type(graph_path, list(graph_types))
    metadata = {
        "format": "spectral_clean_data_v1",
        "dataset": dataset_name,
        "dataset_key": dataset_key,
        "prepared_data": str(paths.prepared),
        "graph_types": list(graph_types),
        "float_precision": 8,
        "incremental_cache": str(paths.cache_dir),
        "reuse_policy": "source_code_id plus source_sha256; numeric code IDs remapped at packaging",
        "consumer_schema": {
            "profile": "all_current_baselines_and_spectra_siam_v1",
            "retained_graph_types": list(graph_types),
            "ast_node_labels": True,
            "non_ast_node_labels": False,
            "precomputed_eigenvalues": True,
            "eigenvalue_limit_per_layer": DEFAULT_EIGENVALUE_LIMIT,
            "eigenvalue_limit_rationale": "SNN K_EIGEN=128; all spectral consumers remain compatible",
            "omitted_derived_fields": ["adjacency.format", "adjacency.directed", "eigenvalue_count"],
        },
        "counts": {
            "codes": len(codes),
            "pairs": {
                **pair_counts,
                "clone": clone_pairs,
                "non_clone": non_clone_pairs,
                "total": pair_total,
            },
            "graph_spectra": {
                "methods": record_count,
                "graph_layers": record_count * len(graph_types),
                "missing_feature_layers": missing_features,
            },
        },
        "spectral_coverage_by_graph_type": coverage,
        "source_preparation": prepared_metadata,
    }
    (paths.clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (paths.clean_dir / "README.md").write_text(
        f"# {dataset_name}\n\n"
        f"{readme_summary} "
        "Graph records contain AST, CFG, DDG, "
        "and composed CPG adjacency plus spectral eigenvalues. Previously built records were reused only after "
        "stable source-ID and source-hash validation.\n",
        encoding="utf-8",
    )
    validate_clean_data_files(paths.clean_dir)
    if create_zip:
        create_clean_data_zip(paths.clean_dir, paths.zip_path)
        metadata["zip"] = str(paths.zip_path)
        (paths.clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def package_all_clone_graphs(
    paths: CloneGraphPaths,
    connection: sqlite3.Connection,
    targets: dict[str, dict[str, object]],
    graph_types: tuple[str, ...],
    *,
    create_zip: bool = True,
) -> dict:
    """Package a clone-only selection, retaining the legacy public API."""
    prepared_metadata = json.loads((paths.prepared / "metadata.json").read_text(encoding="utf-8"))
    pair_total = sum(
        int(details.get("pairs", 0))
        for details in prepared_metadata.get("pairs", {}).values()
    )
    complete_release = prepared_metadata.get("sampling_mode") == "full"
    dataset_name = (
        "Project CodeNet 4L all clone pairs"
        if complete_release
        else f"Project CodeNet 4L uniform {pair_total:,}-clone-pair subset"
    )
    dataset_key = "codenet_4l_all_clones" if complete_release else f"codenet_4l_clone_{pair_total // 1000}k"
    return package_codenet_pair_graphs(
        paths,
        connection,
        targets,
        graph_types,
        dataset_name=dataset_name,
        dataset_key=dataset_key,
        readme_summary=f"All {pair_total:,} selected clone pairs and every unique endpoint are included.",
        create_zip=create_zip,
    )


def archive_sha256_from_prepared(prepared: Path) -> str:
    metadata_path = prepared / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared CodeNet metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = str(metadata.get("source_archive_sha256", ""))
    if not value:
        raise RuntimeError(f"Prepared metadata has no source archive hash: {metadata_path}")
    return value


def format_audit(audit: dict[str, object]) -> str:
    by_language = audit["remaining_by_language"]
    language_text = ", ".join(f"{language}={by_language[language]:,}" for language in LANGUAGES)
    return (
        f"target={audit['target_codes']:,} cached={audit['cached_matching_codes']:,} "
        f"remaining={audit['remaining_codes']:,} ({language_text})"
    )
