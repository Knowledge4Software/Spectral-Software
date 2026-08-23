r"""Build AST/CFG/DDG/CPG graphs for all CodeNet clone endpoints incrementally.

Run from the repository root:
  .\.venv\Scripts\python.exe create_datasets_graphs/codenet_4l/all_clones/run_pipeline.py

Rerun the exact same command after an interruption. Completed graph batches and
compatible records from the existing 12k package are never rebuilt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.codenet_clone_graphs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAPH_TYPES,
    CloneGraphPaths,
    _connect_cache,
    _read_prepared_targets,
    archive_sha256_from_prepared,
    build_missing_graph_batches,
    cache_audit,
    default_paths,
    default_reuse_source,
    format_audit,
    package_all_clone_graphs,
    prepare_all_clone_pairs,
    seed_cache_from_clean_bundle,
)
from spectral_code.evaluation.codenet_preparation import LANGUAGES


STAGES = ("prepare", "reuse", "graphs", "package")


def _stage_index(value: str) -> int:
    return STAGES.index(value)


def _csv_languages(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    invalid = sorted(set(values) - set(LANGUAGES))
    if not values or invalid:
        raise ValueError(f"languages must be chosen from {LANGUAGES}; invalid={invalid}")
    return values


def main() -> None:
    defaults = default_paths()
    reuse_default = default_reuse_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=defaults.archive)
    parser.add_argument("--prepared-dir", type=Path, default=defaults.prepared)
    parser.add_argument("--output-dir", type=Path, default=defaults.output)
    parser.add_argument("--reuse-from", type=Path, default=reuse_default)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--start-at", choices=STAGES, default="prepare")
    parser.add_argument("--stop-after", choices=STAGES, default="package")
    parser.add_argument("--max-batches", type=int, help="Debug/smoke limit; cache remains resumable")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--keep-batch-work", action="store_true")
    parser.add_argument(
        "--joern-python-csharp",
        action="store_true",
        help="Disable the fast source-graph path and run redundant Joern work for Python/C#.",
    )
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if _stage_index(args.start_at) > _stage_index(args.stop_after):
        parser.error("--start-at must not be after --stop-after")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        languages = _csv_languages(args.languages)
    except ValueError as exc:
        parser.error(str(exc))

    paths = CloneGraphPaths(args.archive.resolve(), args.prepared_dir.resolve(), args.output_dir.resolve())
    graph_types = DEFAULT_GRAPH_TYPES
    print("[*] CodeNet all-clone incremental graph build")
    print(f"    archive:  {paths.archive}")
    print(f"    prepared: {paths.prepared}")
    print(f"    output:   {paths.output}")
    print(f"    graphs:   {','.join(graph_types)}")
    print(f"    batch:    {args.batch_size:,} programs")

    if _stage_index(args.start_at) <= _stage_index("prepare") <= _stage_index(args.stop_after):
        metadata = prepare_all_clone_pairs(paths, overwrite=args.force_prepare)
        pair_count = sum(int(item["pairs"]) for item in metadata["pairs"].values())
        print(f"[+] Prepared {metadata['code_count']:,} unique endpoints from {pair_count:,} clone pairs.")
    if not (paths.prepared / "metadata.json").is_file():
        raise FileNotFoundError(f"Prepared clone dataset is missing: {paths.prepared}; run from stage prepare.")

    targets = _read_prepared_targets(paths.prepared)
    paths.output.mkdir(parents=True, exist_ok=True)
    connection = _connect_cache(paths, graph_types, archive_sha256_from_prepared(paths.prepared))
    try:
        if _stage_index(args.start_at) <= _stage_index("reuse") <= _stage_index(args.stop_after):
            if args.reuse_from is not None:
                source = args.reuse_from.resolve()
                reused = seed_cache_from_clean_bundle(paths, connection, targets, source, graph_types)
                print(f"[+] Reused {reused:,} prior graph records from {source}")
            else:
                print("[!] No prior clean CodeNet package found; graphing every endpoint.")
            print("[*] Cache audit:", format_audit(cache_audit(connection, targets)))

        if _stage_index(args.start_at) <= _stage_index("graphs") <= _stage_index(args.stop_after):
            report = build_missing_graph_batches(
                paths,
                connection,
                targets,
                graph_types,
                batch_size=args.batch_size,
                languages=languages,
                max_batches=args.max_batches,
                keep_batch_work=args.keep_batch_work,
                fast_source_graphs=not args.joern_python_csharp,
            )
            print("[+] Newly cached:", json.dumps(report, sort_keys=True))
            print("[*] Cache audit:", format_audit(cache_audit(connection, targets)))

        if _stage_index(args.start_at) <= _stage_index("package") <= _stage_index(args.stop_after):
            if set(languages) != set(LANGUAGES):
                raise RuntimeError("Packaging requires all four languages; rerun without --languages.")
            metadata = package_all_clone_graphs(
                paths, connection, targets, graph_types, create_zip=not args.no_zip
            )
            print(f"[+] Clean package: {paths.clean_dir}")
            if "zip" in metadata:
                print(f"[+] Kaggle ZIP: {metadata['zip']}")
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted. Rerun the same command; completed clone batches remain cached.")
        raise SystemExit(130)
