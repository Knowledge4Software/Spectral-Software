r"""Build the uniform 50k CodeNet clone subset with a shared graph cache.

The subset contains exactly 5,000 clone pairs from each of the ten language
configurations and preserves 70/15/15 train/valid/test quotas per configuration.
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
    default_reuse_source,
    format_audit,
    package_all_clone_graphs,
    prepare_clone_pairs,
    recover_completed_batch_work,
    seed_cache_from_clean_bundle,
)
from spectral_code.evaluation.codenet_preparation import (
    CONFIGURATIONS,
    LANGUAGES,
    default_archive_path,
)
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for


SAMPLE_SIZE = 50_000
STAGES = ("prepare", "reuse", "recover", "graphs", "package")


def _stage_index(value: str) -> int:
    return STAGES.index(value)


def _csv_languages(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    invalid = sorted(set(values) - set(LANGUAGES))
    if not values or invalid:
        raise ValueError(f"languages must be chosen from {LANGUAGES}; invalid={invalid}")
    return values


def _audit_uniform_pairs(metadata: dict) -> None:
    if int(metadata.get("sample_size", 0)) != SAMPLE_SIZE:
        raise RuntimeError(f"Prepared subset has {metadata.get('sample_size')} pairs; expected {SAMPLE_SIZE:,}")
    targets = metadata.get("sample_targets", {})
    per_configuration = {
        configuration: sum(
            int(targets.get(f"{configuration}/clone/{split}", 0))
            for split in ("train", "valid", "test")
        )
        for configuration in CONFIGURATIONS
    }
    invalid = {key: value for key, value in per_configuration.items() if value != 5_000}
    if invalid:
        raise RuntimeError(f"50k clone subset is not uniform by language configuration: {invalid}")
    expected_split = {"train": 3_500, "valid": 750, "test": 750}
    for configuration in CONFIGURATIONS:
        actual = {
            split: int(targets.get(f"{configuration}/clone/{split}", 0))
            for split in expected_split
        }
        if actual != expected_split:
            raise RuntimeError(f"Unexpected split quotas for {configuration}: {actual}")


def main() -> None:
    default_output = output_root_for("codenet_4l_clone_50k")
    shared_cache = output_root_for("codenet_4l_all_clones") / "graph_record_cache"
    old_work = output_root_for("codenet_4l_all_clones") / "_batch_work"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument("--prepared-dir", type=Path, default=DATA_ROOT / "codenet_4l_clone_50k_prepared")
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--shared-cache-dir", type=Path, default=shared_cache)
    parser.add_argument("--reuse-from", type=Path, default=default_reuse_source())
    parser.add_argument("--recover-work-from", type=Path, default=old_work)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--start-at", choices=STAGES, default="prepare")
    parser.add_argument("--stop-after", choices=STAGES, default="package")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--keep-batch-work", action="store_true")
    parser.add_argument("--joern-python-csharp", action="store_true")
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

    paths = CloneGraphPaths(
        args.archive.resolve(),
        args.prepared_dir.resolve(),
        args.output_dir.resolve(),
        args.shared_cache_dir.resolve(),
        "codenet_4l_clone_50k_clean_data.zip",
    )
    graph_types = DEFAULT_GRAPH_TYPES
    print("[*] CodeNet uniform 50k-clone graph build")
    print(f"    archive:      {paths.archive}")
    print(f"    prepared:     {paths.prepared}")
    print(f"    output:       {paths.output}")
    print(f"    shared cache: {paths.cache_dir}")
    print("    pairs:        50,000 = 5,000/configuration = 3,500/750/750 split")
    print(f"    graphs:       {','.join(graph_types)}")

    if _stage_index(args.start_at) <= _stage_index("prepare") <= _stage_index(args.stop_after):
        metadata = prepare_clone_pairs(paths, sample_size=SAMPLE_SIZE, overwrite=args.force_prepare)
        _audit_uniform_pairs(metadata)
        print(f"[+] Prepared {metadata['code_count']:,} unique endpoints from {SAMPLE_SIZE:,} clone pairs.")
    metadata_path = paths.prepared / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared 50k clone subset is missing: {paths.prepared}")
    prepared_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _audit_uniform_pairs(prepared_metadata)

    targets = _read_prepared_targets(paths.prepared)
    paths.output.mkdir(parents=True, exist_ok=True)
    connection = _connect_cache(paths, graph_types, archive_sha256_from_prepared(paths.prepared))
    try:
        if _stage_index(args.start_at) <= _stage_index("reuse") <= _stage_index(args.stop_after):
            if args.reuse_from is not None:
                reused = seed_cache_from_clean_bundle(
                    paths, connection, targets, args.reuse_from.resolve(), graph_types
                )
                print(f"[+] Imported {reused:,} additional records from {args.reuse_from.resolve()}")
            print("[*] Cache audit:", format_audit(cache_audit(connection, targets)))

        if _stage_index(args.start_at) <= _stage_index("recover") <= _stage_index(args.stop_after):
            report = recover_completed_batch_work(
                paths, connection, graph_types, [args.recover_work_from.resolve()]
            )
            print("[+] Stopped-work recovery:", json.dumps(report, sort_keys=True))
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
                raise RuntimeError("Packaging requires all four languages; rerun --start-at package without --languages.")
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
        print("\n[!] Interrupted. Rerun the same command; completed cache and stage shards remain reusable.")
        raise SystemExit(130)

