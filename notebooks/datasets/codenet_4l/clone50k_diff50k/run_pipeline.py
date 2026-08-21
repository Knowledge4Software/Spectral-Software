r"""Build graphs and spectra for the fixed CodeNet 50k clone + 50k non-clone release.

The clone half is deliberately identical to ``clone_50k`` and shares its
durable cache.  Only endpoints introduced by the 50k different-problem pairs
are graph-constructed when the clone build has completed.
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
    format_audit,
    merge_cache_from_cache,
    package_codenet_pair_graphs,
    recover_completed_batch_work,
    seed_cache_from_clean_bundle,
)
from spectral_code.evaluation.codenet_preparation import (
    CONFIGURATIONS,
    LANGUAGES,
    default_archive_path,
    prepare_codenet_dataset,
)
from spectral_code.utils.dataset_paths import DATA_ROOT, output_root_for


SAMPLE_SIZE = 100_000
PAIR_KINDS = ("clone", "nonclone_diff_problem")
STAGES = ("prepare", "reuse", "recover", "graphs", "package")


def _stage_index(value: str) -> int:
    return STAGES.index(value)


def _csv_languages(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    invalid = sorted(set(values) - set(LANGUAGES))
    if not values or invalid:
        raise ValueError(f"languages must be chosen from {LANGUAGES}; invalid={invalid}")
    return values


def _audit_fixed_pairs(metadata: dict) -> None:
    if int(metadata.get("sample_size", 0)) != SAMPLE_SIZE:
        raise RuntimeError(f"Prepared subset has {metadata.get('sample_size')} pairs; expected {SAMPLE_SIZE:,}")
    if tuple(metadata.get("pair_kinds", ())) != PAIR_KINDS:
        raise RuntimeError(f"Unexpected pair kinds: {metadata.get('pair_kinds')}")
    targets = metadata.get("sample_targets", {})
    expected_split = {"train": 3_500, "valid": 750, "test": 750}
    for configuration in CONFIGURATIONS:
        for pair_kind in PAIR_KINDS:
            actual = {
                split: int(targets.get(f"{configuration}/{pair_kind}/{split}", 0))
                for split in expected_split
            }
            if actual != expected_split:
                raise RuntimeError(
                    f"Unexpected quota for {configuration}/{pair_kind}: {actual}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument("--prepared-dir", type=Path, default=DATA_ROOT / "codenet_4l_clone50k_diff50k_prepared")
    parser.add_argument("--output-dir", type=Path, default=output_root_for("codenet_4l_clone50k_diff50k"))
    parser.add_argument(
        "--shared-cache-dir",
        type=Path,
        default=output_root_for("codenet_4l_all_clones") / "graph_record_cache",
    )
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument(
        "--import-cache-dir",
        type=Path,
        action="append",
        default=[],
        help="Import graph records from a compatible cache built on another machine; repeatable.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument(
        "--only-nonclone-endpoints",
        action="store_true",
        help=(
            "Build only endpoints introduced by the different-problem non-clone half. "
            "Use this for a separate partial cache while the clone cache is being built elsewhere."
        ),
    )
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
        "codenet_4l_clone50k_diff50k_clean_data.zip",
    )
    print("[*] CodeNet fixed 50k clone + 50k different-problem graph build")
    print(f"    archive:      {paths.archive}")
    print(f"    prepared:     {paths.prepared}")
    print(f"    output:       {paths.output}")
    print(f"    shared cache: {paths.cache_dir}")
    print("    pairs:        100,000 = 50,000 clone + 50,000 different-problem non-clone")
    print(f"    graphs:       {','.join(DEFAULT_GRAPH_TYPES)}")

    if _stage_index(args.start_at) <= _stage_index("prepare") <= _stage_index(args.stop_after):
        metadata = prepare_codenet_dataset(
            paths.archive,
            paths.prepared,
            configurations=list(CONFIGURATIONS),
            pair_kinds=list(PAIR_KINDS),
            sample_size=SAMPLE_SIZE,
            overwrite=args.force_prepare,
        )
        _audit_fixed_pairs(metadata)
        print(f"[+] Prepared {metadata['code_count']:,} unique endpoints from {SAMPLE_SIZE:,} pairs.")
    metadata_path = paths.prepared / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared 50k+50k subset is missing: {paths.prepared}")
    _audit_fixed_pairs(json.loads(metadata_path.read_text(encoding="utf-8")))

    all_targets = _read_prepared_targets(paths.prepared)
    targets = all_targets
    if args.only_nonclone_endpoints:
        clone_prepared = DATA_ROOT / "codenet_4l_clone_50k_prepared"
        clone_metadata = clone_prepared / "metadata.json"
        if not clone_metadata.is_file():
            raise FileNotFoundError(
                "The fixed clone prepared target is required to identify non-clone-only endpoints: "
                f"{clone_prepared}"
            )
        clone_targets = _read_prepared_targets(clone_prepared)
        targets = {
            source_id: target
            for source_id, target in all_targets.items()
            if source_id not in clone_targets
        }
        print(
            f"[*] Non-clone-only mode: {len(targets):,} endpoints "
            f"({len(all_targets) - len(targets):,} clone endpoints excluded)."
        )
    paths.output.mkdir(parents=True, exist_ok=True)
    connection = _connect_cache(paths, DEFAULT_GRAPH_TYPES, archive_sha256_from_prepared(paths.prepared))
    try:
        if _stage_index(args.start_at) <= _stage_index("reuse") <= _stage_index(args.stop_after):
            if args.reuse_from is not None:
                reused = seed_cache_from_clean_bundle(
                    paths, connection, targets, args.reuse_from.resolve(), DEFAULT_GRAPH_TYPES
                )
                print(f"[+] Imported {reused:,} additional records from {args.reuse_from.resolve()}")
            for source_cache in args.import_cache_dir:
                imported = merge_cache_from_cache(
                    paths, connection, targets, source_cache, DEFAULT_GRAPH_TYPES
                )
                print(f"[+] Imported {imported:,} graph records from cache {source_cache.resolve()}")
            print("[*] Cache audit:", format_audit(cache_audit(connection, targets)))

        if _stage_index(args.start_at) <= _stage_index("recover") <= _stage_index(args.stop_after):
            report = recover_completed_batch_work(paths, connection, DEFAULT_GRAPH_TYPES, [paths.work_dir])
            print("[+] Stopped-work recovery:", json.dumps(report, sort_keys=True))
            print("[*] Cache audit:", format_audit(cache_audit(connection, targets)))

        if _stage_index(args.start_at) <= _stage_index("graphs") <= _stage_index(args.stop_after):
            report = build_missing_graph_batches(
                paths, connection, targets, DEFAULT_GRAPH_TYPES,
                batch_size=args.batch_size, languages=languages, max_batches=args.max_batches,
                keep_batch_work=args.keep_batch_work,
                fast_source_graphs=not args.joern_python_csharp,
            )
            print("[+] Newly cached:", json.dumps(report, sort_keys=True))
            print("[*] Cache audit:", format_audit(cache_audit(connection, targets)))

        if _stage_index(args.start_at) <= _stage_index("package") <= _stage_index(args.stop_after):
            if args.only_nonclone_endpoints:
                raise RuntimeError(
                    "--only-nonclone-endpoints produces a partial cache, not a publishable package. "
                    "Merge its shard files into the completed shared clone cache, then run this runner normally."
                )
            if set(languages) != set(LANGUAGES):
                raise RuntimeError("Packaging requires all four languages; rerun --start-at package without --languages.")
            metadata = package_codenet_pair_graphs(
                paths, connection, targets, DEFAULT_GRAPH_TYPES,
                dataset_name="Project CodeNet 4L fixed 50k clone + 50k different-problem non-clone subset",
                dataset_key="codenet_4l_clone50k_diff50k",
                readme_summary="All 100,000 selected pairs and every unique endpoint are included.",
                create_zip=not args.no_zip,
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
