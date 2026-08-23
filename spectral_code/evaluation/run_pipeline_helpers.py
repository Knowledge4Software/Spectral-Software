from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

from spectral_code.evaluation.bcb_preparation import main as bcb_preparation_main
from spectral_code.utils.dataset_paths import bcb_type_dir, output_root_for
from spectral_code.utils.pipeline_timings import record_pipeline_timing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TYPE3_COMPONENT_VARIANTS = ("3/moderate", "3/strong", "3/very_strong")


def bcb_positive_only_defaults(
    *,
    type3_min_similarity: float | None = None,
    type3_max_similarity: float | None = None,
) -> list[str]:
    """Return the shared BCB positive-only extraction settings."""
    defaults = [
        "--target-pairs", "1000000",
        "--positive-fraction", "1.0",
        "--max-positive-pairs", "1000000",
    ]
    if type3_min_similarity is not None:
        defaults.extend(["--type3-min-similarity", f"{type3_min_similarity:.2f}"])
    if type3_max_similarity is not None:
        defaults.extend(["--type3-max-similarity", f"{type3_max_similarity:.2f}"])
    defaults.extend(["--preselect-positive-pairs-before-code-scan", "--positive-only"])
    return defaults


def bcb_non_clone_defaults() -> list[str]:
    """Return the shared curated false-positive non-clone extraction settings."""
    return [
        "--max-non-clone-code-ids", os.getenv("BCB_NON_CLONE_MAX_CODE_IDS", "0"),
        "--negative-pool", "false-positives",
        "--non-clone-only",
        "--keep-getter-setters",
        "--drop-non-clone-pairs-both-three-line",
        "--write-all-filtered-non-clone-code-ids",
        "--all-filtered-non-clone-pairs",
    ]


def with_default_args(argv: list[str], defaults: list[str]) -> list[str]:
    args = list(argv)
    i = 0
    while i < len(defaults):
        flag = defaults[i]
        has_value = i + 1 < len(defaults) and not defaults[i + 1].startswith("--")
        if flag not in args:
            if has_value:
                args = [flag, defaults[i + 1], *args]
            else:
                args = [flag, *args]
        i += 2 if has_value else 1
    return args


def run_bcb_data_extraction(
    clone_type: str,
    argv: list[str],
    defaults: list[str] | None = None,
    variant: str | None = None,
) -> None:
    data_variant = variant or clone_type
    base_defaults = [
        "--clone-type", clone_type,
        "--output-dir", str(bcb_type_dir(data_variant)),
    ]
    sys.argv = [sys.argv[0], *with_default_args(argv, [*(defaults or []), *base_defaults])]
    previous_variant = os.environ.get("BCB_OUTPUT_VARIANT")
    os.environ["BCB_OUTPUT_VARIANT"] = data_variant
    try:
        bcb_preparation_main()
    finally:
        if previous_variant is None:
            os.environ.pop("BCB_OUTPUT_VARIANT", None)
        else:
            os.environ["BCB_OUTPUT_VARIANT"] = previous_variant


def _run_python_script(relative_path: str, env_overrides: dict[str, str]) -> float:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else os.pathsep.join([str(PROJECT_ROOT), existing_pythonpath])
    )
    env.update(env_overrides)

    command = [sys.executable, str(PROJECT_ROOT / relative_path)]
    print(f"\n[*] Running {relative_path}")
    start = time.perf_counter()
    subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=True)
    return time.perf_counter() - start


def _count_file_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _load_data_jsonl_ids(data_path: Path) -> set[int]:
    ids = set()
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.add(int(json.loads(line)["idx"]))
    return ids


def _resolve_manifest_path(manifest_path: Path, shard_path: str) -> Path:
    path = Path(shard_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _load_usable_graph_method_ids(manifest_path: Path, base_layers: list[str]) -> set[int]:
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    usable_ids = set()
    for shard in manifest.get("shards", []):
        shard_path = _resolve_manifest_path(manifest_path, shard)
        with shard_path.open("rb") as f:
            graph_db = pickle.load(f)
        for method_id, layers in graph_db.items():
            for graph_type in base_layers:
                graph = layers.get(graph_type)
                if graph is not None and graph.number_of_nodes() > 0:
                    usable_ids.add(int(method_id))
                    break
    return usable_ids


def _safe_remove_child(path: Path, parent: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise ValueError(f"Refusing to remove path outside expected parent: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def cleanup_legacy_bcb_type3_all_artifacts() -> dict[str, object]:
    data_root = bcb_type_dir("3")
    output_root = output_root_for("bcb", "3")
    removed: list[str] = []

    for name in [
        "data.jsonl",
        "train.txt",
        "metadata.json",
        "graph_prune_metadata.json",
        "train_positives.txt",
        "type_labels.tsv",
    ]:
        target = data_root / name
        if target.exists():
            _safe_remove_child(target, data_root)
            removed.append(str(target))

    legacy_output_names = [
        "clean_graphs",
        "cpg",
        "dataset_features",
        "dot",
        "java_files",
        "models",
        "reports",
        "spectral_features",
        "timing_stats.json",
        "pipeline_timings.json",
        "skipped_methods_pipeline01.jsonl",
        "skipped_graph_parse_pipeline01.jsonl",
        "trained_bcb_type3_f1_pss_wasserstein.json",
        "pair_scores_trained_bcb_type3_f1_pss_wasserstein.csv",
    ]
    legacy_output_names.extend(path.name for path in output_root.glob("batch_cpg_*.parse.log"))
    for name in legacy_output_names:
        target = output_root / name
        if target.exists():
            _safe_remove_child(target, output_root)
            removed.append(str(target))

    return {
        "legacy_data_root": str(data_root),
        "legacy_output_root": str(output_root),
        "removed": removed,
        "removed_count": len(removed),
        "preserved_variant_dirs": list(TYPE3_COMPONENT_VARIANTS),
    }


def _require_files(root: Path, names: list[str]) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {root}: {', '.join(missing)}")


def _merge_bcb_data_dirs(source_variants: tuple[str, ...], destination_variant: str) -> dict[str, object]:
    destination = bcb_type_dir(destination_variant)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    seen_code_ids: set[int] = set()
    seen_pairs: set[tuple[int, int, int]] = set()
    source_summaries = []
    duplicate_code_records = 0
    duplicate_pairs = 0

    data_out = destination / "data.jsonl"
    train_out = destination / "train.txt"
    with data_out.open("w", encoding="utf-8") as data_dst, train_out.open("w", encoding="utf-8") as train_dst:
        for variant in source_variants:
            source = bcb_type_dir(variant)
            _require_files(source, ["data.jsonl", "train.txt"])
            source_code_records = 0
            source_pairs = 0
            source_unique_code_records = 0
            source_unique_pairs = 0

            with (source / "data.jsonl").open("r", encoding="utf-8") as data_src:
                for line in data_src:
                    if not line.strip():
                        continue
                    source_code_records += 1
                    record = json.loads(line)
                    method_id = int(record["idx"])
                    if method_id in seen_code_ids:
                        duplicate_code_records += 1
                        continue
                    seen_code_ids.add(method_id)
                    source_unique_code_records += 1
                    data_dst.write(json.dumps(record, ensure_ascii=False) + "\n")

            with (source / "train.txt").open("r", encoding="utf-8") as train_src:
                for line in train_src:
                    if not line.strip():
                        continue
                    left, right, label = line.rstrip("\n").split("\t")[:3]
                    left_id = int(left)
                    right_id = int(right)
                    label_id = int(label)
                    pair_key = (*sorted((left_id, right_id)), label_id)
                    source_pairs += 1
                    if pair_key in seen_pairs:
                        duplicate_pairs += 1
                        continue
                    seen_pairs.add(pair_key)
                    source_unique_pairs += 1
                    train_dst.write(f"{left_id}\t{right_id}\t{label_id}\n")

            source_summaries.append(
                {
                    "variant": variant,
                    "data_dir": str(source),
                    "code_records": source_code_records,
                    "unique_code_records_added": source_unique_code_records,
                    "pairs": source_pairs,
                    "unique_pairs_added": source_unique_pairs,
                }
            )

    metadata = {
        "dataset": "bigclonebench",
        "clone_type": "3",
        "variant": destination_variant,
        "merge_source_variants": list(source_variants),
        "source_summaries": source_summaries,
        "code_records": len(seen_code_ids),
        "pairs": len(seen_pairs),
        "duplicate_code_records_skipped": duplicate_code_records,
        "duplicate_pairs_skipped": duplicate_pairs,
        "labels": {"positive": len(seen_pairs), "negative": 0},
        "positive_only": True,
        "generated_by": "merge_bcb_type3_all_artifacts",
    }
    with (destination / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {"destination": str(destination), **metadata}


def _read_manifest(path: Path, expected_format: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != expected_format:
        raise ValueError(f"Unsupported manifest format in {path}: {manifest.get('format')}")
    return manifest


def _write_pickle_shards(
    merged: dict,
    shard_dir: Path,
    *,
    prefix: str,
    shard_size: int,
) -> list[str]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    items = list(merged.items())
    for shard_index, start in enumerate(range(0, len(items), shard_size), start=1):
        shard = dict(items[start:start + shard_size])
        shard_path = shard_dir / f"{prefix}_{shard_index:06d}.pkl"
        with shard_path.open("wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_paths.append(str(shard_path))
    return shard_paths


def _merge_graph_shards(source_variants: tuple[str, ...], destination_variant: str, shard_size: int = 1000) -> dict[str, object]:
    destination_root = output_root_for("bcb", destination_variant)
    clean_root = destination_root / "clean_graphs"
    if clean_root.exists():
        shutil.rmtree(clean_root)
    shard_dir = clean_root / "cleaned_graphs_shards"

    merged = {}
    duplicate_methods = 0
    source_summaries = []
    for variant in source_variants:
        manifest_path = output_root_for("bcb", variant) / "clean_graphs" / "graph_shards_manifest.json"
        manifest = _read_manifest(manifest_path, "cleaned_graph_shards_v1")
        source_methods = 0
        source_added = 0
        for shard in manifest.get("shards", []):
            shard_path = _resolve_manifest_path(manifest_path, shard)
            with shard_path.open("rb") as f:
                graph_db = pickle.load(f)
            for method_id, graphs in graph_db.items():
                source_methods += 1
                key = str(method_id)
                if key in merged:
                    duplicate_methods += 1
                    continue
                merged[key] = graphs
                source_added += 1
        source_summaries.append(
            {
                "variant": variant,
                "manifest": str(manifest_path),
                "methods": source_methods,
                "unique_methods_added": source_added,
            }
        )

    shard_paths = _write_pickle_shards(merged, shard_dir, prefix="graphs", shard_size=shard_size)
    base_layers = ["ast", "cfg", "ddg", "pdg"]
    total_base_layers = 0
    for graphs in merged.values():
        total_base_layers += sum(1 for graph_type in base_layers if graphs.get(graph_type) is not None)

    manifest = {
        "format": "cleaned_graph_shards_v1",
        "shard_size": shard_size,
        "total_methods": len(merged),
        "total_base_layers_cleaned": total_base_layers,
        "shards": shard_paths,
        "merged_from": source_summaries,
        "duplicate_methods_skipped": duplicate_methods,
    }
    clean_root.mkdir(parents=True, exist_ok=True)
    manifest_path = clean_root / "graph_shards_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"manifest": str(manifest_path), **manifest}


def _merge_spectral_shards(source_variants: tuple[str, ...], destination_variant: str, shard_size: int = 1000) -> dict[str, object]:
    destination_root = output_root_for("bcb", destination_variant)
    spectral_root = destination_root / "spectral_features"
    if spectral_root.exists():
        shutil.rmtree(spectral_root)
    shard_dir = spectral_root / "spectral_feature_shards"

    merged = {}
    duplicate_methods = 0
    source_summaries = []
    graph_types: list[str] = []
    mode = None
    dense_max_nodes = None
    approx_top_k = None
    for variant in source_variants:
        manifest_path = output_root_for("bcb", variant) / "spectral_features" / "spectral_features_manifest.json"
        manifest = _read_manifest(manifest_path, "spectral_feature_shards_v1")
        mode = mode or manifest.get("mode")
        dense_max_nodes = dense_max_nodes if dense_max_nodes is not None else manifest.get("dense_max_nodes")
        approx_top_k = approx_top_k if approx_top_k is not None else manifest.get("approx_top_k")
        for graph_type in manifest.get("graph_types", []):
            if graph_type not in graph_types:
                graph_types.append(graph_type)

        source_methods = 0
        source_added = 0
        for shard in manifest.get("shards", []):
            shard_path = _resolve_manifest_path(manifest_path, shard)
            with shard_path.open("rb") as f:
                features_db = pickle.load(f)
            for method_id, features in features_db.items():
                source_methods += 1
                key = str(method_id)
                if key in merged:
                    duplicate_methods += 1
                    continue
                merged[key] = features
                source_added += 1
        source_summaries.append(
            {
                "variant": variant,
                "manifest": str(manifest_path),
                "methods": source_methods,
                "unique_methods_added": source_added,
            }
        )

    shard_paths = _write_pickle_shards(merged, shard_dir, prefix="features", shard_size=shard_size)
    manifest = {
        "format": "spectral_feature_shards_v1",
        "mode": mode or "directed_laplacian",
        "total_methods": len(merged),
        "graph_types": graph_types,
        "shards": shard_paths,
        "dense_max_nodes": dense_max_nodes,
        "approx_top_k": approx_top_k,
        "merged_from": source_summaries,
        "duplicate_methods_skipped": duplicate_methods,
    }
    spectral_root.mkdir(parents=True, exist_ok=True)
    manifest_path = spectral_root / "spectral_features_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"manifest": str(manifest_path), **manifest}


def merge_bcb_type3_all_artifacts() -> dict[str, object]:
    destination_variant = "3/all"
    destination_root = output_root_for("bcb", destination_variant)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    cleanup = cleanup_legacy_bcb_type3_all_artifacts()
    data_summary = _merge_bcb_data_dirs(TYPE3_COMPONENT_VARIANTS, destination_variant)
    graph_summary = _merge_graph_shards(TYPE3_COMPONENT_VARIANTS, destination_variant)
    spectral_summary = _merge_spectral_shards(TYPE3_COMPONENT_VARIANTS, destination_variant)
    seconds = time.perf_counter() - start

    summary = {
        "dataset": "bcb",
        "clone_type": "3",
        "variant": destination_variant,
        "source_variants": list(TYPE3_COMPONENT_VARIANTS),
        "data": data_summary,
        "graphs": graph_summary,
        "spectral": spectral_summary,
        "legacy_cleanup": cleanup,
        "seconds": seconds,
    }
    reports_dir = destination_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "type3_all_merge_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    record_pipeline_timing(
        destination_root / "pipeline_timings.json",
        "01_merge_type3_all",
        seconds,
        {
            "dataset": "bcb",
            "clone_type": "3",
            "variant": destination_variant,
            "summary": str(summary_path),
        },
    )

    print("[+] Type-3 All merge complete.")
    print(f"    Data: {data_summary['destination']}")
    print(f"    Graph manifest: {graph_summary['manifest']}")
    print(f"    Spectral manifest: {spectral_summary['manifest']}")
    print(f"    Summary: {summary_path}")
    return summary


def _prune_pair_file_to_valid_graph_ids(path: Path, valid_ids: set[int]) -> dict[str, int | str | None]:
    stats: dict[str, int | str | None] = {
        "path": str(path),
        "pairs_before": 0,
        "pairs_after": 0,
        "pairs_removed_by_missing_graph": 0,
    }
    if not path.exists():
        stats["path"] = None
        return stats

    temp_path = path.with_name(f"{path.name}.graph_prune.tmp")
    with path.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            stats["pairs_before"] = int(stats["pairs_before"]) + 1
            left_id = int(parts[0])
            right_id = int(parts[1])
            if left_id not in valid_ids or right_id not in valid_ids:
                stats["pairs_removed_by_missing_graph"] = int(stats["pairs_removed_by_missing_graph"]) + 1
                continue
            dst.write(line)
            stats["pairs_after"] = int(stats["pairs_after"]) + 1

    temp_path.replace(path)
    return stats


def prune_bcb_pairs_to_graph_coverage(
    data_dir: Path,
    output_root: Path,
    base_layers: list[str],
) -> dict[str, object]:
    data_path = data_dir / "data.jsonl"
    manifest_path = output_root / "clean_graphs" / "graph_shards_manifest.json"
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Cannot prune graph-invalid pairs; missing {data_path} or {manifest_path}."
        )

    all_data_ids = _load_data_jsonl_ids(data_path)
    usable_graph_ids = _load_usable_graph_method_ids(manifest_path, base_layers)
    skipped_ids = all_data_ids - usable_graph_ids

    file_stats = {
        "train": _prune_pair_file_to_valid_graph_ids(data_dir / "train.txt", usable_graph_ids),
        "train_positives": _prune_pair_file_to_valid_graph_ids(data_dir / "train_positives.txt", usable_graph_ids),
        "type_labels": _prune_pair_file_to_valid_graph_ids(data_dir / "type_labels.tsv", usable_graph_ids),
    }

    metadata = {
        "data_dir": str(data_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "graph_manifest": str(manifest_path.resolve()),
        "base_layers": base_layers,
        "data_jsonl_function_ids": len(all_data_ids),
        "usable_graph_function_ids": len(usable_graph_ids),
        "skipped_graph_function_ids": len(skipped_ids),
        "skipped_methods_before_joern": _count_file_lines(output_root / "skipped_methods_pipeline01.jsonl"),
        "skipped_or_empty_graph_layers": _count_file_lines(output_root / "skipped_graph_parse_pipeline01.jsonl"),
        "files": file_stats,
    }

    metadata_path = data_dir / "graph_prune_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    bcb_metadata_path = data_dir / "metadata.json"
    if bcb_metadata_path.exists():
        with bcb_metadata_path.open("r", encoding="utf-8") as f:
            bcb_metadata = json.load(f)
        bcb_metadata["graph_prune_metadata"] = str(metadata_path.resolve())
        bcb_metadata["graph_pruned_pairs"] = metadata["files"]["train"]
        bcb_metadata["skipped_graph_function_ids"] = metadata["skipped_graph_function_ids"]
        with bcb_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(bcb_metadata, f, indent=2)

    print("[*] Graph coverage pair pruning:")
    print(f"    usable graph functions: {len(usable_graph_ids):,}/{len(all_data_ids):,}")
    print(f"    skipped graph functions: {len(skipped_ids):,}")
    print(
        "    train pairs: "
        f"{file_stats['train']['pairs_before']:,} -> {file_stats['train']['pairs_after']:,} "
        f"(removed {file_stats['train']['pairs_removed_by_missing_graph']:,})"
    )
    print(f"    metadata: {metadata_path}")
    return metadata


def run_bcb_graph_extraction(
    clone_type: str,
    base_layers: list[str] | None = None,
    variant: str | None = None,
) -> None:
    data_variant = variant or clone_type
    data_dir = bcb_type_dir(data_variant)
    output_root = output_root_for("bcb", data_variant)
    base_layers = base_layers or ["ast", "cfg", "ddg", "pdg"]
    max_method_lines_default = "1000" if str(data_variant).lower().replace("-", "_") == "non_clone" else "2000"

    if not (data_dir / "data.jsonl").exists() or not (data_dir / "train.txt").exists():
        raise FileNotFoundError(f"Prepared Type-{clone_type} data is missing in {data_dir}. Run 01_extract_data.py first.")

    env_overrides = {
        "BCB_CLONE_TYPE": clone_type,
        "BCB_DATA_FILE": str(data_dir / "data.jsonl"),
        "BCB_DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(output_root),
        "JOERN_LANGUAGE": "javasrc",
        "JOERN_USE_DIRECT_FRONTEND": os.getenv("JOERN_USE_DIRECT_FRONTEND", "0"),
        "BCB_MAX_METHOD_LINES": os.getenv("BCB_MAX_METHOD_LINES", max_method_lines_default),
        "JOERN_PARSE_CHUNK_SIZE": os.getenv("JOERN_PARSE_CHUNK_SIZE", "1000"),
        "JOERN_PARSE_MIN_CHUNK_SIZE": os.getenv("JOERN_PARSE_MIN_CHUNK_SIZE", "10"),
        "JOERN_PARSE_CHUNK_TIMEOUT_SECONDS": os.getenv("JOERN_PARSE_CHUNK_TIMEOUT_SECONDS", "300"),
        "JOERN_PARSE_INACTIVITY_TIMEOUT_SECONDS": os.getenv("JOERN_PARSE_INACTIVITY_TIMEOUT_SECONDS", "0"),
        "PIPELINE_GRAPH_TYPES": ",".join(base_layers),
        "PIPELINE_BASE_LAYERS": ",".join(base_layers),
    }

    print(f"[*] Prepared data: {data_dir}")
    print(f"[*] Output root: {output_root}")
    total_start = time.perf_counter()
    raw_seconds = _run_python_script("pipelines/01_extract_dataset.py", env_overrides)
    clean_seconds = _run_python_script("pipelines/02_build_graph_db.py", env_overrides)
    graph_prune_metadata = prune_bcb_pairs_to_graph_coverage(data_dir, output_root, base_layers)
    total_seconds = time.perf_counter() - total_start

    timings_path = output_root / "pipeline_timings.json"
    common = {"dataset": "bcb", "clone_type": clone_type, "variant": data_variant, "output_root": str(output_root)}
    record_pipeline_timing(
        timings_path,
        "02_extract_raw_graphs",
        raw_seconds,
        {**common, "script": "pipelines/01_extract_dataset.py"},
    )
    record_pipeline_timing(
        timings_path,
        "02_build_graph_db",
        clean_seconds,
        {**common, "script": "pipelines/02_build_graph_db.py"},
    )
    record_pipeline_timing(
        timings_path,
        "02_extract_graphs",
        total_seconds,
        {
            **common,
            "raw_graph_seconds": raw_seconds,
            "clean_graph_db_seconds": clean_seconds,
            "graph_prune_metadata": str((data_dir / "graph_prune_metadata.json").resolve()),
            "skipped_graph_function_ids": graph_prune_metadata["skipped_graph_function_ids"],
            "train_pairs_after_graph_prune": graph_prune_metadata["files"]["train"]["pairs_after"],
            "train_pairs_removed_by_graph_prune": graph_prune_metadata["files"]["train"]["pairs_removed_by_missing_graph"],
        },
    )


def run_bcb_spectral_feature_extraction(
    clone_type: str,
    graph_types: list[str] | None = None,
    variant: str | None = None,
) -> None:
    data_variant = variant or clone_type
    data_dir = bcb_type_dir(data_variant)
    output_root = output_root_for("bcb", data_variant)
    graph_types = graph_types or ["ast", "cfg", "ddg", "pdg", "cpg"]
    graph_manifest = output_root / "clean_graphs" / "graph_shards_manifest.json"

    if not graph_manifest.exists():
        raise FileNotFoundError(f"Graph manifest is missing in {graph_manifest}. Run 02_extract_graphs.py first.")

    env_overrides = {
        "BCB_CLONE_TYPE": clone_type,
        "BCB_DATA_DIR": str(data_dir),
        "OUTPUT_DIR": str(output_root),
        "SPECTRAL_GRAPH_TYPES": ",".join(graph_types),
    }

    print(f"[*] Prepared data: {data_dir}")
    print(f"[*] Output root: {output_root}")
    print(f"[*] Spectral graph types: {', '.join(graph_types)}")

    seconds = _run_python_script("pipelines/03_extract_spectral_features.py", env_overrides)
    record_pipeline_timing(
        output_root / "pipeline_timings.json",
        "03_extract_spectral_features",
        seconds,
        {
            "dataset": "bcb",
            "clone_type": clone_type,
            "variant": data_variant,
            "output_root": str(output_root),
            "script": "pipelines/03_extract_spectral_features.py",
            "spectral_features_manifest": str(output_root / "spectral_features" / "spectral_features_manifest.json"),
        },
    )


def run_bcb_metric_tuning(
    clone_type: str,
    metrics: list[str],
    stage_name: str = "04_tune_metrics",
    variant: str | None = None,
) -> None:
    from spectral_code.evaluation.pipeline_section_runner import SectionConfig, run_pss_wasserstein_tuning

    data_variant = variant or clone_type
    output_root = output_root_for("bcb", data_variant)
    features_manifest = output_root / "spectral_features" / "spectral_features_manifest.json"
    if not features_manifest.exists():
        raise FileNotFoundError(f"Spectral features are missing in {features_manifest}. Run 03_extract_spectral_features.py first.")

    os.environ.setdefault("TUNING_K_VALUES", "full")
    os.environ.setdefault("TUNING_BALANCED_CHUNK_SIZE", "auto")
    os.environ.setdefault("TUNING_BALANCED_CHUNK_ANCHOR", "positive")
    os.environ["TUNING_METRICS"] = ",".join(metrics)
    previous_extra_manifests = os.environ.get("TUNING_EXTRA_FEATURE_MANIFESTS")
    non_clone_manifest = output_root_for("bcb", "non_clone") / "spectral_features" / "spectral_features_manifest.json"
    if str(data_variant).lower().replace("-", "_") != "non_clone":
        if not non_clone_manifest.exists():
            raise FileNotFoundError(
                f"Shared BCB non-clone spectral features are missing in {non_clone_manifest}. "
                "Run create_datasets_graphs/bigclonebench/build_once.py "
                "--variant non_clone --start-at graphs first."
            )
        os.environ["TUNING_EXTRA_FEATURE_MANIFESTS"] = str(non_clone_manifest)

    start = time.perf_counter()
    try:
        run_pss_wasserstein_tuning(
            SectionConfig(dataset="bcb", variant=data_variant, run_dir=Path(__file__).resolve().parent)
        )
    finally:
        if previous_extra_manifests is None:
            os.environ.pop("TUNING_EXTRA_FEATURE_MANIFESTS", None)
        else:
            os.environ["TUNING_EXTRA_FEATURE_MANIFESTS"] = previous_extra_manifests
    seconds = time.perf_counter() - start
    record_pipeline_timing(
        output_root / "pipeline_timings.json",
        stage_name,
        seconds,
        {
            "dataset": "bcb",
            "clone_type": clone_type,
            "variant": data_variant,
            "output_root": str(output_root),
            "features_manifest": str(features_manifest),
            "k_values": os.environ.get("TUNING_K_VALUES", "full"),
            "metrics": metrics,
        },
    )


def run_bcb_pss_wasserstein_tuning(clone_type: str, variant: str | None = None) -> None:
    run_bcb_metric_tuning(
        clone_type,
        metrics=["pss"],
        stage_name="04_tune_pss_wasserstein",
        variant=variant,
    )
