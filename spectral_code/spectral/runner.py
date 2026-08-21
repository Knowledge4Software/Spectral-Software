import os
import pickle
import time
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
from spectral_code.spectral.extractor import extract_all_spectral_features, spectral_worker_settings


def spectral_shard_workers() -> int:
    """Independent graph shards can use separate ARPACK processes safely."""
    return max(1, int(os.getenv("SPECTRAL_SHARD_WORKERS", "4")))


def _extract_graph_shard_task(task):
    graph_shard_path, feature_shard_path, graph_types, mode = task
    with open(graph_shard_path, "rb") as f:
        graph_db = pickle.load(f)
    result = extract_all_spectral_features(
        graph_db,
        list(graph_types),
        mode=mode,
        show_progress=False,
    )
    features_db = result[0]
    feature_path = Path(feature_shard_path)
    temporary = feature_path.with_name(f"{feature_path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as f:
            pickle.dump(features_db, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(feature_path)
    finally:
        temporary.unlink(missing_ok=True)
    return (str(feature_path), len(graph_db), *result[1:])


def _load_graph_shard_manifest(graph_db_path: str):
    path = Path(graph_db_path)
    if path.is_dir():
        path = path / "graph_shards_manifest.json"
    if path.suffix.lower() != ".json":
        return None
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "cleaned_graph_shards_v1":
        return None
    return path, manifest


def _feature_shard_supports_top_k(features_db: dict, graph_types: list[str], requested_top_k: int) -> bool:
    """Reject resumable sparse shards produced with a smaller spectral window."""
    for method_features in features_db.values():
        if not isinstance(method_features, dict):
            continue
        for gtype in graph_types:
            feature = method_features.get(gtype, {})
            if not isinstance(feature, dict):
                continue
            status = str(feature.get("status", ""))
            if "sparse" not in status:
                continue
            values = feature.get("eigenvalues")
            try:
                value_count = len(values) if values is not None else 0
            except TypeError:
                value_count = 0
            if value_count < requested_top_k:
                return False
    return True


def run_spectral_feature_extraction(graph_db_path: str, features_out_dir: str, timing_file: str, graph_types: list[str], mode: str = "normalized_laplacian", output_filename: str = "spectral_vectors_full.pkl"):
    if not os.path.exists(graph_db_path):
        print(f"[-] Database not found at: {graph_db_path}")
        return None

    os.makedirs(features_out_dir, exist_ok=True)

    spectral_start_time = time.perf_counter()
    print(f"[*] Extracting Full Lossless Spectrum ({mode}) for sensitivity analysis...")

    layer_counts = {gtype: 0 for gtype in graph_types}
    layer_durations = {gtype: 0.0 for gtype in graph_types}
    layer_node_sums = {gtype: 0 for gtype in graph_types}
    layer_skipped = {gtype: 0 for gtype in graph_types}
    layer_approx = {gtype: 0 for gtype in graph_types}

    manifest_info = _load_graph_shard_manifest(graph_db_path)
    if manifest_info:
        _, manifest = manifest_info
        if manifest.get("total_methods", 0) <= 0 or not manifest.get("shards"):
            raise RuntimeError(
                f"Graph shard manifest is empty: {graph_db_path}. "
                "Run pipeline 01 and pipeline 02 first, and confirm pipeline 02 processed methods."
            )

        feature_shard_dir = Path(features_out_dir) / "spectral_feature_shards"
        feature_shard_dir.mkdir(parents=True, exist_ok=True)
        feature_shards = [None] * len(manifest["shards"])
        total_methods = 0
        pending_tasks = []
        requested_top_k = int(os.getenv("SPECTRAL_APPROX_TOPK", "128"))

        for shard_index, graph_shard_path in enumerate(manifest["shards"], start=1):
            feature_shard_path = feature_shard_dir / f"features_{shard_index:06d}.pkl"
            if feature_shard_path.exists():
                # A stopped incremental run may already have completed this
                # shard. Count its durable features instead of recomputing it
                # or reporting a false zero-computation failure upstream.
                with open(feature_shard_path, "rb") as f:
                    existing_features = pickle.load(f)
                if not _feature_shard_supports_top_k(
                    existing_features, graph_types, requested_top_k
                ):
                    print(
                        f"[*] Recomputing {feature_shard_path.name}: its sparse "
                        f"spectrum is shorter than {requested_top_k}."
                    )
                    pending_tasks.append(
                        (
                            str(graph_shard_path),
                            str(feature_shard_path),
                            tuple(graph_types),
                            mode,
                        )
                    )
                    continue
                total_methods += len(existing_features)
                for method_features in existing_features.values():
                    if not isinstance(method_features, dict):
                        continue
                    for gtype in graph_types:
                        feature = method_features.get(gtype, {})
                        if not isinstance(feature, dict):
                            continue
                        values = feature.get("eigenvalues")
                        try:
                            has_values = values is not None and len(values) > 0
                        except TypeError:
                            has_values = values is not None
                        if has_values:
                            layer_counts[gtype] += 1
                        status = str(feature.get("status", ""))
                        if "approx" in status or "sparse" in status:
                            layer_approx[gtype] += 1
                feature_shards[shard_index - 1] = str(feature_shard_path)
                continue
            pending_tasks.append(
                (
                    str(graph_shard_path),
                    str(feature_shard_path),
                    tuple(graph_types),
                    mode,
                )
            )

        shard_workers = min(spectral_shard_workers(), max(1, len(pending_tasks)))
        if pending_tasks:
            print(
                f"[*] Computing {len(pending_tasks)} missing spectral shards "
                f"with {shard_workers} process(es)."
            )
            if shard_workers == 1:
                shard_results = map(_extract_graph_shard_task, pending_tasks)
                executor = None
            else:
                executor = ProcessPoolExecutor(max_workers=shard_workers)
                shard_results = executor.map(_extract_graph_shard_task, pending_tasks)
            try:
                shard_results = tqdm(
                    shard_results,
                    total=len(pending_tasks),
                    desc="Spectral graph shards",
                    unit="shard",
                )
                for result in shard_results:
                    feature_path, method_count, shard_counts, shard_durations, shard_node_sums, shard_skipped, shard_approx = result
                    shard_index = int(Path(feature_path).stem.rsplit("_", 1)[-1]) - 1
                    feature_shards[shard_index] = feature_path
                    total_methods += method_count
                    for gtype in graph_types:
                        layer_counts[gtype] += shard_counts[gtype]
                        layer_durations[gtype] += shard_durations[gtype]
                        layer_node_sums[gtype] += shard_node_sums[gtype]
                        layer_skipped[gtype] += shard_skipped[gtype]
                        layer_approx[gtype] += shard_approx[gtype]
            except BaseException:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = None
                raise
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

        if any(path is None for path in feature_shards):
            raise RuntimeError("At least one spectral feature shard was not produced")

        if total_methods == 0:
            total_methods = manifest.get("total_methods", 0)

        output_path = os.path.join(features_out_dir, "spectral_features_manifest.json")
        feature_manifest = {
            "format": "spectral_feature_shards_v1",
            "mode": mode,
            "total_methods": total_methods,
            "graph_types": graph_types,
            "shards": feature_shards,
            "dense_max_nodes": int(os.getenv("SPECTRAL_MAX_NODES", "2000")),
            "approx_top_k": int(os.getenv("SPECTRAL_APPROX_TOPK", "128")),
            "sparse_solver": os.getenv("SPECTRAL_SPARSE_SOLVER", "shift_invert"),
            "workers": spectral_worker_settings()[0],
            "blas_threads_per_worker": spectral_worker_settings()[1],
            "shard_workers": spectral_shard_workers(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(feature_manifest, f, indent=2)
    else:
        print("[*] Loading cleaned graph database into memory...")
        with open(graph_db_path, "rb") as f:
            graph_db = pickle.load(f)

        features_db, layer_counts, layer_durations, layer_node_sums, layer_skipped, layer_approx = extract_all_spectral_features(
            graph_db, graph_types, mode=mode
        )

        total_methods = len(graph_db)
        output_path = os.path.join(features_out_dir, output_filename)
        print(f"[*] Saving complete lossless features database to: {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(features_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_duration = time.perf_counter() - spectral_start_time

    stats = {}
    if os.path.exists(timing_file):
        try:
            with open(timing_file, "r") as f:
                stats = json.load(f)
        except Exception:
            pass

    stats["total_spectral_extraction_time"] = total_duration
    stats["total_methods_processed"] = total_methods
    stats["spectral_features_path"] = output_path
    stats["spectral_workers"] = spectral_worker_settings()[0]
    stats["spectral_blas_threads_per_worker"] = spectral_worker_settings()[1]
    stats["spectral_shard_workers"] = spectral_shard_workers()
    stats["spectral_sparse_solver"] = os.getenv("SPECTRAL_SPARSE_SOLVER", "shift_invert")
    
    for gtype in graph_types:
        stats[f"spectral_computed_graphs_{gtype}"] = layer_counts[gtype]
        stats[f"spectral_skipped_oversized_{gtype}"] = layer_skipped[gtype]
        stats[f"spectral_approx_topk_graphs_{gtype}"] = layer_approx[gtype]
        stats[f"spectral_total_time_{gtype}"] = layer_durations[gtype]
        stats[f"spectral_avg_time_{gtype}_ms"] = (layer_durations[gtype] / max(1, layer_counts[gtype])) * 1000
        stats[f"avg_nodes_{gtype}"] = layer_node_sums[gtype] / max(1, total_methods)

    with open(timing_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"[+] Full-spectrum integration successful! Features saved to: {output_path}")
    return output_path
