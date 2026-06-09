import os
import pickle
import time
import json
from pathlib import Path
from tqdm import tqdm
from spectral_code.spectral.extractor import extract_all_spectral_features


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
        feature_shard_dir = Path(features_out_dir) / "spectral_feature_shards"
        feature_shard_dir.mkdir(parents=True, exist_ok=True)
        feature_shards = []
        total_methods = 0

        for shard_index, graph_shard_path in enumerate(tqdm(manifest["shards"], desc="Spectral graph shards", unit="shard"), start=1):
            feature_shard_path = feature_shard_dir / f"features_{shard_index:06d}.pkl"
            if feature_shard_path.exists():
                feature_shards.append(str(feature_shard_path))
                continue

            with open(graph_shard_path, "rb") as f:
                graph_db = pickle.load(f)

            features_db, shard_counts, shard_durations, shard_node_sums, shard_skipped, shard_approx = extract_all_spectral_features(
                graph_db, graph_types, mode=mode
            )

            with open(feature_shard_path, "wb") as f:
                pickle.dump(features_db, f, protocol=pickle.HIGHEST_PROTOCOL)
            feature_shards.append(str(feature_shard_path))
            total_methods += len(graph_db)

            for gtype in graph_types:
                layer_counts[gtype] += shard_counts[gtype]
                layer_durations[gtype] += shard_durations[gtype]
                layer_node_sums[gtype] += shard_node_sums[gtype]
                layer_skipped[gtype] += shard_skipped[gtype]
                layer_approx[gtype] += shard_approx[gtype]

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
            "approx_top_k": int(os.getenv("SPECTRAL_APPROX_TOPK", "300")),
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
