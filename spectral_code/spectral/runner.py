import os
import pickle
import time
import json
from spectral_code.spectral.extractor import extract_all_spectral_features

def run_spectral_feature_extraction(graph_db_path: str, features_out_dir: str, timing_file: str, graph_types: list[str], mode: str = "normalized_laplacian", output_filename: str = "spectral_vectors_full.pkl"):
    if not os.path.exists(graph_db_path):
        print(f"[-] Database not found at: {graph_db_path}")
        return None

    os.makedirs(features_out_dir, exist_ok=True)

    print("[*] Loading cleaned graph database into memory...")
    with open(graph_db_path, "rb") as f:
        graph_db = pickle.load(f)

    spectral_start_time = time.perf_counter()
    print(f"[*] Extracting Full Lossless Spectrum ({mode}) for sensitivity analysis...")
    
    features_db, layer_counts, layer_durations, layer_node_sums = extract_all_spectral_features(
        graph_db, graph_types, mode=mode
    )

    total_duration = time.perf_counter() - spectral_start_time

    output_path = os.path.join(features_out_dir, output_filename)
    print(f"[*] Saving complete lossless features database to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(features_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats = {}
    if os.path.exists(timing_file):
        try:
            with open(timing_file, "r") as f:
                stats = json.load(f)
        except Exception:
            pass

    stats["total_spectral_extraction_time"] = total_duration
    stats["total_methods_processed"] = len(graph_db)
    
    for gtype in graph_types:
        stats[f"spectral_computed_graphs_{gtype}"] = layer_counts[gtype]
        stats[f"spectral_total_time_{gtype}"] = layer_durations[gtype]
        stats[f"spectral_avg_time_{gtype}_ms"] = (layer_durations[gtype] / max(1, layer_counts[gtype])) * 1000
        stats[f"avg_nodes_{gtype}"] = layer_node_sums[gtype] / max(1, len(graph_db))

    with open(timing_file, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"[+] Full-spectrum integration successful! Optimization database ready for testing.")
    return features_db
