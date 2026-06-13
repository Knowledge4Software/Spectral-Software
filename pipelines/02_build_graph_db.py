import os
import sys
import json
import shutil
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.project_paths import (
    RAW_FEATURES_DIR, 
    CLEAN_GRAPHS_DIR, 
    TIMING_STATS_FILE, 
    ensure_dirs
)
from spectral_code.preprocessing.cleaner import clean_and_compose_graphs_sharded

BASE_LAYERS = ["ast", "cfg", "ddg", "pdg"]
SHARD_SIZE = int(os.getenv("GRAPH_SHARD_SIZE", "1000"))

def main():
    preprocessing_start_time = time.perf_counter()
    
    # Initialize directory structure
    ensure_dirs()

    raw_json_count = len(list(RAW_FEATURES_DIR.glob("*.json")))
    if raw_json_count == 0:
        raise RuntimeError(
            f"No raw feature JSON files found in {RAW_FEATURES_DIR}. "
            "Run pipeline 01 first and confirm it writes dataset_features/*.json."
        )

    if CLEAN_GRAPHS_DIR.exists():
        shutil.rmtree(CLEAN_GRAPHS_DIR)
    CLEAN_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    shard_dir = CLEAN_GRAPHS_DIR / "cleaned_graphs_shards"
    print(f"[*] Loading raw JSON features from {RAW_FEATURES_DIR}...")
    print(f"[*] Raw feature JSON files: {raw_json_count:,}")
    print(f"[*] Writing cleaned graph shards to {shard_dir}...")
    manifest_path, methods_cleaned, layers_cleaned = clean_and_compose_graphs_sharded(
        str(RAW_FEATURES_DIR),
        BASE_LAYERS,
        str(shard_dir),
        shard_size=SHARD_SIZE,
    )

    total_duration = time.perf_counter() - preprocessing_start_time

    # Update timing stats
    stats = {}
    if TIMING_STATS_FILE.exists():
        try:
            with open(TIMING_STATS_FILE, "r") as f:
                stats = json.load(f)
        except Exception:
            pass

    stats["total_preprocessing_time"] = total_duration
    stats["total_methods_cleaned"] = methods_cleaned
    stats["total_layers_cleaned"] = layers_cleaned + methods_cleaned
    stats["clean_graphs_manifest"] = manifest_path
    stats["clean_graphs_shard_size"] = SHARD_SIZE
    
    with open(TIMING_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[+] Success! Processed {methods_cleaned} methods.")
    print(f"[+] Total layers cleaned: {layers_cleaned}")
    print(f"[+] Graph manifest: {manifest_path}")
    print(f"[+] Prep time: {total_duration:.2f}s.")

if __name__ == "__main__":
    main()
