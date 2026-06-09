import os
import sys
import json
import time
import shutil
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.project_paths import (
    CLEAN_GRAPHS_DIR, 
    SPECTRAL_FEATURES_DIR, 
    TIMING_STATS_FILE,
    ensure_dirs
)
from spectral_code.spectral.runner import run_spectral_feature_extraction

GRAPH_TYPES = ["ast", "cfg", "ddg", "pdg", "cpg"]

def main():
    extraction_start_time = time.perf_counter()
    
    # Initialize directory structure
    ensure_dirs()

    force_rebuild = os.getenv("SPECTRAL_FORCE_REBUILD", "1").strip().lower() not in {"0", "false", "no"}
    if force_rebuild and SPECTRAL_FEATURES_DIR.exists():
        print(f"[*] Removing old spectral outputs: {SPECTRAL_FEATURES_DIR}")
        shutil.rmtree(SPECTRAL_FEATURES_DIR)
        SPECTRAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    GRAPH_DB_PATH = CLEAN_GRAPHS_DIR / "graph_shards_manifest.json"
    
    print(f"[*] Starting Spectral Feature Extraction (Eigenvalues)...")
    print(f"[*] Graph DB: {GRAPH_DB_PATH}")
    print(f"[*] Max graph nodes per layer: {os.getenv('SPECTRAL_MAX_NODES', '2000')}")
    print(f"[*] Approx top-K for oversized graphs: {os.getenv('SPECTRAL_APPROX_TOPK', '300')}")
    print(f"[*] Force rebuild spectral outputs: {force_rebuild}")
    
    run_spectral_feature_extraction(
        graph_db_path=str(GRAPH_DB_PATH),
        features_out_dir=str(SPECTRAL_FEATURES_DIR),
        timing_file=str(TIMING_STATS_FILE),
        graph_types=GRAPH_TYPES,
        mode="directed_laplacian"
    )

    total_duration = time.perf_counter() - extraction_start_time
    print(f"\n[+] Success! Spectral features saved to {SPECTRAL_FEATURES_DIR}")
    print(f"[+] Total time: {total_duration:.2f}s.")

if __name__ == "__main__":
    main()
