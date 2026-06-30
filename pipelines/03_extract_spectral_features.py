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

GRAPH_TYPES = [g.strip().lower() for g in os.getenv("SPECTRAL_GRAPH_TYPES", "ast,cfg,ddg,pdg,cpg").split(",") if g.strip()]

def main():
    extraction_start_time = time.perf_counter()
    
    # Initialize directory structure
    ensure_dirs()

    GRAPH_DB_PATH = CLEAN_GRAPHS_DIR / "graph_shards_manifest.json"
    if not GRAPH_DB_PATH.exists():
        raise FileNotFoundError(
            f"Graph DB not found at {GRAPH_DB_PATH}. "
            "Run pipeline 01 successfully, then pipeline 02 before pipeline 03."
        )

    force_rebuild = os.getenv("SPECTRAL_FORCE_REBUILD", "1").strip().lower() not in {"0", "false", "no"}
    if force_rebuild and SPECTRAL_FEATURES_DIR.exists():
        print(f"[*] Removing old spectral outputs: {SPECTRAL_FEATURES_DIR}")
        shutil.rmtree(SPECTRAL_FEATURES_DIR)
        SPECTRAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting Spectral Feature Extraction (Eigenvalues)...")
    print(f"[*] Graph DB: {GRAPH_DB_PATH}")
    print(f"[*] Max graph nodes per layer: {os.getenv('SPECTRAL_MAX_NODES', '2000')}")
    print(f"[*] Approx top-K for oversized graphs: {os.getenv('SPECTRAL_APPROX_TOPK', '300')}")
    print(f"[*] Force rebuild spectral outputs: {force_rebuild}")
    
    output_path = run_spectral_feature_extraction(
        graph_db_path=str(GRAPH_DB_PATH),
        features_out_dir=str(SPECTRAL_FEATURES_DIR),
        timing_file=str(TIMING_STATS_FILE),
        graph_types=GRAPH_TYPES,
        mode="directed_laplacian"
    )
    if output_path is None:
        raise RuntimeError("Spectral feature extraction failed.")

    with open(TIMING_STATS_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)
    computed_total = sum(int(stats.get(f"spectral_computed_graphs_{gtype}", 0)) for gtype in GRAPH_TYPES)
    if computed_total == 0:
        raise RuntimeError(
            "Spectral extraction completed but produced zero usable graph spectra. "
            "Check pipeline 01/02 outputs and selected SPECTRAL_GRAPH_TYPES."
        )

    total_duration = time.perf_counter() - extraction_start_time
    print(f"\n[+] Success! Spectral features saved to {SPECTRAL_FEATURES_DIR}")
    print(f"[+] Total time: {total_duration:.2f}s.")

if __name__ == "__main__":
    main()
