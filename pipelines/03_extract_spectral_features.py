import os
import sys
import json
import time
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
    
    GRAPH_DB_PATH = CLEAN_GRAPHS_DIR / "cleaned_graphs_db.pkl"
    
    print(f"[*] Starting Spectral Feature Extraction (Eigenvalues)...")
    print(f"[*] Graph DB: {GRAPH_DB_PATH}")
    
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
