import os
import sys
import json
import pickle
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
from spectral_code.preprocessing.cleaner import clean_and_compose_graphs

BASE_LAYERS = ["ast", "cfg", "ddg", "pdg"]

def main():
    preprocessing_start_time = time.perf_counter()
    
    # Initialize directory structure
    ensure_dirs()
    
    print(f"[*] Loading raw JSON features from {RAW_FEATURES_DIR}...")
    cleaned_graphs_db, methods_cleaned, layers_cleaned = clean_and_compose_graphs(
        str(RAW_FEATURES_DIR), BASE_LAYERS
    )

    output_pkl_path = CLEAN_GRAPHS_DIR / "cleaned_graphs_db.pkl"
    print(f"[*] Saving unified graph DB to {output_pkl_path}...")
    with open(output_pkl_path, "wb") as f:
        pickle.dump(cleaned_graphs_db, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    
    with open(TIMING_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[+] Success! Processed {methods_cleaned} methods.")
    print(f"[+] Total layers cleaned: {layers_cleaned}")
    print(f"[+] Prep time: {total_duration:.2f}s.")

if __name__ == "__main__":
    main()
