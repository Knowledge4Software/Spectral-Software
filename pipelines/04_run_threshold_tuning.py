import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.project_paths import SPECTRAL_FEATURES_DIR, OUTPUT_ROOT, ensure_dirs
from spectral_code.evaluation.tuning import run_fast_grid_search , run_fused_fast_grid_search

def main():
    ensure_dirs()
    
    # Path to spectral features
    FEATURES_DB_PATH = SPECTRAL_FEATURES_DIR / "spectral_vectors_full.pkl"
    BCB_DATA_DIR = Path("data")
    
    print(f"[*] Starting Hyperparameter Tuning (Threshold Optimization)...")
    print(f"[*] Features: {FEATURES_DB_PATH}")
    
    # Set n_samples=None to use the full training dataset
    run_fused_fast_grid_search(str(FEATURES_DB_PATH), str(BCB_DATA_DIR), n_samples=None)

if __name__ == "__main__":
    main()
