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
    
    print(f"[*] Starting Comprehensive Hyperparameter Tuning (Threshold Optimization)...")
    print(f"[*] Features: {FEATURES_DB_PATH}")
    
    # Configuration for the "mashit" run (all combinations)
    # k_values=[None] means we use the FULL spectrum (as requested: "فقط روی full کار کن")
    k_values = [None] 
    metrics = ["pss", "heat_kernel", "wasserstein", "jensenshannon"]
    
    # 1. Run Unfused Grid Search for all graph types and all 4 metrics
    print("\n[>>] STEP 1: Running Unfused Grid Search (Single Graph Layers)...")
    run_fast_grid_search(
        str(FEATURES_DB_PATH), 
        str(BCB_DATA_DIR), 
        n_samples=None, 
        k_values=k_values, 
        metrics=metrics,
        out_filename="trained_unfused_models.json"
    )
    
    # 2. Run Fused Grid Search (AST + others) for all 4 metrics
    print("\n[>>] STEP 2: Running Fused Grid Search (AST + Secondary Layers)...")
    run_fused_fast_grid_search(
        str(FEATURES_DB_PATH), 
        str(BCB_DATA_DIR), 
        n_samples=None, 
        k_values=k_values, 
        metrics=metrics,
        out_filename="trained_fused_models.json"
    )
    
    print("\n[+] All tuning completed. Results saved to trained_unfused_models.json and trained_fused_models.json.")

if __name__ == "__main__":
    main()
