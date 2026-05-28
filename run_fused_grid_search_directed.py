import os
from spectral_code.evaluation.tuning import run_fused_fast_grid_search

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Using the directed spectral features we extracted earlier
    FEATURES_DB_PATH = os.path.join(BASE_DIR, "outputs", "spectral_features", "spectral_vectors_directed.pkl")
    BCB_DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Set n_samples=None to use the full training dataset
    # Set k_values=[None] to only test the "Full" eigenvalues array (skips 25, 50, 100 to save time)
    run_fused_fast_grid_search(
        features_db_path=FEATURES_DB_PATH, 
        bcb_data_dir=BCB_DATA_DIR, 
        primary_graph="ast", 
        n_samples=None,
        k_values=[None],
        metrics=["wasserstein", "jensenshannon"],
        out_filename="trained_fused_models_directed.json"
    )

if __name__ == "__main__":
    main()
