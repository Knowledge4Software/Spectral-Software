import os
from spectral_code.evaluation.tuning import run_fast_grid_search

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FEATURES_DB_PATH = os.path.join(BASE_DIR, "outputs", "spectral_features", "spectral_vectors_directed.pkl")
    BCB_DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Set n_samples=None to use the full training dataset
    run_fast_grid_search(FEATURES_DB_PATH, BCB_DATA_DIR, n_samples=None, out_filename="trained_models_directed.json")

if __name__ == "__main__":
    main()
