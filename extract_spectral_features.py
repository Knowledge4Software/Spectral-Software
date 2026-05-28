import os
from spectral_code.spectral.runner import run_spectral_feature_extraction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File path configurations
GRAPH_DB_PATH = os.path.join(BASE_DIR, "outputs", "clean_graphs", "cleaned_graphs_db.pkl")
FEATURES_OUT_DIR = os.path.join(BASE_DIR, "outputs", "spectral_features")
TIMING_FILE = os.path.join(BASE_DIR, "outputs", "timing_stats.json")
GRAPH_TYPES = ["cpg", "ast", "cfg", "ddg", "pdg"]

def main():
    run_spectral_feature_extraction(GRAPH_DB_PATH, FEATURES_OUT_DIR, TIMING_FILE, GRAPH_TYPES)

if __name__ == "__main__":
    main()