import os
import sys
import json
import pickle
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from spectral_code.preprocessing.cleaner import clean_and_compose_graphs

RAW_FEATURES_DIR = os.path.join(BASE_DIR, "outputs", "dataset_features")
CLEAN_GRAPHS_DIR = os.path.join(BASE_DIR, "outputs", "clean_graphs")
TIMING_FILE = os.path.join(BASE_DIR, "outputs", "timing_stats.json")
os.makedirs(CLEAN_GRAPHS_DIR, exist_ok=True)

BASE_LAYERS = ["ast", "cfg", "ddg", "pdg"]

def main():
    preprocessing_start_time = time.perf_counter()
    
    cleaned_graphs_db, methods_cleaned, layers_cleaned = clean_and_compose_graphs(
        RAW_FEATURES_DIR, BASE_LAYERS
    )

    output_pkl_path = os.path.join(CLEAN_GRAPHS_DIR, "cleaned_graphs_db.pkl")
    with open(output_pkl_path, "wb") as f:
        pickle.dump(cleaned_graphs_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_duration = time.perf_counter() - preprocessing_start_time

    stats = {}
    if os.path.exists(TIMING_FILE):
        try:
            with open(TIMING_FILE, "r") as f:
                stats = json.load(f)
        except Exception:
            pass

    stats["total_preprocessing_time"] = total_duration
    stats["total_methods_cleaned"] = methods_cleaned
    stats["total_layers_cleaned"] = layers_cleaned + methods_cleaned
    
    with open(TIMING_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[+] Processed {methods_cleaned} methods with 5 strictly directed layers.")

if __name__ == "__main__":
    main()