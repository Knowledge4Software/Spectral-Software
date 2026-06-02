import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from spectral_code.utils.project_paths import CLEAN_GRAPHS_DIR, SPECTRAL_FEATURES_DIR
    from spectral_code.similarity.pss import PSSSimilarity
    from spectral_code.similarity.heat_kernel import HeatKernelSimilarity
    from spectral_code.similarity.distribution import JensenShannonSimilarity, WassersteinSimilarity
except ImportError as e:
    print(f"[-] Import Error: {e}")
    sys.exit(1)

def generate_csvs():
    GRAPH_DB_PATH = CLEAN_GRAPHS_DIR / "cleaned_graphs_db.pkl"
    FEATURES_DB_PATH = SPECTRAL_FEATURES_DIR / "spectral_vectors_full.pkl"
    DATA_DIR = PROJECT_ROOT / "data"
    
    if not GRAPH_DB_PATH.exists() or not FEATURES_DB_PATH.exists():
        print("[-] Required pickle files not found.")
        return

    print("[*] Loading databases...")
    with open(GRAPH_DB_PATH, "rb") as f:
        graph_db = pickle.load(f)
    with open(FEATURES_DB_PATH, "rb") as f:
        features_db = pickle.load(f)

    layers = ["ast", "cfg", "ddg", "pdg", "cpg"]

    # --- CSV 1: METADATA & EIGENVALUES (Structural information) ---
    print("\n[*] Step 1: Generating structural_metadata_stats.csv...")
    metadata_rows = []
    
    for method_id, layers_data in tqdm(graph_db.items(), desc="Extracting Metadata"):
        row = {"method_id": method_id}
        has_data = False
        for layer in layers:
            g = layers_data.get(layer)
            ev = features_db.get(method_id, {}).get(layer, {}).get("eigenvalues", np.array([]))
            
            n_nodes = g.number_of_nodes() if g else 0
            n_edges = g.number_of_edges() if g else 0
            
            row[f"{layer}_nodes"] = n_nodes
            row[f"{layer}_edges"] = n_edges
            # Store eigenvalues as comma-separated string for inspection
            row[f"{layer}_eigenvalues"] = ",".join([f"{v:.4f}" for v in ev]) if len(ev) > 0 else ""
            if n_nodes > 0:
                has_data = True
        
        if has_data:
            metadata_rows.append(row)

    df_metadata = pd.DataFrame(metadata_rows)
    metadata_out = DATA_DIR / "structural_metadata_stats.csv"
    df_metadata.to_csv(metadata_out, index=False)
    print(f"[+] Saved {len(df_metadata)} methods metadata to {metadata_out}")

    # --- CSV 2: SIMILARITY COMPARISON (Train Split Sample) ---
    TRAIN_FILE = DATA_DIR / "train.txt"
    if not TRAIN_FILE.exists():
        print(f"[-] Train file not found at {TRAIN_FILE}. Skipping Step 2.")
    else:
        print("\n[*] Step 2: Generating similarity_comparison_sample.csv (from train.txt)...")
        train_pairs = []
        with open(TRAIN_FILE, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    train_pairs.append(parts)
        
        SAMPLE_SIZE = 2000 
        import random
        if len(train_pairs) > SAMPLE_SIZE:
            train_pairs = random.sample(train_pairs, SAMPLE_SIZE)
            
        pss_metric = PSSSimilarity()
        hk_metric = HeatKernelSimilarity()
        js_metric = JensenShannonSimilarity()
        ws_metric = WassersteinSimilarity()
        
        similarity_data = []
        
        for id1, id2, label in tqdm(train_pairs, desc="Computing Similarities"):
            if id1 not in features_db or id2 not in features_db:
                continue
                
            row = {
                "id1": id1,
                "id2": id2,
                "label": int(label)
            }
            
            valid_pair = False
            for layer in layers:
                ev1 = features_db[id1].get(layer, {}).get("eigenvalues", np.array([]))
                ev2 = features_db[id2].get(layer, {}).get("eigenvalues", np.array([]))
                
                if len(ev1) > 0 and len(ev2) > 0:
                    row[f"pss_{layer}"] = pss_metric.compute(ev1, ev2)
                    row[f"hk_{layer}"] = hk_metric.compute(ev1, ev2)
                    row[f"js_{layer}"] = js_metric.compute(ev1, ev2)
                    row[f"ws_{layer}"] = ws_metric.compute(ev1, ev2)
                    valid_pair = True
                else:
                    row[f"pss_{layer}"] = np.nan
                    row[f"hk_{layer}"] = np.nan
                    row[f"js_{layer}"] = np.nan
                    row[f"ws_{layer}"] = np.nan
            
            if valid_pair:
                similarity_data.append(row)
        
        df_similarity = pd.DataFrame(similarity_data)
        similarity_out = DATA_DIR / "similarity_comparison_sample.csv"
        df_similarity.to_csv(similarity_out, index=False)
        print(f"[+] Saved {len(df_similarity)} sample pairs (Train) to {similarity_out}")

if __name__ == "__main__":
    generate_csvs()
