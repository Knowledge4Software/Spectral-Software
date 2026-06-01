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

    TEST_FILE = DATA_DIR / "test.txt"
    if not TEST_FILE.exists():
        return

    print("[*] Generating comprehensive_multi_layer_analysis.csv...")
    pairs_to_process = []
    with open(TEST_FILE, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                pairs_to_process.append(parts)
    
    MAX_PAIRS = 5000 
    import random
    if len(pairs_to_process) > MAX_PAIRS:
        pairs_to_process = random.sample(pairs_to_process, MAX_PAIRS)
        
    pss_metric = PSSSimilarity()
    hk_metric = HeatKernelSimilarity()
    js_metric = JensenShannonSimilarity()
    ws_metric = WassersteinSimilarity()
    
    layers = ["ast", "cfg", "ddg", "pdg", "cpg"]
    integrated_data = []
    
    for id1, id2, label in tqdm(pairs_to_process, desc="Computing Multi-Layer Metrics"):
        if id1 not in features_db or id2 not in features_db:
            continue
            
        row = {
            "pair_ids": f"{id1}_{id2}",
            "id1": id1,
            "id2": id2,
            "label": int(label)
        }
        
        valid_pair = False
        for layer in layers:
            ev1 = features_db[id1].get(layer, {}).get("eigenvalues", np.array([]))
            ev2 = features_db[id2].get(layer, {}).get("eigenvalues", np.array([]))
            
            g1 = graph_db.get(id1, {}).get(layer)
            g2 = graph_db.get(id2, {}).get(layer)

            n1 = g1.number_of_nodes() if g1 else 0
            n2 = g2.number_of_nodes() if g2 else 0
            
            row[f"{layer}_n1"] = n1
            row[f"{layer}_n2"] = n2
            
            if n1 > 0 and n2 > 0 and len(ev1) > 0 and len(ev2) > 0:
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
            integrated_data.append(row)
    
    df_integrated = pd.DataFrame(integrated_data)
    output_path = DATA_DIR / "comprehensive_multi_layer_analysis.csv"
    df_integrated.to_csv(output_path, index=False)
    
    print(f"[+] Saved {len(df_integrated)} valid pairs to {output_path}")

if __name__ == "__main__":
    generate_csvs()
