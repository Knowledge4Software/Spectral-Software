import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from spectral_code.evaluation.bcb_dataset import BigCloneBenchLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# We will use the directed features database
FEATURES_DB_PATH = os.path.join(BASE_DIR, "outputs", "spectral_features", "spectral_vectors_directed.pkl")
BCB_DATA_DIR = os.path.join(BASE_DIR, "data")

def pad_or_truncate(v, k):
    """Ensure vector is exactly length k."""
    if len(v) >= k:
        return v[:k]
    return np.pad(v, (0, k - len(v)))

def build_dataset(features_db, pairs, k=30, graph_types=["ast", "cfg", "ddg", "pdg", "cpg"]):
    """
    Constructs a Machine Learning feature vector for each pair.
    We take the Absolute Difference between the top K eigenvalues of each graph layer.
    """
    X, y = [], []
    for pair in tqdm(pairs, desc="Extracting ML Features"):
        left_data = features_db.get(pair.left_id) or features_db.get(str(pair.left_id))
        right_data = features_db.get(pair.right_id) or features_db.get(str(pair.right_id))
        
        if not left_data or not right_data:
            continue
        
        feat = []
        for gtype in graph_types:
            v1 = left_data.get(gtype, {}).get("eigenvalues", np.array([]))
            v2 = right_data.get(gtype, {}).get("eigenvalues", np.array([]))
            
            # Pad or truncate to fixed feature size K for alignment
            v1k = pad_or_truncate(v1, k)
            v2k = pad_or_truncate(v2, k)
            
            # The core feature: the absolute variation between spectral frequencies
            feat.extend(np.abs(v1k - v2k))
            
        X.append(feat)
        y.append(pair.label)
        
    return np.array(X), np.array(y)

def main():
    print(f"[*] Loading Precomputed Directed Graph Features...")
    if not os.path.exists(FEATURES_DB_PATH):
        print(f"[-] Database not found: {FEATURES_DB_PATH}")
        return
        
    with open(FEATURES_DB_PATH, "rb") as f:
        features_db = pickle.load(f)
        
    print(f"[*] Loading BigCloneBench Pairs...")
    loader = BigCloneBenchLoader(data_dir=BCB_DATA_DIR)
    pairs = loader.get_pairs("train")
    
    # Sub-sample to 100,000 pairs to avoid massive RAM usage during algorithm fitting
    np.random.seed(42)
    sample_size = min(100000, len(pairs))
    sampled_indices = np.random.choice(len(pairs), sample_size, replace=False)
    sampled_pairs = [pairs[i] for i in sampled_indices]
    
    # We will use all 5 graph layers and their top 30 Eigenvalues
    # Total feature size per pair: 5 * 30 = 150 features
    graph_layers = ["ast", "cfg", "ddg", "pdg", "cpg"]
    K = 30 
    
    print(f"[*] Assembling 150-Dimensional Feature Matrix...")
    X, y = build_dataset(features_db, sampled_pairs, k=K, graph_types=graph_layers)
    
    print(f"[*] Splitting Data (80% Train, 20% Validation)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    models_to_eval = []
    
    print(f"[*] Training RandomForest (100 Trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    models_to_eval.append(("RandomForest", rf_probs))
    
    if HAS_XGB:
        print(f"[*] Training XGBoost Classifier...")
        xgb_model = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, n_jobs=-1, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        models_to_eval.append(("XGBoost", xgb_probs))
    else:
        print("[!] XGBoost not installed. Run 'pip install xgboost' if you want to test it.")
        
    print("\n[*] Evaluating Machine Learning Thresholds...")
    for name, probs in models_to_eval:
        p, r, t = precision_recall_curve(y_test, probs)
        
        # Calculate F1 mathematically avoiding zero boundaries
        numerator = 2 * p * r
        denominator = p + r
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = np.where(denominator == 0, 0, numerator / denominator)
            
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_th = t[best_idx] if best_idx < len(t) else 1.0
        
        print(f"[+] Model: {name} | Best Cut-Off Thresh={best_th:.4f} -> Val F1={best_f1:.3f}")
        results.append({
            "model": name,
            "best_threshold": float(best_th),
            "val_precision": float(p[best_idx]),
            "val_recall": float(r[best_idx]),
            "val_f1_score": float(best_f1)
        })
        
    out_path = os.path.join(BASE_DIR, "outputs", "trained_ml_models.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[+] ML models validation results saved to: {out_path}")

if __name__ == "__main__":
    main()
