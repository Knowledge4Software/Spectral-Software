import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.utils.project_paths import SPECTRAL_FEATURES_DIR
from spectral_code.similarity.pss import PSSSimilarity

def load_train_pairs(train_file):
    pairs = []
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                pairs.append((parts[0], parts[1], int(parts[2])))
    return pairs

def main():
    FEATURES_DB_PATH = SPECTRAL_FEATURES_DIR / "spectral_vectors_full.pkl"
    TRAIN_FILE = PROJECT_ROOT / "data" / "train.txt"
    
    print(f"[*] Loading features from {FEATURES_DB_PATH}...")
    with open(FEATURES_DB_PATH, "rb") as f:
        features_db = pickle.load(f)
        
    print(f"[*] Loading training pairs from {TRAIN_FILE}...")
    pairs = load_train_pairs(TRAIN_FILE)
    print(f"[*] Total pairs: {len(pairs)}")
    
    pss = PSSSimilarity()
    
    scores = []
    labels = []
    
    print("[*] Computing PSS (Zero-Padding) for AST layer...")
    # Using simple loop with indices for speed (900k is manageable)
    for i, (id1, id2, label) in enumerate(pairs):
        if i % 100000 == 0 and i > 0:
            print(f"    Processed {i} pairs...")
            
        f1 = features_db.get(str(id1))
        f2 = features_db.get(str(id2))
        
        if not f1 or not f2:
            continue
            
        ev1 = f1.get('ast', {}).get('eigenvalues', np.array([]))
        ev2 = f2.get('ast', {}).get('eigenvalues', np.array([]))
        
        if len(ev1) == 0 or len(ev2) == 0:
            score = 0.0
        else:
            score = pss.compute(ev1, ev2)
            
        scores.append(score)
        labels.append(label)
        
    if not scores:
        print("[-] No valid pairs found!")
        return
        
    y_true = np.array(labels)
    y_scores = np.array(scores)
    
    clone_scores = y_scores[y_true == 1]
    non_clone_scores = y_scores[y_true == 0]
    
    print(f"[*] Score Stats: Min={y_scores.min():.4f}, Max={y_scores.max():.4f}, Mean={y_scores.mean():.4f}")
    print(f"[*] Clone Mean:     {clone_scores.mean():.4f} (std: {clone_scores.std():.4f})")
    print(f"[*] Non-Clone Mean: {non_clone_scores.mean():.4f} (std: {non_clone_scores.std():.4f})")
    print(f"[*] Label Stats: Positives={y_true.sum()}, Negatives={len(y_true) - y_true.sum()}")

    print("[*] Finding optimal threshold for F1-score...")
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Avoid div by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Calculate Accuracy at best threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    final_acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("AST + PSS (Zero-Padding) FULL EVALUATION")
    print("="*40)
    print(f"Total Evaluated Pairs: {len(y_true)}")
    print(f"Optimal Threshold:    {best_threshold:.4f}")
    print(f"Best F1 Score:        {best_f1:.4f}")
    print(f"Accuracy:             {final_acc:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
