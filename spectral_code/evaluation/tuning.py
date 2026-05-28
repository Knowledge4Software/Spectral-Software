import os
import pickle
import numpy as np
from tqdm import tqdm
from itertools import product
from spectral_code.evaluation.bcb_dataset import BigCloneBenchLoader
from spectral_code.evaluation.baseline import evaluate_binary
from spectral_code.similarity.pss import PSSSimilarity
from spectral_code.similarity.heat_kernel import HeatKernelSimilarity
from spectral_code.similarity.distribution import WassersteinSimilarity, JensenShannonSimilarity

class PrecomputedSpectralModel:
    def __init__(self, features_db, graph_type, k_eigen, threshold, metric_name):
        self.features_db = features_db
        self.graph_type = graph_type
        self.k_eigen = k_eigen
        self.threshold = threshold
        self.metric_name = metric_name
        
        self.pss_metric = PSSSimilarity()
        self.hk_metric = HeatKernelSimilarity()
        self.wasserstein_metric = WassersteinSimilarity()
        self.js_metric = JensenShannonSimilarity()

    def score_pair(self, pair):
        left_data = self.features_db.get(pair.left_id) or self.features_db.get(str(pair.left_id))
        right_data = self.features_db.get(pair.right_id) or self.features_db.get(str(pair.right_id))
        
        if not left_data or not right_data:
            return 0.0  
            
        v1 = left_data.get(self.graph_type, {}).get("eigenvalues", np.array([]))
        v2 = right_data.get(self.graph_type, {}).get("eigenvalues", np.array([]))
        
        if len(v1) == 0 or len(v2) == 0:
            return 0.0

        if self.k_eigen is not None and self.k_eigen > 0:
            v1 = v1[:self.k_eigen]
            v2 = v2[:self.k_eigen]

        if self.metric_name == "cosine":
            max_len = max(len(v1), len(v2))
            v1_pad = np.pad(v1, (0, max_len - len(v1))) if len(v1) < max_len else v1
            v2_pad = np.pad(v2, (0, max_len - len(v2))) if len(v2) < max_len else v2
            n1, n2 = np.linalg.norm(v1_pad), np.linalg.norm(v2_pad)
            if n1 > 0: v1_pad /= n1
            if n2 > 0: v2_pad /= n2
            return float(np.dot(v1_pad, v2_pad))
            
        elif self.metric_name == "pss":
            return self.pss_metric.compute(v1, v2)
            
        elif self.metric_name == "heat_kernel":
            return self.hk_metric.compute(v1, v2)
            
        elif self.metric_name == "wasserstein":
            return self.wasserstein_metric.compute(v1, v2)
            
        elif self.metric_name == "jensenshannon":
            return self.js_metric.compute(v1, v2)
            
        return 0.0

    def predict(self, pair):
        return int(self.score_pair(pair) >= self.threshold)


import json

def run_fast_grid_search(features_db_path, bcb_data_dir, n_samples=1000, out_filename="trained_models.json"):
    print("[*] Loading your precomputed lossless features...")
    if not os.path.exists(features_db_path):
        print(f"[-] Database not found: {features_db_path}")
        return None
        
    with open(features_db_path, "rb") as f:
        features_db = pickle.load(f)
        
    print(f"[*] Loading Clone Pairs from BigCloneBench (Train dataset, {'Full' if n_samples is None else n_samples} samples)...")
    loader = BigCloneBenchLoader(data_dir=bcb_data_dir)
    # Using 'train' data to find the optimal thresholds (hyperparameter tuning)
    if n_samples is None:
        pairs = loader.get_pairs("train")
    else:
        pairs = loader.sample_pairs(split="train", n=n_samples)

    graph_types = ["ast", "cfg", "ddg", "pdg", "cpg"] # 5 graphs
    k_values = [25, 50, 100, None]                    # 4 K values (None = Full)
    metrics = ["pss", "heat_kernel"]                  # 2 Metrics

    trained_models = []
    
    total_base_combos = len(graph_types) * len(k_values) * len(metrics)
    print(f"[*] Starting Exact Continuous 'Training' Phase over {total_base_combos} core configurations...")
    
    for gtype, k, metric in product(graph_types, k_values, metrics):
        # 1. Precalculate all scores for the current config just ONCE
        model = PrecomputedSpectralModel(features_db, gtype, k, None, metric)
        
        scores = []
        labels = []
        
        desc_inner = f"Testing pairs for {gtype.upper()}-K{k if k else 'Full'}-{metric}"
        for pair in tqdm(pairs, desc=desc_inner, leave=False):
            scores.append(model.score_pair(pair))
            labels.append(pair.label)
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        # 2. Learn the EXACT optimal threshold without static 15-step buckets
        # This uses the Precision-Recall curve logic to evaluate every possible cutoff internally
        from sklearn.metrics import precision_recall_curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        
        # Calculate F1-Score for each possible threshold (prevent div by zero)
        numerator = 2 * precision * recall
        denominator = precision + recall
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = np.where(denominator == 0, 0, numerator / denominator)
            
        # Get the index of the highest F1 score
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        
        # Scikit-learn's precision_recall_curve returns N thresholds for N+1 precision/recall values
        if best_idx < len(pr_thresholds):
            best_th = pr_thresholds[best_idx]
        else:
            best_th = pr_thresholds[-1] if len(pr_thresholds) > 0 else 1.0
            
        print(f"[+] Trained: Layer={gtype.upper()}, K={k if k else 'Full'}, Metric={metric} | Exactly Learned Thresh={best_th:.4f} -> F1={best_f1:.3f}")
        
        trained_models.append({
            "graph_type": gtype,
            "k_eigen": k,
            "metric": metric,
            "best_threshold": float(best_th),
            "train_precision": float(precision[best_idx]),
            "train_recall": float(recall[best_idx]),
            "train_f1_score": float(best_f1)
        })

    # Save to outputs
    out_path = os.path.join(os.path.dirname(features_db_path), "..", out_filename)
    with open(out_path, "w") as f:
        json.dump(trained_models, f, indent=4)
        
    print(f"\n[+] Grid Search Complete! {len(trained_models)} models trained and saved to: {os.path.abspath(out_path)}")
    return trained_models

class PrecomputedFusedSpectralModel:
    def __init__(self, features_db, primary_graph, secondary_graph, k_eigen, threshold, metric_name):
        self.model1 = PrecomputedSpectralModel(features_db, primary_graph, k_eigen, threshold, metric_name)
        self.model2 = PrecomputedSpectralModel(features_db, secondary_graph, k_eigen, threshold, metric_name)
        self.threshold = threshold

    def score_pair(self, pair):
        # Late Fusion Approach (Ensemble): Average the similarity scores calculated independently
        return (self.model1.score_pair(pair) + self.model2.score_pair(pair)) / 2.0

    def predict(self, pair):
        return int(self.score_pair(pair) >= self.threshold)

def run_fused_fast_grid_search(features_db_path, bcb_data_dir, primary_graph="ast", n_samples=1000, k_values=None, metrics=None, out_filename="trained_fused_models.json"):
    print(f"[*] Loading your precomputed lossless features for FUSED models (Primary: {primary_graph.upper()})...")
    if not os.path.exists(features_db_path):
        print(f"[-] Database not found: {features_db_path}")
        return None
        
    with open(features_db_path, "rb") as f:
        features_db = pickle.load(f)
        
    print(f"[*] Loading Clone Pairs from BigCloneBench (Train dataset, {'Full' if n_samples is None else n_samples} samples)...")
    loader = BigCloneBenchLoader(data_dir=bcb_data_dir)
    if n_samples is None:
        pairs = loader.get_pairs("train")
    else:
        pairs = loader.sample_pairs(split="train", n=n_samples)

    secondary_graphs = ["cfg", "ddg", "pdg", "cpg"]   # 4 secondary graphs
    if k_values is None:
        k_values = [25, 50, 100, None]                    # 4 K values (None = Full)
    if metrics is None:
        metrics = ["pss", "heat_kernel"]                  # Defaults

    trained_fused_models = []
    
    total_base_combos = len(secondary_graphs) * len(k_values) * len(metrics)
    print(f"[*] Starting Exact Continuous Fused 'Training' Phase over {total_base_combos} core configurations...")
    
    for sec_gtype, k, metric in product(secondary_graphs, k_values, metrics):
        fused_name = f"{primary_graph.upper()}+{sec_gtype.upper()}"
        model = PrecomputedFusedSpectralModel(features_db, primary_graph, sec_gtype, k, None, metric)
        
        scores = []
        labels = []
        
        desc_inner = f"Testing pairs for {fused_name}-K{k if k else 'Full'}-{metric}"
        for pair in tqdm(pairs, desc=desc_inner, leave=False):
            scores.append(model.score_pair(pair))
            labels.append(pair.label)
            
        scores = np.array(scores)
        labels = np.array(labels)
        
        from sklearn.metrics import precision_recall_curve
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        
        numerator = 2 * precision * recall
        denominator = precision + recall
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_scores = np.where(denominator == 0, 0, numerator / denominator)
            
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        
        if best_idx < len(pr_thresholds):
            best_th = pr_thresholds[best_idx]
        else:
            best_th = pr_thresholds[-1] if len(pr_thresholds) > 0 else 1.0
            
        print(f"[+] Trained Fused: Layer={fused_name}, K={k if k else 'Full'}, Metric={metric} | Exactly Learned Thresh={best_th:.4f} -> F1={best_f1:.3f}")
        
        trained_fused_models.append({
            "graph_type": fused_name,
            "k_eigen": k,
            "metric": metric,
            "best_threshold": float(best_th),
            "train_precision": float(precision[best_idx]),
            "train_recall": float(recall[best_idx]),
            "train_f1_score": float(best_f1)
        })

    # Save to outputs
    out_path = os.path.join(os.path.dirname(features_db_path), "..", out_filename)
    with open(out_path, "w") as f:
        json.dump(trained_fused_models, f, indent=4)
        
    print(f"\n[+] Fused Grid Search Complete! {len(trained_fused_models)} models trained and saved to: {os.path.abspath(out_path)}")
    return trained_fused_models