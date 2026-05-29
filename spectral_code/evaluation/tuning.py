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
from sklearn.metrics import (
            precision_recall_curve,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        )

def run_fast_grid_search(
        features_db_path,
        bcb_data_dir,
        n_samples=1000,
        optimize_for="accuracy",
        out_filename="trained_models.json"
    ):
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
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_thresholds = np.arange(0.0, 1.01, 0.01)
        
        metric_scores = []

        for i, th in enumerate(pr_thresholds):

            preds = (scores >= th).astype(int)

            if optimize_for == "f1":
                score = f1_score(labels, preds, zero_division=0)

            elif optimize_for == "precision":
                score = precision_score(labels, preds, zero_division=0)

            elif optimize_for == "recall":
                score = recall_score(labels, preds, zero_division=0)

            elif optimize_for == "accuracy":
                score = accuracy_score(labels, preds)

            else:
                raise ValueError(f"Unsupported metric: {optimize_for}")

            metric_scores.append(score)

        metric_scores = np.array(metric_scores)

        best_idx = np.argmax(metric_scores)
        best_metric = metric_scores[best_idx]
        
        # Scikit-learn's precision_recall_curve returns N thresholds for N+1 precision/recall values
        if best_idx < len(pr_thresholds):
            best_th = pr_thresholds[best_idx]
        else:
            best_th = pr_thresholds[-1] if len(pr_thresholds) > 0 else 1.0
            
        print(
            f"[+] Trained: "
            f"Layer={gtype.upper()}, "
            f"K={k if k else 'Full'}, "
            f"Metric={metric} | "
            f"Thresh={best_th:.4f} -> "
            f"{optimize_for.upper()}={best_metric:.3f}"
        )

        trained_models.append({
            "optimized_for": optimize_for,
            "best_metric": float(best_metric),
            "graph_type": gtype,
            "k_eigen": k,
            "metric": metric,
            "best_threshold": float(best_th),
            "train_precision": float(precision[best_idx]),
            "train_recall": float(recall[best_idx]),
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

def run_fused_fast_grid_search(features_db_path, bcb_data_dir, primary_graph="ast", 
        n_samples=1000, k_values=None, metrics=None, out_filename="trained_fused_models.json",
        optimize_for="accuracy"):
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

        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_thresholds = np.arange(0.0, 1.01, 0.01)
        
        metric_scores = []

        for i, th in enumerate(pr_thresholds):

            preds = (scores >= th).astype(int)

            if optimize_for == "f1":
                score = f1_score(labels, preds, zero_division=0)

            elif optimize_for == "precision":
                score = precision_score(labels, preds, zero_division=0)

            elif optimize_for == "recall":
                score = recall_score(labels, preds, zero_division=0)

            elif optimize_for == "accuracy":
                score = accuracy_score(labels, preds)

            else:
                raise ValueError(f"Unsupported metric: {optimize_for}")

            metric_scores.append(score)

        metric_scores = np.array(metric_scores)

        best_idx = np.argmax(metric_scores)
        best_metric = metric_scores[best_idx]
        
        if best_idx < len(pr_thresholds):
            best_th = pr_thresholds[best_idx]
        else:
            best_th = pr_thresholds[-1] if len(pr_thresholds) > 0 else 1.0
            
        print(
            f"[+] Trained: "
            f"Layer={gtype.upper()}, "
            f"K={k if k else 'Full'}, "
            f"Metric={metric} | "
            f"Thresh={best_th:.4f} -> "
            f"{optimize_for.upper()}={best_metric:.3f}"
        )
        
        trained_fused_models.append({
            "optimized_for": optimize_for,
            "best_metric": float(best_metric),
            "graph_type": fused_name,
            "k_eigen": k,
            "metric": metric,
            "best_threshold": float(best_th),
            "train_precision": float(precision[best_idx]),
            "train_recall": float(recall[best_idx])
        })

    # Save to outputs
    out_path = os.path.join(os.path.dirname(features_db_path), "..", out_filename)
    with open(out_path, "w") as f:
        json.dump(trained_fused_models, f, indent=4)
        
    print(f"\n[+] Fused Grid Search Complete! {len(trained_fused_models)} models trained and saved to: {os.path.abspath(out_path)}")
    return trained_fused_models