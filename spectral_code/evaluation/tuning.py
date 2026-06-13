import os
import pickle
import numpy as np
import json
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

        elif self.metric_name == "topk_largest":
            # Strip zeros first
            def strip(ev):
                nz = np.nonzero(np.round(ev, 10))[0]
                return ev[:nz[-1] + 1] if len(nz) > 0 else np.array([0.0])
            
            v1_s = strip(v1)
            v2_s = strip(v2)
            
            # Take last 58 (largest)
            v1_sort = np.sort(v1_s)
            v2_sort = np.sort(v2_s)
            
            K = min(len(v1_s), len(v2_s))
            if K == 0: return 0.0
            v1_top = np.sort(v1_s)[-K:]
            v2_top = np.sort(v2_s)[-K:]
            
            dist = np.linalg.norm(v1_top - v2_top)
            return 1.0 / (1.0 + dist)
            
        return 0.0

    def predict(self, pair):
        return int(self.score_pair(pair) >= self.threshold)

from sklearn.metrics import (
            precision_recall_curve,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        )


def _metric_score(labels, preds, optimize_for):
    if optimize_for == "f1":
        return f1_score(labels, preds, zero_division=0)

    if optimize_for == "precision":
        return precision_score(labels, preds, zero_division=0)

    if optimize_for == "recall":
        return recall_score(labels, preds, zero_division=0)

    if optimize_for == "accuracy":
        return accuracy_score(labels, preds)

    raise ValueError(f"Unsupported metric: {optimize_for}")


def _candidate_thresholds(scores):
    unique_scores = np.unique(scores)
    candidates = {0.0, 1.0}

    for score in unique_scores:
        candidates.add(float(score))

    if len(unique_scores) > 1:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
        candidates.update(float(th) for th in midpoints)

    return np.array(sorted(candidates))


def _find_best_threshold(scores, labels, optimize_for):
    thresholds = _candidate_thresholds(scores)
    best_th = 0.0
    best_metric = -1.0

    for th in thresholds:
        preds = (scores >= th).astype(int)
        score = _metric_score(labels, preds, optimize_for)

        if score > best_metric or (score == best_metric and th > best_th):
            best_metric = score
            best_th = th

    return float(best_th), float(best_metric)


def _pair_method_ids(pairs):
    needed = set()
    for pair in pairs:
        needed.add(str(pair.left_id))
        needed.add(str(pair.right_id))
    return needed


def _has_feature_record(features_db, method_id) -> bool:
    return method_id in features_db or str(method_id) in features_db


def _filter_pairs_with_features(pairs, features_db):
    filtered = [
        pair for pair in pairs
        if _has_feature_record(features_db, pair.left_id)
        and _has_feature_record(features_db, pair.right_id)
    ]
    dropped = len(pairs) - len(filtered)
    if dropped:
        print(f"[*] Dropped {dropped:,} pairs with missing method features.")
    if not filtered:
        raise RuntimeError("No pairs remain after dropping pairs with missing method features.")
    return filtered


def _load_features_db(features_db_path, needed_ids=None):
    if not os.path.exists(features_db_path):
        print(f"[-] Database not found: {features_db_path}")
        return None

    if str(features_db_path).lower().endswith(".json"):
        with open(features_db_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest.get("format") != "spectral_feature_shards_v1":
            raise ValueError(f"Unsupported feature manifest: {features_db_path}")

        features_db = {}
        for shard_path in tqdm(manifest["shards"], desc="Loading feature shards", unit="shard"):
            with open(shard_path, "rb") as f:
                shard = pickle.load(f)

            if needed_ids is None:
                features_db.update(shard)
            else:
                for method_id in needed_ids:
                    if method_id in shard:
                        features_db[method_id] = shard[method_id]

            if needed_ids is not None and len(features_db) >= len(needed_ids):
                break

        return features_db

    with open(features_db_path, "rb") as f:
        features_db = pickle.load(f)

    if needed_ids is None:
        return features_db

    return {method_id: features_db[method_id] for method_id in needed_ids if method_id in features_db}

def run_fast_grid_search(
        features_db_path,
        bcb_data_dir,
        n_samples=1000,
        optimize_for="accuracy",
        out_filename="trained_models.json",
        graph_types=None,
        k_values=None,
        metrics=None
    ):
    print(f"[*] Loading Clone Pairs from BigCloneBench (Train dataset, {'Full' if n_samples is None else n_samples} samples)...")
    loader = BigCloneBenchLoader(data_dir=bcb_data_dir)
    # Using 'train' data to find the optimal thresholds (hyperparameter tuning)
    if n_samples is None:
        pairs = loader.get_pairs("train")
    else:
        pairs = loader.sample_pairs(split="train", n=n_samples)

    print("[*] Loading precomputed lossless features for selected pairs...")
    features_db = _load_features_db(features_db_path, needed_ids=_pair_method_ids(pairs))
    if features_db is None:
        return None
    print(f"[*] Loaded spectral features for {len(features_db)} methods used by selected pairs.")
    pairs = _filter_pairs_with_features(pairs, features_db)

    if graph_types is None:
        graph_types = ["ast", "cfg", "ddg", "pdg", "cpg"] # 5 graphs
    if k_values is None:
        k_values = [25, 50, 100, None]                    # 4 K values (None = Full)
    if metrics is None:
        metrics = ["pss", "heat_kernel", "wasserstein", "jensenshannon"]                  # 4 Metrics

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
        
        best_th, best_metric = _find_best_threshold(scores, labels, optimize_for)
            
        print(
            f"[+] Trained: "
            f"Layer={gtype.upper()}, "
            f"K={k if k else 'Full'}, "
            f"Metric={metric} | "
            f"Thresh={best_th:.6f} -> "
            f"{optimize_for.upper()}={best_metric:.3f}"
        )

        best_preds = (scores >= best_th).astype(int)

        trained_models.append({
            "optimized_for": optimize_for,
            "best_metric": float(best_metric),
            "graph_type": gtype,
            "k_eigen": k,
            "metric": metric,
            "best_threshold": float(best_th),
            "train_accuracy": float(accuracy_score(labels, best_preds)),
            "train_precision": float(precision_score(labels, best_preds, zero_division=0)),
            "train_recall": float(recall_score(labels, best_preds, zero_division=0)),
            "train_f1": float(f1_score(labels, best_preds, zero_division=0)),
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
    print(f"[*] Loading Clone Pairs from BigCloneBench (Train dataset, {'Full' if n_samples is None else n_samples} samples)...")
    loader = BigCloneBenchLoader(data_dir=bcb_data_dir)
    if n_samples is None:
        pairs = loader.get_pairs("train")
    else:
        pairs = loader.sample_pairs(split="train", n=n_samples)

    print(f"[*] Loading precomputed lossless features for FUSED models (Primary: {primary_graph.upper()})...")
    features_db = _load_features_db(features_db_path, needed_ids=_pair_method_ids(pairs))
    if features_db is None:
        return None
    print(f"[*] Loaded spectral features for {len(features_db)} methods used by selected pairs.")
    pairs = _filter_pairs_with_features(pairs, features_db)

    secondary_graphs = ["cfg", "ddg", "pdg", "cpg"]   # 4 secondary graphs
    if k_values is None:
        k_values = [25, 50, 100, None]                    # 4 K values (None = Full)
    if metrics is None:
        metrics = ["pss", "heat_kernel", "wasserstein", "jensenshannon"]                  # Defaults

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

        best_th, best_metric = _find_best_threshold(scores, labels, optimize_for)
            
        print(
            f"[+] Trained: "
            f"Layer={fused_name}, "
            f"K={k if k else 'Full'}, "
            f"Metric={metric} | "
            f"Thresh={best_th:.6f} -> "
            f"{optimize_for.upper()}={best_metric:.3f}"
        )

        best_preds = (scores >= best_th).astype(int)
        
        trained_fused_models.append({
            "optimized_for": optimize_for,
            "best_metric": float(best_metric),
            "graph_type": fused_name,
            "k_eigen": k,
            "metric": metric,
            "best_threshold": float(best_th),
            "train_accuracy": float(accuracy_score(labels, best_preds)),
            "train_precision": float(precision_score(labels, best_preds, zero_division=0)),
            "train_recall": float(recall_score(labels, best_preds, zero_division=0)),
            "train_f1": float(f1_score(labels, best_preds, zero_division=0)),
        })

    # Save to outputs
    out_path = os.path.join(os.path.dirname(features_db_path), "..", out_filename)
    with open(out_path, "w") as f:
        json.dump(trained_fused_models, f, indent=4)
        
    print(f"\n[+] Fused Grid Search Complete! {len(trained_fused_models)} models trained and saved to: {os.path.abspath(out_path)}")
    return trained_fused_models
