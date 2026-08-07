import os
import pickle
import random
import csv
import time
import numpy as np
import json
from tqdm import tqdm
from itertools import product
from spectral_code.evaluation.bcb_dataset import BigCloneBenchLoader
from spectral_code.evaluation.baseline import evaluate_binary
from spectral_code.similarity.pss import PSSSimilarity
from spectral_code.similarity.heat_kernel import HeatKernelSimilarity
from spectral_code.similarity.distribution import (
    FisherInformationSimilarity,
    JensenShannonSimilarity,
    WassersteinSimilarity,
)
from spectral_code.evaluation.bcb_dataset import ClonePair

class PrecomputedSpectralModel:
    def __init__(self, features_db, graph_type, k_eigen, threshold, metric_name):
        self.features_db = features_db
        self.graph_type = graph_type
        self.k_eigen = k_eigen
        self.threshold = threshold
        self.metric_name = metric_name
        
        self.pss_metric = PSSSimilarity() if metric_name == "pss" else None
        self.hk_metric = HeatKernelSimilarity() if metric_name == "heat_kernel" else None
        self.wasserstein_metric = (
            WassersteinSimilarity(gamma=float(os.getenv("WASSERSTEIN_GAMMA", "0.1")))
            if metric_name == "wasserstein"
            else None
        )
        self.js_metric = JensenShannonSimilarity() if metric_name == "jensenshannon" else None
        self.fisher_metric = (
            FisherInformationSimilarity(gamma=float(os.getenv("FISHER_GAMMA", "1.0")))
            if metric_name == "fisher"
            else None
        )

    def metric_parameters(self):
        if self.metric_name == "pss":
            metric = self.pss_metric or PSSSimilarity()
            return {
                "pss_gamma": float(metric.gamma),
                "pss_distance_power": float(metric.distance_power),
            }
        if self.metric_name == "wasserstein":
            metric = self.wasserstein_metric or WassersteinSimilarity(
                gamma=float(os.getenv("WASSERSTEIN_GAMMA", "0.1"))
            )
            return {"wasserstein_gamma": float(metric.gamma)}
        if self.metric_name == "fisher":
            metric = self.fisher_metric or FisherInformationSimilarity(
                gamma=float(os.getenv("FISHER_GAMMA", "1.0"))
            )
            return {"fisher_gamma": float(metric.gamma)}
        return {}

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
            # Out-of-place: v1_pad can alias the array cached in features_db,
            # and an in-place divide permanently rescaled the stored spectrum
            # for every later pair that referenced the same method.
            if n1 > 0: v1_pad = v1_pad / n1
            if n2 > 0: v2_pad = v2_pad / n2
            return float(np.dot(v1_pad, v2_pad))
            
        elif self.metric_name == "pss":
            if self.pss_metric is None:
                self.pss_metric = PSSSimilarity()
            return self.pss_metric.compute(v1, v2)
            
        elif self.metric_name == "heat_kernel":
            if self.hk_metric is None:
                self.hk_metric = HeatKernelSimilarity()
            return self.hk_metric.compute(v1, v2)
            
        elif self.metric_name == "wasserstein":
            if self.wasserstein_metric is None:
                self.wasserstein_metric = WassersteinSimilarity(
                    gamma=float(os.getenv("WASSERSTEIN_GAMMA", "0.1"))
                )
            return self.wasserstein_metric.compute(v1, v2)
            
        elif self.metric_name == "jensenshannon":
            if self.js_metric is None:
                self.js_metric = JensenShannonSimilarity()
            return self.js_metric.compute(v1, v2)

        elif self.metric_name == "fisher":
            if self.fisher_metric is None:
                self.fisher_metric = FisherInformationSimilarity(
                    gamma=float(os.getenv("FISHER_GAMMA", "1.0"))
                )
            return self.fisher_metric.compute(v1, v2)

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
        return int(_score_at_or_above_threshold(self.score_pair(pair), self.threshold))

from sklearn.metrics import (
            precision_recall_curve,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
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


def _decision_threshold(threshold):
    return float(np.nextafter(float(threshold), -np.inf))


def _score_at_or_above_threshold(score, threshold):
    return float(score) >= _decision_threshold(threshold)


def _scores_at_or_above_threshold(scores, threshold):
    return np.asarray(scores, dtype=np.float64) >= _decision_threshold(threshold)


def _safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    preds = _scores_at_or_above_threshold(scores, threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": _safe_auc(labels, scores),
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.var(ddof=1) if arr.size > 1 else 0.0)


def _balanced_class_chunks(scores, labels, random_seed=42):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return [], {}

    rng = np.random.default_rng(random_seed)
    chunk_size_raw = os.getenv("TUNING_BALANCED_CHUNK_SIZE", "auto").strip().lower()
    anchor_raw = os.getenv("TUNING_BALANCED_CHUNK_ANCHOR", "auto").strip().lower()

    if anchor_raw in {"positive", "positives", "pos", "clone", "clones"}:
        anchor_label = 1
    elif anchor_raw in {"negative", "negatives", "neg", "nonclone", "non-clone", "non_clones"}:
        anchor_label = 0
    else:
        anchor_label = 1 if len(pos_idx) >= len(neg_idx) else 0

    anchor_idx = rng.permutation(pos_idx if anchor_label == 1 else neg_idx)
    other_idx = np.asarray(neg_idx if anchor_label == 1 else pos_idx)
    anchor_name = "positive" if anchor_label == 1 else "negative"
    other_name = "negative" if anchor_label == 1 else "positive"

    if chunk_size_raw in {"", "0", "auto", "opposite", "full-opposite"}:
        chunk_size = len(other_idx)
        strategy = f"auto_{anchor_name}_chunks_against_sampled_{other_name}"
    else:
        chunk_size = int(chunk_size_raw)
        strategy = f"fixed_{anchor_name}_chunks_against_sampled_{other_name}"

    chunk_size = max(1, min(chunk_size, len(other_idx)))
    chunks = []
    for start in range(0, len(anchor_idx), chunk_size):
        anchor_chunk = anchor_idx[start:start + chunk_size]
        if len(anchor_chunk) == 0:
            continue
        if len(anchor_chunk) == len(other_idx):
            other_chunk = rng.permutation(other_idx)
        else:
            other_chunk = rng.choice(other_idx, size=len(anchor_chunk), replace=False)
        fold_idx = np.concatenate([anchor_chunk, other_chunk])
        chunks.append(fold_idx)

    return chunks, {
        "strategy": strategy,
        "anchor_label": anchor_label,
        "anchor_name": anchor_name,
        "other_name": other_name,
        "chunk_size": chunk_size,
    }


def _balanced_evaluation(scores, labels, optimize_for, random_seed=42):
    enabled = os.getenv("TUNING_BALANCED_EVALUATION", "1").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        threshold, best_metric = _find_best_threshold(scores, labels, optimize_for)
        metrics = _classification_metrics(labels, scores, threshold)
        return {
            "balanced_evaluation_enabled": False,
            "balanced_folds": 1,
            "best_threshold": float(threshold),
            "best_metric": float(best_metric),
            "metrics": metrics,
            "fold_metrics": [],
        }

    chunks, chunk_meta = _balanced_class_chunks(scores, labels, random_seed=random_seed)
    if not chunks:
        threshold, best_metric = _find_best_threshold(scores, labels, optimize_for)
        metrics = _classification_metrics(labels, scores, threshold)
        return {
            "balanced_evaluation_enabled": False,
            "balanced_folds": 1,
            "best_threshold": float(threshold),
            "best_metric": float(best_metric),
            "metrics": metrics,
            "fold_metrics": [],
        }

    fold_metrics = []
    thresholds = []
    best_metrics = []
    combined_labels = []
    combined_scores = []
    combined_preds = []
    for fold_number, fold_idx in enumerate(chunks, start=1):
        fold_scores = scores[fold_idx]
        fold_labels = labels[fold_idx]
        threshold, best_metric = _find_best_threshold(fold_scores, fold_labels, optimize_for)
        fold_preds = _scores_at_or_above_threshold(fold_scores, threshold).astype(int)
        metrics = _classification_metrics(fold_labels, fold_scores, threshold)
        metrics["fold"] = fold_number
        metrics["threshold"] = float(threshold)
        metrics["positive_pairs"] = int(np.sum(fold_labels == 1))
        metrics["negative_pairs"] = int(np.sum(fold_labels == 0))
        metrics["is_partial_positive_fold"] = bool(
            chunk_meta.get("anchor_label") == 1
            and np.sum(fold_labels == 1) < int(chunk_meta.get("chunk_size", 0))
        )
        metrics["is_partial_negative_fold"] = bool(
            chunk_meta.get("anchor_label") == 0
            and np.sum(fold_labels == 0) < int(chunk_meta.get("chunk_size", 0))
        )
        fold_metrics.append(metrics)
        thresholds.append(float(threshold))
        best_metrics.append(float(best_metric))
        combined_labels.append(fold_labels)
        combined_scores.append(fold_scores)
        combined_preds.append(fold_preds)

    combined_labels_arr = np.concatenate(combined_labels)
    combined_scores_arr = np.concatenate(combined_scores)
    combined_preds_arr = np.concatenate(combined_preds)
    summary_metrics = {
        "accuracy": float(accuracy_score(combined_labels_arr, combined_preds_arr)),
        "precision": float(precision_score(combined_labels_arr, combined_preds_arr, zero_division=0)),
        "recall": float(recall_score(combined_labels_arr, combined_preds_arr, zero_division=0)),
        "f1": float(f1_score(combined_labels_arr, combined_preds_arr, zero_division=0)),
        "auc": _safe_auc(combined_labels_arr, combined_scores_arr),
    }
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean, variance = _mean_std([fold[key] for fold in fold_metrics])
        summary_metrics[f"{key}_fold_mean"] = mean
        summary_metrics[f"{key}_variance"] = variance
    threshold_mean, threshold_variance = _mean_std([fold["threshold"] for fold in fold_metrics])
    summary_metrics["threshold"] = threshold_mean
    summary_metrics["threshold_variance"] = threshold_variance

    best_metric_mean, best_metric_variance = _mean_std(best_metrics)
    positives_per_fold = int(np.sum(labels[chunks[0]] == 1)) if chunks else 0
    negatives_per_fold = int(np.sum(labels[chunks[0]] == 0)) if chunks else 0
    total_positive_pairs = int(np.sum(labels == 1))
    total_negative_pairs = int(np.sum(labels == 0))
    positive_pair_evaluations = int(sum(int(fold["positive_pairs"]) for fold in fold_metrics))
    negative_pair_evaluations = int(sum(int(fold["negative_pairs"]) for fold in fold_metrics))
    used_idx = np.unique(np.concatenate(chunks))
    total_positive_pairs_used = int(np.sum(labels[used_idx] == 1))
    total_negative_pairs_used = int(np.sum(labels[used_idx] == 0))
    return {
        "balanced_evaluation_enabled": True,
        "balanced_folds": len(fold_metrics),
        "balanced_chunk_strategy": chunk_meta.get("strategy", ""),
        "balanced_chunk_anchor": chunk_meta.get("anchor_name", ""),
        "balanced_pair_chunk_size": int(chunk_meta.get("chunk_size", 0)),
        "balanced_positive_pairs_per_full_fold": positives_per_fold,
        "balanced_positive_pairs_used": total_positive_pairs_used,
        "balanced_positive_pairs_excluded": total_positive_pairs - total_positive_pairs_used,
        "balanced_positive_pair_evaluations": positive_pair_evaluations,
        "balanced_negative_pairs_per_fold": negatives_per_fold,
        "balanced_negative_pairs_used": total_negative_pairs_used,
        "balanced_negative_pairs_excluded": total_negative_pairs - total_negative_pairs_used,
        "balanced_negative_pair_evaluations": negative_pair_evaluations,
        "best_threshold": float(summary_metrics["threshold"]),
        "best_metric": float(summary_metrics.get(optimize_for, best_metric_mean)),
        "best_metric_fold_mean": float(best_metric_mean),
        "best_metric_variance": float(best_metric_variance),
        "metrics": summary_metrics,
        "fold_metrics": fold_metrics,
    }


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
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    labels = labels[finite_mask]
    if scores.size == 0:
        return 0.0, 0.0

    thresholds = _candidate_thresholds(scores)
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    positives_before = np.concatenate(([0], np.cumsum(sorted_labels == 1)))
    negatives_before = np.concatenate(([0], np.cumsum(sorted_labels == 0)))
    total_pos = int(positives_before[-1])
    total_neg = int(negatives_before[-1])

    starts = np.searchsorted(sorted_scores, np.nextafter(thresholds, -np.inf), side="left")
    tp = total_pos - positives_before[starts]
    fp = total_neg - negatives_before[starts]
    fn = total_pos - tp
    tn = total_neg - fp

    predicted_pos = tp + fp
    precision = np.divide(tp, predicted_pos, out=np.zeros_like(tp, dtype=np.float64), where=predicted_pos > 0)
    recall = np.divide(tp, total_pos, out=np.zeros_like(tp, dtype=np.float64), where=total_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    accuracy = (tp + tn) / max(1, labels.size)

    metric_values = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }[optimize_for]

    # Prefer the larger threshold when scores tie, matching the previous implementation.
    best_metric = np.max(metric_values)
    best_candidates = np.flatnonzero(metric_values == best_metric)
    best_idx = best_candidates[np.argmax(thresholds[best_candidates])]
    return float(thresholds[best_idx]), float(best_metric)


def _pair_method_ids(pairs):
    needed = set()
    for pair in pairs:
        needed.add(str(pair.left_id))
        needed.add(str(pair.right_id))
    return needed


def _has_feature_record(features_db, method_id) -> bool:
    return method_id in features_db or str(method_id) in features_db


def _get_feature_record(features_db, method_id):
    return features_db.get(method_id) or features_db.get(str(method_id))


def _has_informative_eigenvalues(record, graph_type, eps=1e-10) -> bool:
    if not record:
        return False

    values = record.get(graph_type, {}).get("eigenvalues", np.array([]))
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False

    return bool(np.any(np.abs(values) > eps))


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


def _filter_pairs_with_graph_features(pairs, features_db, graph_type, show_progress=False):
    filtered = []
    iterator = tqdm(
        pairs,
        desc=f"Filtering pairs for {graph_type.upper()} features",
        unit="pair",
        dynamic_ncols=True,
        leave=True,
    ) if show_progress else pairs
    for pair in iterator:
        left_data = _get_feature_record(features_db, pair.left_id)
        right_data = _get_feature_record(features_db, pair.right_id)
        if (
            _has_informative_eigenvalues(left_data, graph_type)
            and _has_informative_eigenvalues(right_data, graph_type)
        ):
            filtered.append(pair)

    dropped = len(pairs) - len(filtered)
    if dropped:
        print(f"[*] {graph_type.upper()}: dropped {dropped:,} pairs with missing/all-zero spectra.")

    labels = {pair.label for pair in filtered}
    if len(labels) < 2:
        print(f"[!] {graph_type.upper()}: skipped because valid pairs do not contain both labels.")
        return []

    return filtered


def _pair_scores_csv_path(features_db_path, out_filename):
    root = os.path.abspath(os.path.join(os.path.dirname(features_db_path), ".."))
    stem = os.path.splitext(os.path.basename(out_filename))[0]
    return os.path.join(root, f"pair_scores_{stem}.csv")


def _append_pair_scores_csv(path, pairs, scores, graph_type, k_eigen, metric, model_type="single"):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "left_id",
                "right_id",
                "label",
                "graph_type",
                "k_eigen",
                "metric",
                "score",
            ],
        )
        if not file_exists:
            writer.writeheader()
        k_label = "full" if k_eigen is None else k_eigen
        for pair, score in zip(pairs, scores):
            writer.writerow(
                {
                    "model_type": model_type,
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "label": pair.label,
                    "graph_type": graph_type,
                    "k_eigen": k_label,
                    "metric": metric,
                    "score": float(score),
                }
            )


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


def _load_combined_features_db(features_db_path, pairs):
    needed_ids = _pair_method_ids(pairs)
    features_db = _load_features_db(features_db_path, needed_ids=needed_ids)
    if features_db is None:
        return None

    extra_raw = os.getenv("TUNING_EXTRA_FEATURE_MANIFESTS", "").strip()
    if not extra_raw:
        return features_db

    for extra_path in [item.strip() for item in extra_raw.split(os.pathsep) if item.strip()]:
        missing_ids = needed_ids - set(features_db)
        if not missing_ids:
            break
        extra_db = _load_features_db(extra_path, needed_ids=missing_ids)
        if extra_db:
            features_db.update(extra_db)

    return features_db

def run_fast_grid_search_on_pairs(
        features_db_path,
        pairs: list[ClonePair],
        n_samples=1000,
        optimize_for="accuracy",
        out_filename="trained_models.json",
        graph_types=None,
        k_values=None,
        metrics=None,
        random_seed=42,
    ):
    if n_samples is not None and n_samples < len(pairs):
        rng = random.Random(random_seed)
        pairs = rng.sample(list(pairs), n_samples)

    print("[*] Loading precomputed lossless features for selected pairs...")
    features_db = _load_combined_features_db(features_db_path, pairs)
    if features_db is None:
        return None
    print(f"[*] Loaded spectral features for {len(features_db)} methods used by selected pairs.")
    pairs = _filter_pairs_with_features(pairs, features_db)

    if graph_types is None:
        graph_types = ["ast", "cfg", "ddg", "pdg", "cpg"] # 5 graphs
    if k_values is None:
        k_values = [25, 50, 100, None]                    # 4 K values (None = Full)
    if metrics is None:
        metrics = ["pss"]

    trained_models = []
    save_pair_scores = os.getenv("TUNING_SAVE_PAIR_SCORES", "1").strip().lower() not in {"0", "false", "no"}
    pair_scores_path = _pair_scores_csv_path(features_db_path, out_filename)
    if save_pair_scores and os.path.exists(pair_scores_path):
        os.remove(pair_scores_path)
    if save_pair_scores:
        print(f"[*] Pair-level metric scores will be saved to: {pair_scores_path}")
    
    total_base_combos = len(graph_types) * len(k_values) * len(metrics)
    print(f"[*] Starting Exact Continuous 'Training' Phase over {total_base_combos} core configurations...")
    graph_pairs_by_type = {}
    for gtype in tqdm(graph_types, desc="Preparing graph-specific pair pools", unit="graph", dynamic_ncols=True):
        graph_pairs_by_type[gtype] = _filter_pairs_with_graph_features(
            pairs,
            features_db,
            gtype,
            show_progress=True,
        )
    
    combo_iter = tqdm(
        list(product(graph_types, k_values, metrics)),
        desc="Tuning configs",
        unit="config",
        dynamic_ncols=True,
    )
    for gtype, k, metric in combo_iter:
        config_start = time.perf_counter()
        combo_iter.set_postfix(graph=gtype, k=("Full" if k is None else k), metric=metric)
        graph_pairs = graph_pairs_by_type[gtype]
        if not graph_pairs:
            continue

        # 1. Precalculate all scores for the current config just ONCE
        model = PrecomputedSpectralModel(features_db, gtype, k, None, metric)
        
        scores = []
        labels = []
        
        desc_inner = f"Scoring {gtype.upper()} K={k if k else 'Full'} {metric}"
        score_start = time.perf_counter()
        for pair in tqdm(graph_pairs, desc=desc_inner, unit="pair", leave=False, dynamic_ncols=True):
            scores.append(model.score_pair(pair))
            labels.append(pair.label)
        score_seconds = time.perf_counter() - score_start
            
        scores = np.array(scores)
        labels = np.array(labels)

        if save_pair_scores:
            _append_pair_scores_csv(pair_scores_path, graph_pairs, scores, gtype, k, metric)
        
        threshold_start = time.perf_counter()
        best_th, best_metric = _find_best_threshold(scores, labels, optimize_for)
        balanced_eval = _balanced_evaluation(scores, labels, optimize_for, random_seed=random_seed)
        report_th = balanced_eval["best_threshold"]
        report_metric = balanced_eval["best_metric"]
        threshold_seconds = time.perf_counter() - threshold_start
            
        print(
            f"[+] Trained: "
            f"Layer={gtype.upper()}, "
            f"K={k if k else 'Full'}, "
            f"Metric={metric} | "
            f"Thresh={report_th:.6f} -> "
            f"{optimize_for.upper()}={report_metric:.3f}"
        )

        full_metrics = _classification_metrics(labels, scores, best_th)
        metrics = balanced_eval["metrics"]

        trained_models.append({
            "optimized_for": optimize_for,
            "best_metric": float(report_metric),
            "graph_type": gtype,
            "k_eigen": k,
            "metric": metric,
            "metric_parameters": model.metric_parameters(),
            "best_threshold": float(report_th),
            "decision_threshold": float(_decision_threshold(report_th)),
            "valid_pairs": int(len(labels)),
            "dropped_pairs": int(len(pairs) - len(labels)),
            "positive_pairs": int(np.sum(labels == 1)),
            "negative_pairs": int(np.sum(labels == 0)),
            "balanced_evaluation_enabled": bool(balanced_eval["balanced_evaluation_enabled"]),
            "balanced_folds": int(balanced_eval["balanced_folds"]),
            "balanced_chunk_strategy": balanced_eval.get("balanced_chunk_strategy", ""),
            "balanced_chunk_anchor": balanced_eval.get("balanced_chunk_anchor", ""),
            "balanced_pair_chunk_size": int(balanced_eval.get("balanced_pair_chunk_size", 0)),
            "balanced_positive_pairs_per_full_fold": int(balanced_eval.get("balanced_positive_pairs_per_full_fold", np.sum(labels == 1))),
            "balanced_positive_pairs_used": int(balanced_eval.get("balanced_positive_pairs_used", np.sum(labels == 1))),
            "balanced_positive_pairs_excluded": int(balanced_eval.get("balanced_positive_pairs_excluded", 0)),
            "balanced_positive_pair_evaluations": int(balanced_eval.get("balanced_positive_pair_evaluations", np.sum(labels == 1))),
            "balanced_negative_pairs_per_fold": int(balanced_eval.get("balanced_negative_pairs_per_fold", np.sum(labels == 0))),
            "balanced_negative_pairs_used": int(balanced_eval.get("balanced_negative_pairs_used", np.sum(labels == 0))),
            "balanced_negative_pairs_excluded": int(balanced_eval.get("balanced_negative_pairs_excluded", 0)),
            "balanced_negative_pair_evaluations": int(balanced_eval.get("balanced_negative_pair_evaluations", np.sum(labels == 0))),
            "train_accuracy": float(metrics["accuracy"]),
            "train_precision": float(metrics["precision"]),
            "train_recall": float(metrics["recall"]),
            "train_f1": float(metrics["f1"]),
            "train_auc": float(metrics["auc"]),
            "train_accuracy_fold_mean": float(metrics.get("accuracy_fold_mean", metrics["accuracy"])),
            "train_precision_fold_mean": float(metrics.get("precision_fold_mean", metrics["precision"])),
            "train_recall_fold_mean": float(metrics.get("recall_fold_mean", metrics["recall"])),
            "train_f1_fold_mean": float(metrics.get("f1_fold_mean", metrics["f1"])),
            "train_auc_fold_mean": float(metrics.get("auc_fold_mean", metrics["auc"])),
            "train_accuracy_variance": float(metrics.get("accuracy_variance", 0.0)),
            "train_precision_variance": float(metrics.get("precision_variance", 0.0)),
            "train_recall_variance": float(metrics.get("recall_variance", 0.0)),
            "train_f1_variance": float(metrics.get("f1_variance", 0.0)),
            "train_auc_variance": float(metrics.get("auc_variance", 0.0)),
            "threshold_variance": float(metrics.get("threshold_variance", 0.0)),
            "best_metric_fold_mean": float(balanced_eval.get("best_metric_fold_mean", report_metric)),
            "best_metric_variance": float(balanced_eval.get("best_metric_variance", 0.0)),
            "full_train_threshold": float(best_th),
            "full_train_accuracy": float(full_metrics["accuracy"]),
            "full_train_precision": float(full_metrics["precision"]),
            "full_train_recall": float(full_metrics["recall"]),
            "full_train_f1": float(full_metrics["f1"]),
            "full_train_auc": float(full_metrics["auc"]),
            "balanced_fold_metrics": balanced_eval["fold_metrics"],
            "score_time_seconds": float(score_seconds),
            "threshold_time_seconds": float(threshold_seconds),
            "total_config_time_seconds": float(time.perf_counter() - config_start),
        })

    # Save to outputs
    out_path = os.path.join(os.path.dirname(features_db_path), "..", out_filename)
    with open(out_path, "w") as f:
        json.dump(trained_models, f, indent=4)
        
    print(f"\n[+] Grid Search Complete! {len(trained_models)} models trained and saved to: {os.path.abspath(out_path)}")
    if save_pair_scores:
        print(f"[+] Pair-level scores saved to: {pair_scores_path}")
    return trained_models


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
    pairs = loader.get_pairs("train")
    return run_fast_grid_search_on_pairs(
        features_db_path=features_db_path,
        pairs=pairs,
        n_samples=n_samples,
        optimize_for=optimize_for,
        out_filename=out_filename,
        graph_types=graph_types,
        k_values=k_values,
        metrics=metrics,
    )

class PrecomputedFusedSpectralModel:
    def __init__(self, features_db, primary_graph, secondary_graph, k_eigen, threshold, metric_name):
        self.model1 = PrecomputedSpectralModel(features_db, primary_graph, k_eigen, threshold, metric_name)
        self.model2 = PrecomputedSpectralModel(features_db, secondary_graph, k_eigen, threshold, metric_name)
        self.threshold = threshold

    def score_pair(self, pair):
        # Late Fusion Approach (Ensemble): Average the similarity scores calculated independently
        return (self.model1.score_pair(pair) + self.model2.score_pair(pair)) / 2.0

    def metric_parameters(self):
        return self.model1.metric_parameters()

    def predict(self, pair):
        return int(_score_at_or_above_threshold(self.score_pair(pair), self.threshold))

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
        metrics = ["pss"]

    trained_fused_models = []
    save_pair_scores = os.getenv("TUNING_SAVE_PAIR_SCORES", "1").strip().lower() not in {"0", "false", "no"}
    pair_scores_path = _pair_scores_csv_path(features_db_path, out_filename)
    if save_pair_scores and os.path.exists(pair_scores_path):
        os.remove(pair_scores_path)
    if save_pair_scores:
        print(f"[*] Pair-level fused metric scores will be saved to: {pair_scores_path}")
    
    total_base_combos = len(secondary_graphs) * len(k_values) * len(metrics)
    print(f"[*] Starting Exact Continuous Fused 'Training' Phase over {total_base_combos} core configurations...")
    primary_valid_pairs = _filter_pairs_with_graph_features(pairs, features_db, primary_graph, show_progress=True)
    secondary_valid_pairs = {}
    for gtype in tqdm(secondary_graphs, desc="Preparing fused pair pools", unit="graph", dynamic_ncols=True):
        secondary_valid_pairs[gtype] = _filter_pairs_with_graph_features(
            primary_valid_pairs,
            features_db,
            gtype,
            show_progress=True,
        )
    
    combo_iter = tqdm(
        list(product(secondary_graphs, k_values, metrics)),
        desc="Fused tuning configs",
        unit="config",
        dynamic_ncols=True,
    )
    for sec_gtype, k, metric in combo_iter:
        config_start = time.perf_counter()
        combo_iter.set_postfix(graph=f"{primary_graph}+{sec_gtype}", k=("Full" if k is None else k), metric=metric)
        fused_pairs = secondary_valid_pairs[sec_gtype]
        if not fused_pairs:
            continue

        fused_name = f"{primary_graph.upper()}+{sec_gtype.upper()}"
        model = PrecomputedFusedSpectralModel(features_db, primary_graph, sec_gtype, k, None, metric)
        
        scores = []
        labels = []
        
        desc_inner = f"Scoring {fused_name} K={k if k else 'Full'} {metric}"
        score_start = time.perf_counter()
        for pair in tqdm(fused_pairs, desc=desc_inner, unit="pair", leave=False, dynamic_ncols=True):
            scores.append(model.score_pair(pair))
            labels.append(pair.label)
        score_seconds = time.perf_counter() - score_start
            
        scores = np.array(scores)
        labels = np.array(labels)

        if save_pair_scores:
            _append_pair_scores_csv(pair_scores_path, fused_pairs, scores, fused_name, k, metric, model_type="fused")

        threshold_start = time.perf_counter()
        best_th, best_metric = _find_best_threshold(scores, labels, optimize_for)
        balanced_eval = _balanced_evaluation(scores, labels, optimize_for)
        report_th = balanced_eval["best_threshold"]
        report_metric = balanced_eval["best_metric"]
        threshold_seconds = time.perf_counter() - threshold_start
            
        print(
            f"[+] Trained: "
            f"Layer={fused_name}, "
            f"K={k if k else 'Full'}, "
            f"Metric={metric} | "
            f"Thresh={report_th:.6f} -> "
            f"{optimize_for.upper()}={report_metric:.3f}"
        )

        full_metrics = _classification_metrics(labels, scores, best_th)
        metrics = balanced_eval["metrics"]
        
        trained_fused_models.append({
            "optimized_for": optimize_for,
            "best_metric": float(report_metric),
            "graph_type": fused_name,
            "k_eigen": k,
            "metric": metric,
            "metric_parameters": model.metric_parameters(),
            "best_threshold": float(report_th),
            "decision_threshold": float(_decision_threshold(report_th)),
            "valid_pairs": int(len(labels)),
            "dropped_pairs": int(len(pairs) - len(labels)),
            "positive_pairs": int(np.sum(labels == 1)),
            "negative_pairs": int(np.sum(labels == 0)),
            "balanced_evaluation_enabled": bool(balanced_eval["balanced_evaluation_enabled"]),
            "balanced_folds": int(balanced_eval["balanced_folds"]),
            "balanced_chunk_strategy": balanced_eval.get("balanced_chunk_strategy", ""),
            "balanced_chunk_anchor": balanced_eval.get("balanced_chunk_anchor", ""),
            "balanced_pair_chunk_size": int(balanced_eval.get("balanced_pair_chunk_size", 0)),
            "balanced_positive_pairs_per_full_fold": int(balanced_eval.get("balanced_positive_pairs_per_full_fold", np.sum(labels == 1))),
            "balanced_positive_pairs_used": int(balanced_eval.get("balanced_positive_pairs_used", np.sum(labels == 1))),
            "balanced_positive_pairs_excluded": int(balanced_eval.get("balanced_positive_pairs_excluded", 0)),
            "balanced_positive_pair_evaluations": int(balanced_eval.get("balanced_positive_pair_evaluations", np.sum(labels == 1))),
            "balanced_negative_pairs_per_fold": int(balanced_eval.get("balanced_negative_pairs_per_fold", np.sum(labels == 0))),
            "balanced_negative_pairs_used": int(balanced_eval.get("balanced_negative_pairs_used", np.sum(labels == 0))),
            "balanced_negative_pairs_excluded": int(balanced_eval.get("balanced_negative_pairs_excluded", 0)),
            "balanced_negative_pair_evaluations": int(balanced_eval.get("balanced_negative_pair_evaluations", np.sum(labels == 0))),
            "train_accuracy": float(metrics["accuracy"]),
            "train_precision": float(metrics["precision"]),
            "train_recall": float(metrics["recall"]),
            "train_f1": float(metrics["f1"]),
            "train_auc": float(metrics["auc"]),
            "train_accuracy_fold_mean": float(metrics.get("accuracy_fold_mean", metrics["accuracy"])),
            "train_precision_fold_mean": float(metrics.get("precision_fold_mean", metrics["precision"])),
            "train_recall_fold_mean": float(metrics.get("recall_fold_mean", metrics["recall"])),
            "train_f1_fold_mean": float(metrics.get("f1_fold_mean", metrics["f1"])),
            "train_auc_fold_mean": float(metrics.get("auc_fold_mean", metrics["auc"])),
            "train_accuracy_variance": float(metrics.get("accuracy_variance", 0.0)),
            "train_precision_variance": float(metrics.get("precision_variance", 0.0)),
            "train_recall_variance": float(metrics.get("recall_variance", 0.0)),
            "train_f1_variance": float(metrics.get("f1_variance", 0.0)),
            "train_auc_variance": float(metrics.get("auc_variance", 0.0)),
            "threshold_variance": float(metrics.get("threshold_variance", 0.0)),
            "best_metric_fold_mean": float(balanced_eval.get("best_metric_fold_mean", report_metric)),
            "best_metric_variance": float(balanced_eval.get("best_metric_variance", 0.0)),
            "full_train_threshold": float(best_th),
            "full_train_accuracy": float(full_metrics["accuracy"]),
            "full_train_precision": float(full_metrics["precision"]),
            "full_train_recall": float(full_metrics["recall"]),
            "full_train_f1": float(full_metrics["f1"]),
            "full_train_auc": float(full_metrics["auc"]),
            "balanced_fold_metrics": balanced_eval["fold_metrics"],
            "score_time_seconds": float(score_seconds),
            "threshold_time_seconds": float(threshold_seconds),
            "total_config_time_seconds": float(time.perf_counter() - config_start),
        })

    # Save to outputs
    out_path = os.path.join(os.path.dirname(features_db_path), "..", out_filename)
    with open(out_path, "w") as f:
        json.dump(trained_fused_models, f, indent=4)
        
    print(f"\n[+] Fused Grid Search Complete! {len(trained_fused_models)} models trained and saved to: {os.path.abspath(out_path)}")
    if save_pair_scores:
        print(f"[+] Pair-level fused scores saved to: {pair_scores_path}")
    return trained_fused_models
