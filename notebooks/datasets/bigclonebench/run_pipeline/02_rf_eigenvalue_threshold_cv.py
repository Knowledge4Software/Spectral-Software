from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectral_code.similarity.distribution import WassersteinSimilarity
from spectral_code.similarity.pss import PSSSimilarity
from spectral_code.utils.dataset_paths import OUTPUTS_ROOT, bcb_type_dir, output_root_for


GRAPH_TYPES = ["ast", "cpg"]
DEFAULT_THRESHOLDS = [50, 75, 100]
DEFAULT_VARIANTS = [
    ("type1", "1"),
    ("type2", "2"),
    ("type3_all", "3/all"),
    ("type4", "4"),
    ("non_clone", "non_clone"),
]
DEFAULT_PARTITION_MODES = ["per_graph", "any_shared_graph", "all_graphs"]


@dataclass(frozen=True)
class PairRecord:
    left_id: str
    right_id: str
    label: int
    source_label: str = ""
    source_variant: str = ""
    class_label: str = ""


@dataclass(frozen=True)
class PartitionSpec:
    mode: str
    threshold: int
    graph_type: str | None = None

    @property
    def name(self) -> str:
        graph_suffix = f"_{self.graph_type}" if self.graph_type else ""
        return f"{self.mode}{graph_suffix}_ge{self.threshold}"


def _parse_csv(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _parse_int_csv(raw: str | None, default: list[int]) -> list[int]:
    return [int(item) for item in _parse_csv(raw, [str(value) for value in default])]


def _parse_variants(raw: str | None) -> list[tuple[str, str]]:
    if raw is None or raw.strip() == "":
        return list(DEFAULT_VARIANTS)
    values = []
    known = {label: (label, variant) for label, variant in DEFAULT_VARIANTS}
    known.update({variant: (label, variant) for label, variant in DEFAULT_VARIANTS})
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" in token:
            label, variant = token.split("=", 1)
            values.append((label.strip(), variant.strip()))
        elif token in known:
            values.append(known[token])
        else:
            label = token.lower().replace("/", "_").replace("-", "_")
            values.append((label, token))
    return values


def _variant_pair_path(variant: str) -> Path:
    return bcb_type_dir(variant) / "train.txt"


def _variant_manifest_path(variant: str) -> Path:
    return output_root_for("bcb", variant) / "spectral_features" / "spectral_features_manifest.json"


def _class_label_for_variant(label: str, variant: str, pair_label: int) -> str:
    if pair_label == 0:
        return "non_clone"
    variant_key = variant.strip().lower().replace("-", "_")
    mapping = {
        "1": "type1",
        "2": "type2",
        "3/all": "type3_all",
        "3/moderate": "type3_moderate",
        "3/strong": "type3_strong",
        "3/very_strong": "type3_very_strong",
        "4": "type4",
    }
    return mapping.get(variant_key, label.strip().lower().replace("-", "_").replace("/", "_"))


def _load_pairs(path: Path, *, source_label: str = "", source_variant: str = "") -> list[PairRecord]:
    pairs: list[PairRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            label = int(parts[2])
            pairs.append(
                PairRecord(
                    str(parts[0]),
                    str(parts[1]),
                    label,
                    source_label=source_label,
                    source_variant=source_variant,
                    class_label=_class_label_for_variant(source_label, source_variant, label),
                )
            )
    return pairs


def _load_feature_manifest(manifest_path: Path) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "spectral_feature_shards_v1":
        raise ValueError(f"Unsupported spectral feature manifest: {manifest_path}")

    features: dict[str, dict] = {}
    for shard_path_raw in tqdm(manifest.get("shards", []), desc=f"Loading {manifest_path.parent.parent.name} features", unit="shard"):
        shard_path = Path(shard_path_raw)
        with shard_path.open("rb") as f:
            shard = pickle.load(f)
        features.update({str(method_id): value for method_id, value in shard.items()})
    return features


def _finite_eigenvalues(features_db: dict[str, dict], method_id: str, graph_type: str) -> np.ndarray:
    values = features_db.get(str(method_id), {}).get(graph_type, {}).get("eigenvalues", [])
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _build_count_lookup(features_db: dict[str, dict], graph_types: list[str]) -> dict[str, dict[str, int]]:
    lookup = {graph_type: {} for graph_type in graph_types}
    for method_id, layers in features_db.items():
        for graph_type in graph_types:
            values = layers.get(graph_type, {}).get("eigenvalues", []) if isinstance(layers, dict) else []
            arr = np.asarray(values, dtype=np.float64).ravel()
            lookup[graph_type][str(method_id)] = int(np.isfinite(arr).sum())
    return lookup


def _passes_graph(pair: PairRecord, count_lookup: dict[str, dict[str, int]], graph_type: str, threshold: int) -> bool:
    left_count = count_lookup.get(graph_type, {}).get(pair.left_id, 0)
    right_count = count_lookup.get(graph_type, {}).get(pair.right_id, 0)
    return left_count >= threshold and right_count >= threshold


def _partition_pairs(
    pairs: list[PairRecord],
    spec: PartitionSpec,
    count_lookup: dict[str, dict[str, int]],
    graph_types: list[str],
) -> list[PairRecord]:
    if spec.mode == "per_graph":
        if not spec.graph_type:
            raise ValueError("per_graph partitions require graph_type.")
        return [pair for pair in pairs if _passes_graph(pair, count_lookup, spec.graph_type, spec.threshold)]

    if spec.mode == "any_shared_graph":
        return [
            pair
            for pair in pairs
            if any(_passes_graph(pair, count_lookup, graph_type, spec.threshold) for graph_type in graph_types)
        ]

    if spec.mode == "all_graphs":
        return [
            pair
            for pair in pairs
            if all(_passes_graph(pair, count_lookup, graph_type, spec.threshold) for graph_type in graph_types)
        ]

    raise ValueError(f"Unknown partition mode: {spec.mode}")


def _passes_graph_mixed(
    pair: PairRecord,
    count_lookup_by_variant: dict[str, dict[str, dict[str, int]]],
    graph_type: str,
    threshold: int,
) -> bool:
    lookup = count_lookup_by_variant.get(pair.source_variant, {})
    left_count = lookup.get(graph_type, {}).get(pair.left_id, 0)
    right_count = lookup.get(graph_type, {}).get(pair.right_id, 0)
    return left_count >= threshold and right_count >= threshold


def _partition_pairs_mixed(
    pairs: list[PairRecord],
    spec: PartitionSpec,
    count_lookup_by_variant: dict[str, dict[str, dict[str, int]]],
    graph_types: list[str],
) -> list[PairRecord]:
    if spec.mode == "per_graph":
        if not spec.graph_type:
            raise ValueError("per_graph partitions require graph_type.")
        return [
            pair
            for pair in pairs
            if _passes_graph_mixed(pair, count_lookup_by_variant, spec.graph_type, spec.threshold)
        ]

    if spec.mode == "any_shared_graph":
        return [
            pair
            for pair in pairs
            if any(
                _passes_graph_mixed(pair, count_lookup_by_variant, graph_type, spec.threshold)
                for graph_type in graph_types
            )
        ]

    if spec.mode == "all_graphs":
        return [
            pair
            for pair in pairs
            if all(
                _passes_graph_mixed(pair, count_lookup_by_variant, graph_type, spec.threshold)
                for graph_type in graph_types
            )
        ]

    raise ValueError(f"Unknown partition mode: {spec.mode}")


def _passes_graphs_mixed(
    pair: PairRecord,
    count_lookup_by_variant: dict[str, dict[str, dict[str, int]]],
    graph_types: list[str],
    threshold: int,
    mode: str,
) -> bool:
    checks = [
        _passes_graph_mixed(pair, count_lookup_by_variant, graph_type, threshold)
        for graph_type in graph_types
    ]
    if mode == "all_graphs":
        return all(checks)
    if mode == "any_shared_graph":
        return any(checks)
    raise ValueError(f"Unsupported graph filter mode for balanced RF dataset: {mode}")


def _sample_without_replacement(
    pairs: list[PairRecord],
    target_size: int,
    rng: random.Random,
) -> list[PairRecord]:
    if target_size <= 0:
        return []
    if len(pairs) <= target_size:
        sampled = list(pairs)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(pairs, target_size)


def _build_type123_type4_balanced_dataset(
    all_pairs: list[PairRecord],
    count_lookup_by_variant: dict[str, dict[str, dict[str, int]]],
    *,
    graph_types: list[str],
    threshold: int,
    graph_filter_mode: str,
    type4_ratio: float,
    seed: int,
) -> tuple[list[PairRecord], dict]:
    rng = random.Random(seed)

    def passes(pair: PairRecord) -> bool:
        return _passes_graphs_mixed(pair, count_lookup_by_variant, graph_types, threshold, graph_filter_mode)

    type123_variants = {"1", "2", "3/all"}
    type123_positive = [
        pair for pair in all_pairs
        if pair.source_variant in type123_variants and pair.label == 1 and passes(pair)
    ]
    type4_positive_pool = [
        pair for pair in all_pairs
        if pair.source_variant == "4" and pair.label == 1 and passes(pair)
    ]
    non_clone_pool = [
        pair for pair in all_pairs
        if pair.source_variant == "non_clone" and pair.label == 0 and passes(pair)
    ]

    type4_target = int(round(len(type123_positive) * type4_ratio))
    type4_positive = _sample_without_replacement(type4_positive_pool, type4_target, rng)
    clone_pairs = list(type123_positive) + type4_positive
    non_clone_target = len(clone_pairs)
    non_clone_pairs = _sample_without_replacement(non_clone_pool, non_clone_target, rng)

    dataset_pairs = clone_pairs + non_clone_pairs
    rng.shuffle(dataset_pairs)

    report = {
        "partition": f"type123_plus_type4_{type4_ratio:g}_balanced_ge{threshold}_{graph_filter_mode}",
        "mode": "type123_type4_quarter_balanced",
        "graph_type": ",".join(graph_types),
        "threshold": threshold,
        "pairs": len(dataset_pairs),
        "positive_pairs": len(clone_pairs),
        "negative_pairs": len(non_clone_pairs),
        "type123_positive_pairs": len(type123_positive),
        "type4_positive_pool": len(type4_positive_pool),
        "type4_positive_target": type4_target,
        "type4_positive_sampled": len(type4_positive),
        "non_clone_pool": len(non_clone_pool),
        "non_clone_target": non_clone_target,
        "non_clone_sampled": len(non_clone_pairs),
        "class_counts": dict(Counter(pair.class_label for pair in dataset_pairs)),
    }
    return dataset_pairs, report


def _partition_specs(
    modes: list[str],
    thresholds: list[int],
    graph_types: list[str],
) -> list[PartitionSpec]:
    specs = []
    for threshold in thresholds:
        for mode in modes:
            if mode == "per_graph":
                for graph_type in graph_types:
                    specs.append(PartitionSpec(mode, threshold, graph_type))
            else:
                specs.append(PartitionSpec(mode, threshold))
    return specs


def _stratified_sample_pairs(pairs: list[PairRecord], max_pairs: int, seed: int) -> list[PairRecord]:
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs

    rng = random.Random(seed)
    by_label: dict[int, list[PairRecord]] = {}
    for pair in pairs:
        by_label.setdefault(pair.label, []).append(pair)

    sampled: list[PairRecord] = []
    remaining = max_pairs
    labels = sorted(by_label)
    for index, label in enumerate(labels):
        label_pairs = by_label[label]
        if index == len(labels) - 1:
            take = min(len(label_pairs), remaining)
        else:
            take = min(len(label_pairs), max(1, round(max_pairs * len(label_pairs) / len(pairs))))
        sampled.extend(rng.sample(label_pairs, take) if take < len(label_pairs) else label_pairs)
        remaining -= take

    if len(sampled) > max_pairs:
        sampled = rng.sample(sampled, max_pairs)
    rng.shuffle(sampled)
    return sampled


def _spectrum_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.max(values)),
        float(np.sum(values)),
    )


def _pair_features_for_graph(
    left_values: np.ndarray,
    right_values: np.ndarray,
    pss: PSSSimilarity,
    wasserstein: WassersteinSimilarity,
) -> list[float]:
    left_count = int(left_values.size)
    right_count = int(right_values.size)
    left_mean, left_std, left_max, left_sum = _spectrum_stats(left_values)
    right_mean, right_std, right_max, right_sum = _spectrum_stats(right_values)

    if left_count and right_count:
        pss_score = pss.compute(left_values, right_values)
        wasserstein_score = wasserstein.compute(left_values, right_values)
    else:
        pss_score = 0.0
        wasserstein_score = 0.0

    return [
        float(pss_score),
        float(wasserstein_score),
        float(left_count),
        float(right_count),
        float(min(left_count, right_count)),
        float(max(left_count, right_count)),
        float(abs(left_count - right_count)),
        abs(left_mean - right_mean),
        abs(left_std - right_std),
        abs(left_max - right_max),
        abs(left_sum - right_sum),
    ]


def _feature_names_for_graph(graph_type: str) -> list[str]:
    return [
        f"{graph_type}_pss",
        f"{graph_type}_wasserstein",
        f"{graph_type}_left_count",
        f"{graph_type}_right_count",
        f"{graph_type}_min_count",
        f"{graph_type}_max_count",
        f"{graph_type}_count_abs_diff",
        f"{graph_type}_mean_abs_diff",
        f"{graph_type}_std_abs_diff",
        f"{graph_type}_max_abs_diff",
        f"{graph_type}_sum_abs_diff",
    ]


def _build_feature_matrix(
    pairs: list[PairRecord],
    features_db: dict[str, dict],
    feature_graphs: list[str],
    *,
    pss_gamma: float,
    pss_distance_power: float,
    wasserstein_gamma: float,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    pss = PSSSimilarity(gamma=pss_gamma, distance_power=pss_distance_power)
    wasserstein = WassersteinSimilarity(gamma=wasserstein_gamma)
    feature_names = [name for graph_type in feature_graphs for name in _feature_names_for_graph(graph_type)]

    rows: list[list[float]] = []
    labels: list[int] = []
    skipped = 0

    for pair in tqdm(pairs, desc="Building RF features", unit="pair", leave=False):
        if pair.left_id not in features_db or pair.right_id not in features_db:
            skipped += 1
            continue

        row: list[float] = []
        for graph_type in feature_graphs:
            left_values = _finite_eigenvalues(features_db, pair.left_id, graph_type)
            right_values = _finite_eigenvalues(features_db, pair.right_id, graph_type)
            row.extend(_pair_features_for_graph(left_values, right_values, pss, wasserstein))
        rows.append(row)
        labels.append(pair.label)

    if not rows:
        return np.empty((0, len(feature_names)), dtype=np.float32), np.empty((0,), dtype=np.int8), feature_names, skipped

    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int8), feature_names, skipped


def _build_feature_matrix_mixed(
    pairs: list[PairRecord],
    features_by_variant: dict[str, dict[str, dict]],
    feature_graphs: list[str],
    *,
    target: str,
    pss_gamma: float,
    pss_distance_power: float,
    wasserstein_gamma: float,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    pss = PSSSimilarity(gamma=pss_gamma, distance_power=pss_distance_power)
    wasserstein = WassersteinSimilarity(gamma=wasserstein_gamma)
    feature_names = [name for graph_type in feature_graphs for name in _feature_names_for_graph(graph_type)]

    rows: list[list[float]] = []
    labels: list[int | str] = []
    skipped = 0

    for pair in tqdm(pairs, desc=f"Building {target} RF features", unit="pair", leave=False):
        features_db = features_by_variant.get(pair.source_variant, {})
        if pair.left_id not in features_db or pair.right_id not in features_db:
            skipped += 1
            continue

        row: list[float] = []
        for graph_type in feature_graphs:
            left_values = _finite_eigenvalues(features_db, pair.left_id, graph_type)
            right_values = _finite_eigenvalues(features_db, pair.right_id, graph_type)
            row.extend(_pair_features_for_graph(left_values, right_values, pss, wasserstein))
        rows.append(row)
        labels.append(pair.label if target == "binary" else pair.class_label)

    if not rows:
        return np.empty((0, len(feature_names)), dtype=np.float32), np.empty((0,), dtype=object), feature_names, skipped

    return np.asarray(rows, dtype=np.float32), np.asarray(labels), feature_names, skipped


def _confusion_counts(labels: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    if not set(np.unique(labels).tolist()) <= {0, 1}:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    return {
        "tp": int(np.sum((labels == 1) & (preds == 1))),
        "fp": int(np.sum((labels == 0) & (preds == 1))),
        "tn": int(np.sum((labels == 0) & (preds == 0))),
        "fn": int(np.sum((labels == 1) & (preds == 0))),
    }


def _metrics_from_predictions(
    labels: np.ndarray,
    preds: np.ndarray,
    proba: np.ndarray | None,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    unique_labels = set(np.unique(labels).tolist())
    average = "binary" if unique_labels <= {0, 1} else "macro"
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average=average, zero_division=0)),
        "recall": float(recall_score(labels, preds, average=average, zero_division=0)),
        "f1": float(f1_score(labels, preds, average=average, zero_division=0)),
        "auc": 0.0,
    }
    if proba is None or len(unique_labels) < 2:
        return metrics

    try:
        if unique_labels <= {0, 1}:
            scores = proba[:, 1] if proba.ndim == 2 else proba
            metrics["auc"] = float(roc_auc_score(labels, scores))
        elif classes is not None and proba.ndim == 2 and len(classes) == proba.shape[1]:
            metrics["auc"] = float(
                roc_auc_score(
                    labels,
                    proba,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


def _run_random_forest_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    seed: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    n_jobs: int,
) -> tuple[dict, list[dict], list[dict]]:
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Random Forest CV requires both positive and negative labels.")
    effective_folds = min(folds, int(np.min(counts)))
    if effective_folds < 2:
        raise ValueError("Random Forest CV requires at least two examples per class.")

    cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    fold_rows = []
    aggregate = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    importance_rows: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed + fold,
            n_jobs=n_jobs,
            class_weight="balanced",
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx]) if hasattr(clf, "predict_proba") else None
        metrics = _metrics_from_predictions(y[test_idx], preds, proba, clf.classes_ if proba is not None else None)
        confusion = _confusion_counts(y[test_idx], preds)
        for key in aggregate:
            aggregate[key] += confusion[key]

        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "positive_test_rows": int(np.sum(y[test_idx] == 1)),
                "negative_test_rows": int(np.sum(y[test_idx] == 0)),
                **confusion,
                **metrics,
            }
        )
        all_labels.append(y[test_idx])
        all_preds.append(preds)
        if proba is not None:
            all_proba.append(proba)
        for feature_index, importance in enumerate(clf.feature_importances_):
            importance_rows.append(
                {
                    "fold": fold,
                    "feature_index": int(feature_index),
                    "importance": float(importance),
                }
            )

    labels_all = np.concatenate(all_labels)
    preds_all = np.concatenate(all_preds)
    proba_all = np.concatenate(all_proba) if all_proba else None
    aggregate_metrics = _metrics_from_predictions(labels_all, preds_all, proba_all, labels)

    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    summary = {
        "effective_folds": int(effective_folds),
        "rows": int(len(y)),
        "positive_rows": int(np.sum(y == 1)),
        "negative_rows": int(np.sum(y == 0)),
        **aggregate,
        **{f"{name}_from_oof_predictions": float(aggregate_metrics[name]) for name in metric_names},
    }
    for name in metric_names:
        values = [float(row[name]) for row in fold_rows]
        summary[f"{name}_fold_mean"] = float(np.mean(values))
        summary[f"{name}_fold_std"] = float(np.std(values))
    return summary, fold_rows, importance_rows


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    variants = _parse_variants(args.variants)
    graph_types = _parse_csv(args.graph_types, GRAPH_TYPES)
    thresholds = _parse_int_csv(args.thresholds, DEFAULT_THRESHOLDS)
    modes = _parse_csv(args.partition_modes, DEFAULT_PARTITION_MODES)
    tasks = _parse_csv(args.tasks, ["binary", "multiclass"])
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_ROOT / "bcb" / "rf_eigenvalue_threshold_cv"
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _partition_specs(modes, thresholds, graph_types)
    results: list[dict] = []
    fold_results: list[dict] = []
    importance_results: list[dict] = []
    partition_counts: list[dict] = []

    all_pairs: list[PairRecord] = []
    features_by_variant: dict[str, dict[str, dict]] = {}
    count_lookup_by_variant: dict[str, dict[str, dict[str, int]]] = {}

    for label, variant in variants:
        pair_path = _variant_pair_path(variant)
        manifest_path = _variant_manifest_path(variant)
        if not pair_path.exists():
            raise FileNotFoundError(f"Missing pair file for {label}: {pair_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing spectral manifest for {label}: {manifest_path}")

        print(f"[*] Loading {label} pairs: {pair_path}")
        pairs = _load_pairs(pair_path, source_label=label, source_variant=variant)
        all_pairs.extend(pairs)
        print(f"[*] Loading {label} spectral features: {manifest_path}")
        features_db = _load_feature_manifest(manifest_path)
        features_by_variant[variant] = features_db
        count_lookup_by_variant[variant] = _build_count_lookup(features_db, graph_types)

    print(
        f"[*] Combined BCB pair pool: {len(all_pairs):,} rows "
        f"({sum(pair.label == 1 for pair in all_pairs):,} clone / "
        f"{sum(pair.label == 0 for pair in all_pairs):,} non-clone)."
    )

    def evaluate_partition(
        partition_pairs: list[PairRecord],
        *,
        partition_name: str,
        mode: str,
        graph_type_label: str,
        threshold: int,
        feature_graphs: list[str],
        count_row: dict | None = None,
        apply_sample_cap: bool = False,
    ) -> None:
        binary_label_counts = Counter(pair.label for pair in partition_pairs)
        class_label_counts = Counter(pair.class_label for pair in partition_pairs)
        if count_row is None:
            count_row = {
                "partition": partition_name,
                "mode": mode,
                "graph_type": graph_type_label,
                "threshold": threshold,
                "pairs": len(partition_pairs),
                "positive_pairs": binary_label_counts.get(1, 0),
                "negative_pairs": binary_label_counts.get(0, 0),
                "class_counts": dict(class_label_counts),
            }
        partition_counts.append(count_row)

        print(
            f"[*] RF dataset {partition_name}: {len(partition_pairs):,} pairs "
            f"({binary_label_counts.get(1, 0):,} positive / {binary_label_counts.get(0, 0):,} negative); "
            f"features={','.join(feature_graphs)}"
        )

        for task in tasks:
            print(f"    [*] Task {task}: preparing {partition_name}...", flush=True)
            target_counts = binary_label_counts if task == "binary" else class_label_counts
            if len(partition_pairs) < args.min_pairs:
                results.append(
                    {
                        "task": task,
                        "partition": partition_name,
                        "mode": mode,
                        "graph_type": graph_type_label,
                        "threshold": threshold,
                        "status": "skipped",
                        "reason": f"fewer_than_min_pairs_{args.min_pairs}",
                        "pairs_before_sampling": len(partition_pairs),
                    }
                )
                continue
            if len(target_counts) < 2:
                results.append(
                    {
                        "task": task,
                        "partition": partition_name,
                        "mode": mode,
                        "graph_type": graph_type_label,
                        "threshold": threshold,
                        "status": "skipped",
                        "reason": "requires_at_least_two_target_classes",
                        "pairs_before_sampling": len(partition_pairs),
                        "target_counts": dict(target_counts),
                    }
                )
                continue

            sampled_pairs = (
                _stratified_sample_pairs(partition_pairs, args.max_pairs_per_partition, args.seed)
                if apply_sample_cap
                else list(partition_pairs)
            )
            sample_label = (
                "full balanced dataset"
                if not apply_sample_cap or args.max_pairs_per_partition <= 0
                else f"sample cap {args.max_pairs_per_partition:,}"
            )
            print(
                f"    [*] Task {task}: building features for {len(sampled_pairs):,} pairs "
                f"({sample_label}; graphs={','.join(feature_graphs)})...",
                flush=True,
            )
            X, y, feature_names, skipped_features = _build_feature_matrix_mixed(
                sampled_pairs,
                features_by_variant,
                feature_graphs,
                target=task,
                pss_gamma=args.pss_gamma,
                pss_distance_power=args.pss_distance_power,
                wasserstein_gamma=args.wasserstein_gamma,
            )
            print(
                f"    [*] Task {task}: feature matrix {X.shape[0]:,} x {X.shape[1]:,}; "
                f"skipped {skipped_features:,} pairs.",
                flush=True,
            )

            if X.shape[0] < args.min_pairs or len(np.unique(y)) < 2:
                results.append(
                    {
                        "task": task,
                        "partition": partition_name,
                        "mode": mode,
                        "graph_type": graph_type_label,
                        "threshold": threshold,
                        "status": "skipped",
                        "reason": "insufficient_rows_after_feature_build",
                        "pairs_before_sampling": len(partition_pairs),
                        "pairs_after_sampling": len(sampled_pairs),
                        "feature_rows": int(X.shape[0]),
                        "skipped_feature_pairs": int(skipped_features),
                    }
                )
                continue

            try:
                print(f"    [*] Task {task}: running {args.cv_folds}-fold Random Forest CV...", flush=True)
                cv_summary, fold_rows, importance_rows = _run_random_forest_cv(
                    X,
                    y,
                    folds=args.cv_folds,
                    seed=args.seed,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    n_jobs=args.n_jobs,
                )
                status = "ok"
                reason = ""
                print(
                    f"    [+] Task {task}: done, OOF F1={cv_summary.get('f1_from_oof_predictions', 0.0):.4f}, "
                    f"accuracy={cv_summary.get('accuracy_from_oof_predictions', 0.0):.4f}.",
                    flush=True,
                )
            except ValueError as exc:
                cv_summary = {}
                fold_rows = []
                importance_rows = []
                status = "skipped"
                reason = str(exc)

            result_row = {
                "task": task,
                "partition": partition_name,
                "mode": mode,
                "graph_type": graph_type_label,
                "threshold": threshold,
                "feature_graphs": ",".join(feature_graphs),
                "feature_count": len(feature_names),
                "status": status,
                "reason": reason,
                "pairs_before_sampling": len(partition_pairs),
                "pairs_after_sampling": len(sampled_pairs),
                "feature_rows": int(X.shape[0]),
                "target_counts": dict(Counter(y.tolist())),
                "skipped_feature_pairs": int(skipped_features),
                **cv_summary,
            }
            results.append(result_row)
            for fold_row in fold_rows:
                fold_results.append({**result_row, **fold_row})
            for importance_row in importance_rows:
                feature_index = int(importance_row["feature_index"])
                importance_results.append(
                    {
                        **result_row,
                        "fold": importance_row["fold"],
                        "feature": feature_names[feature_index],
                        "importance": importance_row["importance"],
                    }
                )

    if args.sampling_scheme == "type123_type4_quarter_balanced":
        threshold = int(args.eigen_threshold)
        dataset_pairs, dataset_report = _build_type123_type4_balanced_dataset(
            all_pairs,
            count_lookup_by_variant,
            graph_types=graph_types,
            threshold=threshold,
            graph_filter_mode=args.graph_filter_mode,
            type4_ratio=args.type4_ratio,
            seed=args.seed,
        )
        print(
            "[*] Built balanced RF dataset: "
            f"type123={dataset_report['type123_positive_pairs']:,}, "
            f"type4={dataset_report['type4_positive_sampled']:,}/{dataset_report['type4_positive_target']:,}, "
            f"non_clone={dataset_report['non_clone_sampled']:,}/{dataset_report['non_clone_target']:,}, "
            f"total={dataset_report['pairs']:,}."
        )
        evaluate_partition(
            dataset_pairs,
            partition_name=dataset_report["partition"],
            mode=dataset_report["mode"],
            graph_type_label=dataset_report["graph_type"],
            threshold=threshold,
            feature_graphs=graph_types,
            count_row=dataset_report,
            apply_sample_cap=False,
        )
    elif args.sampling_scheme == "legacy_partitions":
        for spec in specs:
            partition_pairs = _partition_pairs_mixed(all_pairs, spec, count_lookup_by_variant, graph_types)
            feature_graphs = [spec.graph_type] if spec.mode == "per_graph" and spec.graph_type else graph_types
            evaluate_partition(
                partition_pairs,
                partition_name=spec.name,
                mode=spec.mode,
                graph_type_label=spec.graph_type or "",
                threshold=spec.threshold,
                feature_graphs=feature_graphs,
                apply_sample_cap=True,
            )
    else:
        raise ValueError(f"Unknown sampling scheme: {args.sampling_scheme}")

    results_df = pd.DataFrame(results)
    folds_df = pd.DataFrame(fold_results)
    importance_df = pd.DataFrame(importance_results)
    counts_df = pd.DataFrame(partition_counts)
    results_csv = output_dir / "rf_eigenvalue_threshold_cv_results.csv"
    folds_csv = output_dir / "rf_eigenvalue_threshold_cv_folds.csv"
    importance_csv = output_dir / "rf_eigenvalue_threshold_feature_importances.csv"
    counts_csv = output_dir / "rf_eigenvalue_threshold_partition_counts.csv"
    summary_json = output_dir / "rf_eigenvalue_threshold_cv_summary.json"
    results_df.to_csv(results_csv, index=False)
    folds_df.to_csv(folds_csv, index=False)
    importance_df.to_csv(importance_csv, index=False)
    counts_df.to_csv(counts_csv, index=False)

    summary = {
        "seconds": time.perf_counter() - start,
        "schema_version": 2,
        "experiment": "bcb_rf_eigenvalue_threshold_cv",
        "sampling_scheme": args.sampling_scheme,
        "eigen_threshold": args.eigen_threshold,
        "graph_filter_mode": args.graph_filter_mode,
        "type4_ratio": args.type4_ratio,
        "variants": [{"label": label, "variant": variant} for label, variant in variants],
        "graph_types": graph_types,
        "thresholds": thresholds,
        "partition_modes": modes,
        "tasks": tasks,
        "cv_folds": args.cv_folds,
        "max_pairs_per_partition": args.max_pairs_per_partition,
        "min_pairs": args.min_pairs,
        "random_forest": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",
            "seed": args.seed,
        },
        "metric_parameters": {
            "pss_gamma": args.pss_gamma,
            "pss_distance_power": args.pss_distance_power,
            "wasserstein_gamma": args.wasserstein_gamma,
        },
        "outputs": {
            "results_csv": str(results_csv),
            "folds_csv": str(folds_csv),
            "feature_importances_csv": str(importance_csv),
            "partition_counts_csv": str(counts_csv),
        },
        "results": results,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[+] Random Forest eigenvalue-threshold CV complete.")
    print(f"    Results: {results_csv}")
    print(f"    Folds:   {folds_csv}")
    print(f"    Importances: {importance_csv}")
    print(f"    Counts:  {counts_csv}")
    print(f"    Summary: {summary_json}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Partition BCB pairs by minimum eigenvalue-count thresholds and evaluate "
            "Random Forest classifiers with stratified cross-validation."
        )
    )
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Comma-separated variants. Use labels/variants like type1,3/all,non_clone "
            "or custom label=variant. Default: all BCB types plus non_clone."
        ),
    )
    parser.add_argument(
        "--sampling-scheme",
        default="type123_type4_quarter_balanced",
        choices=["type123_type4_quarter_balanced", "legacy_partitions"],
        help=(
            "Default builds one RF dataset from all Type1+Type2+Type3 clone pairs passing the "
            "eigenvalue threshold, adds Type4 clones at --type4-ratio of that count, then adds "
            "an equal number of non-clone pairs. Use legacy_partitions for the older threshold partition sweep."
        ),
    )
    parser.add_argument("--eigen-threshold", type=int, default=50)
    parser.add_argument("--type4-ratio", type=float, default=0.25)
    parser.add_argument(
        "--graph-filter-mode",
        default="all_graphs",
        choices=["all_graphs", "any_shared_graph"],
        help="How the eigen-threshold filter is applied across --graph-types for the balanced RF dataset.",
    )
    parser.add_argument("--thresholds", default="50")
    parser.add_argument("--partition-modes", default="per_graph,any_shared_graph,all_graphs")
    parser.add_argument("--tasks", default="binary,multiclass")
    parser.add_argument(
        "--graph-types",
        default="ast,cpg",
        help="Comma-separated graph types to use for partitions and RF features. Default focuses on AST and CPG.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--max-pairs-per-partition",
        type=int,
        default=50_000,
        help="Stratified sample cap per partition. Use 0 for full partition. Default: 50000.",
    )
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pss-gamma", type=float, default=0.1)
    parser.add_argument("--pss-distance-power", type=float, default=1.0)
    parser.add_argument("--wasserstein-gamma", type=float, default=0.1)
    parser.add_argument("--output-dir", default="")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
