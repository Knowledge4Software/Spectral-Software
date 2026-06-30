from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.bcb_dataset import ClonePair
from spectral_code.evaluation.bcb_cross_type_generalization import (
    NON_CLONE_VARIANT,
    _features_manifest_for,
    _load_features_from_manifests,
    _load_non_clone_pairs,
    _load_positive_pairs,
    _metric_parameters,
    _parse_csv_values,
    _parse_k_values,
)
from spectral_code.evaluation.tuning import (
    PrecomputedSpectralModel,
    _classification_metrics,
    _decision_threshold,
    _find_best_threshold,
)
from spectral_code.utils.dataset_paths import OUTPUTS_ROOT
from spectral_code.utils.pipeline_timings import record_pipeline_timing

TRAIN_VARIANTS = ("1", "2", "3/all")
TYPE4_VARIANT = "4"
DEFAULT_GRAPH_TYPES = ["ast", "cpg"]
DEFAULT_METRICS = ["pss"]


def _finite_eigen_count(features_db: dict, method_id: str, graph_type: str) -> int:
    values = features_db.get(str(method_id), {}).get(graph_type, {}).get("eigenvalues", [])
    arr = np.asarray(values, dtype=np.float64).ravel()
    return int(np.isfinite(arr).sum())


def _pair_passes_graph_filter(pair: ClonePair, features_db: dict, graph_types: list[str], threshold: int, mode: str) -> bool:
    checks = []
    for graph_type in graph_types:
        left_count = _finite_eigen_count(features_db, pair.left_id, graph_type)
        right_count = _finite_eigen_count(features_db, pair.right_id, graph_type)
        checks.append(left_count >= threshold and right_count >= threshold)
    if mode == "all_graphs":
        return all(checks)
    if mode == "any_shared_graph":
        return any(checks)
    raise ValueError(f"Unknown graph filter mode: {mode}")


def _sample_without_replacement(pairs: list[ClonePair], target_size: int, rng: random.Random) -> list[ClonePair]:
    if target_size <= 0:
        return []
    if len(pairs) <= target_size:
        sampled = list(pairs)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(pairs, target_size)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_pairs(features_db: dict, pairs: list[ClonePair], graph_type: str, k_value: int | None, metric: str) -> tuple[np.ndarray, np.ndarray]:
    model = PrecomputedSpectralModel(features_db, graph_type, k_value, None, metric)
    k_label = "full" if k_value is None else str(k_value)
    iterator = tqdm(pairs, desc=f"Scoring {graph_type.upper()} k={k_label} {metric}", unit="pair", dynamic_ncols=True)
    scores = np.asarray([model.score_pair(pair) for pair in iterator], dtype=np.float64)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int8)
    return scores, labels


def run_balanced_type123_type4_threshold_experiment(
    *,
    graph_types: list[str] | None = None,
    metrics: list[str] | None = None,
    k_values: list[int | None] | None = None,
    eigen_threshold: int = 50,
    graph_filter_mode: str = "all_graphs",
    type4_ratio: float = 0.25,
    optimize_for: str = "f1",
    seed: int = 42,
) -> dict:
    start = time.perf_counter()
    graph_types = graph_types or list(DEFAULT_GRAPH_TYPES)
    metrics = metrics or list(DEFAULT_METRICS)
    k_values = k_values or [None]
    rng = random.Random(seed)

    output_root = OUTPUTS_ROOT / "bcb" / "type123_type4_balanced_threshold"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading BCB clone/non-clone pairs...")
    type123_positive_pairs = []
    type123_source_counts = {}
    for variant in TRAIN_VARIANTS:
        positives = _load_positive_pairs(variant)
        type123_source_counts[variant] = len(positives)
        type123_positive_pairs.extend(positives)
    type4_positive_pairs = _load_positive_pairs(TYPE4_VARIANT)
    non_clone_pairs = _load_non_clone_pairs()

    all_candidate_pairs = type123_positive_pairs + type4_positive_pairs + non_clone_pairs
    manifests = [_features_manifest_for(variant) for variant in [*TRAIN_VARIANTS, TYPE4_VARIANT, NON_CLONE_VARIANT]]
    print(f"[*] Loading spectral features for {len(all_candidate_pairs):,} candidate pairs...")
    features_db = _load_features_from_manifests(manifests, all_candidate_pairs)

    print(f"[*] Filtering pairs with {graph_filter_mode} and eigen_threshold={eigen_threshold} over {','.join(graph_types)}...")
    type123_filtered = [
        pair for pair in tqdm(type123_positive_pairs, desc="Filtering Type1+2+3", unit="pair", dynamic_ncols=True)
        if _pair_passes_graph_filter(pair, features_db, graph_types, eigen_threshold, graph_filter_mode)
    ]
    type4_pool = [
        pair for pair in tqdm(type4_positive_pairs, desc="Filtering Type4", unit="pair", dynamic_ncols=True)
        if _pair_passes_graph_filter(pair, features_db, graph_types, eigen_threshold, graph_filter_mode)
    ]
    non_clone_pool = [
        pair for pair in tqdm(non_clone_pairs, desc="Filtering non-clone", unit="pair", dynamic_ncols=True)
        if _pair_passes_graph_filter(pair, features_db, graph_types, eigen_threshold, graph_filter_mode)
    ]

    type4_target = int(round(len(type123_filtered) * type4_ratio))
    type4_sample = _sample_without_replacement(type4_pool, type4_target, rng)
    clone_pairs = list(type123_filtered) + type4_sample
    non_clone_target = len(clone_pairs)
    non_clone_sample = _sample_without_replacement(non_clone_pool, non_clone_target, rng)

    balanced_pairs = clone_pairs + non_clone_sample
    rng.shuffle(balanced_pairs)

    selection = {
        "type123_source_counts": type123_source_counts,
        "type123_positive_after_filter": len(type123_filtered),
        "type4_positive_after_filter": len(type4_pool),
        "type4_positive_target": type4_target,
        "type4_positive_sampled": len(type4_sample),
        "non_clone_after_filter": len(non_clone_pool),
        "non_clone_target": non_clone_target,
        "non_clone_sampled": len(non_clone_sample),
        "selected_positive_pairs": len(clone_pairs),
        "selected_negative_pairs": len(non_clone_sample),
        "selected_total_pairs": len(balanced_pairs),
        "class_counts": dict(Counter(pair.label for pair in balanced_pairs)),
    }
    print(
        "[*] Balanced threshold dataset: "
        f"type123={len(type123_filtered):,}, type4={len(type4_sample):,}/{type4_target:,}, "
        f"non_clone={len(non_clone_sample):,}/{non_clone_target:,}, total={len(balanced_pairs):,}."
    )

    rows = []
    score_rows = []
    for graph_type in graph_types:
        graph_pairs = [
            pair for pair in balanced_pairs
            if _pair_passes_graph_filter(pair, features_db, [graph_type], eigen_threshold, "all_graphs")
        ]
        if not graph_pairs:
            continue
        for k_value in k_values:
            for metric in metrics:
                config_start = time.perf_counter()
                scores, labels = _score_pairs(features_db, graph_pairs, graph_type, k_value, metric)
                threshold, best_metric = _find_best_threshold(scores, labels, optimize_for)
                metrics_row = _classification_metrics(labels, scores, threshold)
                k_label = "full" if k_value is None else k_value
                rows.append(
                    {
                        "optimized_for": optimize_for,
                        "graph_type": graph_type,
                        "metric": metric,
                        "metric_parameters": _metric_parameters(metric),
                        "k_eigen": k_label,
                        "best_threshold": float(threshold),
                        "decision_threshold": float(_decision_threshold(threshold)),
                        "best_metric": float(best_metric),
                        "pairs": int(len(labels)),
                        "positive_pairs": int(np.sum(labels == 1)),
                        "negative_pairs": int(np.sum(labels == 0)),
                        "accuracy": float(metrics_row["accuracy"]),
                        "precision": float(metrics_row["precision"]),
                        "recall": float(metrics_row["recall"]),
                        "f1": float(metrics_row["f1"]),
                        "auc": float(metrics_row["auc"]),
                        "config_seconds": float(time.perf_counter() - config_start),
                    }
                )
                for pair, score in zip(graph_pairs, scores):
                    score_rows.append(
                        {
                            "left_id": pair.left_id,
                            "right_id": pair.right_id,
                            "label": pair.label,
                            "graph_type": graph_type,
                            "metric": metric,
                            "k_eigen": k_label,
                            "score": float(score),
                        }
                    )

    rows.sort(key=lambda row: (row["accuracy"], row["f1"], row["auc"]), reverse=True)
    seconds = time.perf_counter() - start
    summary = {
        "experiment": "bcb_type123_type4_balanced_threshold",
        "train_variants": list(TRAIN_VARIANTS),
        "type4_variant": TYPE4_VARIANT,
        "non_clone_variant": NON_CLONE_VARIANT,
        "graph_types": graph_types,
        "metrics": metrics,
        "k_values": ["full" if value is None else value for value in k_values],
        "eigen_threshold": eigen_threshold,
        "graph_filter_mode": graph_filter_mode,
        "type4_ratio": type4_ratio,
        "optimize_for": optimize_for,
        "seed": seed,
        "selection": selection,
        "seconds": seconds,
        "results": rows,
    }

    summary_json = reports_dir / "balanced_type123_type4_threshold_summary.json"
    results_csv = reports_dir / "balanced_type123_type4_threshold_results.csv"
    scores_csv = reports_dir / "balanced_type123_type4_pair_scores.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(results_csv, rows)
    _write_csv(scores_csv, score_rows)
    record_pipeline_timing(
        output_root / "pipeline_timings.json",
        "00_train_balanced_type123_type4_threshold",
        seconds,
        {"summary_json": str(summary_json), "results_csv": str(results_csv), "scores_csv": str(scores_csv)},
    )

    print("[+] Balanced Type1+2+3 + Type4 threshold experiment finished.")
    print(f"    Summary: {summary_json}")
    print(f"    Results: {results_csv}")
    print(f"    Pair scores: {scores_csv}")
    if rows:
        best = rows[0]
        print(
            "    Best accuracy: "
            f"{best['accuracy']:.4f} ({best['graph_type']}/{best['metric']}/k={best['k_eigen']}, "
            f"threshold={best['best_threshold']:.6f})"
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune PSS thresholds on balanced BCB Type1+2+3 plus Type4 dataset.")
    parser.add_argument("--graph-types", default="ast,cpg")
    parser.add_argument("--metrics", default="pss")
    parser.add_argument("--k-values", default="full")
    parser.add_argument("--eigen-threshold", type=int, default=50)
    parser.add_argument("--graph-filter-mode", choices=["all_graphs", "any_shared_graph"], default="all_graphs")
    parser.add_argument("--type4-ratio", type=float, default=0.25)
    parser.add_argument("--optimize-for", choices=["accuracy", "precision", "recall", "f1"], default="f1")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_balanced_type123_type4_threshold_experiment(
        graph_types=_parse_csv_values(args.graph_types, DEFAULT_GRAPH_TYPES),
        metrics=_parse_csv_values(args.metrics, DEFAULT_METRICS),
        k_values=_parse_k_values(args.k_values),
        eigen_threshold=args.eigen_threshold,
        graph_filter_mode=args.graph_filter_mode,
        type4_ratio=args.type4_ratio,
        optimize_for=args.optimize_for,
        seed=args.seed,
    )
