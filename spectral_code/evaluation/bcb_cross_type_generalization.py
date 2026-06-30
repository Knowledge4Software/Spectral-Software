from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from spectral_code.evaluation.bcb_dataset import BigCloneBenchLoader, ClonePair
from spectral_code.evaluation.pipeline_section_runner import _select_balanced_bcb_tuning_pairs
from spectral_code.evaluation.tuning import (
    PrecomputedSpectralModel,
    _classification_metrics,
    _decision_threshold,
    _filter_pairs_with_features,
    _filter_pairs_with_graph_features,
    _find_best_threshold,
    _load_features_db,
)
from spectral_code.utils.dataset_paths import OUTPUTS_ROOT, bcb_type_dir, output_root_for
from spectral_code.utils.pipeline_timings import record_pipeline_timing


TRAIN_VARIANTS = ("1", "2", "3/all")
TEST_VARIANT = "4"
NON_CLONE_VARIANT = "non_clone"


def _parse_k_values(raw: str | None) -> list[int | None]:
    if raw is None or raw.strip() == "":
        return [None]
    values: list[int | None] = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        values.append(None if value in {"full", "all", "none"} else int(value))
    return values or [None]


def _parse_csv_values(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or raw.strip() == "":
        return default
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return values or default


def _variant_slug(variant: str) -> str:
    return str(variant).strip().lower().replace("-", "_").replace("/", "_")


def _features_manifest_for(variant: str) -> Path:
    return output_root_for("bcb", variant) / "spectral_features" / "spectral_features_manifest.json"


def _load_positive_pairs(variant: str) -> list[ClonePair]:
    data_dir = bcb_type_dir(variant)
    loader = BigCloneBenchLoader(data_dir)
    return [pair for pair in loader.get_pairs("train") if pair.label == 1]


def _load_non_clone_pairs() -> list[ClonePair]:
    data_dir = bcb_type_dir(NON_CLONE_VARIANT)
    loader = BigCloneBenchLoader(data_dir)
    return [pair for pair in loader.get_pairs("train") if pair.label == 0]


def _balanced_clone_vs_nonclone_pairs(
    positive_pairs: list[ClonePair],
    non_clone_pairs: list[ClonePair],
    *,
    spec_key: str,
    seed: int,
) -> tuple[list[ClonePair], dict]:
    return _select_balanced_bcb_tuning_pairs(
        [*positive_pairs, *non_clone_pairs],
        spec_key=spec_key,
        seed=seed,
    )


def _load_features_from_manifests(manifests: list[Path], pairs: list[ClonePair]) -> dict:
    needed_ids = {str(pair.left_id) for pair in pairs} | {str(pair.right_id) for pair in pairs}
    features_db = {}
    for manifest in manifests:
        if not manifest.exists():
            raise FileNotFoundError(f"Spectral features manifest is missing: {manifest}")
        missing_ids = needed_ids - {str(method_id) for method_id in features_db}
        if not missing_ids:
            break
        shard_db = _load_features_db(str(manifest), needed_ids=missing_ids)
        if shard_db:
            features_db.update({str(method_id): features for method_id, features in shard_db.items()})

    missing_after = needed_ids - set(features_db)
    if missing_after:
        print(f"[*] Missing spectral features for {len(missing_after):,} methods after loading all manifests.")
    return features_db


def _prepare_scored_pool(
    *,
    name: str,
    pairs: list[ClonePair],
    manifests: list[Path],
    graph_types: list[str],
) -> tuple[dict, dict[str, list[ClonePair]], list[ClonePair]]:
    print(f"[*] Loading {name} spectral features from {len(manifests)} manifest(s)...")
    features_db = _load_features_from_manifests(manifests, pairs)
    pairs_with_features = _filter_pairs_with_features(pairs, features_db)
    graph_pairs_by_type = {}
    for graph_type in graph_types:
        print(f"[*] {name}: filtering {len(pairs_with_features):,} pairs for {graph_type.upper()} features...")
        graph_pairs_by_type[graph_type] = _filter_pairs_with_graph_features(
            pairs_with_features,
            features_db,
            graph_type,
            show_progress=True,
        )
    return features_db, graph_pairs_by_type, pairs_with_features


def _score_pairs(
    features_db: dict,
    pairs: list[ClonePair],
    graph_type: str,
    k_value: int | None,
    metric: str,
    *,
    label: str,
):
    model = PrecomputedSpectralModel(features_db, graph_type, k_value, None, metric)
    k_label = "full" if k_value is None else str(k_value)
    iterator = tqdm(
        pairs,
        desc=f"Scoring {label} {graph_type.upper()} k={k_label} {metric}",
        unit="pair",
        dynamic_ncols=True,
        leave=True,
    )
    scores = np.asarray([model.score_pair(pair) for pair in iterator], dtype=np.float64)
    labels = np.asarray([pair.label for pair in pairs], dtype=np.int8)
    return scores, labels


def _metric_parameters(metric: str) -> dict:
    return PrecomputedSpectralModel({}, "ast", None, None, metric).metric_parameters()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_bcb_type123_to_type4_threshold_experiment(
    *,
    graph_types: list[str] | None = None,
    metrics: list[str] | None = None,
    k_values: list[int | None] | None = None,
    optimize_for: str = "f1",
    seed: int = 42,
) -> dict:
    start = time.perf_counter()
    graph_types = graph_types or ["ast", "cfg", "ddg", "pdg", "cpg"]
    metrics = metrics or ["pss"]
    k_values = k_values or [None]

    output_root = OUTPUTS_ROOT / "bcb" / "type123_to_type4"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading BCB clone/non-clone pairs...")
    train_positive_pairs = []
    train_source_counts = {}
    for variant in TRAIN_VARIANTS:
        positives = _load_positive_pairs(variant)
        train_source_counts[variant] = len(positives)
        train_positive_pairs.extend(positives)
    test_positive_pairs = _load_positive_pairs(TEST_VARIANT)
    non_clone_pairs = _load_non_clone_pairs()

    train_pairs, train_selection = _balanced_clone_vs_nonclone_pairs(
        train_positive_pairs,
        non_clone_pairs,
        spec_key="bcb_type123_union",
        seed=seed,
    )
    test_pairs, test_selection = _balanced_clone_vs_nonclone_pairs(
        test_positive_pairs,
        non_clone_pairs,
        spec_key="bcb_type4_holdout",
        seed=seed + 1,
    )

    print(
        "[*] Train pair selection: "
        f"{train_selection['selected_positive_pairs']:,} clone pairs + "
        f"{train_selection['selected_negative_pairs']:,} non-clone evaluations "
        f"({train_selection.get('unique_selected_negative_pairs', 0):,} unique) "
        f"across {train_selection['chunk_count']:,} chunk(s)."
    )
    print(
        "[*] Test pair selection: "
        f"{test_selection['selected_positive_pairs']:,} clone pairs + "
        f"{test_selection['selected_negative_pairs']:,} non-clone evaluations "
        f"({test_selection.get('unique_selected_negative_pairs', 0):,} unique) "
        f"across {test_selection['chunk_count']:,} chunk(s)."
    )

    train_manifests = [_features_manifest_for(variant) for variant in TRAIN_VARIANTS]
    train_manifests.append(_features_manifest_for(NON_CLONE_VARIANT))
    test_manifests = [_features_manifest_for(TEST_VARIANT), _features_manifest_for(NON_CLONE_VARIANT)]

    train_features, train_graph_pairs, train_pairs_with_features = _prepare_scored_pool(
        name="train Type1+Type2+Type3/all",
        pairs=train_pairs,
        manifests=train_manifests,
        graph_types=graph_types,
    )
    test_features, test_graph_pairs, test_pairs_with_features = _prepare_scored_pool(
        name="test Type4",
        pairs=test_pairs,
        manifests=test_manifests,
        graph_types=graph_types,
    )

    rows = []
    for graph_type in graph_types:
        train_graph_specific_pairs = train_graph_pairs.get(graph_type, [])
        test_graph_specific_pairs = test_graph_pairs.get(graph_type, [])
        if not train_graph_specific_pairs or not test_graph_specific_pairs:
            continue

        for k_value in k_values:
            for metric in metrics:
                config_start = time.perf_counter()
                train_scores, train_labels = _score_pairs(
                    train_features,
                    train_graph_specific_pairs,
                    graph_type,
                    k_value,
                    metric,
                    label="train",
                )
                threshold, best_metric = _find_best_threshold(train_scores, train_labels, optimize_for)
                train_metrics = _classification_metrics(train_labels, train_scores, threshold)

                test_scores, test_labels = _score_pairs(
                    test_features,
                    test_graph_specific_pairs,
                    graph_type,
                    k_value,
                    metric,
                    label="test",
                )
                test_metrics = _classification_metrics(test_labels, test_scores, threshold)

                rows.append(
                    {
                        "optimized_for": optimize_for,
                        "graph_type": graph_type,
                        "metric": metric,
                        "metric_parameters": _metric_parameters(metric),
                        "k_eigen": "full" if k_value is None else k_value,
                        "best_threshold": float(threshold),
                        "decision_threshold": float(_decision_threshold(threshold)),
                        "train_best_metric": float(best_metric),
                        "train_pairs": int(len(train_labels)),
                        "train_positive_pairs": int(np.sum(train_labels == 1)),
                        "train_negative_pairs": int(np.sum(train_labels == 0)),
                        "train_accuracy": float(train_metrics["accuracy"]),
                        "train_precision": float(train_metrics["precision"]),
                        "train_recall": float(train_metrics["recall"]),
                        "train_f1": float(train_metrics["f1"]),
                        "train_auc": float(train_metrics["auc"]),
                        "test_pairs": int(len(test_labels)),
                        "test_positive_pairs": int(np.sum(test_labels == 1)),
                        "test_negative_pairs": int(np.sum(test_labels == 0)),
                        "test_accuracy": float(test_metrics["accuracy"]),
                        "test_precision": float(test_metrics["precision"]),
                        "test_recall": float(test_metrics["recall"]),
                        "test_f1": float(test_metrics["f1"]),
                        "test_auc": float(test_metrics["auc"]),
                        "config_seconds": float(time.perf_counter() - config_start),
                    }
                )

    rows.sort(key=lambda row: (row["test_accuracy"], row["test_f1"]), reverse=True)
    seconds = time.perf_counter() - start

    summary = {
        "experiment": "bcb_type123_threshold_test_type4",
        "train_variants": list(TRAIN_VARIANTS),
        "test_variant": TEST_VARIANT,
        "non_clone_variant": NON_CLONE_VARIANT,
        "train_positive_source_counts": train_source_counts,
        "train_pair_selection": train_selection,
        "test_pair_selection": test_selection,
        "train_pairs_after_feature_filter": len(train_pairs_with_features),
        "test_pairs_after_feature_filter": len(test_pairs_with_features),
        "graph_types": graph_types,
        "metrics": metrics,
        "k_values": ["full" if value is None else value for value in k_values],
        "optimize_for": optimize_for,
        "seed": seed,
        "seconds": seconds,
        "results": rows,
    }

    json_path = reports_dir / "type123_threshold_test_type4_summary.json"
    csv_path = reports_dir / "type123_threshold_test_type4_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_csv(csv_path, rows)
    record_pipeline_timing(
        output_root / "pipeline_timings.json",
        "01_train_type123_threshold_test_type4",
        seconds,
        {
            "dataset": "bcb",
            "train_variants": list(TRAIN_VARIANTS),
            "test_variant": TEST_VARIANT,
            "summary_json": str(json_path),
            "results_csv": str(csv_path),
        },
    )

    print("[+] Type1+Type2+Type3 -> Type4 experiment finished.")
    print(f"    Summary: {json_path}")
    print(f"    Results: {csv_path}")
    if rows:
        best = rows[0]
        print(
            "    Best test accuracy: "
            f"{best['test_accuracy']:.4f} "
            f"({best['graph_type']}/{best['metric']}/k={best['k_eigen']}, "
            f"threshold={best['best_threshold']:.6f})"
        )
    return summary


def run_from_env() -> dict:
    return run_bcb_type123_to_type4_threshold_experiment(
        graph_types=_parse_csv_values(os.getenv("BCB_CROSS_GRAPH_TYPES"), ["ast", "cfg", "ddg", "pdg", "cpg"]),
        metrics=_parse_csv_values(os.getenv("BCB_CROSS_METRICS"), ["pss"]),
        k_values=_parse_k_values(os.getenv("BCB_CROSS_K_VALUES", os.getenv("TUNING_K_VALUES", "full"))),
        optimize_for=os.getenv("BCB_CROSS_OPTIMIZE_FOR", os.getenv("TUNING_OPTIMIZE_FOR", "f1")).strip().lower(),
        seed=int(os.getenv("BCB_CROSS_SEED", "42")),
    )
