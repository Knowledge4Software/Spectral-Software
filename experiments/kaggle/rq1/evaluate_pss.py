"""Evaluate PSS on learned latent graphs versus conventional program graphs."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_code.similarity.pss import PSSSimilarity


CONVENTIONAL_GRAPH_TYPES = ("ast", "cfg", "ddg", "cpg")
ALL_GRAPH_TYPES = (*CONVENTIONAL_GRAPH_TYPES, "latent")


def _find_file(root: Path, *names: str) -> Path:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(path for path in root.rglob(name) if path.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of {names} below {root}")


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_latent_export(path: Path) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    spectra: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        # Accept both the compact RQ1 archive and Kaggle's full output ZIP,
        # which materializes the compact archive below a same-named folder.
        roots = []
        for name in sorted(names):
            if not name.endswith("manifest.json"):
                continue
            prefix = name[: -len("manifest.json")]
            if prefix + "rq1_pairs.csv.gz" in names and any(
                item.startswith(prefix + "shards/") and item.endswith(".npz") for item in names
            ):
                roots.append(prefix)
        if not roots:
            raise RuntimeError(
                f"{path} contains no RQ1 manifest/pairs/shards bundle; "
                "provide either the compact latent export or the complete Kaggle output ZIP"
            )
        prefix = min(roots, key=lambda value: (value.count("/"), len(value), value))
        manifest = json.loads(archive.read(prefix + "manifest.json"))
        pairs = pd.read_csv(
            io.BytesIO(archive.read(prefix + "rq1_pairs.csv.gz")),
            compression="gzip",
            dtype={"left_id": str, "right_id": str, "split": str},
        )
        shard_names = sorted(
            name for name in names if name.startswith(prefix + "shards/") and name.endswith(".npz")
        )
        if not shard_names:
            raise RuntimeError("RQ1 archive contains no latent graph shards")
        for name in shard_names:
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as shard:
                code_ids = shard["code_ids"].astype(str)
                eigenvalues = shard["eigenvalues"].astype(np.float64, copy=False)
                if len(code_ids) != len(eigenvalues):
                    raise RuntimeError(f"Mismatched IDs/eigenvalues in {name}")
                for code_id, values in zip(code_ids, eigenvalues):
                    if code_id in spectra:
                        raise RuntimeError(f"Duplicate latent code_id {code_id}")
                    spectra[code_id] = values
    if len(spectra) != int(manifest["code_count"]):
        raise RuntimeError(
            f"Latent manifest declares {manifest['code_count']} codes but archive contains {len(spectra)}"
        )
    return manifest, pairs, spectra


def load_conventional_spectra(
    clean_data: Path, required_ids: set[str]
) -> dict[str, dict[str, np.ndarray]]:
    path = _find_file(clean_data, "graph_spectra.jsonl.gz", "graph_spectra.jsonl")
    records: dict[str, dict[str, np.ndarray]] = {}
    with _open_jsonl(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error
            code_id = str(record.get("code_id", ""))
            if code_id not in required_ids:
                continue
            views = {}
            for graph_type in CONVENTIONAL_GRAPH_TYPES:
                values = np.asarray(
                    record.get("graphs", {}).get(graph_type, {}).get("eigenvalues", []),
                    dtype=np.float64,
                )
                if values.size and np.isfinite(values).all():
                    views[graph_type] = values
            records[code_id] = views
    return records


def common_support(
    pairs: pd.DataFrame,
    conventional: dict[str, dict[str, np.ndarray]],
    latent: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    required = {str(value) for value in pairs["left_id"]} | {str(value) for value in pairs["right_id"]}
    available_by_view = {
        graph_type: {code_id for code_id in required if graph_type in conventional.get(code_id, {})}
        for graph_type in CONVENTIONAL_GRAPH_TYPES
    }
    available_by_view["latent"] = required & set(latent)
    common_ids = set.intersection(*(set(values) for values in available_by_view.values()))
    mask = pairs["left_id"].isin(common_ids) & pairs["right_id"].isin(common_ids)
    selected = pairs.loc[mask].copy()
    before_balance = selected.groupby("split").size().astype(int).to_dict()
    validation_balance_dropped = 0
    source_valid = pairs[pairs["split"].eq("valid")]
    source_counts = source_valid["label"].value_counts()
    selected_valid = selected[selected["split"].eq("valid")]
    selected_counts = selected_valid["label"].value_counts()
    # If the official validation split is exactly 50/50, graph-coverage loss
    # must not silently switch checkpoint selection from Accuracy to F1.
    # Remove only the deterministic excess from the majority label while
    # keeping every graph view on identical validation pairs.
    if (
        len(source_counts) == 2
        and source_counts.nunique() == 1
        and len(selected_counts) == 2
        and selected_counts.nunique() != 1
    ):
        target = int(selected_counts.min())
        balanced_valid = pd.concat(
            [group.head(target) for _, group in selected_valid.groupby("label", sort=True)]
        )
        validation_balance_dropped = int(len(selected_valid) - len(balanced_valid))
        selected = pd.concat((selected[selected["split"].ne("valid")], balanced_valid)).sort_index()
    selected = selected.reset_index(drop=True)
    coverage = {
        "required_code_count": len(required),
        "common_code_count": len(common_ids),
        "available_codes_by_graph": {key: len(value) for key, value in available_by_view.items()},
        "input_pairs_by_split": pairs.groupby("split").size().astype(int).to_dict(),
        "common_pairs_before_validation_balance": before_balance,
        "common_pairs_by_split": selected.groupby("split").size().astype(int).to_dict(),
        "excluded_pairs": int(len(pairs) - len(selected)),
        "validation_balance_dropped": validation_balance_dropped,
    }
    for split in ("valid", "test"):
        split_frame = selected[selected["split"].eq(split)]
        if split_frame.empty or split_frame["label"].nunique() != 2:
            raise RuntimeError(f"Common-support {split} split must contain both labels")
    return selected, coverage


def score_pairs(
    pairs: pd.DataFrame,
    conventional: dict[str, dict[str, np.ndarray]],
    latent: dict[str, np.ndarray],
) -> pd.DataFrame:
    pss = PSSSimilarity()
    evaluation_pairs = pairs[pairs["split"].isin(("valid", "test"))].reset_index(drop=True)
    result = evaluation_pairs[[column for column in ("pair_id", "split", "label", "left_id", "right_id") if column in evaluation_pairs]].copy()
    if "pair_id" not in result:
        result["pair_id"] = (
            result["split"].astype(str) + ":" + result["left_id"].astype(str) + ":" + result["right_id"].astype(str)
        )
    left_ids = result["left_id"].astype(str).tolist()
    right_ids = result["right_id"].astype(str).tolist()
    for graph_type in ALL_GRAPH_TYPES:
        if graph_type == "latent":
            left_values = [latent[code_id] for code_id in left_ids]
            right_values = [latent[code_id] for code_id in right_ids]
        else:
            left_values = [conventional[code_id][graph_type] for code_id in left_ids]
            right_values = [conventional[code_id][graph_type] for code_id in right_ids]
        result[f"pss_{graph_type}"] = pss.compute_many(left_values, right_values)
    return result


def choose_validation_threshold(
    labels: np.ndarray, scores: np.ndarray, selection_metric: str
) -> float:
    if selection_metric not in {"accuracy", "f1"}:
        raise ValueError(f"Unsupported validation selection metric: {selection_metric}")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if selection_metric == "f1":
        objective = f1
    else:
        positives = float(labels.sum())
        negatives = float(len(labels) - labels.sum())
        true_positives = recall[:-1] * positives
        false_positives = np.divide(
            true_positives,
            precision[:-1],
            out=np.full_like(true_positives, negatives),
            where=precision[:-1] > 0,
        ) - true_positives
        objective = (true_positives + negatives - false_positives) / len(labels)
    # F1 is the deterministic tie-breaker for accuracy; later thresholds win
    # exact ties to avoid an unnecessarily high predicted-positive rate.
    best = max(range(len(thresholds)), key=lambda index: (objective[index], f1[index], thresholds[index]))
    return float(thresholds[best])


def metric_row(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int8)
    clone_scores, nonclone_scores = scores[labels == 1], scores[labels == 0]
    pooled = math.sqrt(
        max(
            0.0,
            ((len(clone_scores) - 1) * clone_scores.var(ddof=1) + (len(nonclone_scores) - 1) * nonclone_scores.var(ddof=1))
            / max(1, len(scores) - 2),
        )
    )
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "clone_pss_mean": float(clone_scores.mean()),
        "nonclone_pss_mean": float(nonclone_scores.mean()),
        "cohens_d": float((clone_scores.mean() - nonclone_scores.mean()) / pooled) if pooled else 0.0,
        "pairs": int(len(labels)),
    }


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    thresholds = {}
    rows = []
    validation_labels = scores.loc[scores.split.eq("valid"), "label"].to_numpy(dtype=np.int8)
    positives = int(validation_labels.sum())
    negatives = int(len(validation_labels) - positives)
    validation_selection_metric = "accuracy" if positives == negatives else "f1"
    for graph_type in ALL_GRAPH_TYPES:
        valid = scores[scores.split == "valid"]
        score_column = f"pss_{graph_type}"
        threshold = choose_validation_threshold(
            valid.label.to_numpy(), valid[score_column].to_numpy(), validation_selection_metric
        )
        thresholds[graph_type] = threshold
        for split in ("valid", "test"):
            frame = scores[scores.split == split]
            rows.append(
                {
                    "graph_type": graph_type,
                    "split": split,
                    "validation_selection_metric": validation_selection_metric,
                    **metric_row(frame.label.to_numpy(), frame[score_column].to_numpy(), threshold),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap(
    scores: pd.DataFrame, conventional_graph: str, replicates: int, seed: int
) -> dict:
    test = scores[scores.split.eq("test")]
    labels = test["label"].to_numpy(dtype=np.int8)
    latent = test["pss_latent"].to_numpy(dtype=np.float64)
    conventional = test[f"pss_{conventional_graph}"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    by_label = [np.flatnonzero(labels == label) for label in (0, 1)]
    deltas = []
    for _ in range(replicates):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in by_label]
        )
        sampled_labels = labels[indices]
        deltas.append(
            roc_auc_score(sampled_labels, latent[indices])
            - roc_auc_score(sampled_labels, conventional[indices])
        )
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "metric": "test_roc_auc_delta",
        "comparison": f"latent_minus_{conventional_graph}",
        "replicates": int(replicates),
        "seed": int(seed),
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_delta_gt_zero": float((values > 0).mean()),
    }


def evaluate(args: argparse.Namespace) -> dict:
    manifest, pairs, latent = load_latent_export(args.latent_export)
    required_ids = {str(value) for value in pairs.left_id} | {str(value) for value in pairs.right_id}
    conventional = load_conventional_spectra(args.clean_data, required_ids)
    pairs, coverage = common_support(pairs, conventional, latent)
    scores = score_pairs(pairs, conventional, latent)
    summary = summarize(scores)

    valid = summary[summary.split.eq("valid")].set_index("graph_type")
    best_conventional = max(CONVENTIONAL_GRAPH_TYPES, key=lambda graph: valid.loc[graph, "roc_auc"])
    test = summary[summary.split.eq("test")].set_index("graph_type")
    effects = []
    for graph_type in CONVENTIONAL_GRAPH_TYPES:
        effects.append(
            {
                "comparison": f"latent_minus_{graph_type}",
                **{
                    f"delta_{metric}": float(test.loc["latent", metric] - test.loc[graph_type, metric])
                    for metric in ("roc_auc", "average_precision", "f1", "balanced_accuracy")
                },
            }
        )
    bootstrap = paired_bootstrap(scores, best_conventional, args.bootstrap, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "rq1_pair_pss_scores.csv.gz", index=False, compression="gzip")
    summary.to_csv(args.output_dir / "rq1_pss_metrics.csv", index=False)
    pd.DataFrame(effects).to_csv(args.output_dir / "rq1_latent_effects.csv", index=False)
    (args.output_dir / "rq1_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    (args.output_dir / "rq1_paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    report = {
        "dataset": manifest["dataset"],
        "validation_selection_metric": str(summary["validation_selection_metric"].iloc[0]),
        "best_conventional_selected_on_validation_roc_auc": best_conventional,
        "primary_test_effect": float(test.loc["latent", "roc_auc"] - test.loc[best_conventional, "roc_auc"]),
        "paired_bootstrap": bootstrap,
        "coverage": coverage,
        "pss_gamma": float(PSSSimilarity().gamma),
        "pss_distance_power": float(PSSSimilarity().distance_power),
    }
    (args.output_dir / "rq1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--latent-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
