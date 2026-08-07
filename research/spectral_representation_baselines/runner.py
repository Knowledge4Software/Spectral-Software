"""Local spectral-representation baselines for the four V3 clean datasets.

The experiment deliberately uses only precomputed per-graph spectral vectors:
eigenvalue statistics plus a fixed prefix of the sorted spectrum.  Pair labels
are used only by RF/LR; ``no_train`` has no fitted classifier. Thresholds
are always selected on the official validation split and then frozen for test.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.dataset_limitations import (
    affected_languages,
    limitation_for,
    unsupported_layers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = PROJECT_ROOT.parent / "outputs"
DEFAULT_RESULTS_ROOT = OUTPUTS_ROOT / "local_spectral_representation_baselines"
DATASET_DIRS = {
    "codexglue": OUTPUTS_ROOT / "codexglue_v3" / "clean_data",
    "semanticclonebench": OUTPUTS_ROOT / "semanticclonebench_v3" / "clean_data",
    "gptclonebench": OUTPUTS_ROOT / "gptclonebench_v3" / "clean_data",
    "atcoder": OUTPUTS_ROOT / "atcoder_v3" / "clean_data",
}
DISPLAY_NAMES = {
    "codexglue": "CODEXGLUE",
    "semanticclonebench": "SEMANTICCLONEBENCH",
    "gptclonebench": "GPTCLONEBENCH",
    "atcoder": "ATCODER",
}
REQUESTED_GRAPH_TYPES = ("ast", "cfg", "pdg", "ddg", "cpg")


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int


def parse_args(default_dataset: str | None = None, default_method: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local spectral-representation baseline.")
    parser.add_argument("--dataset", choices=tuple(DATASET_DIRS), default=default_dataset, required=default_dataset is None)
    parser.add_argument("--method", choices=("no_train", "rf", "lr"), default=default_method, required=default_method is None)
    parser.add_argument("--clean-data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--graph-types", default="auto", help="auto or comma-separated AST/CFG/DDG/CPG/PDG")
    parser.add_argument("--k-eigen", type=int, default=64)
    parser.add_argument("--include-graph-stats", action="store_true", help="Append node/edge/density statistics to spectral vectors.")
    parser.add_argument("--include-unsupported-layers", action="store_true",
                        help="Score layers a language in this dataset cannot produce (auditing only).")
    parser.add_argument("--max-train-pairs", type=int, default=None, help="Optional reproducible cap; default uses every official train pair.")
    parser.add_argument("--max-valid-pairs", type=int, default=None)
    parser.add_argument("--max-test-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    return parser.parse_args()


def resolve_file(root: Path, *names: str) -> Path:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    matches: list[Path] = []
    for name in names:
        matches.extend(path for path in root.rglob(name) if path.is_file())
    if matches:
        return min(set(matches), key=lambda path: (len(path.relative_to(root).parts), len(path.name), str(path)))
    raise FileNotFoundError(f"Missing one of {names} below {root}")


def is_gzip(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(2) == b"\x1f\x8b"


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if is_gzip(path) else path.open("r", encoding="utf-8")


def load_pairs(root: Path) -> pd.DataFrame:
    path = resolve_file(root, "pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp")
    frame = pd.read_csv(path, compression="gzip" if is_gzip(path) else None)
    required = {"split", "left_id", "right_id", "label"}
    if not required.issubset(frame):
        raise ValueError(f"{path} lacks pair columns {sorted(required - set(frame))}")
    frame = frame.loc[:, ["split", "left_id", "right_id", "label"]].copy()
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["left_id"] = frame["left_id"].astype(str)
    frame["right_id"] = frame["right_id"].astype(str)
    frame["label"] = frame["label"].astype(np.int8)
    return frame


def eigen_stats(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float32)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.zeros(8, dtype=np.float32)
    q25, q50, q75 = np.percentile(values, (25, 50, 75)).astype(np.float32)
    return np.asarray((min(values.size / 2000.0, 10.0), values.mean(), values.std(), values.min(), values.max(), q25, q50, q75), dtype=np.float32)


def graph_stats(layer: dict) -> np.ndarray:
    adjacency = layer.get("adjacency") or {}
    nodes = int(adjacency.get("num_nodes", 0) or 0)
    edges = int(adjacency.get("num_edges", 0) or 0)
    density = edges / max(1, nodes * max(1, nodes - 1))
    return np.asarray((np.log1p(nodes) / 10.0, np.log1p(edges) / 10.0, min(1.0, density)), dtype=np.float32)


def vector_from_layer(layer: dict, k_eigen: int, include_graph_stats: bool) -> np.ndarray | None:
    if not isinstance(layer, dict) or layer.get("spectral_status") not in (None, "ok"):
        return None
    eigenvalues = np.asarray(layer.get("eigenvalues") or (), dtype=np.float32)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if not eigenvalues.size:
        return None
    # Spectrum exports are already sorted. Sorting here makes the baseline
    # robust to legacy records without changing valid current records.
    eigenvalues.sort()
    padded = np.zeros(k_eigen, dtype=np.float32)
    padded[: min(k_eigen, eigenvalues.size)] = eigenvalues[:k_eigen]
    pieces = [eigen_stats(eigenvalues), padded]
    if include_graph_stats:
        pieces.append(graph_stats(layer))
    return np.concatenate(pieces).astype(np.float32, copy=False)


def load_spectral_vectors(root: Path, graph_type: str, k_eigen: int, include_graph_stats: bool) -> dict[str, np.ndarray]:
    path = resolve_file(root, "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp")
    vectors: dict[str, np.ndarray] = {}
    with open_text(path) as stream:
        for line in tqdm(stream, desc=f"{graph_type.upper()} spectral vectors", unit="code"):
            if not line.strip():
                continue
            row = json.loads(line)
            vector = vector_from_layer((row.get("graphs") or {}).get(graph_type), k_eigen, include_graph_stats)
            if vector is not None:
                vectors[str(row.get("code_id"))] = vector
    return vectors


def discover_graph_types(root: Path, requested: str) -> tuple[list[str], list[str]]:
    path = resolve_file(root, "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp")
    available: set[str] = set()
    with open_text(path) as stream:
        for line in stream:
            if line.strip():
                available.update((json.loads(line).get("graphs") or {}).keys())
                break
    if requested.strip().lower() == "auto":
        wanted = list(REQUESTED_GRAPH_TYPES)
    else:
        wanted = [value.strip().lower() for value in requested.split(",") if value.strip()]
    return [value for value in wanted if value in available], [value for value in wanted if value not in available]


def cap_split(frame: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or cap <= 0 or len(frame) <= cap:
        return frame.reset_index(drop=True)
    # Stratified sampling prevents an optional debug cap from accidentally
    # changing the clone/non-clone balance.
    ratio = cap / len(frame)
    pieces = []
    for index, (_, group) in enumerate(frame.groupby("label", sort=True)):
        count = max(1, int(round(len(group) * ratio)))
        pieces.append(group.sample(n=min(count, len(group)), random_state=seed + index))
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).head(cap).reset_index(drop=True)


def usable_splits(pairs: pd.DataFrame, vectors: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    usable = pairs[pairs.left_id.isin(vectors) & pairs.right_id.isin(vectors)].reset_index(drop=True)
    caps = {"train": args.max_train_pairs, "valid": args.max_valid_pairs, "test": args.max_test_pairs}
    splits = {name: cap_split(usable[usable.split == name], caps[name], args.seed + offset) for offset, name in enumerate(("train", "valid", "test"))}
    for name, frame in splits.items():
        labels = set(frame.label.astype(int).unique())
        if labels != {0, 1}:
            raise RuntimeError(f"{name} has no usable two-class split after spectral coverage filtering: {labels}")
    return splits


def code_matrix(vectors: dict[str, np.ndarray]) -> tuple[list[str], dict[str, int], np.ndarray]:
    ids = sorted(vectors)
    return ids, {code_id: index for index, code_id in enumerate(ids)}, np.stack([vectors[code_id] for code_id in ids]).astype(np.float32)


def pair_indices(frame: pd.DataFrame, index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame.left_id.map(index).to_numpy(dtype=np.int64),
        frame.right_id.map(index).to_numpy(dtype=np.int64),
        frame.label.to_numpy(dtype=np.int8),
    )


def pair_features(matrix: np.ndarray, left: np.ndarray, right: np.ndarray, *, chunk: int = 100_000) -> np.ndarray:
    dimension = matrix.shape[1]
    result = np.empty((len(left), 2 * dimension + 2), dtype=np.float32)
    for start in range(0, len(left), chunk):
        end = min(len(left), start + chunk)
        lhs, rhs = matrix[left[start:end]], matrix[right[start:end]]
        diff = np.abs(lhs - rhs)
        product = lhs * rhs
        cosine = (lhs * rhs).sum(axis=1, keepdims=True) / (np.linalg.norm(lhs, axis=1, keepdims=True) * np.linalg.norm(rhs, axis=1, keepdims=True) + 1e-8)
        l2 = np.linalg.norm(lhs - rhs, axis=1, keepdims=True)
        result[start:end] = np.concatenate((diff, product, cosine, l2), axis=1)
    return result


def no_train_scores(matrix: np.ndarray, left: np.ndarray, right: np.ndarray, *, chunk: int = 100_000) -> np.ndarray:
    scores = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), chunk):
        end = min(len(left), start + chunk)
        scores[start:end] = 1.0 / (1.0 + np.linalg.norm(matrix[left[start:end]] - matrix[right[start:end]], axis=1))
    return scores


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Metrics:
    prediction = (scores >= threshold).astype(np.int8)
    tp = int(np.logical_and(prediction == 1, labels == 1).sum())
    fp = int(np.logical_and(prediction == 1, labels == 0).sum())
    tn = int(np.logical_and(prediction == 0, labels == 0).sum())
    fn = int(np.logical_and(prediction == 0, labels == 1).sum())
    return Metrics(
        precision=float(precision_score(labels, prediction, zero_division=0)),
        recall=float(recall_score(labels, prediction, zero_division=0)),
        f1=float(f1_score(labels, prediction, zero_division=0)),
        accuracy=float(accuracy_score(labels, prediction)),
        macro_f1=float(f1_score(labels, prediction, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
        threshold=float(threshold), tp=tp, fp=fp, tn=tn, fn=fn,
    )


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate((np.linspace(.01, .99, 199), np.quantile(scores, np.linspace(0, 1, 301)))))
    best: tuple[float, Metrics] | None = None
    for threshold in candidates:
        current = metrics(labels, scores, float(threshold))
        if best is None or (current.f1, current.balanced_accuracy) > (best[1].f1, best[1].balanced_accuracy):
            best = (float(threshold), current)
    return best[0] if best else .5


def code_languages(root: Path) -> dict[str, str]:
    """``code_id -> language`` from the clean-data bundle."""
    path = resolve_file(root, "codes.jsonl.gz", "codes.jsonl", "codes.jsonl.gz.tmp")
    languages: dict[str, str] = {}
    with open_text(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            languages[str(record.get("code_id"))] = str(record.get("language", "unknown"))
    return languages


def language_breakdown(root: Path, frame: pd.DataFrame, scores: np.ndarray, threshold: float,
                       *, dataset: str, method: str, graph_type: str) -> list[dict]:
    """Split this run's test predictions by the language of each pair.

    Not a per-language retraining: the same predictions that produce the headline
    row are partitioned, so a strong average cannot hide a language the model
    fails on. Cross-language pairs get their own bucket rather than being
    attributed to one side.
    """
    languages = code_languages(root)
    if not languages or not len(frame):
        return []
    labels = frame.label.to_numpy(dtype=np.int64)
    predicted = (np.asarray(scores, dtype=np.float64) >= float(threshold)).astype(np.int64)
    if len(predicted) != len(labels):
        return []

    left = [languages.get(str(value), "unknown") for value in frame.left_id]
    right = [languages.get(str(value), "unknown") for value in frame.right_id]
    keys = [a if a == b else f"{min(a, b)}->{max(a, b)}" for a, b in zip(left, right)]

    out = []
    for key in sorted(set(keys)) + ["ALL"]:
        mask = np.ones(len(keys), dtype=bool) if key == "ALL" else np.asarray([value == key for value in keys])
        subset = metrics(labels[mask], np.asarray(scores)[mask], threshold)
        out.append({"Dataset": dataset, "Method": method, "Graph": graph_type.upper(), "Language": key,
                    "P": subset.precision, "R": subset.recall, "F1": subset.f1, "Acc": subset.accuracy,
                    "Pairs": int(mask.sum()), "Positives": int(labels[mask].sum()),
                    "Threshold": float(threshold)})
    return out


def row_from_metrics(dataset: str, graph_type: str, method: str, test: Metrics, valid: Metrics, *, train_pairs: int, valid_pairs: int, test_pairs: int, seconds: float, status: str = "ok", note: str = "") -> dict:
    return {
        "Dataset": DISPLAY_NAMES[dataset], "Graph": graph_type.upper(), "Method": f"{graph_type.upper()} + {method}", "Status": status, "Note": note,
        "BestValidF1": valid.f1, "P": test.precision, "R": test.recall, "F1": test.f1, "Acc": test.accuracy,
        "MacroF1": test.macro_f1, "BalancedAccuracy": test.balanced_accuracy, "Threshold": test.threshold,
        "TP": test.tp, "FP": test.fp, "TN": test.tn, "FN": test.fn,
        "TrainPairs": train_pairs, "ValidPairs": valid_pairs, "TestPairs": test_pairs,
        "RuntimeSeconds": seconds, "RuntimeMinutes": seconds / 60.0,
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    root = (args.clean_data_dir or DATASET_DIRS[args.dataset]).resolve()
    if not root.exists(): raise FileNotFoundError(f"Clean dataset is not available: {root}")
    out_dir = (args.output_root / args.dataset / args.method).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); pairs = load_pairs(root)
    selected, unavailable = discover_graph_types(root, args.graph_types)
    print(f"Dataset: {DISPLAY_NAMES[args.dataset]} | clean data: {root}")
    print(pairs.groupby(["split", "label"]).size())
    print("Graph types:", selected, "| unavailable:", unavailable)
    rows: list[dict] = []
    language_rows: list[dict] = []
    method_name = {"no_train": "No Train", "rf": "RF", "lr": "LR"}[args.method]
    for graph_type in unavailable:
        rows.append({"Dataset": DISPLAY_NAMES[args.dataset], "Graph": graph_type.upper(), "Method": f"{graph_type.upper()} + {method_name}", "Status": "unavailable", "Note": "This V3 clean dataset has no independent graph layer for this type."})

    # A layer we already know the toolchain cannot produce for one of the
    # dataset's languages is not a weak baseline, it is a meaningless one: those
    # records carry no structure at all. The Kaggle notebooks drop the same
    # layers, so scoring them here would put two different protocols in one table.
    blocked = set(unsupported_layers(f"{args.dataset}_v3"))
    if not args.include_unsupported_layers and blocked:
        skipped = [name for name in selected if name in blocked]
        selected = [name for name in selected if name not in blocked]
        for graph_type in skipped:
            reason = limitation_for(f"{args.dataset}_v3", affected_languages(f"{args.dataset}_v3", graph_type)[0], graph_type)
            print(f"[skip] {graph_type.upper()}: {reason}")
            rows.append({"Dataset": DISPLAY_NAMES[args.dataset], "Graph": graph_type.upper(),
                         "Method": f"{graph_type.upper()} + {method_name}", "Status": "unsupported",
                         "Note": reason})
    for graph_type in selected:
        graph_started = time.perf_counter(); vectors = load_spectral_vectors(root, graph_type, args.k_eigen, args.include_graph_stats)
        splits = usable_splits(pairs, vectors, args)
        ids, index, matrix = code_matrix(vectors)
        arrays = {name: pair_indices(frame, index) for name, frame in splits.items()}
        print(f"{graph_type.upper()}: vectors={len(vectors):,}; train/valid/test={[len(arrays[name][2]) for name in ('train','valid','test')]}")
        if args.method == "no_train":
            valid_scores = no_train_scores(matrix, arrays["valid"][0], arrays["valid"][1]); test_scores = no_train_scores(matrix, arrays["test"][0], arrays["test"][1])
            threshold = best_threshold(arrays["valid"][2], valid_scores); valid = metrics(arrays["valid"][2], valid_scores, threshold); test = metrics(arrays["test"][2], test_scores, threshold)
            extra = {"TrainableParameters": 0, "Device": "cpu"}
        elif args.method in {"rf", "lr"}:
            print(f"{graph_type.upper()} building train pair matrix ({len(arrays['train'][2]):,} pairs)...")
            train_x = pair_features(matrix, arrays["train"][0], arrays["train"][1])
            print(f"{graph_type.upper()} building validation pair matrix ({len(arrays['valid'][2]):,} pairs)...")
            valid_x = pair_features(matrix, arrays["valid"][0], arrays["valid"][1])
            print(f"{graph_type.upper()} building test pair matrix ({len(arrays['test'][2]):,} pairs)...")
            test_x = pair_features(matrix, arrays["test"][0], arrays["test"][1])
            if args.method == "rf":
                print(f"{graph_type.upper()} fitting Random Forest ({args.rf_trees} trees, depth={args.rf_max_depth})...")
                model = RandomForestClassifier(n_estimators=args.rf_trees, max_depth=args.rf_max_depth, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed, verbose=1)
            else:
                print(f"{graph_type.upper()} scaling matrices and fitting Logistic Regression...")
                scaler = StandardScaler(); train_x = scaler.fit_transform(train_x); valid_x = scaler.transform(valid_x); test_x = scaler.transform(test_x)
                model = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1, random_state=args.seed)
            model.fit(train_x, arrays["train"][2]); valid_scores = model.predict_proba(valid_x)[:,1]; test_scores = model.predict_proba(test_x)[:,1]
            threshold = best_threshold(arrays["valid"][2], valid_scores); valid = metrics(arrays["valid"][2], valid_scores, threshold); test = metrics(arrays["test"][2], test_scores, threshold)
            extra = {"TrainableParameters": np.nan, "Device": "cpu"}
            del train_x, valid_x, test_x
        item = row_from_metrics(args.dataset, graph_type, method_name, test, valid, train_pairs=len(arrays["train"][2]), valid_pairs=len(arrays["valid"][2]), test_pairs=len(arrays["test"][2]), seconds=time.perf_counter()-graph_started)
        item.update(extra); rows.append(item)
        print(f"{item['Method']}: F1={item['F1']:.4f}, Acc={item['Acc']:.4f}, minutes={item['RuntimeMinutes']:.2f}")
        language_rows.extend(language_breakdown(
            root, splits["test"], test_scores, threshold,
            dataset=DISPLAY_NAMES[args.dataset], method=item["Method"], graph_type=graph_type))
    results = pd.DataFrame(rows)
    results_path = out_dir / f"{args.dataset}_{args.method}_spectral_representation_results.csv"; results.to_csv(results_path, index=False)
    if language_rows:
        language_path = out_dir / f"{args.dataset}_{args.method}_language_breakdown.csv"
        pd.DataFrame(language_rows).to_csv(language_path, index=False)
        print("Per-language breakdown saved:", language_path)
    metadata = {"Dataset": args.dataset, "Method": args.method, "CleanDataDir": str(root), "Arguments": vars(args), "AvailableGraphTypes": selected, "UnavailableGraphTypes": unavailable, "TotalRuntimeSeconds": time.perf_counter()-started, "CompletedUTC": datetime.now(timezone.utc).isoformat(), "Python": sys.version}
    (out_dir / f"{args.dataset}_{args.method}_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print("\nResults saved:", results_path)
    print(results.to_string(index=False))
    return results


def launch(default_dataset: str, default_method: str) -> None:
    run(parse_args(default_dataset, default_method))


def launch_all(default_dataset: str) -> None:
    """Run No-Train, RF and LR sequentially for one dataset.

    The normal runner parser is reused so every cap and model override has the
    same meaning as it does in an individual launcher.
    """
    args = parse_args(default_dataset, "no_train")
    for method in ("no_train", "rf", "lr"):
        current = argparse.Namespace(**vars(args)); current.method = method
        print("\n" + "=" * 100)
        print(f"Starting {DISPLAY_NAMES[default_dataset]} spectral baseline: {method}")
        print("=" * 100)
        run(current)


if __name__ == "__main__":
    run(parse_args())
