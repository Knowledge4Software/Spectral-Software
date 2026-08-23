import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "spectral_code").exists() and (candidate / "pipelines").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_OUTPUTS_ROOT = Path(os.getenv("OUTPUT_BASE_DIR", PROJECT_ROOT.parent / "outputs")).resolve()
DEFAULT_CLEAN_DATA_DIR = DEFAULT_OUTPUTS_ROOT / "xglue" / "clean_data"
DEFAULT_BASELINE_OUTPUT_DIR = DEFAULT_OUTPUTS_ROOT / "xglue" / "baselines"
GRAPH_TYPES = ["ast", "cfg", "pdg", "ddg", "cpg"]


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    threshold: float


def candidate_files_inside(directory: Path, names: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for name in names:
        files.extend(path for path in directory.rglob(name) if path.is_file())
        files.extend(path for path in directory.rglob(name + ".tmp") if path.is_file())
        files.extend(path for path in directory.rglob(name + ".gz.tmp") if path.is_file())
    return files


def resolve_file(path: Path, *fallback_names: str) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        matches = candidate_files_inside(path, fallback_names or (path.name,))
        if matches:
            return sorted(matches, key=lambda item: (len(item.relative_to(path).parts), len(item.name), str(item)))[0]
    raise FileNotFoundError(f"Expected a file but got: {path}")


def find_file(clean_data_dir: Path, *names: str) -> Path:
    roots = [clean_data_dir, clean_data_dir / "clean_data"]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.is_file():
                return direct
            if direct.is_dir():
                return resolve_file(direct, *names)
        for name in names:
            matches = [path for path in root.rglob(name) if path.is_file()]
            if matches:
                return sorted(matches, key=lambda item: (len(item.relative_to(root).parts), len(item.name), str(item)))[0]
    raise FileNotFoundError(f"Could not find any of {names} under {clean_data_dir}")


def is_gzip_file(path: Path) -> bool:
    path = resolve_file(path)
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def open_text(path: Path):
    path = resolve_file(path)
    if is_gzip_file(path):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_pairs(clean_data_dir: Path) -> pd.DataFrame:
    pairs_path = find_file(clean_data_dir, "pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp")
    compression = "gzip" if is_gzip_file(pairs_path) else None
    pairs = pd.read_csv(
        pairs_path,
        compression=compression,
        dtype={"split": str, "left_id": str, "right_id": str, "label": np.int8},
    )
    pairs["left_id"] = pairs["left_id"].astype(str)
    pairs["right_id"] = pairs["right_id"].astype(str)
    pairs["label"] = pairs["label"].astype(np.int8)
    return pairs[["split", "left_id", "right_id", "label"]]


def eigen_stats(values: list[float]) -> np.ndarray:
    arr = np.asarray(values or [], dtype=np.float32)
    if arr.size == 0:
        return np.zeros(8, dtype=np.float32)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75]).astype(np.float32)
    return np.asarray(
        [
            min(arr.size / 2000.0, 10.0),
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
            float(q25),
            float(q50),
            float(q75),
        ],
        dtype=np.float32,
    )


def pad_eigen(values: list[float], k_eigen: int) -> np.ndarray:
    arr = np.asarray(values or [], dtype=np.float32)
    out = np.zeros(k_eigen, dtype=np.float32)
    if arr.size:
        take = min(k_eigen, arr.size)
        out[:take] = arr[:take]
    return out


def graph_stats(layer: dict) -> np.ndarray:
    adjacency = layer.get("adjacency", {}) if isinstance(layer, dict) else {}
    n = int(adjacency.get("num_nodes", 0) or 0)
    raw_edges = int(adjacency.get("num_edges", 0) or 0)
    possible_edges = max(1, n * max(1, n - 1))
    density = raw_edges / possible_edges
    return np.asarray(
        [
            np.log1p(max(n, 0)) / 10.0,
            np.log1p(max(raw_edges, 0)) / 10.0,
            min(density, 1.0),
        ],
        dtype=np.float32,
    )


def load_code_vectors(clean_data_dir: Path, graph_type: str, k_eigen: int, include_graph_stats: bool) -> dict[str, np.ndarray]:
    graphs_path = find_file(clean_data_dir, "graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp")
    vectors: dict[str, np.ndarray] = {}
    with open_text(graphs_path) as handle:
        for line in tqdm(handle, desc=f"Loading {graph_type.upper()} spectra", unit="code"):
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row.get("code_id"))
            layer = row.get("graphs", {}).get(graph_type, {})
            eig = layer.get("eigenvalues", []) if isinstance(layer, dict) else []
            pieces = [eigen_stats(eig), pad_eigen(eig, k_eigen)]
            if include_graph_stats:
                pieces.append(graph_stats(layer))
            vectors[code_id] = np.concatenate(pieces).astype(np.float32)
    return vectors


def filter_pairs_for_vectors(pairs: pd.DataFrame, vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    mask = pairs.left_id.isin(vectors) & pairs.right_id.isin(vectors)
    return pairs[mask].reset_index(drop=True)


def build_pair_matrix(pairs: pd.DataFrame, vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if pairs.empty:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int8)

    dim = len(next(iter(vectors.values())))
    x = np.empty((len(pairs), dim * 2 + 2), dtype=np.float32)
    labels = pairs.label.to_numpy(dtype=np.int8)

    iterator = tqdm(pairs.itertuples(index=False), total=len(pairs), desc="Building pair features", unit="pair")
    for row_index, row in enumerate(iterator):
        left = vectors[row.left_id]
        right = vectors[row.right_id]
        diff = np.abs(left - right)
        prod = left * right
        denom = (np.linalg.norm(left) * np.linalg.norm(right)) + 1e-8
        cosine = np.asarray([float(np.dot(left, right) / denom)], dtype=np.float32)
        l2 = np.asarray([float(np.linalg.norm(left - right))], dtype=np.float32)
        x[row_index] = np.concatenate([diff, prod, cosine, l2])

    return x, labels


def split_pair_matrices(
    pairs: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    splits: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    usable_pairs = filter_pairs_for_vectors(pairs, vectors)
    print(usable_pairs.groupby(["split", "label"]).size())
    out = {}
    for split in splits:
        split_pairs = usable_pairs[usable_pairs.split == split].reset_index(drop=True)
        print(f"{split}: {len(split_pairs):,} pairs")
        out[split] = build_pair_matrix(split_pairs, vectors)
    return out


def no_train_scores(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.asarray([], dtype=np.float32)
    l2 = x[:, -1]
    return (1.0 / (1.0 + l2)).astype(np.float32)


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Metrics:
    preds = (scores >= threshold).astype(np.int8)
    return Metrics(
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        accuracy=float(accuracy_score(labels, preds)),
        threshold=float(threshold),
    )


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.5
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 301)))
    candidates = np.unique(np.concatenate([candidates, np.asarray([0.5], dtype=np.float32)]))
    best_threshold = 0.5
    best_f1 = -1.0
    best_acc = -1.0
    for threshold in candidates:
        metrics = metrics_at_threshold(labels, scores, float(threshold))
        if (metrics.f1, metrics.accuracy) > (best_f1, best_acc):
            best_threshold = float(threshold)
            best_f1 = metrics.f1
            best_acc = metrics.accuracy
    return best_threshold


def result_row(graph_type: str, method: str, metrics: Metrics, valid_metrics: Metrics) -> dict:
    return {
        "Graph": graph_type.upper(),
        "Method": f"{graph_type.upper()} + {method}",
        "P": metrics.precision,
        "R": metrics.recall,
        "F1": metrics.f1,
        "Acc": metrics.accuracy,
        "Threshold": metrics.threshold,
        "Valid_F1": valid_metrics.f1,
    }


def maybe_sample_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_train_pairs: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_train_pairs is None or len(y_train) <= max_train_pairs:
        return x_train, y_train
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(y_train)), size=max_train_pairs, replace=False)
    return x_train[idx], y_train[idx]


def save_results(rows: list[dict], output_path: Path) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    score_cols = ["P", "R", "F1", "Acc", "Threshold", "Valid_F1"]
    for col in score_cols:
        if col in results:
            results[col] = results[col].astype(float)
    results.to_csv(output_path, index=False)
    print("\nFinal results")
    print(results[["Method", "P", "R", "F1", "Acc"]].to_string(index=False))
    print("Saved:", output_path)
    return results
