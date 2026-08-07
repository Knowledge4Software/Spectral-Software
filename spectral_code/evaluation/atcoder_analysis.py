"""Analysis utilities for the portable ATCoder clean-data export."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from spectral_code.similarity.pss import PSSSimilarity
from spectral_code.utils.dataset_paths import output_root_for


GRAPH_TYPES = ("ast", "cfg", "ddg", "cpg")


def clean_data_dir(root: str | Path | None = None) -> Path:
    return Path(root or output_root_for("atcoder") / "clean_data").resolve()


def load_codes(root: str | Path | None = None, *, include_code: bool = False) -> pd.DataFrame:
    rows = []
    with gzip.open(clean_data_dir(root) / "codes.jsonl.gz", "rt", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            if not include_code:
                row.pop("code", None)
            rows.append(row)
    return pd.DataFrame(rows)


def load_pairs(root: str | Path | None = None) -> pd.DataFrame:
    with gzip.open(clean_data_dir(root) / "pairs.csv.gz", "rt", encoding="utf-8") as src:
        pairs = pd.read_csv(src, dtype={"left_id": str, "right_id": str, "label": int, "split": str})
    pairs["label_name"] = pairs["label"].map({1: "clone", 0: "non_clone"})
    return pairs


def graph_coverage(root: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    root = clean_data_dir(root)
    codes = load_codes(root)
    languages = dict(zip(codes["code_id"].astype(str), codes["language"]))
    counts: Counter[tuple[str, str, str]] = Counter()
    empty_eigenvalues: Counter[tuple[str, str, str]] = Counter()
    graph_ids: set[str] = set()
    with gzip.open(root / "graph_spectra.jsonl.gz", "rt", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            code_id = str(row["code_id"])
            graph_ids.add(code_id)
            for graph_type, value in row["graphs"].items():
                key = (languages[code_id], graph_type, value.get("spectral_status", "missing"))
                counts[key] += 1
                if not value.get("eigenvalue_count", 0):
                    empty_eigenvalues[key] += 1
    pairs = load_pairs(root)
    endpoint_ids = set(pairs["left_id"].astype(str)) | set(pairs["right_id"].astype(str))
    rows = [
        {
            "language": language,
            "graph_type": graph_type,
            "spectral_status": status,
            "codes": count,
            "empty_eigenvalue_layers": empty_eigenvalues[(language, graph_type, status)],
        }
        for (language, graph_type, status), count in sorted(counts.items())
    ]
    summary = {
        "codes": len(languages),
        "graph_records": len(graph_ids),
        "pair_endpoint_ids": len(endpoint_ids),
        "pair_endpoint_ids_missing_graph_records": len(endpoint_ids - graph_ids),
        "missing_or_empty_layers": sum(count for (_, _, status), count in counts.items() if status == "missing_or_empty_graph"),
        "empty_eigenvalue_layers": sum(empty_eigenvalues.values()),
    }
    return pd.DataFrame(rows), summary


def _sample_balanced_pairs(pairs: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    train = pairs[pairs["split"] == "train"]
    chunks = []
    for label in (0, 1):
        group = train[train["label"] == label]
        chunks.append(group.sample(n=min(sample_size // 2, len(group)), random_state=seed + label))
    return pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _read_eigenvalues(root: Path, code_ids: set[str]) -> dict[str, dict[str, np.ndarray]]:
    values: dict[str, dict[str, np.ndarray]] = {}
    with gzip.open(root / "graph_spectra.jsonl.gz", "rt", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            code_id = str(row["code_id"])
            if code_id not in code_ids:
                continue
            values[code_id] = {
                graph_type: np.asarray(graph["eigenvalues"], dtype=float)
                for graph_type, graph in row["graphs"].items()
                if graph.get("eigenvalue_count", 0)
            }
    return values


def load_graph_records(root: str | Path | None, code_ids: set[str]) -> dict[str, dict]:
    """Load only selected sparse graph records from the final compressed export."""
    records: dict[str, dict] = {}
    with gzip.open(clean_data_dir(root) / "graph_spectra.jsonl.gz", "rt", encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            code_id = str(row["code_id"])
            if code_id in code_ids:
                records[code_id] = row
    return records


def select_pair_examples(
    root: str | Path | None = None,
    *,
    graph_type: str = "cpg",
    per_label: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Choose clone and non-clone pairs with two usable spectral graph records."""
    pairs = load_pairs(root)
    candidates = []
    for label in (1, 0):
        group = pairs[pairs["label"] == label]
        candidates.append(group.sample(n=min(len(group), max(200, per_label * 25)), random_state=seed + label))
    sampled = pd.concat(candidates, ignore_index=True)
    ids = set(sampled["left_id"].astype(str)) | set(sampled["right_id"].astype(str))
    graphs = load_graph_records(root, ids)
    selected: list[dict] = []
    for label in (1, 0):
        count = 0
        for pair in sampled[sampled["label"] == label].itertuples(index=False):
            left = graphs.get(str(pair.left_id), {}).get("graphs", {}).get(graph_type, {})
            right = graphs.get(str(pair.right_id), {}).get("graphs", {}).get(graph_type, {})
            if left.get("eigenvalue_count", 0) and right.get("eigenvalue_count", 0):
                selected.append({"pair": pair._asdict(), "left_graph": left, "right_graph": right})
                count += 1
                if count == per_label:
                    break
    return selected


def _draw_sparse_graph(axis, graph_layer: dict, title: str, max_nodes: int) -> None:
    adjacency = graph_layer.get("adjacency", {})
    graph = nx.DiGraph()
    node_count = min(int(adjacency.get("num_nodes", 0)), max_nodes)
    node_types = adjacency.get("node_types", [])
    for node in range(node_count):
        graph.add_node(node, label=str(node_types[node]) if node < len(node_types) else "node")
    for source, target in zip(adjacency.get("row", []), adjacency.get("col", [])):
        if int(source) < node_count and int(target) < node_count:
            graph.add_edge(int(source), int(target))
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    if not graph:
        axis.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        return
    layout = nx.spring_layout(graph, seed=42, k=1.2 / max(1, np.sqrt(len(graph))))
    nx.draw_networkx_nodes(graph, layout, node_size=28, node_color="#4C78A8", ax=axis)
    nx.draw_networkx_edges(graph, layout, width=0.45, alpha=0.55, arrows=False, ax=axis)
    if int(adjacency.get("num_nodes", 0)) > max_nodes:
        axis.text(0.5, -0.06, f"First {max_nodes:,} of {adjacency['num_nodes']:,} nodes", ha="center", transform=axis.transAxes, fontsize=8)


def plot_pair_code_graph_spectra(
    root: str | Path | None,
    example: dict,
    *,
    graph_type: str = "cpg",
    max_code_lines: int = 32,
    max_graph_nodes: int = 80,
):
    """Render both codes, sparse graphs, and spectra side-by-side for one pair."""
    pair = example["pair"]
    code_lookup = load_codes(root, include_code=True).set_index("code_id")
    left_id, right_id = str(pair["left_id"]), str(pair["right_id"])
    left_code, right_code = str(code_lookup.loc[left_id, "code"]), str(code_lookup.loc[right_id, "code"])
    label = "Clone" if int(pair["label"]) else "Non-clone"
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), gridspec_kw={"height_ratios": [1.3, 1, 0.9]})
    for axis, code, code_id, side in ((axes[0, 0], left_code, left_id, "Left"), (axes[0, 1], right_code, right_id, "Right")):
        visible = code.splitlines()[:max_code_lines]
        suffix = "\n…" if len(code.splitlines()) > max_code_lines else ""
        axis.axis("off")
        axis.set_title(f"{side} code — {code_id}", loc="left", fontsize=10)
        axis.text(0, 1, "\n".join(visible) + suffix, family="monospace", fontsize=7, va="top", transform=axis.transAxes)
    _draw_sparse_graph(axes[1, 0], example["left_graph"], f"Left {graph_type.upper()} graph", max_graph_nodes)
    _draw_sparse_graph(axes[1, 1], example["right_graph"], f"Right {graph_type.upper()} graph", max_graph_nodes)
    for axis, graph, side in ((axes[2, 0], example["left_graph"], "Left"), (axes[2, 1], example["right_graph"], "Right")):
        values = np.asarray(graph.get("eigenvalues", []), dtype=float)
        axis.plot(np.arange(len(values)), values, marker="o", markersize=2.2, linewidth=1.0)
        axis.set_title(f"{side} {graph_type.upper()} eigenvalues ({len(values):,})", fontsize=10)
        axis.set_xlabel("Eigenvalue index")
        axis.set_ylabel("Value")
    figure.suptitle(f"{label} pair — {left_id} vs {right_id}", fontsize=14, y=0.995)
    figure.tight_layout()
    return figure


def sample_similarity_scores(root: str | Path | None = None, *, sample_size: int = 10000, seed: int = 42) -> pd.DataFrame:
    root = clean_data_dir(root)
    sampled = _sample_balanced_pairs(load_pairs(root), sample_size, seed)
    code_ids = set(sampled["left_id"].astype(str)) | set(sampled["right_id"].astype(str))
    eigenvalues = _read_eigenvalues(root, code_ids)
    pss = PSSSimilarity()
    rows = []
    for pair in sampled.itertuples(index=False):
        left, right = eigenvalues.get(str(pair.left_id), {}), eigenvalues.get(str(pair.right_id), {})
        for graph_type in GRAPH_TYPES:
            left_values, right_values = left.get(graph_type), right.get(graph_type)
            if left_values is None or right_values is None or not len(left_values) or not len(right_values):
                continue
            rows.append({
                "split": pair.split,
                "label": int(pair.label),
                "label_name": pair.label_name,
                "graph_type": graph_type,
                "pss": pss.compute(left_values, right_values),
                "wasserstein": float(wasserstein_distance(left_values, right_values)),
            })
    return pd.DataFrame(rows)


def threshold_summary(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for graph_type, group in scores.groupby("graph_type"):
        labels, values = group["label"].to_numpy(dtype=int), group["pss"].to_numpy(dtype=float)
        best = None
        for threshold in np.unique(np.quantile(values, np.linspace(0.01, 0.99, 99))):
            predictions = (values >= threshold).astype(int)
            row = {
                "graph_type": graph_type, "threshold": float(threshold),
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "accuracy": float(accuracy_score(labels, predictions)),
                "roc_auc": float(roc_auc_score(labels, values)), "support": int(len(labels)),
            }
            if best is None or (row["f1"], row["roc_auc"], row["accuracy"]) > (best["f1"], best["roc_auc"], best["accuracy"]):
                best = row
        if best:
            rows.append(best)
    thresholds = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    distributions = scores.groupby(["graph_type", "label_name"])["pss"].agg(["count", "mean", "median", "std"]).reset_index()
    return thresholds, distributions


def write_analysis_reports(root: str | Path | None = None, *, sample_size: int = 10000, seed: int = 42) -> dict[str, Path]:
    clean_root = clean_data_dir(root)
    reports = clean_root.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    codes, pairs = load_codes(clean_root), load_pairs(clean_root)
    coverage, coverage_summary = graph_coverage(clean_root)
    scores = sample_similarity_scores(clean_root, sample_size=sample_size, seed=seed)
    thresholds, distributions = threshold_summary(scores)
    paths = {
        "code_counts": reports / "atcoder_code_counts.csv",
        "pair_counts": reports / "atcoder_pair_counts.csv",
        "coverage": reports / "atcoder_graph_coverage.csv",
        "coverage_summary": reports / "atcoder_graph_coverage_summary.json",
        "scores": reports / "atcoder_similarity_sample.csv",
        "thresholds": reports / "atcoder_threshold_sample.csv",
        "distributions": reports / "atcoder_pss_distribution_sample.csv",
    }
    codes.groupby("language").size().reset_index(name="codes").to_csv(paths["code_counts"], index=False)
    pairs.groupby(["split", "label_name"]).size().reset_index(name="pairs").to_csv(paths["pair_counts"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    paths["coverage_summary"].write_text(json.dumps(coverage_summary, indent=2), encoding="utf-8")
    scores.to_csv(paths["scores"], index=False)
    thresholds.to_csv(paths["thresholds"], index=False)
    distributions.to_csv(paths["distributions"], index=False)
    return paths
