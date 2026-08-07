from __future__ import annotations

import html
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from spectral_code.evaluation.bcb_dataset import BigCloneBenchLoader, ClonePair
from spectral_code.evaluation.semantic_dataset import SemanticBenchmarkLoader
from spectral_code.evaluation.semantic_preparation import default_semantic_prepared_dir
from spectral_code.evaluation.tuning import PrecomputedSpectralModel, _load_features_db
from spectral_code.utils.dataset_paths import (
    bcb_type_dir,
    output_root_for,
    semantic_dump_path,
    xglue_dir,
)

try:
    from IPython.display import HTML, Markdown, display
except ModuleNotFoundError:  # pragma: no cover - notebook-only helpers
    HTML = Markdown = None

    def display(*items):
        for item in items:
            print(item)


GRAPH_TYPES_DEFAULT = ["ast", "cfg", "ddg", "pdg", "cpg"]
METRICS_DEFAULT = ["pss"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    data_dir: Path | None
    features_manifest: Path
    output_root: Path
    variant: str | None = None


def bcb_spec(clone_type: int | str) -> DatasetSpec:
    clone_type = str(clone_type)
    output_root = output_root_for("bcb", clone_type)
    normalized = clone_type.lower().replace("-", "_")
    if normalized in {"non_clone", "nonclone", "false_positives"}:
        return DatasetSpec(
            key="bcb_non_clone",
            display_name="BigCloneBench Non-clones",
            data_dir=bcb_type_dir("non_clone"),
            features_manifest=output_root / "spectral_features" / "spectral_features_manifest.json",
            output_root=output_root,
            variant="non_clone",
        )
    return DatasetSpec(
        key=f"bcb_type{clone_type}",
        display_name=f"BigCloneBench Type {clone_type}",
        data_dir=bcb_type_dir(clone_type),
        features_manifest=output_root / "spectral_features" / "spectral_features_manifest.json",
        output_root=output_root,
        variant=clone_type,
    )


def xglue_spec() -> DatasetSpec:
    output_root = output_root_for("xglue")
    return DatasetSpec(
        key="xglue",
        display_name="XGLUE",
        data_dir=xglue_dir(),
        features_manifest=output_root / "spectral_features" / "spectral_features_manifest.json",
        output_root=output_root,
    )


def semantic_spec(language: str) -> DatasetSpec:
    lang_key = language.strip().lower()
    output_root = output_root_for("semantic_benchmark", lang_key)
    return DatasetSpec(
        key=f"semantic_{lang_key}",
        display_name=f"Semantic Benchmark ({language})",
        data_dir=default_semantic_prepared_dir(language),
        features_manifest=output_root / "spectral_features" / "spectral_features_manifest.json",
        output_root=output_root,
        variant=language,
    )


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _md(text: str) -> None:
    display(Markdown(text) if Markdown is not None else text)


def configure_notebook_style() -> None:
    """Apply a consistent visual style for all analysis notebook plots."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": True,
        }
    )


def notebook_output_dir(spec: DatasetSpec, config_roots: dict[str, str] | None = None) -> str:
    """Return a dynamically joined notebook artifact directory for a dataset."""
    roots = config_roots or {}
    outputs_root = roots.get("outputs_root") or str(spec.output_root.parent)
    path = os.path.join(outputs_root, "notebook_analysis", spec.key)
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError as exc:
        print(f"[!] Could not create notebook artifact directory: {path} ({exc})")
    return path


def load_pairs_for_spec(spec: DatasetSpec, negative_ratio: float = 1.0) -> list[ClonePair]:
    if spec.key.startswith("bcb_"):
        loader = BigCloneBenchLoader(spec.data_dir)
        pairs = loader.get_pairs("train")
        if spec.key != "bcb_non_clone" and all(pair.label == 1 for pair in pairs):
            non_clone_dir = bcb_type_dir("non_clone")
            if (non_clone_dir / "train.txt").exists() and (non_clone_dir / "data.jsonl").exists():
                non_clone_loader = BigCloneBenchLoader(non_clone_dir)
                pairs = pairs + [pair for pair in non_clone_loader.get_pairs("train") if pair.label == 0]
        return pairs

    if spec.key == "xglue":
        loader = BigCloneBenchLoader(spec.data_dir)
        return loader.get_pairs("train")

    if spec.key.startswith("semantic_"):
        prepared_dir = spec.data_dir or default_semantic_prepared_dir(spec.variant or "Python")
        if (prepared_dir / "data.jsonl").exists() and (prepared_dir / "train.txt").exists():
            loader = BigCloneBenchLoader(prepared_dir)
            return loader.get_pairs("train")
        loader = SemanticBenchmarkLoader(spec.variant or "Python")
        return loader.get_pairs(negative_ratio=max(0.0, negative_ratio))

    raise ValueError(f"Unsupported dataset spec: {spec}")


def dataset_overview_rows(specs: list[DatasetSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = {
            "dataset": spec.display_name,
            "key": spec.key,
            "data_dir": str(spec.data_dir) if spec.data_dir else "",
            "data_dir_exists": bool(spec.data_dir and spec.data_dir.exists()),
            "output_root": str(spec.output_root),
            "output_root_exists": spec.output_root.exists(),
            "features_manifest": str(spec.features_manifest),
            "features_manifest_exists": spec.features_manifest.exists(),
        }

        if spec.key.startswith("semantic_"):
            row["semantic_dump_path"] = str(semantic_dump_path())
            row["prepared_data_dir"] = str(default_semantic_prepared_dir(spec.variant or "Python"))
        else:
            row["semantic_dump_path"] = ""
            row["prepared_data_dir"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def pair_stats_dataframe(pairs: list[ClonePair]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "left_id": [pair.left_id for pair in pairs],
            "right_id": [pair.right_id for pair in pairs],
            "label": [pair.label for pair in pairs],
            "label_name": ["clone" if pair.label == 1 else "nonclone" for pair in pairs],
            "clone_type": [pair.clone_type for pair in pairs],
            "left_chars": [len(pair.left_code) for pair in pairs],
            "right_chars": [len(pair.right_code) for pair in pairs],
            "left_lines": [pair.left_code.count("\n") + 1 for pair in pairs],
            "right_lines": [pair.right_code.count("\n") + 1 for pair in pairs],
        }
    )


def artifact_status_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    data_dir = spec.data_dir or Path("")
    paths = {
        "data_dir": data_dir,
        "data_jsonl": data_dir / "data.jsonl",
        "train_txt": data_dir / "train.txt",
        "metadata_json": data_dir / "metadata.json",
        "type_labels_tsv": data_dir / "type_labels.tsv",
        "output_root": spec.output_root,
        "raw_features_dir": spec.output_root / "dataset_features",
        "clean_graphs_dir": spec.output_root / "clean_graphs",
        "graph_manifest": spec.output_root / "clean_graphs" / "graph_shards_manifest.json",
        "spectral_manifest": spec.features_manifest,
        "timing_stats": spec.output_root / "timing_stats.json",
        "reports_dir": spec.output_root / "reports",
    }
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.is_file() else None,
            }
            for name, path in paths.items()
        ]
    )


def timing_stats_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    stats = _read_json(spec.output_root / "timing_stats.json")
    if not isinstance(stats, dict):
        return pd.DataFrame()
    rows = []
    for key, value in stats.items():
        if isinstance(value, (int, float, str)) or value is None:
            rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)


def skipped_graph_parse_dataframe(spec: DatasetSpec, limit: int = 50) -> pd.DataFrame:
    """Load skipped/empty graph parse records written by pipeline 01."""
    path = spec.output_root / "skipped_graph_parse_pipeline01.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line_number > limit:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"line_number": line_number, "reason": "invalid JSONL row", "raw": line.strip()})
    return pd.DataFrame(rows)


def plot_timing_stats(spec: DatasetSpec) -> None:
    """Plot timing metrics from timing_stats.json using readable horizontal bars."""
    timing = timing_stats_dataframe(spec)
    if timing.empty:
        _md("No timing stats were found yet.")
        return
    numeric = timing[pd.to_numeric(timing["value"], errors="coerce").notna()].copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    timing_rows = numeric[numeric["metric"].str.contains("time|duration", case=False, regex=True)].copy()
    if timing_rows.empty:
        return
    timing_rows = timing_rows.sort_values("value", ascending=True).tail(18)
    plt.figure(figsize=(10, max(4, 0.35 * len(timing_rows))))
    sns.barplot(data=timing_rows, y="metric", x="value", color="#4C78A8")
    plt.title("Pipeline timing metrics")
    plt.xlabel("Seconds")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_graph_coverage_from_timing(spec: DatasetSpec) -> None:
    """Plot DOT mapping and spectral coverage metrics when available."""
    stats = _read_json(spec.output_root / "timing_stats.json")
    if not isinstance(stats, dict):
        return
    rows = []
    for key, value in stats.items():
        if key.startswith(("dot_mapped_", "dot_missing_", "spectral_computed_graphs_", "spectral_skipped_oversized_")):
            prefix, graph_type = key.rsplit("_", 1)
            rows.append({"metric": prefix, "graph_type": graph_type, "count": int(value or 0)})
    coverage_df = pd.DataFrame(rows)
    if coverage_df.empty:
        return
    plt.figure(figsize=(11, 4.2))
    sns.barplot(data=coverage_df, x="graph_type", y="count", hue="metric")
    plt.title("Graph extraction and spectral feature coverage")
    plt.xlabel("Graph layer")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def graph_manifest_summary_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    graph_manifest = _read_json(spec.output_root / "clean_graphs" / "graph_shards_manifest.json")
    spectral_manifest = _read_json(spec.features_manifest)
    raw_dir = spec.output_root / "dataset_features"
    rows = [
        {
            "stage": "raw_features",
            "value": len(list(raw_dir.glob("*.json"))) if raw_dir.exists() else 0,
            "detail": "JSON feature files",
        }
    ]
    if isinstance(graph_manifest, dict):
        rows.extend(
            [
                {"stage": "clean_graphs", "value": graph_manifest.get("total_methods", 0), "detail": "methods"},
                {
                    "stage": "clean_graphs",
                    "value": graph_manifest.get("total_base_layers_cleaned", 0),
                    "detail": "base graph layers",
                },
                {"stage": "clean_graphs", "value": len(graph_manifest.get("shards", [])), "detail": "shards"},
            ]
        )
    if isinstance(spectral_manifest, dict):
        rows.extend(
            [
                {"stage": "spectral_features", "value": spectral_manifest.get("total_methods", 0), "detail": "methods"},
                {
                    "stage": "spectral_features",
                    "value": ", ".join(spectral_manifest.get("graph_types", [])),
                    "detail": "graph types",
                },
                {"stage": "spectral_features", "value": len(spectral_manifest.get("shards", [])), "detail": "shards"},
            ]
        )
    return pd.DataFrame(rows)


def plot_graph_manifest_summary(spec: DatasetSpec) -> None:
    """Plot high-level method/layer/shard counts from graph and spectral manifests."""
    summary = graph_manifest_summary_dataframe(spec)
    if summary.empty:
        return
    numeric = summary[pd.to_numeric(summary["value"], errors="coerce").notna()].copy()
    if numeric.empty:
        return
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric["label"] = numeric["stage"] + " / " + numeric["detail"]
    plt.figure(figsize=(10, max(3.5, 0.35 * len(numeric))))
    sns.barplot(data=numeric, y="label", x="value", color="#54A24B")
    plt.title("Pipeline artifact volume summary")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_pair_size_distributions(pair_df: pd.DataFrame) -> None:
    if pair_df.empty:
        _md("No pairs were loaded.")
        return
    plot_df = pair_df.melt(
        id_vars=["label_name"],
        value_vars=["left_lines", "right_lines", "left_chars", "right_chars"],
        var_name="measure",
        value_name="value",
    )
    g = sns.displot(
        data=plot_df,
        x="value",
        hue="label_name",
        col="measure",
        col_wrap=2,
        kind="hist",
        bins=40,
        facet_kws={"sharex": False, "sharey": False},
        height=3.2,
        aspect=1.35,
    )
    g.fig.suptitle("Pair size distributions", y=1.03)
    plt.show()


def display_dataset_overview(spec: DatasetSpec, negative_ratio: float = 1.0) -> pd.DataFrame:
    _md(f"## {spec.display_name} dataset overview")
    display(artifact_status_dataframe(spec))
    pairs = load_pairs_for_spec(spec, negative_ratio=negative_ratio)
    pair_df = pair_stats_dataframe(pairs)
    display(pair_df["label_name"].value_counts(dropna=False).rename_axis("label").to_frame("pairs"))
    display(pair_df[["left_chars", "right_chars", "left_lines", "right_lines"]].describe().T)
    plot_pair_size_distributions(pair_df)
    return pair_df


def _balanced_sample_pairs(pairs: list[ClonePair], sample_size: int | None = None, seed: int = 42) -> list[ClonePair]:
    if sample_size is None or sample_size >= len(pairs):
        return list(pairs)

    rng = random.Random(seed)
    positives = [pair for pair in pairs if pair.label == 1]
    negatives = [pair for pair in pairs if pair.label == 0]
    if positives and negatives:
        per_label = max(1, sample_size // 2)
        sample = rng.sample(positives, min(per_label, len(positives)))
        sample.extend(rng.sample(negatives, min(sample_size - len(sample), len(negatives))))
        if len(sample) < sample_size:
            remaining = [pair for pair in pairs if pair not in sample]
            sample.extend(rng.sample(remaining, min(sample_size - len(sample), len(remaining))))
        rng.shuffle(sample)
        return sample

    return rng.sample(list(pairs), sample_size)


def _graph_manifest_paths(spec: DatasetSpec) -> list[Path]:
    manifest = _read_json(spec.output_root / "clean_graphs" / "graph_shards_manifest.json")
    if not isinstance(manifest, dict):
        return []
    paths = []
    for raw_path in manifest.get("shards", []):
        path = Path(raw_path)
        if not path.exists() and not path.is_absolute():
            path = spec.output_root / "clean_graphs" / path
        paths.append(path)
    return paths


def load_clean_graph_records(
    spec: DatasetSpec,
    method_ids: set[str | int] | None = None,
    max_records: int | None = None,
) -> dict[str, dict[str, nx.DiGraph | None]]:
    wanted = {str(method_id) for method_id in method_ids} if method_ids is not None else None
    records: dict[str, dict[str, nx.DiGraph | None]] = {}
    for shard_path in _graph_manifest_paths(spec):
        if not shard_path.exists():
            continue
        with shard_path.open("rb") as f:
            shard = pickle.load(f)
        for method_id, graphs in shard.items():
            key = str(method_id)
            if wanted is not None and key not in wanted:
                continue
            records[key] = graphs
            if wanted is not None and wanted.issubset(records.keys()):
                return records
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def _format_code_html(title: str, code: str, max_lines: int = 80) -> str:
    lines = code.splitlines()
    truncated = len(lines) > max_lines
    shown = "\n".join(lines[:max_lines])
    if truncated:
        shown += f"\n... ({len(lines) - max_lines} more lines)"
    return (
        "<div style='width:50%; padding:0 10px; box-sizing:border-box;'>"
        f"<h4 style='margin:0 0 8px 0'>{html.escape(title)}</h4>"
        "<pre style='white-space:pre-wrap; font-size:12px; line-height:1.35; "
        "border:1px solid #ddd; padding:10px; max-height:460px; overflow:auto;'>"
        f"{html.escape(shown)}</pre></div>"
    )


def display_pair_code(pair: ClonePair, max_lines: int = 80) -> None:
    body = (
        "<div style='display:flex; gap:8px; align-items:stretch;'>"
        + _format_code_html(f"Left {pair.left_id}", pair.left_code, max_lines=max_lines)
        + _format_code_html(f"Right {pair.right_id}", pair.right_code, max_lines=max_lines)
        + "</div>"
    )
    display(HTML(body) if HTML is not None else body)


def _safe_subgraph(graph: nx.Graph, max_nodes: int) -> nx.Graph:
    if graph.number_of_nodes() <= max_nodes:
        return graph
    selected = list(graph.nodes())[:max_nodes]
    return graph.subgraph(selected).copy()


def _draw_graph(ax, graph: nx.Graph | None, title: str, graph_type: str, max_nodes: int = 80) -> None:
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if graph is None or graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "missing / empty", ha="center", va="center")
        return

    graph = _safe_subgraph(nx.DiGraph(graph), max_nodes=max_nodes)
    color_map = {
        "ast": "#4C78A8",
        "cfg": "#F58518",
        "ddg": "#54A24B",
        "pdg": "#B279A2",
        "cpg": "#E45756",
    }
    try:
        pos = nx.kamada_kawai_layout(graph.to_undirected())
    except Exception:
        pos = nx.spring_layout(graph.to_undirected(), seed=42)

    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35, width=0.8, arrows=graph.is_directed(), arrowsize=8)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=90 if graph.number_of_nodes() < 45 else 40,
        node_color=color_map.get(graph_type, "#72B7B2"),
        linewidths=0.4,
        edgecolors="white",
        alpha=0.92,
    )
    if graph.number_of_nodes() <= 35:
        labels = {}
        for node, data in graph.nodes(data=True):
            label = str(data.get("label") or data.get("block_type") or node)
            labels[node] = label.replace('"', "").replace("\\n", "\n").splitlines()[0][:18]
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=6)
    ax.text(
        0.01,
        0.02,
        f"{graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
    )


def display_pair_graphs(
    spec: DatasetSpec,
    pair: ClonePair,
    graph_types: list[str] | None = None,
    max_nodes: int = 80,
) -> None:
    graph_types = graph_types or GRAPH_TYPES_DEFAULT
    records = load_clean_graph_records(spec, method_ids={pair.left_id, pair.right_id})
    left = records.get(str(pair.left_id), {})
    right = records.get(str(pair.right_id), {})
    fig, axes = plt.subplots(2, len(graph_types), figsize=(4.2 * len(graph_types), 7.2), squeeze=False)
    for col, graph_type in enumerate(graph_types):
        _draw_graph(axes[0][col], left.get(graph_type), f"Left {pair.left_id} - {graph_type.upper()}", graph_type, max_nodes)
        _draw_graph(axes[1][col], right.get(graph_type), f"Right {pair.right_id} - {graph_type.upper()}", graph_type, max_nodes)
    fig.suptitle(f"{spec.display_name}: code pair graphs", y=1.01, fontsize=14)
    fig.tight_layout()
    plt.show()


def display_pair_code_and_graphs(
    spec: DatasetSpec,
    pair: ClonePair | None = None,
    pair_index: int = 0,
    label: int | None = 1,
    graph_types: list[str] | None = None,
    max_code_lines: int = 80,
    max_nodes: int = 80,
) -> ClonePair:
    pairs = load_pairs_for_spec(spec)
    candidates = [item for item in pairs if label is None or item.label == label]
    if not candidates:
        candidates = pairs
    if pair is None:
        pair = candidates[min(pair_index, len(candidates) - 1)]
    label_name = "clone" if pair.label == 1 else "nonclone"
    _md(f"### Pair {pair.left_id} - {pair.right_id} ({label_name})")
    display_pair_code(pair, max_lines=max_code_lines)
    display_pair_graphs(spec, pair, graph_types=graph_types, max_nodes=max_nodes)
    return pair


def display_pipeline_validation(
    spec: DatasetSpec,
    graph_types: list[str] | None = None,
    sample_label: int | None = 1,
) -> None:
    """Display a rich validation report for prepared data, graph outputs, timing, and examples."""
    configure_notebook_style()
    _md(f"## {spec.display_name} pipeline validation")
    display(artifact_status_dataframe(spec))
    summary = graph_manifest_summary_dataframe(spec)
    if not summary.empty:
        _md("### Graph and spectral artifact summary")
        display(summary)
        plot_graph_manifest_summary(spec)
    timing = timing_stats_dataframe(spec)
    if not timing.empty:
        _md("### Timing and coverage metrics")
        display(timing)
        plot_timing_stats(spec)
        plot_graph_coverage_from_timing(spec)
    skipped = skipped_graph_parse_dataframe(spec)
    if not skipped.empty:
        _md("### Skipped or empty Joern graph layers")
        display(skipped)
    graph_manifest = spec.output_root / "clean_graphs" / "graph_shards_manifest.json"
    if graph_manifest.exists():
        _md("### Representative validated code pair and graph layers")
        display_pair_code_and_graphs(spec, label=sample_label, graph_types=graph_types)
    else:
        _md(f"Graph manifest is not ready yet: `{graph_manifest}`")


def _manifest_graph_types(spec: DatasetSpec) -> list[str]:
    manifest = _read_json(spec.features_manifest)
    if isinstance(manifest, dict) and manifest.get("graph_types"):
        return list(manifest["graph_types"])
    return GRAPH_TYPES_DEFAULT


def compute_similarity_score_dataframe(
    spec: DatasetSpec,
    sample_size: int | None = 1000,
    graph_types: list[str] | None = None,
    metrics: list[str] | None = None,
    seed: int = 42,
    negative_ratio: float = 1.0,
) -> pd.DataFrame:
    if not spec.features_manifest.exists():
        raise FileNotFoundError(f"Spectral features manifest not found: {spec.features_manifest}")

    pairs = load_pairs_for_spec(spec, negative_ratio=negative_ratio)
    pairs = _balanced_sample_pairs(pairs, sample_size=sample_size, seed=seed)
    needed_ids = {str(pair.left_id) for pair in pairs} | {str(pair.right_id) for pair in pairs}
    features_db = _load_features_db(str(spec.features_manifest), needed_ids=needed_ids)
    if not features_db:
        return pd.DataFrame()

    graph_types = graph_types or _manifest_graph_types(spec)
    metrics = metrics or METRICS_DEFAULT
    rows = []
    for graph_type in graph_types:
        for metric in metrics:
            model = PrecomputedSpectralModel(features_db, graph_type, None, None, metric)
            for pair in pairs:
                score = model.score_pair(pair)
                rows.append(
                    {
                        "dataset": spec.display_name,
                        "graph_type": graph_type,
                        "metric": metric,
                        "left_id": pair.left_id,
                        "right_id": pair.right_id,
                        "label": pair.label,
                        "label_name": "clone" if pair.label == 1 else "nonclone",
                        "score": score,
                        "left_lines": pair.left_code.count("\n") + 1,
                        "right_lines": pair.right_code.count("\n") + 1,
                    }
                )
    return pd.DataFrame(rows)


def summarize_similarity_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    return (
        score_df.groupby(["graph_type", "metric", "label_name"])["score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values(["metric", "graph_type", "label_name"])
    )


def plot_similarity_score_distributions(score_df: pd.DataFrame) -> None:
    if score_df.empty:
        _md("No similarity scores to plot.")
        return

    g = sns.displot(
        data=score_df,
        x="score",
        hue="label_name",
        row="metric",
        col="graph_type",
        kind="hist",
        bins=35,
        common_norm=False,
        facet_kws={"sharex": False, "sharey": False},
        height=2.7,
        aspect=1.15,
    )
    g.fig.suptitle("Clone vs non-clone similarity score distributions", y=1.02)
    plt.show()

    plt.figure(figsize=(max(8, 1.5 * score_df["graph_type"].nunique() * score_df["metric"].nunique()), 4.8))
    sns.boxplot(data=score_df, x="graph_type", y="score", hue="label_name")
    plt.title("Score separation by graph type")
    plt.tight_layout()
    plt.show()


def display_similarity_distribution_report(
    spec: DatasetSpec,
    sample_size: int | None = 1000,
    graph_types: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    _md(f"## {spec.display_name} similarity distributions")
    score_df = compute_similarity_score_dataframe(
        spec,
        sample_size=sample_size,
        graph_types=graph_types,
        metrics=metrics,
    )
    summary = summarize_similarity_scores(score_df)
    display(summary)
    plot_similarity_score_distributions(score_df)
    return score_df


def load_tuning_results(spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    for path in sorted(spec.output_root.glob("trained_*.json")):
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if isinstance(item, dict):
                row = dict(item)
                row["source_file"] = str(path)
                rows.append(row)
    return pd.DataFrame(rows)


def plot_tuning_results(tuning_df: pd.DataFrame) -> None:
    if tuning_df.empty:
        _md("No tuning rows to plot.")
        return

    metric_col = "train_f1" if "train_f1" in tuning_df.columns else "best_metric"
    order_df = tuning_df.sort_values(metric_col, ascending=False)
    plt.figure(figsize=(max(9, 1.1 * len(order_df)), 4.8))
    sns.barplot(data=order_df, x="graph_type", y=metric_col, hue="metric")
    plt.ylim(0, 1.02)
    plt.title(f"Tuning performance by graph and metric ({metric_col})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    if {"graph_type", "metric", "best_threshold"}.issubset(tuning_df.columns):
        pivot = tuning_df.pivot_table(index="graph_type", columns="metric", values="best_threshold", aggfunc="max")
        plt.figure(figsize=(7, max(3, 0.55 * len(pivot))))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Best threshold heatmap")
        plt.tight_layout()
        plt.show()


def display_tuning_report(spec: DatasetSpec, include_pair_examples: bool = True) -> pd.DataFrame:
    _md(f"## {spec.display_name} tuning report")
    tuning_df = load_tuning_results(spec)
    if tuning_df.empty:
        _md(f"No tuning result JSON found under `{spec.output_root}`.")
        return tuning_df

    sort_cols = [col for col in ["train_f1", "train_accuracy", "best_metric"] if col in tuning_df.columns]
    display(tuning_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else tuning_df)
    plot_tuning_results(tuning_df)
    if include_pair_examples and (spec.output_root / "clean_graphs" / "graph_shards_manifest.json").exists():
        ranked = tuning_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else tuning_df
        top_graph = str(ranked.iloc[0].get("graph_type", "cpg"))
        graph_types = [top_graph.lower()] if "+" not in top_graph else [part.lower() for part in top_graph.split("+")]
        _md("### Code and graph examples for tuning sanity check")
        display_pair_code_and_graphs(spec, label=1, graph_types=graph_types)
        display_pair_code_and_graphs(spec, label=0, graph_types=graph_types)
    return tuning_df


def _code_axis(ax, title: str, code: str, max_lines: int = 38) -> None:
    """Render source code into a matplotlib axis using monospaced text."""
    lines = code.splitlines()
    shown = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        shown += f"\n... ({len(lines) - max_lines} more lines)"
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10)
    ax.text(
        0,
        1,
        shown,
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.5,
        transform=ax.transAxes,
        bbox={"facecolor": "#F8F8F8", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.5"},
    )


def _method_code_map(pairs: list[ClonePair]) -> dict[str, str]:
    """Build a method/snippet id to source-code lookup from clone-pair records."""
    code_by_id: dict[str, str] = {}
    for pair in pairs:
        code_by_id[str(pair.left_id)] = pair.left_code
        code_by_id[str(pair.right_id)] = pair.right_code
    return code_by_id


def display_code_graph_side_by_side_examples(
    spec: DatasetSpec,
    n_examples: int = 3,
    graph_type: str = "cpg",
    seed: int = 42,
    max_code_lines: int = 38,
    max_nodes: int = 80,
) -> list[str]:
    """Show random source snippets beside their extracted Joern graph visualization."""
    configure_notebook_style()
    pairs = load_pairs_for_spec(spec)
    code_by_id = _method_code_map(pairs)
    rng = random.Random(seed)
    method_ids = list(code_by_id)
    rng.shuffle(method_ids)
    selected_ids = method_ids[: min(n_examples, len(method_ids))]
    graph_records = load_clean_graph_records(spec, method_ids=set(selected_ids))
    skipped = []

    _md(f"## Visual Step 1 - Code vs. {graph_type.upper()} graph side-by-side")
    for method_id in selected_ids:
        graph = graph_records.get(str(method_id), {}).get(graph_type)
        if graph is None or graph.number_of_nodes() == 0:
            skipped.append({"method_id": method_id, "reason": f"missing/empty {graph_type} graph"})

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
        _code_axis(axes[0], f"Source code: {method_id}", code_by_id[str(method_id)], max_lines=max_code_lines)
        _draw_graph(axes[1], graph, f"Joern {graph_type.upper()} graph: {method_id}", graph_type, max_nodes=max_nodes)
        fig.suptitle(f"{spec.display_name} random example {method_id}", y=1.02)
        fig.tight_layout()
        plt.show()

    if skipped:
        _md("Skipped or incomplete graph renders")
        display(pd.DataFrame(skipped))
    return selected_ids


def _feature_record(features_db: dict, method_id: str | int) -> dict:
    """Fetch a feature record regardless of integer/string id representation."""
    return features_db.get(method_id) or features_db.get(str(method_id)) or {}


def _eigenvalues_for(features_db: dict, method_id: str | int, graph_type: str) -> np.ndarray:
    """Return finite eigenvalues for one method and graph layer."""
    values = _feature_record(features_db, method_id).get(graph_type, {}).get("eigenvalues", [])
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _load_features_for_pairs(spec: DatasetSpec, pairs: list[ClonePair]) -> dict:
    """Load only spectral feature rows needed by the selected pairs."""
    if not spec.features_manifest.exists():
        raise FileNotFoundError(f"Spectral features manifest not found: {spec.features_manifest}")
    needed_ids = {str(pair.left_id) for pair in pairs} | {str(pair.right_id) for pair in pairs}
    features_db = _load_features_db(str(spec.features_manifest), needed_ids=needed_ids)
    return features_db or {}


def _pair_has_eigenvalues(pair: ClonePair, features_db: dict, graph_type: str) -> bool:
    left = _eigenvalues_for(features_db, pair.left_id, graph_type)
    right = _eigenvalues_for(features_db, pair.right_id, graph_type)
    return left.size > 0 and right.size > 0


def _select_pairs_for_labels(
    pairs: list[ClonePair],
    features_db: dict,
    graph_type: str,
    per_label: int = 2,
    seed: int = 42,
) -> list[ClonePair]:
    """Select true clone and true non-clone pairs that have usable spectra."""
    rng = random.Random(seed)
    selected: list[ClonePair] = []
    for label in (1, 0):
        candidates = [pair for pair in pairs if pair.label == label and _pair_has_eigenvalues(pair, features_db, graph_type)]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_label])
    return selected


def plot_pair_eigenvalue_comparison(pair: ClonePair, features_db: dict, graph_type: str = "cpg") -> None:
    """Plot eigenvalue arrays for both snippets in a pair on the same axis."""
    left = _eigenvalues_for(features_db, pair.left_id, graph_type)
    right = _eigenvalues_for(features_db, pair.right_id, graph_type)
    max_len = max(left.size, right.size)
    if max_len == 0:
        _md(f"No eigenvalues available for pair {pair.left_id}-{pair.right_id}.")
        return

    plt.figure(figsize=(10, 3.8))
    plt.plot(np.arange(left.size), left, marker="o", linewidth=1.3, markersize=2.5, label=f"{pair.left_id}")
    plt.plot(np.arange(right.size), right, marker="s", linewidth=1.3, markersize=2.5, label=f"{pair.right_id}")
    plt.title(f"Eigenvalue distribution comparison ({graph_type.upper()})")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.legend(title="Snippet")
    plt.tight_layout()
    plt.show()


def display_clone_nonclone_pair_inspection(
    spec: DatasetSpec,
    graph_type: str = "cpg",
    per_label: int = 2,
    seed: int = 42,
    max_code_lines: int = 70,
    max_nodes: int = 80,
) -> list[ClonePair]:
    """Display clone/non-clone code pairs, graphs, and eigenvalue comparisons."""
    configure_notebook_style()
    _md(f"## Visual Step 2 - Clone vs. non-clone pair inspection ({graph_type.upper()})")
    pairs = load_pairs_for_spec(spec)
    sample_for_loading = _balanced_sample_pairs(pairs, sample_size=None, seed=seed)
    features_db = _load_features_for_pairs(spec, sample_for_loading)
    selected_pairs = _select_pairs_for_labels(sample_for_loading, features_db, graph_type, per_label=per_label, seed=seed)

    if not selected_pairs:
        _md("No clone/non-clone pairs with usable spectral features were found.")
        return []

    for pair in selected_pairs:
        label_name = "Clone" if pair.label == 1 else "Non-clone"
        _md(f"### {label_name} pair: {pair.left_id} vs {pair.right_id}")
        display_pair_code(pair, max_lines=max_code_lines)
        display_pair_graphs(spec, pair, graph_types=[graph_type], max_nodes=max_nodes)
        plot_pair_eigenvalue_comparison(pair, features_db, graph_type=graph_type)
    return selected_pairs


def compute_metric_score_dataframe(
    spec: DatasetSpec,
    sample_size: int | None = 1000,
    graph_types: list[str] | None = None,
    seed: int = 42,
    negative_ratio: float = 1.0,
) -> pd.DataFrame:
    """Compute PSS similarity and raw Wasserstein distance for clone/non-clone pairs."""
    from spectral_code.similarity.pss import PSSSimilarity

    pairs = load_pairs_for_spec(spec, negative_ratio=negative_ratio)
    pairs = _balanced_sample_pairs(pairs, sample_size=sample_size, seed=seed)
    features_db = _load_features_for_pairs(spec, pairs)
    graph_types = graph_types or _manifest_graph_types(spec)
    pss_metric = PSSSimilarity()
    rows = []
    skipped = []

    for graph_type in graph_types:
        for pair in pairs:
            left = _eigenvalues_for(features_db, pair.left_id, graph_type)
            right = _eigenvalues_for(features_db, pair.right_id, graph_type)
            if left.size == 0 or right.size == 0:
                skipped.append({"graph_type": graph_type, "left_id": pair.left_id, "right_id": pair.right_id})
                continue
            rows.append(
                {
                    "dataset": spec.display_name,
                    "graph_type": graph_type,
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "label": pair.label,
                    "label_name": "Clone" if pair.label == 1 else "Non-clone",
                    "metric": "PSS score",
                    "score": pss_metric.compute(left, right),
                    "direction": "higher_is_clone",
                }
            )
            rows.append(
                {
                    "dataset": spec.display_name,
                    "graph_type": graph_type,
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "label": pair.label,
                    "label_name": "Clone" if pair.label == 1 else "Non-clone",
                    "metric": "Wasserstein distance",
                    "score": float(wasserstein_distance(left, right)),
                    "direction": "lower_is_clone",
                }
            )

    score_df = pd.DataFrame(rows)
    if skipped:
        skipped_df = pd.DataFrame(skipped).drop_duplicates()
        score_df.attrs["skipped_pairs"] = skipped_df
    return score_df


def plot_metric_distribution_overlay(score_df: pd.DataFrame) -> None:
    """Plot clone/non-clone density overlays for PSS and Wasserstein metrics."""
    configure_notebook_style()
    if score_df.empty:
        _md("No metric scores were available for distribution plots.")
        return

    palette = {"Clone": "#2E8B57", "Non-clone": "#C44E52"}
    g = sns.displot(
        data=score_df,
        x="score",
        hue="label_name",
        row="metric",
        col="graph_type",
        kind="hist",
        kde=True,
        stat="density",
        common_norm=False,
        bins=36,
        palette=palette,
        facet_kws={"sharex": False, "sharey": False},
        height=2.8,
        aspect=1.2,
    )
    g.set_axis_labels("Metric value", "Density")
    g.fig.suptitle("Visual Step 3 - Clone vs. non-clone score distributions", y=1.03)
    plt.show()


def display_statistical_distribution_plots(
    spec: DatasetSpec,
    sample_size: int | None = 1000,
    graph_types: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute and display statistical distribution plots for PSS and Wasserstein."""
    _md("## Visual Step 3 - Statistical distribution plots")
    score_df = compute_metric_score_dataframe(spec, sample_size=sample_size, graph_types=graph_types, seed=seed)
    if not score_df.empty:
        summary = (
            score_df.groupby(["graph_type", "metric", "label_name"])["score"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        display(summary)
    skipped = score_df.attrs.get("skipped_pairs")
    if isinstance(skipped, pd.DataFrame) and not skipped.empty:
        _md("Pairs skipped because spectral features were missing")
        display(skipped.head(25))
    plot_metric_distribution_overlay(score_df)
    return score_df


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Build threshold candidates from unique scores and adjacent midpoints."""
    unique = np.unique(scores)
    if unique.size == 0:
        return unique
    candidates = {float(unique[0]), float(unique[-1])}
    candidates.update(float(value) for value in unique)
    if unique.size > 1:
        candidates.update(float(value) for value in (unique[:-1] + unique[1:]) / 2.0)
    return np.asarray(sorted(candidates), dtype=np.float64)


def run_threshold_sweep(score_df: pd.DataFrame, optimize_for: str = "f1") -> tuple[pd.DataFrame, dict]:
    """Sweep thresholds for every graph/metric and return all rows plus the best config."""
    rows = []
    if score_df.empty:
        return pd.DataFrame(), {}

    for (graph_type, metric), group in score_df.groupby(["graph_type", "metric"]):
        labels = group["label"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=np.float64)
        direction = str(group["direction"].iloc[0])
        if len(np.unique(labels)) < 2:
            continue

        auc_scores = -scores if direction == "lower_is_clone" else scores
        try:
            roc_auc = float(roc_auc_score(labels, auc_scores))
        except ValueError:
            roc_auc = float("nan")

        for threshold in _candidate_thresholds(scores):
            preds = (scores <= threshold).astype(int) if direction == "lower_is_clone" else (scores >= threshold).astype(int)
            row = {
                "graph_type": graph_type,
                "metric": metric,
                "direction": direction,
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "roc_auc": roc_auc,
                "support": int(labels.size),
            }
            rows.append(row)

    sweep_df = pd.DataFrame(rows)
    if sweep_df.empty:
        return sweep_df, {}
    best_row = sweep_df.sort_values([optimize_for, "roc_auc", "accuracy"], ascending=False).iloc[0].to_dict()
    return sweep_df, best_row


def display_global_threshold_tuning_summary(
    spec: DatasetSpec,
    graph_types: list[str] | None = None,
    optimize_for: str = "f1",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Run a full-subdataset threshold sweep and print the best classification report."""
    configure_notebook_style()
    _md("## Visual Step 4 - Global hyperparameter tuning and metrics summary")
    score_df = compute_metric_score_dataframe(spec, sample_size=None, graph_types=graph_types, seed=seed)
    sweep_df, best = run_threshold_sweep(score_df, optimize_for=optimize_for)
    if sweep_df.empty or not best:
        _md("No valid threshold sweep could be computed.")
        return sweep_df, {}

    display(
        sweep_df.sort_values([optimize_for, "roc_auc", "accuracy"], ascending=False)
        .head(20)
        .reset_index(drop=True)
    )

    best_scores = score_df[
        (score_df["graph_type"] == best["graph_type"]) & (score_df["metric"] == best["metric"])
    ].copy()
    labels = best_scores["label"].to_numpy(dtype=int)
    scores = best_scores["score"].to_numpy(dtype=np.float64)
    if best["direction"] == "lower_is_clone":
        preds = (scores <= best["threshold"]).astype(int)
        auc_scores = -scores
    else:
        preds = (scores >= best["threshold"]).astype(int)
        auc_scores = scores

    report = classification_report(labels, preds, target_names=["Non-clone", "Clone"], zero_division=0)
    accuracy = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, auc_scores)
    _md(
        "\n".join(
            [
                "### Optimal threshold report",
                f"- Graph: `{best['graph_type']}`",
                f"- Metric: `{best['metric']}`",
                f"- Threshold: `{best['threshold']:.6f}`",
                f"- Accuracy: `{accuracy:.4f}`",
                f"- ROC-AUC: `{roc_auc:.4f}`",
            ]
        )
    )
    print(report)

    plt.figure(figsize=(8.5, 4.5))
    palette = {"Clone": "#2E8B57", "Non-clone": "#C44E52"}
    sns.histplot(
        data=best_scores,
        x="score",
        hue="label_name",
        kde=True,
        stat="density",
        common_norm=False,
        bins=40,
        palette=palette,
    )
    plt.axvline(best["threshold"], color="black", linestyle="--", linewidth=1.6, label="Optimal threshold")
    plt.title(f"Best threshold separation: {best['graph_type']} / {best['metric']}")
    plt.xlabel(best["metric"])
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return sweep_df, best


def run_tuning_for_spec(
    spec: DatasetSpec,
    pairs: list[ClonePair],
    metrics: list[str] | None = None,
    graph_types: list[str] | None = None,
    k_values: list[int | None] | None = None,
    n_samples: int | None = None,
    optimize_for: str = "f1",
    out_filename: str | None = None,
) -> list[dict] | None:
    from spectral_code.evaluation.tuning import run_fast_grid_search_on_pairs

    spec.output_root.mkdir(parents=True, exist_ok=True)
    if out_filename is None:
        out_filename = f"trained_{spec.key}_{optimize_for}_pss_wasserstein.json"

    return run_fast_grid_search_on_pairs(
        features_db_path=str(spec.features_manifest),
        pairs=pairs,
        n_samples=n_samples,
        optimize_for=optimize_for,
        out_filename=out_filename,
        graph_types=graph_types or GRAPH_TYPES_DEFAULT,
        k_values=k_values or [None],
        metrics=metrics or METRICS_DEFAULT,
    )


def save_json_report(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_dataframe_report(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
