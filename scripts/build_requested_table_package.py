"""Build exactly the four tables requested for the current paper checkpoint."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "research" / "latent_graph_learning" / "reports" / "artifacts"
OUT = ROOT.parent / "outputs" / "paper_tables_current"
LOCAL_SPECTRAL = ROOT.parent / "outputs" / "local_spectral_representation_baselines"
BENCHMARKS = ("BigCloneBench", "SemanticCloneBench", "GPTCloneBench", "ATCoder")
METRICS = ("P", "R", "F1", "Acc")
REPRESENTATIVES = (
    ("Our method", "SPECTRA-Siam"),
    ("Graph-based", "ASTNN"),
    ("Hybrid", "DeepSim"),
    ("Raw code", "RtvNN"),
)
OUR_METHOD_VARIANTS = (
    ("proposed_eigen_only", "Proposed eigenvalue-only"),
    ("topology_only", "Topology only"),
    ("canonical", "Canonical graph"),
    ("lexical", "Canonical + source lexical"),
)


def metric(value: object) -> str:
    return "-" if pd.isna(value) else f"{float(value):.3f}"


def family(method: str) -> str:
    if method == "SPECTRA-Siam" or method in {name for _, name in OUR_METHOD_VARIANTS}:
        return "Our method"
    if method == "DeepSim":
        return "Hybrid graph + code"
    if method in {"Deckard", "RtvNN", "CDLH"}:
        return "Raw code"
    if any(method.endswith(f" + {learner}") for learner in ("No Train", "RF", "LR", "SNN")):
        return "Spectral representations"
    # GNNs, ASTNN, FA-AST models, and spectral features derived from a graph
    # all consume graph structure; keep them in the same graph-based family.
    return "Graph-based"


def add_local_spectral_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Include No-Train/RF/LR outputs produced by the local spectral runner."""
    benchmark_for_dataset = {
        "CODEXGLUE": "BigCloneBench", "SEMANTICCLONEBENCH": "SemanticCloneBench",
        "GPTCLONEBENCH": "GPTCloneBench", "ATCODER": "ATCoder",
    }
    additions = []
    for path in sorted(LOCAL_SPECTRAL.glob("*/*/*_spectral_representation_results.csv")):
        frame = pd.read_csv(path)
        if "Status" in frame:
            frame = frame[frame["Status"].eq("ok")].copy()
        if frame.empty:
            continue
        benchmark = benchmark_for_dataset.get(str(frame["Dataset"].iloc[0]).upper())
        if benchmark is None:
            continue
        additions.append(pd.DataFrame({
            "Benchmark": benchmark, "Method": frame["Method"].astype(str),
            "P": pd.to_numeric(frame["P"]), "R": pd.to_numeric(frame["R"]),
            "F1": pd.to_numeric(frame["F1"]), "Acc": pd.to_numeric(frame["Acc"]),
            "RuntimeMinutes": pd.to_numeric(frame.get("RuntimeMinutes")),
            "TrainPairs": pd.to_numeric(frame.get("TrainPairs")),
            "ValidPairs": pd.to_numeric(frame.get("ValidPairs")),
            "TestPairs": pd.to_numeric(frame.get("TestPairs")),
            "TrainableParameters": (pd.to_numeric(frame["TrainableParameters"], errors="coerce")
                                   if "TrainableParameters" in frame else np.nan),
            "RunProfile": "final_full", "Source": str(path),
        }))
    if not additions:
        return raw
    combined = pd.concat([raw, *additions], ignore_index=True)
    return combined.drop_duplicates(["Benchmark", "Method"], keep="first")


def center_group_label(rows, groups, column: int) -> None:
    """Visually merge consecutive equal group values in a table column."""
    start = 0
    while start < len(groups):
        end = start
        while end + 1 < len(groups) and groups[end + 1] == groups[start]:
            end += 1
        label = groups[start]
        for row in range(start, end + 1):
            rows[row][column] = ""
        rows[(start + end) // 2][column] = label
        start = end + 1


def save_table(rows, headers, title, path, widths, spans, best_cells=(), groups=(), note=None, merge_columns=()):
    figure_width = max(14, min(45, 1.05 * len(headers) + 4))
    figure_height = max(4.8, 0.34 * (len(rows) + 6))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center",
                       colWidths=widths, bbox=[0.005, 0.035, 0.99, 0.83])
    table.auto_set_font_size(False); table.set_fontsize(7.0 if len(headers) > 18 else 7.8); table.scale(1, 1.15)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        if row == 0:
            cell.set_facecolor("#e2e8f0"); cell.set_text_props(weight="bold")
    for row, column in best_cells:
        table[(row, column)].set_facecolor("#bbf7d0"); table[(row, column)].set_text_props(weight="bold")
    for row in range(2, len(groups) + 1):
        if groups[row - 1] != groups[row - 2]:
            for column in range(len(headers)):
                table[(row, column)].set_linewidth(1.2)
    for column in merge_columns:
        start = 0
        while start < len(groups):
            end = start
            while end + 1 < len(groups) and groups[end + 1] == groups[start]:
                end += 1
            if end > start:
                for row in range(start + 1, end + 2):
                    cell = table[(row, column)]
                    cell.visible_edges = "LR" if row <= end else "BLR"
                table[(start + 1, column)].visible_edges = "TLR"
            start = end + 1
    cumulative = np.cumsum([0.0, *widths])
    for label, first, last in spans:
        axis.text((cumulative[first] + cumulative[last + 1]) / 2, 0.895, label,
                  transform=axis.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold")
    figure.suptitle(title, fontweight="bold", fontsize=14, y=0.985)
    if note:
        figure.text(.5, .006, note, ha="center", fontsize=7.2, style="italic")
    figure.savefig(path, dpi=260, bbox_inches="tight"); plt.close(figure)


def merge_feature_ablation_into_general(raw: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    """Replace the single canonical row with the four full-data controlled variants."""
    base = raw[~raw.Method.eq("SPECTRA-Siam")].copy()
    variants = []
    for variant, display in OUR_METHOD_VARIANTS:
        selected = ablation[ablation.Variant.eq(variant)].copy()
        if selected.empty:
            continue
        selected["Method"] = display
        variants.append(selected)
    if not variants:
        return raw
    return pd.concat([base, *variants], ignore_index=True, sort=False)


def make_general(raw: pd.DataFrame, ablation: pd.DataFrame) -> None:
    order = ["Graph-based", "Spectral representations", "Hybrid graph + code", "Raw code", "Our method"]
    raw = merge_feature_ablation_into_general(raw, ablation).assign(Family=lambda frame: frame.Method.map(family))
    rows, group_ids, best = [], [], set()
    for current_family in order:
        if current_family == "Our method":
            methods = [name for _, name in OUR_METHOD_VARIANTS if name in set(raw.Method)]
        else:
            methods = sorted(raw[raw.Family.eq(current_family)].Method.unique())
        for index, method_name in enumerate(methods):
            row = [current_family if index == 0 else "", method_name]
            for benchmark in BENCHMARKS:
                value = raw[(raw.Method.eq(method_name)) & (raw.Benchmark.eq(benchmark))]
                item = value.iloc[0] if not value.empty else None
                row.extend(metric(item[name]) if item is not None else "-" for name in METRICS)
            rows.append(row); group_ids.append(current_family)
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        top = raw[raw.Benchmark.eq(benchmark)].F1.max()
        for row_index, row in enumerate(rows, start=1):
            value = raw[(raw.Method.eq(row[1])) & (raw.Benchmark.eq(benchmark))].F1
            if not value.empty and np.isclose(value.iloc[0], top): best.add((row_index, 2 + benchmark_index * 4 + 2))
    center_group_label(rows, group_ids, 0)
    widths = [.14, .13] + [(1 - .27) / 16] * 16
    save_table(rows, ["Method family", "Method"] + list(METRICS) * 4,
               "General benchmark comparison — official test sets", OUT / "01_general_benchmark_table.png", widths,
               [(name, 2 + i * 4, 5 + i * 4) for i, name in enumerate(BENCHMARKS)], best, group_ids,
                "Our method is expanded into the four full-data Experiment-4 readout/input variants. Spectral-representation baselines use graph spectra with No Train/RF/LR/SNN; DeepSim combines graph and source-code features; RtvNN/CDLH/Deckard use raw code.",
               merge_columns=(0,))


def pretty_language(value: str) -> str:
    names = {"java": "Java", "python": "Python", "c": "C", "csharp": "C#"}
    return " ↔ ".join(names.get(item, item) for item in value.split("->"))


def make_language(language: pd.DataFrame) -> None:
    language = language.copy()
    language["Setting"] = np.where(language.Language.str.contains("->", regex=False), "Cross-language", "Single-language")
    rank = {name: i for i, name in enumerate(BENCHMARKS)}
    strata = language[["Setting", "Benchmark", "Language"]].drop_duplicates()
    strata["_setting_order"] = strata["Setting"].map({"Single-language": 0, "Cross-language": 1})
    strata["_benchmark_order"] = strata["Benchmark"].map(rank)
    strata = strata.sort_values(["_setting_order", "_benchmark_order", "Language"]).drop(columns=["_setting_order", "_benchmark_order"])
    rows, groups, best = [], [], set()
    for row_index, item in enumerate(strata.itertuples(index=False), start=1):
        row = [item.Setting, pretty_language(item.Language), item.Benchmark]
        for _, method_name in REPRESENTATIVES:
            candidate = language[(language.Benchmark.eq(item.Benchmark)) & (language.Language.eq(item.Language)) & (language.Method.eq(method_name))]
            result = candidate.iloc[0] if not candidate.empty else None
            row.extend(metric(result[name]) if result is not None else "-" for name in METRICS)
        rows.append(row); groups.append(item.Setting)
        for method_index, (_, method_name) in enumerate(REPRESENTATIVES):
            values = language[(language.Benchmark.eq(item.Benchmark)) & (language.Language.eq(item.Language)) & (language.Method.eq(method_name))].F1
            if not values.empty:
                available = language[(language.Benchmark.eq(item.Benchmark)) & (language.Language.eq(item.Language)) & (language.Method.isin([x[1] for x in REPRESENTATIVES]))].F1.max()
                if np.isclose(values.iloc[0], available): best.add((row_index, 3 + method_index * 4 + 2))
    center_group_label(rows, groups, 0)
    widths = [.08, .10, .12] + [(1 - .30) / 16] * 16
    spans = [(f"{label}\n{method}", 3 + index * 4, 6 + index * 4) for index, (label, method) in enumerate(REPRESENTATIVES)]
    save_table(rows, ["Setting", "Language pair", "Dataset"] + list(METRICS) * 4,
               "Language-wise evaluation — selected representative of each method family",
               OUT / "03_language_breakdown_table.png", widths, spans, best, groups,
               "Representatives selected from the general table: SPECTRA-Siam (ours), ASTNN (graph-based), DeepSim (hybrid), and RtvNN (raw code).",
               merge_columns=(0,))


def parameter_count(value: object, method: str) -> str:
    if method.endswith(" + No Train"):
        return "0"
    if pd.isna(value):
        return "N/A"
    value = float(value)
    return f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.1f}K"


def make_parameter_table(raw: pd.DataFrame) -> None:
    raw = raw.assign(Family=raw.Method.map(family))
    order = ["Graph-based", "Spectral representations", "Hybrid graph + code", "Raw code", "Our method"]
    rows, groups = [], []
    for current_family in order:
        for method_name in sorted(raw[raw.Family.eq(current_family)].Method.unique()):
            row = ["", method_name]
            for benchmark in BENCHMARKS:
                candidate = raw[(raw.Method.eq(method_name)) & (raw.Benchmark.eq(benchmark))]
                item = candidate.iloc[0] if not candidate.empty else None
                row.append(parameter_count(item.get("TrainableParameters", np.nan), method_name) if item is not None else "-")
            rows.append(row); groups.append(current_family)
    center_group_label(rows, groups, 0)
    save_table(rows, ["Method family", "Method", *BENCHMARKS],
               "Trainable parameter count by method", OUT / "02_trainable_parameter_counts.png",
               [.22, .28] + [.125] * 4,
               [("Trainable parameters", 2, 5)], groups=groups,
               note="N/A = non-neural classical model (RF/LR), for which trainable neural-network parameter count is not defined.",
               merge_columns=(0,))


def make_experiment(frame: pd.DataFrame, row_key: str, values, labels, title: str, filename: str) -> None:
    # Experiment 4 is intentionally represented by the four Our-method rows
    # in the general table, rather than repeated as a standalone figure.
    if row_key == "Variant":
        return
    if row_key == "LatentNodes":
        filename = "04_experiment_1_latent_capacity.png"
    rows, best = [], set()
    for row_index, value in enumerate(values, start=1):
        row = [labels[value]]
        for benchmark in BENCHMARKS:
            candidate = frame[(frame.Benchmark.eq(benchmark)) & (frame[row_key].eq(value))]
            item = candidate.iloc[0] if not candidate.empty else None
            row.extend(metric(item[name]) if item is not None else "-" for name in METRICS)
        rows.append(row)
    for index, benchmark in enumerate(BENCHMARKS):
        top = frame[frame.Benchmark.eq(benchmark)].F1.max()
        for row_index, value in enumerate(values, start=1):
            score = frame[(frame.Benchmark.eq(benchmark)) & (frame[row_key].eq(value))].F1
            if not score.empty and np.isclose(score.iloc[0], top): best.add((row_index, 1 + index * 4 + 2))
    widths = [.20] + [.80 / 16] * 16
    save_table(rows, ["Configuration"] + list(METRICS) * 4, title, OUT / filename, widths,
               [(name, 1 + i * 4, 4 + i * 4) for i, name in enumerate(BENCHMARKS)], best)


def transfer_methods(frame: pd.DataFrame) -> list[str]:
    preferred = ("ASTNN", "RtvNN", "DeepSim", "SPECTRA-Siam")
    return [method for method in preferred if method in set(frame.Method)]


def make_cross_dataset_transfer(frame: pd.DataFrame) -> None:
    targets = ("ATCoder", "GPTCloneBench", "SemanticCloneBench")
    methods = transfer_methods(frame)
    rows, best = [], set()
    for row_index, method in enumerate(methods, start=1):
        row = [method]
        for target in targets:
            candidate = frame[(frame.Method.eq(method)) & (frame.Benchmark.eq(target))]
            item = candidate.iloc[0] if not candidate.empty else None
            row.extend(metric(item[name]) if item is not None else "-" for name in METRICS)
        rows.append(row)
    for target_index, target in enumerate(targets):
        top = pd.to_numeric(frame[frame.Benchmark.eq(target)].F1, errors="coerce").max()
        for row_index, method in enumerate(methods, start=1):
            score = frame[(frame.Method.eq(method)) & (frame.Benchmark.eq(target))].F1
            if not score.empty and np.isclose(float(score.iloc[0]), top):
                best.add((row_index, 1 + target_index * 4 + 2))
    save_table(
        rows, ["Method"] + list(METRICS) * len(targets),
        "Experiment 2 — zero-shot transfer trained on BigCloneBench",
        OUT / "05_experiment_2_cross_dataset_transfer.png", [.16] + [.84 / 12] * 12,
        [(target, 1 + i * 4, 4 + i * 4) for i, target in enumerate(targets)], best,
        note="Every method trains once on the same 250k BigCloneBench pairs; the source-validation threshold is frozen on the three target test sets.",
    )


def make_cross_language_transfer(frame: pd.DataFrame) -> None:
    language_order = {"java": 0, "python": 1, "c": 2, "csharp": 3}
    conditions = sorted(
        frame[["TrainedOnLanguage", "TestLanguage"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda item: (language_order.get(item[0], 99), language_order.get(item[1], 99)),
    )
    methods = transfer_methods(frame)
    rows, groups, best = [], [], set()
    for row_index, (source, target) in enumerate(conditions, start=1):
        groups.append(source)
        row = [source, target]
        available = frame[(frame.TrainedOnLanguage.eq(source)) & (frame.TestLanguage.eq(target))]
        for method in methods:
            candidate = available[available.Method.eq(method)]
            item = candidate.iloc[0] if not candidate.empty else None
            row.extend(metric(item[name]) if item is not None else "-" for name in METRICS)
        top = pd.to_numeric(available.F1, errors="coerce").max()
        for method_index, method in enumerate(methods):
            score = available[available.Method.eq(method)].F1
            if not score.empty and np.isclose(float(score.iloc[0]), top):
                best.add((row_index, 2 + method_index * 4 + 2))
        rows.append(row)
    names = {"java": "Java", "python": "Python", "c": "C", "csharp": "C#"}
    for row in rows:
        row[0] = names.get(row[0], row[0]); row[1] = names.get(row[1], row[1])
    groups = [names.get(value, value) for value in groups]
    center_group_label(rows, groups, 0)
    save_table(
        rows, ["Train language", "Test language"] + list(METRICS) * len(methods),
        "Experiment 3 — cross-language transfer on SemanticCloneBench",
        OUT / "06_experiment_3_cross_language_transfer.png", [.09, .09] + [.82 / (4 * len(methods))] * (4 * len(methods)),
        [(method, 2 + i * 4, 5 + i * 4) for i, method in enumerate(methods)], best, groups,
        note="For each row, the model trains on the source language, chooses a threshold on source validation, and evaluates that frozen model on the target language.",
        merge_columns=(0,),
    )


def main() -> None:
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    raw = add_local_spectral_results(pd.read_csv(ARTIFACTS / "all_results_long.csv"))
    language = pd.read_csv(ARTIFACTS / "language_breakdown_long.csv")
    latent = pd.read_csv(ARTIFACTS / "latent_capacity_long.csv")
    ablation = pd.read_csv(ARTIFACTS / "feature_ablation_long.csv")
    cross_dataset = pd.read_csv(ARTIFACTS / "cross_dataset_transfer_long.csv")
    cross_language = pd.read_csv(ARTIFACTS / "cross_language_transfer_long.csv")
    make_general(raw, ablation); make_language(language)
    make_experiment(latent, "LatentNodes", [16, 24, 32, 48], {16: "16 latent nodes", 24: "24 latent nodes", 32: "32 latent nodes", 48: "48 latent nodes"}, "Experiment 1 — latent graph capacity", "03_experiment_1_latent_capacity.png")
    make_experiment(ablation, "Variant", ["proposed_eigen_only", "topology_only", "canonical", "lexical"], {"proposed_eigen_only": "Proposed: eigenvalue-only", "topology_only": "Topology only", "canonical": "Canonical graph", "lexical": "Canonical + source lexical"}, "Experiment 4 — SPECTRA-Siam readout and feature ablation", "04_experiment_4_feature_ablation.png")
    make_parameter_table(raw)
    make_cross_dataset_transfer(cross_dataset)
    make_cross_language_transfer(cross_language)
    (OUT / "README.txt").write_text("Current-paper tables. Green cells mark the best F1 in each comparison column.\n", encoding="utf-8")
    print("Created", OUT)


if __name__ == "__main__":
    main()
