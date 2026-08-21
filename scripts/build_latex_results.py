"""Generate the publication LaTeX tables from the same CSV artifacts as the PNG package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_requested_table_package import (
    ARTIFACTS,
    BENCHMARKS,
    METRICS,
    OUR_METHOD_VARIANTS,
    REPRESENTATIVES,
    add_local_spectral_results,
    family,
    merge_feature_ablation_into_general,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "paper" / "results.tex"
FAMILY_ORDER = ["Graph-based", "Spectral representations", "Hybrid graph + code", "Raw code", "Our method"]


def tex(value: object) -> str:
    text = str(value)
    return (text.replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
                .replace("%", r"\%").replace("$", r"\$"))


def metric(value: object, best: bool = False) -> str:
    if pd.isna(value):
        return "--"
    result = f"{float(value):.3f}"
    return rf"\best{{{result}}}" if best else result


def parameter(value: object, method: str) -> str:
    if method.endswith(" + No Train"):
        return "0"
    if pd.isna(value):
        return "N/A"
    value = float(value)
    return f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.1f}K"


def long_header(first_columns: list[str], groups: list[str]) -> list[str]:
    columns = "l" * len(first_columns) + "r" * (4 * len(groups))
    top = " & ".join(["" for _ in first_columns] + [rf"\multicolumn{{4}}{{c}}{{{tex(group)}}}" for group in groups]) + r" \\"
    sub = " & ".join(first_columns + list(METRICS) * len(groups)) + r" \\"
    return [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", top, r"\cmidrule(lr){%d-%d}" % (len(first_columns) + 1, len(first_columns) + 4 * len(groups)), sub, r"\midrule"]


def family_rows(frame: pd.DataFrame, method_names: dict[str, list[str]], benchmarks: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for family_name in FAMILY_ORDER:
        methods = method_names.get(family_name, [])
        if not methods:
            continue
        for index, method_name in enumerate(methods):
            values = [rf"\multirow{{{len(methods)}}}{{*}}{{{tex(family_name)}}}" if index == 0 else "", tex(method_name)]
            for benchmark in benchmarks:
                item = frame[(frame.Method.eq(method_name)) & (frame.Benchmark.eq(benchmark))]
                item = item.iloc[0] if not item.empty else None
                highest = pd.to_numeric(frame[frame.Benchmark.eq(benchmark)].F1, errors="coerce").max()
                for name in METRICS:
                    values.append(metric(item[name] if item is not None else np.nan,
                                         name == "F1" and item is not None and np.isclose(float(item[name]), highest)))
            lines.append(" & ".join(values) + r" \\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    return lines


def general_table(raw: pd.DataFrame, ablation: pd.DataFrame) -> str:
    frame = merge_feature_ablation_into_general(raw, ablation).copy()
    frame["Family"] = frame.Method.map(family)
    names: dict[str, list[str]] = {}
    for current_family in FAMILY_ORDER:
        if current_family == "Our method":
            names[current_family] = [name for _, name in OUR_METHOD_VARIANTS if name in set(frame.Method)]
        else:
            names[current_family] = sorted(frame[frame.Family.eq(current_family)].Method.unique())
    lines = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\scriptsize",
             r"\caption{General benchmark comparison on official test sets. Green cells mark the best F1 in each benchmark.}", r"\label{tab:general}", r"\resizebox{\linewidth}{!}{%"]
    lines.extend(long_header(["Method family", "Method"], list(BENCHMARKS)))
    lines.extend(family_rows(frame, names, BENCHMARKS))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}",
                  r"\vspace{2pt}\parbox{\linewidth}{\footnotesize Our method is expanded into the four full-data readout/input variants from Experiment~4.}",
                  r"\end{table}", r"\end{landscape}", r"\clearpage"])
    return "\n".join(lines)


def parameter_table(raw: pd.DataFrame) -> str:
    frame = raw.copy(); frame["Family"] = frame.Method.map(family)
    lines = [r"\begin{table}[p]", r"\centering", r"\small",
             r"\caption{Trainable parameter count by method. N/A denotes a non-neural classical model.}", r"\label{tab:parameters}",
             r"\begin{tabular}{llrrrr}", r"\toprule", r"Method family & Method & BigCloneBench & SemanticCloneBench & GPTCloneBench & ATCoder \\", r"\midrule"]
    for current_family in FAMILY_ORDER:
        methods = sorted(frame[frame.Family.eq(current_family)].Method.unique())
        for index, method_name in enumerate(methods):
            row = [rf"\multirow{{{len(methods)}}}{{*}}{{{tex(current_family)}}}" if index == 0 else "", tex(method_name)]
            for benchmark in BENCHMARKS:
                item = frame[(frame.Method.eq(method_name)) & (frame.Benchmark.eq(benchmark))]
                row.append(parameter(item.iloc[0].get("TrainableParameters", np.nan), method_name) if not item.empty else "--")
            lines.append(" & ".join(row) + r" \\")
        if methods:
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule": lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\clearpage"])
    return "\n".join(lines)


def language_display(value: str) -> str:
    names = {"java": "Java", "python": "Python", "c": "C", "csharp": r"C\#"}
    return r" $\leftrightarrow$ ".join(names.get(part, tex(part)) for part in value.split("->"))


def language_table(language: pd.DataFrame) -> str:
    frame = language.copy()
    frame["Setting"] = np.where(frame.Language.str.contains("->", regex=False), "Cross-language", "Single-language")
    rank = {name: index for index, name in enumerate(BENCHMARKS)}
    strata = frame[["Setting", "Benchmark", "Language"]].drop_duplicates()
    strata["SettingOrder"] = strata.Setting.map({"Single-language": 0, "Cross-language": 1})
    strata["BenchmarkOrder"] = strata.Benchmark.map(rank)
    strata = strata.sort_values(["SettingOrder", "BenchmarkOrder", "Language"])
    rows = list(strata[["Setting", "Benchmark", "Language"]].itertuples(index=False, name=None))
    methods = [method for _, method in REPRESENTATIVES]
    lines = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\scriptsize",
             r"\caption{Language-wise evaluation using one representative per method family.}", r"\label{tab:language}", r"\resizebox{\linewidth}{!}{%"]
    lines.extend(long_header(["Setting", "Language pair", "Dataset"], [label for label, _ in REPRESENTATIVES]))
    for index, (setting, benchmark, language_name) in enumerate(rows):
        count = sum(item[0] == setting for item in rows)
        first = index == 0 or rows[index - 1][0] != setting
        row = [rf"\multirow{{{count}}}{{*}}{{{tex(setting)}}}" if first else "", language_display(language_name), tex(benchmark)]
        available = frame[(frame.Setting.eq(setting)) & (frame.Benchmark.eq(benchmark)) & (frame.Language.eq(language_name))]
        highest = pd.to_numeric(available[available.Method.isin(methods)].F1, errors="coerce").max()
        for method_name in methods:
            item = available[available.Method.eq(method_name)]
            item = item.iloc[0] if not item.empty else None
            for name in METRICS:
                row.append(metric(item[name] if item is not None else np.nan,
                                  name == "F1" and item is not None and np.isclose(float(item[name]), highest)))
        lines.append(" & ".join(row) + r" \\")
        if index + 1 < len(rows) and rows[index + 1][0] != setting:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", r"\end{landscape}", r"\clearpage"])
    return "\n".join(lines)


def experiment_one(latent: pd.DataFrame) -> str:
    labels = {16: "16 latent nodes", 24: "24 latent nodes", 32: "32 latent nodes", 48: "48 latent nodes"}
    lines = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\small", r"\caption{Experiment 1: latent graph capacity.}", r"\label{tab:latent}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrrrrrrrrrrr}", r"\toprule"]
    lines.extend(long_header(["Configuration"], list(BENCHMARKS))[2:])
    for nodes, label in labels.items():
        row = [label]
        for benchmark in BENCHMARKS:
            item = latent[(latent.LatentNodes.eq(nodes)) & (latent.Benchmark.eq(benchmark))]
            item = item.iloc[0] if not item.empty else None
            highest = pd.to_numeric(latent[latent.Benchmark.eq(benchmark)].F1, errors="coerce").max()
            for name in METRICS:
                row.append(metric(item[name] if item is not None else np.nan, name == "F1" and item is not None and np.isclose(float(item[name]), highest)))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", r"\end{landscape}", r"\clearpage"])
    return "\n".join(lines)


def cross_dataset_table(frame: pd.DataFrame) -> str:
    targets = ("ATCoder", "GPTCloneBench", "SemanticCloneBench")
    methods = ("ASTNN", "RtvNN", "DeepSim", "SPECTRA-Siam")
    lines = [r"\begin{table}[p]", r"\centering", r"\small", r"\caption{Experiment 2: zero-shot transfer trained on 250k BigCloneBench pairs.}", r"\label{tab:crossdataset}", r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule"]
    lines.extend(long_header(["Method"], list(targets))[2:])
    for method_name in methods:
        row = [method_name]
        for benchmark in targets:
            item = frame[(frame.Method.eq(method_name)) & (frame.Benchmark.eq(benchmark))]
            item = item.iloc[0] if not item.empty else None
            highest = pd.to_numeric(frame[frame.Benchmark.eq(benchmark)].F1, errors="coerce").max()
            for name in METRICS:
                row.append(metric(item[name] if item is not None else np.nan, name == "F1" and item is not None and np.isclose(float(item[name]), highest)))
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\clearpage"])
    return "\n".join(lines)


def cross_language_table(frame: pd.DataFrame) -> str:
    methods = ("ASTNN", "RtvNN", "DeepSim", "SPECTRA-Siam")
    order = {"java": 0, "python": 1, "c": 2, "csharp": 3}
    conditions = sorted(frame[["TrainedOnLanguage", "TestLanguage"]].drop_duplicates().itertuples(index=False, name=None), key=lambda item: (order.get(item[0], 99), order.get(item[1], 99)))
    names = {"java": "Java", "python": "Python", "c": "C", "csharp": r"C\#"}
    lines = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\scriptsize", r"\caption{Experiment 3: cross-language transfer on SemanticCloneBench.}", r"\label{tab:crosslanguage}", r"\resizebox{\linewidth}{!}{%"]
    lines.extend(long_header(["Train language", "Test language"], list(methods)))
    for index, (source, target) in enumerate(conditions):
        count = sum(left == source for left, _ in conditions)
        first = index == 0 or conditions[index - 1][0] != source
        row = [rf"\multirow{{{count}}}{{*}}{{{names.get(source, tex(source))}}}" if first else "", names.get(target, tex(target))]
        available = frame[(frame.TrainedOnLanguage.eq(source)) & (frame.TestLanguage.eq(target))]
        highest = pd.to_numeric(available.F1, errors="coerce").max()
        for method_name in methods:
            item = available[available.Method.eq(method_name)]
            item = item.iloc[0] if not item.empty else None
            for name in METRICS:
                row.append(metric(item[name] if item is not None else np.nan, name == "F1" and item is not None and np.isclose(float(item[name]), highest)))
        lines.append(" & ".join(row) + r" \\")
        if index + 1 < len(conditions) and conditions[index + 1][0] != source:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", r"\end{landscape}"])
    return "\n".join(lines)


def main() -> None:
    raw = add_local_spectral_results(pd.read_csv(ARTIFACTS / "all_results_long.csv"))
    language = pd.read_csv(ARTIFACTS / "language_breakdown_long.csv")
    latent = pd.read_csv(ARTIFACTS / "latent_capacity_long.csv")
    ablation = pd.read_csv(ARTIFACTS / "feature_ablation_long.csv")
    cross_dataset = pd.read_csv(ARTIFACTS / "cross_dataset_transfer_long.csv")
    cross_language = pd.read_csv(ARTIFACTS / "cross_language_transfer_long.csv")
    preamble = r"""\documentclass[10pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage{booktabs,multirow,graphicx,pdflscape}
\usepackage[table]{xcolor}
\definecolor{bestgreen}{RGB}{187,247,208}
\newcommand{\best}[1]{\cellcolor{bestgreen}\textbf{#1}}
\title{Spectral CCD Results}
\date{}
\begin{document}
\maketitle
\section{Results}
"""
    body = [general_table(raw, ablation), parameter_table(raw), language_table(language), experiment_one(latent), cross_dataset_table(cross_dataset), cross_language_table(cross_language)]
    TARGET.write_text(preamble + "\n\n".join(body) + "\n\\end{document}\n", encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
