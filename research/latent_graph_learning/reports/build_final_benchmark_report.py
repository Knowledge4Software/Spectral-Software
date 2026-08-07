"""Build publication-ready benchmark, language, and ablation report figures.

The input is the collection of Kaggle result ZIPs in ``../outputs/kaggle``.
The builder deliberately reads ZIPs in place and writes only into this
report's ``artifacts`` directory.  It also validates that a result embedded in
an archive belongs to the archive's benchmark, preventing accidental copy/paste
mix-ups from silently entering a paper table.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_ROOT = PROJECT_ROOT.parent / "outputs"
KAGGLE_ROOT = OUTPUTS_ROOT / "kaggle"
ARTIFACTS = PROJECT_ROOT / "research" / "latent_graph_learning" / "reports" / "artifacts"

FOLDER_TO_BENCHMARK = {
    "CodeXGlue": "BigCloneBench",  # Project reporting convention.
    "SemanticClone": "SemanticCloneBench",
    "GPTClone": "GPTCloneBench",
    "ATCoder": "ATCoder",
}
BENCHMARK_ORDER = tuple(FOLDER_TO_BENCHMARK.values())
BENCHMARK_TO_CLEAN_DATA = {
    "BigCloneBench": "codexglue_v3",
    "SemanticCloneBench": "semanticclonebench_v3",
    "GPTCloneBench": "gptclonebench_v3",
    "ATCoder": "atcoder_v3",
}
BENCHMARK_SLUGS = {
    "BigCloneBench": {"codexglue", "codexgluev3"},
    "SemanticCloneBench": {"semanticclonebench", "semanticclonebenchv3"},
    "GPTCloneBench": {"gptclonebench", "gptclonebenchv3"},
    "ATCoder": {"atcoder", "atcoderv3"},
}
METRICS = ("P", "R", "F1", "Acc")
LATENT_ORDER = (16, 24, 32, 48)
ABLATION_ORDER = ("topology_only", "typed_topology", "source_lexical")


def slug(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def number(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def fmt_metric(value: object) -> str:
    return "-" if pd.isna(value) else f"{float(value):.3f}"


def fmt_time(value: object) -> str:
    return "-" if pd.isna(value) else f"{float(value):.1f}"


def fmt_params(value: object) -> str:
    if pd.isna(value):
        return "-"
    value = float(value)
    if value == 0:
        return "0"
    return f"{value / 1_000_000:.2f}M" if value >= 1_000_000 else f"{value / 1_000:.1f}K"


def normalized_method(raw_method: object, archive_name: str) -> str:
    value = str(raw_method).strip().replace("CDHL", "CDLH")
    archive = Path(archive_name).stem.lower()
    if value.startswith("GNN-") or " + " in value:
        return value
    if archive.endswith("_snn"):
        return f"{value.upper()} + SNN"
    if archive.endswith("_gnn"):
        return f"GNN-{value.upper()}"
    return value


def method_family(method: str) -> str:
    if method == "SPECTRA-Siam":
        return "Our Method"
    if method.startswith("GNN-"):
        return "GNN Baselines"
    # This reproduction feeds both an AST adjacency matrix/node statistics and
    # a hashed raw-source-token feature vector into the encoder.  It is not a
    # graph-only baseline, even though one of its branches is graph-based.
    if method == "DeepSim":
        return "Hybrid Graph + Code Baselines"
    if method in {"CDLH", "RtvNN", "Deckard"}:
        return "Non-graph Code Baselines"
    if " + " in method:
        return "Spectral Representation Baselines"
    return "Other Graph-based Learning Methods"


def method_order(observed: Iterable[str]) -> list[str]:
    priority = {
        "GNN-AST": 10, "GNN-CFG": 11, "GNN-DDG": 12, "GNN-CPG": 13,
        "Deckard": 20, "RtvNN": 21, "CDLH": 22,
        "ASTNN": 30, "DeepSim": 31, "FA-AST+GGNN": 32, "FA-AST+GMN": 33,
        "SPECTRA-Siam": 40,
    }
    for graph_index, graph in enumerate(("AST", "CFG", "DDG", "CPG")):
        for method_index, method in enumerate(("No Train", "RF", "LR", "SNN")):
            priority[f"{graph} + {method}"] = 100 + graph_index * 10 + method_index
    return sorted(set(observed), key=lambda value: (priority.get(value, 999), value))


def archive_csv(bundle: zipfile.ZipFile, predicate) -> tuple[str, pd.DataFrame] | None:
    candidates = [name for name in bundle.namelist() if predicate(name.lower())]
    if len(candidates) != 1:
        return None
    name = candidates[0]
    return name, pd.read_csv(io.BytesIO(bundle.read(name)))


def expected_benchmark_ok(frame: pd.DataFrame, result_name: str, benchmark: str) -> bool:
    """Reject an archive if either a Dataset field or its result filename disagrees."""
    expected = BENCHMARK_SLUGS[benchmark]
    if "Dataset" in frame:
        declared = {slug(value) for value in frame["Dataset"].dropna().unique()}
        if declared and not declared.issubset(expected):
            return False
    result_slug = slug(Path(result_name).stem)
    explicit_other = set().union(*BENCHMARK_SLUGS.values()) - expected
    return not any(other and other in result_slug for other in explicit_other)


def main_result_records(archive: Path, benchmark: str, warnings: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(archive) as bundle:
        result = archive_csv(bundle, lambda name: name.endswith("_results.csv") and "combined" not in name)
        if result is None:
            warnings.append(f"Skipped {archive.name}: expected exactly one primary *_results.csv.")
            return pd.DataFrame(), pd.DataFrame()
        result_name, frame = result
        if not expected_benchmark_ok(frame, result_name, benchmark):
            warnings.append(f"Skipped {archive.name}: embedded result {result_name} belongs to a different dataset.")
            return pd.DataFrame(), pd.DataFrame()
        if "RunProfile" in frame and frame["RunProfile"].astype(str).str.lower().eq("final_full").any():
            frame = frame[frame["RunProfile"].astype(str).str.lower().eq("final_full")].copy()
        records = pd.DataFrame({
            "Benchmark": benchmark,
            "Method": [normalized_method(value, archive.name) for value in frame["Method"]],
            "Graph": frame["Method"].astype(str).str.upper().where(
                frame["Method"].astype(str).str.upper().isin(["AST", "CFG", "DDG", "CPG"]), ""
            ),
            "P": number(frame, "P"), "R": number(frame, "R"), "F1": number(frame, "F1"), "Acc": number(frame, "Acc"),
            "BestValidF1": number(frame, "BestValidF1"), "BestEpoch": number(frame, "BestEpoch"),
            "RuntimeSeconds": number(frame, "RuntimeSeconds"), "RuntimeMinutes": number(frame, "RuntimeMinutes"),
            "TrainableParameters": number(frame, "TrainableParameters"),
            "TrainPairs": number(frame, "TrainPairs"), "ValidPairs": number(frame, "ValidPairs"), "TestPairs": number(frame, "TestPairs"),
            "RunProfile": frame.get("RunProfile", pd.Series("unknown", index=frame.index)).astype(str),
            "Source": str(archive),
        })
        language = archive_csv(bundle, lambda name: name.endswith("_language_breakdown.csv"))
        if language is None:
            language_records = pd.DataFrame()
        else:
            _, breakdown = language
            language_records = pd.DataFrame({
                "Benchmark": benchmark,
                "Method": [normalized_method(value, archive.name) for value in breakdown["Method"]],
                "Language": breakdown["Language"].astype(str),
                "P": number(breakdown, "P"), "R": number(breakdown, "R"),
                "F1": number(breakdown, "F1"), "Acc": number(breakdown, "Acc"),
                "Pairs": number(breakdown, "Pairs"), "Source": str(archive),
            })
            language_records = language_records[~language_records["Language"].eq("ALL")].copy()
        return records, language_records


def language_label(left: str, right: str) -> str:
    return left if left == right else "->".join(sorted((left, right)))


def load_code_languages(benchmark: str) -> dict[str, str]:
    data_dir = OUTPUTS_ROOT / BENCHMARK_TO_CLEAN_DATA[benchmark] / "clean_data"
    candidates = (data_dir / "codes.jsonl.gz", data_dir / "codes.jsonl")
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Clean code export missing for {benchmark}: {data_dir}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return {str(item["code_id"]): str(item["language"]).lower() for item in map(json.loads, handle) if item}


def derive_spectra_language_records(archive: Path, benchmark: str, warnings: list[str]) -> pd.DataFrame:
    """Recover SPECTRA-Siam language metrics from its saved test predictions."""
    with zipfile.ZipFile(archive) as bundle:
        prediction_names = [name for name in bundle.namelist() if name.endswith("_predictions.csv.gz")]
        if len(prediction_names) != 1:
            return pd.DataFrame()
        try:
            languages = load_code_languages(benchmark)
        except FileNotFoundError as error:
            warnings.append(str(error))
            return pd.DataFrame()
        predictions = pd.read_csv(io.BytesIO(bundle.read(prediction_names[0])), compression="gzip")
    predictions["left_language"] = predictions["left_id"].astype(str).map(languages)
    predictions["right_language"] = predictions["right_id"].astype(str).map(languages)
    predictions = predictions.dropna(subset=["left_language", "right_language"]).copy()
    predictions["Language"] = [language_label(left, right) for left, right in zip(predictions.left_language, predictions.right_language)]
    rows = []
    for language, group in predictions.groupby("Language", sort=True):
        labels = group["label"].astype(int).to_numpy()
        predicted = group["prediction"].astype(int).to_numpy()
        tp = int(((labels == 1) & (predicted == 1)).sum())
        fp = int(((labels == 0) & (predicted == 1)).sum())
        tn = int(((labels == 0) & (predicted == 0)).sum())
        fn = int(((labels == 1) & (predicted == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "Benchmark": benchmark, "Method": "SPECTRA-Siam", "Language": language,
            "P": precision, "R": recall, "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "Acc": (tp + tn) / len(group) if len(group) else np.nan, "Pairs": len(group), "Source": str(archive),
        })
    return pd.DataFrame(rows)


def read_baseline_results() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    mains: list[pd.DataFrame] = []
    languages: list[pd.DataFrame] = []
    for archive in sorted(KAGGLE_ROOT.rglob("*.zip")):
        benchmark = FOLDER_TO_BENCHMARK.get(archive.parent.name)
        if benchmark is None:
            continue
        main, language = main_result_records(archive, benchmark, warnings)
        if main.empty:
            continue
        mains.append(main)
        if not language.empty:
            languages.append(language)
        if main["Method"].eq("SPECTRA-Siam").any():
            derived = derive_spectra_language_records(archive, benchmark, warnings)
            if not derived.empty:
                languages.append(derived)
    raw = pd.concat(mains, ignore_index=True) if mains else pd.DataFrame()
    language = pd.concat(languages, ignore_index=True) if languages else pd.DataFrame()
    raw = raw[~raw["Method"].str.contains("PDG", case=False, na=False)].copy()
    raw["Family"] = raw["Method"].map(method_family)
    raw = raw.sort_values(["Benchmark", "Method", "F1", "TestPairs"], ascending=[True, True, False, False])
    raw = raw.drop_duplicates(["Benchmark", "Method"], keep="first").reset_index(drop=True)
    if not language.empty:
        language = language[~language["Method"].str.contains("PDG", case=False, na=False)].copy()
        language = language.sort_values(["Benchmark", "Method", "Language", "F1", "Pairs"], ascending=[True, True, True, False, False])
        language = language.drop_duplicates(["Benchmark", "Method", "Language"], keep="first").reset_index(drop=True)
    return raw, language, warnings


def add_group_boundaries(table, groups: list[str]) -> None:
    for row in range(2, len(groups) + 1):
        if groups[row - 1] != groups[row - 2]:
            for column in range(max(column for _, column in table.get_celld())):
                table[(row, column)].set_linewidth(1.2)
                table[(row, column)].set_edgecolor("#1f2937")


def save_table(
    cell_text: list[list[str]],
    headers: list[str],
    title: str,
    output_name: str,
    widths: list[float] | None = None,
    groups: list[str] | None = None,
    best_cells: set[tuple[int, int]] | None = None,
    group_spans: list[tuple[str, int, int]] | None = None,
    note: str | None = None,
) -> Path:
    total_columns = len(headers)
    figure_width = max(13.5, min(42, 1.1 * total_columns + 4.5))
    figure_height = max(4.7, min(22, 0.34 * (len(cell_text) + 5)))
    fig, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.axis("off")
    axis.set_position([0.006, 0.035, 0.988, 0.9])
    if widths is None:
        widths = [1 / total_columns] * total_columns
    table = axis.table(
        cellText=cell_text, colLabels=headers, cellLoc="center", colLoc="center",
        colWidths=widths, bbox=[0, 0.025, 1, 0.87],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8 if total_columns > 18 else 7.7)
    table.scale(1, 1.18)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#64748b")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
    for row, column in best_cells or set():
        table[(row, column)].set_facecolor("#bbf7d0")
        table[(row, column)].set_text_props(weight="bold")
    if groups:
        add_group_boundaries(table, groups)
    if group_spans:
        cumulative = np.cumsum([0.0, *widths])
        for label, first, last in group_spans:
            center = (cumulative[first] + cumulative[last + 1]) / 2
            axis.text(center, 0.913, label, transform=axis.transAxes, ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontweight="bold", fontsize=14, y=0.988)
    if note:
        fig.text(0.5, 0.006, note, ha="center", va="bottom", fontsize=7.1, style="italic")
    target = ARTIFACTS / output_name
    fig.savefig(target, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return target


def make_main_tables(raw: pd.DataFrame) -> list[Path]:
    methods = method_order(raw.Method)
    metric_rows: list[list[str]] = []
    efficiency_rows: list[list[str]] = []
    groups: list[str] = []
    metric_best: set[tuple[int, int]] = set()
    for row_index, method in enumerate(methods, start=1):
        selected = raw[raw.Method.eq(method)]
        family = method_family(method)
        groups.append(family)
        metric_row = [family, method]
        efficiency_row = [family, method]
        for benchmark in BENCHMARK_ORDER:
            item = selected[selected.Benchmark.eq(benchmark)]
            item = item.iloc[0] if not item.empty else None
            metric_row.extend(fmt_metric(item[metric]) if item is not None else "-" for metric in METRICS)
            efficiency_row.extend([
                fmt_time(item.RuntimeMinutes) if item is not None else "-",
                fmt_params(item.TrainableParameters) if item is not None else "-",
            ])
        metric_rows.append(metric_row)
        efficiency_rows.append(efficiency_row)
    for benchmark_index, benchmark in enumerate(BENCHMARK_ORDER):
        best = raw.loc[raw.Benchmark.eq(benchmark), "F1"].max()
        for row_index, method in enumerate(methods, start=1):
            value = raw[(raw.Benchmark.eq(benchmark)) & (raw.Method.eq(method))]["F1"]
            if not value.empty and np.isclose(value.iloc[0], best):
                metric_best.add((row_index, 2 + benchmark_index * 4 + 2))
    metric_headers = ["Method family", "Method"] + list(METRICS) * len(BENCHMARK_ORDER)
    efficiency_headers = ["Method family", "Method"] + ["Time\n(min)", "Params"] * len(BENCHMARK_ORDER)
    metric_widths = [0.135, 0.13] + [(1 - 0.265) / (4 * len(BENCHMARK_ORDER))] * (4 * len(BENCHMARK_ORDER))
    efficiency_widths = [0.16, 0.18] + [(1 - 0.34) / (2 * len(BENCHMARK_ORDER))] * (2 * len(BENCHMARK_ORDER))
    metric_spans = [(benchmark, 2 + index * 4, 2 + index * 4 + 3) for index, benchmark in enumerate(BENCHMARK_ORDER)]
    efficiency_spans = [(benchmark, 2 + index * 2, 2 + index * 2 + 1) for index, benchmark in enumerate(BENCHMARK_ORDER)]
    return [
        save_table(metric_rows, metric_headers, "Clone-detection benchmark: test-set comparison", "paper_wide_test_metrics.png", metric_widths, groups, metric_best, metric_spans),
        save_table(efficiency_rows, efficiency_headers, "Clone-detection benchmark: run-level efficiency comparison", "paper_wide_efficiency_metrics.png", efficiency_widths, groups, None, efficiency_spans),
    ]


def stratum_label(benchmark: str, language: str) -> str:
    names = {"csharp": "C#", "python": "Python", "java": "Java", "c": "C"}
    display_language = " -> ".join(names.get(item, item) for item in language.split("->"))
    return f"{benchmark}\n{display_language}"


def make_language_tables(raw: pd.DataFrame, language: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    if language.empty:
        return outputs
    language["Setting"] = np.where(language.Language.str.contains("->", regex=False), "Cross-language", "Single-language")
    for setting, tag in (("Single-language", "single_language"), ("Cross-language", "cross_language")):
        subset = language[language.Setting.eq(setting)].copy()
        if subset.empty:
            continue
        strata = sorted(
            subset[["Benchmark", "Language"]].drop_duplicates().itertuples(index=False, name=None),
            key=lambda item: (BENCHMARK_ORDER.index(item[0]), item[1]),
        )
        methods = method_order(subset.Method)
        rows: list[list[str]] = []
        groups: list[str] = []
        best_cells: set[tuple[int, int]] = set()
        for row_index, method in enumerate(methods, start=1):
            groups.append(method_family(method))
            row = [method_family(method), method]
            for benchmark, language_name in strata:
                item = subset[(subset.Benchmark.eq(benchmark)) & (subset.Language.eq(language_name)) & (subset.Method.eq(method))]
                item = item.iloc[0] if not item.empty else None
                row.extend(fmt_metric(item[metric]) if item is not None else "-" for metric in METRICS)
            rows.append(row)
        for stratum_index, (benchmark, language_name) in enumerate(strata):
            best = subset[(subset.Benchmark.eq(benchmark)) & (subset.Language.eq(language_name))]["F1"].max()
            for row_index, method in enumerate(methods, start=1):
                value = subset[(subset.Benchmark.eq(benchmark)) & (subset.Language.eq(language_name)) & (subset.Method.eq(method))]["F1"]
                if not value.empty and np.isclose(value.iloc[0], best):
                    best_cells.add((row_index, 2 + stratum_index * 4 + 2))
        headers = ["Method family", "Method"] + list(METRICS) * len(strata)
        widths = [0.15, 0.15] + [(1 - 0.3) / (4 * len(strata))] * (4 * len(strata))
        spans = [(stratum_label(benchmark, language_name), 2 + index * 4, 2 + index * 4 + 3) for index, (benchmark, language_name) in enumerate(strata)]
        outputs.append(save_table(
            rows, headers, f"{setting} evaluation: test metrics by language / language pair",
            f"paper_{tag}_test_metrics.png", widths, groups, best_cells, spans,
        ))

        efficiency_rows: list[list[str]] = []
        for method in methods:
            row = [method_family(method), method]
            for benchmark, language_name in strata:
                available = not subset[(subset.Benchmark.eq(benchmark)) & (subset.Language.eq(language_name)) & (subset.Method.eq(method))].empty
                item = raw[(raw.Benchmark.eq(benchmark)) & (raw.Method.eq(method))]
                item = item.iloc[0] if available and not item.empty else None
                row.extend([fmt_time(item.RuntimeMinutes) if item is not None else "-", fmt_params(item.TrainableParameters) if item is not None else "-"])
            efficiency_rows.append(row)
        headers = ["Method family", "Method"] + ["Time\n(min)", "Params"] * len(strata)
        widths = [0.17, 0.18] + [(1 - 0.35) / (2 * len(strata))] * (2 * len(strata))
        spans = [(stratum_label(benchmark, language_name), 2 + index * 2, 2 + index * 2 + 1) for index, (benchmark, language_name) in enumerate(strata)]
        outputs.append(save_table(
            efficiency_rows, headers, f"{setting} evaluation: associated run efficiency",
            f"paper_{tag}_run_efficiency.png", widths, groups, None, spans,
            "Runtime and parameter count belong to the full benchmark run; they are repeated only to associate a language stratum with its model run.",
        ))

    # The all-benchmark view above is useful for an appendix, while these
    # dataset-specific views remain readable at normal paper/table scale and
    # make every language condition independently citable.
    for benchmark in BENCHMARK_ORDER:
        subset = language[language.Benchmark.eq(benchmark)].copy()
        if subset.empty:
            continue
        setting = "Cross-language" if subset.Language.str.contains("->", regex=False).any() else "Single-language"
        strata = sorted(subset.Language.drop_duplicates())
        methods = method_order(subset.Method)
        rows: list[list[str]] = []
        groups: list[str] = []
        best_cells: set[tuple[int, int]] = set()
        for row_index, method in enumerate(methods, start=1):
            groups.append(method_family(method))
            row = [method_family(method), method]
            for language_name in strata:
                item = subset[(subset.Language.eq(language_name)) & (subset.Method.eq(method))]
                item = item.iloc[0] if not item.empty else None
                row.extend(fmt_metric(item[metric]) if item is not None else "-" for metric in METRICS)
            rows.append(row)
        for language_index, language_name in enumerate(strata):
            best = subset[subset.Language.eq(language_name)]["F1"].max()
            for row_index, method in enumerate(methods, start=1):
                value = subset[(subset.Language.eq(language_name)) & (subset.Method.eq(method))]["F1"]
                if not value.empty and np.isclose(value.iloc[0], best):
                    best_cells.add((row_index, 2 + language_index * 4 + 2))
        headers = ["Method family", "Method"] + list(METRICS) * len(strata)
        widths = [0.19, 0.19] + [(1 - 0.38) / (4 * len(strata))] * (4 * len(strata))
        spans = [(stratum_label(benchmark, language_name).split("\n", 1)[1], 2 + index * 4, 2 + index * 4 + 3) for index, language_name in enumerate(strata)]
        file_stem = slug(benchmark)
        outputs.append(save_table(
            rows, headers, f"{benchmark}: {setting.lower()} test metrics by language / language pair",
            f"paper_{file_stem}_language_test_metrics.png", widths, groups, best_cells, spans,
        ))
        efficiency_rows: list[list[str]] = []
        for method in methods:
            row = [method_family(method), method]
            for language_name in strata:
                available = not subset[(subset.Language.eq(language_name)) & (subset.Method.eq(method))].empty
                item = raw[(raw.Benchmark.eq(benchmark)) & (raw.Method.eq(method))]
                item = item.iloc[0] if available and not item.empty else None
                row.extend([fmt_time(item.RuntimeMinutes) if item is not None else "-", fmt_params(item.TrainableParameters) if item is not None else "-"])
            efficiency_rows.append(row)
        headers = ["Method family", "Method"] + ["Time\n(min)", "Params"] * len(strata)
        widths = [0.22, 0.22] + [(1 - 0.44) / (2 * len(strata))] * (2 * len(strata))
        spans = [(stratum_label(benchmark, language_name).split("\n", 1)[1], 2 + index * 2, 2 + index * 2 + 1) for index, language_name in enumerate(strata)]
        outputs.append(save_table(
            efficiency_rows, headers, f"{benchmark}: associated run efficiency by language condition",
            f"paper_{file_stem}_language_run_efficiency.png", widths, groups, None, spans,
            "Runtime and parameter count are measured once for the full benchmark run, not separately for each language condition.",
        ))
    return outputs


def read_experiment_results(folder: str, suffix: str, warnings: list[str]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for archive in sorted((KAGGLE_ROOT / folder).glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            found = archive_csv(bundle, lambda name: name.endswith(suffix))
            if found is None:
                warnings.append(f"Skipped experiment archive {archive.name}: expected exactly one {suffix}.")
                continue
            _, frame = found
        dataset_slug = slug(frame["Dataset"].iloc[0])
        benchmark = next((key for key, aliases in BENCHMARK_SLUGS.items() if dataset_slug in aliases), None)
        if benchmark is None:
            warnings.append(f"Skipped experiment archive {archive.name}: unknown Dataset={frame['Dataset'].iloc[0]!r}.")
            continue
        frame = frame.copy()
        frame["Benchmark"] = benchmark
        frame["Source"] = str(archive)
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def make_experiment_tables(frame: pd.DataFrame, kind: str) -> list[Path]:
    if frame.empty:
        return []
    if kind == "latent_capacity":
        row_key, values, title = "LatentNodes", list(LATENT_ORDER), "Latent graph capacity"
        labels = {value: f"{value} latent nodes" for value in values}
    else:
        row_key, values, title = "Variant", list(ABLATION_ORDER), "SPECTRA-Siam feature ablation"
        labels = {
            "topology_only": "Topology only",
            "typed_topology": "Typed topology + lexical sketch",
            "source_lexical": "Typed topology + source lexical",
        }
    metric_rows: list[list[str]] = []
    metric_best: set[tuple[int, int]] = set()
    selection_rows: list[list[str]] = []
    for row_index, value in enumerate(values, start=1):
        metric_row = [labels[value]]
        selection_row = [labels[value]]
        for benchmark in BENCHMARK_ORDER:
            item = frame[(frame.Benchmark.eq(benchmark)) & (frame[row_key].eq(value))]
            item = item.iloc[0] if not item.empty else None
            metric_row.extend(fmt_metric(item[metric]) if item is not None else "-" for metric in METRICS)
            selection_row.extend([
                fmt_metric(item.MacroF1) if item is not None else "-",
                fmt_metric(item.BalancedAccuracy) if item is not None else "-",
                fmt_metric(item.BestValidF1) if item is not None else "-",
                fmt_time(number(pd.DataFrame([item]), "RuntimeSeconds").iloc[0] / 60) if item is not None else "-",
            ])
        metric_rows.append(metric_row)
        selection_rows.append(selection_row)
    for benchmark_index, benchmark in enumerate(BENCHMARK_ORDER):
        best = number(frame[frame.Benchmark.eq(benchmark)], "F1").max()
        for row_index, value in enumerate(values, start=1):
            candidate = frame[(frame.Benchmark.eq(benchmark)) & (frame[row_key].eq(value))]
            if not candidate.empty and np.isclose(number(candidate, "F1").iloc[0], best):
                metric_best.add((row_index, 1 + benchmark_index * 4 + 2))
    metric_headers = ["Configuration"] + list(METRICS) * len(BENCHMARK_ORDER)
    selection_headers = ["Configuration"] + ["Macro F1", "Balanced\nAcc", "Valid F1", "Time\n(min)"] * len(BENCHMARK_ORDER)
    metric_widths = [0.2] + [0.8 / (4 * len(BENCHMARK_ORDER))] * (4 * len(BENCHMARK_ORDER))
    selection_widths = [0.2] + [0.8 / (4 * len(BENCHMARK_ORDER))] * (4 * len(BENCHMARK_ORDER))
    spans = [(benchmark, 1 + index * 4, 1 + index * 4 + 3) for index, benchmark in enumerate(BENCHMARK_ORDER)]
    prefix = "latent_capacity" if kind == "latent_capacity" else "feature_ablation"
    return [
        save_table(metric_rows, metric_headers, f"{title}: test-set comparison", f"{prefix}_test_metrics.png", metric_widths, None, metric_best, spans),
        save_table(selection_rows, selection_headers, f"{title}: validation and efficiency evidence", f"{prefix}_selection_efficiency.png", selection_widths, None, None, spans),
    ]


def write_csv_artifacts(raw: pd.DataFrame, language: pd.DataFrame, latent: pd.DataFrame, ablation: pd.DataFrame) -> None:
    raw.to_csv(ARTIFACTS / "all_results_long.csv", index=False)
    language.to_csv(ARTIFACTS / "language_breakdown_long.csv", index=False)
    latent.to_csv(ARTIFACTS / "latent_capacity_long.csv", index=False)
    ablation.to_csv(ARTIFACTS / "feature_ablation_long.csv", index=False)


def main() -> dict[str, object]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw, language, warnings = read_baseline_results()
    if raw.empty:
        raise FileNotFoundError(f"No usable baseline result ZIPs found under {KAGGLE_ROOT}")
    latent = read_experiment_results("01_latent_capacity", "_capacity_results.csv", warnings)
    ablation = read_experiment_results("04_feature_ablation", "_ablation_results.csv", warnings)
    write_csv_artifacts(raw, language, latent, ablation)
    images = [*make_main_tables(raw), *make_language_tables(raw, language)]
    images.extend(make_experiment_tables(latent, "latent_capacity"))
    images.extend(make_experiment_tables(ablation, "feature_ablation"))
    warning_path = ARTIFACTS / "report_input_warnings.txt"
    warning_path.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    manifest = [
        "# Final benchmark artifact manifest",
        "",
        f"- Baseline rows: {len(raw):,}",
        f"- Language-breakdown rows: {len(language):,}",
        f"- Latent-capacity rows: {len(latent):,}",
        f"- Feature-ablation rows: {len(ablation):,}",
        "- Generated figures:",
        *[f"  - `{path.name}`" for path in images],
    ]
    if warnings:
        manifest.extend(["", "- Input warnings are documented in `report_input_warnings.txt`."])
    (ARTIFACTS / "paper_report_summary.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Generated {len(images)} figures in {ARTIFACTS}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return {"artifacts": ARTIFACTS, "images": images, "warnings": warnings}


if __name__ == "__main__":
    main()
