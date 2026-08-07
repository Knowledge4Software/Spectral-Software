"""Generate ATCoder analysis notebooks mirroring the XGLUE analysis stages."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _source(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source(text)}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


SETUP = """from pathlib import Path
import sys
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / 'spectral_code').exists():
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError('Open this notebook from the repository.')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import importlib
import spectral_code.evaluation.atcoder_analysis as atcoder_analysis
importlib.reload(atcoder_analysis)
from spectral_code.evaluation.atcoder_analysis import *

sns.set_theme(style='whitegrid', context='notebook')
ROOT = clean_data_dir()
ROOT"""


NOTEBOOKS = {
    "00_temp_small_graph_inspection.ipynb": [
        markdown("# ATCoder — Small Graph Inspection\n\nFinal-export sanity checks, equivalent to the XGLUE graph-inspection stage."),
        code(SETUP),
        markdown("## Artifact readiness"),
        code("coverage, summary = graph_coverage(ROOT)\nsummary\ncoverage"),
        markdown("## Random code records"),
        code("codes = load_codes(ROOT, include_code=True)\ncodes[['code_id', 'language', 'source_function_id', 'line_count', 'char_count', 'code']].sample(3, random_state=42)"),
        markdown("## Graph-layer status"),
        code("sns.catplot(data=coverage, x='graph_type', y='codes', hue='spectral_status', col='language', kind='bar'); plt.show()"),
    ],
    "01_dataset_analysis.ipynb": [
        markdown("# ATCoder — Dataset Analysis\n\nDataset overview, split balance, language composition, and source-size summaries."),
        code(SETUP),
        code("codes = load_codes(ROOT)\npairs = load_pairs(ROOT)\nprint('codes:', len(codes), 'pairs:', len(pairs))\ndisplay(codes.groupby('language').size().rename('codes').to_frame())\ndisplay(pairs.groupby(['split', 'label_name']).size().rename('pairs').to_frame())"),
        markdown("## Source-size distributions"),
        code("display(codes.groupby('language')[['line_count', 'char_count']].describe())\nsns.histplot(data=codes, x='line_count', hue='language', bins=60, log_scale=(True, False)); plt.show()"),
        markdown("## Split and label balance"),
        code("sns.countplot(data=pairs, x='split', hue='label_name', order=['train', 'valid', 'test']); plt.show()"),
    ],
    "02_graph_and_spectral_analysis.ipynb": [
        markdown("# ATCoder — Graph and Spectral Analysis\n\nGraph/spectral coverage plus sampled PSS and Wasserstein score distributions."),
        code(SETUP),
        code("coverage, summary = graph_coverage(ROOT)\nsummary\ncoverage"),
        code("sns.catplot(data=coverage, x='graph_type', y='codes', hue='spectral_status', col='language', kind='bar'); plt.show()"),
        markdown("## Clone and non-clone examples: code, graph, and spectrum"),
        code("examples = select_pair_examples(ROOT, graph_type='cpg', per_label=2, seed=42)\nfor example in examples:\n    display(pd.DataFrame([example['pair']]))\n    display(plot_pair_code_graph_spectra(ROOT, example, graph_type='cpg', max_code_lines=32, max_graph_nodes=80))"),
        markdown("## Balanced train-pair sample"),
        code("scores = sample_similarity_scores(ROOT, sample_size=10_000, seed=42)\nscores.groupby(['graph_type', 'label_name']).size().rename('scores').to_frame()"),
        code("sns.displot(data=scores, x='pss', hue='label_name', col='graph_type', kind='hist', stat='density', common_norm=False, bins=40, col_wrap=2); plt.show()"),
        code("sns.displot(data=scores, x='wasserstein', hue='label_name', col='graph_type', kind='hist', stat='density', common_norm=False, bins=40, col_wrap=2); plt.show()"),
    ],
    "03_tuning_analysis.ipynb": [
        markdown("# ATCoder — Tuning Analysis\n\nReproducible balanced-sample threshold sweep for the PSS spectral baseline."),
        code(SETUP),
        code("scores = sample_similarity_scores(ROOT, sample_size=10_000, seed=42)\nthresholds, distributions = threshold_summary(scores)\ndisplay(thresholds)\ndisplay(distributions)"),
        code("sns.barplot(data=thresholds, x='graph_type', y='f1'); plt.ylim(0, 1); plt.show()"),
        markdown("## Save reports"),
        code("paths = write_analysis_reports(ROOT, sample_size=10_000, seed=42)\npaths"),
    ],
    "04_baseline_results_table.ipynb": [
        markdown("# ATCoder — Baseline Results Table\n\nDisplays saved spectral-baseline results. Future CPU/GPU baseline CSVs can be added under `outputs/atcoder/baselines/`, following the XGLUE convention."),
        code(SETUP),
        code("reports = ROOT.parent / 'reports'\nthresholds = pd.read_csv(reports / 'atcoder_threshold_sample.csv') if (reports / 'atcoder_threshold_sample.csv').exists() else pd.DataFrame()\nthresholds"),
    ],
}


def main() -> None:
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    for filename, cells in NOTEBOOKS.items():
        notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
        path = HERE / filename
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
