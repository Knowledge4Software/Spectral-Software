"""Render a score-distribution histogram for every RQ1 table row.

Produces one PNG per (graph view, downstream learner) so the paper can be
built from whichever pair of panels reads most clearly, plus a contact sheet
of all of them and a summary CSV.

The corpus and every spectrum are loaded once and reused for all rows; the
per-row work is only the downstream learner. Raw test scores are cached to
``scores.npz``, so re-plotting with different styling never retrains anything.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.rq1.evaluate_pss import common_support, load_conventional_spectra, load_latent_export
from research.rq1.run_table import (
    CONVENTIONAL_GRAPH_TYPES,
    LogisticRegression,
    RandomForestClassifier,
    StandardScaler,
    choose_threshold,
    default_run_args,
    matrices_and_splits,
    metrics,
    pair_features,
    pss_scores,
    run_snn,
    select_metric,
)

GRAPHS = ("latent", *CONVENTIONAL_GRAPH_TYPES)
LEARNERS = ("No Train", "RF", "LR", "SNN")
RQ2_PREDICTIONS = {
    "atcoder": _ROOT.parent / "outputs" / "kaggle" / "RQ2" / "ATCoder" / "ATCoder_spectra_siam_Lexical.zip",
    "xglue": _ROOT.parent / "outputs" / "kaggle" / "RQ2" / "CodeXGlue" / "CodeXGlue_spectra_siam_lexical.zip",
}


def display_name(graph: str, learner: str) -> str:
    prefix = "SPECTRA-Siam Spectrum" if graph == "latent" else graph.upper()
    return f"{prefix} + {learner}"


def slug(graph: str, learner: str) -> str:
    return f"{graph}_{learner.replace(' ', '')}"


class Corpus:
    """Everything the rows share: pairs, spectra, and per-graph split arrays."""

    def __init__(self, clean_data: Path, latent_export: Path) -> None:
        started = time.perf_counter()
        _manifest, pairs, latent = load_latent_export(latent_export)
        required = set(pairs.left_id.astype(str)) | set(pairs.right_id.astype(str))
        conventional = load_conventional_spectra(clean_data, required)
        self.pairs, _coverage = common_support(pairs, conventional, latent)
        common = set(self.pairs.left_id.astype(str)) | set(self.pairs.right_id.astype(str))
        self.spectra = {
            "latent": {code: value for code, value in latent.items() if code in common}
        }
        self.spectra.update({
            graph: {code: views[graph] for code, views in conventional.items() if code in common}
            for graph in CONVENTIONAL_GRAPH_TYPES
        })
        self._cache: dict[str, tuple] = {}
        print(f"corpus ready in {time.perf_counter() - started:.0f}s: "
              f"{len(self.pairs):,} pairs, {len(common):,} codes")

    def arrays(self, graph: str):
        """Feature matrix and split indices for one graph view, built once."""
        if graph not in self._cache:
            self._cache[graph] = matrices_and_splits(self.pairs, self.spectra[graph])
        return self._cache[graph]


def score_row(corpus: Corpus, graph: str, learner: str, args) -> tuple[np.ndarray, np.ndarray, dict]:
    """Test-split labels, scores, and metrics for one table row.

    Mirrors ``run_dataset``'s per-row logic so the numbers line up with the
    published table rather than being a second, slightly different pipeline.
    """
    matrix, splits = corpus.arrays(graph)
    primary = select_metric(splits["valid"].labels)
    if learner == "No Train":
        scores = pss_scores(corpus.pairs, corpus.spectra[graph])
        valid_scores, test_scores = scores["valid"], scores["test"]
    elif learner in {"RF", "LR"}:
        train_x = pair_features(matrix, splits["train"].left, splits["train"].right)
        valid_x = pair_features(matrix, splits["valid"].left, splits["valid"].right)
        test_x = pair_features(matrix, splits["test"].left, splits["test"].right)
        index = GRAPHS.index(graph)
        if learner == "RF":
            classifier = RandomForestClassifier(
                n_estimators=args.rf_trees, max_depth=args.rf_max_depth, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=args.rf_jobs, random_state=args.seed + index,
            )
        else:
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            valid_x, test_x = scaler.transform(valid_x), scaler.transform(test_x)
            classifier = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=args.seed + index
            )
        classifier.fit(train_x, splits["train"].labels)
        valid_scores = classifier.predict_proba(valid_x)[:, 1]
        test_scores = classifier.predict_proba(test_x)[:, 1]
        del train_x, valid_x, test_x, classifier
    else:
        valid_scores, test_scores, _threshold, _details = run_snn(
            matrix, splits, args, args.seed + GRAPHS.index(graph)
        )
    threshold = choose_threshold(splits["valid"].labels, valid_scores, primary)
    row_metrics = metrics(splits["test"].labels, test_scores, threshold)
    return splits["test"].labels, np.asarray(test_scores, dtype=np.float64), row_metrics


def direct_spectra_siam_scores(dataset: str) -> tuple[np.ndarray, np.ndarray] | None:
    """SPECTRA-Siam's own trained classifier, from its RQ2 predictions."""
    archive_path = RQ2_PREDICTIONS.get(dataset)
    if archive_path is None or not Path(archive_path).is_file():
        return None
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.endswith("predictions.csv.gz")]
        if not names:
            return None
        frame = pd.read_csv(io.BytesIO(archive.read(names[0])), compression="gzip")
    return frame.label.to_numpy(dtype=np.int8), frame.clone_probability.to_numpy(dtype=np.float64)


def plot_one(labels: np.ndarray, scores: np.ndarray, title: str, subtitle: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for value, color, name in ((0, "#4C72B0", "Non-clone pairs"), (1, "#C44E52", "Clone pairs")):
        subset = np.clip(scores[labels == value], 0.0, 1.0)
        if not len(subset):
            continue
        axis.hist(subset, bins=40, range=(0, 1), density=True, alpha=0.55,
                  color=color, label=name, edgecolor="white", linewidth=0.3)
        if subset.std() > 1e-6:
            kde = stats.gaussian_kde(subset)
            grid = np.linspace(0, 1, 400)
            axis.plot(grid, kde(grid), color=color, linestyle="--", linewidth=1.4)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Output score")
    axis.set_ylabel("Density")
    axis.set_title(title, fontsize=11)
    axis.text(0.5, -0.28, subtitle, transform=axis.transAxes, ha="center", va="top", fontsize=9)
    axis.grid(axis="y", alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=8.5, frameon=True, framealpha=0.9)
    figure.tight_layout()
    # PNG for quick review, PDF because the paper embeds vector figures.
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_contact_sheet(entries: list[dict], path: Path) -> None:
    """All rows on one page, so the clearest separation is easy to spot."""
    columns = 4
    rows = -(-len(entries) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, entry in zip(axes, entries):
        labels, scores = entry["labels"], entry["scores"]
        for value, color in ((0, "#4C72B0"), (1, "#C44E52")):
            subset = np.clip(scores[labels == value], 0.0, 1.0)
            if len(subset):
                axis.hist(subset, bins=32, range=(0, 1), density=True, alpha=0.55, color=color)
        axis.set_title(f"{entry['name']}\nF1={entry['F1']:.3f}  Acc={entry['Acc']:.3f}", fontsize=8)
        axis.set_xlim(0, 1)
        axis.tick_params(labelsize=7)
    for axis in axes[len(entries):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("atcoder", "xglue"), required=True)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--latent-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graphs", nargs="*", default=list(GRAPHS))
    parser.add_argument("--learners", nargs="*", default=list(LEARNERS))
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--rf-jobs", type=int, default=4)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parsed = parser.parse_args()

    run_args = default_run_args(device=parsed.device)
    run_args.rf_jobs, run_args.rf_trees, run_args.rf_max_depth = (
        parsed.rf_jobs, parsed.rf_trees, parsed.rf_max_depth
    )
    run_args.seed = parsed.seed

    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(parsed.clean_data, parsed.latent_export)

    entries: list[dict] = []
    cache: dict[str, np.ndarray] = {}
    for graph in parsed.graphs:
        for learner in parsed.learners:
            name = display_name(graph, learner)
            started = time.perf_counter()
            print(f"[{name}] scoring...", flush=True)
            labels, scores, row_metrics = score_row(corpus, graph, learner, run_args)
            elapsed = time.perf_counter() - started
            entry = {
                "graph": graph, "learner": learner, "name": name,
                "labels": labels, "scores": scores,
                "F1": row_metrics["F1"], "Acc": row_metrics["Acc"],
                "P": row_metrics["P"], "R": row_metrics["R"],
                "minutes": elapsed / 60.0,
            }
            entries.append(entry)
            cache[f"{slug(graph, learner)}_labels"] = labels
            cache[f"{slug(graph, learner)}_scores"] = scores
            plot_one(
                labels, scores, name,
                f"F1={row_metrics['F1']:.3f}   Acc={row_metrics['Acc']:.3f}   "
                f"P={row_metrics['P']:.3f}   R={row_metrics['R']:.3f}",
                parsed.output_dir / f"{slug(graph, learner)}.png",
            )
            print(f"    F1={row_metrics['F1']:.4f} Acc={row_metrics['Acc']:.4f} "
                  f"({elapsed/60:.1f} min)", flush=True)

    if not parsed.skip_direct:
        direct = direct_spectra_siam_scores(parsed.dataset)
        if direct is None:
            print("SPECTRA-Siam direct predictions not found; skipping that panel")
        else:
            labels, scores = direct
            predicted = (scores >= 0.5).astype(np.int8)
            accuracy = float((predicted == labels).mean())
            entries.append({
                "graph": "direct", "learner": "model", "name": "SPECTRA-Siam (full model)",
                "labels": labels, "scores": scores, "F1": np.nan, "Acc": accuracy,
                "P": np.nan, "R": np.nan, "minutes": 0.0,
            })
            cache["direct_labels"], cache["direct_scores"] = labels, scores
            plot_one(labels, scores, "SPECTRA-Siam (full model)",
                     f"trained pair classifier   Acc@0.5={accuracy:.3f}",
                     parsed.output_dir / "direct_full_model.png")

    np.savez_compressed(parsed.output_dir / "scores.npz", **cache)
    plot_contact_sheet(entries, parsed.output_dir / "_contact_sheet.png")

    # One archive of every vector figure, ready to drop into the paper.
    pdfs = sorted(parsed.output_dir.glob("*.pdf"))
    archive_path = parsed.output_dir / f"{parsed.dataset}_figures_pdf.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for pdf in pdfs:
            archive.write(pdf, pdf.name)
    print(f"wrote {archive_path} ({len(pdfs)} PDFs)")
    summary = pd.DataFrame(
        [{key: entry[key] for key in ("name", "graph", "learner", "P", "R", "F1", "Acc", "minutes")}
         for entry in entries]
    ).sort_values("F1", ascending=False)
    summary.to_csv(parsed.output_dir / "summary.csv", index=False)
    print(f"\nwrote {len(entries)} histograms to {parsed.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
