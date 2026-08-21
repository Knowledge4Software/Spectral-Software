"""Embed language-stratified test reporting in every SPECTRA-Siam notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_HELPER = '''def save_spectra_language_breakdown(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[Path, pd.DataFrame]:
    """Save test metrics by unordered endpoint-language pair plus ALL."""
    required = {"left_language", "right_language", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Language breakdown requires pair columns: {missing}")
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if len(probabilities) != len(frame):
        raise RuntimeError(
            f"Language breakdown received {len(probabilities)} probabilities for {len(frame)} test pairs"
        )

    left_languages = frame["left_language"].astype(str).str.strip().str.lower().tolist()
    right_languages = frame["right_language"].astype(str).str.strip().str.lower().tolist()
    language_keys = np.asarray([
        left if left == right else f"{min(left, right)}->{max(left, right)}"
        for left, right in zip(left_languages, right_languages)
    ], dtype=object)

    rows = []
    groups = [(key, language_keys == key) for key in sorted(set(language_keys.tolist()))]
    groups.append(("ALL", np.ones(len(frame), dtype=bool)))
    for language, mask in groups:
        group_labels = labels[mask]
        group_probabilities = probabilities[mask]
        metrics = metric_summary(group_labels, group_probabilities, threshold)
        rows.append({
            "Dataset": DATASET_KEY,
            "Method": "SPECTRA-Siam",
            "MethodVariant": METHOD_VARIANT,
            "GraphType": "learned_latent",
            "Language": language,
            "LanguageScope": (
                "all" if language == "ALL" else "cross_language" if "->" in language else "same_language"
            ),
            "P": metrics["Precision"],
            "R": metrics["Recall"],
            "F1": metrics["F1"],
            "Acc": metrics["Accuracy"],
            "MacroF1": metrics["MacroF1"],
            "BalancedAccuracy": metrics["BalancedAccuracy"],
            "Threshold": metrics["Threshold"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
            "Pairs": metrics["Pairs"],
            "Positives": int((group_labels == 1).sum()),
            "Negatives": int((group_labels == 0).sum()),
        })

    breakdown = pd.DataFrame(rows)
    output_path = WORK_DIR / f"{DATASET_KEY}_{METHOD_VARIANT}_spectra_siam_language_breakdown.csv"
    breakdown.to_csv(output_path, index=False)
    print("Language breakdown:")
    display(breakdown[["Language", "LanguageScope", "P", "R", "F1", "Acc", "Pairs", "Positives"]])
    print("Saved language breakdown:", output_path)
    return output_path, breakdown
'''


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def patch_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    full_source = "\n".join(source(cell) for cell in notebook["cells"])
    if "class CanonicalSpectraSiam" not in full_source or "def run_experiment()" not in full_source:
        return False

    changed = False
    run_cell = next(cell for cell in notebook["cells"] if "def run_experiment()" in source(cell))
    run_source = source(run_cell)
    if "def save_spectra_language_breakdown" not in run_source:
        marker = "\ndef choose_threshold("
        if marker not in run_source:
            raise RuntimeError(f"Could not locate threshold helper in {path}")
        run_source = run_source.replace(marker, "\n\n" + LANGUAGE_HELPER + marker, 1)
        changed = True

    if "language_breakdown_path, language_breakdown_frame = save_spectra_language_breakdown" not in run_source:
        prediction_lines = (
            'predictions.to_csv(WORK_DIR / f"spectra_siam_{RUN_TAG}_predictions.csv.gz", index=False, compression="gzip")',
            'predictions.to_csv(WORK_DIR / f"spectra_siam_{DATASET_KEY}_predictions.csv.gz", index=False, compression="gzip")',
            'predictions.to_csv(WORK_DIR / f"{OUTPUT_STEM}_predictions.csv.gz", index=False, compression="gzip")',
        )
        prediction_line = next((item for item in prediction_lines if item in run_source), None)
        if prediction_line is None:
            raise RuntimeError(f"Could not locate prediction output in {path}")
        run_source = run_source.replace(
            prediction_line,
            prediction_line
            + "\n    language_breakdown_path, language_breakdown_frame = save_spectra_language_breakdown(\n"
            + "        frames[\"test\"], test_probabilities, checkpoint[\"best\"][\"threshold\"]\n"
            + "    )",
            1,
        )
        changed = True

    if 'result["language_breakdown_csv"]' not in run_source:
        marker = '    result["comparison_csv"] = str(comparison_path)\n'
        if marker not in run_source:
            raise RuntimeError(f"Could not locate comparison result assignment in {path}")
        run_source = run_source.replace(
            marker,
            marker
            + '    result["language_breakdown_csv"] = str(language_breakdown_path)\n'
            + '    result["language_breakdown_rows"] = int(len(language_breakdown_frame))\n',
            1,
        )
        changed = True

    if '"language_breakdown": str(language_breakdown_path)' not in run_source:
        markers = (
            '        "predictions": str(WORK_DIR / f"spectra_siam_{RUN_TAG}_predictions.csv.gz"),\n',
            '        "predictions": str(WORK_DIR / f"spectra_siam_{DATASET_KEY}_predictions.csv.gz"),\n',
            '        "predictions": str(WORK_DIR / f"{OUTPUT_STEM}_predictions.csv.gz"),\n',
        )
        marker = next((item for item in markers if item in run_source), None)
        if marker is None:
            raise RuntimeError(f"Could not locate output manifest in {path}")
        run_source = run_source.replace(
            marker,
            marker + '        "language_breakdown": str(language_breakdown_path),\n',
            1,
        )
        changed = True

    if changed:
        set_source(run_cell, run_source)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    candidates = sorted((ROOT / "kaggle").rglob("*.ipynb"))
    changed = [path for path in candidates if patch_notebook(path)]
    print(f"Synchronized SPECTRA language breakdown in {len(changed)} notebook(s).")
    for path in changed:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
