from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sync_spectra_language_breakdown import LANGUAGE_HELPER


ROOT = Path(__file__).resolve().parents[1]


def spectra_notebooks() -> list[Path]:
    paths = []
    for path in (ROOT / "experiments").rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        # RQ3 reuses the model/helper cells but deliberately has a separate
        # multi-scope transfer runner and a Table-7 result contract, validated
        # by test_rq3_notebooks.py rather than this main-run breakdown test.
        if (
            "class CanonicalSpectraSiam" in source
            and "def run_experiment()" in source
            and "def run_rq3_spectra" not in source
        ):
            paths.append(path)
    return sorted(paths)


def test_every_spectra_run_saves_language_breakdown_and_compiles():
    paths = spectra_notebooks()
    # A lower bound, not an exact count: notebooks are added for new studies
    # (RQ3, the sensitivity sweeps, the per-seed stability runs) and each one
    # must still satisfy the contract below. Pinning the exact number only
    # breaks this test whenever a study is added.
    assert len(paths) >= 33
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "def save_spectra_language_breakdown" in source, path
        assert "language_breakdown_path, language_breakdown_frame = save_spectra_language_breakdown" in source, path
        assert '"language_breakdown": str(language_breakdown_path)' in source, path
        assert 'result["language_breakdown_csv"]' in source, path
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{path}#cell-{index}", "exec")


def test_language_helper_groups_same_and_cross_language_pairs(tmp_path: Path):
    def metric_summary(labels, probabilities, threshold):
        labels = np.asarray(labels, dtype=np.int64)
        predicted = (np.asarray(probabilities) >= threshold).astype(np.int64)
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        tn = int(((predicted == 0) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        accuracy = (tp + tn) / max(1, len(labels))
        return {
            "Precision": tp / max(1, tp + fp), "Recall": tp / max(1, tp + fn),
            "F1": accuracy, "Accuracy": accuracy, "MacroF1": accuracy,
            "BalancedAccuracy": accuracy, "Threshold": float(threshold),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn, "Pairs": len(labels),
        }

    namespace = {
        "pd": pd, "np": np, "Path": Path, "WORK_DIR": tmp_path,
        "DATASET_KEY": "codenet-4l", "METHOD_VARIANT": "input_only_lex",
        "metric_summary": metric_summary, "display": lambda value: None,
    }
    exec(LANGUAGE_HELPER, namespace)
    frame = pd.DataFrame({
        "left_language": ["java", "cpp", "python", "csharp"],
        "right_language": ["java", "python", "cpp", "csharp"],
        "label": [1, 0, 1, 0],
    })
    path, breakdown = namespace["save_spectra_language_breakdown"](
        frame, np.asarray([0.9, 0.2, 0.8, 0.1]), 0.5,
    )
    assert path.exists()
    assert path.name.endswith("_language_breakdown.csv")
    assert set(breakdown.Language) == {"java", "cpp->python", "csharp", "ALL"}
    assert int(breakdown.loc[breakdown.Language == "ALL", "Pairs"].iloc[0]) == 4


def test_all_codenet_method_variants_have_language_output():
    directory = ROOT / "experiments/kaggle/rq2/codenet/method"
    assert {path.name for path in directory.glob("spectra_siam_*.ipynb")} == {
        "spectra_siam_topo.ipynb", "spectra_siam_label.ipynb", "spectra_siam_lex.ipynb",
    }
    for path in directory.glob("spectra_siam_*.ipynb"):
        source = path.read_text(encoding="utf-8")
        assert "learned_latent" in source
        assert "cross_language" in source
        assert "spectra_siam_language_breakdown.csv" in source
