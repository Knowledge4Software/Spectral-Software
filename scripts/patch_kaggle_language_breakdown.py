"""Make every baseline report its test F1 per language, not just overall.

Three of the four benchmarks are multi-language, so a single F1 hides which
language a model actually handles: a model can look competitive while scoring
0.9 on Java and collapsing on Python. The method notebook already keeps
per-pair predictions; the baselines did not, so the split could not be
recovered afterwards.

This injects a self-contained helper and one call per baseline. It does **not**
retrain anything per language: it partitions the *existing* test predictions by
the language of the pair, exactly as asked. A pair whose two endpoints differ in
language is reported under ``java->python`` style keys so cross-language pairs
stay visible instead of being dropped or double counted.

Each run writes ``/kaggle/working/<dataset>_language_breakdown.csv``.

    python scripts/patch_kaggle_language_breakdown.py
    python scripts/patch_kaggle_language_breakdown.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"
DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")

HELPER_MARKER = "# === per-language breakdown helper"
HELPER = '''# === per-language breakdown helper (patched by scripts/patch_kaggle_language_breakdown.py) ===
# Splits an already-computed set of test predictions by the language of each
# pair. No retraining and no separate per-language model: this is the same run,
# reported per language so a strong average cannot hide a collapsed language.
import gzip as _gzip
import json as _json
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd

_LANGUAGE_CACHE = {}
LANGUAGE_BREAKDOWN_ROWS = []


def _resolve_codes_file():
    for root in (_Path("/kaggle/input"), _Path("/kaggle/working"), _Path(".")):
        if not root.exists():
            continue
        for name in ("codes.jsonl.gz", "codes.jsonl", "codes.jsonl.gz.tmp"):
            for path in root.rglob(name):
                if path.is_file():
                    return path
    return None


def _open_any(path):
    with open(path, "rb") as probe:
        packed = probe.read(2) == b"\\x1f\\x8b"
    return _gzip.open(path, "rt", encoding="utf-8") if packed else open(path, "r", encoding="utf-8")


def code_languages():
    """``code_id -> language`` from the attached clean-data bundle."""
    if _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE
    path = _resolve_codes_file()
    if path is None:
        print("[language-breakdown] codes.jsonl not found; breakdown will be skipped.")
        return _LANGUAGE_CACHE
    with _open_any(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            record = _json.loads(line)
            code_id = str(record.get("code_id", record.get("id", record.get("idx", ""))))
            _LANGUAGE_CACHE[code_id] = str(record.get("language", record.get("lang", "unknown")))
    print(f"[language-breakdown] languages loaded for {len(_LANGUAGE_CACHE):,} codes.")
    return _LANGUAGE_CACHE


def _binary_scores(labels, predicted):
    labels = _np.asarray(labels, dtype=_np.int64)
    predicted = _np.asarray(predicted, dtype=_np.int64)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "P": precision, "R": recall, "F1": f1,
        "Acc": (tp + tn) / max(1, len(labels)),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Pairs": int(len(labels)), "Positives": int((labels == 1).sum()),
    }


def record_language_breakdown(frame, scores, threshold, *, dataset, method, graph_type=None):
    """Partition this run's test predictions by pair language and record them."""
    languages = code_languages()
    if not languages or frame is None or not len(frame):
        return []
    scores = _np.asarray(scores, dtype=_np.float64).reshape(-1)
    labels = _np.asarray(frame["label"], dtype=_np.int64).reshape(-1)
    if len(scores) != len(labels):
        print(f"[language-breakdown] skipped {method}: {len(scores)} scores vs {len(labels)} labels.")
        return []
    predicted = (scores >= float(threshold)).astype(_np.int64)

    left = [languages.get(str(value), "unknown") for value in frame["left_id"]]
    right = [languages.get(str(value), "unknown") for value in frame["right_id"]]
    # Cross-language pairs get their own bucket instead of being attributed to
    # one side; ATCoder is entirely java<->python and would otherwise vanish.
    keys = [a if a == b else f"{min(a, b)}->{max(a, b)}" for a, b in zip(left, right)]

    rows = []
    for key in sorted(set(keys)):
        mask = _np.asarray([value == key for value in keys])
        row = {"Dataset": dataset, "Method": method, "GraphType": graph_type or "", "Language": key}
        row.update(_binary_scores(labels[mask], predicted[mask]))
        row["Threshold"] = float(threshold)
        rows.append(row)
    overall = {"Dataset": dataset, "Method": method, "GraphType": graph_type or "", "Language": "ALL"}
    overall.update(_binary_scores(labels, predicted))
    overall["Threshold"] = float(threshold)
    rows.append(overall)

    LANGUAGE_BREAKDOWN_ROWS.extend(rows)
    table = _pd.DataFrame(LANGUAGE_BREAKDOWN_ROWS)
    out_path = _Path("/kaggle/working") / f"{dataset}_language_breakdown.csv"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
    except OSError:
        out_path = _Path(f"{dataset}_language_breakdown.csv")
        table.to_csv(out_path, index=False)
    print(f"\\n[language-breakdown] {method}{'/' + graph_type if graph_type else ''}")
    print(_pd.DataFrame(rows)[["Language", "P", "R", "F1", "Acc", "Pairs", "Positives"]].to_string(index=False))
    print(f"[language-breakdown] written to {out_path}")
    return rows
'''

# (anchor, replacement, needs graph_type) per notebook.
INJECTIONS: dict[str, tuple[str, str, bool]] = {}


def _standard(threshold_var: str, method: str) -> tuple[str, str, bool]:
    anchor = (
        f'    test_metrics = metric_dict(test_df["label"].to_numpy(), test_scores, {threshold_var})\n'
    )
    call = (
        f"    record_language_breakdown(test_df, test_scores, {threshold_var}, "
        f'dataset=DATASET_KEY_FOR_BREAKDOWN, method="{method}")\n'
    )
    return anchor, anchor + call, False


for _name, _threshold, _method in [
    ("astnn_baseline.ipynb", "best_threshold", "ASTNN"),
    ("cdlh_baseline.ipynb", "best_threshold", "CDLH"),
    ("deckard_baseline.ipynb", "threshold", "Deckard"),
    ("deepsim_baseline.ipynb", "best_threshold", "DeepSim"),
    ("fa_ast_ggnn_baseline.ipynb", "best_threshold", "FA-AST+GGNN"),
    ("fa_ast_gmn_baseline.ipynb", "best_threshold", "FA-AST+GMN"),
    ("rtvnn_baseline.ipynb", "best_threshold", "RtvNN"),
]:
    INJECTIONS[_name] = _standard(_threshold, _method)

# The GNN notebook hides the probabilities inside evaluate(); compute them once
# and derive the same metrics from them so nothing is run twice.
INJECTIONS["gnn_baselines.ipynb"] = (
    "        test_metrics = evaluate(model, te, graphs, DEVICE, selected_threshold)\n",
    "        test_probs, test_labels = predict_probs(model, te, graphs, DEVICE)\n"
    "        test_metrics = binary_metrics(test_labels, test_probs, threshold=selected_threshold)\n"
    "        record_language_breakdown(te, test_probs, selected_threshold, "
    'dataset=DATASET_KEY_FOR_BREAKDOWN, method=f"GNN-{graph_type.upper()}", graph_type=graph_type)\n',
    True,
)

# build_graph_data() converts its train/valid/test frames to numeric index
# arrays and drops the frames, so `test_df` does not exist inside
# train_one_graph(). scripts/fix_snn_breakdown_scope.py makes that function also
# return the frames; the breakdown reads the test frame from there.
INJECTIONS["snn_baselines.ipynb"] = (
    "        test_metrics = metrics_at_threshold(y_test, test_scores, best_threshold)\n",
    "        test_metrics = metrics_at_threshold(y_test, test_scores, best_threshold)\n"
    '        record_language_breakdown(split_frames["test"], test_scores, best_threshold, '
    'dataset=DATASET_KEY_FOR_BREAKDOWN, method=f"{graph_type.upper()} + SNN", graph_type=graph_type)\n',
    True,
)


def patch_notebook(path: Path, dataset: str, apply: bool) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    anchor, replacement, _ = INJECTIONS[path.name]

    joined = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    already = replacement in joined
    has_helper = HELPER_MARKER in joined

    if already and has_helper:
        return "up to date"

    if not has_helper:
        helper_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": (
                HELPER + f'\nDATASET_KEY_FOR_BREAKDOWN = "{dataset}"\n'
            ).splitlines(keepends=True),
        }
        notebook["cells"].insert(1, helper_cell)

    if not already:
        hit = False
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if anchor in source:
                cell["source"] = source.replace(anchor, replacement, 1).splitlines(keepends=True)
                hit = True
                break
        if not hit:
            return "ANCHOR NOT FOUND"

    if apply:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()

    problems = 0
    for dataset in args.datasets:
        folder = KAGGLE_ROOT / dataset / "baselines"
        for name in sorted(INJECTIONS):
            path = folder / name
            if not path.is_file():
                print(f"  {dataset:24s} {name:26s} MISSING FILE")
                problems += 1
                continue
            status = patch_notebook(path, dataset, apply=not args.check)
            if status not in {"patched", "up to date"}:
                problems += 1
            if args.check and status == "patched":
                problems += 1
                status = "STALE"
            print(f"  {dataset:24s} {name:26s} {status}")

    if problems:
        print(f"\n[-] {problems} notebook(s) need attention.")
        return 1
    print("\n[+] every baseline now reports test F1 per language alongside the overall score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
