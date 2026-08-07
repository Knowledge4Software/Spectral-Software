"""Create one all-input cross-dataset Kaggle notebook for each selected method.

Each notebook prepares a namespaced temporary clean-data directory with
CodeXGLUE train/valid and the three target test splits, trains exactly once,
then reports a separate row for each target.  It is deliberately self-contained
for Kaggle: attach the four final ``*_clean_data.zip`` datasets and Run All.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
OUT = KAGGLE / "experiments" / "02_cross_dataset"
METHODS = ("astnn", "rtvnn", "deepsim")


def read_nb(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_nb(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cell_source(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, value: str) -> None:
    cell["source"] = [line + "\n" for line in value.splitlines()]


PREPARE_DATA = r'''
# Build one private, namespaced clean-data directory from the four Kaggle inputs.
# train+valid = CodeXGLUE; test = AtCoder + GPTCloneBench + SemanticCloneBench.
from pathlib import Path
import csv, gzip, json, zipfile

INPUT_ROOT = Path("/kaggle/input")
WORK_DIR = Path("/kaggle/working")
TRANSFER_CLEAN_DATA_DIR = WORK_DIR / "cross_dataset_all_inputs" / "clean_data"
DATASET_ORDER = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")
ID_PREFIX = {"codexglue_v3": "cx", "atcoder_v3": "at", "gptclonebench_v3": "gp", "semanticclonebench_v3": "sc"}

def _gzip(path):
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"

def _open(path, mode="rt"):
    return gzip.open(path, mode, encoding="utf-8") if _gzip(path) else path.open(mode, encoding="utf-8")

def _artifact(root, names):
    for name in names:
        direct = root / name
        if direct.is_file(): return direct
        # Kaggle may unpack ``codes.jsonl.gz.tmp`` *inside a directory named*
        # ``codes.jsonl``.  Never return that directory: callers open an
        # artifact as text, so only a real file is valid here.
        found = [item for item in root.rglob(name) if item.is_file()]
        if found: return sorted(found, key=lambda x: (len(x.parts), str(x)))[0]
    return None

def _valid(root):
    return (_artifact(root, ("codes.jsonl.gz", "codes.jsonl", "codes.jsonl.gz.tmp"))
            and _artifact(root, ("pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp"))
            and _artifact(root, ("graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp")))

# Kaggle often exposes a ZIP as a file rather than extracting it. Extract only
# archive inputs that actually contain a clean-data export.
for archive in INPUT_ROOT.rglob("*.zip"):
    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            if not (any("codes.jsonl" in n for n in names) and any("pairs.csv" in n for n in names)
                    and any("graph_spectra.jsonl" in n for n in names)):
                continue
            destination = WORK_DIR / "attached_clean_data" / archive.stem
            marker = destination / ".done"
            if not marker.exists():
                destination.mkdir(parents=True, exist_ok=True)
                zf.extractall(destination)
                marker.write_text("ok", encoding="utf-8")
    except zipfile.BadZipFile:
        pass

roots = [path.parent for path in list(INPUT_ROOT.rglob("metadata.json"))
         + list((WORK_DIR / "attached_clean_data").rglob("metadata.json"))]
dataset_roots = {}
for root in roots:
    try:
        manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        continue
    key = manifest.get("dataset_key")
    if key in DATASET_ORDER and _valid(root):
        dataset_roots[key] = root
missing = [key for key in DATASET_ORDER if key not in dataset_roots]
if missing:
    raise FileNotFoundError("Attach all four final clean-data ZIPs. Missing: " + ", ".join(missing))
print("Inputs:", {key: str(value) for key, value in dataset_roots.items()})

def _read_pairs(root):
    path = _artifact(root, ("pairs.csv.gz", "pairs.csv", "pairs.csv.gz.tmp"))
    with _open(path) as handle:
        return list(csv.DictReader(handle))

selected, wanted = [], {key: set() for key in DATASET_ORDER}
for key in DATASET_ORDER:
    keep_split = {"train", "valid"} if key == "codexglue_v3" else {"test"}
    for row in _read_pairs(dataset_roots[key]):
        split = str(row["split"]).strip().lower()
        if split not in keep_split: continue
        prefix = ID_PREFIX[key]
        selected.append({"split": split, "left_id": f"{prefix}_{row['left_id']}",
                         "right_id": f"{prefix}_{row['right_id']}", "label": row["label"],
                         "label_name": row.get("label_name", ""),
                         "test_dataset": key if split == "test" else ""})
        wanted[key].update((str(row["left_id"]), str(row["right_id"])))

if TRANSFER_CLEAN_DATA_DIR.exists():
    # The marker means this exact, deterministic materialization already exists.
    marker = TRANSFER_CLEAN_DATA_DIR / ".ready"
    if not marker.exists():
        raise RuntimeError(f"Refusing to reuse incomplete directory: {TRANSFER_CLEAN_DATA_DIR}")
else:
    TRANSFER_CLEAN_DATA_DIR.mkdir(parents=True)
    with gzip.open(TRANSFER_CLEAN_DATA_DIR / "pairs.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "left_id", "right_id", "label", "label_name", "test_dataset"])
        writer.writeheader(); writer.writerows(selected)
    for out_name, candidates in (("codes.jsonl.gz", ("codes.jsonl.gz", "codes.jsonl", "codes.jsonl.gz.tmp")),
                                 ("graph_spectra.jsonl.gz", ("graph_spectra.jsonl.gz", "graph_spectra.jsonl", "graph_spectra.jsonl.gz.tmp"))):
        with gzip.open(TRANSFER_CLEAN_DATA_DIR / out_name, "wt", encoding="utf-8") as output:
            for key in DATASET_ORDER:
                source = _artifact(dataset_roots[key], candidates)
                with _open(source) as input_handle:
                    for line in input_handle:
                        if not line.strip(): continue
                        record = json.loads(line)
                        raw_id = str(record.get("code_id"))
                        if raw_id not in wanted[key]: continue
                        record["code_id"] = f"{ID_PREFIX[key]}_{raw_id}"
                        record["source_dataset"] = key
                        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    (TRANSFER_CLEAN_DATA_DIR / "metadata.json").write_text(json.dumps({
        "bundle": "cross_dataset_all_inputs", "train_valid_from": "codexglue_v3",
        "test_from": list(DATASET_ORDER[1:]), "id_prefixes": ID_PREFIX}, indent=2), encoding="utf-8")
    (TRANSFER_CLEAN_DATA_DIR / ".ready").write_text("ok", encoding="utf-8")
print("Prepared:", TRANSFER_CLEAN_DATA_DIR, "pairs=", len(selected))
'''


TEST_LIMIT = '''
    def limit_transfer_test(frame: pd.DataFrame) -> pd.DataFrame:
        part = frame[frame["split"] == "test"].copy()
        if "test_dataset" not in part.columns:
            raise RuntimeError("Prepared transfer data lacks test_dataset provenance.")
        pieces = []
        for number, (target_dataset, group) in enumerate(part.groupby("test_dataset", sort=True)):
            if MAX_TEST_PAIRS is not None and len(group) > MAX_TEST_PAIRS:
                # Allocate the cap across labels by largest remainder. This
                # preserves the official target-test clone/non-clone ratio as
                # closely as an integer sample permits (exactly for 50/50).
                counts = group["label"].value_counts().sort_index()
                exact = MAX_TEST_PAIRS * counts / len(group)
                allocation = np.floor(exact).astype(int)
                remaining = int(MAX_TEST_PAIRS - allocation.sum())
                for label in sorted(counts.index, key=lambda label: (-float(exact[label] - allocation[label]), label)):
                    if remaining <= 0:
                        break
                    if allocation[label] < counts[label]:
                        allocation[label] += 1
                        remaining -= 1
                group = pd.concat([
                    group[group["label"] == label].sample(n=int(allocation[label]), random_state=SEED + 200 + number * 10 + int(label))
                    for label in counts.index
                ], ignore_index=True)
            label_counts = group["label"].value_counts().sort_index().to_dict()
            print(f"test {target_dataset}: pairs={len(group):,}, label counts={label_counts}")
            pieces.append(group)
        return pd.concat(pieces, ignore_index=True)
'''


TARGET_METRICS = '''
    transfer_target_rows = []
    for target_dataset, target_frame in test_df.groupby("test_dataset", sort=True):
        target_scores = predict_scores(model, target_frame.reset_index(drop=True))
        target_metrics = metric_dict(target_frame["label"].to_numpy(), target_scores, best_threshold)
        transfer_target_rows.append({
            "Experiment": "cross_dataset_transfer", "Method": METHOD_NAME,
            "TrainedOn": "codexglue_v3", "TestedOn": target_dataset,
            "BestEpoch": best_epoch, "BestValidF1": best_f1, **target_metrics,
            "Threshold": best_threshold, "TrainPairs": len(train_df),
            "ValidPairs": len(valid_df), "TestPairs": len(target_frame),
            "RunProfile": RUN_PROFILE,
        })
    results_df = pd.DataFrame(transfer_target_rows)
'''


FINAL_RUNNER = '''
from IPython.display import display
import pandas as pd

print("\\n" + "=" * 96)
print("Training once on CodeXGLUE; evaluating all three target datasets")
print("=" * 96)
combined_results = run_one_dataset("cross_dataset_all_inputs")
combined_results.to_csv(Path("/kaggle/working") / f"{RUN_LABEL}_all_targets_results.csv", index=False)
display(combined_results)
print("Saved:", Path("/kaggle/working") / f"{RUN_LABEL}_all_targets_results.csv")
'''


def make_baseline(method: str, index: int) -> None:
    notebook = copy.deepcopy(read_nb(KAGGLE / "codexglue_v3" / "baselines" / f"{method}_baseline.ipynb"))
    config = cell_source(notebook["cells"][0])
    config = config.replace('DATASET_KEYS = ("codexglue",)', 'DATASET_KEYS = ("cross_dataset_all_inputs",)')
    config = config.replace('RUN_PROFILE = "final_full"', 'RUN_PROFILE = "transfer_250k"')
    anchor = '# Final paper protocol: use every available pair in each official split.\n'
    transfer_preset = '''# Cross-dataset transfer budget: train once on a substantial CodeXGLUE sample,
# choose the frozen threshold from a bounded source validation set, and evaluate
# up to 20k pairs per target dataset.
RUN_PRESETS["transfer_250k"] = {
    **RUN_PRESETS["comparison_50k"],
    "max_train_pairs": 250_000,
    "max_valid_pairs": 20_000,
    "max_test_pairs": 20_000,
}

'''
    if anchor not in config:
        raise RuntimeError(f"Preset insertion point missing in {method} config.")
    config = config.replace(anchor, transfer_preset + anchor, 1)
    config = config.replace(f'RUN_LABEL = "{method}_baseline"', f'RUN_LABEL = "cross_dataset_all_{method}"')
    set_source(notebook["cells"][0], config)
    notebook["cells"][2]["source"] = [
        f"# Cross-dataset transfer — {method.upper()}\n", "\n",
        "Attach the four final clean-data ZIPs. This notebook trains once on CodeXGLUE train/valid, "
        "freezes that validation threshold, then evaluates AtCoder, GPTCloneBench, and SemanticCloneBench.\n",
    ]
    notebook["cells"].insert(1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                                  "source": [line + "\n" for line in PREPARE_DATA.splitlines()]})
    body = cell_source(notebook["cells"][4])
    body = body.replace('KAGGLE_DATA_ROOT = Path("/kaggle/input") / DATASET_KEY', 'KAGGLE_DATA_ROOT = TRANSFER_CLEAN_DATA_DIR', 1)
    body = body.replace('return [root for root in [KAGGLE_DATA_ROOT, Path("/kaggle/input")] if root.exists()]', 'return [KAGGLE_DATA_ROOT] if KAGGLE_DATA_ROOT.exists() else []', 1)
    marker = '    def metric_dict(labels, scores, threshold: float) -> dict:\n'
    body = body.replace(marker, TEST_LIMIT + '\n' + marker, 1)
    body = body.replace('test_df = maybe_limit_split(pairs_df, "test", MAX_TEST_PAIRS, SEED + 2)', 'test_df = limit_transfer_test(pairs_df)', 1)
    record_lines = [line for line in body.splitlines() if 'record_language_breakdown(test_df, test_scores' in line]
    if len(record_lines) != 1:
        raise RuntimeError(f"Could not find one language-breakdown call for {method}.")
    body = body.replace(record_lines[0], TARGET_METRICS.rstrip(), 1)
    runner = 'from IPython.display import display\nimport pandas as pd\n\nall_dataset_results = {}'
    cut = body.find(runner)
    if cut < 0:
        raise RuntimeError(f"Could not replace final runner for {method}.")
    set_source(notebook["cells"][4], body[:cut] + FINAL_RUNNER)
    write_nb(OUT / f"0{index}_{method}_train_xglue_test_all.ipynb", notebook)


def main() -> None:
    for index, method in enumerate(METHODS, start=2):
        make_baseline(method, index)
        print("created", OUT / f"0{index}_{method}_train_xglue_test_all.ipynb")


if __name__ == "__main__":
    main()
