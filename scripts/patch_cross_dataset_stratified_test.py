"""Patch the SPECTRA cross-dataset notebook to cap test sets stratified by label."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle" / "experiments" / "02_cross_dataset" / "01_spectra_siam_train_on_codexglue.ipynb"
OLD = '''        if maximum is not None and len(frame) > maximum:
            pieces = []
            for _, group in frame.groupby("label", sort=True):
                size = max(1, round(maximum * len(group) / len(frame)))
                pieces.append(group.sample(n=min(size, len(group)), random_state=seed))
            frame = pd.concat(pieces).sample(frac=1, random_state=seed).head(maximum)
'''
NEW = '''        if maximum is not None and len(frame) > maximum:
            # Largest-remainder allocation preserves the official clone/non-clone
            # ratio under a cap; a balanced target therefore remains exactly 50/50.
            counts = frame["label"].value_counts().sort_index()
            exact = maximum * counts / len(frame)
            allocation = np.floor(exact).astype(int)
            remaining = int(maximum - allocation.sum())
            for label in sorted(counts.index, key=lambda label: (-float(exact[label] - allocation[label]), label)):
                if remaining <= 0:
                    break
                if allocation[label] < counts[label]:
                    allocation[label] += 1
                    remaining -= 1
            frame = pd.concat([
                group.sample(n=int(allocation[label]), random_state=seed + int(label))
                for label, group in frame.groupby("label", sort=True)
            ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"{split} label counts: {frame['label'].value_counts().sort_index().to_dict()}")
'''


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    replaced = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = "".join(cell["source"])
        if OLD in text:
            cell["source"] = text.replace(OLD, NEW, 1).splitlines(keepends=True)
            replaced = True
            break
    if not replaced:
        raise RuntimeError("Expected SPECTRA sampling block was not found.")
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("patched", NOTEBOOK)


if __name__ == "__main__":
    main()
