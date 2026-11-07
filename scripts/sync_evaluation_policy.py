"""Synchronize explicit benchmark selection and checkpoint ordering in legacy notebooks.

These notebooks retain their original four-epoch schedules. New converged,
resumable experiments live under revision/ instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def update_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    if not any(token in source for token in ("class CanonicalSpectraSiam", "def _validation_selection", "def validation_selection_context")):
        return False
    match = re.search(r'DATASET_KEY\s*=\s*["\']([^"\']+)', source)
    if match is None:
        match = re.search(r'DATASET_KEYS\s*=\s*\(\s*["\']([^"\']+)', source)
    dataset = match.group(1).lower() if match else str(path).lower()
    metric = "Accuracy" if any(key in dataset for key in ("atcoder", "at-coder", "codenet")) else "F1"
    changed = False
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        old = s = "".join(cell.get("source", []))
        if "RUN_PROFILE =" in s and "VALIDATION_SELECTION_METRIC =" not in s:
            s += f'\n# Fixed benchmark policy; graph filtering must not change the primary metric.\nVALIDATION_SELECTION_METRIC = "{metric}"\n'
        s = re.sub(r'VALIDATION_SELECTION_METRIC = ["\'](?:Accuracy|F1)["\']',
                   f'VALIDATION_SELECTION_METRIC = "{metric}"', s)
        s = s.replace('validation_selection_metric = "Accuracy" if validation_is_balanced else "F1"',
                      'validation_selection_metric = VALIDATION_SELECTION_METRIC')
        s = s.replace('"metric": "Accuracy" if balanced else "F1"',
                      '"metric": globals().get("VALIDATION_SELECTION_METRIC", "Accuracy" if balanced else "F1")')
        s = s.replace('if context["balanced"]:', 'if context["metric"] == "Accuracy":')
        # The current epoch must be eligible before stopping for session time.
        guard = "        if SESSION_BUDGET_HOURS is not None:\n"
        ranking = "        selection_ranking = (\n"
        save = '            torch.save({"model_state": model.state_dict(), "config": asdict(config), "best": best}, best_path)\n'
        if guard in s and ranking in s and save in s and s.index(guard) < s.index(ranking):
            start, end = s.index(guard), s.index(ranking)
            block = s[start:end]
            s = s[:start] + s[end:]
            s = s.replace(save, save + block, 1)
        if s != old:
            cell["source"] = s.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return changed


if __name__ == "__main__":
    paths = list((ROOT / "experiments/kaggle").rglob("*.ipynb"))
    paths += [ROOT / "spectral_code/templates/spectra_siam_kaggle_template.ipynb"]
    print(f"Updated {sum(update_notebook(path) for path in paths)} notebook(s)")
