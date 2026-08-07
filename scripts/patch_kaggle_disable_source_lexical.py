"""Turn the source-token residual off in every run except the ablation that studies it.

The configuration carried a dataset-specific exception::

    USE_SOURCE_LEXICAL = DATASET_KEY == "at-coder"
    EMBEDDING_CONTRASTIVE_WEIGHT = 0.30 if DATASET_KEY == "at-coder" else 0.0

so AtCoder - and only AtCoder - fed the model a hashed sketch of the raw source
tokens plus an extra embedding-alignment loss. That is precisely the benchmark
carrying the paper's strongest claim (the only structural method that survives
cross-language), and it makes that number partly lexical while every other
column is not. Two runs that differ in their inputs cannot sit in the same table.

After this patch:

* every method notebook and every experiment runs with `USE_SOURCE_LEXICAL =
  False` and no AtCoder-only contrastive term;
* `04_feature_ablation` still switches the flag per arm, because measuring what
  the source tokens are worth is the entire point of that experiment.

    python scripts/patch_kaggle_disable_source_lexical.py
    python scripts/patch_kaggle_disable_source_lexical.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"
TEMPLATE = PROJECT_ROOT / "research" / "latent_graph_learning" / "notebooks" / "spectra_siam_kaggle_template.ipynb"

OLD_LEXICAL = 'USE_SOURCE_LEXICAL = DATASET_KEY == "at-coder"\n'
NEW_LEXICAL = (
    "# Raw source tokens stay out of every reported run: the structural claim has to\n"
    "# be measured on structure. 04_feature_ablation overrides this per arm, which is\n"
    "# the one place the residual is the subject of the experiment rather than a\n"
    "# silent advantage on a single benchmark.\n"
    "USE_SOURCE_LEXICAL = False\n"
)

OLD_CONTRASTIVE = 'EMBEDDING_CONTRASTIVE_WEIGHT = 0.30 if DATASET_KEY == "at-coder" else 0.0\n'
NEW_CONTRASTIVE = (
    "# Was applied to AtCoder only, which made its loss function different from every\n"
    "# other column in the table.\n"
    "EMBEDDING_CONTRASTIVE_WEIGHT = 0.0\n"
)

REPLACEMENTS = ((OLD_LEXICAL, NEW_LEXICAL), (OLD_CONTRASTIVE, NEW_CONTRASTIVE))


def patch(path: Path, apply: bool) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        original = source
        for old, new in REPLACEMENTS:
            if old in source:
                source = source.replace(old, new, 1)
        if source != original:
            cell["source"] = source.splitlines(keepends=True)
            changed = True
    if changed and apply:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    verify = json.loads(path.read_text(encoding="utf-8")) if apply else notebook
    joined = "\n".join("".join(c["source"]) for c in verify["cells"] if c["cell_type"] == "code")
    if OLD_LEXICAL in joined or OLD_CONTRASTIVE in joined:
        return "STILL DATASET-CONDITIONAL"
    return "patched" if changed else "up to date"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    paths = sorted(KAGGLE_ROOT.glob("*_v3/method/spectra_siam.ipynb"))
    paths += sorted(KAGGLE_ROOT.glob("experiments/*/*.ipynb"))
    if TEMPLATE.is_file():
        paths.append(TEMPLATE)

    problems = 0
    for path in paths:
        status = patch(path, apply=not args.check)
        if status == "STILL DATASET-CONDITIONAL":
            problems += 1
        if args.check and status == "patched":
            problems += 1
            status = "STALE"
        try:
            label = path.relative_to(KAGGLE_ROOT)
        except ValueError:
            label = path.relative_to(PROJECT_ROOT)
        print(f"  {str(label):56s} {status}")

    if problems:
        print(f"\n[-] {problems} notebook(s) still enable the residual on a single benchmark.")
        return 1
    print("\n[+] source tokens are off everywhere except the ablation arm that measures them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
