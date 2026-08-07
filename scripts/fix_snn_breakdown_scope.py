"""Give the SNN baseline access to the pair frame its breakdown call needs.

The per-language patch assumed `test_df` was in scope at the call site. In
`snn_baselines` it is not: `build_graph_data()` builds train/valid/test frames,
converts them to index arrays, and discards the frames, so `train_one_graph()`
only ever sees `PairArrays` (numeric indices, no `left_id`/`right_id`). The call
therefore raised `NameError: name 'test_df' is not defined` after training had
already finished - the worst possible time on a capped Kaggle session.

`build_graph_data` now also returns the frames, and the breakdown reads the test
frame from there. Nothing about the model, the splits, or the metrics changes:
the same rows, in the same order, are simply also handed back so the pair ids
can be recovered.

    python scripts/fix_snn_breakdown_scope.py
    python scripts/fix_snn_breakdown_scope.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"

# 1. Return the frames alongside the arrays.
OLD_RETURN = """        return matrix, {
            "train": split_arrays(train_df),
            "valid": split_arrays(valid_df),
            "test": split_arrays(test_df),
        }, len(used_ids)
"""
NEW_RETURN = """        # The frames are returned as well: split_arrays() reduces them to numeric
        # indices, which cannot be mapped back to code ids for the per-language
        # breakdown.
        return matrix, {
            "train": split_arrays(train_df),
            "valid": split_arrays(valid_df),
            "test": split_arrays(test_df),
        }, len(used_ids), {"train": train_df, "valid": valid_df, "test": test_df}
"""

# 2. Unpack the extra value at the single call site.
OLD_CALL = "        code_matrix, split_data, code_count = build_graph_data(pairs_df, vectors, seed)\n"
NEW_CALL = "        code_matrix, split_data, code_count, split_frames = build_graph_data(pairs_df, vectors, seed)\n"

# 3. Read the test frame from what was returned.
OLD_BREAKDOWN = (
    "        record_language_breakdown(test_df, test_scores, best_threshold, "
    'dataset=DATASET_KEY_FOR_BREAKDOWN, method=f"{graph_type.upper()} + SNN", graph_type=graph_type)\n'
)
NEW_BREAKDOWN = (
    "        record_language_breakdown(split_frames[\"test\"], test_scores, best_threshold, "
    'dataset=DATASET_KEY_FOR_BREAKDOWN, method=f"{graph_type.upper()} + SNN", graph_type=graph_type)\n'
)

EDITS = [
    (OLD_RETURN, NEW_RETURN, "return frames"),
    (OLD_CALL, NEW_CALL, "unpack frames"),
    (OLD_BREAKDOWN, NEW_BREAKDOWN, "breakdown uses returned frame"),
]


def patch(path: Path, apply: bool) -> tuple[str, list[str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        original = source
        for old, new, _ in EDITS:
            if old in source:
                # The unpacking line appears more than once in some notebooks;
                # replacing every occurrence keeps them consistent.
                source = source.replace(old, new)
        if source != original:
            cell["source"] = source.splitlines(keepends=True)
            changed = True

    if changed and apply:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    verify = json.loads(path.read_text(encoding="utf-8")) if apply else notebook
    joined = "\n".join("".join(c["source"]) for c in verify["cells"] if c["cell_type"] == "code")
    missing = [label for _, new, label in EDITS if new not in joined]
    return ("patched" if changed else "up to date"), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    problems = 0
    for path in sorted(KAGGLE_ROOT.glob("*_v3/baselines/snn_baselines.ipynb")):
        status, missing = patch(path, apply=not args.check)
        if missing:
            problems += 1
            print(f"  {str(path.relative_to(KAGGLE_ROOT)):52s} {status}  MISSING: {missing}")
        else:
            if args.check and status == "patched":
                problems += 1
                status = "STALE"
            print(f"  {str(path.relative_to(KAGGLE_ROOT)):52s} {status}")

    if problems:
        print(f"\n[-] {problems} notebook(s) need attention.")
        return 1
    print("\n[+] the SNN breakdown now reads the frame build_graph_data returns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
