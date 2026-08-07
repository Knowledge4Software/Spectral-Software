"""Add a `bounded_10h` run profile and make it the default everywhere.

`final_full` uses every pair in every split. On BigCloneBench that is 901k train
and 415k validation pairs, and the validation split only exists to pick one
scalar threshold - so most of a long run is spent on something that does not
change the model. Measured on the previous full run, SPECTRA-Siam took 3.5h on
BigCloneBench and every baseline finished inside 1.1h.

Two things have changed since those numbers were measured:

* AtCoder graphs are much larger now that a submission's functions are merged
  rather than one arbitrary function kept (mean AST 42 -> 200 nodes), so its
  runtime scales up even though its pair count did not.
* CodeXGLUE graphs are unchanged (mean 153.4 nodes before and after), because
  every BigCloneBench record is a single method. Its old timings still hold.

`bounded_10h` therefore keeps the training data large enough to stay comparable
with the published table, and caps validation and test - the two splits whose
size buys almost nothing - so no single Kaggle session approaches the limit.

    python scripts/patch_kaggle_run_budget.py
    python scripts/patch_kaggle_run_budget.py --check
    python scripts/patch_kaggle_run_budget.py --profile final_full   # revert
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"

MARKER = "# --- bounded run budget (scripts/patch_kaggle_run_budget.py) ---"

# Baselines and the method share the same keys, but the method notebook also
# reads "epochs" from the preset while several baselines additionally use
# "patience"; spreading an existing preset keeps whatever else each one needs.
BUDGET_BLOCK = f'''{MARKER}
# Kaggle sessions are capped, and a run that dies at the limit produces nothing.
# Training data stays large so results remain comparable with the published
# table; validation and test are capped because a bigger validation split only
# sharpens one threshold, and a bigger test split only tightens an error bar we
# do not report.
RUN_PRESETS["bounded_10h"] = {{
    **RUN_PRESETS["comparison_50k"],
    "max_train_pairs": 200_000,
    "max_valid_pairs": 20_000,
    "max_test_pairs": 20_000,
}}
'''

METHOD_BUDGET_BLOCK = f'''{MARKER}
# SPECTRA-Siam is the slowest model in the table (3.5h on BigCloneBench at full
# data) and its cost grows with graph size, which the merge fix increased on
# AtCoder. Fewer, larger epochs keep the number of pair presentations close to
# the published run while bounding wall time.
RUN_PRESETS["bounded_10h"] = {{
    **RUN_PRESETS["comparison_50k"],
    "max_train_pairs": 200_000,
    "max_valid_pairs": 20_000,
    "max_test_pairs": 20_000,
    "epochs": 4,
}}
'''

ANCHOR = re.compile(r'^(\s*)RUN_PRESETS\["final_full"\] = \{.*?\n\1\}\n', re.S | re.M)
PROFILE_LINE = re.compile(r'^(\s*)RUN_PROFILE = "(\w+)"\n', re.M)


def patch_notebook(path: Path, profile: str, is_method: bool, apply: bool) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    block = METHOD_BUDGET_BLOCK if is_method else BUDGET_BLOCK

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "RUN_PRESETS" not in source:
            continue
        original = source

        if MARKER not in source:
            match = ANCHOR.search(source)
            if not match:
                return "ANCHOR NOT FOUND"
            indent = match.group(1)
            indented = "".join(
                (indent + line if line.strip() else line) for line in block.splitlines(keepends=True)
            )
            source = source[: match.end()] + "\n" + indented + source[match.end():]

        source = PROFILE_LINE.sub(lambda m: f'{m.group(1)}RUN_PROFILE = "{profile}"\n', source, count=1)

        if source != original:
            changed = True
            cell["source"] = source.splitlines(keepends=True)
        break

    if changed and apply:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return "patched" if changed else "up to date"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    # The main runs are deliberately full-data; `bounded_10h` exists as a
    # ready-made fallback for when a session is at risk, not as the default.
    parser.add_argument("--profile", default="final_full",
                        help="Profile to select as the default (bounded_10h caps train/valid/test).")
    args = parser.parse_args()

    targets = sorted(KAGGLE_ROOT.glob("*_v3/baselines/*.ipynb"))
    methods = sorted(KAGGLE_ROOT.glob("*_v3/method/spectra_siam.ipynb"))
    # Each sweep notebook trains 3-4 times in one session, so the per-training
    # budget must be a fraction of a single run's. They keep comparison_50k,
    # which is also the profile that makes their arms comparable to each other.
    experiments = sorted(KAGGLE_ROOT.glob("experiments/*/*.ipynb"))

    problems = 0
    for path in targets + methods:
        is_method = path.parent.name == "method"
        status = patch_notebook(path, args.profile, is_method, apply=not args.check)
        if status == "ANCHOR NOT FOUND":
            problems += 1
        if args.check and status == "patched":
            problems += 1
            status = "STALE"
        print(f"  {str(path.relative_to(KAGGLE_ROOT)):52s} {status}")

    for path in experiments:
        # Still give them the preset so it can be selected by hand, but do not
        # make a multi-training notebook default to the big profile.
        status = patch_notebook(path, "comparison_50k", True, apply=not args.check)
        if args.check and status == "patched":
            problems += 1
            status = "STALE"
        print(f"  {str(path.relative_to(KAGGLE_ROOT)):52s} {status}  (kept comparison_50k)")

    if problems:
        print(f"\n[-] {problems} notebook(s) need attention.")
        return 1
    print(f"\n[+] baselines and method default to {args.profile!r}; "
          f"multi-training experiments stay on 'comparison_50k'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
