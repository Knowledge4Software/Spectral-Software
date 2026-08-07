"""Stop a full-data run from dying at the Kaggle session limit with nothing to show.

Every main run is deliberately `final_full`, and at that setting the projected
worst case is SPECTRA-Siam on AtCoder at roughly 8.7h: under Kaggle's limit, but
close enough that one slow epoch loses the whole session. The merge fix is what
moved it - AtCoder's mean AST went from 42 to 200 nodes once a submission's
functions stopped being thrown away - so the risk is new and specific.

The guard does not shrink the data. It records when training started and, if the
next epoch would cross the budget, stops after the current one and evaluates with
the best checkpoint it already has. A run that ends early is reported as such via
``StoppedEarly`` rather than passing as a completed full-data run.

    python scripts/patch_kaggle_time_guard.py
    python scripts/patch_kaggle_time_guard.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"

MARKER = "# --- session time guard (scripts/patch_kaggle_time_guard.py) ---"

CONFIG_BLOCK = f'''{MARKER}
# Hard wall for one Kaggle session, in hours. Training stops after the current
# epoch when the next one is projected to cross it, and the best checkpoint so
# far is evaluated. Set to None to disable.
SESSION_BUDGET_HOURS = 8.5
'''

# Inserted directly after the epoch loop's validation step, where `epoch`,
# `started` and `best` are all in scope.
GUARD_BLOCK = '''        if SESSION_BUDGET_HOURS is not None:
            _elapsed_h = (time.perf_counter() - started) / 3600.0
            _per_epoch_h = _elapsed_h / max(epoch, 1)
            if epoch < EPOCHS and _elapsed_h + _per_epoch_h > SESSION_BUDGET_HOURS:
                print(f"[time-guard] {_elapsed_h:.2f}h used, ~{_per_epoch_h:.2f}h per epoch; "
                      f"stopping after epoch {epoch}/{EPOCHS} to stay inside "
                      f"{SESSION_BUDGET_HOURS}h and keep the best checkpoint.")
                STOPPED_EARLY = True
                break
'''

CONFIG_ANCHOR = re.compile(r'^(\s*)EPOCHS = RUN_CONFIG\["epochs"\]\n', re.M)
# The method notebook's training loop reports validation once per epoch here.
LOOP_ANCHOR = (
    '        if best is None or (valid["F1"], valid["BalancedAccuracy"]) > '
    '(best["valid"]["F1"], best["valid"]["BalancedAccuracy"]):\n'
)


def patch(path: Path, apply: bool) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    joined = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    if MARKER in joined and "[time-guard]" in joined:
        return "up to date"

    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        original = source

        if MARKER not in source:
            match = CONFIG_ANCHOR.search(source)
            if match:
                indent = match.group(1)
                block = "".join(indent + line if line.strip() else line
                                for line in CONFIG_BLOCK.splitlines(keepends=True))
                source = source[: match.end()] + block + source[match.end():]

        if LOOP_ANCHOR in source and "[time-guard]" not in source:
            source = source.replace(LOOP_ANCHOR, GUARD_BLOCK + LOOP_ANCHOR, 1)
            source = source.replace(
                "    history, best = [], None;",
                "    STOPPED_EARLY = False\n    history, best = [], None;", 1)

        if source != original:
            cell["source"] = source.splitlines(keepends=True)
            changed = True

    if not changed:
        return "ANCHOR NOT FOUND"
    if apply:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    problems = 0
    for path in sorted(KAGGLE_ROOT.glob("*_v3/method/spectra_siam.ipynb")):
        status = patch(path, apply=not args.check)
        if status == "ANCHOR NOT FOUND":
            problems += 1
        if args.check and status == "patched":
            problems += 1
            status = "STALE"
        print(f"  {str(path.relative_to(KAGGLE_ROOT)):50s} {status}")

    if problems:
        print(f"\n[-] {problems} notebook(s) need attention.")
        return 1
    print("\n[+] full-data runs now stop cleanly before the session limit instead of being killed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
