"""Stop the graph baselines from scoring layers a dataset cannot actually provide.

Joern's C# frontend crashes while building the CFG/DDG overlays
(``AssertionError: ... malformed cpg``), so in every benchmark that contains C#
those two layers are empty or degenerate for the C# half:

    csharp cfg   coverage   0.0%
    csharp ddg   coverage  94.1%   100% body-less 3-node graphs

A baseline trained on such a layer is not a weak baseline, it is a meaningless
one: a quarter of its records carry no structure at all. Reporting it next to
the method would understate the comparison rather than strengthen it. Until a
tree-sitter CFG/DDG builder exists for C# (the Python branch already has one),
those layers are dropped per dataset instead of silently scored.

The AST and CPG layers stay: C# AST comes from tree-sitter and is healthy
(correlation 0.88-0.91), and CPG is dominated by it.

Run from the repository root::

    python scripts/patch_kaggle_layer_support.py            # apply
    python scripts/patch_kaggle_layer_support.py --check    # report only, exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"

from spectral_code.evaluation.dataset_limitations import (
    CSHARP_OVERLAY_FAILURE,
    unsupported_layers,
)

DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")
# Single source of truth, shared with scripts/check_graph_health.py.
UNSUPPORTED_LAYERS: dict[str, list[str]] = {name: unsupported_layers(name) for name in DATASETS}
REASON = CSHARP_OVERLAY_FAILURE.replace("\n", " ")

MARKER = "# --- layer support (patched by scripts/patch_kaggle_layer_support.py) ---"


def _supported_block(dataset: str) -> str:
    dropped = UNSUPPORTED_LAYERS[dataset]
    if dropped:
        note = f"    # {REASON}\n"
    else:
        note = "    # Every language in this benchmark provides all four layers.\n"
    return (
        f"{MARKER}\n"
        f"    UNSUPPORTED_GRAPH_LAYERS = {dropped!r}\n"
        f"{note}"
        "    GRAPH_TYPES = [layer for layer in GRAPH_TYPES if layer not in UNSUPPORTED_GRAPH_LAYERS]\n"
        "    print('graph layers scored:', GRAPH_TYPES, '| dropped:', UNSUPPORTED_GRAPH_LAYERS)\n"
    )


ANCHORS = {
    "gnn_baselines.ipynb": '    GRAPH_TYPES = ["ast", "cfg", "ddg", "cpg"]\n',
    "snn_baselines.ipynb": "    GRAPH_TYPES = [str(name).lower() for name in GRAPH_TYPES]\n",
}


def patch_notebook(path: Path, dataset: str, apply: bool) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    anchor = ANCHORS[path.name]
    block = _supported_block(dataset)

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if anchor not in source:
            continue
        if MARKER in source:
            existing_start = source.index(MARKER)
            existing_end = source.index("\n", source.index("print('graph layers scored:'", existing_start)) + 1
            if source[existing_start:existing_end] == block:
                return "up to date"
            source = source[:existing_start] + block + source[existing_end:]
        else:
            source = source.replace(anchor, anchor + block, 1)
        if apply:
            cell["source"] = source.splitlines(keepends=True)
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        return "patched"
    return "anchor not found"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Report without writing; exit 1 if anything is stale.")
    args = parser.parse_args()

    stale = 0
    for dataset in sorted(UNSUPPORTED_LAYERS):
        folder = KAGGLE_ROOT / dataset / "baselines"
        if not folder.is_dir():
            print(f"[!] missing {folder}")
            continue
        for name in ANCHORS:
            path = folder / name
            if not path.is_file():
                print(f"[!] missing {path}")
                continue
            status = patch_notebook(path, dataset, apply=not args.check)
            if status != "up to date":
                stale += 1
            dropped = UNSUPPORTED_LAYERS[dataset] or ["-"]
            print(f"  {dataset:24s} {name:22s} drop={','.join(dropped):9s} {status}")

    if args.check and stale:
        print(f"\n[-] {stale} notebook(s) need patching: run scripts/patch_kaggle_layer_support.py")
        return 1
    print(f"\n[+] {'checked' if args.check else 'applied'}; C#-bearing benchmarks now score AST/CPG only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
