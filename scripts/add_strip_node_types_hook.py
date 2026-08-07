"""Add the STRIP_NODE_TYPES switch the topology-only ablation needs.

The encoder already has switches for the lexical sketch and for source tokens,
but no way to remove the *canonical node categories*. Without that, the
"pure topology" arm of the ablation cannot be run: every node would still carry
``Control_If`` / ``Call_Expr`` / ``Literal_Num``, which is exactly the signal the
arm is supposed to withhold.

The switch collapses every node to one canonical id, leaving adjacency as the
only input. It defaults to ``False`` everywhere, so the canonical runs and the
published results are unaffected.

    python scripts/add_strip_node_types_hook.py
    python scripts/add_strip_node_types_hook.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ROOT = PROJECT_ROOT / "kaggle"

CONFIG_ANCHOR = "USE_SOURCE_LEXICAL = DATASET_KEY == \"at-coder\"\n"
CONFIG_ADDITION = (
    "# Topology-only ablation: collapse every canonical category to a single id so\n"
    "# the encoder sees adjacency and nothing else. False for all normal runs.\n"
    "STRIP_NODE_TYPES = False\n"
)

BUILDER_ANCHOR = "    canonical_ids = np.asarray([raw_id + 1 for raw_id in raw_type_ids], dtype=np.int64)\n"
BUILDER_REPLACEMENT = (
    "    canonical_ids = np.asarray([raw_id + 1 for raw_id in raw_type_ids], dtype=np.int64)\n"
    "    if strip_node_types:\n"
    "        # One shared id for every node: adjacency survives, categories do not.\n"
    "        canonical_ids = np.ones_like(canonical_ids)\n"
)

# The runnable notebooks carry an extra ``source_slots`` parameter that the
# template does not, so each anchor is matched in both spellings.
def _variants(tail: str, addition: str, indent: str, extras: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs = []
    for extra in extras:
        anchor = f"{indent}lexical_slots: int = DEFAULT_LEXICAL_SLOTS,\n{extra}{tail}"
        pairs.append((anchor, f"{indent}lexical_slots: int = DEFAULT_LEXICAL_SLOTS,\n{extra}{addition}{tail}"))
    return pairs


SIGNATURE_VARIANTS = _variants(
    ") -> CanonicalGraph | None:\n",
    "    strip_node_types: bool = False,\n",
    "    ",
    ('    source_code: str = "",\n    source_slots: int = 96,\n', "    source_slots: int = 96,\n", ""),
)
LOAD_SIGNATURE_VARIANTS = _variants(
    "    ) -> dict[str, CanonicalGraph]:\n",
    "        strip_node_types: bool = False,\n",
    "        ",
    ("        source_slots: int = 96,\n", ""),
)

CORPUS_VARIANTS = [
    (
        "                    lexical_slots=lexical_slots,\n" + extra + "                )\n",
        "                    lexical_slots=lexical_slots,\n" + extra
        + "                    strip_node_types=strip_node_types,\n                )\n",
    )
    for extra in (
        '                    source_code=self.code_texts.get(code_id, ""),\n                    source_slots=source_slots,\n',
        "",
    )
]

CALL_ANCHOR = "    graphs = corpus.load_graphs(pair_code_ids(*frames.values()), max_nodes=MAX_AST_NODES"
CALL_REPLACEMENT = (
    "    graphs = corpus.load_graphs(pair_code_ids(*frames.values()), max_nodes=MAX_AST_NODES,"
    " strip_node_types=STRIP_NODE_TYPES"
)

EDITS = [
    (BUILDER_ANCHOR, BUILDER_REPLACEMENT, "canonical id collapse"),
    (CALL_ANCHOR, CALL_REPLACEMENT, "prepare_experiment_data call"),
]
ALTERNATIVES = [
    # The runnable notebooks declare USE_SOURCE_LEXICAL; the reference template
    # predates it, so anchor on the lexical dropout it does have.
    (
        [
            (CONFIG_ANCHOR, CONFIG_ANCHOR + CONFIG_ADDITION),
            ("LEXICAL_DROPOUT = 0.30\n", "LEXICAL_DROPOUT = 0.30\n" + CONFIG_ADDITION),
        ],
        "config switch",
    ),
    (SIGNATURE_VARIANTS, "graph builder signature"),
    (LOAD_SIGNATURE_VARIANTS, "load_graphs signature"),
    (CORPUS_VARIANTS, "load_graphs forwarding"),
]


def patch(path: Path, apply: bool) -> tuple[str, list[str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    applied, skipped = [], []
    changed = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        original = source
        for anchor, replacement, label in EDITS:
            if replacement in source:
                continue
            if anchor in source:
                source = source.replace(anchor, replacement, 1)
                applied.append(label)
        for variants, label in ALTERNATIVES:
            if any(replacement in source for _, replacement in variants):
                continue
            for anchor, replacement in variants:
                if anchor in source:
                    source = source.replace(anchor, replacement, 1)
                    applied.append(label)
                    break
        if source != original:
            changed = True
            cell["source"] = source.splitlines(keepends=True)
    if changed and apply:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    # Verify against the reassembled cell sources: a notebook stores source as a
    # list of lines, so a multi-line snippet never appears verbatim in the file.
    verified = json.loads(path.read_text(encoding="utf-8")) if apply else notebook
    joined = "\n".join("".join(cell["source"]) for cell in verified["cells"] if cell["cell_type"] == "code")
    for _, replacement, label in EDITS:
        if replacement not in joined:
            skipped.append(label)
    for variants, label in ALTERNATIVES:
        if not any(replacement in joined for _, replacement in variants):
            skipped.append(label)
    return ("patched" if changed else "up to date"), skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    notebooks = sorted(KAGGLE_ROOT.glob("*/method/spectra_siam.ipynb"))
    notebooks.append(PROJECT_ROOT / "research" / "latent_graph_learning" / "notebooks" / "spectra_siam_kaggle_template.ipynb")

    incomplete = 0
    for path in notebooks:
        if not path.is_file():
            continue
        status, missing = patch(path, apply=not args.check)
        label = path.relative_to(PROJECT_ROOT)
        if missing:
            incomplete += 1
            print(f"  {str(label):58s} {status}  MISSING: {', '.join(missing)}")
        else:
            print(f"  {str(label):58s} {status}")

    if incomplete:
        print(f"\n[!] {incomplete} notebook(s) do not expose every hook; the topology-only arm needs all of them.")
        return 1
    print("\n[+] STRIP_NODE_TYPES available in every method notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
