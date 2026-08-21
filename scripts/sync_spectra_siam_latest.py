"""Synchronize every runnable SPECTRA-Siam notebook to the current method.

The source of truth is the CodeNet clone-vs-different-problem notebook.  This
script deliberately preserves a notebook's dataset protocol and declared input
ablation. It copies the complete model cell from the source notebook, then
standardizes the fixed-band graph-signal spectral readout and disables the
source-token residual. Cross-language experiments regain AST+DDG relations.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KAGGLE = ROOT / "kaggle"
REFERENCE = (
    KAGGLE
    / "experiments"
    / "06_codenet_nonclone_scopes"
    / "02_clone_vs_diff_problem.ipynb"
)
RQ1_NOTEBOOKS = ROOT / "research" / "rq1" / "notebooks"
TEMPLATE = ROOT / "research" / "latent_graph_learning" / "notebooks"


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def write_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def replace_assignment(value: str, name: str, rhs: str) -> tuple[str, int]:
    return re.subn(
        rf"^{re.escape(name)}\s*=\s*.*$",
        f"{name} = {rhs}",
        value,
        flags=re.MULTILINE,
    )


def _is_spectra(notebook: dict) -> bool:
    return any("class CanonicalSpectraSiam" in source(cell) for cell in notebook.get("cells", []))


def _reference_model_source() -> str:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    candidates = [
        source(cell)
        for cell in reference["cells"]
        if "class CanonicalSpectraConfig" in source(cell)
        and "class CanonicalSpectraEncoder" in source(cell)
        and "class CanonicalSpectraSiam" in source(cell)
        and "def canonical_spectra_loss" in source(cell)
    ]
    if len(candidates) != 1:
        raise RuntimeError("Could not locate exactly one complete reference model cell")
    return candidates[0]


def _update_notebook(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    notebook = json.loads(raw)
    if not _is_spectra(notebook):
        return False

    changed = False
    reference_model = _reference_model_source()
    model_cells = [
        cell
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "class CanonicalSpectraConfig" in source(cell)
        and "class CanonicalSpectraEncoder" in source(cell)
        and "class CanonicalSpectraSiam" in source(cell)
        and "def canonical_spectra_loss" in source(cell)
    ]
    if len(model_cells) != 1:
        raise RuntimeError(f"Expected one complete SPECTRA model cell in {path}, found {len(model_cells)}")
    if source(model_cells[0]) != reference_model:
        write_source(model_cells[0], reference_model)
        changed = True
    cross_language = "experiments/03_cross_language/" in path.as_posix()
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        value = source(cell)
        updated = value

        # These are literal runtime settings in the main method notebooks.
        updated, count = replace_assignment(
            updated, "READOUT_MODE", '"graph_signal_spectral"'
        )
        updated, count = replace_assignment(updated, "USE_SOURCE_LEXICAL", "False")

        # Feature-ablation notebooks set their runnable arm from this mapping.
        updated = updated.replace('readout_mode="hybrid"', 'readout_mode="graph_signal_spectral"')
        updated = updated.replace("readout_mode='hybrid'", "readout_mode='graph_signal_spectral'")
        updated = updated.replace("use_source_lexical=True", "use_source_lexical=False")
        # Feature-ablation arms are cumulative: canonical is topology plus
        # labels, while lexical additionally enables lexical node sketches.
        # It must never silently become a duplicate of the full lexical arm.
        updated = updated.replace(
            '("canonical", dict(strip_node_types=False, use_node_lexical=True, use_source_lexical=False, readout_mode="graph_signal_spectral"))',
            '("canonical", dict(strip_node_types=False, use_node_lexical=False, use_source_lexical=False, readout_mode="graph_signal_spectral"))',
        )

        # Cross-language has an old, later override to AST-only.  Its graph
        # cache includes projected DDG, so it must use the same AST+DDG input.
        if cross_language:
            updated = re.sub(
                r"^INPUT_RELATION_INDICES\s*=\s*\(0,\)\s*(?:#.*)?$",
                "INPUT_RELATION_INDICES = (0, 2)  # AST + DDG",
                updated,
                flags=re.MULTILINE,
            )

        if updated != value:
            write_source(cell, updated)
            changed = True

    if not changed:
        return False
    path.write_text(json.dumps(notebook, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    return True


def validate() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference_model = next(
        source(cell)
        for cell in reference["cells"]
        if "class CanonicalSpectraSiam" in source(cell)
    )
    required = (
        "self.block_sizes = (config.density_bins, config.heat_samples)",
        "torch.split(left_spectrum, self.block_sizes, dim=-1)",
        "return self.classifier(spectral_pair_features).squeeze(-1)",
        '"hybrid", "eigenvalue_only", "graph_signal_spectral"',
    )
    if any(fragment not in reference_model for fragment in required):
        raise RuntimeError("Reference notebook does not contain the required block-wise head")

    failures: list[str] = []
    for folder in (KAGGLE, RQ1_NOTEBOOKS, TEMPLATE):
        for path in sorted(folder.rglob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not _is_spectra(notebook):
                continue
            text = "\n".join(source(cell) for cell in notebook["cells"])
            if any(fragment not in text for fragment in required):
                failures.append(f"missing block comparator: {path.relative_to(ROOT)}")
            if 'READOUT_MODE = "hybrid"' in text or 'readout_mode="hybrid"' in text:
                failures.append(f"stale hybrid readout: {path.relative_to(ROOT)}")
            if 'USE_SOURCE_LEXICAL = True' in text or 'use_source_lexical=True' in text:
                failures.append(f"stale source lexical residual: {path.relative_to(ROOT)}")
            if "readout_mode must be 'hybrid' or 'eigenvalue_only'" in text:
                failures.append(f"stale encoder without graph-signal readout: {path.relative_to(ROOT)}")
            if "experiments/03_cross_language/" in path.as_posix() and "INPUT_RELATION_INDICES = (0,)" in text:
                failures.append(f"stale AST-only cross-language config: {path.relative_to(ROOT)}")
    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> None:
    changed = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(KAGGLE.rglob("*.ipynb")) + sorted(TEMPLATE.rglob("*.ipynb"))
        if _update_notebook(path)
    ]
    print(f"Synchronized {len(changed)} notebook(s)")
    for path in changed:
        print(path)
    # RQ1 export notebooks are generated from the two main-method notebooks;
    # regenerate them here so a later sync cannot leave stale exports behind.
    from research.rq1.build_notebooks import main as build_rq1_notebooks

    build_rq1_notebooks()
    validate()
    print("Validated all SPECTRA-Siam notebooks against the latest block-wise design.")


if __name__ == "__main__":
    main()
