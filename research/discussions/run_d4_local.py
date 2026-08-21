"""Rebuild the Section 6.4 latent-graph figure locally, without Kaggle.

The Kaggle notebook ``kaggle/d4/01_latent_graph_analysis.ipynb`` is
inference-only: it loads the finished CodeNet lexical checkpoint and runs a
forward pass over the validation fragments. Nothing is retrained, so it runs on
a CPU laptop given the two inputs it needs, both of which live outside the repo:

  * clean data   data/benchmarks/codenet.zip
  * checkpoint   outputs/kaggle/RQ2/codenet/CodeNet_spectra_siam_Lexical.zip

This script executes the notebook's own cells in order, so the model, the data
reader, and the analysis stay defined in exactly one place. Only the Kaggle
paths are redirected and the GPU guard is relaxed.
"""
from __future__ import annotations

import argparse
import json
import linecache
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = _ROOT / "kaggle/d4/01_latent_graph_analysis.ipynb"
# The dataset the RQ2 CodeNet checkpoint was trained on: its splits
# (69,779 / 14,959 / 14,955 graph-evaluable pairs) come from here, and the
# rest of the paper uses this same export. The smaller nonclone_12k subset
# is a different dataset and must not be used with this checkpoint.
CLEAN_DATA = _ROOT.parent / "data/benchmarks/codenet.zip"
CHECKPOINT_ZIP = (_ROOT.parent
                  / "outputs/kaggle/RQ2/codenet/CodeNet_spectra_siam_Lexical.zip")
CHECKPOINT_MEMBER = "spectra_siam_codenet-4l_input_only_lex_final.pt"


def stage_inputs(root: Path) -> Path:
    """Point the notebook at the real inputs instead of copying them.

    The clean-data export is a ~1.4 GB ZIP and is never duplicated: the
    notebook scans /kaggle/input recursively and unpacks a clean-data archive
    itself, so a hard link is enough. A hard link needs no administrator
    rights, unlike a symlink, and costs no disk space. Only the checkpoint (a
    few MB) is extracted, because it lives inside a different zip.
    """
    kaggle_input = root / "input"
    data = kaggle_input / "codenet-4l"
    data.mkdir(parents=True, exist_ok=True)

    target = data / CLEAN_DATA.name
    try:
        target.hardlink_to(CLEAN_DATA)
    except (OSError, NotImplementedError, AttributeError):
        # A hard link cannot cross volumes; copying is correct but slow.
        shutil.copy2(CLEAN_DATA, target)

    checkpoint_dir = kaggle_input / "spectra-siam-codenet"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CHECKPOINT_ZIP) as handle:
        with handle.open(CHECKPOINT_MEMBER) as source:
            (checkpoint_dir / CHECKPOINT_MEMBER).write_bytes(source.read())
    return kaggle_input


def code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell.get("cell_type") == "code"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=_ROOT / "kaggle/latex/d4/figures")
    parser.add_argument("--keep-workdir", action="store_true",
                        help="do not delete the staged inputs afterwards")
    args = parser.parse_args()

    for path, what in ((NOTEBOOK, "notebook"), (CLEAN_DATA, "clean data"),
                       (CHECKPOINT_ZIP, "checkpoint archive")):
        if not path.exists():
            raise SystemExit(f"missing {what}: {path}")

    root = Path(tempfile.mkdtemp(prefix="d4_local_"))
    try:
        kaggle_input = stage_inputs(root)
        work = root / "working"
        work.mkdir(parents=True, exist_ok=True)

        cells = code_cells()
        print(f"{len(cells)} code cells; staged inputs under {root}")

        # The notebook resolves everything from these two globals, and calls
        # display() for its inline tables.
        namespace: dict = {
            "__name__": "__main__",
            "KAGGLE_INPUT_OVERRIDE": kaggle_input,
            "WORK_DIR_OVERRIDE": work,
            "_WORK_DIR_OVERRIDE": work,
            "display": print,
        }

        started = time.perf_counter()
        for index, source in enumerate(cells):
            # Redirect the two Kaggle roots and relax the GPU requirement.
            source = source.replace('Path("/kaggle/input")',
                                    "KAGGLE_INPUT_OVERRIDE")
            source = source.replace('Path("/kaggle/working")',
                                    "WORK_DIR_OVERRIDE")
            source = source.replace('_Path("/kaggle/working")',
                                    "WORK_DIR_OVERRIDE")
            print(f"\n=== cell {index + 1}/{len(cells)} ===", flush=True)
            # One cell rebuilds the encoder's forward() with
            # inspect.getsource(). Jupyter serves that from its own cell
            # cache; under exec() there is no file to read, so register the
            # cell with linecache under the name we compile it as.
            name = f"{NOTEBOOK.name}#cell-{index}"
            linecache.cache[name] = (len(source), None,
                                     source.splitlines(keepends=True), name)
            exec(compile(source, name, "exec"), namespace)
        elapsed = time.perf_counter() - started

        args.output_dir.mkdir(parents=True, exist_ok=True)
        produced = []
        for name in ("d4_latent_graph_analysis.pdf", "d4_latent_graph_analysis.png",
                     "d4_triplet_eigenvalues.csv"):
            source_path = work / name
            if source_path.is_file():
                destination = args.output_dir / name.replace(
                    "d4_latent_graph_analysis", "latent_graph_discussion")
                shutil.copy2(source_path, destination)
                produced.append(destination)

        print(f"\nfinished in {elapsed/60:.1f} min")
        for path in produced:
            print(f"wrote {path}")
        if not produced:
            print(f"no outputs found under {work}")
    finally:
        if args.keep_workdir:
            print(f"staged inputs kept at {root}")
        else:
            # Only hard links and extracted copies live here, so removing
            # the staging tree never touches the real export.
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
