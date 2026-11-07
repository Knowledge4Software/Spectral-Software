"""Paths for the revision workspace, independent of the scientific runtime.

The generator still describes notebook families internally. This mapping places
each notebook in the execution stage where a person should look for it.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REVISION_ROOT = ROOT / "revision"
QUEUE_DIR = "queues"
PLAN_DIR = "00_plan"


def notebook_path(relative: str | Path) -> Path:
    """Map an internal notebook family to its single maintained stage path."""
    path = PurePosixPath(str(relative).replace("\\", "/"))
    group, *parts = path.parts
    name = path.name
    if group == "08_analysis":
        return Path("05_analysis", name)
    if group == "04_modern_baseline":
        return Path("optional", "frozen_qwen", name)
    if group == "03_transfer":
        return Path("04_transfer", "spectra", name)
    if group == "06_transfer_baselines":
        return Path("04_transfer", "baselines", name)

    seed = int(path.stem.rsplit("seed", 1)[1])
    if group in {"01_readout_controls", "02_loss_controls", "05_baselines"}:
        dataset = "codenet"
        is_baseline = group == "05_baselines" or name == f"codenet_full_seed{seed}.ipynb"
        kind = "baselines" if is_baseline else "architecture" if group == "01_readout_controls" else "losses"
    elif group == "07_cross_benchmark":
        dataset = parts[0]
        is_external = "baselines" in parts
        is_baseline = is_external or path.stem.startswith("full_seed")
        kind = "baselines" if is_baseline else "controls"
        if not is_external:
            name = f"{dataset}_{name}"
    else:
        raise ValueError(f"Unknown revision notebook family: {relative}")

    if seed != 42:
        return Path("03_selected_repeats", dataset, kind, name)
    if is_baseline:
        return Path("01_baselines_seed42", dataset, name)
    if dataset != "codenet":
        return Path("02_controls_seed42", "optional_cross_benchmark", dataset, name)
    return Path("02_controls_seed42", dataset, kind, name)
