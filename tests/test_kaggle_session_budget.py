"""Kaggle revision jobs are standalone and resume below the 12-hour cap.

The old feature-ablation notebook package was retired. These checks cover the
maintained generator and the actual resumable session contract instead.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.build_revision_notebooks import build
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.training import ARMS
from scripts.revision_notebook_extras import EXTERNAL_METHODS, REFERENCE_METHODS


def test_one_arm_and_seed_per_notebook_with_reserved_session_time(tmp_path):
    paths = build(tmp_path)
    assert len(paths) == len(ARMS) * 3 + 4 + len(EXTERNAL_METHODS) * 6 + (2 * len(REFERENCE_METHODS) - 1) * 3 + 18 + 3
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        config = "".join(notebook["cells"][1]["source"])
        assignments = {node.targets[0].id: ast.literal_eval(node.value)
                       for node in ast.parse(config).body if isinstance(node, ast.Assign)
                       and isinstance(node.value, ast.Constant)}
        for i, notebook_cell in enumerate(notebook["cells"]):
            if notebook_cell["cell_type"] == "code":
                compile("".join(notebook_cell["source"]), f"{path}#{i}", "exec")
        if "05_analysis" in path.parts:
            if path.stem == "validation_screening":
                assert notebook["metadata"]["kaggle"]["accelerator"] == "none"
                assert "collect_validation" in "".join(notebook["cells"][3]["source"])
            continue
        if "frozen_qwen" in path.parts:
            assert assignments["PHASE"] == "encode"
            continue
        if "METHOD" in assignments:
            assert assignments["METHOD"] in EXTERNAL_METHODS
            assert assignments["SEED"] in (42, 123, 231)
            assert assignments["SESSION_HOURS"] == 10.0
            assert assignments["FINAL_TEST"] is False
            if assignments["TRANSFER_KEY"]:
                assert (assignments["MAX_EPOCHS"], assignments["MIN_EPOCHS"], assignments["PATIENCE"]) == (30, 30, 1000)
            else:
                assert assignments["MAX_EPOCHS"] == 100
            continue
        assert assignments["ARM"] in ARMS
        assert assignments["SEED"] in (42, 123, 231)
        assert assignments["SESSION_HOURS"] == 10.0
        assert assignments["FINAL_TEST"] is False
        if "04_transfer" in path.parts:
            assert (assignments["MAX_EPOCHS"], assignments["MIN_EPOCHS"], assignments["PATIENCE"]) == (30, 30, 1000)
        else:
            assert (assignments["MAX_EPOCHS"], assignments["MIN_EPOCHS"], assignments["PATIENCE"]) == (100, 10, 10)
        assert RunConfig().reserve_minutes == 45.0
        run = "".join(notebook["cells"][3]["source"])
        assert "max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS, patience=PATIENCE" in run
        assert "resume=Path(RESUME_CHECKPOINT)" in run
        assert 'if summary["complete"]:' in run
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path}#{i}", "exec")


def test_embedded_runtime_matches_tested_source_files(tmp_path):
    notebook = json.loads(build(tmp_path)[0].read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][2]["source"]))
    node = next(node for node in tree.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "_sources")
    sources = ast.literal_eval(node.value)
    root = Path(__file__).resolve().parents[1] / "spectral_code/revision"
    assert sources == {p.name: p.read_text(encoding="utf-8") for p in root.glob("*.py")}
