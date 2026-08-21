from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "kaggle/exploratory/04_feature_ablation"
VARIANTS = (
    "proposed_graph_signal",
    "topology_only",
    "canonical",
    "lexical",
)


def source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_combined_codexglue_ablation_uses_bounded_profile():
    path = ABLATION_DIR / "codexglue_v3.ipynb"
    assert 'RUN_PROFILE = "bounded_10h"' in source(path)


def test_full_codexglue_ablation_is_split_into_one_arm_per_notebook():
    for variant in VARIANTS:
        path = ABLATION_DIR / f"codexglue_v3_feature_ablation_{variant}.ipynb"
        notebook_source = source(path)
        assert 'RUN_PROFILE = "final_full"' in notebook_source, path
        assert f'ABLATION_RUN_TAG = "{variant}"' in notebook_source, path
        assert notebook_source.count("dict(strip_node_types=") == 1, path
        assert f'("{variant}",' in notebook_source, path
        assert "{ABLATION_RUN_TAG}_feature_ablation_results.csv" in notebook_source, path


def test_split_ablation_code_cells_compile():
    for variant in VARIANTS:
        path = ABLATION_DIR / f"codexglue_v3_feature_ablation_{variant}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{path}#cell-{index}", "exec")
