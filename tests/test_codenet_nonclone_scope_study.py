"""The retained scope-study preparation contract.

Notebook-only checks for retired exploratory packages were removed. Current
controlled experiments and pair selection have executable tests in
 test_revision_protocol.py, test_revision_model.py and test_revision_resume.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_pipeline_runner_excludes_mutations_and_targets_12k():
    runner = (ROOT / "create_datasets_graphs/codenet_4l/nonclone_scope_study/run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert 'PAIR_KINDS = "clone,hard_nonclone,nonclone_diff_problem"' in runner
    assert 'GRAPH_TYPES = "ast,cfg,ddg,cpg"' in runner
    assert "SAMPLE_SIZE = 12_000" in runner
    assert "MIN_PROGRAM_LINES = 20" in runner
    assert "MAX_PROGRAM_LINES = 50" in runner
    assignment = runner.index("PAIR_KINDS =")
    assert "nonclone_mutation" not in runner[assignment:assignment + 100]
