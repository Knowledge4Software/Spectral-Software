import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from spectral_code.revision.screening import collect_validation
from spectral_code.revision.statistics import _family
from spectral_code.revision.transfer import FOCUSED_QUOTAS, select_transfer_frames
from spectral_code.revision.transfer_matrix import bucket
from spectral_code.revision.versioning import TRAINING_SOURCES, training_implementation_hash


def test_reporting_updates_do_not_change_training_but_model_changes_do(tmp_path):
    source = Path(__file__).resolve().parents[1] / "spectral_code/revision"
    for name in TRAINING_SOURCES:
        shutil.copyfile(source / name, tmp_path / name)
    initial = training_implementation_hash(tmp_path)
    (tmp_path / "screening.py").write_text("# reporting update")
    assert training_implementation_hash(tmp_path) == initial
    with (tmp_path / "model.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# model update\n")
    assert training_implementation_hash(tmp_path) != initial


@pytest.fixture(scope="module")
def transfer_data():
    result = {}
    buckets = {b for train, valid in FOCUSED_QUOTAS.values() for b in (*train, *valid)}
    for split, per_class in (("train", 3100), ("valid", 650), ("test", 4)):
        result[split] = pd.DataFrame([
            {"pair_id": f"{split}-{b}-{label}-{i}", "left_id": f"{split}-l-{b}-{label}-{i}",
             "right_id": f"{split}-r-{b}-{label}-{i}", "label": label,
             "configuration_id": bucket(*b.split("-"))}
            for b in sorted(buckets) for label in (0, 1) for i in range(per_class)])
    return result


def test_focused_paths_match_both_budgets_and_target_exposure(transfer_data):
    tests = []
    for key in FOCUSED_QUOTAS:
        selected, metadata = select_transfer_frames(transfer_data, key, 6000, 42)
        assert len(selected["train"]) == 6000
        assert len(selected["valid"]) == 1200
        assert selected["train"].label.sum() == 3000
        assert selected["valid"].label.sum() == 600
        assert metadata["selected_pairs_hash"]
        tests.append(selected["test"].pair_id.tolist())
        if key == "focused|source_only":
            assert set(selected["train"].configuration_id) == {"java"}
            assert metadata["target_containing_valid_pairs"] == 0
        elif key != "focused|target_reference":
            assert metadata["target_containing_train_pairs"] == 1500
            assert metadata["target_containing_valid_pairs"] == 300
        for split, quota_key in (("train", "train_quotas"), ("valid", "valid_quotas")):
            assert selected[split].configuration_id.value_counts().to_dict() == {
                bucket(*b.split("-")): n for b, n in metadata[quota_key].items()}
    assert all(test == tests[0] for test in tests)


def test_focused_sampling_identity_and_fixed_budget(transfer_data):
    a, ma = select_transfer_frames(transfer_data, "focused|two_hop", 6000, 42)
    b, mb = select_transfer_frames(transfer_data, "focused|two_hop", 6000, 42)
    c, mc = select_transfer_frames(transfer_data, "focused|two_hop", 6000, 123)
    pd.testing.assert_frame_equal(a["train"], b["train"])
    assert ma["selected_pairs_hash"] == mb["selected_pairs_hash"] != mc["selected_pairs_hash"]
    with pytest.raises(ValueError, match="exactly 6000"):
        select_transfer_frames(transfer_data, "focused|direct", 12000, 42)
    tiny = {k: v.head(2) for k, v in transfer_data.items()}
    with pytest.raises(ValueError, match="Insufficient"):
        select_transfer_frames(tiny, "focused|direct", 6000, 42)


@pytest.mark.parametrize("label,expected", [
    ("transfer/full/focused|direct", "full"),
    ("transfer/mean_max/focused|source_only", "mean_max"),
    ("transfer/attention/table8|J->P", "attention"),
    ("transfer/table8|J->P", "full"),
    ("baseline/codebert/focused|direct", "baseline/codebert"),
])
def test_transfer_analysis_preserves_readout_arm(label, expected):
    assert _family(label) == expected


def _status(root, name, *, identity="run1", arm="full", complete=False, epochs=1):
    path = root / name
    path.mkdir()
    state = {"identity": identity, "arm": arm, "implementation_hash": "code", "data_hash": "support",
             "training": {"dataset": "codenet", "seed": 42, "selection_metric": "Accuracy"},
             "history": [{"epoch": i} for i in range(1, epochs + 1)], "optimizer_steps": epochs * 10,
             "complete": complete, "status": "complete" if complete else "incomplete_session_budget",
             "best_epoch": 1, "best_valid": {"Accuracy": .8, "F1": .7}}
    (path / "status.json").write_text(json.dumps(state))
    # Deliberately unreadable inputs: validation collection must never open them.
    (path / "final_test.json").write_text("invalid json, do not read")
    (path / "progress.pt").write_bytes(b"not a checkpoint")
    return state


def test_validation_collector_deduplicates_resume_and_excludes_incomplete_ranks(tmp_path):
    source = tmp_path / "input"; source.mkdir()
    _status(source, "old")
    _status(source, "new", complete=True, epochs=2)
    _status(source, "unfinished", identity="run2", arm="baseline/astnn")
    queue = tmp_path / "queue.csv"
    pd.DataFrame([{"dataset": "codenet", "arm": arm, "seed": 42}
                  for arm in ("full", "baseline/astnn", "baseline/codebert")]).to_csv(queue, index=False)
    frame = collect_validation(source, tmp_path / "out", queue)
    assert len(frame) == 2
    assert frame.set_index("arm").loc["full", "validated_epochs"] == 2
    assert frame.set_index("arm").loc["full", "validation_rank"] == 1
    assert pd.isna(frame.set_index("arm").loc["baseline/astnn", "validation_rank"])
    ledger = pd.read_csv(tmp_path / "out/run_ledger.csv")
    assert ledger.set_index("arm").loc["baseline/codebert", "status"] == "not_started"


def test_validation_collector_refuses_competing_configurations(tmp_path):
    source = tmp_path / "input"; source.mkdir()
    _status(source, "candidate1", identity="one")
    _status(source, "candidate2", identity="two")
    with pytest.raises(ValueError, match="Multiple configurations"):
        collect_validation(source, tmp_path / "out")


def test_input_fix_reuses_only_unchanged_external_baselines(tmp_path):
    from spectral_code.revision.screening import INPUT_V1_HASH, INPUT_V2_HASH
    source = tmp_path / 'input'; source.mkdir()
    states = [('fixed', 'full', INPUT_V2_HASH), ('baseline', 'baseline/astnn', INPUT_V1_HASH)]
    for name,arm,implementation in states:
        state = _status(source,name,identity=name,arm=arm,complete=True)
        state['implementation_hash'] = implementation
        (source/name/'status.json').write_text(json.dumps(state))
    frame = collect_validation(source,tmp_path/'out')
    assert frame.implementation_hash.nunique() == 2
    assert frame.comparison_implementation_hash.nunique() == 1
    state = json.loads((source/'baseline/status.json').read_text())
    state['arm'] = 'mean_max'
    (source/'baseline/status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='one common support and training implementation'):
        collect_validation(source,tmp_path/'out_bad')
    state['arm'] = 'baseline/astnn'; state['data_hash'] = 'different_support'
    (source/'baseline/status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='one common support and training implementation'):
        collect_validation(source,tmp_path/'out_bad_data')


def test_empty_validation_ledger_is_still_useful(tmp_path):
    queue = tmp_path / "queue.csv"
    pd.DataFrame([{"dataset": "codenet", "arm": "full", "seed": 42}]).to_csv(queue, index=False)
    assert collect_validation(tmp_path / "empty", tmp_path / "out", queue).empty
    assert pd.read_csv(tmp_path / "out/run_ledger.csv").iloc[0].status == "not_started"
