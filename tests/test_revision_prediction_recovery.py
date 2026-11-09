"""Invalid AMP scores must never reach threshold selection or discard pairs."""
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from spectral_code.revision import training
from spectral_code.revision.protocol import RunConfig


def test_finite_amp_prediction_path_is_unchanged(monkeypatch):
    expected = np.array([.1, .9])
    run = Mock(return_value=expected)
    monkeypatch.setattr(training, "_prediction_pass", run)
    result = training.predict(Mock(), [0, 1], {}, RunConfig(), torch.device("cuda"))
    assert result is expected
    assert run.call_count == 1 and run.call_args.kwargs == {"amp": True}


def test_bad_amp_retries_whole_pass_and_records_precision(monkeypatch):
    expected = np.array([.2, .8])
    run = Mock(side_effect=[np.array([np.nan, .9]), expected])
    monkeypatch.setattr(training, "_prediction_pass", run)
    events = []
    result = training.predict(Mock(), [0, 1], {}, RunConfig(), torch.device("cuda"), on_event=events.append)
    assert result is expected and run.call_count == 2
    assert [c.kwargs["amp"] for c in run.call_args_list] == [True, False]
    assert all(c.args[1] == [0, 1] for c in run.call_args_list)
    assert [e["action"] for e in events] == ["retry_entire_prediction_fp32", "prediction_recovered"]


@pytest.mark.parametrize("amp,calls", [(True, 2), (False, 1)])
def test_persistent_invalid_scores_fail_without_masking(monkeypatch, amp, calls):
    run = Mock(return_value=np.array([np.nan, np.inf]))
    monkeypatch.setattr(training, "_prediction_pass", run)
    with pytest.raises(training.NumericalPredictionFailure, match="persisted in FP32"):
        training.predict(Mock(), [0, 1], {}, RunConfig(amp=amp), torch.device("cuda"))
    assert run.call_count == calls


def test_retry_preserves_session_deadline(monkeypatch):
    run = Mock(side_effect=[np.array([np.nan]), training.SessionBudgetReached()])
    monkeypatch.setattr(training, "_prediction_pass", run)
    with pytest.raises(training.SessionBudgetReached):
        training.predict(Mock(), [0], {}, RunConfig(), torch.device("cuda"))


def test_validation_failure_checkpoint_resumes_without_repeating_optimizer_steps(tmp_path, monkeypatch):
    from test_revision_numerics import _setup
    frames, graphs, config, model = _setup()
    original = training.predict
    monkeypatch.setattr(training, "predict", Mock(side_effect=training.NumericalPredictionFailure("bad scores")))
    with pytest.raises(training.NumericalPredictionFailure):
        training.train_session(frames, graphs, config, model, training.LOSS_WEIGHTS, tmp_path / "partial", data_hash="data")
    checkpoint = tmp_path / "partial/progress.pt"
    before = torch.load(checkpoint, weights_only=False)
    assert before["epoch"] == 1 and before["batch"] == 4 and before["optimizer_steps"] == 2
    monkeypatch.setattr(training, "predict", original)
    recovered = training.train_session(frames, graphs, config, model, training.LOSS_WEIGHTS, tmp_path / "resumed",
                                       data_hash="data", resume=checkpoint)
    reference = training.train_session(frames, graphs, config, model, training.LOSS_WEIGHTS, tmp_path / "reference", data_hash="data")
    assert recovered["history"] == reference["history"]
    assert recovered["optimizer_steps"] == reference["optimizer_steps"] == 4
