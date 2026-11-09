import copy
import json
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from spectral_code.revision.numerics import (VALIDATION_RECOVERY_SOURCE, MAX_AMP_ATTEMPTS,
                                             NumericalStepFailure, optimizer_group)
from spectral_code.revision.protocol import RunConfig, fingerprint, run_identity
from spectral_code.revision.training import (LOSS_WEIGHTS, atomic_save, restore_rng, rng_state,
                                             train_session)


class TinyPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(.25)
        self.linear = nn.Linear(3, 1)

    def forward(self, left, right, unused):
        return self.linear(self.dropout(self.norm(left + right))).squeeze(-1), {}


class BadScaledGradient(torch.autograd.Function):
    threshold = float("inf")

    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("inf")) if gradient.abs().max() > BadScaledGradient.threshold else gradient


def loss_fn(logits, auxiliary, labels, **kwargs):
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return BadScaledGradient.apply(loss), {}


def group():
    return [(torch.tensor([[.1, .5, .9], [.4, .3, .2]]), torch.ones(2, 3) * i,
             torch.tensor([0., 1.])) for i in (1, 2)]


@pytest.fixture(autouse=True)
def reset_failure_threshold():
    torch.set_num_threads(1)
    BadScaledGradient.threshold = float("inf")
    yield
    BadScaledGradient.threshold = float("inf")


def run_group(model, optimizer, scaler, **kwargs):
    return optimizer_group(model, optimizer, scaler, group(), config=RunConfig(),
                           device=torch.device("cpu"), losses={}, loss_function=loss_fn,
                           positive_class_weight=1., **kwargs)


def test_amp_overflow_replays_dropout_and_buffers_without_skipping_an_update():
    torch.manual_seed(42)
    model = TinyPair().train(); reference = copy.deepcopy(model)
    a = torch.optim.AdamW(model.parameters(), lr=.001)
    b = torch.optim.AdamW(reference.parameters(), lr=.001)
    before = rng_state()
    BadScaledGradient.threshold = 6.  # weighted gradient: scale16 ->8 fails, scale8 ->4 succeeds
    contributions, events = run_group(model, a, torch.amp.GradScaler("cpu", init_scale=16.))
    after = rng_state()
    restore_rng(before)
    expected, no_events = run_group(reference, b, torch.amp.GradScaler("cpu", init_scale=8.))
    assert contributions == expected and not no_events
    assert len(events) == 1 and events[0]["next_loss_scale"] == 8.
    assert torch.equal(after["torch"], rng_state()["torch"])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    assert model.norm.num_batches_tracked.item() == 2
    assert all(value["step"].item() == 1 for value in a.state.values())


def test_finite_group_is_bitwise_identical_to_original_accumulation_sequence():
    torch.manual_seed(42)
    model = TinyPair().train(); original = copy.deepcopy(model)
    a = torch.optim.AdamW(model.parameters(), lr=.001)
    b = torch.optim.AdamW(original.parameters(), lr=.001)
    before = rng_state()
    contributions, events = run_group(model, a, torch.amp.GradScaler("cpu", enabled=False))
    after = rng_state()
    restore_rng(before)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    expected = []
    for left, right, labels in group():
        logits, auxiliary = original(left, right, None)
        loss, _ = loss_fn(logits, auxiliary, labels)
        scaler.scale(loss * len(labels) / 4).backward()
        expected.append((float(loss.detach()) * len(labels), len(labels)))
    scaler.unscale_(b)
    torch.nn.utils.clip_grad_norm_(original.parameters(), 1.)
    scaler.step(b); scaler.update(); b.zero_grad(set_to_none=True)
    assert contributions == expected and not events
    assert torch.equal(after["torch"], rng_state()["torch"])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)


def test_nonfinite_first_loss_can_backoff_before_any_backward_initializes_scaler():
    torch.manual_seed(42)
    model = TinyPair().train()
    optimizer = torch.optim.AdamW(model.parameters())
    calls = []
    def first_loss_nan(logits, auxiliary, labels, **kwargs):
        calls.append(None)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return (loss * float("nan") if len(calls) == 1 else loss), {}
    contributions, events = optimizer_group(model, optimizer, torch.amp.GradScaler("cpu", init_scale=16.),
        group(), config=RunConfig(), device=torch.device("cpu"), losses={},
        loss_function=first_loss_nan, positive_class_weight=1.)
    assert len(contributions) == 2 and len(calls) == 3
    assert events[0]["reason"] == "nonfinite_loss"
    assert model.norm.num_batches_tracked.item() == 2
    assert all(value["step"].item() == 1 for value in optimizer.state.values())


def test_persistent_scaled_overflow_gets_one_unscaled_fp32_attempt():
    torch.manual_seed(42)
    model = TinyPair().train(); reference = copy.deepcopy(model)
    a = torch.optim.AdamW(model.parameters(), lr=.001)
    b = torch.optim.AdamW(reference.parameters(), lr=.001)
    before = rng_state()
    BadScaledGradient.threshold = .6
    contributions, events = run_group(model, a, torch.amp.GradScaler("cpu", init_scale=16.))
    restore_rng(before)
    expected, no_events = run_group(reference, b, torch.amp.GradScaler("cpu", enabled=False))
    assert contributions == expected and not no_events
    assert len(events) == MAX_AMP_ATTEMPTS + 1
    assert events[-1]["action"] == "group_recovered"
    assert events[-1]["precision"] == "fp32_fallback"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    assert all(value["step"].item() == 1 for value in a.state.values())


def test_fp32_failure_restores_pre_group_state_and_never_steps_optimizer():
    torch.manual_seed(42)
    model = TinyPair().train()
    initial = copy.deepcopy(model.state_dict()); before = rng_state()
    optimizer = torch.optim.AdamW(model.parameters())
    BadScaledGradient.threshold = -1.
    with pytest.raises(NumericalStepFailure) as failure:
        run_group(model, optimizer, torch.amp.GradScaler("cpu", enabled=False))
    assert failure.value.diagnostic["events"][-1]["nonfinite_parameters"]
    assert torch.equal(before["torch"], rng_state()["torch"])
    assert not optimizer.state
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def _setup():
    from test_revision_resume import _data
    from spectral_code.revision.model import RevisionModelConfig
    frames, graphs = _data()
    config = RunConfig(max_epochs=2, min_epochs=2, patience=3, batch_size=2,
                       effective_batch_size=4, max_nodes=8, amp=False)
    model = RevisionModelConfig(hidden_dim=16, latent_nodes=4, max_input_nodes=8,
                                 spectral_bands=3, spectral_signals=2, density_bins=4,
                                 heat_samples=3, chebyshev_degree=3)
    return frames, graphs, config, model


def test_persistent_failure_saves_a_safe_cursor_and_diagnostic(tmp_path):
    frames, graphs, config, model = _setup()
    BadScaledGradient.threshold = -1.
    with pytest.raises(FloatingPointError, match="Safe pre-group checkpoint"):
        train_session(frames, graphs, config, model, {}, tmp_path, data_hash="data", loss_function=loss_fn)
    state = torch.load(tmp_path / "progress.pt", weights_only=False)
    assert state["batch"] == state["optimizer_steps"] == state["seen"] == 0
    assert not state["complete"] and state["stop_reason"] == "nonfinite_after_fp32"
    assert not state["optimizer"]["state"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "failed_nonfinite"
    assert json.loads((tmp_path / "numerical_failure.json").read_text())["events"]
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    assert not (tmp_path / "final_test.json").exists()


def test_approved_upgrade_requires_opt_in_exact_settings_and_records_lineage(tmp_path):
    from test_revision_resume import Clock
    frames, graphs, config, model = _setup()
    partial = tmp_path / "partial"
    train_session(frames, graphs, config, model, LOSS_WEIGHTS, partial, data_hash="data",
                  clock=Clock(5), experiment_label="full")
    state = torch.load(partial / "progress.pt", weights_only=False)
    assert state["optimizer_steps"] > 0 and not state["complete"]
    legacy_hash = VALIDATION_RECOVERY_SOURCE
    state["implementation_hash"] = legacy_hash
    state["identity"] = fingerprint({"run": run_identity(config, asdict(model), LOSS_WEIGHTS, "data"),
                                      "implementation": legacy_hash, "arm": "full"})
    for name in ("numerical_policy", "numerical_events", "lineage"):
        state.pop(name, None)
    legacy = tmp_path / "legacy.pt"; atomic_save(state, legacy)
    with pytest.raises(ValueError, match="identity mismatch"):
        train_session(frames, graphs, config, model, LOSS_WEIGHTS, tmp_path / "no_opt_in",
                      data_hash="data", resume=legacy, experiment_label="full")
    with pytest.raises(ValueError, match="identity mismatch"):
        train_session(frames, graphs, replace(config, seed=123), model, LOSS_WEIGHTS, tmp_path / "wrong_seed",
                      data_hash="data", resume=legacy, experiment_label="full", allow_numerical_upgrade=True)
    migrated = tmp_path / "migrated"
    result = train_session(frames, graphs, config, model, LOSS_WEIGHTS, migrated, data_hash="data",
                           resume=legacy, experiment_label="full", allow_numerical_upgrade=True)
    full = tmp_path / "full"
    train_session(frames, graphs, config, model, LOSS_WEIGHTS, full, data_hash="data", experiment_label="full")
    a = torch.load(migrated / "progress.pt", weights_only=False)
    b = torch.load(full / "progress.pt", weights_only=False)
    assert a["history"] == b["history"]
    assert a["identity"] == b["identity"] != state["identity"]
    assert result["lineage"][0]["from_implementation_hash"] == legacy_hash
    for name, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
    for value in state["model"].values():
        if value.is_floating_point():
            value.view(-1)[0] = float("nan"); break
    invalid = tmp_path / "invalid.pt"; atomic_save(state, invalid)
    with pytest.raises(ValueError, match="non-finite weights"):
        train_session(frames, graphs, config, model, LOSS_WEIGHTS, tmp_path / "invalid",
                      data_hash="data", resume=invalid, experiment_label="full", allow_numerical_upgrade=True)
