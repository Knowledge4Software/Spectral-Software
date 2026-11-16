import pytest
import torch

from spectral_code.revision import learning_probe as probe
from spectral_code.revision.model import RevisionModelConfig, RevisionPairModel
from spectral_code.revision.training import SessionBudgetReached
from spectral_code.revision.versioning import training_implementation_hash
from spectral_code.revision.screening import VALIDATION_RECOVERY_HASH


def test_deadline_is_strict_and_never_extended(monkeypatch):
    monkeypatch.setattr(probe.time, "monotonic", lambda: 9.0)
    clock = probe.ProbeClock(10, started=0)
    clock.check()
    monkeypatch.setattr(probe.time, "monotonic", lambda: 10.0)
    with pytest.raises(SessionBudgetReached):clock.check()
    assert clock.deadline == 10


def test_identical_seed_has_identical_initial_weights_and_different_seed_differs():
    torch.set_num_threads(1)
    config = RevisionModelConfig(hidden_dim=16, latent_nodes=4)
    torch.manual_seed(123); a = probe.initial_weights_hash(RevisionPairModel(config))
    torch.manual_seed(123); b = probe.initial_weights_hash(RevisionPairModel(config))
    torch.manual_seed(231); c = probe.initial_weights_hash(RevisionPairModel(config))
    assert a == b and a != c
    assert training_implementation_hash() == VALIDATION_RECOVERY_HASH
