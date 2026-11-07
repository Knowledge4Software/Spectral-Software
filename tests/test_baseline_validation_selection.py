from __future__ import annotations

import numpy as np

from spectral_code.faithful_graph_baselines.notebook_runtime import _selection_key, _validation_selection
from spectral_code.spectral_representation_baselines.runner import Metrics, selection_key, validation_selection


def _metrics(*, f1: float, accuracy: float, balanced_accuracy: float = 0.0) -> Metrics:
    return Metrics(0.0, 0.0, f1, accuracy, 0.0, balanced_accuracy, 0.5, 0, 0, 0, 0)


def test_balanced_validation_means_exactly_equal_clone_and_nonclone_counts():
    assert _validation_selection(np.array([0, 1, 0, 1]))["metric"] == "Accuracy"
    assert validation_selection(np.array([0, 1, 0, 1]))["metric"] == "Accuracy"
    assert _validation_selection(np.array([0, 0, 1]))["metric"] == "F1"
    assert validation_selection(np.array([0, 0, 1]))["metric"] == "F1"


def test_balanced_validation_prioritizes_accuracy_and_imbalanced_prioritizes_f1():
    higher_f1 = _metrics(f1=0.90, accuracy=0.70, balanced_accuracy=0.80)
    higher_acc = _metrics(f1=0.80, accuracy=0.90, balanced_accuracy=0.75)
    balanced = np.array([0, 1, 0, 1])
    imbalanced = np.array([0, 0, 1])

    assert _selection_key({"F1": higher_f1.f1, "Acc": higher_f1.accuracy, "BalancedAccuracy": higher_f1.balanced_accuracy}, balanced) < _selection_key({"F1": higher_acc.f1, "Acc": higher_acc.accuracy, "BalancedAccuracy": higher_acc.balanced_accuracy}, balanced)
    assert selection_key(higher_f1, balanced) < selection_key(higher_acc, balanced)
    assert _selection_key({"F1": higher_f1.f1, "Acc": higher_f1.accuracy, "BalancedAccuracy": higher_f1.balanced_accuracy}, imbalanced) > _selection_key({"F1": higher_acc.f1, "Acc": higher_acc.accuracy, "BalancedAccuracy": higher_acc.balanced_accuracy}, imbalanced)
    assert selection_key(higher_f1, imbalanced) > selection_key(higher_acc, imbalanced)


def test_explicit_accuracy_policy_survives_graph_coverage_imbalance(monkeypatch):
    from spectral_code.faithful_graph_baselines import notebook_runtime
    from spectral_code.spectral_representation_baselines import runner
    labels = np.array([1] * 101 + [0] * 100)
    monkeypatch.setattr(notebook_runtime, "VALIDATION_SELECTION_METRIC", "Accuracy", raising=False)
    monkeypatch.setattr(runner, "VALIDATION_SELECTION_METRIC", "Accuracy", raising=False)
    assert _validation_selection(labels)["metric"] == "Accuracy"
    assert validation_selection(labels)["metric"] == "Accuracy"
    assert not _validation_selection(labels)["balanced"]
    assert _selection_key({"Acc": .8, "F1": .7}, labels) == (.8, .7)
