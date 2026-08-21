from __future__ import annotations

import numpy as np

from research.faithful_graph_baselines.notebook_runtime import _selection_key, _validation_selection
from research.spectral_representation_baselines.runner import Metrics, selection_key, validation_selection


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
