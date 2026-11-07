"""Explicit validation policy, common metrics, and matched transfer budgets."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


@dataclass(frozen=True)
class RunConfig:
    dataset: str = "codenet"
    seed: int = 42
    selection_metric: str = "Accuracy"
    max_epochs: int = 100
    patience: int = 10
    min_epochs: int = 10
    batch_size: int = 32
    effective_batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_nodes: int = 256
    session_hours: float = 10.0
    reserve_minutes: float = 45.0
    checkpoint_minutes: float = 10.0
    amp: bool = True

    def __post_init__(self):
        if self.selection_metric not in {"Accuracy", "F1"}:
            raise ValueError("Selection metric must be explicit: Accuracy or F1")
        if not 0 < self.session_hours <= 11:
            raise ValueError("Session budget must be at most 11h, below Kaggle's 12h cap")
        if not 0 < self.reserve_minutes < self.session_hours * 60:
            raise ValueError("Reserve must fit inside the session budget")
        if min(self.batch_size, self.effective_batch_size, self.patience, self.max_epochs) <= 0:
            raise ValueError("Training sizes and limits must be positive")
        if self.effective_batch_size < self.batch_size or self.effective_batch_size % self.batch_size:
            raise ValueError("Effective batch must be an integer multiple of the physical batch")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def runtime_forecast(max_epochs: int, completed_epochs: int, elapsed_seconds: float,
                     remaining_seconds: float | None) -> dict:
    """Planning estimate only; never changes the scientific training schedule.

    Elapsed training/validation time includes work from previous sessions and
    retries. A 15% planning margin supplements the session's output reserve.
    """
    epochs_left = max(0, max_epochs - completed_epochs)
    result = {"completed_epochs": completed_epochs, "epochs_left_to_cap": epochs_left,
              "planning_margin": 0.15, "estimate_only": True}
    if completed_epochs <= 0 or elapsed_seconds <= 0:
        return {**result, "status": "awaiting_first_validated_epoch"}
    seconds_per_epoch = elapsed_seconds / completed_epochs
    result.update(status="estimated", seconds_per_epoch=seconds_per_epoch,
                  hours_to_epoch_cap=epochs_left * seconds_per_epoch / 3600)
    if remaining_seconds is not None:
        available = max(0.0, remaining_seconds) * 0.85
        capacity = int(available / seconds_per_epoch)
        result.update(estimated_more_epochs_this_session=capacity,
                      epoch_cap_likely_fits_this_session=epochs_left <= capacity)
    return result


def metrics(labels, scores, threshold: float) -> dict:
    y, p = np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Metrics require equally sized, finite, nonempty inputs")
    pred = p >= threshold
    two_classes = set(np.unique(y)) == {0, 1}
    return {
        "P": float(precision_score(y, pred, zero_division=0)),
        "R": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y, pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "MacroF1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "AUROC": float(roc_auc_score(y, p)) if two_classes else None,
        "AUPRC_AP": float(average_precision_score(y, p)) if two_classes else None,
        "MCC": float(matthews_corrcoef(y, pred)), "Threshold": float(threshold),
        "Pairs": int(len(y)), "Positives": int(y.sum()),
        "TP": int(((y == 1) & pred).sum()), "FP": int(((y == 0) & pred).sum()),
        "TN": int(((y == 0) & ~pred).sum()), "FN": int(((y == 1) & ~pred).sum()),
    }


def choose_threshold(labels, scores, primary: str) -> tuple[float, dict]:
    """Exact validation threshold optimization; no labels from test are used."""
    if primary not in {"Accuracy", "F1"}:
        raise ValueError("Primary metric must be Accuracy or F1")
    y, p = np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(p).all():
        raise ValueError("Validation requires both classes and finite probabilities")
    precision, recall, thresholds = precision_recall_curve(y, p)
    # Include the all-negative operating point, missing from sklearn thresholds.
    candidates = np.r_[thresholds, np.nextafter(p.max(), np.inf)]
    tp = np.r_[recall[:-1] * y.sum(), 0.0]
    fp = np.r_[np.divide(tp[:-1], precision[:-1], out=np.zeros_like(tp[:-1]), where=precision[:-1] > 0) - tp[:-1], 0.0]
    acc = (tp + (len(y) - y.sum()) - fp) / len(y)
    f1 = np.divide(2 * tp, tp + fp + y.sum(), out=np.zeros_like(tp), where=(tp + fp + y.sum()) > 0)
    objective, tie = (acc, f1) if primary == "Accuracy" else (f1, acc)
    index = max(range(len(candidates)), key=lambda i: (round(objective[i], 12), round(tie[i], 12), candidates[i]))
    threshold = float(candidates[index])
    return threshold, metrics(y, p, threshold)


def matched_training_pairs(frame: pd.DataFrame, buckets: tuple[str, ...], total: int, seed: int) -> pd.DataFrame:
    """Same total and label budget per path, balanced within each language bucket.

    Fail if the exact budget is impossible; never quietly use a smaller sample.
    Target-language exposure still needs to be reported separately by the caller.
    """
    buckets = tuple(sorted(set(buckets)))
    if total <= 0 or total % (2 * len(buckets)):
        raise ValueError("Budget must divide equally into every configuration/label bucket")
    n = total // (2 * len(buckets))
    parts = []
    for bucket in buckets:
        for label in (0, 1):
            group = frame[frame.configuration_id.eq(bucket) & frame.label.eq(label)]
            if len(group) < n:
                raise ValueError(f"Insufficient {bucket}/{label}: need {n}, have {len(group)}")
            parts.append(group.sample(n, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def run_identity(config: RunConfig, model_config: dict, losses: dict, data_hash: str) -> str:
    # Session length/checkpoint cadence can change between resume sessions.
    settings = asdict(config)
    for key in ("session_hours", "reserve_minutes", "checkpoint_minutes"):
        settings.pop(key)
    return fingerprint({"training": settings, "model": model_config, "losses": losses, "data": data_hash})
