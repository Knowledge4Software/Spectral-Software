from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from spectral_code.config import PipelineConfig
from spectral_code.pipeline import Pipeline
from .bcb_dataset import ClonePair


def _pad_and_normalize(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(v1), len(v2))

    if len(v1) < max_len:
        v1 = np.pad(v1, (0, max_len - len(v1)))
    if len(v2) < max_len:
        v2 = np.pad(v2, (0, max_len - len(v2)))

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 > 0:
        v1 = v1 / n1
    if n2 > 0:
        v2 = v2 / n2

    return v1, v2


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1, v2 = _pad_and_normalize(v1, v2)
    return float(np.dot(v1, v2))


class SpectralCloneBaseline:
    """
    Uses the pipeline's eigenvalue spectrum as a feature vector.
    Similarity score = cosine similarity between normalized spectra.
    """
    def __init__(self, config: PipelineConfig, lang: str = "java", threshold: float = 0.90):
        self.config = config
        self.lang = lang
        self.threshold = threshold

        if config.graph_type == "cfg" and lang != "python":
            raise ValueError(
                "Current CFGGraphBuilder only supports Python. "
                "For the BCB Java data in this repo, use graph_type='ast' "
                "or add a Java CFG builder."
            )

        self.pipeline = Pipeline(config)
        self._cache: dict[int, np.ndarray] = {}

    def embed(self, code_id: int, code: str) -> np.ndarray:
        if code_id not in self._cache:
            result = self.pipeline.run(code, lang=self.lang)
            eigvals = np.asarray(result["eigenvalues"], dtype=float)
            self._cache[code_id] = np.sort(eigvals)
        return self._cache[code_id]

    def score_pair(self, pair: ClonePair) -> float:
        v1 = self.embed(pair.left_id, pair.left_code)
        v2 = self.embed(pair.right_id, pair.right_code)
        return cosine_similarity(v1, v2)

    def predict(self, pair: ClonePair) -> int:
        return int(self.score_pair(pair) >= self.threshold)

    def tune_threshold(
        self,
        pairs: Iterable[ClonePair],
        grid: Iterable[float] | None = None,
    ) -> tuple[float, dict[str, float]]:
        pairs = list(pairs)
        if grid is None:
            grid = np.linspace(0.6, 0.95, 15)

        best_threshold = self.threshold
        best_f1 = -1.0
        best_metrics: dict[str, float] = {}

        for t in grid:
            self.threshold = float(t)
            metrics = evaluate_binary(self, pairs)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = float(t)
                best_metrics = metrics

        self.threshold = best_threshold
        return best_threshold, best_metrics


def evaluate_binary(model: SpectralCloneBaseline, pairs: list[ClonePair]) -> dict[str, float]:
    tp = fp = tn = fn = 0

    for pair in pairs:
        pred = model.predict(pair)

        if pair.label == 1 and pred == 1:
            tp += 1
        elif pair.label == 0 and pred == 1:
            fp += 1
        elif pair.label == 0 and pred == 0:
            tn += 1
        elif pair.label == 1 and pred == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_by_type(model: SpectralCloneBaseline, sampled: dict[str, list[ClonePair]]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}

    for ctype, pairs in sampled.items():
        if not pairs:
            report[ctype] = {"n": 0, "correct": 0, "success_rate": 0.0}
            continue

        correct = sum(1 for p in pairs if model.predict(p) == 1)
        report[ctype] = {
            "n": len(pairs),
            "correct": correct,
            "success_rate": correct / len(pairs),
        }

    return report


def save_report(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
