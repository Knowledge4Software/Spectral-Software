from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spectral_code.revision.embeddings import fit_probe, pair_vectors
from spectral_code.revision.training import SessionBudgetReached


class Clock:
    def __init__(self, allowed=10000):
        self.allowed = allowed

    def check(self):
        self.allowed -= 1
        if self.allowed < 0:
            raise SessionBudgetReached()


def test_frozen_probe_uses_train_scaling_resumes_and_keeps_test_opt_in(tmp_path):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(6, 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=-1, keepdims=True)
    np.save(embeddings / "embeddings.npy", matrix)
    (embeddings / "embedding_ids.json").write_text(json.dumps([str(i) for i in range(6)]))
    (embeddings / "embeddings.json").write_text(json.dumps({"complete": True, "identity": "frozen",
                                                          "model": "synthetic", "resolved_revision": "test"}))
    frame = pd.DataFrame([{"left_id": str(i % 6), "right_id": str((i + 1) % 6), "label": i % 2} for i in range(12)])
    frames = {"train": frame, "valid": frame.iloc[:6], "test": frame.iloc[6:]}
    a, b = tmp_path / "full", tmp_path / "resumed"
    full = fit_probe(frames, embeddings, a, clock=Clock())
    partial = fit_probe(frames, embeddings, b, clock=Clock(2))
    assert not partial["complete"]
    resumed = fit_probe(frames, embeddings, b, clock=Clock())
    assert full == resumed
    assert not (b / "final_test.json").exists()
    result = fit_probe(frames, embeddings, b, clock=Clock(), final_test=True)
    assert result["probe_test"]["Pairs"] == 6
    assert result["encoder_training"] == "frozen"
    assert result["cosine_test"]["AUPRC_AP"] is not None
    np.testing.assert_array_equal(pair_vectors(matrix, [0, 1], [2, 3]), pair_vectors(matrix, [2, 3], [0, 1]))
