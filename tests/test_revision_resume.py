from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from spectral_code.revision.data import CanonicalGraph
from spectral_code.revision.model import RevisionModelConfig
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.training import (LOSS_WEIGHTS, SessionBudgetReached, evaluate_final,
                                             train_session)


class Clock:
    def __init__(self, allowed=10000):
        self.allowed = allowed

    def check(self):
        self.allowed -= 1
        if self.allowed < 0:
            raise SessionBudgetReached()


def _data():
    graphs = {}
    for i in range(6):
        nodes = 3 + i % 3
        edges = np.array([(a, b) for j in range(nodes - 1) for a, b in ((j, j + 1), (j + 1, j))], dtype=np.int32)
        graphs[str(i)] = CanonicalGraph(str(i), "java", np.arange(nodes) + i + 1,
                                        np.full((nodes, 4), i + 1, dtype=np.int64),
                                        np.zeros(96, dtype=np.int64),
                                        (edges, edges.copy(), edges.copy()), {})
    rows = [{"left_id": str(i % 6), "right_id": str((i + 1) % 6), "label": i % 2,
             "pair_id": str(i), "configuration_id": "java"} for i in range(8)]
    frame = pd.DataFrame(rows)
    return {"train": frame, "valid": frame.iloc[:4], "test": frame.iloc[4:]}, graphs


@pytest.mark.parametrize("allowed", (2, 3))
def test_mid_epoch_and_pre_validation_resume_match_uninterrupted_training(tmp_path, allowed):
    torch.set_num_threads(1)
    frames, graphs = _data()
    config = RunConfig(max_epochs=2, min_epochs=2, patience=3, batch_size=2, effective_batch_size=4,
                       max_nodes=8, amp=False)
    model = RevisionModelConfig(hidden_dim=16, latent_nodes=4, max_input_nodes=8,
                                 spectral_bands=3, spectral_signals=2, density_bins=4,
                                 heat_samples=3, chebyshev_degree=3)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train_session(frames, graphs, config, model, LOSS_WEIGHTS, full, data_hash="same-data", clock=Clock())
    interrupted = train_session(frames, graphs, config, model, LOSS_WEIGHTS, resumed,
                                data_hash="same-data", clock=Clock(allowed))
    assert interrupted["status"] == "incomplete_session_budget"
    assert not interrupted["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_final(resumed / "progress.pt", frames, graphs, config, resumed, expected_data_hash="same-data")
    summary = train_session(frames, graphs, config, model, LOSS_WEIGHTS, resumed,
                            data_hash="same-data", clock=Clock())
    assert summary["complete"]
    assert summary["runtime_forecast"]["completed_epochs"] == 2
    assert summary["runtime_forecast"]["epochs_left_to_cap"] == 0
    a = torch.load(full / "progress.pt", weights_only=False)
    b = torch.load(resumed / "progress.pt", weights_only=False)
    assert a["optimizer_steps"] == b["optimizer_steps"] == 4
    assert a["history"] == b["history"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    assert not (resumed / "final_test.json").exists()
    pending = evaluate_final(resumed / "progress.pt", frames, graphs, config, resumed,
                             expected_data_hash="same-data", clock=Clock(0))
    assert pending["status"] == "final_test_pending_session_budget"
    assert not (resumed / "final_test.json").exists()
    with pytest.raises(ValueError, match="identity mismatch"):
        train_session(frames, graphs, replace(config, seed=123), model, LOSS_WEIGHTS, resumed,
                      data_hash="same-data", clock=Clock())
    result = evaluate_final(resumed / "progress.pt", frames, graphs, config, resumed,
                            expected_data_hash="same-data")
    assert result["test"]["Pairs"] == 4
    assert result["test"]["AUPRC_AP"] is not None
    assert evaluate_final(resumed / "progress.pt", frames, graphs, config, resumed,
                          expected_data_hash="same-data") == result
