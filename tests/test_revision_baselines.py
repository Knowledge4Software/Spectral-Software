from __future__ import annotations

import gzip
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from spectral_code.revision.legacy import LegacyConfig, prepare_legacy
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.training import train_session, evaluate_final, SessionBudgetReached
from spectral_code.revision.transformer import TokenBatch, TokenCorpus, TransformerConfig, TransformerPairModel, task_contrastive


def _fixture(root):
    root.mkdir()
    adjacency = {"node_ids": ["0", "1", "2", "3"],
                 "node_types": ["Module", "FunctionDef", "Return", "Name"],
                 "row": [0, 1, 2], "col": [1, 2, 3]}
    codes = [{"code_id": str(i), "language": "python", "code": f"return {i}", "problem_id": str(i // 2)} for i in range(6)]
    with gzip.open(root / "codes.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in codes:
            stream.write(json.dumps(row) + "\n")
    with gzip.open(root / "graph_spectra.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in codes:
            stream.write(json.dumps({"code_id": row["code_id"], "graphs": {
                name: {"adjacency": adjacency, "eigenvalues": [0, .5, 1, 2]} for name in ("ast", "cfg", "ddg", "cpg")}}) + "\n")
    frame = pd.DataFrame([{"pair_id": str(i), "left_id": str(i % 6), "right_id": str((i + 1) % 6), "label": i % 2,
                           "split": "train", "configuration_id": "python"} for i in range(8)])
    frame.to_csv(root / "pairs.csv.gz", index=False)
    return {"train": frame, "valid": frame.iloc[:4], "test": frame.iloc[4:]}


@pytest.mark.parametrize("method", ("astnn", "rtvnn", "cdlh", "deepsim", "fa_ast_ggnn", "fa_ast_gmn"))
def test_real_baseline_adapters_train_and_release_symmetric_final_predictions(tmp_path, method):
    torch.set_num_threads(1)
    frames = _fixture(tmp_path / "data")
    config = RunConfig(max_epochs=1, min_epochs=1, batch_size=2, effective_batch_size=4, max_nodes=8, amp=False)
    output = tmp_path / method
    packed, settings, factory = prepare_legacy(tmp_path / "data", frames, method, config, output)
    summary = train_session(frames, packed, config, settings, {}, output, data_hash="fixture", model_factory=factory)
    assert summary["complete"]
    result = evaluate_final(output / "progress.pt", frames, packed, config, output, expected_data_hash="fixture",
                            model_factory=factory, model_config_factory=LegacyConfig)
    assert result["test"]["Pairs"] == 4
    model = factory(settings).eval()
    a, b, _ = next(iter(packed.pair_loader(frames["valid"], config)))
    with torch.no_grad():
        torch.testing.assert_close(model(a, b)[0], model(b, a)[0], atol=1e-5, rtol=1e-5)
    if method == "astnn":
        from spectral_code.revision.benchmark import benchmark_model
        weights = {name: p.detach().clone() for name, p in model.named_parameters()}
        timing = benchmark_model(model, frames["train"], packed, config, output / "benchmark.json", warmup=0, batches=2)
        assert timing["status"] == "complete"
        assert timing["inference"]["pairs"] == timing["forward_backward"]["pairs"] == 4
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.mark.parametrize("objective", ("pair_bce", "c4_adapted", "pair_bce_contrastive"))
def test_transformer_objectives_have_finite_gradients_and_symmetric_scores(objective):
    torch.set_num_threads(1)
    model = TransformerPairModel(TransformerConfig(objective=objective, pooling="last_token"), encoder=TinyEncoder()).eval()
    a = TokenBatch({"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), "attention_mask": torch.ones(2, 3)}, torch.tensor([0, 1]))
    b = TokenBatch({"input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]), "attention_mask": torch.ones(2, 3)}, torch.tensor([0, 1]))
    scores, aux = model(a, b)
    torch.testing.assert_close(scores, model(b, a)[0])
    loss, _ = model.compute_loss(scores, aux, torch.tensor([1., 0.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.embedding.weight.grad.abs().sum() > 0


def test_task_contrastive_handles_same_task_batches_without_false_negatives():
    vectors = torch.randn(4, 8, requires_grad=True)
    assert task_contrastive(vectors, torch.zeros(4, dtype=torch.long)).item() == 0
    loss = task_contrastive(vectors, torch.tensor([0, 0, 1, 1]))
    assert loss > 0
    loss.backward()
    assert torch.isfinite(vectors.grad).all()


def test_transformer_adapter_resumes_the_same_optimizer_trajectory(tmp_path):
    torch.set_num_threads(1)
    frames = _fixture(tmp_path / "data")
    tokens = np.arange(18).reshape(6, 3) + 1
    packed = TokenCorpus(tokens, np.ones_like(tokens), [str(i) for i in range(6)], [0, 0, 1, 1, 2, 2])
    config = RunConfig(max_epochs=2, min_epochs=2, batch_size=2, effective_batch_size=4, amp=False)
    settings = TransformerConfig(pooling="last_token")
    factory = lambda value: TransformerPairModel(value, encoder=TinyEncoder())
    class Clock:
        def __init__(self, count=10000):
            self.count = count
        def check(self):
            self.count -= 1
            if self.count < 0:
                raise SessionBudgetReached()
    for name, count in (("full", 10000), ("resumed", 2)):
        train_session(frames, packed, config, settings, {}, tmp_path / name, data_hash="tokens",
                      model_factory=factory, clock=Clock(count))
    summary = train_session(frames, packed, config, settings, {}, tmp_path / "resumed", data_hash="tokens",
                            model_factory=factory, clock=Clock())
    assert summary["complete"]
    a = torch.load(tmp_path / "full/progress.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/progress.pt", weights_only=False)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
