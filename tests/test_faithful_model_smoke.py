from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from spectral_code.faithful_graph_baselines.core import CDLHModel
from spectral_code.faithful_graph_baselines.notebook_runtime import (
    MAX_EDGES,
    MAX_NODES,
    MAX_STATEMENTS,
    RELATIONS,
    _DepthBucketBatchSampler,
    _EndpointGroupedDepthBatchSampler,
    _load_fa_arrays_streaming,
    _PairModel,
    _adaptive_train_step,
)


def _tree_arrays(rows: int) -> dict[str, np.ndarray]:
    arrays = {
        "node_types": np.zeros((rows, MAX_NODES), dtype=np.int32),
        "edge_parents": np.full((rows, MAX_EDGES), -1, dtype=np.int16),
        "edge_children": np.full((rows, MAX_EDGES), -1, dtype=np.int16),
        "depths": np.zeros((rows, MAX_NODES), dtype=np.int16),
        "binary_depths": np.zeros((rows, MAX_NODES), dtype=np.int16),
        "left_child": np.full((rows, MAX_NODES), -1, dtype=np.int16),
        "right_sibling": np.full((rows, MAX_NODES), -1, dtype=np.int16),
        "statements": np.full((rows, MAX_STATEMENTS), -1, dtype=np.int16),
    }
    arrays["node_types"][:, :2] = (1, 2)
    arrays["statements"][:, 0] = 0
    return arrays


def _flow_arrays(rows: int) -> dict[str, np.ndarray]:
    arrays = {
        "node_types": np.zeros((rows, MAX_NODES), dtype=np.int32),
        "edge_src": np.full((rows, len(RELATIONS), MAX_EDGES), -1, dtype=np.int16),
        "edge_dst": np.full((rows, len(RELATIONS), MAX_EDGES), -1, dtype=np.int16),
    }
    arrays["node_types"][:, :2] = (1, 2)
    return arrays


@pytest.mark.parametrize("method", ("astnn", "rtvnn", "cdlh", "deepsim", "fa_ast_ggnn", "fa_ast_gmn"))
def test_each_trainable_faithful_model_completes_forward_and_backward(method: str):
    rows = 2
    if method in {"astnn", "rtvnn", "cdlh"}:
        arrays = _tree_arrays(rows)
    elif method == "deepsim":
        arrays = {
            "node_types": np.zeros((rows, MAX_NODES), dtype=np.int32),
            "numeric": np.zeros((rows, MAX_NODES, 12), dtype=np.float32),
            "edge_src": np.full((rows, MAX_EDGES), -1, dtype=np.int16),
            "edge_dst": np.full((rows, MAX_EDGES), -1, dtype=np.int16),
        }
        arrays["node_types"][:, :2] = (1, 2)
    else:
        arrays = _flow_arrays(rows)

    model = _PairModel(method, arrays, vocab_size=8)
    labels = torch.tensor([0.0, 1.0])
    logits, auxiliary, left, right = model(torch.tensor([0, 1]), torch.tensor([1, 0]))
    assert logits.shape == labels.shape
    assert torch.isfinite(logits).all()
    auxiliary = auxiliary.mean()
    loss = (
        CDLHModel.loss(left, right, labels)
        if method == "cdlh"
        else torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    )
    if method == "rtvnn":
        loss = loss + 0.05 * auxiliary
    loss.backward()
    assert torch.isfinite(loss.detach())


def test_adaptive_train_step_bisects_an_oom_batch_without_changing_logical_batch():
    class ArtificialOOMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, left, right):
            if len(left) > 1:
                raise torch.OutOfMemoryError("synthetic CUDA out of memory")
            logits = self.bias.expand(len(left))
            auxiliary = logits.new_zeros(())
            codes = logits.unsqueeze(-1)
            return logits, auxiliary, codes, codes

    model = ArtificialOOMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss, microbatch = _adaptive_train_step(
        model,
        "astnn",
        optimizer,
        scaler,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        torch.tensor([0.0, 1.0]),
        torch.device("cpu"),
        maximum_microbatch=2,
    )
    assert np.isfinite(loss)
    assert microbatch == 1


def test_cdlh_depth_bucketing_keeps_every_pair_and_reduces_batch_depth_spread():
    depths = np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.int16)
    sampler = _DepthBucketBatchSampler(depths, batch_size=2, seed=42)
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(len(depths)))
    assert all(len(batch) == 2 for batch in batches)
    # Each batch is drawn from a contiguous depth bucket, regardless of the
    # randomized order of those buckets.
    assert max(int(depths[batch].max() - depths[batch].min()) for batch in batches) <= 1


def test_cdlh_endpoint_grouping_keeps_every_pair_once():
    left = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    right = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    sampler = _EndpointGroupedDepthBatchSampler(
        left, right, np.arange(len(left), dtype=np.int16), batch_size=4, seed=42,
    )
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(len(left)))
    assert all(len(batch) == 4 for batch in batches)
    assert {len(set(left[batch])) for batch in batches} == {1}


def test_cdlh_duplicate_rows_are_encoded_once_without_changing_scores():
    arrays = _tree_arrays(rows=2)
    model = _PairModel("cdlh", arrays, vocab_size=8).eval()
    left_rows, right_rows = torch.tensor([0, 0]), torch.tensor([1, 1])
    with torch.no_grad():
        logits, _, left_hash, right_hash = model(left_rows, right_rows)
        expected_left, _ = model._single(left_rows)
        expected_right, _ = model._single(right_rows)
        expected_logits = 4.0 * (expected_left * expected_right).mean(-1)
    torch.testing.assert_close(left_hash, expected_left)
    torch.testing.assert_close(right_hash, expected_right)
    torch.testing.assert_close(logits, expected_logits)


def test_fa_ast_streaming_loader_packs_without_retaining_raw_graph_dicts(tmp_path):
    def layer(node_type: str):
        return {"adjacency": {"num_nodes": 1, "node_ids": ["n0"], "node_types": [node_type], "row": [], "col": []}}

    rows = [
        {"code_id": "a", "graphs": {name: layer("METHOD") for name in ("ast", "cfg", "ddg", "cpg")} },
        {"code_id": "b", "graphs": {name: layer("CALL") for name in ("ast", "cfg", "ddg", "cpg")} },
    ]
    path = tmp_path / "graphs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    arrays, vocab, usable, id_to_row = _load_fa_arrays_streaming(
        path, ["a", "b"], {"a"}, ("ast", "cfg", "ddg", "cpg"),
    )
    assert usable == {"a", "b"}
    assert set(id_to_row) == {"a", "b"}
    assert arrays["node_types"].shape[0] == 2
    assert arrays["node_types"][id_to_row["a"], 0] == vocab["METHOD"]


@pytest.mark.parametrize("method", ("fa_ast_ggnn", "fa_ast_gmn"))
def test_fa_ast_scatter_uses_matching_dtype_under_autocast(method: str):
    arrays = _flow_arrays(rows=2)
    # Exercise a real relation rather than only padded edge slots.
    arrays["edge_src"][:, 0, 0] = 0
    arrays["edge_dst"][:, 0, 0] = 1
    model = _PairModel(method, arrays, vocab_size=8)
    labels = torch.tensor([0.0, 1.0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, _, _, _ = model(torch.tensor([0, 1]), torch.tensor([1, 0]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss.detach())


@pytest.mark.parametrize("method", ("fa_ast_ggnn", "fa_ast_gmn"))
def test_fa_ast_padding_crop_preserves_the_padded_graph_result(method: str):
    arrays = _flow_arrays(rows=2)
    arrays["node_types"][0, :2] = (1, 2)
    arrays["node_types"][1, :4] = (1, 3, 2, 4)
    arrays["edge_src"][0, 0, :2] = (0, 1)
    arrays["edge_dst"][0, 0, :2] = (1, 0)
    arrays["edge_src"][1, 0, :4] = (0, 1, 2, 3)
    arrays["edge_dst"][1, 0, :4] = (1, 2, 3, 0)
    model = _PairModel(method, arrays, vocab_size=8).eval()
    left_rows, right_rows = torch.tensor([0]), torch.tensor([1])
    with torch.no_grad():
        _, _, cropped_left, cropped_right = model(left_rows, right_rows)
        if method == "fa_ast_gmn":
            full_left, full_right = model.encoder.forward_pair(
                model.node_types[left_rows].long(), model.edge_src[left_rows].long(), model.edge_dst[left_rows].long(),
                model.node_types[right_rows].long(), model.edge_src[right_rows].long(), model.edge_dst[right_rows].long(),
            )
        else:
            full_left, _ = model._single(left_rows)
            full_right, _ = model._single(right_rows)
    torch.testing.assert_close(cropped_left, full_left)
    torch.testing.assert_close(cropped_right, full_right)
