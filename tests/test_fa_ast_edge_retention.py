from __future__ import annotations

import ast as _ast
from pathlib import Path

import numpy as np

_RUNTIME = Path(__file__).resolve().parents[1] / "spectral_code" / "faithful_graph_baselines" / "notebook_runtime.py"
_WANTED = {"_map_relation", "_flow_arrays", "_flow_universe"}

# notebook_runtime imports torch at module scope, but FA-AST packing is pure
# NumPy.  Loading only the packing functions keeps this regression test
# runnable wherever the notebooks are generated.
_TREE = _ast.parse(_RUNTIME.read_text(encoding="utf-8"))
_MODULE = _ast.Module(
    body=[node for node in _TREE.body if isinstance(node, _ast.FunctionDef) and node.name in _WANTED],
    type_ignores=[],
)
MAX_NODES, MAX_EDGES = 128, 256
_NAMESPACE: dict = {
    "np": np,
    "MAX_NODES": MAX_NODES,
    "MAX_EDGES": MAX_EDGES,
    "RELATIONS": tuple("abcdef"),
}
exec(compile(_MODULE, str(_RUNTIME), "exec"), _NAMESPACE)
_flow_arrays = _NAMESPACE["_flow_arrays"]

VOCAB = {"<pad>": 0, "<unk>": 1, "T": 2}


def _program_graph(total_nodes: int, *, seed: int = 0) -> dict:
    """Build an export-shaped graph with program-like locality.

    Every layer orders its nodes independently, exactly as the clean-data
    exporter writes them.
    """
    rng = np.random.default_rng(seed)
    ids = [f"n{index}" for index in range(total_nodes)]

    ast_rows, ast_cols = [], []
    for child in range(1, total_nodes):
        ast_rows.append(int(rng.integers(max(0, child - 8), child)))
        ast_cols.append(child)
    cfg_rows, cfg_cols = list(range(total_nodes - 1)), list(range(1, total_nodes))
    ddg_rows, ddg_cols = [], []
    for node in range(total_nodes):
        for other in rng.integers(max(0, node - 20), min(total_nodes, node + 20), 2):
            if node != int(other):
                ddg_rows.append(node)
                ddg_cols.append(int(other))

    graphs = {}
    for name, (rows, cols) in {
        "ast": (ast_rows, ast_cols),
        "cfg": (cfg_rows, cfg_cols),
        "ddg": (ddg_rows, ddg_cols),
    }.items():
        order = list(rng.permutation(total_nodes))
        remap = {node: index for index, node in enumerate(order)}
        graphs[name] = {
            "node_ids": [ids[index] for index in order],
            "node_types": ["T"] * total_nodes,
            "row": [remap[value] for value in rows],
            "col": [remap[value] for value in cols],
        }
    order = list(rng.permutation(total_nodes))
    graphs["cpg"] = {
        "node_ids": [ids[index] for index in order],
        "node_types": ["T"] * total_nodes,
        "row": [],
        "col": [],
    }
    return graphs


def test_every_relation_keeps_edges_on_graphs_larger_than_the_window():
    """Graphs far larger than MAX_NODES must still carry message-passing edges.

    Retaining an arbitrary node prefix leaves almost every edge with an
    endpoint outside the window.  The packed graph becomes edgeless, GGNN/GMN
    message passing degenerates to a constant, and training stalls at
    ln(2) loss with 0.5 ROC-AUC instead of learning anything.
    """
    for total_nodes in (300, 600, 1200, 2000):
        _types, sources, targets = _flow_arrays(_program_graph(total_nodes), VOCAB)
        valid = (sources >= 0) & (targets >= 0)
        per_relation = valid.sum(axis=1)
        assert per_relation.min() > 0, (
            f"{total_nodes}-node graph lost every edge of at least one relation: {per_relation.tolist()}"
        )
        assert valid.sum() >= 500, (
            f"{total_nodes}-node graph kept only {int(valid.sum())} edges"
        )


def test_packed_edge_indices_stay_inside_the_node_window():
    for total_nodes in (128, 2000):
        _types, sources, targets = _flow_arrays(_program_graph(total_nodes), VOCAB)
        assert int(sources.max()) < MAX_NODES
        assert int(targets.max()) < MAX_NODES


def test_selection_is_deterministic():
    first = _flow_arrays(_program_graph(1500), VOCAB)
    second = _flow_arrays(_program_graph(1500), VOCAB)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
