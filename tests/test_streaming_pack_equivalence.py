"""Streaming packers must reproduce the in-memory packers exactly.

CodeNet's endpoint count makes the original "decode every graph, then pack"
path exceed Kaggle host RAM.  The streaming replacements only change when
memory is released, so any divergence here would silently change published
baseline numbers rather than merely making a run fit.
"""
from __future__ import annotations

import ast as _ast
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "research" / "faithful_graph_baselines"
MAX_NODES, MAX_EDGES, MAX_STATEMENTS = 128, 256, 32
TREE_NAMES = (
    "node_types", "edge_parents", "edge_children", "depths",
    "binary_depths", "left_child", "right_sibling", "statements",
)


def _load_namespace() -> dict:
    """Execute only the NumPy-level helpers, so torch is not required."""
    namespace: dict = {
        "np": np,
        "json": json,
        "Path": Path,
        "MAX_NODES": MAX_NODES,
        "MAX_EDGES": MAX_EDGES,
        "MAX_STATEMENTS": MAX_STATEMENTS,
        "tqdm": lambda iterable, **_kwargs: iterable,
        "Sequence": list,
        "_open_text": lambda path: open(path, encoding="utf-8"),
    }
    core_wanted = {
        "children_from_edges", "postorder_depths", "left_child_right_sibling",
        "deckard_subtree_vectors", "deckard_pair_similarity",
    }
    runtime_wanted = {
        "_tree_arrays", "_statement_roots", "_adjacency", "_training_type_vocabulary",
        "_load_tree_arrays_streaming", "_load_deckard_vectors_streaming",
    }
    for filename, wanted in (("core.py", core_wanted), ("notebook_runtime.py", runtime_wanted)):
        tree = _ast.parse((_ROOT / filename).read_text(encoding="utf-8"))
        body = [
            node for node in tree.body
            if (isinstance(node, _ast.FunctionDef) and node.name in wanted)
            or (
                isinstance(node, _ast.Assign)
                and any(getattr(target, "id", "") == "_TREE_ARRAY_SPEC" for target in node.targets)
            )
        ]
        exec(compile(_ast.Module(body=body, type_ignores=[]), filename, "exec"), namespace)
    return namespace


NS = _load_namespace()


@pytest.fixture(scope="module")
def export(tmp_path_factory) -> tuple[Path, list[dict], list[str], set[str]]:
    """Write an export-shaped JSONL file with varied AST sizes."""
    rng = np.random.default_rng(0)
    codes = []
    for index in range(40):
        nodes = int(rng.integers(20, 200))
        rows, cols = [], []
        for child in range(1, nodes):
            rows.append(int(rng.integers(max(0, child - 6), child)))
            cols.append(child)
        codes.append({
            "code_id": f"c{index}",
            "graphs": {"ast": {"adjacency": {
                "num_nodes": nodes,
                "num_edges": len(rows),
                "node_ids": [f"n{position}" for position in range(nodes)],
                "node_types": [f"T{int(rng.integers(0, 12))}" for _ in range(nodes)],
                "row": rows,
                "col": cols,
            }}},
        })
    path = tmp_path_factory.mktemp("export") / "graph_spectra.jsonl"
    path.write_text("\n".join(json.dumps(code) for code in codes), encoding="utf-8")
    code_ids = [code["code_id"] for code in codes]
    return path, codes, code_ids, set(code_ids[:25])


def _decoded(codes: list[dict]) -> dict:
    return {
        code["code_id"]: {name: NS["_adjacency"](value) for name, value in code["graphs"].items()}
        for code in codes
    }


def test_streaming_tree_pack_matches_in_memory_pack(export):
    path, codes, code_ids, train_ids = export
    raw = _decoded(codes)

    vocabulary = NS["_training_type_vocabulary"](raw, train_ids, "ast")
    packed = [NS["_tree_arrays"](raw[code_id]["ast"], vocabulary)[:8] for code_id in code_ids]
    expected = {name: np.stack([row[index] for row in packed]) for index, name in enumerate(TREE_NAMES)}

    arrays, streamed_vocabulary, usable, _id_to_row = NS["_load_tree_arrays_streaming"](
        path, code_ids, train_ids,
    )

    assert streamed_vocabulary == vocabulary
    assert usable == set(code_ids)
    for name in TREE_NAMES:
        assert np.array_equal(arrays[name], expected[name]), name


def test_streaming_deckard_vectors_match_in_memory_vectors(export):
    path, codes, code_ids, train_ids = export
    raw = _decoded(codes)

    vocabulary = NS["_training_type_vocabulary"](raw, train_ids, "ast")
    expected = {}
    for code_id in code_ids:
        types, *_rest, children = NS["_tree_arrays"](raw[code_id]["ast"], vocabulary)
        nodes = int((types != 0).sum())
        subtree_vectors, sizes = NS["deckard_subtree_vectors"](
            types[:nodes], children, len(vocabulary), min_nodes=3,
        )
        order = np.argsort(sizes)[-32:]
        expected[code_id] = (
            subtree_vectors[order].astype(np.uint8), sizes[order].astype(np.uint8),
        )

    vectors, usable = NS["_load_deckard_vectors_streaming"](path, code_ids, train_ids)

    assert usable == set(code_ids)
    for code_id in code_ids:
        assert np.array_equal(vectors[code_id][0], expected[code_id][0]), code_id
        assert np.array_equal(vectors[code_id][1], expected[code_id][1]), code_id


def test_streaming_deckard_scores_are_unchanged(export):
    path, codes, code_ids, train_ids = export
    raw = _decoded(codes)
    vocabulary = NS["_training_type_vocabulary"](raw, train_ids, "ast")

    def in_memory(code_id):
        types, *_rest, children = NS["_tree_arrays"](raw[code_id]["ast"], vocabulary)
        nodes = int((types != 0).sum())
        subtree_vectors, sizes = NS["deckard_subtree_vectors"](
            types[:nodes], children, len(vocabulary), min_nodes=3,
        )
        order = np.argsort(sizes)[-32:]
        return subtree_vectors[order].astype(np.uint8), sizes[order].astype(np.uint8)

    vectors, _usable = NS["_load_deckard_vectors_streaming"](path, code_ids, train_ids)
    left_ids, right_ids = code_ids[:20], code_ids[20:]

    before = [
        NS["deckard_pair_similarity"](*in_memory(left), *in_memory(right))
        for left, right in zip(left_ids, right_ids)
    ]
    after = [
        NS["deckard_pair_similarity"](*vectors[left], *vectors[right])
        for left, right in zip(left_ids, right_ids)
    ]
    assert np.allclose(before, after)
