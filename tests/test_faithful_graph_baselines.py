import numpy as np
import pytest
import torch

from spectral_code.faithful_graph_baselines.core import (
    ASTNNEncoder,
    CDLHModel,
    DeepSimEncoder,
    FAASTGGNN,
    FAASTGMN,
    RtvNNEncoder,
    deckard_pair_similarity,
    deckard_subtree_vectors,
    left_child_right_sibling,
)


def _synthetic_graph_batch():
    batch, nodes, edges, relations = 2, 6, 8, 6
    node_types = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 3, 2, 4, 5, 0]])
    parents = torch.full((batch, edges), -1)
    children = torch.full((batch, edges), -1)
    parents[:, :3] = torch.tensor([[0, 0, 1], [0, 1, 1]])
    children[:, :3] = torch.tensor([[1, 2, 3], [1, 2, 3]])
    depths = torch.tensor([[2, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0]])
    statements = torch.tensor([[1, 2, -1], [1, 2, 3]])
    sources = torch.full((batch, relations, edges), -1)
    targets = torch.full_like(sources, -1)
    sources[:, :, :3] = parents[:, :3].unsqueeze(1)
    targets[:, :, :3] = children[:, :3].unsqueeze(1)
    return node_types, parents, children, depths, statements, sources, targets


def test_deckard_identical_subtrees_have_unit_similarity():
    children = [[1, 2], [3], [], []]
    vectors, sizes = deckard_subtree_vectors([1, 2, 3, 4], children, 8)
    assert deckard_pair_similarity(vectors, sizes, vectors, sizes) == pytest.approx(1.0)


def test_deckard_vectorized_subtree_reduction_matches_a_direct_reference():
    children = [[1, 2], [3, 4], [4], [], [], []]
    node_types, vocab_size = [1, 2, 3, 2, 4, 1], 8
    vectors, sizes = deckard_subtree_vectors(node_types, children, vocab_size)

    # Reference is the former scalar reduction, retained here to lock the
    # accelerated implementation to exactly the same characteristic vectors.
    from spectral_code.faithful_graph_baselines.core import postorder_depths
    levels = postorder_depths(children)
    counts = np.zeros((len(node_types), vocab_size + 1), dtype=np.float32)
    reference_sizes = np.ones(len(node_types), dtype=np.int32)
    for level in range(int(levels.max(initial=0)) + 1):
        for node in np.flatnonzero(levels == level):
            token = node_types[node]
            if 0 <= token < vocab_size:
                counts[node, token] += 1.0
            for child in children[node]:
                if 0 <= child < len(node_types) and levels[child] < levels[node]:
                    counts[node] += counts[child]
                    reference_sizes[node] += reference_sizes[child]
            counts[node, -1] = reference_sizes[node]
    keep = reference_sizes >= 3
    np.testing.assert_array_equal(vectors, counts[keep])
    np.testing.assert_array_equal(sizes, reference_sizes[keep])


def test_deckard_vectorized_pair_score_matches_per_left_subtree_reference():
    left = np.asarray([[1, 1, 3], [1, 2, 4]], dtype=np.float32)
    right = np.asarray([[1, 1, 3], [2, 0, 2], [1, 2, 4]], dtype=np.float32)
    left_sizes, right_sizes = np.asarray([3, 4]), np.asarray([3, 2, 4])
    expected = 0.0
    for index, size in enumerate(left_sizes):
        candidates = np.flatnonzero(np.abs(right_sizes - size) / max(size, 1) <= 0.2)
        if len(candidates):
            denom = np.linalg.norm(right[candidates], axis=1) * max(np.linalg.norm(left[index]), 1e-8)
            expected = max(expected, float(((right[candidates] @ left[index]) / np.maximum(denom, 1e-8)).max()))
    assert deckard_pair_similarity(left, left_sizes, right, right_sizes) == pytest.approx(expected)


def test_all_neural_backbones_forward_and_backward():
    node_types, parents, children, depths, statements, sources, targets = _synthetic_graph_batch()
    batch = node_types.size(0)

    outputs = [
        ASTNNEncoder(10)(node_types, parents, children, depths, statements),
        RtvNNEncoder(10)(node_types, parents, children, depths)[0],
        DeepSimEncoder(10)(node_types, torch.zeros(batch, node_types.size(1), 12), sources[:, 0], targets[:, 0]),
        FAASTGGNN(10)(node_types, sources, targets),
    ]

    tree = [[1, 2], [3], [], []]
    left, right = left_child_right_sibling(tree)
    left = torch.tensor(np.tile(np.r_[left, -1, -1], (batch, 1)))
    right = torch.tensor(np.tile(np.r_[right, -1, -1], (batch, 1)))
    outputs.append(CDLHModel(10).encode(node_types, left, right, depths))
    outputs.extend(FAASTGMN(10).forward_pair(node_types, sources, targets, node_types, sources, targets))

    loss = sum(output.square().mean() for output in outputs)
    loss.backward()
    assert all(torch.isfinite(output).all() for output in outputs)
