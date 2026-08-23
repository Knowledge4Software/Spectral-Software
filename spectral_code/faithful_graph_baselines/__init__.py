"""Paper-faithful adaptations of clone-detection baselines.

The implementations retain this project's exported program graphs while
matching the defining representation and learning logic of each baseline.
They are adaptations, not byte-for-byte executions of upstream repositories.
"""

from .core import (
    ASTNNEncoder,
    BinaryTreeLSTMEncoder,
    CDLHModel,
    DeepSimEncoder,
    FAASTGGNN,
    FAASTGMN,
    RtvNNEncoder,
    deckard_pair_similarity,
    deckard_subtree_vectors,
    left_child_right_sibling,
)

__all__ = [
    "ASTNNEncoder",
    "BinaryTreeLSTMEncoder",
    "CDLHModel",
    "DeepSimEncoder",
    "FAASTGGNN",
    "FAASTGMN",
    "RtvNNEncoder",
    "deckard_pair_similarity",
    "deckard_subtree_vectors",
    "left_child_right_sibling",
]
