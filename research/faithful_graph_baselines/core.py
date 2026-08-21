"""Core models for the repository's paper-faithful baseline adaptations.

The code deliberately accepts tensors derived from our exported AST/CFG/DDG
graphs.  It reproduces the defining algorithmic ideas of the cited methods,
without claiming parser- or preprocessing-equivalence with upstream projects.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def children_from_edges(num_nodes: int, rows: Sequence[int], cols: Sequence[int]) -> list[list[int]]:
    """Return a cycle-safe ordered child list for an exported AST."""
    children = [[] for _ in range(num_nodes)]
    seen: set[tuple[int, int]] = set()
    for parent, child in zip(rows, cols):
        parent, child = int(parent), int(child)
        edge = (parent, child)
        if 0 <= parent < num_nodes and 0 <= child < num_nodes and parent != child and edge not in seen:
            children[parent].append(child)
            seen.add(edge)
    return children


def postorder_depths(children: Sequence[Sequence[int]]) -> np.ndarray:
    """Compute bottom-up levels, breaking malformed back-edges safely."""
    n = len(children)
    state = np.zeros(n, dtype=np.int8)
    depth = np.zeros(n, dtype=np.int16)

    def visit(node: int) -> int:
        if state[node] == 2:
            return int(depth[node])
        if state[node] == 1:
            return 0
        state[node] = 1
        child_depths = [visit(c) for c in children[node] if state[c] != 1]
        depth[node] = 0 if not child_depths else min(255, 1 + max(child_depths))
        state[node] = 2
        return int(depth[node])

    for index in range(n):
        visit(index)
    return depth


def left_child_right_sibling(children: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert an n-ary AST to the lossless left-child/right-sibling binary form."""
    n = len(children)
    left = np.full(n, -1, dtype=np.int64)
    right = np.full(n, -1, dtype=np.int64)
    for parent, child_list in enumerate(children):
        clean = [int(c) for c in child_list if 0 <= int(c) < n and int(c) != parent]
        if not clean:
            continue
        left[parent] = clean[0]
        for current, sibling in zip(clean, clean[1:]):
            right[current] = sibling
    return left, right


def deckard_subtree_vectors(
    node_types: Sequence[int],
    children: Sequence[Sequence[int]],
    vocab_size: int,
    *,
    min_nodes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct DECKARD characteristic vectors for every eligible AST subtree.

    Each vector is a count of AST node categories in a rooted subtree.  The
    final coordinate stores subtree size, matching DECKARD's size-aware vector
    grouping while allowing direct scoring of benchmark-supplied pairs.
    """
    n = len(node_types)
    counts = np.zeros((n, vocab_size + 1), dtype=np.float32)
    sizes = np.ones(n, dtype=np.int32)
    levels = postorder_depths(children)
    # The earlier direct transcription performed one Python-level dense-vector
    # addition per AST edge. On CodeNet that is millions of tiny NumPy calls.
    # ``np.add.at`` performs the identical child-to-parent reductions in a
    # compiled loop, while retaining duplicate/multi-parent edges and the
    # cycle guard used by the original implementation.
    parent_rows: list[int] = []
    child_rows: list[int] = []
    for parent, child_list in enumerate(children):
        for child in child_list:
            child = int(child)
            if 0 <= child < n and levels[child] < levels[parent]:
                parent_rows.append(parent)
                child_rows.append(child)
    parents = np.asarray(parent_rows, dtype=np.intp)
    child_indices = np.asarray(child_rows, dtype=np.intp)
    node_levels = levels.astype(np.intp, copy=False)
    for level in range(int(levels.max(initial=0)) + 1):
        nodes = np.flatnonzero(node_levels == level)
        if not len(nodes):
            continue
        tokens = np.asarray(node_types, dtype=np.int64)[nodes]
        valid_tokens = (tokens >= 0) & (tokens < vocab_size)
        counts[nodes[valid_tokens], tokens[valid_tokens]] += 1.0
        if len(parents):
            edge_mask = node_levels[parents] == level
            if edge_mask.any():
                np.add.at(counts, parents[edge_mask], counts[child_indices[edge_mask]])
                np.add.at(sizes, parents[edge_mask], sizes[child_indices[edge_mask]])
        counts[nodes, -1] = sizes[nodes]
    keep = sizes >= min_nodes
    return counts[keep], sizes[keep]


def deckard_pair_similarity(
    left_vectors: np.ndarray,
    left_sizes: np.ndarray,
    right_vectors: np.ndarray,
    right_sizes: np.ndarray,
    *,
    size_tolerance: float = 0.20,
) -> float:
    """DECKARD-style best subtree similarity for one supplied code pair."""
    if not len(left_vectors) or not len(right_vectors):
        return 0.0
    # Full-data runs retain integer subtree-category counts compactly.  Promote
    # only this pair's small retained vectors for the cosine computation so
    # uint8 storage cannot overflow a matrix product.
    left_vectors = np.asarray(left_vectors, dtype=np.float32)
    right_vectors = np.asarray(right_vectors, dtype=np.float32)
    left_sizes = np.asarray(left_sizes, dtype=np.float32)
    right_sizes = np.asarray(right_sizes, dtype=np.float32)
    # Pair evaluation replaces the original corpus-wide LSH candidate lookup;
    # the characteristic vectors and size filter remain unchanged.  Score all
    # retained subtree pairs in one BLAS multiplication instead of issuing up
    # to 32 Python/NumPy matrix products per benchmark pair.
    compatible = (
        np.abs(left_sizes[:, None] - right_sizes[None, :])
        / np.maximum(left_sizes[:, None], 1.0)
    ) <= size_tolerance
    if not compatible.any():
        return 0.0
    denominator = np.maximum(
        np.linalg.norm(left_vectors, axis=1)[:, None] * np.linalg.norm(right_vectors, axis=1)[None, :],
        1e-8,
    )
    similarities = (left_vectors @ right_vectors.T) / denominator
    return float(similarities[compatible].max(initial=0.0))


def _masked_max(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    result = masked.max(dim=dim).values
    return torch.where(torch.isfinite(result), result, torch.zeros_like(result))


class RecursiveTreeEncoder(nn.Module):
    """Bottom-up recursive tree encoder over an exported AST."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.node = nn.Linear(embed_dim, hidden_dim)
        self.child = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, node_types: Tensor, parents: Tensor, children: Tensor, depths: Tensor) -> Tensor:
        mask = node_types.ne(0)
        embedded = self.embedding(node_types)
        hidden = torch.zeros((*node_types.shape, self.node.out_features), device=node_types.device, dtype=embedded.dtype)
        valid_edges = parents.ge(0) & children.ge(0)
        max_depth = int(depths.max().detach().cpu()) if depths.numel() else 0
        for level in range(max_depth + 1):
            aggregate = torch.zeros_like(hidden)
            valid = valid_edges & depths.gather(1, children.clamp_min(0)).lt(level + 1)
            child_index = children.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
            child_values = hidden.gather(1, child_index) * valid.unsqueeze(-1)
            parent_index = parents.clamp_min(0).unsqueeze(-1).expand_as(child_values)
            aggregate.scatter_add_(1, parent_index, child_values)
            count = torch.zeros_like(hidden[..., :1])
            count.scatter_add_(1, parents.clamp_min(0).unsqueeze(-1), valid.unsqueeze(-1).to(hidden.dtype))
            candidate = torch.tanh(self.node(embedded) + self.child(aggregate / count.clamp_min(1.0)))
            update = depths.eq(level) & mask
            hidden = torch.where(update.unsqueeze(-1), candidate, hidden)
        return hidden * mask.unsqueeze(-1)


class ASTNNEncoder(nn.Module):
    """ASTNN block sequence: recursive subtree encoding, BiGRU, max pooling."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 100, code_dim: int = 200):
        super().__init__()
        self.tree = RecursiveTreeEncoder(vocab_size, embed_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, code_dim)

    def forward(self, node_types: Tensor, parents: Tensor, children: Tensor, depths: Tensor, statements: Tensor) -> Tensor:
        nodes = self.tree(node_types, parents, children, depths)
        statement_mask = statements.ge(0)
        indices = statements.clamp_min(0).unsqueeze(-1).expand(-1, -1, nodes.size(-1))
        blocks = nodes.gather(1, indices) * statement_mask.unsqueeze(-1)
        sequence, _ = self.gru(blocks)
        pooled = _masked_max(sequence, statement_mask, 1)
        return torch.tanh(self.projection(pooled))


class RtvNNEncoder(nn.Module):
    """Recursive tree/vector encoder with a reconstruction objective."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 192, code_dim: int = 192):
        super().__init__()
        self.tree = RecursiveTreeEncoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, embed_dim)
        self.output = nn.Linear(hidden_dim * 2, code_dim)

    def forward(self, node_types: Tensor, parents: Tensor, children: Tensor, depths: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.tree(node_types, parents, children, depths)
        mask = node_types.ne(0)
        mean = (nodes * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = _masked_max(nodes, mask, 1)
        code = torch.tanh(self.output(torch.cat([mean, maximum], dim=-1)))
        target = self.tree.embedding(node_types).detach()
        reconstruction = F.mse_loss(self.decoder(nodes)[mask], target[mask]) if mask.any() else nodes.sum() * 0
        return code, reconstruction


class BinaryTreeLSTMEncoder(nn.Module):
    """Binary Tree-LSTM over left-child/right-sibling AST conversion."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gates = nn.Linear(embed_dim + hidden_dim * 2, hidden_dim * 5)
        self.hidden_dim = hidden_dim

    def _encode_embedded(
        self, embedded: Tensor, node_types: Tensor, left: Tensor, right: Tensor, depths: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the bottom-up tree recurrence for already embedded nodes."""
        batch, nodes = node_types.shape
        h = embedded.new_zeros(batch, nodes, self.hidden_dim)
        c = embedded.new_zeros(batch, nodes, self.hidden_dim)
        active_nodes = node_types.ne(0)
        max_depth = int(depths.max().detach().cpu()) if depths.numel() else 0
        for level in range(max_depth + 1):
            # Only nodes at this level are written; evaluating the gates for the
            # whole padded batch and discarding the rest costs the padded width
            # at every level.  Selecting the level's rows first is the same
            # recurrence on exactly the rows that survive the update.
            rows, columns = (depths.eq(level) & active_nodes).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            left_rows = left[rows, columns]
            right_rows = right[rows, columns]
            left_ok = left_rows.ge(0).unsqueeze(-1)
            right_ok = right_rows.ge(0).unsqueeze(-1)
            hl = h[rows, left_rows.clamp_min(0)] * left_ok
            hr = h[rows, right_rows.clamp_min(0)] * right_ok
            cl = c[rows, left_rows.clamp_min(0)] * left_ok
            cr = c[rows, right_rows.clamp_min(0)] * right_ok
            i, fl, fr, o, u = self.gates(
                torch.cat([embedded[rows, columns], hl, hr], -1)
            ).chunk(5, -1)
            new_c = torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(fl) * cl + torch.sigmoid(fr) * cr
            new_h = torch.sigmoid(o) * torch.tanh(new_c)
            h = h.index_put((rows, columns), new_h)
            c = c.index_put((rows, columns), new_c)
        return h, c

    def forward(self, node_types: Tensor, left: Tensor, right: Tensor, depths: Tensor) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(node_types)
        # A deep, padded CodeNet AST otherwise retains one BxNxH activation
        # graph per tree level. Checkpointing preserves the forward computation
        # and recomputes it during backward, which is safe on a 16-GB Kaggle T4.
        if self.training and torch.is_grad_enabled():
            return checkpoint(
                self._encode_embedded, embedded, node_types, left, right, depths,
                use_reentrant=False,
            )
        return self._encode_embedded(embedded, node_types, left, right, depths)


class CDLHModel(nn.Module):
    """Binary Tree-LSTM followed by a supervised continuous hash layer."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 192, hash_bits: int = 32):
        super().__init__()
        self.tree_lstm = BinaryTreeLSTMEncoder(vocab_size, embed_dim, hidden_dim)
        self.hash = nn.Linear(hidden_dim, hash_bits)

    def encode(self, node_types: Tensor, left: Tensor, right: Tensor, depths: Tensor) -> Tensor:
        hidden, _ = self.tree_lstm(node_types, left, right, depths)
        mask = node_types.ne(0)
        return torch.tanh(self.hash(_masked_max(hidden, mask, 1)))

    @staticmethod
    def loss(left_hash: Tensor, right_hash: Tensor, labels: Tensor, quantization_weight: float = 0.01) -> Tensor:
        similarity = (left_hash * right_hash).mean(-1)
        pair_loss = F.binary_cross_entropy_with_logits(4.0 * similarity, labels.float())
        quantization = ((left_hash.abs() - 1.0) ** 2).mean() + ((right_hash.abs() - 1.0) ** 2).mean()
        return pair_loss + quantization_weight * quantization


def relation_message(hidden: Tensor, edge_src: Tensor, edge_dst: Tensor, transform: nn.Module) -> Tensor:
    """Batched sparse relation aggregation from padded edge-index tensors."""
    valid = edge_src.ge(0) & edge_dst.ge(0)
    src = edge_src.clamp_min(0)
    dst = edge_dst.clamp_min(0)
    src_hidden = hidden.gather(1, src.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
    # Embeddings remain float32 under AMP while Linear outputs may be fp16 or
    # bfloat16. scatter_add_ requires an exact dtype match between destination
    # and source, so normalize the message tensor to the recurrent state dtype.
    messages = (transform(src_hidden) * valid.unsqueeze(-1)).to(hidden.dtype)
    output = torch.zeros_like(hidden)
    output.scatter_add_(1, dst.unsqueeze(-1).expand_as(messages), messages)
    counts = torch.zeros_like(hidden[..., :1])
    counts.scatter_add_(1, dst.unsqueeze(-1), valid.unsqueeze(-1).to(hidden.dtype))
    return output / counts.clamp_min(1.0)


class DeepSimEncoder(nn.Module):
    """Control/data-flow semantic-matrix encoder for the exported graphs.

    DeepSim's original semantic matrix combines control- and data-flow
    information.  The project export cannot reproduce WALA instruction
    categories across all four languages, but it can retain both graph views
    and their structural roles rather than silently reducing the method to a
    CFG-only encoder.
    """

    def __init__(self, vocab_size: int, numeric_dim: int = 12, embed_dim: int = 96, hidden_dim: int = 192, code_dim: int = 192):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.self_proj = nn.Linear(embed_dim + numeric_dim, hidden_dim)
        self.cfg_message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sequence = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output = nn.Linear(hidden_dim * 2, code_dim)

    def forward(self, node_types: Tensor, numeric: Tensor, cfg_src: Tensor, cfg_dst: Tensor) -> Tensor:
        mask = node_types.ne(0)
        semantic_matrix = torch.cat([self.embedding(node_types), numeric], -1)
        semantic_hidden = self.self_proj(semantic_matrix)
        hidden = torch.tanh(semantic_hidden + relation_message(semantic_hidden, cfg_src, cfg_dst, self.cfg_message))
        sequence, _ = self.sequence(hidden)
        return F.normalize(self.output(_masked_max(sequence, mask, 1)), dim=-1)


class FAASTGGNN(nn.Module):
    """Relation-aware GGNN on our AST+CFG+DDG flow-augmented graph."""

    def __init__(self, vocab_size: int, relations: int = 6, hidden_dim: int = 192, code_dim: int = 192, steps: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.transforms = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relations))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(hidden_dim, code_dim)
        self.steps = steps

    def propagate(self, hidden: Tensor, edge_src: Tensor, edge_dst: Tensor) -> Tensor:
        # All six directed relations share the same padded edge layout.  Fuse
        # their linear projections and scatter operations into two kernels.
        # This is algebraically identical to summing ``relation_message`` for
        # every relation, but avoids six Python/CUDA dispatch chains at each
        # GGNN step (the principal full-CodeNet bottleneck).
        relations = len(self.transforms)
        weights = torch.stack([transform.weight for transform in self.transforms])
        transformed = torch.einsum("bni,roi->brno", hidden, weights).to(hidden.dtype)
        valid = edge_src.ge(0) & edge_dst.ge(0)
        src = edge_src.clamp_min(0)
        dst = edge_dst.clamp_min(0)
        messages = transformed.gather(
            2, src.unsqueeze(-1).expand(-1, -1, -1, hidden.size(-1)),
        ) * valid.unsqueeze(-1)
        relation_aggregate = hidden.new_zeros(hidden.size(0), relations, hidden.size(1), hidden.size(2))
        relation_aggregate.scatter_add_(
            2, dst.unsqueeze(-1).expand_as(messages), messages,
        )
        counts = hidden.new_zeros(hidden.size(0), relations, hidden.size(1), 1)
        counts.scatter_add_(2, dst.unsqueeze(-1), valid.unsqueeze(-1).to(hidden.dtype))
        aggregate = (relation_aggregate / counts.clamp_min(1.0)).sum(1)
        return self.gru(aggregate.flatten(0, 1), hidden.flatten(0, 1)).view_as(hidden)

    def pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        weights = torch.sigmoid(self.gate(hidden)) * mask.unsqueeze(-1)
        pooled = (weights * hidden).sum(1) / weights.sum(1).clamp_min(1e-6)
        return F.normalize(self.output(pooled), dim=-1)

    def forward(self, node_types: Tensor, edge_src: Tensor, edge_dst: Tensor) -> Tensor:
        mask = node_types.ne(0)
        hidden = self.embedding(node_types)
        for _ in range(self.steps):
            hidden = self.propagate(hidden, edge_src, edge_dst) * mask.unsqueeze(-1)
        return self.pool(hidden, mask)


class FAASTGMN(FAASTGGNN):
    """FA-AST GGNN with pair-aware cross-graph matching at every step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.embedding.embedding_dim
        self.match_gru = nn.GRUCell(hidden_dim * 2, hidden_dim)

    @staticmethod
    def cross_attention(left: Tensor, right: Tensor, left_mask: Tensor, right_mask: Tensor) -> tuple[Tensor, Tensor]:
        scale = left.size(-1) ** -0.5
        scores = torch.bmm(left, right.transpose(1, 2)) * scale
        left_scores = scores.masked_fill(~right_mask.unsqueeze(1), -1e4)
        right_scores = scores.transpose(1, 2).masked_fill(~left_mask.unsqueeze(1), -1e4)
        left_match = torch.bmm(torch.softmax(left_scores, -1), right)
        right_match = torch.bmm(torch.softmax(right_scores, -1), left)
        return left_match, right_match

    def forward_pair(
        self,
        left_types: Tensor,
        left_src: Tensor,
        left_dst: Tensor,
        right_types: Tensor,
        right_src: Tensor,
        right_dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lm, rm = left_types.ne(0), right_types.ne(0)
        left, right = self.embedding(left_types), self.embedding(right_types)
        for _ in range(self.steps):
            lp, rp = self.propagate(left, left_src, left_dst), self.propagate(right, right_src, right_dst)
            lx, rx = self.cross_attention(lp, rp, lm, rm)
            left = self.match_gru(torch.cat([lp, lp - lx], -1).flatten(0, 1), left.flatten(0, 1)).view_as(left)
            right = self.match_gru(torch.cat([rp, rp - rx], -1).flatten(0, 1), right.flatten(0, 1)).view_as(right)
            left, right = left * lm.unsqueeze(-1), right * rm.unsqueeze(-1)
        return self.pool(left, lm), self.pool(right, rm)
