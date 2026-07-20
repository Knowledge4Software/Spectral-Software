"""Lightweight Siamese latent-graph encoder for code-clone experiments.

The input is a fixed-size sequence of AST node types and depths. Cross
attention pools variable-size ASTs into a small learned latent graph. Its
normalized Laplacian eigenvalues are differentiable PyTorch features, so the
spectral representation is optimized jointly with clone supervision.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 96,
        latent_nodes: int = 24,
        attention_heads: int = 4,
        attention_layers: int = 1,
        max_depth: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads.")

        self.latent_nodes = latent_nodes
        self.type_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(max_depth + 1, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ast_attention = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
        self.latent_queries = nn.Parameter(torch.empty(latent_nodes, hidden_dim))
        nn.init.normal_(self.latent_queries, std=hidden_dim**-0.5)
        self.pool_attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout, batch_first=True)
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.adjacency_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.adjacency_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.spectrum_projector = nn.Sequential(
            nn.LayerNorm(latent_nodes),
            nn.Linear(latent_nodes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim * 2)
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_spectrum: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode padded ASTs into L2-normalized latent spectral vectors."""
        if node_types.ndim != 2 or depths.shape != node_types.shape or mask.shape != node_types.shape:
            raise ValueError("node_types, depths, and mask must have matching [batch, nodes] shapes.")

        # Embedding indices must be int32/int64; compact AST inputs may use int16.
        node_types = node_types.long()
        depths = depths.long().clamp(min=0, max=self.depth_embedding.num_embeddings - 1)
        mask = mask.bool()
        tokens = self.input_norm(self.type_embedding(node_types) + self.depth_embedding(depths))
        ast_states = self.ast_attention(tokens, src_key_padding_mask=~mask)

        queries = self.latent_queries.unsqueeze(0).expand(node_types.size(0), -1, -1)
        latent, _ = self.pool_attention(queries, ast_states, ast_states, key_padding_mask=~mask, need_weights=False)
        latent = self.latent_norm(latent)

        query = self.adjacency_query(latent)
        key = self.adjacency_key(latent)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.size(-1))
        scores = 0.5 * (scores + scores.transpose(1, 2))
        adjacency = torch.sigmoid(scores)
        adjacency = adjacency * (1.0 - torch.eye(self.latent_nodes, device=adjacency.device, dtype=adjacency.dtype))

        degree = adjacency.sum(dim=-1).clamp_min(1e-6)
        normalized_adjacency = adjacency * degree.rsqrt().unsqueeze(-1) * degree.rsqrt().unsqueeze(-2)
        laplacian = torch.eye(self.latent_nodes, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0) - normalized_adjacency
        eigenvalues = torch.linalg.eigvalsh(laplacian)

        pooled_latent = latent.mean(dim=1)
        spectrum = self.spectrum_projector(eigenvalues)
        embedding = F.normalize(self.output_projector(self.output_norm(torch.cat([pooled_latent, spectrum], dim=1))), dim=1)
        return (embedding, eigenvalues) if return_spectrum else embedding


class SiameseLatentGraphModel(nn.Module):
    def __init__(self, encoder: LatentGraphEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        left_types: torch.Tensor,
        left_depths: torch.Tensor,
        left_mask: torch.Tensor,
        right_types: torch.Tensor,
        right_depths: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left = self.encoder(left_types, left_depths, left_mask)
        right = self.encoder(right_types, right_depths, right_mask)
        cosine = (left * right).sum(dim=1).clamp(-1.0, 1.0)
        logits = self.logit_scale.clamp(1.0, 30.0) * cosine + self.logit_bias
        return logits, cosine


def clone_loss(
    logits: torch.Tensor,
    cosine: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "hybrid",
    margin: float = 0.25,
    hard_negative_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine calibrated pair BCE with a contrastive metric objective."""
    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    positive = labels * (1.0 - cosine).square()
    negative_hinge = (1.0 - labels) * F.relu(cosine - margin).square()
    hard_weight = 1.0 + (hard_negative_weight - 1.0) * (1.0 - labels) * (cosine.detach() > margin).float()
    metric = (positive + hard_weight * negative_hinge).mean()

    if objective == "bce":
        loss = bce
    elif objective == "contrastive":
        loss = metric
    elif objective == "hybrid":
        loss = bce + metric
    else:
        raise ValueError("objective must be one of: bce, contrastive, hybrid.")
    return loss, {"bce": bce.detach(), "metric": metric.detach()}
