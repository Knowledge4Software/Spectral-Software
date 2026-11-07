"""Canonical SPECTRA-Siam model for mono- and zero-shot cross-language runs.

The encoder consumes only stable canonical category IDs plus a bounded hashed
lexical sketch. Parser-specific raw node strings never index an embedding
table. Input topology is retained as a soft latent-graph prior.
"""

from __future__ import annotations

from .canonical_types import NUM_CANONICAL_TYPES
ADAPTIVE_TEMPERATURE_MIN = 0.20
ADAPTIVE_TEMPERATURE_MAX = 1.20
USE_ADAPTIVE_TEMPERATURE = True
USE_SPECTRAL_ATTENTION = False
SPECTRAL_VARIANCE_FLOOR = 0.03
TEMPERATURE_END = 0.30

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F




@dataclass(frozen=True)
class RevisionModelConfig:
    canonical_vocab_size: int = NUM_CANONICAL_TYPES + 1  # zero is padding
    lexical_buckets: int = 4096
    hidden_dim: int = 256
    max_input_nodes: int = 256
    latent_nodes: int = 32
    slot_iterations: int = 3
    attention_heads: int = 4
    lexical_dropout: float = 0.30
    use_node_lexical: bool = True
    use_source_lexical: bool = False
    readout_mode: str = "graph_signal_spectral"
    adjacency_mode: str = "learned"
    pooling: str = "spectral"
    target_density: float = 0.15
    chebyshev_degree: int = 12
    density_bins: int = 32
    heat_samples: int = 24
    spectral_bands: int = 12
    spectral_signals: int = 8
    dropout: float = 0.10
    relation_indices: tuple[int, ...] = (0, 2)  # AST plus DDG projected onto the AST backbone
    prior_relation_indices: tuple[int, ...] | None = None  # keep encoder inputs fixed in adjacency-view controls

    @property
    def descriptor_dim(self) -> int:
        return self.density_bins + self.heat_samples + self.spectral_bands * self.spectral_signals

    @property
    def eigenvalue_descriptor_dim(self) -> int:
        return self.density_bins + self.heat_samples

    @property
    def readout_descriptor_dim(self) -> int:
        if self.pooling == "energy_only":
            return self.spectral_bands * self.spectral_signals
        if self.readout_mode == "eigenvalue_only":
            return self.eigenvalue_descriptor_dim
        return self.descriptor_dim


class RelationGraphLayer(nn.Module):
    def __init__(self, dim: int, relations: int = 3, dropout: float = 0.10) -> None:
        super().__init__()
        self.self_linear = nn.Linear(dim, dim)
        self.relation_linears = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(relations))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, edges: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        message = self.self_linear(states)
        for relation, linear in enumerate(self.relation_linears):
            adjacency = edges[:, relation]
            degree = adjacency.sum(-1, keepdim=True).clamp_min(1.0)
            message = message + torch.bmm(adjacency / degree, linear(states))
        return self.norm(states + self.dropout(F.gelu(message))) * mask.unsqueeze(-1)


class SlotAttentionInduction(nn.Module):
    def __init__(self, dim: int, slots: int, iterations: int) -> None:
        super().__init__()
        self.slots = slots
        self.iterations = iterations
        self.scale = dim**-0.5
        self.slot_seed = nn.Parameter(torch.randn(1, slots, dim) * 0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.slot_norm = nn.LayerNorm(dim)
        self.to_query = nn.Linear(dim, dim, bias=False)
        self.to_key = nn.Linear(dim, dim, bias=False)
        self.to_value = nn.Linear(dim, dim, bias=False)
        self.update = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(states)
        key, value = self.to_key(normalized), self.to_value(normalized)
        slots = self.slot_seed.expand(states.size(0), -1, -1)
        for _ in range(self.iterations):
            scores = torch.einsum("bsd,bnd->bns", self.to_query(self.slot_norm(slots)), key) * self.scale
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e4)
            assignments = torch.softmax(scores, dim=-1) * mask.unsqueeze(-1)
            weights = assignments / assignments.sum(1, keepdim=True).clamp_min(1e-6)
            updates = torch.einsum("bns,bnd->bsd", weights, value)
            slots = self.update(updates.reshape(-1, updates.size(-1)), slots.reshape(-1, slots.size(-1))).view_as(slots)
            slots = slots + self.mlp(slots)
        return slots, assignments


class LatentGraphRefinement(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(dim, dim, bias=False)
        self.self_linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1e-6)
        message = torch.bmm(adjacency / degree, self.message(states))
        return self.norm(states + self.dropout(F.gelu(self.self_linear(states) + message)))


class MultiScaleSpectralExtractor(nn.Module):
    """Graph spectrum plus attributed graph-signal spectral energies.

    In ``graph_signal_spectral`` mode, canonical/lexical information can reach
    the classifier only after node signals have been filtered by polynomial
    functions of the learned latent Laplacian.  Squared energy is aggregated
    over nodes, making the descriptor invariant to latent-node permutations.
    There is no state pooling or raw-feature concatenation in this branch.
    """

    def __init__(self, dim: int, config: RevisionModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.readout_mode in {"hybrid", "graph_signal_spectral"}:
            self.signal = nn.Linear(dim, config.spectral_signals, bias=False)
        else:
            self.signal = None
        self.band_attention = (
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, config.spectral_bands))
            if config.readout_mode == "hybrid"
            else None
        )

        # Chebyshev coefficients use uniform angular quadrature, not uniform
        # Laplacian eigenvalues (which gives a different integration measure).
        theta = (torch.arange(256) + 0.5) * math.pi / 256
        grid = 1 + torch.cos(theta)
        coefficients = []
        for center in torch.linspace(0.05, 1.95, config.spectral_bands):
            response = torch.exp(-0.5 * ((grid - center) / 0.18) ** 2)
            coefficients.append(
                torch.stack(
                    [
                        (response * torch.cos(order * theta)).mean() * (1 if order == 0 else 2)
                        for order in range(config.chebyshev_degree + 1)
                    ]
                )
            )
        self.register_buffer("chebyshev_coefficients", torch.stack(coefficients))
        self.register_buffer("density_centers", torch.linspace(0, 2, config.density_bins))
        self.register_buffer("heat_times", torch.logspace(-2, 2, config.heat_samples))

    def forward(self, states: torch.Tensor, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Eigensolver and graph filtering remain float32 under AMP.
        with torch.autocast(device_type=states.device.type, enabled=False):
            states, adjacency = states.float(), adjacency.float()
            batch_size, node_count, _ = adjacency.shape
            identity = torch.eye(node_count, device=adjacency.device).unsqueeze(0)
            adjacency = torch.nan_to_num(adjacency, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2)) * (1 - identity)
            degree = adjacency.sum(-1).clamp_min(1e-6)
            normalized = adjacency * degree.rsqrt().unsqueeze(-1) * degree.rsqrt().unsqueeze(-2)
            laplacian = identity - normalized
            laplacian = 0.5 * (laplacian + laplacian.transpose(1, 2))

            if self.config.readout_mode in {"eigenvalue_only", "graph_signal_spectral"}:
                eigenvalues = torch.linalg.eigvalsh(laplacian).clamp(0, 2)
            else:
                with torch.no_grad():
                    eigenvalues = torch.linalg.eigvalsh(laplacian).clamp(0, 2)
            density = torch.exp(
                -0.5 * ((eigenvalues.unsqueeze(-1) - self.density_centers) / 0.08) ** 2
            ).mean(1)
            heat_trace = torch.exp(-eigenvalues.unsqueeze(-1) * self.heat_times).mean(1)

            if self.config.readout_mode == "eigenvalue_only":
                descriptor = torch.cat((density, heat_trace), dim=-1)
                band_weights = torch.full(
                    (batch_size, self.config.spectral_bands),
                    1.0 / self.config.spectral_bands,
                    device=adjacency.device,
                )
            else:
                if self.signal is None:
                    raise RuntimeError("Graph-signal projection was not initialized")
                # A learned channel projection creates a compact attributed
                # graph signal. RMS normalization prevents raw signal scale
                # from acting as a non-spectral shortcut.
                signal = self.signal(states)
                signal = signal / signal.square().mean(1, keepdim=True).add(1e-6).sqrt()

                # T_k(L-I)X is a Chebyshev approximation of spectral filters
                # g_b(L)X. No node state skips these graph-dependent operators.
                shifted = laplacian - identity
                terms = [signal, torch.bmm(shifted, signal)]
                for _ in range(2, self.config.chebyshev_degree + 1):
                    terms.append(2 * torch.bmm(shifted, terms[-1]) - terms[-2])
                filtered = torch.einsum(
                    "bkse,ck->bcse", torch.stack(terms, dim=1), self.chebyshev_coefficients
                )
                raw_energy = torch.log1p(filtered.square().mean(2))

                if self.config.readout_mode == "graph_signal_spectral":
                    # Fixed bands prevent an input-conditioned state gate from
                    # becoming a covert lexical/node-embedding bypass.
                    band_weights = torch.full(
                        (batch_size, self.config.spectral_bands),
                        1.0 / self.config.spectral_bands,
                        device=adjacency.device,
                    )
                    energy = raw_energy.reshape(batch_size, -1)
                else:
                    if self.band_attention is None:
                        raise RuntimeError("Hybrid spectral attention was not initialized")
                    band_weights = (
                        torch.softmax(self.band_attention(states.mean(1)), dim=-1)
                        if USE_SPECTRAL_ATTENTION
                        else torch.full(
                            (batch_size, self.config.spectral_bands),
                            1.0 / self.config.spectral_bands,
                            device=adjacency.device,
                        )
                    )
                    energy = (raw_energy * band_weights.unsqueeze(-1)).reshape(batch_size, -1)
                descriptor = torch.cat((density, heat_trace, energy), dim=-1)
        return descriptor, eigenvalues, band_weights


class RevisionEncoder(nn.Module):
    def __init__(self, config: RevisionModelConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if config.readout_mode not in {"eigenvalue_only", "graph_signal_spectral"}:
            raise ValueError("unsupported readout_mode")
        if config.adjacency_mode not in {"learned", "projected"} or config.pooling not in {"spectral", "mean_max", "attention", "energy_only"}:
            raise ValueError("Unsupported controlled architecture")
        if config.readout_mode != "hybrid" and config.use_source_lexical:
            raise ValueError("source lexical residual is incompatible with spectral-only readouts")
        self.config = config
        dim = config.hidden_dim
        self.canonical_embedding = nn.Embedding(config.canonical_vocab_size, dim, padding_idx=0)
        self.lexical_embedding = nn.Embedding(config.lexical_buckets, dim, padding_idx=0)
        # Start lexical fallback at roughly 20% of the canonical signal.  It
        # can help when an identifier/literal is available, but cannot become
        # a direct raw-node-type lookup table.
        self.lexical_gate_logit = nn.Parameter(torch.tensor(-1.38629436))
        self.lexical_dropout = nn.Dropout(config.lexical_dropout)
        if config.use_source_lexical:
            self.source_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
            self.source_gate_logit = nn.Parameter(torch.tensor(-0.61903921))
        else:
            self.source_projection = None
            self.register_parameter("source_gate_logit", None)
        self.input_norm = nn.LayerNorm(dim)
        if not config.relation_indices:
            raise ValueError("At least one input relation must be enabled.")
        self.graph_layers = nn.ModuleList(
            [RelationGraphLayer(dim, relations=len(config.relation_indices), dropout=config.dropout) for _ in range(2)]
        )
        self.slot_attention = SlotAttentionInduction(dim, config.latent_nodes, config.slot_iterations)
        self.adjacency_query = nn.Linear(dim, dim, bias=False)
        self.adjacency_key = nn.Linear(dim, dim, bias=False)
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        self.temperature_bias = nn.Parameter(torch.tensor(-0.55))
        self.temperature_complexity = nn.Parameter(torch.tensor(1.25))
        self.refinement = nn.ModuleList([LatentGraphRefinement(dim, config.dropout) for _ in range(2)])
        if config.readout_mode == "eigenvalue_only" and config.pooling == "spectral":
            self.refinement.requires_grad_(False)
        spectral_config = replace(config, readout_mode="eigenvalue_only") if config.pooling in {"mean_max", "attention"} else config
        self.spectral = MultiScaleSpectralExtractor(dim, spectral_config)
        if config.adjacency_mode == "projected":
            for module in (self.adjacency_query, self.adjacency_key):
                module.requires_grad_(False)
            for parameter in (self.prior_scale, self.temperature_bias, self.temperature_complexity):
                parameter.requires_grad_(False)
        # A common 152-D positive descriptor and the same symmetric pair head
        # keep the readout controls comparable. Exact parameter counts are saved.
        self.pool_projection = (nn.Linear(dim * 2 if config.pooling == "mean_max" else dim,
                                         config.readout_descriptor_dim)
                                if config.pooling in {"mean_max", "attention"} else None)
        self.pool_attention = nn.Linear(dim, 1) if config.pooling == "attention" else None
        self.output = (
            nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
            if config.readout_mode == "hybrid"
            else None
        )

    def forward(self, batch, temperature: float = 0.30):
        canonical = self.canonical_embedding(batch.canonical_ids.long())
        if self.config.use_node_lexical:
            lexical = self.lexical_embedding(batch.lexical_ids.long()).mean(-2)
            lexical = self.lexical_dropout(lexical) if self.training else lexical
            canonical = canonical + torch.sigmoid(self.lexical_gate_logit) * lexical
        input_features = self.input_norm(canonical) * batch.mask.unsqueeze(-1)
        enabled_edges = batch.edges[:, self.config.relation_indices]
        states = input_features
        for layer in self.graph_layers:
            states = layer(states, enabled_edges, batch.mask)

        slots, assignment = self.slot_attention(states, batch.mask)
        assignment = torch.nan_to_num(assignment)
        assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-6)
        reconstruction = torch.bmm(assignment, slots)
        reconstruction_loss = (
            (input_features - reconstruction).square().mean(-1) * batch.mask
        ).sum() / batch.mask.sum().clamp_min(1)

        input_adjacency = enabled_edges.sum(1).clamp(max=1)
        prior_adjacency = (batch.edges[:, self.config.prior_relation_indices].sum(1).clamp(max=1)
                           if self.config.prior_relation_indices is not None else input_adjacency)
        structural_prior = torch.bmm(
            assignment.transpose(1, 2), torch.bmm(prior_adjacency, assignment)
        )
        structural_prior = structural_prior / structural_prior.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        head_dim = self.config.hidden_dim // self.config.attention_heads
        with torch.autocast(device_type=slots.device.type, enabled=False):
            query = self.adjacency_query(slots.float()).view(
                -1, self.config.latent_nodes, self.config.attention_heads, head_dim
            ).transpose(1, 2)
            key = self.adjacency_key(slots.float()).view(
                -1, self.config.latent_nodes, self.config.attention_heads, head_dim
            ).transpose(1, 2)
            score = torch.matmul(query, key.transpose(-1, -2)).mean(1) / math.sqrt(head_dim)
            score = 0.5 * (score + score.transpose(1, 2)) + self.prior_scale.float() * structural_prior.float()
            score = torch.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20, 20)
            identity = torch.eye(self.config.latent_nodes, device=score.device).unsqueeze(0)
            node_complexity = batch.mask.float().sum(-1) / self.config.max_input_nodes
            edge_complexity = enabled_edges.float().sum(dim=(1, 2, 3)) / batch.mask.float().sum(-1).square().clamp_min(1.0)
            complexity = (0.75 * node_complexity + 0.25 * edge_complexity.clamp(0, 1)).clamp(0, 1)
            adaptive_temperature = ADAPTIVE_TEMPERATURE_MIN + (ADAPTIVE_TEMPERATURE_MAX - ADAPTIVE_TEMPERATURE_MIN) * torch.sigmoid(self.temperature_bias.float() + F.softplus(self.temperature_complexity.float()) * complexity) if USE_ADAPTIVE_TEMPERATURE else torch.full_like(complexity, TEMPERATURE_END)
            # Keep a differentiable weighted graph. A fixed top-k gave
            # every example nearly the same edge count and erased precisely the
            # graph-to-graph spectral variation that the readout needs.
            adjacency = torch.sigmoid(score / adaptive_temperature[:, None, None]) * (1 - identity)
            adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
            if self.config.adjacency_mode == "projected":
                adjacency = 0.5 * (structural_prior.float() + structural_prior.float().transpose(1, 2)) * (1 - identity)

        projected_topology = torch.bmm(assignment, torch.bmm(adjacency.to(assignment.dtype), assignment.transpose(1, 2)))
        projected_topology = projected_topology.clamp(1e-6, 1 - 1e-6)
        topology_mask = batch.mask.unsqueeze(1) & batch.mask.unsqueeze(2)
        topology_mask = topology_mask & ~torch.eye(
            topology_mask.size(-1), device=topology_mask.device, dtype=torch.bool
        ).unsqueeze(0)
        topology_target = torch.maximum(input_adjacency, input_adjacency.transpose(1, 2)).to(projected_topology.dtype)
        positive_edges = (topology_target * topology_mask).sum()
        negative_edges = ((1 - topology_target) * topology_mask).sum()
        topology_positive_weight = (negative_edges / positive_edges.clamp_min(1)).detach().clamp(1, 20)
        topology_element = -(
            topology_positive_weight * topology_target * projected_topology.log()
            + (1 - topology_target) * (1 - projected_topology).log()
        )
        topology_reconstruction_loss = (
            topology_element * topology_mask
        ).sum() / topology_mask.sum().clamp_min(1)

        latent = slots
        for layer in self.refinement:
            latent = layer(latent, adjacency.to(latent.dtype))
        descriptor, eigenvalues, band_weights = self.spectral(latent, adjacency)
        if self.config.pooling == "mean_max":
            descriptor = F.softplus(self.pool_projection(torch.cat((latent.mean(1), latent.amax(1)), dim=-1)))
        elif self.config.pooling == "attention":
            weights = torch.softmax(self.pool_attention(latent), dim=1)
            descriptor = F.softplus(self.pool_projection((latent * weights).sum(1)))
        elif self.config.pooling == "energy_only":
            descriptor = descriptor[:, self.config.density_bins + self.config.heat_samples:]
        # No unused learned embedding projection. This diagnostic vector has
        # no path to the classifier or a nonzero loss term.
        spectral_embedding = descriptor.to(latent.dtype)
        if self.config.readout_mode in {"eigenvalue_only", "graph_signal_spectral"}:
            # In graph_signal_spectral mode this embedding is derived only from
            # g_b(L)X energies and adjacency-spectrum summaries.
            embedding = F.normalize(spectral_embedding, dim=-1)
        else:
            if self.output is None:
                raise RuntimeError("Hybrid latent fusion module was not initialized")
            embedding = F.normalize(
                self.output(torch.cat((latent.mean(1), spectral_embedding), dim=-1)), dim=-1
            )
        if self.config.use_source_lexical:
            if self.source_projection is None or self.source_gate_logit is None:
                raise RuntimeError("Source lexical residual was not initialized")
            source = self.lexical_embedding(batch.source_lexical_ids.long()).mean(1)
            source = self.source_projection(source)
            embedding = F.normalize(embedding + torch.sigmoid(self.source_gate_logit) * source, dim=-1)
        density = adjacency.sum(dim=(1, 2)) / (
            self.config.latent_nodes * (self.config.latent_nodes - 1)
        )
        density_loss = (density - self.config.target_density).square().mean()
        connectivity_loss = F.relu(0.03 - eigenvalues[:, 1]).square().mean()
        auxiliary = {
            "reconstruction": reconstruction_loss,
            "topology_reconstruction": topology_reconstruction_loss,
            "density": density_loss,
            "connectivity": connectivity_loss,
            "adaptive_temperature": adaptive_temperature,
            "spectral_attention_entropy": -(band_weights * band_weights.clamp_min(1e-8).log()).sum(-1),
            "graph": 2 * density_loss + 0.1 * connectivity_loss,
        }
        if getattr(self, "capture_diagnostics", False):
            auxiliary.update(assignment=assignment, adjacency=adjacency, eigenvalues=eigenvalues)
        return embedding, descriptor, auxiliary


class RevisionPairModel(nn.Module):
    """Symmetric comparator over separately normalized spectral blocks only."""

    def __init__(self, config: RevisionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = RevisionEncoder(config)
        # Density and heat trace are always separate comparison blocks.  The
        # attributed Chebyshev-energy block exists only in spectral-signal modes.
        self.block_sizes = (config.density_bins, config.heat_samples)
        if config.readout_mode != "eigenvalue_only":
            self.block_sizes += (config.spectral_bands * config.spectral_signals,)
        if config.pooling == "energy_only":
            self.block_sizes = (config.spectral_bands * config.spectral_signals,)
        pair_dim = 3 * config.readout_descriptor_dim + 3 * len(self.block_sizes)
        pair_hidden_dim = config.hidden_dim + config.hidden_dim // 2
        pair_bottleneck_dim = config.hidden_dim - config.hidden_dim // 8
        self.classifier = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(pair_hidden_dim, pair_bottleneck_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(pair_bottleneck_dim, 1),
        )

    def _spectral_pair_features(
        self, left_spectrum: torch.Tensor, right_spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_blocks = torch.split(left_spectrum, self.block_sizes, dim=-1)
        right_blocks = torch.split(right_spectrum, self.block_sizes, dim=-1)
        interactions = []
        summaries = []
        block_cosines = []
        for left_block, right_block in zip(left_blocks, right_blocks):
            left_log = torch.log1p(left_block.clamp_min(0))
            right_log = torch.log1p(right_block.clamp_min(0))
            left_shape = F.normalize(left_log, p=2, dim=-1)
            right_shape = F.normalize(right_log, p=2, dim=-1)
            difference = (left_shape - right_shape).abs()
            interactions.extend((difference, left_shape * right_shape, (left_log - right_log).abs()))
            block_cosine = F.cosine_similarity(left_shape, right_shape)
            block_cosines.append(block_cosine)
            summaries.extend(
                (
                    block_cosine,
                    difference.mean(-1),
                    difference.square().mean(-1).add(1e-8).sqrt(),
                )
            )
        # The head sees only symmetric, block-wise spectral comparisons.  It
        # never receives an individual descriptor, code embedding, pooled node
        # state, lexical vector, or source-token feature.
        pair_features = torch.cat((*interactions, torch.stack(summaries, dim=-1)), dim=-1)
        spectral_cosine = torch.stack(block_cosines, dim=-1).mean(-1)
        return pair_features, spectral_cosine

    def forward(self, left, right, temperature: float = 0.30):
        left_embedding, left_spectrum, left_aux = self.encoder(left, temperature)
        right_embedding, right_spectrum, right_aux = self.encoder(right, temperature)
        spectral_pair_features, spectral_cosine = self._spectral_pair_features(
            left_spectrum, right_spectrum
        )
        auxiliary = {key: 0.5 * (left_aux[key] + right_aux[key]) for key in left_aux}
        auxiliary["spectral_cosine"] = spectral_cosine
        # Embedding similarity is diagnostic metadata only.  It is excluded
        # from the classifier and every non-zero training-loss term.
        auxiliary["embedding_cosine"] = F.cosine_similarity(left_embedding, right_embedding)
        spectral_batch = torch.cat(
            (F.normalize(left_spectrum, dim=-1), F.normalize(right_spectrum, dim=-1)), dim=0
        )
        spectral_std = torch.sqrt(spectral_batch.var(dim=0, unbiased=False) + 1e-4).mean()
        auxiliary["spectral_variance"] = F.relu(SPECTRAL_VARIANCE_FLOOR - spectral_std)
        return self.classifier(spectral_pair_features).squeeze(-1), auxiliary

def canonical_spectra_loss(
    logits: torch.Tensor,
    auxiliary: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    positive_class_weight: float = 1.0,
    spectral_weight: float = 0.30,
    auc_ranking_weight: float = 0.10,
    reconstruction_weight: float = 0.05,
    topology_reconstruction_weight: float = 0.05,
    spectral_variance_weight: float = 0.05,
    graph_weight: float = 0.01,
    embedding_contrastive_weight: float = 0.0,
    embedding_negative_margin: float = 0.20,
    hard_negative_weight: float = 0.20,
    hard_negative_margin: float = 0.10,
    hard_negative_fraction: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = labels.float()
    if positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be positive")
    classification = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.as_tensor(positive_class_weight, device=logits.device, dtype=logits.dtype),
    )
    positive = labels * (1 - auxiliary["spectral_cosine"]).square()
    negative = (1 - labels) * F.relu(auxiliary["spectral_cosine"] - 0.25).square()
    spectral_contrastive = (positive + negative).mean()
    positive_logits = logits[labels > 0.5]
    negative_logits = logits[labels < 0.5]
    if positive_logits.numel() and negative_logits.numel():
        auc_ranking = F.softplus(
            -(positive_logits.unsqueeze(1) - negative_logits.unsqueeze(0))
        ).mean()
    else:
        auc_ranking = logits.new_zeros(())

    if embedding_contrastive_weight != 0:
        raise ValueError("embedding_contrastive_weight must remain zero for descriptor-only SPECTRA-Siam")
    embedding_contrastive = logits.new_zeros(())

    # Hard-negative mining is based on the block-wise spectral comparison,
    # never on a standalone embedding similarity.
    known_negative_scores = F.relu(
        auxiliary["spectral_cosine"][labels < 0.5] - hard_negative_margin
    ).square()
    if known_negative_scores.numel():
        hard_count = max(1, int(known_negative_scores.numel() * hard_negative_fraction))
        hard_negative = torch.topk(known_negative_scores, hard_count).values.mean()
    else:
        hard_negative = logits.new_zeros(())


    total = (
        classification
        + spectral_weight * spectral_contrastive
        + auc_ranking_weight * auc_ranking
        + hard_negative_weight * hard_negative
        + reconstruction_weight * auxiliary["reconstruction"]
        + topology_reconstruction_weight * auxiliary["topology_reconstruction"]
        + spectral_variance_weight * auxiliary["spectral_variance"]
        + graph_weight * auxiliary["graph"]
    )
    parts = {
        "classification": classification.detach(),
        "spectral_contrastive": spectral_contrastive.detach(),
        "auc_ranking": auc_ranking.detach(),
        "embedding_contrastive": embedding_contrastive.detach(),
        "hard_negative": hard_negative.detach(),
        "reconstruction": auxiliary["reconstruction"].detach(),
        "topology_reconstruction": auxiliary["topology_reconstruction"].detach(),
        "spectral_variance": auxiliary["spectral_variance"].detach(),
        "graph": auxiliary["graph"].detach(),
    }
    return total, parts
