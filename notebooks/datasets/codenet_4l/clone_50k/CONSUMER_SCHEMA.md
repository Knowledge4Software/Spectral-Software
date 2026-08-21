# CodeNet 50k consumer schema

The production package is the union of inputs used by the current Kaggle run
order, not only the SPECTRA-Siam notebook.

| Consumer | Required package data |
| --- | --- |
| DECKARD, ASTNN, RtvNN, CDLH | AST adjacency, node IDs, node types |
| DeepSim | CFG adjacency, node IDs, node types |
| FA-AST GGNN/GMN | AST, CFG, DDG adjacency and aligned node metadata |
| observed-graph GNN baselines | AST, CFG, DDG, CPG adjacency and spectra |
| SNN and local spectral baselines | AST, CFG, DDG, CPG spectra |
| SPECTRA-Siam | source text, AST labels/types/adjacency, DDG alignment |

Consequently, all four graph layers and their precomputed eigenvalues remain in
the release. The compact export omits only fields no current consumer reads:

- `adjacency.format`;
- `adjacency.directed`;
- derived `eigenvalue_count` (consumers can use `len(eigenvalues)`);
- `node_labels` for CFG, DDG, and CPG (AST labels are retained).

Every exported layer is capped at 128 low-frequency eigenvalues. This is the
raw spectral window used by the production suite (`SNN` uses `K_EIGEN=128`);
GNN and local spectral baselines consume the same consistent 128-value
representation, while SPECTRA-Siam recomputes its latent spectrum.
Oversized layers use sparse shift-invert with a correctness-preserving fallback
to ARPACK smallest-magnitude mode. It limits BLAS to one thread per method and
runs four independent 500-method graph shards by default. The optimized 16 GB
defaults are `SPECTRAL_SHARD_WORKERS=4`, `SPECTRAL_WORKERS=1`,
`SPECTRAL_BLAS_THREADS=1`, `SPECTRAL_APPROX_TOPK=128`, and
`SPECTRAL_SPARSE_SOLVER=shift_invert`.
