from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from spectral_code.revision.model import RevisionModelConfig, RevisionPairModel, canonical_spectra_loss


def _batch(size=8, padded=8):
    ids = torch.zeros(2, padded, dtype=torch.long)
    ids[:, :size] = torch.arange(1, size + 1) % 20 + 1
    lexical = torch.ones(2, padded, 4, dtype=torch.long)
    mask = ids.ne(0)
    edges = torch.zeros(2, 3, padded, padded)
    for i in range(size - 1):
        edges[:, 0, i, i + 1] = edges[:, 0, i + 1, i] = 1
    return SimpleNamespace(canonical_ids=ids, lexical_ids=lexical, mask=mask, edges=edges)


def _config(**kwargs):
    return RevisionModelConfig(hidden_dim=16, latent_nodes=4, spectral_bands=3, spectral_signals=2,
                               density_bins=4, heat_samples=3, chebyshev_degree=3, **kwargs)


@pytest.mark.parametrize("changes", ({}, {"pooling": "mean_max"}, {"pooling": "attention"},
                                    {"adjacency_mode": "projected"}, {"readout_mode": "eigenvalue_only"},
                                    {"pooling": "energy_only"}))
def test_readout_controls_have_finite_backward_and_symmetric_predictions(changes):
    torch.set_num_threads(1)
    model = RevisionPairModel(_config(**changes)).eval()
    a, b = _batch(6), _batch(7)
    ab, aux = model(a, b)
    ba, _ = model(b, a)
    torch.testing.assert_close(ab, ba)
    loss, _ = canonical_spectra_loss(ab, aux, torch.tensor([0., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_adding_masked_padding_does_not_change_the_prediction():
    torch.set_num_threads(1)
    model = RevisionPairModel(_config()).eval()
    with torch.no_grad():
        compact, _ = model(_batch(6, 6), _batch(6, 6))
        padded, _ = model(_batch(6, 12), _batch(6, 12))
    torch.testing.assert_close(compact, padded, atol=2e-6, rtol=2e-5)


def test_spectral_descriptor_is_invariant_to_joint_latent_node_permutation():
    from spectral_code.revision.model import MultiScaleSpectralExtractor
    torch.manual_seed(42)
    extractor = MultiScaleSpectralExtractor(16, _config()).eval()
    states = torch.randn(2, 4, 16)
    adjacency = torch.rand(2, 4, 4)
    adjacency = (adjacency + adjacency.transpose(1, 2)) / 2
    adjacency = adjacency * (1 - torch.eye(4))
    permutation = torch.tensor([2, 0, 3, 1])
    a, eigen_a, _ = extractor(states, adjacency)
    b, eigen_b, _ = extractor(states[:, permutation], adjacency[:, permutation][:, :, permutation])
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(eigen_a, eigen_b, atol=2e-6, rtol=2e-5)
