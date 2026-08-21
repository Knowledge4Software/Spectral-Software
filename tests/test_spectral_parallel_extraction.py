from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import csgraph

from spectral_code.evaluation.clean_data_export import _graph_to_sparse_adjacency
from spectral_code.spectral.extractor import _approximate_sparse_spectrum, extract_all_spectral_features


def _graph(offset: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in range(6):
        graph.add_node(node, type=f"TYPE_{node}", label=f"label_{offset}_{node}")
    graph.add_edges_from((node, node + 1) for node in range(5))
    graph.add_edge(0, 3)
    return graph


def test_parallel_spectral_extraction_matches_sequential(monkeypatch) -> None:
    graph_db = {
        method: {kind: _graph(method) for kind in ("ast", "cfg", "ddg", "cpg")}
        for method in range(6)
    }
    monkeypatch.setenv("SPECTRAL_WORKERS", "1")
    sequential = extract_all_spectral_features(graph_db, ["ast", "cfg", "ddg", "cpg"])
    monkeypatch.setenv("SPECTRAL_WORKERS", "4")
    monkeypatch.setenv("SPECTRAL_BLAS_THREADS", "1")
    parallel = extract_all_spectral_features(graph_db, ["ast", "cfg", "ddg", "cpg"])

    assert sequential[1] == parallel[1]
    assert sequential[3:] == parallel[3:]
    for method in graph_db:
        for kind in graph_db[method]:
            np.testing.assert_allclose(
                sequential[0][method][kind]["eigenvalues"],
                parallel[0][method][kind]["eigenvalues"],
            )
            assert sequential[0][method][kind]["status"] == parallel[0][method][kind]["status"]


def test_compact_adjacency_keeps_only_ast_labels() -> None:
    graph = _graph(1)
    ast = _graph_to_sparse_adjacency(graph, include_node_labels=True)
    cfg = _graph_to_sparse_adjacency(graph, include_node_labels=False)

    assert ast["node_labels"] == [f"label_1_{node}" for node in range(6)]
    assert "node_labels" not in cfg
    assert "format" not in ast
    assert "directed" not in ast


def test_shift_invert_sparse_spectrum_matches_low_exact_eigenvalues(monkeypatch) -> None:
    graph = nx.path_graph(160).to_directed()
    monkeypatch.setenv("SPECTRAL_SPARSE_SOLVER", "shift_invert")
    actual, solver = _approximate_sparse_spectrum(graph, 32)
    adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float64, format="csr")
    adjacency = adjacency + adjacency.T
    adjacency.data[:] = 1.0
    expected = np.linalg.eigvalsh(csgraph.laplacian(adjacency, normed=True).toarray())[:32]
    assert solver == "shift_invert"
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
