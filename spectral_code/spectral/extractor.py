import os
import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from spectral_code.spectral.factory import create_spectral_analyzer


DEFAULT_MAX_NODES = int(os.getenv("SPECTRAL_MAX_NODES", "2000"))
DEFAULT_APPROX_TOPK = int(os.getenv("SPECTRAL_APPROX_TOPK", "300"))


def _approximate_sparse_spectrum(graph, k: int):
    """Approximate large-graph spectra without materializing dense NxN matrices."""
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return np.array([], dtype=np.float64)

    if node_count <= 2:
        return np.zeros(node_count, dtype=np.float64)

    effective_k = min(k, node_count - 2)
    if effective_k <= 0:
        return np.array([], dtype=np.float64)

    nodelist = list(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodelist,
        dtype=np.float64,
        weight=None,
        format="csr",
    )

    if graph.is_directed():
        adjacency = adjacency + adjacency.T
        adjacency.data[:] = 1.0
        adjacency.eliminate_zeros()

    laplacian = csgraph.laplacian(adjacency, normed=True)
    laplacian = sp.csr_matrix(laplacian, dtype=np.float64)

    eigvals = eigsh(
        laplacian,
        k=effective_k,
        which="SM",
        return_eigenvectors=False,
        tol=float(os.getenv("SPECTRAL_APPROX_TOL", "1e-3")),
        maxiter=int(os.getenv("SPECTRAL_APPROX_MAXITER", "5000")),
    )
    eigvals = np.real(eigvals).astype(np.float64)
    return np.sort(eigvals)


def extract_all_spectral_features(graph_db: dict, graph_types: list[str], mode: str = "normalized_laplacian"):
    analyzer = create_spectral_analyzer(
        mode=mode,
        solver_name="dense",
        k=0,
        solver_kwargs={},
        spectral_kwargs={}
    )
    
    features_db = {}
    layer_counts = {gtype: 0 for gtype in graph_types}
    layer_durations = {gtype: 0.0 for gtype in graph_types}
    layer_node_sums = {gtype: 0 for gtype in graph_types}
    layer_skipped = {gtype: 0 for gtype in graph_types}
    layer_approx = {gtype: 0 for gtype in graph_types}

    for method_id, layers in tqdm(graph_db.items(), desc="Processing Methods", unit="method"):
        features_db[method_id] = {}
        for gtype in graph_types:
            graph = layers.get(gtype)
            if graph is not None and graph.number_of_nodes() > 0:
                nodes_count = graph.number_of_nodes()
                layer_node_sums[gtype] += nodes_count

                if nodes_count > DEFAULT_MAX_NODES:
                    t0 = time.perf_counter()
                    try:
                        approx_eigs = _approximate_sparse_spectrum(graph, DEFAULT_APPROX_TOPK)
                        elapsed_time = time.perf_counter() - t0
                        features_db[method_id][gtype] = {
                            "eigenvalues": approx_eigs,
                            "compute_time_seconds": elapsed_time,
                            "status": "ok_sparse_topk",
                            "nodes": nodes_count,
                            "max_dense_nodes": DEFAULT_MAX_NODES,
                            "top_k": len(approx_eigs),
                        }
                        layer_counts[gtype] += 1
                        layer_durations[gtype] += elapsed_time
                        layer_approx[gtype] += 1
                    except Exception:
                        layer_skipped[gtype] += 1
                        features_db[method_id][gtype] = {
                            "eigenvalues": np.array([], dtype=np.float64),
                            "compute_time_seconds": 0.0,
                            "status": "failed_sparse_topk",
                            "nodes": nodes_count,
                            "max_dense_nodes": DEFAULT_MAX_NODES,
                            "top_k_requested": DEFAULT_APPROX_TOPK,
                        }
                    continue
                
                t0 = time.perf_counter()
                try:
                    raw_eigs, _ = analyzer.analyze(graph)
                    elapsed_time = time.perf_counter() - t0
                    features_db[method_id][gtype] = {
                        "eigenvalues": raw_eigs,
                        "compute_time_seconds": elapsed_time,
                        "status": "ok",
                        "nodes": nodes_count,
                    }
                    layer_counts[gtype] += 1
                    layer_durations[gtype] += elapsed_time
                except Exception:
                    features_db[method_id][gtype] = {
                        "eigenvalues": np.array([], dtype=np.float64),
                        "compute_time_seconds": 0.0,
                        "status": "failed_exception",
                        "nodes": nodes_count,
                    }
            else:
                features_db[method_id][gtype] = {
                    "eigenvalues": np.array([], dtype=np.float64),
                    "compute_time_seconds": 0.0,
                    "status": "missing_or_empty_graph",
                    "nodes": 0,
                }
                
    return features_db, layer_counts, layer_durations, layer_node_sums, layer_skipped, layer_approx
