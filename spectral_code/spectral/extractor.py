import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from spectral_code.spectral.factory import create_spectral_analyzer

try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:  # pragma: no cover - optional runtime guard
    threadpool_limits = None


DEFAULT_MAX_NODES = int(os.getenv("SPECTRAL_MAX_NODES", "2000"))
# The baseline suite uses a fixed 128-value spectral window. Computing 300
# sparse eigenvalues for oversized graphs spent most of the dataset-build time
# beyond that consumer contract. Existing 300-value records remain compatible
# because packaging takes their leading 128 values.
DEFAULT_APPROX_TOPK = int(os.getenv("SPECTRAL_APPROX_TOPK", "128"))


def spectral_worker_settings() -> tuple[int, int]:
    """Return method-level workers and BLAS threads used by each worker.

    Dense eigensolvers normally claim every logical CPU through BLAS.  That is
    inefficient for the many small and medium code graphs in these datasets.
    Running several independent methods concurrently with one BLAS thread per
    task preserves the exact solver while avoiding nested oversubscription.
    """
    # One method worker was fastest on the Ryzen/OpenBLAS production host once
    # nested BLAS fan-out was disabled. Extra workers remain opt-in because
    # ARPACK's sparse path does not scale reliably with Python threads.
    workers = max(1, int(os.getenv("SPECTRAL_WORKERS", "1")))
    blas_threads = max(1, int(os.getenv("SPECTRAL_BLAS_THREADS", "1")))
    return workers, blas_threads


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

    tolerance = float(os.getenv("SPECTRAL_APPROX_TOL", "1e-3"))
    max_iterations = int(os.getenv("SPECTRAL_APPROX_MAXITER", "5000"))
    solver = os.getenv("SPECTRAL_SPARSE_SOLVER", "shift_invert").strip().lower()
    if solver not in {"shift_invert", "smallest_magnitude"}:
        raise ValueError(
            "SPECTRAL_SPARSE_SOLVER must be 'shift_invert' or 'smallest_magnitude'"
        )

    if solver == "shift_invert":
        # Asking ARPACK for ``which=SM`` converges extremely slowly on the
        # many near-zero eigenvalues of large source-code graphs.  A tiny
        # positive shift obtains the same low-frequency end of the normalized
        # Laplacian spectrum through a sparse factorization.  On representative
        # 38k--90k node CodeNet layers this was 333x--1175x faster than the old
        # k=300 path, while remaining inside the existing 1e-3 tolerance.
        try:
            eigvals = eigsh(
                laplacian,
                k=effective_k,
                sigma=float(os.getenv("SPECTRAL_SHIFT_INVERT_SIGMA", "1e-6")),
                which="LM",
                return_eigenvectors=False,
                tol=tolerance,
                maxiter=max_iterations,
            )
            solver_used = "shift_invert"
        except Exception:
            # Some singular/pathological sparse factorizations can fail.  Keep
            # the proven ARPACK path as a correctness-preserving per-layer
            # fallback instead of dropping that layer from the release.
            eigvals = eigsh(
                laplacian,
                k=effective_k,
                which="SM",
                return_eigenvectors=False,
                tol=tolerance,
                maxiter=max_iterations,
            )
            solver_used = "smallest_magnitude_fallback"
    else:
        eigvals = eigsh(
            laplacian,
            k=effective_k,
            which="SM",
            return_eigenvectors=False,
            tol=tolerance,
            maxiter=max_iterations,
        )
        solver_used = "smallest_magnitude"
    eigvals = np.real(eigvals).astype(np.float64)
    return np.sort(eigvals), solver_used


def _extract_method_features(item, graph_types: tuple[str, ...], mode: str):
    method_id, layers = item
    analyzer = create_spectral_analyzer(
        mode=mode,
        solver_name="dense",
        k=0,
        # The stored graph representation contains eigenvalues only.  Avoid
        # materialising a full N x N eigenvector matrix for every graph while
        # preserving the exact same dense spectrum.
        solver_kwargs={"eigenvalues_only": True},
        spectral_kwargs={}
    )
    
    method_features = {}
    layer_counts = {gtype: 0 for gtype in graph_types}
    layer_durations = {gtype: 0.0 for gtype in graph_types}
    layer_node_sums = {gtype: 0 for gtype in graph_types}
    layer_skipped = {gtype: 0 for gtype in graph_types}
    layer_approx = {gtype: 0 for gtype in graph_types}

    for gtype in graph_types:
        graph = layers.get(gtype)
        if graph is not None and graph.number_of_nodes() > 0:
            nodes_count = graph.number_of_nodes()
            layer_node_sums[gtype] += nodes_count

            if nodes_count > DEFAULT_MAX_NODES:
                t0 = time.perf_counter()
                try:
                    approx_eigs, sparse_solver = _approximate_sparse_spectrum(
                        graph, DEFAULT_APPROX_TOPK
                    )
                    elapsed_time = time.perf_counter() - t0
                    method_features[gtype] = {
                        "eigenvalues": approx_eigs,
                        "compute_time_seconds": elapsed_time,
                        "status": "ok_sparse_topk",
                        "nodes": nodes_count,
                        "max_dense_nodes": DEFAULT_MAX_NODES,
                        "top_k": len(approx_eigs),
                        "sparse_solver": sparse_solver,
                    }
                    layer_counts[gtype] += 1
                    layer_durations[gtype] += elapsed_time
                    layer_approx[gtype] += 1
                except Exception:
                    layer_skipped[gtype] += 1
                    method_features[gtype] = {
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
                method_features[gtype] = {
                    "eigenvalues": raw_eigs,
                    "compute_time_seconds": elapsed_time,
                    "status": "ok",
                    "nodes": nodes_count,
                }
                layer_counts[gtype] += 1
                layer_durations[gtype] += elapsed_time
            except Exception:
                method_features[gtype] = {
                    "eigenvalues": np.array([], dtype=np.float64),
                    "compute_time_seconds": 0.0,
                    "status": "failed_exception",
                    "nodes": nodes_count,
                }
        else:
            method_features[gtype] = {
                "eigenvalues": np.array([], dtype=np.float64),
                "compute_time_seconds": 0.0,
                "status": "missing_or_empty_graph",
                "nodes": 0,
            }

    return (
        method_id,
        method_features,
        layer_counts,
        layer_durations,
        layer_node_sums,
        layer_skipped,
        layer_approx,
    )


def extract_all_spectral_features(
    graph_db: dict,
    graph_types: list[str],
    mode: str = "normalized_laplacian",
    *,
    show_progress: bool = True,
):
    features_db = {}
    layer_counts = {gtype: 0 for gtype in graph_types}
    layer_durations = {gtype: 0.0 for gtype in graph_types}
    layer_node_sums = {gtype: 0 for gtype in graph_types}
    layer_skipped = {gtype: 0 for gtype in graph_types}
    layer_approx = {gtype: 0 for gtype in graph_types}
    workers, blas_threads = spectral_worker_settings()
    worker = partial(_extract_method_features, graph_types=tuple(graph_types), mode=mode)
    limit_context = (
        threadpool_limits(limits=blas_threads)
        if threadpool_limits is not None
        else nullcontext()
    )

    with limit_context:
        if workers == 1 or len(graph_db) <= 1:
            results = map(worker, graph_db.items())
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spectral")
            results = executor.map(worker, graph_db.items())
        try:
            for result in tqdm(
                results,
                total=len(graph_db),
                desc="Processing Methods",
                unit="method",
                disable=not show_progress,
            ):
                method_id, features, counts, durations, node_sums, skipped, approx = result
                features_db[method_id] = features
                for gtype in graph_types:
                    layer_counts[gtype] += counts[gtype]
                    layer_durations[gtype] += durations[gtype]
                    layer_node_sums[gtype] += node_sums[gtype]
                    layer_skipped[gtype] += skipped[gtype]
                    layer_approx[gtype] += approx[gtype]
        except BaseException:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                executor = None
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    return features_db, layer_counts, layer_durations, layer_node_sums, layer_skipped, layer_approx
