from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx

import spectral_code.spectral.extractor as spectral_extractor
from spectral_code.spectral.runner import run_spectral_feature_extraction


def test_existing_spectral_shard_counts_as_resumed_work(tmp_path: Path) -> None:
    graph_shard = tmp_path / "graphs.pkl"
    graph_shard.write_bytes(pickle.dumps({1: {}}))
    graph_manifest = tmp_path / "graph_shards_manifest.json"
    graph_manifest.write_text(
        json.dumps({
            "format": "cleaned_graph_shards_v1",
            "total_methods": 1,
            "shards": [str(graph_shard)],
        }),
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features"
    shard_dir = feature_dir / "spectral_feature_shards"
    shard_dir.mkdir(parents=True)
    feature = {kind: {"eigenvalues": [0.0], "status": "exact"} for kind in ("ast", "cfg", "ddg", "cpg")}
    (shard_dir / "features_000001.pkl").write_bytes(pickle.dumps({1: feature}))
    timing = tmp_path / "timing.json"
    timing.write_text("{}", encoding="utf-8")

    result = run_spectral_feature_extraction(
        str(graph_manifest),
        str(feature_dir),
        str(timing),
        ["ast", "cfg", "ddg", "cpg"],
        mode="directed_laplacian",
    )

    assert Path(result).is_file()
    stats = json.loads(timing.read_text(encoding="utf-8"))
    assert stats["total_methods_processed"] == 1
    assert stats["spectral_computed_graphs_ast"] == 1
    assert stats["spectral_computed_graphs_cpg"] == 1


def test_missing_spectral_shards_can_run_in_separate_processes(tmp_path: Path, monkeypatch) -> None:
    graph = nx.DiGraph([(0, 1), (1, 2)])
    for node in graph:
        graph.nodes[node].update(type="NODE", label=str(node))
    graph_shards = []
    for index in range(2):
        path = tmp_path / f"graphs_{index:06d}.pkl"
        path.write_bytes(
            pickle.dumps({index: {kind: graph for kind in ("ast", "cfg", "ddg", "cpg")}})
        )
        graph_shards.append(str(path))
    graph_manifest = tmp_path / "graph_shards_manifest.json"
    graph_manifest.write_text(
        json.dumps({
            "format": "cleaned_graph_shards_v1",
            "total_methods": 2,
            "shards": graph_shards,
        }),
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features"
    timing = tmp_path / "timing.json"
    timing.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SPECTRAL_SHARD_WORKERS", "2")
    monkeypatch.setenv("SPECTRAL_WORKERS", "1")
    monkeypatch.setenv("SPECTRAL_BLAS_THREADS", "1")

    result = run_spectral_feature_extraction(
        str(graph_manifest),
        str(feature_dir),
        str(timing),
        ["ast", "cfg", "ddg", "cpg"],
    )

    manifest = json.loads(Path(result).read_text(encoding="utf-8"))
    assert manifest["total_methods"] == 2
    assert manifest["shard_workers"] == 2
    assert all(Path(path).is_file() for path in manifest["shards"])


def test_sparse_shard_with_smaller_top_k_is_recomputed(tmp_path: Path, monkeypatch) -> None:
    graph = nx.path_graph(130, create_using=nx.DiGraph)
    for node in graph:
        graph.nodes[node].update(type="NODE", label=str(node))
    graph_shard = tmp_path / "graphs.pkl"
    graph_shard.write_bytes(
        pickle.dumps({1: {kind: graph for kind in ("ast", "cfg", "ddg", "cpg")}})
    )
    graph_manifest = tmp_path / "graph_shards_manifest.json"
    graph_manifest.write_text(
        json.dumps({
            "format": "cleaned_graph_shards_v1",
            "total_methods": 1,
            "shards": [str(graph_shard)],
        }),
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features"
    shard_dir = feature_dir / "spectral_feature_shards"
    shard_dir.mkdir(parents=True)
    stale = {
        kind: {"eigenvalues": [0.0] * 64, "status": "ok_sparse_topk", "nodes": 130}
        for kind in ("ast", "cfg", "ddg", "cpg")
    }
    (shard_dir / "features_000001.pkl").write_bytes(pickle.dumps({1: stale}))
    timing = tmp_path / "timing.json"
    timing.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("SPECTRAL_MAX_NODES", "10")
    monkeypatch.setenv("SPECTRAL_APPROX_TOPK", "128")
    monkeypatch.setenv("SPECTRAL_SHARD_WORKERS", "1")
    monkeypatch.setattr(spectral_extractor, "DEFAULT_MAX_NODES", 10)
    monkeypatch.setattr(spectral_extractor, "DEFAULT_APPROX_TOPK", 128)

    result = run_spectral_feature_extraction(
        str(graph_manifest),
        str(feature_dir),
        str(timing),
        ["ast", "cfg", "ddg", "cpg"],
    )

    manifest = json.loads(Path(result).read_text(encoding="utf-8"))
    recomputed = pickle.loads(Path(manifest["shards"][0]).read_bytes())
    assert len(recomputed[1]["ast"]["eigenvalues"]) == 128
