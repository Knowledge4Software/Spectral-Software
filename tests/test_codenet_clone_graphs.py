from __future__ import annotations

import gzip
import json
import pickle
import zipfile
from pathlib import Path

import pytest

from spectral_code.evaluation.codenet_clone_graphs import (
    CloneGraphPaths,
    _connect_cache,
    _iter_cached_target_records,
    _manifest_has_methods,
    _open_clean_bundle,
    _raw_stage_complete,
    _safe_remove_batch_work,
    _spectral_manifest_complete,
    _write_cache_shard,
    cache_audit,
    merge_cache_from_cache,
    recover_completed_batch_work,
    seed_cache_from_clean_bundle,
)


GRAPH_TYPES = ("ast", "cfg", "ddg", "cpg")


def _graph_record(code_id: str) -> dict:
    graph = {
        "adjacency": {
            "format": "adjacency_coo_v1",
            "directed": True,
            "num_nodes": 1,
            "num_edges": 0,
            "node_ids": ["n0"],
            "node_types": ["METHOD"],
            "node_labels": ["main"],
            "row": [],
            "col": [],
        },
        "eigenvalues": [0.0],
        "eigenvalue_count": 1,
        "spectral_status": "exact",
    }
    return {"code_id": code_id, "graphs": {kind: graph for kind in GRAPH_TYPES}}


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_reuse_uses_stable_source_id_and_remaps_numeric_id(tmp_path: Path) -> None:
    reusable = tmp_path / "old" / "clean_data"
    reusable.mkdir(parents=True)
    _write_gzip_jsonl(
        reusable / "codes.jsonl.gz",
        [{"code_id": "7", "source_code_id": "source-a", "source_sha256": "hash-a"}],
    )
    _write_gzip_jsonl(reusable / "graph_spectra.jsonl.gz", [_graph_record("7")])

    paths = CloneGraphPaths(tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "output")
    targets = {
        "source-a": {"code_id": 99, "language": "python", "source_sha256": "hash-a"},
        "source-b": {"code_id": 100, "language": "java", "source_sha256": "hash-b"},
    }
    connection = _connect_cache(paths, GRAPH_TYPES, "archive-hash")
    try:
        assert seed_cache_from_clean_bundle(paths, connection, targets, reusable, GRAPH_TYPES) == 1
        assert seed_cache_from_clean_bundle(paths, connection, targets, reusable, GRAPH_TYPES) == 0
        assert cache_audit(connection, targets) == {
            "target_codes": 2,
            "cached_matching_codes": 1,
            "remaining_codes": 1,
            "remaining_by_language": {"python": 0, "java": 1, "cpp": 0, "csharp": 0},
        }
        exported = list(_iter_cached_target_records(paths, connection, {"source-a": targets["source-a"]}, GRAPH_TYPES))
        assert len(exported) == 1
        assert exported[0]["code_id"] == "99"
        assert set(exported[0]["graphs"]) == set(GRAPH_TYPES)
        assert "format" not in exported[0]["graphs"]["ast"]["adjacency"]
        assert "directed" not in exported[0]["graphs"]["ast"]["adjacency"]
        assert "eigenvalue_count" not in exported[0]["graphs"]["ast"]
        assert "node_labels" in exported[0]["graphs"]["ast"]["adjacency"]
        assert "node_labels" not in exported[0]["graphs"]["cfg"]["adjacency"]
    finally:
        connection.close()


def test_reuse_rejects_same_source_id_with_changed_source_hash(tmp_path: Path) -> None:
    reusable = tmp_path / "clean_data"
    reusable.mkdir()
    _write_gzip_jsonl(
        reusable / "codes.jsonl.gz",
        [{"code_id": "1", "source_code_id": "source-a", "source_sha256": "old-hash"}],
    )
    _write_gzip_jsonl(reusable / "graph_spectra.jsonl.gz", [_graph_record("1")])
    paths = CloneGraphPaths(tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "output")
    targets = {"source-a": {"code_id": 1, "language": "cpp", "source_sha256": "new-hash"}}
    connection = _connect_cache(paths, GRAPH_TYPES, "archive-hash")
    try:
        assert seed_cache_from_clean_bundle(paths, connection, targets, reusable, GRAPH_TYPES) == 0
        assert cache_audit(connection, targets)["remaining_codes"] == 1
    finally:
        connection.close()


def test_packaging_caps_cached_spectra_at_consumer_limit(tmp_path: Path) -> None:
    paths = CloneGraphPaths(tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "output")
    targets = {"source-a": {"code_id": 1, "language": "python", "source_sha256": "hash-a"}}
    graphs = _graph_record("1")["graphs"]
    for layer in graphs.values():
        layer["eigenvalues"] = [float(value) for value in range(180)]
    connection = _connect_cache(paths, GRAPH_TYPES, "archive-hash")
    try:
        assert _write_cache_shard(
            paths,
            connection,
            [{"source_code_id": "source-a", "source_sha256": "hash-a", "graphs": graphs}],
            shard_name="long-spectrum.jsonl.gz",
            graph_types=GRAPH_TYPES,
        ) == 1
        record = next(_iter_cached_target_records(paths, connection, targets, GRAPH_TYPES))
        assert all(len(layer["eigenvalues"]) == 128 for layer in record["graphs"].values())
    finally:
        connection.close()


def test_clean_bundle_zip_streams_nested_clean_data(tmp_path: Path) -> None:
    archive_path = tmp_path / "bundle.zip"
    code_payload = gzip.compress((json.dumps({"code_id": "1"}) + "\n").encode())
    graph_payload = gzip.compress((json.dumps(_graph_record("1")) + "\n").encode())
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("clean_data/codes.jsonl.gz", code_payload)
        archive.writestr("clean_data/graph_spectra.jsonl.gz", graph_payload)
    with _open_clean_bundle(archive_path) as (codes, graphs):
        assert json.loads(next(codes))["code_id"] == "1"
        assert json.loads(next(graphs))["code_id"] == "1"


def test_batch_cleanup_cannot_escape_work_root(tmp_path: Path) -> None:
    work_root = tmp_path / "output" / "_batch_work"
    work_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside"):
        _safe_remove_batch_work(work_root, work_root)
    with pytest.raises(ValueError, match="outside"):
        _safe_remove_batch_work(tmp_path / "elsewhere", work_root)


def test_clone_subset_can_share_the_existing_graph_cache(tmp_path: Path) -> None:
    shared = tmp_path / "shared-cache"
    paths = CloneGraphPaths(
        tmp_path / "source.zip",
        tmp_path / "prepared",
        tmp_path / "subset-output",
        shared,
        "subset.zip",
    )
    assert paths.cache_dir == shared
    assert paths.cache_index == shared / "index.sqlite3"
    assert paths.zip_path == tmp_path / "subset-output" / "subset.zip"


def test_disjoint_machine_caches_merge_records_without_copying_sqlite_index(tmp_path: Path) -> None:
    source_paths = CloneGraphPaths(tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "source-output")
    destination_paths = CloneGraphPaths(
        tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "destination-output"
    )
    targets = {"source-a": {"code_id": 99, "language": "python", "source_sha256": "hash-a"}}
    source = _connect_cache(source_paths, GRAPH_TYPES, "archive-hash")
    try:
        assert _write_cache_shard(
            source_paths,
            source,
            [{"source_code_id": "source-a", "source_sha256": "hash-a", "graphs": _graph_record("1")["graphs"]}],
            shard_name="source.jsonl.gz",
            graph_types=GRAPH_TYPES,
        ) == 1
    finally:
        source.close()
    destination = _connect_cache(destination_paths, GRAPH_TYPES, "archive-hash")
    try:
        assert merge_cache_from_cache(
            destination_paths, destination, targets, source_paths.cache_dir, GRAPH_TYPES
        ) == 1
        assert cache_audit(destination, targets)["remaining_codes"] == 0
        assert merge_cache_from_cache(
            destination_paths, destination, targets, source_paths.cache_dir, GRAPH_TYPES
        ) == 0
    finally:
        destination.close()


def test_merge_skips_sparse_records_from_an_old_64_value_run(tmp_path: Path) -> None:
    source_paths = CloneGraphPaths(tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "source-output")
    destination_paths = CloneGraphPaths(
        tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "destination-output"
    )
    targets = {"source-a": {"code_id": 99, "language": "csharp", "source_sha256": "hash-a"}}
    graphs = _graph_record("1")["graphs"]
    graphs["cpg"] = dict(graphs["cpg"])
    graphs["cpg"]["spectral_status"] = "ok_sparse_topk"
    graphs["cpg"]["eigenvalues"] = [0.0] * 64
    source = _connect_cache(source_paths, GRAPH_TYPES, "archive-hash")
    try:
        _write_cache_shard(
            source_paths,
            source,
            [{"source_code_id": "source-a", "source_sha256": "hash-a", "graphs": graphs}],
            shard_name="old-k64.jsonl.gz",
            graph_types=GRAPH_TYPES,
        )
    finally:
        source.close()

    destination = _connect_cache(destination_paths, GRAPH_TYPES, "archive-hash")
    try:
        assert merge_cache_from_cache(
            destination_paths,
            destination,
            targets,
            source_paths.cache_dir,
            GRAPH_TYPES,
        ) == 0
        assert cache_audit(destination, targets)["remaining_codes"] == 1
    finally:
        destination.close()


def test_interrupted_batch_stage_markers_are_validated(tmp_path: Path) -> None:
    output = tmp_path / "output"
    raw = output / "dataset_features"
    raw.mkdir(parents=True)
    for index in range(3):
        (raw / f"{index}.json").write_text("{}", encoding="utf-8")
    (output / "timing_stats.json").write_text(
        json.dumps({
            "total_methods": 3,
            "total_raw_extraction_time": 1.2,
            "dot_mapped_ast": 3,
            "dot_mapped_cfg": 3,
            "dot_mapped_ddg": 3,
        }),
        encoding="utf-8",
    )
    assert _raw_stage_complete(output, 3)
    assert not _raw_stage_complete(output, 4)

    graph_manifest = output / "clean_graphs" / "graph_shards_manifest.json"
    graph_manifest.parent.mkdir()
    graph_manifest.write_text(
        json.dumps({"total_methods": 3, "total_base_layers_cleaned": 9}),
        encoding="utf-8",
    )
    assert _manifest_has_methods(graph_manifest, 3)

    feature_shard = output / "spectral_features" / "features.pkl"
    feature_shard.parent.mkdir()
    feature_shard.write_bytes(pickle.dumps({}))
    spectral_manifest = feature_shard.parent / "spectral_features_manifest.json"
    spectral_manifest.write_text(
        json.dumps({
            "total_methods": 3,
            "graph_types": list(GRAPH_TYPES),
            "approx_top_k": 128,
            "shards": [str(feature_shard)],
        }),
        encoding="utf-8",
    )
    assert _spectral_manifest_complete(spectral_manifest, 3, GRAPH_TYPES)
    feature_shard.unlink()
    assert not _spectral_manifest_complete(spectral_manifest, 3, GRAPH_TYPES)


def test_recovery_reads_jsonl_with_unicode_line_separator_inside_source(tmp_path: Path) -> None:
    paths = CloneGraphPaths(tmp_path / "source.zip", tmp_path / "prepared", tmp_path / "output")
    batch = paths.work_dir / "java_batch"
    input_dir = batch / "input"
    input_dir.mkdir(parents=True)
    record = {"idx": 1, "source_code_id": "java:1", "func": "class Main { // a\u2028b\n }"}
    (input_dir / "data.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    connection = _connect_cache(paths, GRAPH_TYPES, "archive-hash")
    try:
        # The incomplete batch is deliberately skipped, but parsing its JSONL
        # must not mistake U+2028 inside source text for a record delimiter.
        report = recover_completed_batch_work(paths, connection, GRAPH_TYPES, [paths.work_dir])
        assert report["incomplete_skipped"] == 1
    finally:
        connection.close()
