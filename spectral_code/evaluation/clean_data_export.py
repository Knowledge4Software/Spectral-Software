"""Export any prepared dataset as the compact, final clean-data format."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
import shutil
import zipfile
from pathlib import Path

from spectral_code.utils.artifact_cleanup import cleanup_finalized_pipeline_artifacts


SPLITS = ("train", "valid", "test")
CLEAN_DATA_REQUIRED_FILES = (
    "codes.jsonl.gz",
    "pairs.csv.gz",
    "graph_spectra.jsonl.gz",
    "metadata.json",
    "README.md",
)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_shards(manifest_path: Path) -> list[Path]:
    manifest = _read_json(manifest_path)
    paths = []
    for raw in manifest.get("shards", []):
        path = Path(raw)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest shard: {path}")
        paths.append(path)
    return paths


def _read_splits(data_dir: Path) -> tuple[dict[str, list[tuple[int, int, int]]], set[int]]:
    splits: dict[str, list[tuple[int, int, int]]] = {}
    code_ids: set[int] = set()
    for split in SPLITS:
        path = data_dir / f"{split}.txt"
        if not path.exists():
            continue
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    raise ValueError(f"Malformed row {line_number} in {path}")
                left, right, label = map(int, parts)
                rows.append((left, right, label))
                code_ids.update((left, right))
        splits[split] = rows
    if "train" not in splits:
        raise FileNotFoundError(f"Required split not found: {data_dir / 'train.txt'}")
    return splits, code_ids


def _read_codes(data_path: Path, needed_ids: set[int]) -> dict[int, str]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing prepared code file: {data_path}")
    codes: dict[int, str] = {}
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            method_id = int(row["idx"])
            if method_id in needed_ids:
                codes[method_id] = row["func"]
    missing = needed_ids - set(codes)
    if missing:
        raise RuntimeError(f"{len(missing):,} pair-referenced methods are missing from {data_path}.")
    return codes


def _graph_to_sparse_adjacency(graph: object, *, include_node_labels: bool = True) -> dict:
    """Serialize only adjacency fields consumed by the benchmark suite."""
    if graph is None:
        adjacency = {
            "num_nodes": 0,
            "num_edges": 0,
            "node_ids": [],
            "node_types": [],
            "row": [],
            "col": [],
        }
        if include_node_labels:
            adjacency["node_labels"] = []
        return adjacency
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    rows, cols = [], []
    for source, target in graph.edges():
        rows.append(index[source])
        cols.append(index[target])
    adjacency = {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "node_ids": [str(node) for node in nodes],
        "node_types": [str(graph.nodes[node].get("type", "<unknown>")) for node in nodes],
        "row": rows,
        "col": cols,
    }
    if include_node_labels:
        adjacency["node_labels"] = [str(graph.nodes[node].get("label", "")) for node in nodes]
    return adjacency


def write_clean_codes(output_dir: Path, codes: dict[int, str]) -> None:
    """Write code records using the portable clean-data schema."""
    output_path = output_dir / "codes.jsonl.gz"
    temporary = output_path.with_name(output_path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as dst:
        for method_id in sorted(codes):
            code = codes[method_id]
            dst.write(
                json.dumps(
                    {
                        "code_id": str(method_id),
                        "code": code,
                        "line_count": code.count("\n") + 1,
                        "char_count": len(code),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(output_path)


def validate_clean_data_files(output_dir: Path) -> None:
    """Fail before archiving if a portable clean-data export is incomplete."""
    missing = [name for name in CLEAN_DATA_REQUIRED_FILES if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Clean-data export is missing required files: {missing}")


def _pair_metadata_value(value: object) -> object:
    """Convert structured provenance to a stable CSV value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def write_clean_pairs(
    output_dir: Path,
    splits: dict[str, list[tuple[int, int, int]]],
    pair_metadata: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, int]:
    """Write labelled pairs and optional aligned construction provenance.

    The four core columns remain identical for every dataset. Optional source
    fields are retained so targeted evaluations (for example, mutation or
    injection negatives) do not have to infer pair construction from code.
    """
    counts = {split: len(rows) for split, rows in splits.items()}
    pair_metadata = pair_metadata or {}
    for split, metadata_rows in pair_metadata.items():
        expected = len(splits.get(split, []))
        if len(metadata_rows) != expected:
            raise ValueError(
                f"Pair metadata for {split} has {len(metadata_rows):,} rows; expected {expected:,}."
            )
    core_fields = ["split", "left_id", "right_id", "label", "label_name"]
    metadata_fields = sorted(
        {
            str(field)
            for rows in pair_metadata.values()
            for record in rows
            for field in record
            if str(field) not in core_fields
        }
    )
    with gzip.open(output_dir / "pairs.csv.gz", "wt", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=core_fields + metadata_fields)
        writer.writeheader()
        for split in SPLITS:
            metadata_rows = pair_metadata.get(split, [{} for _ in splits.get(split, [])])
            for (left, right, label), metadata in zip(splits.get(split, []), metadata_rows):
                row = {
                    "split": split,
                    "left_id": left,
                    "right_id": right,
                    "label": label,
                    "label_name": "clone" if label else "non_clone",
                }
                row.update(
                    {
                        str(field): _pair_metadata_value(value)
                        for field, value in metadata.items()
                        if str(field) in metadata_fields
                    }
                )
                writer.writerow(row)
    return counts


def create_clean_data_zip(output_dir: Path, zip_path: Path) -> Path:
    """Create the Kaggle-uploadable archive while preserving the clean_data directory."""
    validate_clean_data_files(output_dir)
    # The large corpus payloads are already gzip-compressed.  Deflating them a
    # second time costs many minutes for negligible savings and can make a
    # Kaggle-ready CodeNet release time out before its ZIP is finalized.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                compression = zipfile.ZIP_STORED if path.suffix.lower() == ".gz" else zipfile.ZIP_DEFLATED
                archive.write(path, path.relative_to(output_dir.parent), compress_type=compression)
    return zip_path


def _feature_for(features: dict, method_id: object, graph_type: str) -> dict:
    method_features = features.get(method_id) or features.get(str(method_id)) or features.get(int(method_id)) or {}
    result = method_features.get(graph_type, {}) if isinstance(method_features, dict) else {}
    return result if isinstance(result, dict) else {}


def _rounded_eigenvalues(feature: dict, precision: int | None) -> list[float]:
    """Convert list- or NumPy-backed eigenvalues without boolean coercion."""
    values = feature.get("eigenvalues")
    if values is None:
        return []
    try:
        iterator = iter(values)
    except TypeError:
        iterator = iter((values,))
    return [
        float(value) if precision is None else round(float(value), precision)
        for value in iterator
    ]


def _export_graph_spectra(
    output_root: Path,
    output_path: Path,
    needed_ids: set[int],
    graph_types: list[str],
    precision: int | None,
) -> dict[str, int]:
    graph_manifest = output_root / "clean_graphs" / "graph_shards_manifest.json"
    feature_manifest = output_root / "spectral_features" / "spectral_features_manifest.json"
    if not graph_manifest.exists() or not feature_manifest.exists():
        raise FileNotFoundError("Clean graph and spectral manifests are required before final export.")
    graph_shards = _resolve_shards(graph_manifest)
    feature_shards = _resolve_shards(feature_manifest)
    if len(graph_shards) != len(feature_shards):
        raise RuntimeError("Graph and spectral manifests have different shard counts.")

    written: set[int] = set()
    layers = 0
    missing_features = 0
    temporary = output_path.with_name(output_path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as dst:
        for graph_path, feature_path in zip(graph_shards, feature_shards):
            with graph_path.open("rb") as f:
                graph_shard = pickle.load(f)
            with feature_path.open("rb") as f:
                feature_shard = pickle.load(f)
            for raw_id, graph_layers in graph_shard.items():
                method_id = int(raw_id)
                if method_id not in needed_ids or method_id in written:
                    continue
                exported = {}
                for graph_type in graph_types:
                    feature = _feature_for(feature_shard, raw_id, graph_type)
                    eigenvalues = _rounded_eigenvalues(feature, precision)
                    if not eigenvalues:
                        missing_features += 1
                    graph = graph_layers.get(graph_type) if isinstance(graph_layers, dict) else None
                    exported[graph_type] = {
                        "adjacency": _graph_to_sparse_adjacency(
                            graph, include_node_labels=graph_type == "ast"
                        ),
                        "eigenvalues": eigenvalues,
                        "spectral_status": feature.get("status", "missing"),
                    }
                    layers += 1
                dst.write(json.dumps({"code_id": str(method_id), "graphs": exported}, separators=(",", ":")) + "\n")
                written.add(method_id)
    missing = needed_ids - written
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing):,} pair-referenced methods have no cleaned graph.")
    temporary.replace(output_path)
    return {"methods": len(written), "graph_layers": layers, "missing_feature_layers": missing_features}


def export_graph_spectra_from_sources(
    source_output_roots: list[Path],
    output_path: Path,
    needed_ids: set[int],
    graph_types: list[str] | tuple[str, ...],
    precision: int | None,
) -> dict[str, int]:
    """Export selected graph/spectral records from one or more extracted datasets."""
    written: set[int] = set()
    graph_layers = 0
    missing_features = 0
    temporary = output_path.with_name(output_path.name + ".tmp")

    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as dst:
        for source_output_root in source_output_roots:
            graph_manifest = source_output_root / "clean_graphs" / "graph_shards_manifest.json"
            feature_manifest = source_output_root / "spectral_features" / "spectral_features_manifest.json"
            if not graph_manifest.exists() or not feature_manifest.exists():
                raise FileNotFoundError(f"Missing graph/spectral manifests under {source_output_root}.")
            graph_shards = _resolve_shards(graph_manifest)
            feature_shards = _resolve_shards(feature_manifest)
            if len(graph_shards) != len(feature_shards):
                raise RuntimeError(f"Graph/spectral shard mismatch under {source_output_root}.")

            for graph_path, feature_path in zip(graph_shards, feature_shards):
                with graph_path.open("rb") as f:
                    graph_shard = pickle.load(f)
                with feature_path.open("rb") as f:
                    feature_shard = pickle.load(f)
                for raw_id, layers in graph_shard.items():
                    method_id = int(raw_id)
                    if method_id not in needed_ids or method_id in written:
                        continue
                    exported = {}
                    for graph_type in graph_types:
                        feature = _feature_for(feature_shard, raw_id, graph_type)
                        eigenvalues = _rounded_eigenvalues(feature, precision)
                        if not eigenvalues:
                            missing_features += 1
                        graph = layers.get(graph_type) if isinstance(layers, dict) else None
                        exported[graph_type] = {
                            "adjacency": _graph_to_sparse_adjacency(
                                graph, include_node_labels=graph_type == "ast"
                            ),
                            "eigenvalues": eigenvalues,
                            "spectral_status": feature.get("status", "missing"),
                        }
                        graph_layers += 1
                    dst.write(json.dumps({"code_id": str(method_id), "graphs": exported}, separators=(",", ":")) + "\n")
                    written.add(method_id)

    missing = needed_ids - written
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing):,} selected methods have no cleaned graph.")
    temporary.replace(output_path)
    return {"methods": len(written), "graph_layers": graph_layers, "missing_feature_layers": missing_features}


def export_clean_dataset(
    *,
    data_dir: Path,
    output_root: Path,
    output_dir: Path | None = None,
    graph_types: list[str],
    float_precision: int | None = 8,
    cleanup_intermediates: bool = True,
    overwrite: bool = True,
    create_zip: bool = False,
) -> dict[str, object]:
    """Create codes/pairs/graph-spectra files and optionally remove regenerable artefacts."""
    data_dir = data_dir.resolve()
    output_root = output_root.resolve()
    output_dir = (output_dir or output_root / "clean_data").resolve()
    if output_root not in output_dir.parents:
        raise ValueError("Final clean-data output must be inside the dataset output root.")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Final clean-data output already exists: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits, needed_ids = _read_splits(data_dir)
    codes = _read_codes(data_dir / "data.jsonl", needed_ids)
    write_clean_codes(output_dir, codes)
    pair_counts = write_clean_pairs(output_dir, splits)
    label_counts = {
        "clone": sum(label == 1 for rows in splits.values() for _, _, label in rows),
        "non_clone": sum(label == 0 for rows in splits.values() for _, _, label in rows),
    }

    graph_summary = _export_graph_spectra(output_root, output_dir / "graph_spectra.jsonl.gz", needed_ids, graph_types, float_precision)
    cleanup = cleanup_finalized_pipeline_artifacts(output_root, compute_size=True) if cleanup_intermediates else {"skipped": True}
    metadata = {
        "format": "spectral_clean_data_v1",
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "graph_types": graph_types,
        "float_precision": "full" if float_precision is None else float_precision,
        "counts": {"codes": len(codes), "pairs": {**pair_counts, **label_counts, "total": sum(pair_counts.values())}, "graph_spectra": graph_summary},
        "cleanup_after_export": cleanup,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Clean dataset output\n\nThis directory is the final pipeline artefact: source code, labelled pairs, and sparse graph/spectral records. Regenerable Joern, graph, and spectral shard artefacts were removed after this export.\n",
        encoding="utf-8",
    )
    if create_zip:
        zip_path = output_dir.parent / f"{output_dir.name}.zip"
        create_clean_data_zip(output_dir, zip_path)
        metadata["zip"] = str(zip_path)
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
