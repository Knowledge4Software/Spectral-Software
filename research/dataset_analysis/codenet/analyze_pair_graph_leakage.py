#!/usr/bin/env python3
"""Analyze endpoint reuse and random pair-split leakage in pair datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from calculate_clean_zip_hardness import canonical_language


RANDOM_SEED = 20260813
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_BITS = np.array([1, 2, 4], dtype=np.uint8)


@dataclass
class PairGraphData:
    dataset: str
    dataset_key: str
    source_zip: Path
    code_ids: list[str]
    languages: list[str]
    source_hashes: list[str]
    left: np.ndarray
    right: np.ndarray
    labels: np.ndarray
    pair_kinds: np.ndarray
    official_splits: np.ndarray | None
    source_metadata: dict


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0 or values.sum() == 0:
        return 0.0
    positions = np.arange(1, values.size + 1, dtype=np.float64)
    return float(
        np.dot(2 * positions - values.size - 1, values)
        / (values.size * values.sum())
    )


def degree_statistics(degrees: np.ndarray) -> dict:
    degrees = np.asarray(degrees, dtype=np.int64)
    active = degrees[degrees > 0]
    if active.size == 0:
        raise ValueError("The pair graph has no active node")
    descending = np.sort(active)[::-1]
    top_count = max(1, math.ceil(active.size * 0.01))
    result = {
        "all_code_count": int(degrees.size),
        "active_code_count": int(active.size),
        "isolated_code_count": int(np.sum(degrees == 0)),
        "minimum_all_codes": int(np.min(degrees)),
        "minimum_active": int(np.min(active)),
        "mean_active": float(np.mean(active, dtype=np.float64)),
        "std_active": float(np.std(active, dtype=np.float64)),
        "p25_active": float(np.quantile(active, 0.25)),
        "median_active": float(np.quantile(active, 0.50)),
        "p75_active": float(np.quantile(active, 0.75)),
        "p90_active": float(np.quantile(active, 0.90)),
        "p95_active": float(np.quantile(active, 0.95)),
        "p99_active": float(np.quantile(active, 0.99)),
        "maximum_active": int(np.max(active)),
        "gini_active": _gini(active),
        "top_1_percent_incidence_share": float(
            np.sum(descending[:top_count], dtype=np.int64)
            / np.sum(active, dtype=np.int64)
        ),
    }
    for threshold in (2, 5, 10, 20, 50, 100):
        count = int(np.sum(active >= threshold))
        result[f"degree_ge_{threshold}_count"] = count
        result[f"degree_ge_{threshold}_ratio"] = float(count / active.size)
    return result


def degree_arrays(
    left: np.ndarray,
    right: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return incidence degree, unique-neighbor degree, and unique edge arrays."""

    degree = np.bincount(left, minlength=node_count) + np.bincount(
        right, minlength=node_count
    )
    low = np.minimum(left, right).astype(np.uint64, copy=False)
    high = np.maximum(left, right).astype(np.uint64, copy=False)
    packed = (low << np.uint64(32)) | high
    unique_packed = np.unique(packed)
    unique_left = (unique_packed >> np.uint64(32)).astype(np.int32)
    unique_right = (unique_packed & np.uint64(0xFFFFFFFF)).astype(np.int32)
    unique_degree = np.bincount(unique_left, minlength=node_count) + np.bincount(
        unique_right, minlength=node_count
    )
    self_nodes = unique_left[unique_left == unique_right]
    if self_nodes.size:
        unique_degree[self_nodes] -= 1
    return (
        degree.astype(np.int32),
        unique_degree.astype(np.int32),
        unique_left,
        unique_right,
    )


def _node_split_masks(
    left: np.ndarray,
    right: np.ndarray,
    edge_splits: np.ndarray,
    node_count: int,
) -> np.ndarray:
    masks = np.zeros(node_count, dtype=np.uint8)
    bits = SPLIT_BITS[edge_splits]
    np.bitwise_or.at(masks, left, bits)
    np.bitwise_or.at(masks, right, bits)
    return masks


def _hash_split_statistics(
    source_hashes: Sequence[str],
    node_masks: np.ndarray,
) -> dict:
    masks_by_hash: dict[str, int] = {}
    for source_hash, mask in zip(source_hashes, node_masks):
        if not source_hash or mask == 0:
            continue
        masks_by_hash[source_hash] = masks_by_hash.get(source_hash, 0) | int(mask)
    counts = Counter(masks_by_hash.values())
    multi_count = sum(count for mask, count in counts.items() if int(mask).bit_count() >= 2)
    all_three = counts.get(7, 0)
    return {
        "unique_active_source_hashes": len(masks_by_hash),
        "source_hashes_in_multiple_splits": multi_count,
        "source_hashes_in_multiple_splits_ratio": (
            multi_count / len(masks_by_hash) if masks_by_hash else 0.0
        ),
        "source_hashes_in_all_three_splits": all_three,
        "source_hashes_in_all_three_splits_ratio": (
            all_three / len(masks_by_hash) if masks_by_hash else 0.0
        ),
    }


def split_leakage_statistics(
    left: np.ndarray,
    right: np.ndarray,
    edge_splits: np.ndarray,
    node_count: int,
    source_hashes: Sequence[str],
) -> tuple[dict, np.ndarray]:
    masks = _node_split_masks(left, right, edge_splits, node_count)
    active = masks > 0
    bit_counts = np.bitwise_count(masks)
    active_count = int(np.sum(active))
    multi_count = int(np.sum(bit_counts >= 2))
    all_three_count = int(np.sum(masks == 7))
    train_nodes = (masks & 1) != 0
    pair_overlap = {}
    for split_index, split_name in ((1, "validation"), (2, "test")):
        selected = edge_splits == split_index
        selected_count = int(np.sum(selected))
        if selected_count == 0:
            pair_overlap[split_name] = {
                "pair_count": 0,
                "either_endpoint_seen_in_train_ratio": None,
                "both_endpoints_seen_in_train_ratio": None,
                "endpoint_incidences_seen_in_train_ratio": None,
            }
            continue
        left_seen = train_nodes[left[selected]]
        right_seen = train_nodes[right[selected]]
        pair_overlap[split_name] = {
            "pair_count": selected_count,
            "either_endpoint_seen_in_train_ratio": float(np.mean(left_seen | right_seen)),
            "both_endpoints_seen_in_train_ratio": float(np.mean(left_seen & right_seen)),
            "endpoint_incidences_seen_in_train_ratio": float(
                (np.sum(left_seen) + np.sum(right_seen)) / (2 * selected_count)
            ),
        }

    mask_counts = {str(mask): int(np.sum(masks == mask)) for mask in range(1, 8)}
    split_node_counts = {
        split_name: int(np.sum((masks & int(SPLIT_BITS[index])) != 0))
        for index, split_name in enumerate(SPLIT_NAMES)
    }
    result = {
        "pair_counts": {
            split_name: int(np.sum(edge_splits == index))
            for index, split_name in enumerate(SPLIT_NAMES)
        },
        "active_code_count": active_count,
        "active_codes_in_multiple_splits": multi_count,
        "active_codes_in_multiple_splits_ratio": multi_count / active_count,
        "active_codes_in_all_three_splits": all_three_count,
        "active_codes_in_all_three_splits_ratio": all_three_count / active_count,
        "node_counts_by_split": split_node_counts,
        "node_counts_by_exact_split_mask": mask_counts,
        "evaluation_pair_endpoint_overlap_with_train": pair_overlap,
        "exact_source_hash_overlap": _hash_split_statistics(source_hashes, masks),
    }
    return result, masks


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.component_size = np.ones(size, dtype=np.int32)

    def find(self, node: int) -> int:
        parent = int(self.parent[node])
        while parent != node:
            grandparent = int(self.parent[parent])
            self.parent[node] = grandparent
            node = parent
            parent = grandparent
        return node

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.component_size[root_left] < self.component_size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.component_size[root_left] += self.component_size[root_right]


def connected_component_statistics(
    unique_left: np.ndarray,
    unique_right: np.ndarray,
    degrees: np.ndarray,
) -> dict:
    union_find = UnionFind(len(degrees))
    for left, right in zip(unique_left, unique_right):
        union_find.union(int(left), int(right))
    active_nodes = np.flatnonzero(degrees > 0)
    roots = np.fromiter(
        (union_find.find(int(node)) for node in active_nodes),
        dtype=np.int32,
        count=active_nodes.size,
    )
    _, sizes = np.unique(roots, return_counts=True)
    sizes = sizes.astype(np.int64)
    return {
        "component_count": int(sizes.size),
        "singleton_component_count": int(np.sum(sizes == 1)),
        "minimum_component_nodes": int(np.min(sizes)),
        "median_component_nodes": float(np.quantile(sizes, 0.50)),
        "p95_component_nodes": float(np.quantile(sizes, 0.95)),
        "p99_component_nodes": float(np.quantile(sizes, 0.99)),
        "maximum_component_nodes": int(np.max(sizes)),
        "largest_component_active_node_ratio": float(np.max(sizes) / active_nodes.size),
    }


def _load_codenet(dataset_zip: Path) -> PairGraphData:
    with zipfile.ZipFile(dataset_zip, "r") as archive:
        program_files = sorted(
            name
            for name in archive.namelist()
            if name.startswith("programs__") and name.endswith(".parquet")
        )
        code_ids: list[str] = []
        languages: list[str] = []
        source_hashes: list[str] = []
        for name in program_files:
            table = pq.read_table(
                io.BytesIO(archive.read(name)),
                columns=["submission_id", "language", "source_sha256"],
            )
            code_ids.extend(str(value) for value in table.column("submission_id").to_pylist())
            languages.extend(
                canonical_language(value) for value in table.column("language").to_pylist()
            )
            source_hashes.extend(str(value) for value in table.column("source_sha256").to_pylist())
        id_to_row = {code_id: index for index, code_id in enumerate(code_ids)}
        if len(id_to_row) != len(code_ids):
            raise ValueError("Duplicate CodeNet submission IDs")

        left_parts = []
        right_parts = []
        label_parts = []
        kind_parts = []
        for name in sorted(
            member
            for member in archive.namelist()
            if member.startswith("pairs__") and member.endswith(".parquet")
        ):
            table = pq.read_table(
                io.BytesIO(archive.read(name)),
                columns=["submission_id_a", "submission_id_b", "label", "pair_kind"],
            )
            left_parts.append(
                np.fromiter(
                    (id_to_row[str(value)] for value in table.column("submission_id_a").to_pylist()),
                    dtype=np.int32,
                    count=table.num_rows,
                )
            )
            right_parts.append(
                np.fromiter(
                    (id_to_row[str(value)] for value in table.column("submission_id_b").to_pylist()),
                    dtype=np.int32,
                    count=table.num_rows,
                )
            )
            label_parts.append(np.asarray(table.column("label").to_numpy(), dtype=np.int8))
            kind_parts.append(np.asarray(table.column("pair_kind").to_pylist(), dtype=object))
        summary = json.loads(archive.read("summary.json"))
    return PairGraphData(
        dataset="CodeNet 4L",
        dataset_key="codenet_4l",
        source_zip=dataset_zip,
        code_ids=code_ids,
        languages=languages,
        source_hashes=source_hashes,
        left=np.concatenate(left_parts),
        right=np.concatenate(right_parts),
        labels=np.concatenate(label_parts),
        pair_kinds=np.concatenate(kind_parts),
        official_splits=None,
        source_metadata=summary,
    )


def _load_clean_zip(dataset_zip: Path) -> PairGraphData:
    code_ids: list[str] = []
    languages: list[str] = []
    source_hashes: list[str] = []
    with zipfile.ZipFile(dataset_zip, "r") as archive:
        metadata = json.loads(archive.read("clean_data/metadata.json"))
        with archive.open("clean_data/codes.jsonl.gz") as compressed:
            with gzip.GzipFile(fileobj=compressed) as gzip_stream:
                with io.TextIOWrapper(gzip_stream, encoding="utf-8") as text:
                    for line in text:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        source = str(row.get("code") or "")
                        code_ids.append(str(row["code_id"]))
                        languages.append(canonical_language(row.get("language")))
                        source_hashes.append(
                            hashlib.sha256(source.encode("utf-8")).hexdigest()
                        )
        id_to_row = {code_id: index for index, code_id in enumerate(code_ids)}
        left: list[int] = []
        right: list[int] = []
        labels: list[int] = []
        splits: list[int] = []
        split_map = {"train": 0, "valid": 1, "validation": 1, "test": 2}
        with archive.open("clean_data/pairs.csv.gz") as compressed:
            with gzip.GzipFile(fileobj=compressed) as gzip_stream:
                with io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="") as text:
                    for row in csv.DictReader(text):
                        left.append(id_to_row[str(row["left_id"])])
                        right.append(id_to_row[str(row["right_id"])])
                        labels.append(int(row["label"]))
                        splits.append(split_map[str(row["split"])])
    label_array = np.asarray(labels, dtype=np.int8)
    return PairGraphData(
        dataset=str(metadata.get("dataset")),
        dataset_key=str(metadata.get("dataset_key")),
        source_zip=dataset_zip,
        code_ids=code_ids,
        languages=languages,
        source_hashes=source_hashes,
        left=np.asarray(left, dtype=np.int32),
        right=np.asarray(right, dtype=np.int32),
        labels=label_array,
        pair_kinds=np.where(label_array == 1, "clone", "non_clone"),
        official_splits=np.asarray(splits, dtype=np.int8),
        source_metadata=metadata,
    )


def load_dataset(dataset_zip: Path) -> PairGraphData:
    with zipfile.ZipFile(dataset_zip, "r") as archive:
        names = set(archive.namelist())
    if "clean_data/codes.jsonl.gz" in names:
        return _load_clean_zip(dataset_zip)
    if any(name.startswith("programs__") for name in names):
        return _load_codenet(dataset_zip)
    raise ValueError("Unsupported dataset ZIP layout")


def _degree_by_group(
    groups: Sequence[str],
    degrees: np.ndarray,
) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for group, degree in zip(groups, degrees):
        grouped[str(group)].append(int(degree))
    return [
        {"group": group, **degree_statistics(np.asarray(values, dtype=np.int32))}
        for group, values in sorted(grouped.items())
        if any(values)
    ]


def analyze(data: PairGraphData) -> tuple[dict, pa.Table]:
    node_count = len(data.code_ids)
    degree, unique_degree, unique_left, unique_right = degree_arrays(
        data.left, data.right, node_count
    )
    rng = np.random.default_rng(RANDOM_SEED)
    random_values = rng.random(data.left.size)
    random_splits = np.where(random_values < 0.8, 0, np.where(random_values < 0.9, 1, 2)).astype(
        np.int8
    )
    random_leakage, random_masks = split_leakage_statistics(
        data.left,
        data.right,
        random_splits,
        node_count,
        data.source_hashes,
    )
    official_masks = np.zeros(node_count, dtype=np.uint8)
    official_leakage = None
    if data.official_splits is not None:
        official_leakage, official_masks = split_leakage_statistics(
            data.left,
            data.right,
            data.official_splits,
            node_count,
            data.source_hashes,
        )

    by_pair_kind = []
    for pair_kind in sorted(set(str(value) for value in data.pair_kinds)):
        selected = data.pair_kinds == pair_kind
        kind_degree = np.bincount(data.left[selected], minlength=node_count) + np.bincount(
            data.right[selected], minlength=node_count
        )
        by_pair_kind.append({"pair_kind": pair_kind, **degree_statistics(kind_degree)})

    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": data.dataset,
        "dataset_key": data.dataset_key,
        "source_zip": str(data.source_zip.resolve()),
        "graph_definition": {
            "nodes": "Stored source-code IDs.",
            "edges": "Published pair rows; labels do not affect connectivity.",
            "pair_incidence_degree": "Number of pair rows containing the code; reversed/duplicate rows count again.",
            "unique_neighbor_degree": "Number of distinct code IDs paired with the code.",
        },
        "counts": {
            "stored_codes": node_count,
            "pair_rows": int(data.left.size),
            "positive_pair_rows": int(np.sum(data.labels == 1)),
            "negative_pair_rows": int(np.sum(data.labels == 0)),
            "unique_undirected_edges": int(unique_left.size),
            "duplicate_or_reversed_edge_occurrences": int(data.left.size - unique_left.size),
            "self_pair_rows": int(np.sum(data.left == data.right)),
        },
        "pair_incidence_degree": degree_statistics(degree),
        "unique_neighbor_degree": degree_statistics(unique_degree),
        "pair_incidence_degree_by_language": _degree_by_group(data.languages, degree),
        "pair_incidence_degree_by_pair_kind": by_pair_kind,
        "connected_components": connected_component_statistics(
            unique_left, unique_right, degree
        ),
        "random_pair_split_simulation": {
            "seed": RANDOM_SEED,
            "requested_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            **random_leakage,
        },
        "official_split_leakage": official_leakage,
    }
    endpoint_table = pa.table(
        {
            "code_id": pa.array(data.code_ids, type=pa.string()),
            "language": pa.array(data.languages, type=pa.string()),
            "source_sha256": pa.array(data.source_hashes, type=pa.string()),
            "pair_incidence_degree": pa.array(degree, type=pa.int32()),
            "unique_neighbor_degree": pa.array(unique_degree, type=pa.int32()),
            "random_pair_split_mask": pa.array(random_masks, type=pa.uint8()),
            "official_split_mask": pa.array(official_masks, type=pa.uint8()),
        }
    )
    return result, endpoint_table


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def write_outputs(result: dict, endpoint_table: pa.Table, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pair_graph_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        endpoint_table,
        output_dir / "endpoint_degrees.parquet",
        compression="zstd",
    )

    degree = endpoint_table.column("pair_incidence_degree").to_numpy()
    unique_degree = endpoint_table.column("unique_neighbor_degree").to_numpy()
    max_degree = int(max(np.max(degree), np.max(unique_degree)))
    with (output_dir / "degree_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "degree",
                "pair_incidence_node_count",
                "unique_neighbor_node_count",
            ],
        )
        writer.writeheader()
        incidence_counts = np.bincount(degree, minlength=max_degree + 1)
        neighbor_counts = np.bincount(unique_degree, minlength=max_degree + 1)
        for value in range(max_degree + 1):
            writer.writerow(
                {
                    "degree": value,
                    "pair_incidence_node_count": int(incidence_counts[value]),
                    "unique_neighbor_node_count": int(neighbor_counts[value]),
                }
            )

    stats = result["pair_incidence_degree"]
    unique_stats = result["unique_neighbor_degree"]
    random = result["random_pair_split_simulation"]
    validation = random["evaluation_pair_endpoint_overlap_with_train"]["validation"]
    test = random["evaluation_pair_endpoint_overlap_with_train"]["test"]
    components = result["connected_components"]
    lines = [
        f"# Pair-Graph Leakage Report: {result['dataset']}",
        "",
        "Each source-code fragment is a node and each labeled pair is an edge. "
        "`Pair-incidence degree` is the number of pairs in which a fragment appears.",
        "",
        "## Degree summary",
        "",
        "| Metric | Pair-incidence degree | Unique-neighbor degree |",
        "|---|---:|---:|",
        f"| Minimum active | {stats['minimum_active']} | {unique_stats['minimum_active']} |",
        f"| Mean active | {stats['mean_active']:.4f} | {unique_stats['mean_active']:.4f} |",
        f"| Median active | {stats['median_active']:.2f} | {unique_stats['median_active']:.2f} |",
        f"| P95 active | {stats['p95_active']:.2f} | {unique_stats['p95_active']:.2f} |",
        f"| P99 active | {stats['p99_active']:.2f} | {unique_stats['p99_active']:.2f} |",
        f"| Maximum active | {stats['maximum_active']} | {unique_stats['maximum_active']} |",
        f"| Gini | {stats['gini_active']:.4f} | {unique_stats['gini_active']:.4f} |",
        "",
        "## Random pair-split simulation (80/10/10)",
        "",
        f"- Fragments present in more than one split: **{random['active_codes_in_multiple_splits']:,} / "
        f"{random['active_code_count']:,} ({_percentage(random['active_codes_in_multiple_splits_ratio'])})**",
        f"- Fragments present in all three splits: **{random['active_codes_in_all_three_splits']:,} "
        f"({_percentage(random['active_codes_in_all_three_splits_ratio'])})**",
        f"- Validation pairs with at least one endpoint seen in training: "
        f"**{_percentage(validation['either_endpoint_seen_in_train_ratio'])}**",
        f"- Validation pairs with both endpoints seen in training: "
        f"**{_percentage(validation['both_endpoints_seen_in_train_ratio'])}**",
        f"- Test pairs with at least one endpoint seen in training: "
        f"**{_percentage(test['either_endpoint_seen_in_train_ratio'])}**",
        f"- Test pairs with both endpoints seen in training: "
        f"**{_percentage(test['both_endpoints_seen_in_train_ratio'])}**",
        "",
        "## Connected components",
        "",
        f"- Number of components: {components['component_count']:,}",
        f"- Largest component: {components['maximum_component_nodes']:,} nodes "
        f"({_percentage(components['largest_component_active_node_ratio'])})",
        "",
    ]
    official = result["official_split_leakage"]
    if official is not None:
        official_validation = official["evaluation_pair_endpoint_overlap_with_train"]["validation"]
        official_test = official["evaluation_pair_endpoint_overlap_with_train"]["test"]
        lines.extend(
            [
                "## Existing official split",
                "",
                f"- Fragments present in multiple splits: {official['active_codes_in_multiple_splits']:,} "
                f"({_percentage(official['active_codes_in_multiple_splits_ratio'])})",
                f"- Validation pairs with an endpoint seen in training: "
                f"{_percentage(official_validation['either_endpoint_seen_in_train_ratio'])}",
                f"- Test pairs with an endpoint seen in training: "
                f"{_percentage(official_test['either_endpoint_seen_in_train_ratio'])}",
                f"- Source hashes present in multiple splits: "
                f"{official['exact_source_hash_overlap']['source_hashes_in_multiple_splits']:,}",
                "",
            ]
        )
    lines.extend(
        [
            "## Recommended split construction",
            "",
            "1. Do not split on code ID when solutions to the same problem must remain together; "
            "use `semantic_group_id = identical_problem_cluster_id or problem_id` as the unit.",
            "2. Connect or deduplicate identical source hashes before assigning splits.",
            "3. Assign groups to train, validation, and test first; create pairs only within a split.",
            "4. Construct different-problem negatives from distinct problem groups in the same split.",
            "5. Apply the degree cap over the complete split instead of independently per bucket.",
            "6. Verify zero overlap of fragment IDs, source hashes, and semantic groups across splits.",
            "7. For an existing pair set, split nodes first and remove cross-split edges. "
            "Connected-component assignment is practical only when no giant component dominates.",
            "",
        ]
    )
    (output_dir / "pair_graph_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_dataset(args.dataset_zip)
    result, endpoint_table = analyze(data)
    write_outputs(result, endpoint_table, args.output_dir)
    stats = result["pair_incidence_degree"]
    random = result["random_pair_split_simulation"]
    print(f"Dataset: {result['dataset']}")
    print(
        f"Degree min/mean/max: {stats['minimum_active']} / "
        f"{stats['mean_active']:.4f} / {stats['maximum_active']}"
    )
    print(
        "Random pair-split codes in multiple splits: "
        f"{100 * random['active_codes_in_multiple_splits_ratio']:.2f}%"
    )
    print(f"Wrote outputs to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
