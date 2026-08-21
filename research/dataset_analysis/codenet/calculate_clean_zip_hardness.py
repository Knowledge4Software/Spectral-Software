#!/usr/bin/env python3
"""Run the shared hardness metric on a spectral_clean_data_v1 ZIP."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from calculate_hardness import (
    FEATURE_COUNT,
    SIMILARITY_NAME,
    fingerprint_jaccard,
    structural_fingerprint,
    summarize,
)


LANGUAGE_ALIASES = {
    "c": "C",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "cs": "C#",
    "java": "Java",
    "python": "Python",
    "py": "Python",
}


def canonical_language(value: object) -> str:
    text = str(value or "unknown").strip()
    return LANGUAGE_ALIASES.get(text.lower(), text)


def source_zip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_codes(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, int], dict[str, str], Counter[str], int]:
    fingerprints: dict[str, int] = {}
    languages: dict[str, str] = {}
    language_counts: Counter[str] = Counter()
    zero_syntax = 0
    with archive.open("clean_data/codes.jsonl.gz") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as gzip_stream:
            with io.TextIOWrapper(gzip_stream, encoding="utf-8") as text:
                for line_number, line in enumerate(text, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    code_id = str(row["code_id"])
                    if code_id in fingerprints:
                        raise ValueError(f"Duplicate code_id at line {line_number}: {code_id}")
                    language = canonical_language(row.get("language"))
                    source = str(row.get("code") or "")
                    fingerprints[code_id] = structural_fingerprint(source, language)
                    languages[code_id] = language
                    language_counts[language] += 1
                    # A fingerprint with only BEGIN/END has exactly three set
                    # features: BEGIN, END, and their bigram.
                    if fingerprints[code_id].bit_count() <= 3:
                        zero_syntax += 1
    return fingerprints, languages, language_counts, zero_syntax


def load_pairs(archive: zipfile.ZipFile) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with archive.open("clean_data/pairs.csv.gz") as compressed:
        with gzip.GzipFile(fileobj=compressed, mode="rb") as gzip_stream:
            with io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="") as text:
                reader = csv.DictReader(text)
                required = {"split", "left_id", "right_id", "label"}
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"Missing pair columns: {sorted(missing)}")
                for row in reader:
                    rows.append(
                        {
                            "split": str(row["split"]),
                            "left_id": str(row["left_id"]),
                            "right_id": str(row["right_id"]),
                            "label": int(row["label"]),
                        }
                    )
    return rows


def values_summary(values: list[float], role: str) -> dict:
    return summarize(np.asarray(values, dtype=np.float64), role)


def calculate(dataset_zip: Path) -> dict:
    started = time.perf_counter()
    with zipfile.ZipFile(dataset_zip, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC validation failed for {bad_member}")
        metadata = json.loads(archive.read("clean_data/metadata.json"))
        fingerprints, languages, language_counts, zero_syntax = load_codes(archive)
        pairs = load_pairs(archive)

    by_label: dict[int, list[float]] = defaultdict(list)
    unique_by_label: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    by_split_label: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_configuration_label: dict[tuple[str, int], list[float]] = defaultdict(list)
    pair_configurations: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    duplicate_directed_pairs = 0
    duplicate_undirected_pairs = 0
    self_pairs = 0
    missing_endpoints: set[str] = set()
    undirected_label_conflicts = 0
    directed_seen: set[tuple[str, str]] = set()
    undirected_seen: set[tuple[str, str]] = set()

    for row in pairs:
        left_id = str(row["left_id"])
        right_id = str(row["right_id"])
        label = int(row["label"])
        split = str(row["split"])
        if label not in {0, 1}:
            raise ValueError(f"Non-binary label encountered: {label}")
        if left_id not in fingerprints:
            missing_endpoints.add(left_id)
            continue
        if right_id not in fingerprints:
            missing_endpoints.add(right_id)
            continue
        if left_id == right_id:
            self_pairs += 1
        directed = (left_id, right_id)
        undirected = tuple(sorted(directed))
        if directed in directed_seen:
            duplicate_directed_pairs += 1
        if undirected in undirected_seen:
            duplicate_undirected_pairs += 1
        directed_seen.add(directed)
        undirected_seen.add(undirected)

        left_language = languages[left_id]
        right_language = languages[right_id]
        configuration = "-".join(sorted((left_language, right_language)))
        similarity = fingerprint_jaccard(fingerprints[left_id], fingerprints[right_id])
        by_label[label].append(similarity)
        opposite_label = 1 - label
        if undirected in unique_by_label[opposite_label]:
            undirected_label_conflicts += 1
        unique_by_label[label].setdefault(undirected, similarity)
        by_split_label[(split, label)].append(similarity)
        by_configuration_label[(configuration, label)].append(similarity)
        pair_configurations[configuration] += 1
        split_counts[split] += 1
        label_counts[label] += 1

    if missing_endpoints:
        raise ValueError(f"{len(missing_endpoints)} pair endpoint IDs are missing from codes")
    if not by_label[1] or not by_label[0]:
        raise ValueError("Both positive and negative pairs are required")

    per_split = []
    for split in sorted(split_counts):
        per_split.append(
            {
                "split": split,
                "positive": values_summary(by_split_label[(split, 1)], "positive"),
                "negative": values_summary(by_split_label[(split, 0)], "negative"),
            }
        )
    per_configuration = []
    for configuration in sorted(pair_configurations):
        entry = {
            "configuration_id": configuration,
            "pair_count": pair_configurations[configuration],
        }
        if by_configuration_label[(configuration, 1)]:
            entry["positive"] = values_summary(
                by_configuration_label[(configuration, 1)], "positive"
            )
        if by_configuration_label[(configuration, 0)]:
            entry["negative"] = values_summary(
                by_configuration_label[(configuration, 0)], "negative"
            )
        per_configuration.append(entry)

    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": metadata.get("dataset"),
        "dataset_key": metadata.get("dataset_key"),
        "dataset_zip": str(dataset_zip.resolve()),
        "dataset_zip_sha256": source_zip_sha256(dataset_zip),
        "run_mode": "all_pairs",
        "elapsed_seconds": time.perf_counter() - started,
        "similarity": {
            "name": SIMILARITY_NAME,
            "feature_count": FEATURE_COUNT,
            "definition": (
                "Exact set Jaccard over canonical language-neutral syntactic token "
                "unigrams and adjacent token-category bigrams. Identifiers and literals "
                "are delexicalized; comments are ignored."
            ),
            "comparable_with_codenet_hardness_metrics": True,
            "semantic_equivalence_score": False,
        },
        "programs": {
            "count": len(fingerprints),
            "language_counts": dict(sorted(language_counts.items())),
            "zero_syntax_programs": zero_syntax,
        },
        "pairs": {
            "count": len(pairs),
            "positive_count": label_counts[1],
            "negative_count": label_counts[0],
            "split_counts": dict(sorted(split_counts.items())),
            "configuration_counts": dict(sorted(pair_configurations.items())),
        },
        "integrity": {
            "self_pairs": self_pairs,
            "duplicate_directed_pair_occurrences": duplicate_directed_pairs,
            "duplicate_undirected_pair_occurrences": duplicate_undirected_pairs,
            "missing_endpoint_ids": 0,
            "undirected_pair_label_conflict_occurrences": undirected_label_conflicts,
        },
        "requested_metrics": {
            "positive_syntactic_distance": values_summary(by_label[1], "positive"),
            "negative_hardness": values_summary(by_label[0], "negative"),
        },
        "unique_undirected_pair_sensitivity": {
            "definition": (
                "Each unordered endpoint pair is counted once per label. This removes "
                "duplicate and reversed-pair weighting for comparison with CodeNet."
            ),
            "positive_syntactic_distance": values_summary(
                list(unique_by_label[1].values()), "positive"
            ),
            "negative_hardness": values_summary(
                list(unique_by_label[0].values()), "negative"
            ),
        },
        "per_split": per_split,
        "per_configuration": per_configuration,
        "source_metadata": metadata,
    }


def write_outputs(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hardness_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    positive = results["requested_metrics"]["positive_syntactic_distance"]
    negative = results["requested_metrics"]["negative_hardness"]
    unique_positive = results["unique_undirected_pair_sensitivity"][
        "positive_syntactic_distance"
    ]
    unique_negative = results["unique_undirected_pair_sensitivity"]["negative_hardness"]
    csv_rows = [
        {"scope": "global", "group": "all", "pair_kind": "positive", **positive},
        {"scope": "global", "group": "all", "pair_kind": "negative", **negative},
        {
            "scope": "global_unique_undirected",
            "group": "all",
            "pair_kind": "positive",
            **unique_positive,
        },
        {
            "scope": "global_unique_undirected",
            "group": "all",
            "pair_kind": "negative",
            **unique_negative,
        },
    ]
    for row in results["per_split"]:
        csv_rows.append(
            {"scope": "split", "group": row["split"], "pair_kind": "positive", **row["positive"]}
        )
        csv_rows.append(
            {"scope": "split", "group": row["split"], "pair_kind": "negative", **row["negative"]}
        )
    for row in results["per_configuration"]:
        for key in ("positive", "negative"):
            if key in row:
                csv_rows.append(
                    {
                        "scope": "configuration",
                        "group": row["configuration_id"],
                        "pair_kind": key,
                        **row[key],
                    }
                )
    fields = ["scope", "group", "pair_kind"] + list(positive.keys())
    with (output_dir / "hardness_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        f"# {results['dataset']} hardness metrics",
        "",
        f"- Source ZIP: `{results['dataset_zip']}`",
        f"- SHA-256: `{results['dataset_zip_sha256']}`",
        f"- Programs: {results['programs']['count']:,}",
        f"- Pairs: {results['pairs']['count']:,} (all pairs; no sampling)",
        f"- Similarity: `{results['similarity']['name']}`",
        "",
        "| Metric | Pair count | Value | Mean sim | Median sim | P95 | Sim >= 0.75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Positive Syntactic Distance | {positive['pair_count']:,} | "
            f"{positive['positive_syntactic_distance']:.6f} | "
            f"{positive['mean_syntactic_similarity']:.6f} | {positive['median']:.6f} | "
            f"{positive['p95']:.6f} | {100 * positive['similarity_ge_0_75_ratio']:.2f}% |"
        ),
        (
            f"| Negative Hardness | {negative['pair_count']:,} | "
            f"{negative['negative_hardness']:.6f} | "
            f"{negative['mean_syntactic_similarity']:.6f} | {negative['median']:.6f} | "
            f"{negative['p95']:.6f} | {100 * negative['similarity_ge_0_75_ratio']:.2f}% |"
        ),
        (
            f"| Positive Distance (unique unordered) | {unique_positive['pair_count']:,} | "
            f"{unique_positive['positive_syntactic_distance']:.6f} | "
            f"{unique_positive['mean_syntactic_similarity']:.6f} | "
            f"{unique_positive['median']:.6f} | {unique_positive['p95']:.6f} | "
            f"{100 * unique_positive['similarity_ge_0_75_ratio']:.2f}% |"
        ),
        (
            f"| Negative Hardness (unique unordered) | {unique_negative['pair_count']:,} | "
            f"{unique_negative['negative_hardness']:.6f} | "
            f"{unique_negative['mean_syntactic_similarity']:.6f} | "
            f"{unique_negative['median']:.6f} | {unique_negative['p95']:.6f} | "
            f"{100 * unique_negative['similarity_ge_0_75_ratio']:.2f}% |"
        ),
        "",
        "## By split",
        "",
        "| Split | Positive distance | Negative hardness | Positive pairs | Negative pairs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results["per_split"]:
        lines.append(
            f"| {row['split']} | {row['positive']['positive_syntactic_distance']:.6f} | "
            f"{row['negative']['negative_hardness']:.6f} | "
            f"{row['positive']['pair_count']:,} | {row['negative']['pair_count']:,} |"
        )
    lines.extend(
        [
            "",
            "The metric is a syntactic proxy, not an AST edit distance or proof of semantic equivalence.",
            "",
        ]
    )
    (output_dir / "hardness_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dataset_zip.is_file():
        raise FileNotFoundError(args.dataset_zip)
    results = calculate(args.dataset_zip)
    write_outputs(results, args.output_dir)
    positive = results["requested_metrics"]["positive_syntactic_distance"]
    negative = results["requested_metrics"]["negative_hardness"]
    print(f"Dataset: {results['dataset']}")
    print(f"Positive Syntactic Distance: {positive['positive_syntactic_distance']:.6f}")
    print(f"Negative Hardness: {negative['negative_hardness']:.6f}")
    print(f"Processed pairs: {results['pairs']['count']:,}")
    print(f"Wrote outputs to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
