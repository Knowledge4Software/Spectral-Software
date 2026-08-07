"""Build leakage-safe, Kaggle-ready BigCloneBench Type-1--3 and Type-4 datasets.

The BCB relational dump has no official binary train/valid/test split.  This
script creates two reproducible benchmarks from the already extracted BCB
source variants in ``data/bcb``:

* ``bcb_type123_v4``: Type-1 + Type-2 + all Type-3 clones vs. curated BCB
  false-positive non-clones.
* ``bcb_type4_v4``: Type-4 clones vs. the same curated non-clones.

The split is *code-disjoint*: a method ID may occur in exactly one of train,
valid, or test.  Cross-split pairs are dropped, then each split is balanced
exactly 1:1 using only known BCB labels.  In particular, Type-4 positives are
capped by the approximately 260k curated negatives; no unlabelled random pair
is ever called a non-clone.

After pair selection, the regular Joern -> cleaned graph -> spectral pipeline
runs for each benchmark and writes a compact ``*_clean_data.zip`` that can be
attached directly to the Kaggle method and baseline notebooks.

Run from the repository root::

    .venv\\Scripts\\python.exe scripts\\build_bcb_type_benchmarks.py

Useful options::

    # inspect selected pair counts without writing files or running Joern
    .venv\\Scripts\\python.exe scripts\\build_bcb_type_benchmarks.py --dry-run

    # rebuild only Type-4 after deliberately removing its old target folders
    .venv\\Scripts\\python.exe scripts\\build_bcb_type_benchmarks.py --datasets bcb_type4_v4 --force

The output directories are versioned and do not alter BCB4DATA/BCB4STRICT or
any of the four V3 benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.clean_data_export import (  # noqa: E402
    _export_graph_spectra,
    create_clean_data_zip,
    write_clean_codes,
    write_clean_pairs,
)
from spectral_code.utils.dataset_paths import DATA_ROOT, OUTPUTS_ROOT  # noqa: E402


SPLITS = ("train", "valid", "test")
SPLIT_WEIGHTS = {"train": 0.70, "valid": 0.15, "test": 0.15}
GRAPH_TYPES = ("ast", "cfg", "ddg", "cpg")
JOERN_LANGUAGE = "javasrc"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    positive_variants: tuple[str, ...]
    positive_candidate_cap: int | None
    negative_candidate_cap: int


SPECS = {
    "bcb_type123_v4": DatasetSpec(
        key="bcb_type123_v4",
        title="BigCloneBench Type-1--3 (code-disjoint, balanced)",
        positive_variants=("type1", "type2", "type3/all"),
        positive_candidate_cap=None,
        negative_candidate_cap=250_000,
    ),
    "bcb_type4_v4": DatasetSpec(
        key="bcb_type4_v4",
        title="BigCloneBench Type-4 (code-disjoint, balanced)",
        positive_variants=("type4",),
        # BCB has roughly 260k curated negatives.  This cap keeps Type-4
        # balanced without manufacturing negatives from unlabelled pairs.
        positive_candidate_cap=250_000,
        negative_candidate_cap=250_000,
    ),
}


def canonical(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def bcb_source_dir(variant: str) -> Path:
    return DATA_ROOT / "bcb" / variant


def read_pairs(path: Path, expected_label: int) -> list[tuple[int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing BCB source pairs: {path}")
    rows: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed pair at {path}:{line_number}")
            left, right, label = map(int, fields)
            if label != expected_label:
                raise ValueError(f"Expected label={expected_label} at {path}:{line_number}, found {label}")
            key = canonical(left, right)
            if key not in seen:
                rows.append((key[0], key[1], label))
                seen.add(key)
    return rows


def merge_positive_sources(variants: Iterable[str]) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    merged: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    counts: dict[str, int] = {}
    for variant in variants:
        rows = read_pairs(bcb_source_dir(variant) / "train.txt", expected_label=1)
        counts[variant] = len(rows)
        for row in rows:
            key = canonical(row[0], row[1])
            if key not in seen:
                merged.append(row)
                seen.add(key)
    return merged, counts


def sample_rows(rows: list[tuple[int, int, int]], cap: int | None, rng: random.Random) -> list[tuple[int, int, int]]:
    if cap is None or cap >= len(rows):
        return list(rows)
    return rng.sample(rows, cap)


def allocate(total: int) -> dict[str, int]:
    """Largest-remainder allocation with exact total and 70/15/15 proportions."""
    raw = {split: total * SPLIT_WEIGHTS[split] for split in SPLITS}
    result = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(result.values())
    for split in sorted(SPLITS, key=lambda item: (raw[item] - result[item], item), reverse=True)[:remainder]:
        result[split] += 1
    return result


def assign_codes(code_ids: set[int], seed: int) -> dict[int, str]:
    ids = list(code_ids)
    random.Random(seed).shuffle(ids)
    counts = allocate(len(ids))
    assignment: dict[int, str] = {}
    start = 0
    for split in SPLITS:
        for code_id in ids[start:start + counts[split]]:
            assignment[code_id] = split
        start += counts[split]
    return assignment


def rows_within_assigned_split(
    rows: list[tuple[int, int, int]], assignment: dict[int, str]
) -> dict[str, list[tuple[int, int, int]]]:
    result = {split: [] for split in SPLITS}
    for row in rows:
        left_split = assignment[row[0]]
        if left_split == assignment[row[1]]:
            result[left_split].append(row)
    return result


def largest_feasible_per_class(positive: dict[str, list], negative: dict[str, list]) -> int:
    upper = sum(min(len(positive[split]), len(negative[split])) for split in SPLITS)
    low, high = 0, upper
    while low < high:
        middle = (low + high + 1) // 2
        quota = allocate(middle)
        feasible = all(quota[split] <= len(positive[split]) and quota[split] <= len(negative[split]) for split in SPLITS)
        if feasible:
            low = middle
        else:
            high = middle - 1
    return low


def select_code_disjoint_balanced(
    positive_rows: list[tuple[int, int, int]],
    negative_rows: list[tuple[int, int, int]],
    *,
    seed: int,
    search_trials: int,
) -> tuple[dict[str, list[tuple[int, int, int]]], dict]:
    all_ids = {code_id for left, right, _ in positive_rows + negative_rows for code_id in (left, right)}
    if not all_ids:
        raise RuntimeError("No candidate BCB IDs available.")

    best: tuple[int, dict[int, str], dict[str, list], dict[str, list]] | None = None
    for offset in tqdm(range(search_trials), desc="Searching code-disjoint split", unit="trial"):
        assignment = assign_codes(all_ids, seed + offset)
        positive = rows_within_assigned_split(positive_rows, assignment)
        negative = rows_within_assigned_split(negative_rows, assignment)
        per_class = largest_feasible_per_class(positive, negative)
        if best is None or per_class > best[0]:
            best = (per_class, assignment, positive, negative)

    assert best is not None
    per_class, assignment, positive_by_split, negative_by_split = best
    if per_class < 100:
        raise RuntimeError(
            f"Code-disjoint assignment retained only {per_class} pairs per class. "
            "Increase candidate caps or inspect the BCB source artifacts."
        )
    quota = allocate(per_class)
    selected: dict[str, list[tuple[int, int, int]]] = {}
    rng = random.Random(seed + 10_000)
    for split in SPLITS:
        selected_rows = rng.sample(positive_by_split[split], quota[split])
        selected_rows.extend(rng.sample(negative_by_split[split], quota[split]))
        rng.shuffle(selected_rows)
        selected[split] = selected_rows

    used_ids = {code_id for rows in selected.values() for left, right, _ in rows for code_id in (left, right)}
    split_ids = {
        split: {code_id for left, right, _ in rows for code_id in (left, right)}
        for split, rows in selected.items()
    }
    if any(split_ids[first] & split_ids[second] for index, first in enumerate(SPLITS) for second in SPLITS[index + 1:]):
        raise RuntimeError("Internal error: code-disjoint split validation failed.")

    summary = {
        "seed": seed,
        "search_trials": search_trials,
        "candidate_code_ids": len(all_ids),
        "candidate_pairs": {"clone": len(positive_rows), "non_clone": len(negative_rows)},
        "within_split_pairs_before_balancing": {
            split: {"clone": len(positive_by_split[split]), "non_clone": len(negative_by_split[split])}
            for split in SPLITS
        },
        "selected_per_class": per_class,
        "selected_quota_per_class": quota,
        "used_code_ids": len(used_ids),
        "code_ids_by_split": {split: len(split_ids[split]) for split in SPLITS},
        "dropped_cross_split_pairs": {
            "clone": len(positive_rows) - sum(len(positive_by_split[split]) for split in SPLITS),
            "non_clone": len(negative_rows) - sum(len(negative_by_split[split]) for split in SPLITS),
        },
    }
    return selected, summary


def load_codes(source_variants: Iterable[str], needed_ids: set[int]) -> dict[int, str]:
    codes: dict[int, str] = {}
    for variant in source_variants:
        path = bcb_source_dir(variant) / "data.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing BCB source code file: {path}")
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                code_id = int(row["idx"])
                if code_id not in needed_ids:
                    continue
                code = str(row["func"])
                previous = codes.get(code_id)
                if previous is not None and previous != code:
                    raise RuntimeError(f"Conflicting source code for BCB method {code_id}.")
                codes[code_id] = code
    missing = needed_ids - set(codes)
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(f"{len(missing):,} selected BCB methods are absent from source data; examples={sample}")
    return codes


def write_prepared_dataset(path: Path, splits: dict[str, list[tuple[int, int, int]]], codes: dict[int, str], metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (path / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as dst:
            for left, right, label in splits[split]:
                dst.write(f"{left}\t{right}\t{label}\n")
    with (path / "data.jsonl").open("w", encoding="utf-8", newline="\n") as dst:
        for code_id in sorted(codes):
            dst.write(json.dumps({"idx": code_id, "func": codes[code_id], "lang": "java"}, ensure_ascii=False) + "\n")
    (path / "selection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def output_is_graph_complete(output_root: Path, expected_methods: int) -> bool:
    for relative in ("clean_graphs/graph_shards_manifest.json", "spectral_features/spectral_features_manifest.json"):
        path = output_root / relative
        if not path.is_file():
            return False
        try:
            if int(json.loads(path.read_text(encoding="utf-8")).get("total_methods", -1)) != expected_methods:
                return False
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    return True


def run_pipeline(prepared_dir: Path, output_root: Path, *, dry_run: bool) -> None:
    env = os.environ.copy()
    env.update({
        "BCB_DATA_FILE": str(prepared_dir / "data.jsonl"),
        "BCB_DATA_DIR": str(prepared_dir),
        "OUTPUT_DIR": str(output_root),
        "JOERN_LANGUAGE": JOERN_LANGUAGE,
        "PIPELINE_GRAPH_TYPES": "ast,cfg,ddg",
        "PIPELINE_BASE_LAYERS": "ast,cfg,ddg",
        "SPECTRAL_GRAPH_TYPES": ",".join(GRAPH_TYPES),
        "JOERN_PARSE_CHUNK_SIZE": os.getenv("JOERN_PARSE_CHUNK_SIZE", "500"),
        "PIPELINE_CLEAN_INTERMEDIATE_ARTIFACTS": "1",
        "PIPELINE_CLEAN_RAW_FEATURES": "1",
    })
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else os.pathsep.join([str(PROJECT_ROOT), existing_pythonpath])
    for script in ("pipelines/01_extract_dataset.py", "pipelines/02_build_graph_db.py", "pipelines/03_extract_spectral_features.py"):
        command = [sys.executable, str(PROJECT_ROOT / script)]
        if dry_run:
            print("[dry-run]", " ".join(command), "OUTPUT_DIR=", output_root)
        else:
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def graph_method_ids(output_root: Path) -> set[int]:
    manifest_path = output_root / "clean_graphs" / "graph_shards_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids: set[int] = set()
    for raw in manifest["shards"]:
        shard_path = Path(raw)
        if not shard_path.is_absolute():
            shard_path = manifest_path.parent / shard_path
        with shard_path.open("rb") as src:
            ids.update(int(code_id) for code_id in pickle.load(src))
    return ids


def trim_to_graph_coverage(splits: dict[str, list[tuple[int, int, int]]], available_ids: set[int], seed: int) -> dict[str, list[tuple[int, int, int]]]:
    """Keep code-disjointness while restoring exact per-split class balance after Joern failures."""
    rng = random.Random(seed + 20_000)
    result: dict[str, list[tuple[int, int, int]]] = {}
    for split in SPLITS:
        usable = [row for row in splits[split] if row[0] in available_ids and row[1] in available_ids]
        positive = [row for row in usable if row[2] == 1]
        negative = [row for row in usable if row[2] == 0]
        count = min(len(positive), len(negative))
        if count < 100:
            raise RuntimeError(f"{split} has only {count} graph-covered pairs per class after graph extraction.")
        rows = rng.sample(positive, count) + rng.sample(negative, count)
        rng.shuffle(rows)
        result[split] = rows
    return result


def split_counts(splits: dict[str, list[tuple[int, int, int]]]) -> dict[str, dict[str, int]]:
    result = {}
    for split, rows in splits.items():
        labels = Counter(label for _, _, label in rows)
        result[split] = {
            "pairs": len(rows),
            "clone": labels[1],
            "non_clone": labels[0],
            "codes": len({code_id for left, right, _ in rows for code_id in (left, right)}),
        }
        if labels[0] != labels[1]:
            raise RuntimeError(f"{split} is not balanced: {dict(labels)}")
    return result


def package_clean_data(spec: DatasetSpec, output_root: Path, prepared_dir: Path, selected: dict[str, list[tuple[int, int, int]]], source_variants: list[str], metadata: dict) -> Path:
    available_ids = graph_method_ids(output_root)
    final_splits = trim_to_graph_coverage(selected, available_ids, seed=int(metadata["seed"]))
    needed_ids = {code_id for rows in final_splits.values() for left, right, _ in rows for code_id in (left, right)}
    codes = load_codes(source_variants, needed_ids)

    clean_dir = output_root / "clean_data"
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    clean_dir.mkdir(parents=True)
    pair_counts = write_clean_pairs(clean_dir, final_splits)
    write_clean_codes(clean_dir, codes)
    graph_summary = _export_graph_spectra(output_root, clean_dir / "graph_spectra.jsonl.gz", needed_ids, list(GRAPH_TYPES), precision=8)
    if graph_summary["methods"] != len(codes):
        raise RuntimeError("Graph/code coverage mismatch during final BCB package export.")

    metadata = {
        **metadata,
        "format": "spectral_clean_data_v1",
        "dataset": spec.key,
        "title": spec.title,
        "language": "java",
        "graph_types": list(GRAPH_TYPES),
        "split_strategy": "seeded_code_disjoint_70_15_15_then_exact_class_balance",
        "negative_protocol": "curated BigCloneBench false-positive pairs only; no synthetic or unlabelled negatives",
        "leakage_controls": {
            "code_disjoint_splits": True,
            "pair_duplicates_removed": True,
            "positive_negative_conflicts_removed": True,
            "post_joern_graph_coverage_rebalanced": True,
        },
        "counts": {
            "codes": len(codes),
            "pairs": {**pair_counts, "total": sum(pair_counts.values())},
            "split_labels": split_counts(final_splits),
            "graph_spectra": graph_summary,
        },
    }
    (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (clean_dir / "README.md").write_text(
        f"# {spec.title}\n\n"
        "This is a Java-only BigCloneBench binary clone benchmark. Type labels are taken from the BCB extraction protocol; "
        "non-clones are curated BCB false positives. Train, validation, and test have no shared method IDs and every split is exactly class-balanced.\n",
        encoding="utf-8",
    )
    zip_path = create_clean_data_zip(clean_dir, output_root / f"{spec.key}_clean_data.zip")
    metadata["zip"] = str(zip_path)
    (clean_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return zip_path


def build_one(spec: DatasetSpec, args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    positive_rows, source_positive_counts = merge_positive_sources(spec.positive_variants)
    negative_rows = read_pairs(bcb_source_dir("non_clone") / "train.txt", expected_label=0)
    positive_keys = {canonical(left, right) for left, right, _ in positive_rows}
    negative_rows = [row for row in negative_rows if canonical(row[0], row[1]) not in positive_keys]
    positive_candidates = sample_rows(positive_rows, spec.positive_candidate_cap, rng)
    negative_candidates = sample_rows(negative_rows, spec.negative_candidate_cap, rng)
    if not positive_candidates or not negative_candidates:
        raise RuntimeError(f"{spec.key}: empty candidate pool.")

    selected, selection = select_code_disjoint_balanced(
        positive_candidates, negative_candidates, seed=args.seed, search_trials=args.split_search_trials
    )
    summary = split_counts(selected)
    print(f"\n=== {spec.key} selection ===")
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    prepared_dir = DATA_ROOT / "bcb_prepared" / spec.key
    output_root = OUTPUTS_ROOT / "bcb" / spec.key
    if args.force:
        for target in (prepared_dir, output_root):
            if target.exists():
                print(f"[force cleanup] {target}")
                shutil.rmtree(target)
    if (output_root / f"{spec.key}_clean_data.zip").is_file() and not args.force:
        print(f"[skip ready] {spec.key}: {output_root / f'{spec.key}_clean_data.zip'}")
        return

    source_variants = [*spec.positive_variants, "non_clone"]
    needed_ids = {code_id for rows in selected.values() for left, right, _ in rows for code_id in (left, right)}
    codes = load_codes(source_variants, needed_ids)
    metadata = {
        "seed": args.seed,
        "source_variants": {"positive": list(spec.positive_variants), "negative": "non_clone"},
        "source_positive_pair_counts": source_positive_counts,
        "source_negative_pair_count": len(negative_rows),
        "candidate_caps": {"positive": spec.positive_candidate_cap, "negative": spec.negative_candidate_cap},
        "selection": selection,
        "pre_graph_split_labels": summary,
    }
    write_prepared_dataset(prepared_dir, selected, codes, metadata)

    if not output_is_graph_complete(output_root, len(codes)):
        run_pipeline(prepared_dir, output_root, dry_run=False)
    else:
        print(f"[resume graph-ready] {spec.key}: {len(codes):,} methods")
    zip_path = package_clean_data(spec, output_root, prepared_dir, selected, source_variants, metadata)
    print(f"[+] {spec.key} ready: {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-search-trials", type=int, default=64, help="Random code assignment trials; higher retains more within-split pairs.")
    parser.add_argument("--force", action="store_true", help="Delete only the selected new bcb_*_v4 prepared/output targets before rebuilding.")
    parser.add_argument("--dry-run", action="store_true", help="Select and print pair counts without writing or running Joern.")
    args = parser.parse_args()
    if args.split_search_trials < 1:
        parser.error("--split-search-trials must be at least 1.")
    for key in args.datasets:
        build_one(SPECS[key], args)


if __name__ == "__main__":
    main()
