"""Create a leakage-resistant BCB Type-4 benchmark from existing clean graphs.

The legacy BCB4DATA export mixes Type-4 positives and curated false-positive
non-clones, then splits pairs.  That permits code IDs (and their source
membership) to leak across splits.  This exporter keeps only code IDs that
occur in *both* labelled sources, assigns IDs to mutually exclusive splits,
and uses only pairs whose two endpoints remain in the same split.

It reuses the already exported portable graph records, so no Joern rerun is
needed.  The resulting data is intentionally much smaller: validity of the
evaluation takes precedence over preserving the legacy pair count.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.clean_data_export import create_clean_data_zip
from spectral_code.utils.dataset_paths import OUTPUTS_ROOT, bcb_type_dir


SPLITS = ("train", "valid", "test")
DEFAULT_SPLIT_RATIOS = (0.60, 0.20, 0.20)


def _canonical(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _read_pairs(path: Path, expected_label: int) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed pair at {path}:{line_number}")
            left, right, raw_label = fields
            if int(raw_label) != expected_label:
                raise ValueError(f"Unexpected label at {path}:{line_number}")
            key = _canonical(left, right)
            if left == right or key in seen:
                continue
            seen.add(key)
            rows.append((left, right, expected_label))
    return rows


def _endpoint_ids(rows: list[tuple[str, str, int]]) -> set[str]:
    return {code_id for left, right, _ in rows for code_id in (left, right)}


def _assign_ids(code_ids: set[str], seed: int, ratios: tuple[float, float, float]) -> dict[str, str]:
    ordered = sorted(code_ids)
    random.Random(seed).shuffle(ordered)
    train_end = int(len(ordered) * ratios[0])
    valid_end = train_end + int(len(ordered) * ratios[1])
    return {
        code_id: SPLITS[0] if index < train_end else SPLITS[1] if index < valid_end else SPLITS[2]
        for index, code_id in enumerate(ordered)
    }


def _within_split_rows(
    rows: list[tuple[str, str, int]], assignment: dict[str, str]
) -> tuple[dict[str, dict[int, list[tuple[str, str, int]]]], Counter]:
    kept = {split: {0: [], 1: []} for split in SPLITS}
    discarded = Counter()
    for row in rows:
        left, right, label = row
        left_split, right_split = assignment[left], assignment[right]
        if left_split != right_split:
            discarded[label] += 1
            continue
        kept[left_split][label].append(row)
    return kept, discarded


def _score(kept: dict[str, dict[int, list[tuple[str, str, int]]]]) -> tuple[int, int, int]:
    """Keep every split trainable before maximizing holdout size."""
    per_split = [min(len(kept[split][0]), len(kept[split][1])) for split in SPLITS]
    return (min(per_split), min(per_split[1:]), sum(per_split))


def _select_balanced_rows(
    kept: dict[str, dict[int, list[tuple[str, str, int]]]], seed: int
) -> dict[str, list[tuple[str, str, int]]]:
    selected: dict[str, list[tuple[str, str, int]]] = {}
    for split_index, split in enumerate(SPLITS):
        count = min(len(kept[split][0]), len(kept[split][1]))
        if count == 0:
            raise RuntimeError(f"No balanced examples available for {split}.")
        rng = random.Random(seed + split_index)
        rows = rng.sample(kept[split][0], count) + rng.sample(kept[split][1], count)
        rng.shuffle(rows)
        selected[split] = rows
    return selected


def _filter_jsonl_gzip(source: Path, destination: Path, needed_ids: set[str]) -> int:
    temporary = destination.with_name(destination.name + ".tmp")
    written: set[str] = set()
    with gzip.open(source, "rt", encoding="utf-8") as src, gzip.open(temporary, "wt", encoding="utf-8", newline="") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            code_id = str(row["code_id"])
            if code_id in needed_ids:
                if code_id in written:
                    raise RuntimeError(f"Duplicate code record in {source}: {code_id}")
                dst.write(line)
                written.add(code_id)
    missing = needed_ids - written
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{len(missing):,} selected code IDs are missing from {source}.")
    temporary.replace(destination)
    return len(written)


def _write_pairs(path: Path, rows_by_split: dict[str, list[tuple[str, str, int]]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=("split", "left_id", "right_id", "label", "label_name"))
        writer.writeheader()
        for split in SPLITS:
            for left, right, label in rows_by_split[split]:
                writer.writerow({
                    "split": split,
                    "left_id": left,
                    "right_id": right,
                    "label": label,
                    "label_name": "clone" if label else "non_clone",
                })


def _validate(rows_by_split: dict[str, list[tuple[str, str, int]]]) -> dict[str, dict[str, int]]:
    ids_by_split = {
        split: _endpoint_ids(rows)
        for split, rows in rows_by_split.items()
    }
    for index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[index + 1:]:
            overlap = ids_by_split[left_split] & ids_by_split[right_split]
            if overlap:
                raise RuntimeError(f"Code leakage between {left_split} and {right_split}: {len(overlap):,} IDs")

    summary: dict[str, dict[str, int]] = {}
    for split, rows in rows_by_split.items():
        labels = Counter(label for _, _, label in rows)
        if labels[0] != labels[1]:
            raise RuntimeError(f"{split} is not class balanced: {dict(labels)}")
        summary[split] = {"pairs": len(rows), "clone": labels[1], "non_clone": labels[0], "codes": len(ids_by_split[split])}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a code-disjoint, source-balanced BCB Type-4 export.")
    parser.add_argument("--positive-variant", default="4")
    parser.add_argument("--negative-variant", default="non_clone")
    parser.add_argument("--source-dataset", default="BCB4DATA", help="Existing clean export used to reuse graphs and code records.")
    parser.add_argument("--dataset-name", default="BCB4STRICT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=5_000)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target export.")
    args = parser.parse_args()
    if args.search_trials < 1:
        raise ValueError("--search-trials must be at least 1.")
    if Path(args.dataset_name).name != args.dataset_name:
        raise ValueError("--dataset-name must be a single directory name.")

    positive_rows = _read_pairs(bcb_type_dir(args.positive_variant) / "train.txt", expected_label=1)
    negative_rows = _read_pairs(bcb_type_dir(args.negative_variant) / "train.txt", expected_label=0)
    common_ids = _endpoint_ids(positive_rows) & _endpoint_ids(negative_rows)
    if not common_ids:
        raise RuntimeError("The positive and negative sources have no shared code IDs.")

    source_keys = {_canonical(left, right) for left, right, _ in positive_rows}
    contradictory = source_keys & {_canonical(left, right) for left, right, _ in negative_rows}
    if contradictory:
        raise RuntimeError(f"Sources contain {len(contradictory):,} contradictory pair labels.")

    eligible_positive = [row for row in positive_rows if row[0] in common_ids and row[1] in common_ids]
    eligible_negative = [row for row in negative_rows if row[0] in common_ids and row[1] in common_ids]
    if not eligible_positive or not eligible_negative:
        raise RuntimeError("No labelled pairs remain after source balancing.")

    all_eligible = eligible_positive + eligible_negative
    best_assignment = None
    best_kept = None
    best_discarded = None
    best_score = None
    for offset in range(args.search_trials):
        assignment = _assign_ids(common_ids, args.seed + offset, DEFAULT_SPLIT_RATIOS)
        kept, discarded = _within_split_rows(all_eligible, assignment)
        score = _score(kept)
        if best_score is None or score > best_score:
            best_assignment, best_kept, best_discarded, best_score = assignment, kept, discarded, score

    assert best_assignment is not None and best_kept is not None and best_discarded is not None
    selected = _select_balanced_rows(best_kept, args.seed)
    split_summary = _validate(selected)
    needed_ids = _endpoint_ids([row for rows in selected.values() for row in rows])

    source_dir = OUTPUTS_ROOT / "bcb" / args.source_dataset / "clean_data"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source clean export: {source_dir}")
    output_root = OUTPUTS_ROOT / "bcb" / args.dataset_name
    output_dir = output_root / "clean_data"
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target exists: {output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_dir.mkdir(parents=True)

    _write_pairs(output_dir / "pairs.csv.gz", selected)
    code_count = _filter_jsonl_gzip(source_dir / "codes.jsonl.gz", output_dir / "codes.jsonl.gz", needed_ids)
    graph_count = _filter_jsonl_gzip(source_dir / "graph_spectra.jsonl.gz", output_dir / "graph_spectra.jsonl.gz", needed_ids)
    metadata = {
        "format": "spectral_clean_data_v1",
        "dataset": args.dataset_name,
        "source_dataset": args.source_dataset,
        "source_variants": {"positive": args.positive_variant, "negative": args.negative_variant},
        "split_strategy": "code_disjoint_source_intersection_balanced",
        "leakage_controls": {
            "all_code_ids_present_in_both_label_sources": True,
            "code_disjoint_splits": True,
            "pair_duplicates_removed": True,
            "conflicting_labels": 0,
        },
        "seed": args.seed,
        "search_trials": args.search_trials,
        "eligible_before_code_split": {"clone": len(eligible_positive), "non_clone": len(eligible_negative), "common_code_ids": len(common_ids)},
        "discarded_cross_split_pairs": {"clone": best_discarded[1], "non_clone": best_discarded[0]},
        "split_labels": split_summary,
        "counts": {"codes": code_count, "graph_spectra": {"methods": graph_count}, "pairs": {"total": sum(item["pairs"] for item in split_summary.values())}},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# BCB4STRICT clean export\n\n"
        "A code-disjoint, class-balanced BCB Type-4 benchmark. Every retained code ID occurs in both the positive and curated non-clone source pools, preventing source-membership shortcuts.\n",
        encoding="utf-8",
    )
    if not args.no_zip:
        zip_path = create_clean_data_zip(output_dir, output_root / f"{args.dataset_name}.zip")
        metadata["zip"] = str(zip_path)
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[+] {args.dataset_name} ready: {output_dir}")
    for split in SPLITS:
        print(f"    {split}: {split_summary[split]}")
    print(f"    codes/graphs: {code_count:,}/{graph_count:,}; common source IDs: {len(common_ids):,}")


if __name__ == "__main__":
    main()
