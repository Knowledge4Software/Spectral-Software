"""Create the BCB4DATA clean export from Type-4 clones and curated non-clones."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.clean_data_export import (
    create_clean_data_zip,
    export_graph_spectra_from_sources,
    write_clean_codes,
    write_clean_pairs,
)
from spectral_code.utils.artifact_cleanup import cleanup_intermediate_artifacts
from spectral_code.utils.dataset_paths import OUTPUTS_ROOT, bcb_type_dir, output_root_for


SPLITS = ("train", "valid", "test")
GRAPH_TYPES = ("ast", "cfg", "ddg", "pdg", "cpg")
DEFAULT_TARGET_PAIRS = 380_000

# The clean BCB export follows XGLUE's original pair split sizes and label
# proportions. Train is almost balanced; valid/test deliberately keep XGLUE's
# substantially larger non-clone class.
XGLUE_REFERENCE_COUNTS = {
    "train": {"clone": 450_862, "non_clone": 450_166},
    "valid": {"clone": 53_839, "non_clone": 361_577},
    "test": {"clone": 56_820, "non_clone": 358_596},
}


def _canonical(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left <= right else (right, left)


def _read_labeled_pairs(path: Path, expected_label: int) -> list[tuple[int, int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing pair file: {path}")
    rows: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"Malformed pair on line {line_number} in {path}")
            left, right, label = map(int, fields)
            if label != expected_label:
                raise ValueError(f"Expected label={expected_label} in {path}, found {label}.")
            key = _canonical(left, right)
            if key not in seen:
                rows.append((left, right, label))
                seen.add(key)
    return rows


def _allocate_proportionally(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Allocate an exact total with largest-remainder rounding."""
    if total < 0 or not weights or sum(weights.values()) <= 0:
        raise ValueError("A non-negative total and positive allocation weights are required.")
    denominator = sum(weights.values())
    ideal = {key: total * weight / denominator for key, weight in weights.items()}
    allocated = {key: math.floor(value) for key, value in ideal.items()}
    remainder = total - sum(allocated.values())
    for key in sorted(weights, key=lambda item: (ideal[item] - allocated[item], weights[item], item), reverse=True)[:remainder]:
        allocated[key] += 1
    return allocated


def _xglue_profile_quotas(target_pairs: int) -> dict[str, dict[str, int]]:
    split_totals = _allocate_proportionally(
        target_pairs,
        {split: sum(labels.values()) for split, labels in XGLUE_REFERENCE_COUNTS.items()},
    )
    return {
        split: _allocate_proportionally(split_totals[split], XGLUE_REFERENCE_COUNTS[split])
        for split in SPLITS
    }


def _split_pairs_by_quota(
    positives: list[tuple[int, int, int]],
    negatives: list[tuple[int, int, int]],
    quotas: dict[str, dict[str, int]],
    rng: random.Random,
) -> dict[str, list[tuple[int, int, int]]]:
    splits = {split: [] for split in SPLITS}
    for label_name, rows in (("clone", positives), ("non_clone", negatives)):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        start = 0
        for split in SPLITS:
            stop = start + quotas[split][label_name]
            splits[split].extend(shuffled[start:stop])
            start = stop
        if start != len(shuffled):
            raise RuntimeError(f"Unused {label_name} pairs while applying the XGLUE split profile.")
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def _read_codes(source_dirs: list[Path], needed_ids: set[int]) -> dict[int, str]:
    codes: dict[int, str] = {}
    for source_dir in source_dirs:
        data_path = source_dir / "data.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing source code file: {data_path}")
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                method_id = int(row["idx"])
                if method_id not in needed_ids:
                    continue
                code = row["func"]
                if method_id in codes and codes[method_id] != code:
                    raise RuntimeError(f"Conflicting code for method {method_id}.")
                codes[method_id] = code
    missing = needed_ids - set(codes)
    if missing:
        raise RuntimeError(f"{len(missing):,} selected methods are absent from source data files.")
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable BCB4DATA clean export.")
    parser.add_argument("--positive-variant", default="4")
    parser.add_argument("--negative-variant", default="non_clone")
    parser.add_argument("--dataset-name", default="BCB4DATA")
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=DEFAULT_TARGET_PAIRS,
        help=(
            f"Total exported pairs (default: {DEFAULT_TARGET_PAIRS:,}). "
            "The XGLUE label profile needs more non-clones than clones; 400,000 exceeds "
            "the available unique BCB non-clone pairs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--float-precision", type=int, default=8)
    parser.add_argument("--no-zip", action="store_true", help="Skip the Kaggle-uploadable zip archive.")
    parser.add_argument(
        "--cleanup-raw-intermediates",
        "--cleanup-source-intermediates",
        dest="cleanup_raw_intermediates",
        action="store_true",
        help=(
            "Remove only regenerable Joern raw files and diagnostics after export. "
            "clean_graphs and spectral_features are always preserved."
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Replace an existing BCB4DATA export (the default).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail instead of replacing an existing BCB4DATA export.",
    )
    args = parser.parse_args()

    if args.target_pairs < 3:
        raise ValueError("--target-pairs must be an integer of at least 3.")
    if args.float_precision < -1:
        raise ValueError("--float-precision must be -1 or non-negative.")

    positive_dir = bcb_type_dir(args.positive_variant)
    negative_dir = bcb_type_dir(args.negative_variant)
    positive_root = output_root_for("bcb", args.positive_variant)
    negative_root = output_root_for("bcb", args.negative_variant)
    if Path(args.dataset_name).name != args.dataset_name:
        raise ValueError("--dataset-name must be a single directory name.")
    bcb_exports_root = (OUTPUTS_ROOT / "bcb").resolve()
    output_root = (bcb_exports_root / args.dataset_name).resolve()
    output_dir = output_root / "clean_data"
    if bcb_exports_root not in output_root.parents:
        raise ValueError(f"Refusing to write outside the BCB outputs directory: {output_root}")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_root}. Omit --no-overwrite to replace it.")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    positives = _read_labeled_pairs(positive_dir / "train.txt", expected_label=1)
    negatives = _read_labeled_pairs(negative_dir / "train.txt", expected_label=0)
    quotas = _xglue_profile_quotas(args.target_pairs)
    positive_count = sum(quotas[split]["clone"] for split in SPLITS)
    negative_count = sum(quotas[split]["non_clone"] for split in SPLITS)
    if positive_count > len(positives) or negative_count > len(negatives):
        raise RuntimeError(
            f"Need {positive_count:,} positive and {negative_count:,} negative pairs; "
            f"available: {len(positives):,} positive, {len(negatives):,} negative."
        )
    selected_positives = rng.sample(positives, positive_count)
    selected_negatives = rng.sample(negatives, negative_count)
    if {_canonical(left, right) for left, right, _ in selected_positives} & {
        _canonical(left, right) for left, right, _ in selected_negatives
    }:
        raise RuntimeError("Selected sources contain contradictory pair labels.")

    splits = _split_pairs_by_quota(selected_positives, selected_negatives, quotas, rng)
    for split, rows in splits.items():
        clone_count = sum(label == 1 for _, _, label in rows)
        non_clone_count = sum(label == 0 for _, _, label in rows)
        expected = quotas[split]
        if clone_count != expected["clone"] or non_clone_count != expected["non_clone"]:
            raise RuntimeError(
                f"{split} does not match the XGLUE profile: "
                f"clone={clone_count:,}/{expected['clone']:,}, "
                f"non_clone={non_clone_count:,}/{expected['non_clone']:,}."
            )
    needed_ids = {method_id for rows in splits.values() for left, right, _ in rows for method_id in (left, right)}
    codes = _read_codes([positive_dir, negative_dir], needed_ids)
    pair_counts = write_clean_pairs(output_dir, splits)
    write_clean_codes(output_dir, codes)
    graph_summary = export_graph_spectra_from_sources(
        [positive_root, negative_root],
        output_dir / "graph_spectra.jsonl.gz",
        needed_ids,
        GRAPH_TYPES,
        None if args.float_precision == -1 else args.float_precision,
    )

    split_counts = {
        split: {
            "pairs": len(rows),
            "clone": sum(label == 1 for _, _, label in rows),
            "non_clone": sum(label == 0 for _, _, label in rows),
        }
        for split, rows in splits.items()
    }
    metadata: dict[str, object] = {
        "format": "spectral_clean_data_v1",
        "dataset": args.dataset_name,
        "source_variants": {"positive": args.positive_variant, "negative": args.negative_variant},
        "graph_types": list(GRAPH_TYPES),
        "float_precision": "full" if args.float_precision == -1 else args.float_precision,
        "split_strategy": "xglue_pair_stratified_profile",
        "class_balance": "matches XGLUE's per-split clone/non_clone proportions",
        "xglue_reference_counts": XGLUE_REFERENCE_COUNTS,
        "counts": {
            "codes": len(codes),
            "pairs": {
                **pair_counts,
                "clone": positive_count,
                "non_clone": negative_count,
                "total": args.target_pairs,
            },
            "split_labels": split_counts,
            "graph_spectra": graph_summary,
        },
    }
    if not args.no_zip:
        zip_path = output_root / f"{args.dataset_name}.zip"
        metadata["zip"] = str(zip_path)
    if args.cleanup_raw_intermediates:
        source_cleanup = []
        for source_root in (positive_root, negative_root):
            summary = cleanup_intermediate_artifacts(
                source_root,
                include_dataset_features=True,
                include_legacy_dirs=True,
                include_post_spectral_diagnostics=True,
                compute_size=True,
            )
            source_cleanup.append(summary)
        metadata["source_cleanup"] = source_cleanup

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# BCB4DATA clean export\n\n"
        "Portable clean-data schema shared with XGLUE: code records, labelled pairs, "
        "and sparse graph/spectral records. Graph adjacency includes node types and labels "
        "for structural AST models.\n",
        encoding="utf-8",
    )
    if not args.no_zip:
        create_clean_data_zip(output_dir, Path(metadata["zip"]))

    print(f"[+] BCB4DATA ready: {output_dir}")
    for split, counts in split_counts.items():
        print(f"    {split}: {counts['pairs']:,} pairs ({counts['clone']:,} clone, {counts['non_clone']:,} non-clone)")
    print(f"    codes: {len(codes):,}; graph records: {graph_summary['methods']:,}")
    if not args.no_zip:
        print(f"    zip: {metadata['zip']}")


if __name__ == "__main__":
    main()
