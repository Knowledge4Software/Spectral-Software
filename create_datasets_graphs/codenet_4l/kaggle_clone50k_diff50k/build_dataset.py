r"""Create the graph-free Kaggle release for the fixed CodeNet 50k+50k study.

Output layout intentionally mirrors the source CodeNet archive for the two
needed pair kinds:

* 50,000 fixed clone pairs (the exact clone sample prepared for graphing);
* 50,000 different-problem non-clone pairs;
* 5,000 pairs per language configuration and per pair kind;
* 3,500/750/750 train/validation/test pairs in every bucket.

The output includes only the four Program parquet files filtered to referenced
endpoints and no graph artefacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.codenet_preparation import (
    ARCHIVE_SPLITS,
    CONFIGURATIONS,
    LANGUAGES,
    SAMPLE_SEED,
    SPLITS,
    _sample_rank,
    _sample_targets,
    default_archive_path,
    default_prepared_dir,
)
from spectral_code.preprocessing.language_support import normalize_source_language
from spectral_code.utils.dataset_paths import DATA_ROOT


PAIR_KINDS = ("clone", "nonclone_diff_problem")
PAIR_SIZE_PER_KIND = 50_000
PAIR_SIZE_PER_CONFIGURATION = 5_000
SPLIT_QUOTAS = {"train": 3_500, "valid": 750, "test": 750}


def _prepared_clone_pair_ids(prepared_dir: Path) -> dict[tuple[str, str], set[str]]:
    """Read the exact clone IDs chosen by the existing 50k graph target."""
    path = prepared_dir / "pair_provenance.jsonl.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared 50k clone provenance is missing: {path}")
    result = {(configuration, split): set() for configuration in CONFIGURATIONS for split in SPLITS}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("pair_kind") != "clone":
                continue
            split = str(row["split"])
            configuration = str(row["configuration_id"])
            result[(configuration, split)].add(str(row["pair_id"]))
    invalid = {
        f"{configuration}/{split}": len(ids)
        for (configuration, split), ids in result.items()
        if len(ids) != SPLIT_QUOTAS[split]
    }
    if invalid:
        raise RuntimeError(f"Prepared clone target is not the expected fixed 50k sample: {invalid}")
    return result


def _selected_indices(table: pa.Table, count: int) -> list[int]:
    pair_ids = table["pair_id"].to_pylist()
    if len(pair_ids) < count:
        raise RuntimeError(f"Source pair bucket has {len(pair_ids):,} rows; expected at least {count:,}")
    ranked = heapq.nsmallest(
        count,
        ((hashlib.sha256(f"{SAMPLE_SEED}:{pair_id}".encode("utf-8")).digest(), index)
         for index, pair_id in enumerate(pair_ids)),
    )
    return [index for _, index in ranked]


def _read_table(archive: zipfile.ZipFile, member: str) -> pa.Table:
    try:
        return pq.read_table(io.BytesIO(archive.read(member)))
    except KeyError as exc:
        raise FileNotFoundError(f"Source archive member is missing: {member}") from exc


def _write_pairs_and_collect_endpoints(
    archive: zipfile.ZipFile,
    output_dir: Path,
    prepared_clone_ids: dict[tuple[str, str], set[str]],
) -> tuple[dict[str, set[str]], list[dict[str, object]]]:
    endpoints = {language: set() for language in LANGUAGES}
    summary: list[dict[str, object]] = []
    for pair_kind in PAIR_KINDS:
        quotas = _sample_targets(PAIR_SIZE_PER_KIND, CONFIGURATIONS, (pair_kind,))
        for configuration in CONFIGURATIONS:
            for split in SPLITS:
                archive_split = ARCHIVE_SPLITS[split]
                member = f"pairs__{configuration}__{pair_kind}__{archive_split}.parquet"
                table = _read_table(archive, member)
                expected_count = quotas[(configuration, pair_kind, split)]
                if expected_count != SPLIT_QUOTAS[split]:
                    raise RuntimeError(f"Unexpected quota for {configuration}/{pair_kind}/{split}: {expected_count}")
                indices = _selected_indices(table, expected_count)
                selected = table.take(pa.array(indices, type=pa.int64()))
                pair_ids = {str(value) for value in selected["pair_id"].to_pylist()}
                if pair_kind == "clone" and pair_ids != prepared_clone_ids[(configuration, split)]:
                    raise RuntimeError(
                        f"Clone selection drift for {configuration}/{split}; refusing to publish a different sample."
                    )
                labels = set(selected["label"].to_pylist())
                expected_label = 1 if pair_kind == "clone" else 0
                if labels != {expected_label}:
                    raise RuntimeError(f"Unexpected labels in {member}: {labels}")
                if set(selected["configuration_id"].to_pylist()) != {configuration}:
                    raise RuntimeError(f"Configuration mismatch in {member}")
                if set(selected["pair_kind"].to_pylist()) != {pair_kind}:
                    raise RuntimeError(f"Pair kind mismatch in {member}")
                if set(selected["split"].to_pylist()) != {archive_split}:
                    raise RuntimeError(f"Split mismatch in {member}")

                output_path = output_dir / member
                pq.write_table(selected, output_path, compression="zstd", version="2.6")
                for side in ("a", "b"):
                    ids = selected[f"program_id_{side}"].to_pylist()
                    languages = selected[f"language_{side}"].to_pylist()
                    for source_id, language in zip(ids, languages):
                        normalized = normalize_source_language(str(language))
                        if normalized not in endpoints:
                            raise RuntimeError(f"Unsupported language {language!r} in {member}")
                        endpoints[normalized].add(str(source_id))
                summary.append({
                    "configuration_id": configuration,
                    "pair_kind": pair_kind,
                    "split": archive_split,
                    "pairs": len(selected),
                    "unique_pair_ids": len(pair_ids),
                    "output_file": member,
                })
    return endpoints, summary


def _write_program_subset(
    archive: zipfile.ZipFile,
    output_dir: Path,
    endpoints: dict[str, set[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for language in LANGUAGES:
        member = f"programs__{language}.parquet"
        raw = io.BytesIO(archive.read(member))
        source = pq.ParquetFile(raw)
        program_index = source.schema_arrow.get_field_index("program_id")
        if program_index < 0:
            raise RuntimeError(f"program_id column is missing from {member}")
        wanted = pa.array(sorted(endpoints[language]), type=pa.string())
        output_path = output_dir / member
        writer = pq.ParquetWriter(output_path, source.schema_arrow, compression="zstd", version="2.6")
        written = 0
        try:
            for batch in source.iter_batches(batch_size=10_000):
                mask = pc.is_in(batch.column(program_index), value_set=wanted)
                selected = batch.filter(mask)
                if selected.num_rows:
                    writer.write_batch(selected)
                    written += selected.num_rows
        finally:
            writer.close()
        if written != len(endpoints[language]):
            raise RuntimeError(
                f"Program subset {language} wrote {written:,}/{len(endpoints[language]):,} referenced endpoints"
            )
        counts[language] = written
    return counts


def _write_metadata(
    output_dir: Path,
    *,
    archive: Path,
    endpoints: dict[str, set[str]],
    program_counts: dict[str, int],
    bucket_summary: list[dict[str, object]],
    prepared_dir: Path,
) -> None:
    metadata = {
        "format": "codenet_4l_pair_release_v1",
        "dataset": "CodeNet 4L fixed 50k clone + 50k different-problem non-clone subset",
        "source_archive": str(archive),
        "clone_pair_source": str(prepared_dir),
        "pair_kinds": list(PAIR_KINDS),
        "total_pairs": 100_000,
        "pairs_per_pair_kind": 50_000,
        "pairs_per_configuration_per_kind": PAIR_SIZE_PER_CONFIGURATION,
        "split_quotas_per_configuration_per_kind": {
            "train": 3_500,
            "validation": 750,
            "test": 750,
        },
        "sampling": {
            "method": "deterministic SHA-256 rank over original pair_id",
            "seed": SAMPLE_SEED,
            "clone_selection_matches_current_graph_target": True,
        },
        "endpoint_counts_by_language": {language: len(endpoints[language]) for language in LANGUAGES},
        "program_counts_by_language": program_counts,
        "contains_graphs": False,
        "pair_files": bucket_summary,
    }
    (output_dir / "build_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output_dir / "bucket_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bucket_summary[0]))
        writer.writeheader()
        writer.writerows(bucket_summary)
    (output_dir / "README.md").write_text(
        "# CodeNet 4L fixed 50k clone + 50k diff-problem release\n\n"
        "This graph-free Kaggle upload contains 100,000 labelled pairs: 50,000 clones and 50,000 "
        "different-problem non-clones. Each of the ten language configurations contributes exactly 5,000 pairs "
        "per pair kind, split 3,500/750/750 across train/validation/test. The clone pair IDs are identical to the "
        "fixed 50k target used by the graph-construction pipeline. Pair and program Parquet schemas are preserved "
        "from the source CodeNet release; only unneeded pair kinds, unreferenced programs, and graph artefacts are omitted.\n",
        encoding="utf-8",
    )


def _zip_directory(output_dir: Path, zip_path: Path) -> None:
    temporary = zip_path.with_name(zip_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output_dir.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    temporary.replace(zip_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument("--prepared-clones", type=Path, default=DATA_ROOT / "codenet_4l_clone_50k_prepared")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs" / "kaggle_datasets" / "codenet_4l_clone50k_diff50k",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=PROJECT_ROOT.parent / "outputs" / "kaggle_datasets" / "codenet_4l_clone50k_diff50k.zip",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory/ZIP.")
    args = parser.parse_args()
    archive_path = args.archive.resolve()
    prepared_dir = args.prepared_clones.resolve()
    output_dir = args.output_dir.resolve()
    zip_path = args.zip_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"CodeNet source archive is missing: {archive_path}")
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_dir}; pass --force to replace it.")
        shutil.rmtree(output_dir)
    if zip_path.exists() and not args.force:
        raise FileExistsError(f"ZIP already exists: {zip_path}; pass --force to replace it.")
    output_dir.mkdir(parents=True, exist_ok=False)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Building graph-free CodeNet Kaggle pair release")
    print(f"    source:   {archive_path}")
    print(f"    clones:   {prepared_dir}")
    print(f"    output:   {output_dir}")
    print(f"    zip:      {zip_path}")
    prepared_clone_ids = _prepared_clone_pair_ids(prepared_dir)
    with zipfile.ZipFile(archive_path) as source:
        endpoints, bucket_summary = _write_pairs_and_collect_endpoints(source, output_dir, prepared_clone_ids)
        program_counts = _write_program_subset(source, output_dir, endpoints)
    if len(bucket_summary) != 60:
        raise RuntimeError(f"Expected 60 pair buckets; wrote {len(bucket_summary)}")
    if sum(int(row["pairs"]) for row in bucket_summary) != 100_000:
        raise RuntimeError("Pair counts do not sum to 100,000")
    _write_metadata(
        output_dir,
        archive=archive_path,
        endpoints=endpoints,
        program_counts=program_counts,
        bucket_summary=bucket_summary,
        prepared_dir=prepared_dir,
    )
    _zip_directory(output_dir, zip_path)
    print(f"[+] Pair files: {len(bucket_summary)}")
    print(f"[+] Program rows: {sum(program_counts.values()):,} ({program_counts})")
    print(f"[+] Kaggle ZIP: {zip_path}")


if __name__ == "__main__":
    main()

