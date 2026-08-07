"""Build clean-data bundles that make *any* baseline run a cross-dataset transfer.

Every baseline already trains on the ``train`` split, fits its vocabulary from
the training ids, picks its threshold on ``valid`` and reports ``test``. So a
transfer experiment needs no change to any baseline at all: it needs a bundle
whose train/valid come from the source benchmark and whose test comes from the
target one. Running the stock notebook on that bundle *is* the zero-shot
transfer, with the source-fitted vocabulary and source-selected threshold
carried over exactly as they should be.

The one thing that must not be got wrong is identity. Every benchmark numbers
its codes from 1, so merging two of them without namespacing would make
CodeXGLUE code 42 and AtCoder code 42 the same record, and every downstream
graph lookup would silently read the wrong program. Each id is therefore
prefixed with a short dataset tag before anything is merged.

Usage (from the repository root, after the datasets have been rebuilt)::

    python scripts/build_transfer_bundles.py
    python scripts/build_transfer_bundles.py --source codexglue_v3 --targets atcoder_v3
    python scripts/build_transfer_bundles.py --list

Each bundle is written to ``outputs/transfer_bundles/<name>/clean_data`` and
zipped next to it. Attach one ZIP to Kaggle and run any baseline notebook
unchanged; set its ``RUN_PROFILE`` as usual.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT.parent / "outputs"
BUNDLE_ROOT = OUTPUTS_ROOT / "transfer_bundles"

DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")
TAGS = {
    "codexglue_v3": "cx",
    "atcoder_v3": "at",
    "gptclonebench_v3": "gp",
    "semanticclonebench_v3": "sc",
}


def _open_text(path: Path):
    with path.open("rb") as probe:
        packed = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if packed else path.open("r", encoding="utf-8")


def _clean_dir(dataset: str) -> Path:
    path = OUTPUTS_ROOT / dataset / "clean_data"
    if not path.is_dir():
        raise FileNotFoundError(
            f"{dataset} has not been built: {path} is missing. "
            "Run scripts/rebuild_datasets_for_kaggle.py first."
        )
    return path


def _read_pairs(clean: Path, split: str) -> list[dict]:
    rows = []
    with _open_text(clean / "pairs.csv.gz") as stream:
        for row in csv.DictReader(stream):
            if str(row["split"]).strip().lower() == split:
                rows.append(row)
    return rows


def _tagged(tag: str, value: str) -> str:
    return f"{tag}_{value}"


def build_bundle(source: str, target: str, overwrite: bool) -> dict:
    source_clean, target_clean = _clean_dir(source), _clean_dir(target)
    source_tag, target_tag = TAGS[source], TAGS[target]

    name = f"train_{source}__test_{target}"
    out_dir = BUNDLE_ROOT / name / "clean_data"
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace it.")
        shutil.rmtree(out_dir.parent)
    out_dir.mkdir(parents=True)

    # train/valid come from the source, test from the target.
    plan = [
        (source, source_clean, source_tag, "train", "train"),
        (source, source_clean, source_tag, "valid", "valid"),
        (target, target_clean, target_tag, "test", "test"),
    ]

    needed: dict[str, set[str]] = {source: set(), target: set()}
    pair_rows: list[dict] = []
    for dataset, clean, tag, split_in, split_out in plan:
        for row in _read_pairs(clean, split_in):
            left, right = _tagged(tag, row["left_id"]), _tagged(tag, row["right_id"])
            needed[dataset].add(str(row["left_id"]))
            needed[dataset].add(str(row["right_id"]))
            pair_rows.append({
                "split": split_out, "left_id": left, "right_id": right,
                "label": row["label"], "label_name": row.get("label_name", ""),
            })

    counts = {split: sum(1 for row in pair_rows if row["split"] == split) for split in ("train", "valid", "test")}
    for split, count in counts.items():
        if count == 0:
            shutil.rmtree(out_dir.parent)
            raise RuntimeError(f"{name}: the {split} split is empty; nothing to transfer.")

    with gzip.open(out_dir / "pairs.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "left_id", "right_id", "label", "label_name"])
        writer.writeheader()
        writer.writerows(pair_rows)

    languages: set[str] = set()
    code_count = 0
    with gzip.open(out_dir / "codes.jsonl.gz", "wt", encoding="utf-8", newline="") as handle:
        for dataset, clean, tag in ((source, source_clean, source_tag), (target, target_clean, target_tag)):
            wanted = needed[dataset]
            with _open_text(clean / "codes.jsonl.gz") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    code_id = str(record.get("code_id"))
                    if code_id not in wanted:
                        continue
                    record["code_id"] = _tagged(tag, code_id)
                    # Keep the origin so a per-language or per-source breakdown
                    # can tell the two halves apart after the run.
                    record["source_dataset"] = dataset
                    languages.add(str(record.get("language", "unknown")))
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    code_count += 1

    graph_count = 0
    with gzip.open(out_dir / "graph_spectra.jsonl.gz", "wt", encoding="utf-8", newline="") as handle:
        for dataset, clean, tag in ((source, source_clean, source_tag), (target, target_clean, target_tag)):
            wanted = needed[dataset]
            with _open_text(clean / "graph_spectra.jsonl.gz") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    code_id = str(record.get("code_id"))
                    if code_id not in wanted:
                        continue
                    record["code_id"] = _tagged(tag, code_id)
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    graph_count += 1

    missing = (len(needed[source]) + len(needed[target])) - graph_count
    metadata = {
        "format": "spectral_clean_data_v1",
        "bundle": "cross_dataset_transfer",
        "train_valid_from": source,
        "test_from": target,
        "id_prefixes": {source: source_tag, target: target_tag},
        "counts": {"codes": code_count, "graph_spectra": graph_count, "pairs": counts},
        "languages": sorted(languages),
        "note": (
            "train/valid come from the source benchmark and test from the target. Running any "
            "stock baseline on this bundle performs zero-shot transfer: the vocabulary is fitted "
            "on the source training ids and the threshold is selected on the source validation "
            "split, exactly as in a normal run."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        f"# Transfer bundle: train on {source}, test on {target}\n\n"
        f"- train/valid pairs: {counts['train']:,} / {counts['valid']:,} from `{source}`\n"
        f"- test pairs: {counts['test']:,} from `{target}`\n"
        f"- codes: {code_count:,}; graph records: {graph_count:,}\n\n"
        f"Code ids are prefixed (`{source_tag}_` / `{target_tag}_`) because both benchmarks "
        "number their codes from 1.\n\nAttach the ZIP and run any baseline or the method "
        "notebook unchanged.\n",
        encoding="utf-8",
    )

    zip_path = BUNDLE_ROOT / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir.parent))

    return {
        "name": name, "zip": zip_path, "counts": counts, "codes": code_count,
        "graphs": graph_count, "missing_graphs": missing, "languages": sorted(languages),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="codexglue_v3", choices=DATASETS)
    parser.add_argument("--targets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--list", action="store_true", help="Show which datasets are built and exit.")
    args = parser.parse_args()

    if args.list:
        for dataset in DATASETS:
            path = OUTPUTS_ROOT / dataset / "clean_data"
            print(f"  {dataset:24s} {'built' if path.is_dir() else 'NOT BUILT'}  {path}")
        return 0

    targets = args.targets or [name for name in DATASETS if name != args.source]
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"source (train+valid): {args.source}")
    print(f"targets (test)      : {', '.join(targets)}\n")

    failures = []
    for target in targets:
        if target == args.source:
            continue
        try:
            report = build_bundle(args.source, target, args.overwrite)
        except (FileNotFoundError, RuntimeError, FileExistsError) as exc:
            failures.append(f"{target}: {exc}")
            print(f"  [-] {target}: {exc}")
            continue
        counts = report["counts"]
        print(
            f"  [+] {report['name']}\n"
            f"      pairs train/valid/test = {counts['train']:,}/{counts['valid']:,}/{counts['test']:,}"
            f" | codes={report['codes']:,} graphs={report['graphs']:,}"
            f"{' | MISSING GRAPHS=' + str(report['missing_graphs']) if report['missing_graphs'] else ''}\n"
            f"      languages={report['languages']}\n"
            f"      {report['zip']}"
        )

    if failures:
        print(f"\n[-] {len(failures)} bundle(s) failed.")
        return 1
    print(f"\n[+] Attach a bundle ZIP and run any baseline notebook unchanged to get its transfer score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
