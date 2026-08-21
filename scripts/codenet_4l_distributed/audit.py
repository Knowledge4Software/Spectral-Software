"""Audit distributed CodeNet 50k-clone + 50k-non-clone caches/releases."""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECTRALS_ROOT = PROJECT_ROOT.parent
LANGUAGES = ("python", "java", "cpp", "csharp")
GRAPH_TYPES = ("ast", "cfg", "ddg", "cpg")
EXPECTED_ENDPOINTS = 135_068
EXPECTED_PAIRS = 100_000
SPECTRAL_LIMIT = 128


def _targets(prepared: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    with (prepared / "data.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result[str(row["source_code_id"])] = row
    return result


def _selected_targets(scope: str) -> dict[str, dict]:
    combined = _targets(SPECTRALS_ROOT / "data" / "codenet_4l_clone50k_diff50k_prepared")
    clones = _targets(SPECTRALS_ROOT / "data" / "codenet_4l_clone_50k_prepared")
    if scope == "combined":
        return combined
    if scope == "clone":
        return clones
    return {source_id: row for source_id, row in combined.items() if source_id not in clones}


def _cache_rows(cache_dir: Path) -> dict[str, tuple[str, str]]:
    index = cache_dir / "index.sqlite3"
    if not index.is_file() or not (cache_dir / "shards").is_dir():
        raise FileNotFoundError(f"Cache must contain index.sqlite3 and shards: {cache_dir}")
    connection = sqlite3.connect(f"file:{index.as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(source_id): (str(source_sha256), str(shard))
            for source_id, source_sha256, shard in connection.execute(
                "SELECT source_code_id, source_sha256, shard FROM records"
            )
        }
    finally:
        connection.close()


def _matching(targets: dict[str, dict], indexed: dict[str, tuple[str, str]]) -> set[str]:
    result = set()
    for source_id, target in targets.items():
        cached = indexed.get(source_id)
        if cached is None:
            continue
        target_hash = str(target.get("source_sha256", ""))
        if target_hash and cached[0] and target_hash != cached[0]:
            continue
        result.add(source_id)
    return result


def _check_spectra(
    cache_dir: Path,
    indexed: dict[str, tuple[str, str]],
    selected: set[str],
) -> tuple[int, int]:
    by_shard: dict[str, set[str]] = defaultdict(set)
    for source_id in selected:
        by_shard[indexed[source_id][1]].add(source_id)
    found: set[str] = set()
    short: list[tuple[str, str, int]] = []
    for shard_name, wanted in sorted(by_shard.items()):
        shard = cache_dir / "shards" / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Indexed cache shard is missing: {shard}")
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                source_id = str(record.get("source_code_id", ""))
                if source_id not in wanted or indexed[source_id][1] != shard_name:
                    continue
                found.add(source_id)
                graphs = record.get("graphs", {})
                if set(graphs) != set(GRAPH_TYPES):
                    raise RuntimeError(f"{source_id} does not contain exactly {GRAPH_TYPES}")
                for kind in GRAPH_TYPES:
                    layer = graphs[kind]
                    status = str(layer.get("spectral_status", layer.get("status", ""))).lower()
                    values = layer.get("eigenvalues") or []
                    if "sparse" in status and "fail" not in status and len(values) < SPECTRAL_LIMIT:
                        short.append((source_id, kind, len(values)))
    missing_records = selected - found
    if missing_records:
        raise RuntimeError(f"{len(missing_records):,} indexed records were not found in their current shards")
    if short:
        example = ", ".join(f"{sid}/{kind}={count}" for sid, kind, count in short[:5])
        raise RuntimeError(
            f"{len(short):,} successful sparse layers are shorter than {SPECTRAL_LIMIT}: {example}"
        )
    return len(found), len(short)


def _find_zip_member(archive: zipfile.ZipFile, filename: str) -> str:
    matches = [name for name in archive.namelist() if Path(name).name == filename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} in ZIP; found {matches}")
    return matches[0]


def _audit_zip(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Final ZIP is missing: {path}")
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read(_find_zip_member(archive, "metadata.json")))
        counts = metadata.get("counts", {})
        pairs = counts.get("pairs", {})
        if int(counts.get("codes", -1)) != EXPECTED_ENDPOINTS:
            raise RuntimeError(f"Final ZIP code count is not {EXPECTED_ENDPOINTS:,}: {counts.get('codes')}")
        if (
            int(pairs.get("clone", -1)) != 50_000
            or int(pairs.get("non_clone", -1)) != 50_000
            or int(pairs.get("total", -1)) != EXPECTED_PAIRS
        ):
            raise RuntimeError(f"Final ZIP pair counts are wrong: {pairs}")
        consumer = metadata.get("consumer_schema", {})
        if int(consumer.get("eigenvalue_limit_per_layer", -1)) != SPECTRAL_LIMIT:
            raise RuntimeError(f"Final ZIP does not declare the {SPECTRAL_LIMIT}-value contract")
        graph_member = _find_zip_member(archive, "graph_spectra.jsonl.gz")
        graph_count = 0
        short_sparse = 0
        with archive.open(graph_member) as raw, gzip.open(raw, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                graph_count += 1
                graphs = row.get("graphs", {})
                if set(graphs) != set(GRAPH_TYPES):
                    raise RuntimeError(f"ZIP graph record {row.get('code_id')} has wrong graph layers")
                for layer in graphs.values():
                    status = str(layer.get("spectral_status", "")).lower()
                    values = layer.get("eigenvalues") or []
                    if len(values) > SPECTRAL_LIMIT:
                        raise RuntimeError("ZIP contains a spectrum longer than its declared limit")
                    if "sparse" in status and "fail" not in status and len(values) < SPECTRAL_LIMIT:
                        short_sparse += 1
        if graph_count != EXPECTED_ENDPOINTS or short_sparse:
            raise RuntimeError(
                f"ZIP graph audit failed: records={graph_count:,}, short_sparse={short_sparse:,}"
            )
    print(f"[OK] Final ZIP: {path}")
    print(f"     {EXPECTED_PAIRS:,} pairs = 50,000 clone + 50,000 non-clone")
    print(f"     {EXPECTED_ENDPOINTS:,} endpoint graph records; AST/CFG/DDG/CPG; sparse K={SPECTRAL_LIMIT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("clone", "nonclone", "combined"), default="combined")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--check-spectra", action="store_true")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    targets = _selected_targets(args.scope)
    indexed = _cache_rows(cache_dir)
    present = _matching(targets, indexed)
    missing = set(targets) - present
    present_by_language = Counter(str(targets[source_id]["lang"]) for source_id in present)
    missing_by_language = Counter(str(targets[source_id]["lang"]) for source_id in missing)
    print(f"[AUDIT] scope={args.scope} target={len(targets):,} cached={len(present):,} remaining={len(missing):,}")
    print("        cached_by_language=", {lang: present_by_language[lang] for lang in LANGUAGES})
    print("        remaining_by_language=", {lang: missing_by_language[lang] for lang in LANGUAGES})
    if args.require_complete and missing:
        raise RuntimeError(f"Cache is incomplete: {len(missing):,} target endpoints remain")
    if args.check_spectra:
        checked, short = _check_spectra(cache_dir, indexed, present)
        print(f"[OK] Checked {checked:,} current cache records; short successful sparse layers={short}")
    if args.zip:
        _audit_zip(args.zip.resolve())


if __name__ == "__main__":
    main()
