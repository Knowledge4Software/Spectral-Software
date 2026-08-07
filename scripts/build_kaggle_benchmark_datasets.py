"""Build fresh, portable graph/spectral Kaggle datasets from ``data/DataSets.zip``.

This is intentionally the one local command for the V3 releases.  It keeps
languages separate during Joern extraction, preserves the official splits and
labels, then merges all language branches into one ``clean_data`` ZIP per
dataset.  The resulting archives can be attached directly to the SPECTRA-Siam
and baseline Kaggle notebooks.

Examples
--------
Dry-run the exact plan (no writes)::

    python scripts/build_kaggle_benchmark_datasets.py --dry-run

Build everything from scratch, deleting only prior *V3* build artefacts::

    python scripts/build_kaggle_benchmark_datasets.py --clean-first

Resume one failed language without deleting completed work::

    python scripts/build_kaggle_benchmark_datasets.py --datasets gptclonebench_v3 --languages python

Run from the repository root.  Joern is resolved from ``JOERN_HOME``, PATH,
or the project's existing ``C:/joern-cli`` fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.v3_benchmark_preparation import (
    GRAPH_TYPES,
    SPECS,
    default_archive_path,
    default_prepared_dir,
    export_v3_clean_dataset,
    prepare_v3_benchmark,
)
from spectral_code.utils.dataset_paths import OUTPUTS_ROOT, output_root_for


DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")
JOERN_LANGUAGES = {"java": "javasrc", "python": "pythonsrc", "c": "c", "csharp": "csharpsrc"}
LEGACY_OUTPUTS = {"codexglue_v3": output_root_for("xglue"), "atcoder_v3": output_root_for("atcoder")}


def _run(script: str, env_updates: dict[str, str], dry_run: bool) -> None:
    command = [sys.executable, str(PROJECT_ROOT / script)]
    if dry_run:
        print("[dry-run]", " ".join(command), "OUTPUT_DIR=", env_updates["OUTPUT_DIR"])
        return
    env = os.environ.copy()
    env.update(env_updates)
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not old_pythonpath else os.pathsep.join([str(PROJECT_ROOT), old_pythonpath])
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _language_env(prepared_dir: Path, output_root: Path, language: str) -> dict[str, str]:
    base_layers = [kind for kind in GRAPH_TYPES if kind != "cpg"]
    return {
        "BCB_DATA_FILE": str(prepared_dir / language / "data.jsonl"),
        "BCB_DATA_DIR": str(prepared_dir),
        "OUTPUT_DIR": str(output_root / language),
        "JOERN_LANGUAGE": JOERN_LANGUAGES[language],
        "PIPELINE_GRAPH_TYPES": ",".join(base_layers),
        "PIPELINE_BASE_LAYERS": ",".join(base_layers),
        "SPECTRAL_GRAPH_TYPES": ",".join(GRAPH_TYPES),
        # Bounded chunks make a single malformed submission recoverable.
        "JOERN_PARSE_CHUNK_SIZE": os.getenv("JOERN_PARSE_CHUNK_SIZE", "500"),
        "PIPELINE_CLEAN_INTERMEDIATE_ARTIFACTS": "1",
        "PIPELINE_CLEAN_RAW_FEATURES": "1",
    }


def _validate_clean_output(dataset: str, output_root: Path) -> dict:
    clean = output_root / "clean_data"
    required = [clean / name for name in ("codes.jsonl.gz", "pairs.csv.gz", "graph_spectra.jsonl.gz", "metadata.json", "README.md")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"{dataset}: incomplete final clean-data output: {missing}")
    metadata = json.loads((clean / "metadata.json").read_text(encoding="utf-8"))
    counts = metadata["counts"]
    if counts["graph_spectra"]["methods"] != counts["codes"]:
        raise RuntimeError(f"{dataset}: graph/code coverage mismatch: {counts['graph_spectra']['methods']} vs {counts['codes']}")
    return metadata


def _clean_targets(dataset: str) -> list[Path]:
    # Do not silently delete legacy XGLUE/ATCoder/BCB exports.  This runner
    # writes versioned V3 outputs and only replaces its own artefacts.
    return [default_prepared_dir(dataset), output_root_for(dataset)]


def _remove_legacy_if_requested(dataset: str, output_root: Path, requested: bool) -> None:
    legacy = LEGACY_OUTPUTS.get(dataset)
    if requested and legacy and legacy.exists() and legacy.resolve() != output_root.resolve():
        print(f"[remove legacy after success] {legacy}")
        shutil.rmtree(legacy)


def _record_count(path: Path) -> int:
    with path.open("rb") as src:
        return sum(block.count(b"\n") for block in iter(lambda: src.read(1024 * 1024), b""))


def _language_is_complete(prepared: Path, output_root: Path, language: str) -> bool:
    expected = _record_count(prepared / language / "data.jsonl")
    for relative in ("clean_graphs/graph_shards_manifest.json", "spectral_features/spectral_features_manifest.json"):
        manifest_path = output_root / language / relative
        if not manifest_path.is_file():
            return False
        try:
            if int(json.loads(manifest_path.read_text(encoding="utf-8")).get("total_methods", -1)) != expected:
                return False
        except (ValueError, json.JSONDecodeError):
            return False
    # A manifest can be present even when a frontend produced empty graphs.
    # Re-run that language if its primary AST coverage is below publication
    # tolerance; this catches the earlier C# Joern-export failure on resume.
    stats_path = output_root / language / "timing_stats.json"
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        ast_ready = int(stats.get("spectral_computed_graphs_ast", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return ast_ready / max(expected, 1) >= 0.99


def _require_language_dependencies(language: str) -> None:
    """Fail before a costly extraction when its mandatory parser is absent."""
    if language != "csharp":
        return
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_c_sharp  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "C# AST recovery requires tree-sitter and tree-sitter-c-sharp in the active Python environment. "
            "Run: python -m pip install -r requirements.txt"
        ) from exc


def build_dataset(dataset: str, archive: Path, languages_filter: set[str] | None, clean_first: bool, remove_legacy_after_success: bool, refresh_prepared: bool, package_only: bool, dry_run: bool, force: bool = False) -> None:
    prepared = default_prepared_dir(dataset)
    output_root = output_root_for(dataset)
    if force and not dry_run and output_root.exists():
        # A rebuild triggered by an extraction fix must not stop at the stale
        # archive it replaces, nor resume from language branches extracted by
        # the old code: a half-old, half-new bundle is the worst outcome.
        print(f"[force] removing previous output tree {output_root}")
        shutil.rmtree(output_root)
    if clean_first:
        for target in _clean_targets(dataset):
            if target.exists():
                print(f"[clean] {target}")
                if not dry_run:
                    shutil.rmtree(target)
    if dry_run:
        print(f"[dry-run] prepare {dataset} from {archive}")
        return
    if not clean_first and not refresh_prepared and not package_only and not force:
        try:
            ready = _validate_clean_output(dataset, output_root)
            if (output_root / f"{dataset}_clean_data.zip").is_file():
                print(f"[skip ready] {dataset}: {ready['counts']['codes']:,} codes already packaged.")
                _remove_legacy_if_requested(dataset, output_root, remove_legacy_after_success)
                return
        except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError):
            pass
    report = prepare_v3_benchmark(
        dataset,
        archive,
        prepared,
        overwrite=refresh_prepared,
        # A fresh V3 build must never depend on a legacy graph-ID mapping.
        reuse_existing_atcoder_ids=False,
    )
    languages = sorted(report["codes_by_language"])
    if package_only:
        if languages_filter is not None:
            raise ValueError("--package-only cannot be combined with --languages.")
        languages = []
    if languages_filter is not None:
        invalid = languages_filter - set(languages)
        if invalid:
            raise ValueError(f"{dataset}: requested absent languages: {sorted(invalid)}")
        languages = [language for language in languages if language in languages_filter]
    if not languages and not package_only:
        raise RuntimeError(f"{dataset}: no languages selected.")
    print(f"\n=== {dataset}: {report['code_count']:,} codes; languages={languages} ===")
    for language in languages:
        if languages_filter is None and _language_is_complete(prepared, output_root, language):
            print(f"[skip complete language] {dataset}/{language}")
            continue
        _require_language_dependencies(language)
        env = _language_env(prepared, output_root, language)
        _run("pipelines/01_extract_dataset.py", env, dry_run)
        _run("pipelines/02_build_graph_db.py", env, dry_run)
        _run("pipelines/03_extract_spectral_features.py", env, dry_run)
    # A partial-language run is a repair operation, not a publishable dataset.
    if languages_filter is not None and set(languages) != set(report["codes_by_language"]):
        print(f"[resume] {dataset}: selected languages finished; rerun without --languages to package the full dataset.")
        return
    metadata = export_v3_clean_dataset(
        dataset, prepared, output_root,
        # Fresh IDs force a full extraction; never reuse a legacy ATCoder ZIP.
        cleanup_intermediates=True, create_zip=True,
    )
    checked = _validate_clean_output(dataset, output_root)
    print(f"[+] {dataset} ready: {metadata['zip']}")
    print(f"    codes={checked['counts']['codes']:,}, pairs={checked['counts']['pairs']['total']:,}, graph layers={checked['counts']['graph_spectra']['graph_layers']:,}")
    _remove_legacy_if_requested(dataset, output_root, remove_legacy_after_success)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--languages", help="Optional comma-separated repair subset: java,python,c,csharp")
    parser.add_argument("--clean-first", action="store_true", help="Delete only prior V3 prepared/output folders before rebuilding.")
    parser.add_argument("--remove-legacy-after-success", action="store_true", help="After a validated replacement ZIP exists, remove legacy outputs/xglue and outputs/atcoder if replaced.")
    parser.add_argument("--refresh-prepared", action="store_true", help="Regenerate data.jsonl from the archive while retaining existing graph outputs.")
    parser.add_argument("--package-only", action="store_true", help="Do not extract graphs; only validate and package existing language outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without reading/writing dataset outputs.")
    parser.add_argument("--force", action="store_true", help="Rebuild even when a packaged clean_data ZIP already exists.")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        parser.error(f"Archive not found: {archive}")
    language_filter = None
    if args.languages:
        language_filter = {value.strip().lower() for value in args.languages.split(",") if value.strip()}
        invalid = language_filter - set(JOERN_LANGUAGES)
        if invalid:
            parser.error(f"Unsupported languages: {sorted(invalid)}")
    print("Output root:", OUTPUTS_ROOT)
    for dataset in args.datasets:
        build_dataset(dataset, archive, language_filter, args.clean_first, args.remove_legacy_after_success, args.refresh_prepared, args.package_only, args.dry_run, args.force)
    if "codexglue_v3" in args.datasets:
        print("\nNote: codexglue_v3 is the fresh, official-split replacement for legacy XGLUE output.")
    print("\nBigCloneBench is intentionally not included: DataSets.zip contains its raw 22M-function relational release, not an official binary train/valid/test split. A defensible BCB protocol (negative construction and code-disjoint split) must be selected before any graph extraction.")


if __name__ == "__main__":
    main()
