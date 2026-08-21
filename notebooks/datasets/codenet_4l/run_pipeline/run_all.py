"""Prepare, graph, spectral-export, and package Project CodeNet 4L.

Examples (from the repository root):
  python notebooks/datasets/codenet_4l/run_pipeline/run_all.py --stop-after 01
  python notebooks/datasets/codenet_4l/run_pipeline/run_all.py --sample-size 800 --stop-after 01 --force-prepare
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.codenet_preparation import (
    CONFIGURATIONS,
    DEFAULT_SAMPLE_SIZE,
    GRAPH_TYPES,
    LANGUAGES,
    PAIR_KINDS,
    default_archive_path,
    default_prepared_dir,
    export_codenet_clean_dataset,
    prepare_codenet_dataset,
)
from spectral_code.preprocessing.language_support import joern_language
from spectral_code.utils.dataset_paths import output_root_for


STAGES = ("01", "02", "03", "05")


def _csv_values(raw: str, allowed: tuple[str, ...], option: str) -> list[str]:
    values = [value.strip().lower() for value in raw.split(",") if value.strip()]
    invalid = sorted(set(values) - set(allowed))
    if not values or invalid:
        raise ValueError(f"{option} must be chosen from {allowed}; invalid={invalid}")
    return values


def _run(script: str, env_updates: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(PROJECT_ROOT / script)]
    print("[*]", " ".join(command))
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        if exc.returncode in {130, -1073741510}:
            print("\n[!] CodeNet pipeline interrupted; rerun the same command to restart the current language safely.")
            raise SystemExit(130) from None
        raise


def _language_env(
    prepared: Path,
    output_root: Path,
    language: str,
    graph_types: list[str],
) -> dict[str, str]:
    base_layers = [kind for kind in graph_types if kind != "cpg"]
    return {
        "BCB_DATA_FILE": str(prepared / language / "data.jsonl"),
        "BCB_DATA_DIR": str(prepared),
        "OUTPUT_DIR": str(output_root / language),
        "JOERN_LANGUAGE": joern_language(language),
        "PIPELINE_GRAPH_TYPES": ",".join(base_layers),
        "PIPELINE_BASE_LAYERS": ",".join(base_layers),
        "SPECTRAL_GRAPH_TYPES": ",".join(graph_types),
        "JOERN_PARSE_CHUNK_SIZE": os.getenv("JOERN_PARSE_CHUNK_SIZE", "500"),
        "BCB_MAX_METHOD_LINES": os.getenv("CODENET_MAX_PROGRAM_LINES", "0"),
        "BCB_MAX_METHOD_CHARS": os.getenv("CODENET_MAX_PROGRAM_CHARS", "0"),
        "BCB_MAX_LONGEST_LINE": os.getenv("CODENET_MAX_LONGEST_LINE", "0"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument("--prepared-dir", type=Path, default=default_prepared_dir())
    parser.add_argument("--output-dir", type=Path, default=output_root_for("codenet_4l"))
    parser.add_argument("--start-at", choices=STAGES, default="01")
    parser.add_argument("--stop-after", choices=STAGES, default="05")
    parser.add_argument("--configurations", default=",".join(CONFIGURATIONS))
    parser.add_argument("--pair-kinds", default=",".join(PAIR_KINDS))
    parser.add_argument(
        "--graph-types",
        default=",".join(GRAPH_TYPES),
        help="Comma-separated graph layers to extract, spectrally encode, and export.",
    )
    parser.add_argument("--languages", help="Comma-separated graph-extraction subset")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Optional uniform pair cap; omitted means all rows in the selected source buckets.",
    )
    parser.add_argument("--min-program-lines", type=int, help="Inclusive minimum for both pair endpoints")
    parser.add_argument("--max-program-lines", type=int, help="Inclusive maximum for both pair endpoints")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    if int(args.start_at) > int(args.stop_after):
        parser.error("--start-at must not be after --stop-after")
    try:
        configurations = _csv_values(args.configurations, CONFIGURATIONS, "--configurations")
        pair_kinds = _csv_values(args.pair_kinds, PAIR_KINDS, "--pair-kinds")
        graph_types = _csv_values(args.graph_types, tuple(GRAPH_TYPES), "--graph-types")
    except ValueError as exc:
        parser.error(str(exc))

    prepared = args.prepared_dir.resolve()
    output_root = args.output_dir.resolve()
    if int(args.start_at) <= 1 <= int(args.stop_after):
        report = prepare_codenet_dataset(
            args.archive,
            prepared,
            configurations=configurations,
            pair_kinds=pair_kinds,
            sample_size=args.sample_size,
            min_program_lines=args.min_program_lines,
            max_program_lines=args.max_program_lines,
            overwrite=args.force_prepare,
        )
        print(f"[+] Prepared {report['code_count']:,} codes and {sum(v['pairs'] for v in report['pairs'].values()):,} pairs")
    metadata_path = prepared / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Prepared CodeNet input missing: {prepared}; run stage 01 first.")
    metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
    if int(args.start_at) > 1:
        prepared_mode = metadata.get("sampling_mode")
        prepared_sample_size = int(metadata.get("sample_size", 0))
        requested_line_filter = (
            None
            if args.min_program_lines is None and args.max_program_lines is None
            else {
                "minimum": args.min_program_lines,
                "maximum": args.max_program_lines,
                "inclusive": True,
                "applies_to": "both_pair_endpoints",
            }
        )
        if args.sample_size is None and prepared_mode != "full":
            parser.error(
                f"Prepared data is a {prepared_sample_size:,}-pair subset, but this command requests the full "
                f"selected release. Pass --sample-size {prepared_sample_size}, or rebuild stage 01 with --force-prepare."
            )
        if args.sample_size is not None and prepared_sample_size != args.sample_size:
            parser.error(
                f"Prepared data contains {prepared_sample_size:,} pairs, but --sample-size is {args.sample_size:,}. "
                "Run stage 01 with --force-prepare or pass the prepared size."
            )
        if metadata.get("configurations") != configurations or metadata.get("pair_kinds") != pair_kinds:
            parser.error(
                "Prepared configurations/pair kinds do not match this command. Pass the same --configurations and "
                "--pair-kinds used for stage 01, or rebuild with --force-prepare."
            )
        if metadata.get("program_line_filter") != requested_line_filter:
            parser.error(
                "Prepared source-line filter does not match this command. Pass the same "
                "--min-program-lines/--max-program-lines values used for stage 01, or rebuild with --force-prepare."
            )
    available_languages = sorted(metadata["codes_by_language"])
    languages = available_languages
    if args.languages:
        try:
            languages = _csv_values(args.languages, LANGUAGES, "--languages")
        except ValueError as exc:
            parser.error(str(exc))
        missing = sorted(set(languages) - set(available_languages))
        if missing:
            parser.error(f"Prepared data has no programs for languages: {missing}")

    for language in languages:
        env = _language_env(prepared, output_root, language, graph_types)
        if int(args.start_at) <= 2 <= int(args.stop_after):
            _run("pipelines/01_extract_dataset.py", env)
            _run("pipelines/02_build_graph_db.py", env)
        if int(args.start_at) <= 3 <= int(args.stop_after):
            _run("pipelines/03_extract_spectral_features.py", env)

    if int(args.start_at) <= 5 <= int(args.stop_after):
        if args.languages and set(languages) != set(available_languages):
            raise RuntimeError("Final export requires every prepared language; rerun stage 05 without --languages.")
        report = export_codenet_clean_dataset(
            prepared,
            output_root,
            graph_types=graph_types,
            create_zip=not args.no_zip,
            cleanup_intermediates=not args.keep_intermediates,
        )
        print(f"[+] Clean data: {output_root / 'clean_data'}")
        if "zip" in report:
            print(f"[+] ZIP: {report['zip']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] CodeNet pipeline interrupted; rerun the same command to restart the current language safely.")
        raise SystemExit(130)
