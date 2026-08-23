"""Run the ATCoder Java/Python extraction pipeline and make one Kaggle upload."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.atcoder_preparation import (
    COMMON_GRAPH_TYPES,
    default_archive_path,
    default_prepared_dir,
    export_atcoder_clean_dataset,
    prepare_atcoder_dataset,
)
from spectral_code.utils.dataset_paths import output_root_for


LANGUAGES = {"java": "javasrc", "python": "pythonsrc"}
STAGES = ("01", "02", "03", "05")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _run_pipeline_script(script: str, env_overrides: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_overrides)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else os.pathsep.join([str(PROJECT_ROOT), existing_pythonpath])
    command = [sys.executable, str(PROJECT_ROOT / script)]
    print("[*]", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _language_env(language: str, prepared_dir: Path, output_root: Path) -> dict[str, str]:
    base_layers = [kind for kind in COMMON_GRAPH_TYPES if kind != "cpg"]
    return {
        "BCB_DATA_FILE": str(prepared_dir / language / "data.jsonl"),
        "BCB_DATA_DIR": str(prepared_dir),
        "OUTPUT_DIR": str(output_root / language),
        "JOERN_LANGUAGE": LANGUAGES[language],
        "PIPELINE_GRAPH_TYPES": ",".join(base_layers),
        "PIPELINE_BASE_LAYERS": ",".join(base_layers),
        "SPECTRAL_GRAPH_TYPES": ",".join(COMMON_GRAPH_TYPES),
        "BCB_MAX_METHOD_LINES": os.getenv("ATCODER_MAX_METHOD_LINES", "0"),
        "BCB_MAX_METHOD_CHARS": os.getenv("ATCODER_MAX_METHOD_CHARS", "0"),
        "BCB_MAX_LONGEST_LINE": os.getenv("ATCODER_MAX_LONGEST_LINE", "0"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=STAGES, default="01")
    parser.add_argument("--stop-after", choices=STAGES, default="05")
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument(
        "--languages",
        default="java,python",
        help="Comma-separated extraction languages; use java when repairing only the Java branch.",
    )
    args = parser.parse_args()
    if int(args.start_at) > int(args.stop_after):
        parser.error("--start-at must not be after --stop-after")
    languages = [value.strip().lower() for value in args.languages.split(",") if value.strip()]
    invalid_languages = sorted(set(languages) - set(LANGUAGES))
    if not languages or invalid_languages:
        parser.error(f"--languages must be chosen from: {', '.join(LANGUAGES)}")

    prepared_dir = default_prepared_dir()
    output_root = output_root_for("atcoder")
    include_invalid = _env_flag("ATCODER_INCLUDE_INVALID_GENERATED_NEGATIVES", False)

    if int(args.start_at) <= 1 <= int(args.stop_after):
        summary = prepare_atcoder_dataset(
            args.archive,
            prepared_dir,
            include_invalid_generated_negatives=include_invalid,
            overwrite=_env_flag("ATCODER_FORCE_PREPARE", False),
        )
        print(f"[+] Prepared {summary['function_count']:,} functions and {summary['retained_pairs_total']:,} pairs.")

    if not (prepared_dir / "data.jsonl").is_file():
        raise FileNotFoundError(f"Prepared ATCoder data missing: {prepared_dir}. Run stage 01 first.")

    if int(args.start_at) <= 2 <= int(args.stop_after):
        for language in languages:
            print(f"\n[*] Extracting {language} graphs")
            env = _language_env(language, prepared_dir, output_root)
            _run_pipeline_script("pipelines/01_extract_dataset.py", env)
            _run_pipeline_script("pipelines/02_build_graph_db.py", env)

    if int(args.start_at) <= 3 <= int(args.stop_after):
        for language in languages:
            print(f"\n[*] Extracting {language} spectra")
            _run_pipeline_script("pipelines/03_extract_spectral_features.py", _language_env(language, prepared_dir, output_root))

    if int(args.start_at) <= 5 <= int(args.stop_after):
        metadata = export_atcoder_clean_dataset(
            prepared_dir,
            output_root,
            create_zip=_env_flag("ATCODER_EXPORT_ZIP", True),
            cleanup_intermediates=not _env_flag("ATCODER_KEEP_INTERMEDIATES", False),
        )
        print(f"[+] Kaggle upload folder: {output_root / 'clean_data'}")
        if "zip" in metadata:
            print(f"[+] Kaggle upload ZIP: {metadata['zip']}")


if __name__ == "__main__":
    main()
