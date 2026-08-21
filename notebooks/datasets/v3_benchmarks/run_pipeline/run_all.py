"""Prepare, graph, spectral-export, and package a V3 benchmark.

Examples (run from repository root):
  python notebooks/datasets/v3_benchmarks/run_pipeline/run_all.py atcoder_v3 --stop-after 01
  python notebooks/datasets/v3_benchmarks/run_pipeline/run_all.py gptclonebench_v3
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

from spectral_code.evaluation.v3_benchmark_preparation import (
    GRAPH_TYPES, SPECS, default_archive_path, default_prepared_dir,
    export_v3_clean_dataset, prepare_atcoder_v3_repair_subset, prepare_v3_benchmark,
)
from spectral_code.utils.dataset_paths import output_root_for
from spectral_code.preprocessing.language_support import joern_language


JOERN_LANGUAGES = {
    language: joern_language(language)
    for language in ("java", "python", "c", "cpp", "csharp")
}
STAGES = ("01", "02", "03", "05")


def _run(script: str, env_updates: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(env_updates)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(PROJECT_ROOT / script)], cwd=PROJECT_ROOT, env=env, check=True)


def _language_env(prepared: Path, root: Path, language: str) -> dict[str, str]:
    return {
        "BCB_DATA_FILE": str(prepared / language / "data.jsonl"),
        "BCB_DATA_DIR": str(prepared), "OUTPUT_DIR": str(root / language),
        "JOERN_LANGUAGE": JOERN_LANGUAGES[language],
        "PIPELINE_GRAPH_TYPES": ",".join([kind for kind in GRAPH_TYPES if kind != "cpg"]),
        "PIPELINE_BASE_LAYERS": ",".join([kind for kind in GRAPH_TYPES if kind != "cpg"]),
        "SPECTRAL_GRAPH_TYPES": ",".join(GRAPH_TYPES),
        "JOERN_PARSE_CHUNK_SIZE": os.getenv("JOERN_PARSE_CHUNK_SIZE", "500"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(SPECS))
    parser.add_argument("--archive", type=Path, default=default_archive_path())
    parser.add_argument("--start-at", choices=STAGES, default="01")
    parser.add_argument("--stop-after", choices=STAGES, default="05")
    parser.add_argument("--languages", help="Comma-separated subset, e.g. java,python")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--repair-missing-atcoder-v3", action="store_true", help="Parse only ATCoder V3 codes absent from the older graph export.")
    args = parser.parse_args()
    if int(args.start_at) > int(args.stop_after):
        parser.error("--start-at must not be after --stop-after")
    prepared, root = default_prepared_dir(args.dataset), output_root_for(args.dataset)
    if int(args.start_at) <= 1 <= int(args.stop_after):
        report = prepare_v3_benchmark(args.dataset, args.archive, prepared, overwrite=args.force_prepare)
        print(f"[+] Prepared {report['dataset']}: {report['code_count']:,} codes; {report['pairs']}")
    if not (prepared / "metadata.json").is_file():
        raise FileNotFoundError(f"Prepared input missing: {prepared}; run stage 01 first.")
    metadata = __import__("json").loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    languages = sorted(metadata["codes_by_language"])
    if args.languages:
        requested = [value.strip().lower() for value in args.languages.split(",") if value.strip()]
        invalid = sorted(set(requested) - set(languages))
        if invalid:
            parser.error(f"Languages unavailable in prepared data: {invalid}")
        languages = requested
    repair_only = args.repair_missing_atcoder_v3
    if repair_only:
        if args.dataset != "atcoder_v3":
            parser.error("--repair-missing-atcoder-v3 is only valid for atcoder_v3.")
        if int(args.start_at) > 2 or int(args.stop_after) < 3:
            parser.error("Repair needs stages 02 and 03.")
        report = prepare_atcoder_v3_repair_subset(prepared)
        print(f"[+] ATCoder V3 repair subset: {report}")
        languages = [language for language in languages if report.get(language, 0)]
    for language in languages:
        if repair_only:
            env = _language_env(prepared / "repair", root / "repair", language)
        else:
            env = _language_env(prepared, root, language)
        if int(args.start_at) <= 2 <= int(args.stop_after):
            _run("pipelines/01_extract_dataset.py", env)
            _run("pipelines/02_build_graph_db.py", env)
        if int(args.start_at) <= 3 <= int(args.stop_after):
            _run("pipelines/03_extract_spectral_features.py", env)
    if int(args.start_at) <= 5 <= int(args.stop_after):
        if repair_only:
            raise RuntimeError("Repair extraction is complete; rerun stage 05 without --repair-missing-atcoder-v3.")
        if args.languages:
            raise RuntimeError("Final export needs every language; rerun stage 05 without --languages.")
        report = export_v3_clean_dataset(args.dataset, prepared, root)
        print(f"[+] Kaggle zip: {report['zip']}")


if __name__ == "__main__":
    main()
