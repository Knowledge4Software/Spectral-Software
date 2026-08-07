import os
import sys
import json
import shutil
import time
from tqdm import tqdm
from multiprocessing import cpu_count
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.exists():
            return False

        loaded = False
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded = True
        return loaded

# Ensure project root is in sys.path for professional imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import our new path utility
from spectral_code.utils.project_paths import (
    RAW_FEATURES_DIR, 
    OUTPUT_ROOT, 
    TIMING_STATS_FILE, 
    ensure_dirs
)
from spectral_code.utils.dataset_paths import bcb_type_dir
from spectral_code.preprocessing.data_unpacker import unpack_jsonl_to_java
from spectral_code.preprocessing.joern_runner import run_joern_parse, run_joern_export, run_joern_method_map
from spectral_code.preprocessing.graph_parser import (
    process_single_method,
    build_dot_index,
    build_dot_index_with_method_map,
    threaded_bounded_map,
)

load_dotenv()

JOERN_PARSE_BAT = os.getenv("JOERN_PARSE_BAT")
JOERN_EXPORT_BAT = os.getenv("JOERN_EXPORT_BAT")
BCB_CLONE_TYPE = os.getenv("BCB_CLONE_TYPE", "1")
DEFAULT_DATA_FILE = bcb_type_dir(BCB_CLONE_TYPE) / "data.jsonl"
DATA_FILE = Path(os.getenv("BCB_DATA_FILE", str(DEFAULT_DATA_FILE)))

# Temporary folders inside the output root (helps with disk space and speed)
RUN_ID = os.getenv("BCB_RUN_ID", f"{int(time.time())}_{os.getpid()}")
BATCH_TEMP_DIR = OUTPUT_ROOT / f"batch_src_{RUN_ID}"
JOERN_BASE_OUT = OUTPUT_ROOT / f"joern_raw_graphs_{RUN_ID}"
BATCH_CPG_BIN = OUTPUT_ROOT / f"batch_cpg_{RUN_ID}.bin"
BATCH_CPG_CHUNKS = BATCH_CPG_BIN.with_name(f"{BATCH_CPG_BIN.stem}_chunks")
BATCH_CPG_MANIFEST = Path(f"{BATCH_CPG_BIN}.chunks.json")
SKIPPED_METHODS_FILE = OUTPUT_ROOT / "skipped_methods_pipeline01.jsonl"
GRAPH_PARSE_SKIPPED_FILE = OUTPUT_ROOT / "skipped_graph_parse_pipeline01.jsonl"

GRAPH_TYPES = [g.strip().lower() for g in os.getenv("PIPELINE_GRAPH_TYPES", "ast,cfg,ddg,pdg").split(",") if g.strip()]
USE_LEGACY_DOT_MAPPING = os.getenv("JOERN_LEGACY_DOT_MAPPING", "0").strip().lower() in {"1", "true", "yes"}


def _parse_nonnegative_int_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name, str(default)).strip()
    if raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    if raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _io_worker_count(env_name: str, multiplier: int = 2, max_default: int = 32) -> int:
    default = min(max_default, max(4, cpu_count() * multiplier))
    return _parse_positive_int_env(env_name, default)


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _cleanup_temp_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except PermissionError as exc:
        print(f"[!] Could not delete temporary path still in use: {path} ({exc})")


def main():
    extraction_start_time = time.perf_counter()

    with tqdm(total=8, desc="Pipeline 01", unit="step") as pipeline_bar:
        # Initialize directory structure
        ensure_dirs()
        pipeline_bar.update(1)

        # Fresh run folders avoid Windows lock failures on stale Joern log files.
        for folder in tqdm([RAW_FEATURES_DIR, BATCH_TEMP_DIR, JOERN_BASE_OUT], desc="Preparing folders", unit="folder"):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
        if SKIPPED_METHODS_FILE.exists():
            SKIPPED_METHODS_FILE.unlink()
        if GRAPH_PARSE_SKIPPED_FILE.exists():
            GRAPH_PARSE_SKIPPED_FILE.unlink()
        pipeline_bar.update(1)

        print(f"[*] Unpacking {DATA_FILE} to {BATCH_TEMP_DIR}...")
        max_method_lines = _parse_nonnegative_int_env("BCB_MAX_METHOD_LINES", 0)
        max_method_chars = _parse_nonnegative_int_env("BCB_MAX_METHOD_CHARS", 0)
        max_longest_line = _parse_nonnegative_int_env("BCB_MAX_LONGEST_LINE", 0)
        if max_method_lines or max_method_chars or max_longest_line:
            print("[*] Method size filter enabled:")
            print(f"    BCB_MAX_METHOD_LINES={max_method_lines or 'disabled'}")
            print(f"    BCB_MAX_METHOD_CHARS={max_method_chars or 'disabled'}")
            print(f"    BCB_MAX_LONGEST_LINE={max_longest_line or 'disabled'}")
            print(f"    skipped report: {SKIPPED_METHODS_FILE}")

        method_ids = unpack_jsonl_to_java(
            str(DATA_FILE),
            str(BATCH_TEMP_DIR),
            max_lines=max_method_lines,
            max_chars=max_method_chars,
            max_longest_line=max_longest_line,
            skipped_report_path=str(SKIPPED_METHODS_FILE),
        )
        if not method_ids:
            print("[-] No methods found. Check your data file.")
            return
        pipeline_bar.update(1)

        total_methods = len(method_ids)
        skipped_methods = _count_lines(SKIPPED_METHODS_FILE)
        if skipped_methods:
            print(f"[*] Skipped {skipped_methods:,} oversized methods before Joern parse.")

        # 1. Joern Parse
        per_method_cpg_time, total_cpg_time = run_joern_parse(
            JOERN_PARSE_BAT, str(BATCH_TEMP_DIR), str(BATCH_CPG_BIN), total_methods
        )
        pipeline_bar.update(1)

        # 2. Joern Export
        per_layer_export_time, layer_total_times = run_joern_export(
            JOERN_EXPORT_BAT, str(BATCH_CPG_BIN), str(JOERN_BASE_OUT), GRAPH_TYPES, total_methods
        )
        pipeline_bar.update(1)

        # 3. Build Global Index of DOT files for robust mapping.
        #
        # A DOT names its method, never its file, so the legacy index guessed the
        # owner from an ``m_<id>`` marker or from export order. That silently
        # mismapped every C submission and reduced each multi-function file to one
        # arbitrary function. The CPG's own method table removes the guesswork.
        print("[*] Building DOT mapping index...")
        if USE_LEGACY_DOT_MAPPING:
            print("[!] JOERN_LEGACY_DOT_MAPPING=1: using marker/ordinal DOT mapping. "
                  "This is unsafe for C and for files defining several functions.")
            dot_index = build_dot_index(
                str(JOERN_BASE_OUT),
                GRAPH_TYPES,
                method_ids=method_ids,
                chunk_manifest_path=str(BATCH_CPG_MANIFEST),
            )
        else:
            method_map = run_joern_method_map(str(BATCH_CPG_BIN))
            dot_index = build_dot_index_with_method_map(
                str(JOERN_BASE_OUT),
                GRAPH_TYPES,
                method_map,
                method_ids=method_ids,
            )
        pipeline_bar.update(1)

        missing_dot_mappings = {
            gtype: total_methods - len(dot_index.get(gtype, {}))
            for gtype in GRAPH_TYPES
        }
        print("[*] DOT mapping coverage:")
        for gtype in GRAPH_TYPES:
            mapped = len(dot_index.get(gtype, {}))
            missing = missing_dot_mappings[gtype]
            print(f"    {gtype.upper()}: mapped={mapped:,} missing={missing:,}")

        # 4. Parallel DOT to JSON Parsing
        tasks = []
        for idx in tqdm(method_ids, desc="Preparing conversion tasks", unit="method"):
            paths = {g: dot_index[g].get(idx) for g in GRAPH_TYPES}
            tasks.append((
                idx,
                paths,
                str(RAW_FEATURES_DIR),
                per_method_cpg_time,
                per_layer_export_time,
                str(BATCH_TEMP_DIR),
            ))

        print(f"[*] Parsing {len(tasks)} DOT files into JSON format...")
        conversion_workers = _io_worker_count("DOT_CONVERSION_WORKERS")
        print(f"[*] DOT conversion workers: {conversion_workers}")
        parse_results = list(tqdm(
            threaded_bounded_map(process_single_method, tasks, max_workers=conversion_workers),
            total=len(tasks),
            desc="JSON conversion",
            unit="method",
        ))
        skipped_graph_layers = [
            skipped
            for result in parse_results
            for skipped in result.get("skipped_layers", [])
        ]
        fallback_graph_layers = [
            fallback
            for result in parse_results
            for fallback in result.get("fallback_layers", [])
        ]
        if skipped_graph_layers:
            with GRAPH_PARSE_SKIPPED_FILE.open("w", encoding="utf-8") as f:
                for skipped in skipped_graph_layers:
                    f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
            print(f"[*] Logged {len(skipped_graph_layers):,} skipped/empty graph layers to {GRAPH_PARSE_SKIPPED_FILE}.")
        if fallback_graph_layers:
            print(f"[*] Recovered {len(fallback_graph_layers):,} graph layers with source-code fallbacks.")
        pipeline_bar.update(1)

        # Cleanup large temporary files
        for path in tqdm(
            [BATCH_TEMP_DIR, JOERN_BASE_OUT, BATCH_CPG_BIN, BATCH_CPG_CHUNKS, BATCH_CPG_MANIFEST],
            desc="Cleaning temporary files",
            unit="path",
        ):
            _cleanup_temp_path(path)
        pipeline_bar.update(1)

    # Save timing stats
    total_duration = time.perf_counter() - extraction_start_time
    stats = {
        "total_raw_extraction_time": total_duration,
        "total_methods": total_methods,
        "skipped_methods_pipeline01": skipped_methods,
        "skipped_methods_file": str(SKIPPED_METHODS_FILE) if skipped_methods else None,
        "skipped_graph_layers_pipeline01": len(skipped_graph_layers),
        "skipped_graph_layers_file": str(GRAPH_PARSE_SKIPPED_FILE) if skipped_graph_layers else None,
        "fallback_graph_layers_pipeline01": len(fallback_graph_layers),
        "python_ast_fallback_layers_pipeline01": sum(
            1
            for fallback in fallback_graph_layers
            if fallback.get("source") == "python_ast_fallback"
        ),
        "python_cfg_fallback_layers_pipeline01": sum(
            1
            for fallback in fallback_graph_layers
            if fallback.get("source") == "python_cfg_fallback"
        ),
        "python_ddg_fallback_layers_pipeline01": sum(
            1
            for fallback in fallback_graph_layers
            if fallback.get("source") == "python_ddg_fallback"
        ),
        "max_method_lines_filter": max_method_lines,
        "max_method_chars_filter": max_method_chars,
        "max_longest_line_filter": max_longest_line,
        "cpg_generation_time": total_cpg_time,
        "amortized_cpg_per_method_s": per_method_cpg_time
    }
    
    for gtype in GRAPH_TYPES:
        stats[f"export_time_{gtype}"] = layer_total_times[f"raw_{gtype}_export_time"]
        stats[f"amortized_export_{gtype}_ms"] = per_layer_export_time[gtype] * 1000
        stats[f"dot_mapped_{gtype}"] = total_methods - missing_dot_mappings[gtype]
        stats[f"dot_missing_{gtype}"] = missing_dot_mappings[gtype]

    with open(TIMING_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\n[+] Success! All structures exported to {RAW_FEATURES_DIR}")
    print(f"[+] Total time: {total_duration:.2f}s.")

if __name__ == "__main__":
    main()
