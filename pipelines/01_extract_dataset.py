import os
import sys
import json
import shutil
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path
from dotenv import load_dotenv

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
from spectral_code.preprocessing.data_unpacker import unpack_jsonl_to_java
from spectral_code.preprocessing.joern_runner import run_joern_parse, run_joern_export
from spectral_code.preprocessing.graph_parser import process_single_method, build_dot_index

load_dotenv()

JOERN_PARSE_BAT = os.getenv("JOERN_PARSE_BAT")
JOERN_EXPORT_BAT = os.getenv("JOERN_EXPORT_BAT")
DEFAULT_DATA_FILE = PROJECT_ROOT / "bench_data" / "bcb_full_type1" / "data.jsonl"
DATA_FILE = Path(os.getenv("BCB_DATA_FILE", str(DEFAULT_DATA_FILE)))

# Temporary folders inside the output root (helps with disk space and speed)
BATCH_TEMP_DIR = OUTPUT_ROOT / "batch_java_src"
JOERN_BASE_OUT = OUTPUT_ROOT / "joern_raw_graphs"
BATCH_CPG_BIN = OUTPUT_ROOT / "batch_cpg.bin"

GRAPH_TYPES = ["ast", "cfg", "ddg", "pdg"]

def main():
    extraction_start_time = time.perf_counter()

    with tqdm(total=8, desc="Pipeline 01", unit="step") as pipeline_bar:
        # Initialize directory structure
        ensure_dirs()
        pipeline_bar.update(1)

        # Cleanup previous temp runs
        for folder in tqdm([RAW_FEATURES_DIR, BATCH_TEMP_DIR, JOERN_BASE_OUT], desc="Preparing folders", unit="folder"):
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
        pipeline_bar.update(1)

        print(f"[*] Unpacking {DATA_FILE} to {BATCH_TEMP_DIR}...")
        method_ids = unpack_jsonl_to_java(str(DATA_FILE), str(BATCH_TEMP_DIR))
        if not method_ids:
            print("[-] No methods found. Check your data file.")
            return
        pipeline_bar.update(1)

        total_methods = len(method_ids)

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

        # 3. Build Global Index of DOT files for robust mapping
        print("[*] Building DOT mapping index...")
        dot_index = build_dot_index(str(JOERN_BASE_OUT), GRAPH_TYPES)
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
            tasks.append((idx, paths, str(RAW_FEATURES_DIR), per_method_cpg_time, per_layer_export_time))

        print(f"[*] Parsing {len(tasks)} DOT files into JSON format...")
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(
                pool.imap_unordered(process_single_method, tasks),
                total=len(tasks),
                desc="JSON conversion",
                unit="method",
            ))
        pipeline_bar.update(1)

        # Cleanup large temporary files
        for path in tqdm([BATCH_TEMP_DIR, JOERN_BASE_OUT, BATCH_CPG_BIN], desc="Cleaning temporary files", unit="path"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
        pipeline_bar.update(1)

    # Save timing stats
    total_duration = time.perf_counter() - extraction_start_time
    stats = {
        "total_raw_extraction_time": total_duration,
        "total_methods": total_methods,
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
