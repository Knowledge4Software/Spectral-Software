import os
import json
import shutil
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from dotenv import load_dotenv
load_dotenv()


from spectral_code.preprocessing.data_unpacker import unpack_jsonl_to_java
from spectral_code.preprocessing.joern_runner import run_joern_parse, run_joern_export
from spectral_code.preprocessing.graph_parser import process_single_method

JOERN_PARSE_BAT = os.getenv("JOERN_PARSE_BAT")
JOERN_EXPORT_BAT = os.getenv("JOERN_EXPORT_BAT")

DATA_FILE = os.path.join("data", "data.jsonl")
OUTPUT_DIR = os.path.join("outputs", "dataset_features")
BATCH_TEMP_DIR = os.path.join("outputs", "batch_java_src")
JOERN_BASE_OUT = os.path.join("outputs", "joern_raw_graphs")
TIMING_FILE = os.path.join("outputs", "timing_stats.json")

GRAPH_TYPES = ["ast", "cfg", "ddg", "pdg"]

def main():
    extraction_start_time = time.perf_counter()
    
    for folder in [OUTPUT_DIR, BATCH_TEMP_DIR, JOERN_BASE_OUT]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    method_ids = unpack_jsonl_to_java(DATA_FILE, BATCH_TEMP_DIR)
    if not method_ids:
        return
    
    total_methods = len(method_ids)
    batch_cpg = os.path.abspath(os.path.join("outputs", "batch_cpg.bin"))
    
    per_method_cpg_time, total_cpg_time = run_joern_parse(
        JOERN_PARSE_BAT, BATCH_TEMP_DIR, batch_cpg, total_methods
    )
    
    per_layer_export_time, layer_total_times = run_joern_export(
        JOERN_EXPORT_BAT, batch_cpg, JOERN_BASE_OUT, GRAPH_TYPES, total_methods
    )

    tasks = [
        (pos, idx, JOERN_BASE_OUT, OUTPUT_DIR, GRAPH_TYPES, per_method_cpg_time, per_layer_export_time)
        for pos, idx in enumerate(method_ids)
    ]
    
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_method, tasks), total=len(tasks), desc="Parsing DOTs into JSON"))

    shutil.rmtree(BATCH_TEMP_DIR, ignore_errors=True)
    shutil.rmtree(JOERN_BASE_OUT, ignore_errors=True)
    if os.path.exists(batch_cpg):
        os.remove(batch_cpg)

    total_duration = time.perf_counter() - extraction_start_time
    stats = {
        "total_raw_extraction_time": total_duration,
        "total_json_files": total_methods,
        "cpg_generation_time": total_cpg_time,
        "amortized_cpg_per_method_s": per_method_cpg_time
    }
    
    for gtype in GRAPH_TYPES:
        stats[f"export_time_{gtype}"] = layer_total_times[f"raw_{gtype}_export_time"]
        stats[f"amortized_export_{gtype}_ms"] = per_layer_export_time[gtype] * 1000

    with open(TIMING_FILE, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"\n[+] Success! All structures exported in {total_duration:.2f}s.")

if __name__ == "__main__":
    main()