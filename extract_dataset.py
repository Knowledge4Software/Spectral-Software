import os
import json
import subprocess
import shutil
import time
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==================== CONFIG ====================
DATA_FILE = os.path.join("data", "data.jsonl")
OUTPUT_DIR = os.path.join("outputs", "dataset_features")

BATCH_TEMP_DIR = os.path.join("outputs", "batch_java_src")
JOERN_BASE_OUT = os.path.join("outputs", "joern_raw_graphs")

JOERN_PARSE_BAT = r"C:\joern-cli\joern-parse.bat"
JOERN_EXPORT_BAT = r"C:\joern-cli\joern-export.bat"

GRAPH_TYPES = ["ast", "cfg", "ddg", "pdg"]

def process_single_method(args):
    """
    Worker function executed in parallel to parse DOT files for a single method.
    """
    position_idx, idx, per_method_cpg_time, per_layer_export_time = args
    results = {}
    
    for gtype in GRAPH_TYPES:
        layer_folder = os.path.join(JOERN_BASE_OUT, gtype)
        target_dot = f"{position_idx}-{gtype}.dot"
        file_path = os.path.join(layer_folder, target_dot)

        graph = nx.DiGraph()
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                graph = nx.drawing.nx_pydot.read_dot(file_path)
            except Exception:
                pass
                    
        graph_data = nx.node_link_data(graph) if graph.number_of_nodes() > 0 else None
        
        cpg_t = per_method_cpg_time
        export_t = per_layer_export_time[gtype]
        
        results[gtype] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "graph_data": graph_data,
            "eigenvalues": None,
            "metrics": {
                "cpg_time": cpg_t,
                "export_time": export_t,
                "total_graph_time": cpg_t + export_t,
                "spectral_time": 0.0,
                "total_time": cpg_t + export_t
            }
        }
        
    out_file = os.path.join(OUTPUT_DIR, f"{idx}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"idx": int(idx), "features": results}, f)
        
    return True

def main():
    # Reset and prepare fresh directories
    for folder in [OUTPUT_DIR, BATCH_TEMP_DIR, JOERN_BASE_OUT]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"[-] Data file not found: {DATA_FILE}")
        return

    # -------------------------------------------------------------
    # STEP 1: Unpack JSONL methods into standard Java source files
    # -------------------------------------------------------------
    print("[*] Step 1: Unpacking JSONL methods into Java classes...")
    method_ids = []
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for line in tqdm(lines, desc="Unpacking Java files"):
        record = json.loads(line)
        idx = str(record.get("idx", "unknown"))
        func_code = record.get("func", "")
        
        valid_java = f"public class Method_{idx} {{\n{func_code}\n}}"
        with open(os.path.join(BATCH_TEMP_DIR, f"Method_{idx}.java"), "w", encoding="utf-8") as jf:
            jf.write(valid_java)
        method_ids.append(idx)
        
    total_methods = len(method_ids)
    print(f"[+] Generated {total_methods} Java source files.")

    # -------------------------------------------------------------
    # STEP 2: Execute a SINGLE Joern-Parse for the entire batch
    # -------------------------------------------------------------
    print(f"\n[*] Step 2: Running Joern-Parse for {total_methods} files...")
    batch_cpg = os.path.join("outputs", "batch_cpg.bin")
    if os.path.exists(batch_cpg):
        os.remove(batch_cpg)
        
    start_cpg = time.perf_counter()
    subprocess.run([JOERN_PARSE_BAT, BATCH_TEMP_DIR, "--output", batch_cpg], 
                   capture_output=True, text=True, shell=True)
    end_cpg = time.perf_counter()
    
    per_method_cpg_time = (end_cpg - start_cpg) / total_methods if total_methods > 0 else 0
    print(f"[+] Bulk CPG created. Amortized CPG time per method: {per_method_cpg_time:.4f}s")

    # -------------------------------------------------------------
    # STEP 3: Execute Joern-Export for all 4 layers
    # -------------------------------------------------------------
    print("\n[*] Step 3: Exporting all 4 layers (AST, CFG, DDG, PDG)...")
    per_layer_export_time = {}

    for gtype in GRAPH_TYPES:
        layer_out = os.path.join(JOERN_BASE_OUT, gtype)
        
        start_export = time.perf_counter()
        subprocess.run([JOERN_EXPORT_BAT, batch_cpg, f"--repr={gtype}", "--out", layer_out], 
                       capture_output=True, text=True, shell=True)
        end_export = time.perf_counter()
        
        per_layer_export_time[gtype] = (end_export - start_export) / total_methods if total_methods > 0 else 0
        print(f"[+] Layer '{gtype.upper()}' exported.")

    # -------------------------------------------------------------
    # STEP 4: Parallel Mapping and Parsing using Multiprocessing
    # -------------------------------------------------------------
    num_workers = cpu_count()
    print(f"\n[*] Step 4: Parallel parsing and mapping using {num_workers} CPU cores...")
    
    # Prepare task arguments for the process pool
    tasks = [
        (position_idx, idx, per_method_cpg_time, per_layer_export_time)
        for position_idx, idx in enumerate(method_ids)
    ]
    
    # Execute tasks across multiple CPU cores
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_single_method, tasks), total=len(tasks), desc="Saving Dataset Features"))

    # Clean up heavy raw directories and cache binaries
    shutil.rmtree(BATCH_TEMP_DIR, ignore_errors=True)
    shutil.rmtree(JOERN_BASE_OUT, ignore_errors=True)
    if os.path.exists(batch_cpg):
        os.remove(batch_cpg)
        
    print(f"\n[+] Success! All {total_methods} methods mapped and saved perfectly to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()