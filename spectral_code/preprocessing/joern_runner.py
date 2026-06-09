import os
import time
import subprocess
import platform
from tqdm import tqdm


def _run_with_seconds_progress(cmd, desc: str, shell: bool = False):
    process = subprocess.Popen(
        cmd,
        shell=shell,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    with tqdm(desc=desc, unit="s") as pbar:
        while process.poll() is None:
            time.sleep(1)
            pbar.update(1)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_joern_parse(joern_parse_bat: str, src_dir: str, batch_cpg_path: str, total_methods: int):
    abs_src = os.path.abspath(src_dir)
    abs_cpg = os.path.abspath(batch_cpg_path)
    
    if os.path.exists(abs_cpg):
        os.remove(abs_cpg)
        
    print("[*] Running Joern-Parse (Warnings suppressed)...")
    start_cpg = time.perf_counter()

    _run_with_seconds_progress(
        [joern_parse_bat, abs_src, "--language", "javasrc", "--output", abs_cpg],
        desc="Joern parse",
        shell=platform.system() == "Windows",
    )

    end_cpg = time.perf_counter()
    duration = end_cpg - start_cpg
    
    return duration / total_methods if total_methods > 0 else 0, duration

def run_joern_export(joern_export_bat: str, batch_cpg_path: str, base_out_dir: str, graph_types: list[str], total_methods: int):
    abs_cpg = os.path.abspath(batch_cpg_path)
    per_layer_export_time = {}
    layer_total_times = {}

    print("[*] Running Joern-Export for base layers...")
    for gtype in tqdm(graph_types, desc="Exporting graph layers", unit="layer"):
        layer_out = os.path.abspath(os.path.join(base_out_dir, gtype))
        start_export = time.perf_counter()
        _run_with_seconds_progress(
            [joern_export_bat, abs_cpg, f"--repr={gtype}", "--out", layer_out],
            desc=f"Export {gtype.upper()}",
            shell=platform.system() == "Windows",
        )
        end_export = time.perf_counter()
        
        duration = end_export - start_export
        layer_total_times[f"raw_{gtype}_export_time"] = duration
        per_layer_export_time[gtype] = duration / total_methods if total_methods > 0 else 0
        
    return per_layer_export_time, layer_total_times
