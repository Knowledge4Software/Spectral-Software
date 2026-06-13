import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from tqdm import tqdm


def _chunk_manifest_path(cpg_path: str) -> str:
    return f"{cpg_path}.chunks.json"


def _chunk_root(cpg_path: str) -> Path:
    cpg = Path(cpg_path)
    return cpg.with_name(f"{cpg.stem}_chunks")


def _bat_safe_cmd(cmd: list[str]) -> list[str]:
    if os.name == "nt" and cmd and str(cmd[0]).lower().endswith((".bat", ".cmd")):
        return ["cmd.exe", "/d", "/c", *cmd]
    return cmd


def _tail(path: str, max_lines: int = 80) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except OSError:
        return ""


def _run_with_seconds_progress(cmd: list[str], desc: str, log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"    log: {log_path}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            _bat_safe_cmd(cmd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        with tqdm(desc=desc, unit="s") as pbar:
            while process.poll() is None:
                time.sleep(1)
                pbar.update(1)

    if process.returncode != 0:
        log_tail = _tail(log_path)
        if log_tail:
            print(f"\n[-] {desc} failed. Last log lines:\n{log_tail}\n")
        raise subprocess.CalledProcessError(process.returncode, cmd)


def _joern_parse_cmd(joern_parse_bat: str, src_dir: str, cpg_path: str) -> list[str]:
    # Use joern-parse, not javasrc2cpg directly: joern-parse applies the default
    # semantic overlays required for non-empty CFG/DDG/PDG exports.
    return [joern_parse_bat, src_dir, "--language", "javasrc", "--output", cpg_path]


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _parse_chunk_size(total_methods: int) -> int:
    raw = os.getenv("JOERN_PARSE_CHUNK_SIZE", "10000").strip()
    try:
        chunk_size = int(raw)
    except ValueError:
        chunk_size = 10000

    if chunk_size <= 0:
        return total_methods
    return chunk_size


def run_joern_parse(joern_parse_bat: str, src_dir: str, batch_cpg_path: str, total_methods: int):
    abs_src = os.path.abspath(src_dir)
    abs_cpg = os.path.abspath(batch_cpg_path)

    if not joern_parse_bat:
        raise ValueError("JOERN_PARSE_BAT is not configured.")
    if not os.path.isdir(abs_src):
        raise FileNotFoundError(f"Joern source directory not found: {abs_src}")

    if os.path.exists(abs_cpg):
        os.remove(abs_cpg)
    manifest_path = _chunk_manifest_path(abs_cpg)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    chunk_root = _chunk_root(abs_cpg)
    if chunk_root.exists():
        shutil.rmtree(chunk_root)

    java_files = sorted(Path(abs_src).glob("*.java"))
    chunk_size = _parse_chunk_size(total_methods)

    print("[*] Running Joern parse with default overlays...")
    print("[*] Chunking keeps memory bounded while preserving CFG/DDG/PDG overlays.")

    start_cpg = time.perf_counter()

    if len(java_files) > chunk_size:
        chunk_root.mkdir(parents=True, exist_ok=True)
        logs_dir = chunk_root / "logs"
        chunks = []

        for chunk_idx, start in enumerate(range(0, len(java_files), chunk_size)):
            chunk_files = java_files[start:start + chunk_size]
            chunk_src = chunk_root / f"src_{chunk_idx:04d}"
            chunk_src.mkdir(parents=True, exist_ok=True)
            chunk_cpg = chunk_root / f"cpg_{chunk_idx:04d}.bin"
            chunk_log = logs_dir / f"parse_{chunk_idx:04d}.log"

            for src_file in tqdm(chunk_files, desc=f"Preparing parse chunk {chunk_idx + 1}", unit="file"):
                _link_or_copy(src_file, chunk_src / src_file.name)

            if chunk_cpg.exists():
                chunk_cpg.unlink()

            _run_with_seconds_progress(
                _joern_parse_cmd(joern_parse_bat, str(chunk_src), str(chunk_cpg)),
                desc=f"Joern parse chunk {chunk_idx + 1}",
                log_path=str(chunk_log),
            )

            shutil.rmtree(chunk_src, ignore_errors=True)
            chunks.append({
                "index": chunk_idx,
                "cpg_path": str(chunk_cpg),
                "method_count": len(chunk_files),
                "log_path": str(chunk_log),
            })

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"chunked": True, "chunks": chunks, "total_methods": total_methods}, f, indent=2)
    else:
        _run_with_seconds_progress(
            _joern_parse_cmd(joern_parse_bat, abs_src, abs_cpg),
            desc="Joern parse",
            log_path=str(Path(abs_cpg).with_suffix(".parse.log")),
        )

    end_cpg = time.perf_counter()
    duration = end_cpg - start_cpg

    return duration / total_methods if total_methods > 0 else 0, duration


def _cpg_inputs(batch_cpg_path: str) -> list[dict]:
    abs_cpg = os.path.abspath(batch_cpg_path)
    manifest_path = _chunk_manifest_path(abs_cpg)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return manifest.get("chunks", [])
    return [{"index": 0, "cpg_path": abs_cpg}]


def run_joern_export(joern_export_bat: str, batch_cpg_path: str, base_out_dir: str, graph_types: list[str], total_methods: int):
    if not joern_export_bat:
        raise ValueError("JOERN_EXPORT_BAT is not configured.")

    cpg_inputs = _cpg_inputs(batch_cpg_path)
    per_layer_export_time = {}
    layer_total_times = {}
    logs_dir = Path(base_out_dir) / "_logs"

    print("[*] Running Joern-Export for base layers...")
    for gtype in tqdm(graph_types, desc="Exporting graph layers", unit="layer"):
        start_export = time.perf_counter()

        for cpg_info in cpg_inputs:
            chunk_idx = int(cpg_info.get("index", 0))
            cpg_path = os.path.abspath(cpg_info["cpg_path"])
            if len(cpg_inputs) == 1:
                layer_out = os.path.abspath(os.path.join(base_out_dir, gtype))
                log_name = f"export_{gtype}.log"
            else:
                layer_out = os.path.abspath(os.path.join(base_out_dir, gtype, f"chunk_{chunk_idx:04d}"))
                log_name = f"export_{gtype}_{chunk_idx:04d}.log"

            layer_out_path = Path(layer_out)
            if layer_out_path.exists():
                shutil.rmtree(layer_out_path)
            layer_out_path.parent.mkdir(parents=True, exist_ok=True)

            _run_with_seconds_progress(
                [joern_export_bat, cpg_path, f"--repr={gtype}", "--out", layer_out],
                desc=f"Export {gtype.upper()} chunk {chunk_idx + 1}/{len(cpg_inputs)}",
                log_path=str(logs_dir / log_name),
            )

        end_export = time.perf_counter()

        duration = end_export - start_export
        layer_total_times[f"raw_{gtype}_export_time"] = duration
        per_layer_export_time[gtype] = duration / total_methods if total_methods > 0 else 0

    return per_layer_export_time, layer_total_times
