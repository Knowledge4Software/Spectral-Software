import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from spectral_code.preprocessing.language_support import (
    canonical_language_from_joern,
    joern_language,
)

try:
    import psutil
except ModuleNotFoundError:
    psutil = None


def _chunk_manifest_path(cpg_path: str) -> str:
    return f"{cpg_path}.chunks.json"


def _chunk_root(cpg_path: str) -> Path:
    cpg = Path(cpg_path)
    return cpg.with_name(f"{cpg.stem}_chunks")


def _bat_safe_cmd(cmd: list[str]) -> list[str]:
    if os.name == "nt" and cmd and str(cmd[0]).lower().endswith((".bat", ".cmd")):
        return ["cmd.exe", "/d", "/c", *cmd]
    return cmd


def _resolve_joern_cmd(configured: str | None, env_name: str, windows_name: str, posix_name: str) -> str:
    def existing_command(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute() or path.parent != Path("."):
            return str(path) if path.exists() else None
        return shutil.which(value)

    joern_home = os.getenv("JOERN_HOME")
    joern_home_candidates = []
    if joern_home:
        joern_home_path = Path(joern_home)
        joern_home_candidates.extend([joern_home_path / windows_name, joern_home_path / posix_name])

    common_candidates = []
    if os.name == "nt":
        common_candidates.extend([Path("C:/joern-cli") / windows_name, Path("C:/tools/joern-cli") / windows_name])

    candidates = [
        existing_command(configured),
        existing_command(os.getenv(env_name)),
        *(str(path) for path in joern_home_candidates if path.exists()),
        shutil.which(windows_name),
        shutil.which(posix_name),
        *(str(path) for path in common_candidates if path.exists()),
        configured,
        os.getenv(env_name),
        windows_name if os.name == "nt" else posix_name,
    ]
    return next(str(candidate) for candidate in candidates if candidate)


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _tail(path: str, max_lines: int = 80) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except OSError:
        return ""


def _format_bytes(size: int) -> str:
    value = float(max(0, size))
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or suffix == "GiB":
            return f"{int(value)}B" if suffix == "B" else f"{value:.1f}{suffix}"
        value /= 1024
    return f"{value:.1f}GiB"


def _file_size(path: str | None) -> int:
    if not path:
        return 0
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _process_tree_stats(process: subprocess.Popen) -> dict[str, int | float] | None:
    if psutil is None:
        return None

    try:
        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return None

    cpu_seconds = 0.0
    memory_bytes = 0
    live_count = 0
    for proc in processes:
        try:
            times = proc.cpu_times()
            memory = proc.memory_info()
        except psutil.Error:
            continue
        cpu_seconds += float(times.user + times.system)
        memory_bytes += int(memory.rss)
        live_count += 1

    return {
        "cpu_seconds": cpu_seconds,
        "memory_bytes": memory_bytes,
        "processes": live_count,
    }


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            process.kill()
    else:
        process.kill()


def _run_with_seconds_progress(
    cmd: list[str],
    desc: str,
    log_path: str,
    output_path: str | None = None,
    timeout_seconds: int | None = None,
    inactivity_timeout_seconds: int | None = None,
    env: dict[str, str] | None = None,
):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"    log: {log_path}")

    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            _bat_safe_cmd(cmd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        last_log_size = -1
        last_log_activity = time.monotonic()
        last_stats = None
        last_stats_at = time.monotonic()
        try:
            with tqdm(desc=desc, unit="s", dynamic_ncols=True) as pbar:
                while process.poll() is None:
                    time.sleep(1)
                    pbar.update(1)
                    try:
                        log_size = Path(log_path).stat().st_size
                    except OSError:
                        log_size = last_log_size
                    if log_size != last_log_size:
                        last_log_size = log_size
                        last_log_activity = time.monotonic()
                    postfix = {
                        "log": _format_bytes(max(log_size, 0)),
                        "idle": f"{int(time.monotonic() - last_log_activity)}s",
                    }
                    output_size = _file_size(output_path)
                    if output_size:
                        postfix["out"] = _format_bytes(output_size)
                    stats = _process_tree_stats(process)
                    if stats is not None:
                        now = time.monotonic()
                        cpu_seconds = float(stats["cpu_seconds"])
                        cpu_rate = 0.0
                        if last_stats is not None:
                            elapsed = max(now - last_stats_at, 1e-9)
                            cpu_delta = max(cpu_seconds - float(last_stats["cpu_seconds"]), 0.0)
                            cpu_rate = cpu_delta / elapsed
                        last_stats = stats
                        last_stats_at = now
                        postfix["cpu"] = f"{cpu_seconds:.0f}s"
                        postfix["cpu/s"] = f"{cpu_rate:.1f}"
                        postfix["mem"] = _format_bytes(int(stats["memory_bytes"]))
                        postfix["procs"] = str(stats["processes"])
                    pbar.set_postfix(postfix, refresh=False)
                    if timeout_seconds is not None and pbar.n >= timeout_seconds:
                        _terminate_process_tree(process)
                        log_tail = _tail(log_path)
                        if log_tail:
                            print(f"\n[-] {desc} timed out after {timeout_seconds}s. Last log lines:\n{log_tail}\n")
                        raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                    if (
                        inactivity_timeout_seconds is not None
                        and time.monotonic() - last_log_activity >= inactivity_timeout_seconds
                    ):
                        _terminate_process_tree(process)
                        log_tail = _tail(log_path)
                        if log_tail:
                            print(
                                f"\n[-] {desc} had no log activity for {inactivity_timeout_seconds}s. "
                                f"Last log lines:\n{log_tail}\n"
                            )
                        raise subprocess.TimeoutExpired(cmd, inactivity_timeout_seconds)
        except KeyboardInterrupt:
            _terminate_process_tree(process)
            try:
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
            print(f"\n[!] {desc} interrupted; the Joern child process tree was stopped.")
            raise

    if process.returncode != 0:
        log_tail = _tail(log_path)
        if log_tail:
            print(f"\n[-] {desc} failed. Last log lines:\n{log_tail}\n")
        raise subprocess.CalledProcessError(process.returncode, cmd)


def _has_usable_cpg(cpg_path: str) -> bool:
    path = Path(cpg_path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _dot_count(path: str) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.dot"))


def _joern_parse_cmd(joern_parse_bat: str, src_dir: str, cpg_path: str) -> list[str]:
    source_language = canonical_language_from_joern(os.getenv("JOERN_LANGUAGE", "javasrc"))
    frontend = joern_language(source_language)
    if frontend == "javasrc" and _truthy_env("JOERN_USE_DIRECT_FRONTEND", "0"):
        javasrc2cpg = _resolve_joern_cmd(
            os.getenv("JAVASRC2CPG_BAT"),
            "JAVASRC2CPG_BAT",
            "javasrc2cpg.bat",
            "javasrc2cpg",
        )
        return [javasrc2cpg, src_dir, "--output", cpg_path]
    return [joern_parse_bat, src_dir, "--language", frontend, "--output", cpg_path]


def _joern_parse_mode() -> str:
    source_language = canonical_language_from_joern(os.getenv("JOERN_LANGUAGE", "javasrc"))
    frontend = joern_language(source_language)
    if frontend == "javasrc" and _truthy_env("JOERN_USE_DIRECT_FRONTEND", "0"):
        return "direct javasrc2cpg frontend"
    if source_language == "cpp":
        return "joern-parse wrapper (C++ via c2cpg)"
    return f"joern-parse wrapper ({frontend})"


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _parse_chunk_size(total_methods: int) -> int:
    raw = os.getenv("JOERN_PARSE_CHUNK_SIZE", "1000").strip()
    try:
        chunk_size = int(raw)
    except ValueError:
        chunk_size = 0

    if chunk_size <= 0:
        return total_methods
    return chunk_size


def _parse_min_chunk_size() -> int:
    raw = os.getenv("JOERN_PARSE_MIN_CHUNK_SIZE", "10").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _optional_positive_int_env(name: str, default: str = "") -> int | None:
    raw = os.getenv(name, default).strip()
    if not raw or raw.lower() in {"0", "none", "false", "no"}:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return value


def _source_files(src_dir: str) -> list[Path]:
    return sorted(path for path in Path(src_dir).iterdir() if path.is_file())


def _method_id_from_source_file(path: Path) -> str:
    stem = path.stem
    return stem[2:] if stem.startswith("m_") else stem


def _prepare_chunk_source(chunk_files: list[Path], chunk_src: Path) -> None:
    if chunk_src.exists():
        shutil.rmtree(chunk_src, ignore_errors=True)
    chunk_src.mkdir(parents=True, exist_ok=True)
    for src_file in tqdm(chunk_files, desc=f"Preparing {chunk_src.name}", unit="file"):
        _link_or_copy(src_file, chunk_src / src_file.name)


def _parse_chunk_with_adaptive_split(
    joern_parse_bat: str,
    chunk_files: list[Path],
    chunk_root: Path,
    logs_dir: Path,
    next_index: list[int],
    min_chunk_size: int,
    timeout_seconds: int | None,
    inactivity_timeout_seconds: int | None,
) -> tuple[list[dict], list[dict]]:
    chunk_idx = next_index[0]
    next_index[0] += 1
    chunk_src = chunk_root / f"src_{chunk_idx:04d}"
    chunk_cpg = chunk_root / f"cpg_{chunk_idx:04d}.bin"
    chunk_log = logs_dir / f"parse_{chunk_idx:04d}.log"

    _prepare_chunk_source(chunk_files, chunk_src)
    if chunk_cpg.exists():
        chunk_cpg.unlink()

    try:
        _run_with_seconds_progress(
            _joern_parse_cmd(joern_parse_bat, str(chunk_src), str(chunk_cpg)),
            desc=f"Joern parse chunk {chunk_idx + 1} ({len(chunk_files)} files)",
            log_path=str(chunk_log),
            output_path=str(chunk_cpg),
            timeout_seconds=timeout_seconds,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if _has_usable_cpg(str(chunk_cpg)):
            print(
                f"[!] Joern parse chunk {chunk_idx + 1} returned {type(exc).__name__}, "
                f"but a usable CPG was produced at {chunk_cpg}. Continuing."
            )
        elif len(chunk_files) > min_chunk_size:
            print(
                f"[!] Joern parse chunk {chunk_idx + 1} failed/timed out for {len(chunk_files):,} files; "
                "splitting it into smaller chunks."
            )
            shutil.rmtree(chunk_src, ignore_errors=True)
            midpoint = len(chunk_files) // 2
            left_chunks, left_skipped = _parse_chunk_with_adaptive_split(
                joern_parse_bat,
                chunk_files[:midpoint],
                chunk_root,
                logs_dir,
                next_index,
                min_chunk_size,
                timeout_seconds,
                inactivity_timeout_seconds,
            )
            right_chunks, right_skipped = _parse_chunk_with_adaptive_split(
                joern_parse_bat,
                chunk_files[midpoint:],
                chunk_root,
                logs_dir,
                next_index,
                min_chunk_size,
                timeout_seconds,
                inactivity_timeout_seconds,
            )
            return left_chunks + right_chunks, left_skipped + right_skipped
        else:
            print(
                f"[!] Skipping {len(chunk_files):,} files after Joern parse failed/timed out at "
                f"minimum chunk size {min_chunk_size}. See {chunk_log}."
            )
            shutil.rmtree(chunk_src, ignore_errors=True)
            return [], [{
                "index": chunk_idx,
                "method_count": len(chunk_files),
                "log_path": str(chunk_log),
                "reason": type(exc).__name__,
                "files": [path.name for path in chunk_files],
            }]

    shutil.rmtree(chunk_src, ignore_errors=True)
    return [{
        "index": chunk_idx,
        "cpg_path": str(chunk_cpg),
        "method_count": len(chunk_files),
        "log_path": str(chunk_log),
        "files": [path.name for path in chunk_files],
        "method_ids": [_method_id_from_source_file(path) for path in chunk_files],
    }], []


def run_joern_parse(joern_parse_bat: str, src_dir: str, batch_cpg_path: str, total_methods: int):
    joern_parse_bat = _resolve_joern_cmd(joern_parse_bat, "JOERN_PARSE_BAT", "joern-parse.bat", "joern-parse")
    abs_src = os.path.abspath(src_dir)
    abs_cpg = os.path.abspath(batch_cpg_path)

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

    source_files = _source_files(abs_src)
    chunk_size = _parse_chunk_size(total_methods)
    min_chunk_size = _parse_min_chunk_size()
    parse_timeout_seconds = _optional_positive_int_env("JOERN_PARSE_CHUNK_TIMEOUT_SECONDS", "300")
    parse_inactivity_timeout_seconds = _optional_positive_int_env("JOERN_PARSE_INACTIVITY_TIMEOUT_SECONDS")

    chunking_enabled = len(source_files) > chunk_size

    parse_mode = _joern_parse_mode()
    print("[*] Running Joern CPG generation...")
    print(f"[*] Joern parse mode: {parse_mode}")
    if parse_mode == "direct javasrc2cpg frontend":
        print(
            "[!] Direct javasrc2cpg mode skips Joern's default overlays; "
            "CFG/DDG/PDG exports may be empty or incomplete. Prefer the "
            "joern-parse wrapper unless you only need AST."
        )
    if chunking_enabled:
        print("[*] Chunking keeps memory bounded while preserving CFG/DDG/PDG overlays.")
    else:
        print("[*] Parsing the whole batch in one Joern run; no parse chunks will be created.")
    print(
        "[*] Joern parse chunking: "
        + (f"{chunk_size:,} files per chunk" if chunking_enabled else "disabled")
    )
    print(f"[*] Joern adaptive min chunk size: {min_chunk_size:,}")
    print(f"[*] Joern parse chunk timeout: {str(parse_timeout_seconds) + 's' if parse_timeout_seconds else 'disabled'}")
    print(
        "[*] Joern parse inactivity timeout: "
        + (str(parse_inactivity_timeout_seconds) + "s" if parse_inactivity_timeout_seconds else "disabled")
    )
    print(
        "[*] Joern progress shows elapsed time, log size, idle time, CPU/memory, "
        "process count, and CPG output size when available."
    )

    start_cpg = time.perf_counter()

    if chunking_enabled:
        chunk_root.mkdir(parents=True, exist_ok=True)
        logs_dir = chunk_root / "logs"
        chunks = []
        skipped_chunks = []
        next_index = [0]
        initial_chunks = [
            source_files[start:start + chunk_size]
            for start in range(0, len(source_files), chunk_size)
        ]

        for chunk_files in tqdm(initial_chunks, desc="Joern parse chunks", unit="chunk"):
            parsed, skipped = _parse_chunk_with_adaptive_split(
                joern_parse_bat,
                chunk_files,
                chunk_root,
                logs_dir,
                next_index,
                min_chunk_size,
                parse_timeout_seconds,
                parse_inactivity_timeout_seconds,
            )
            chunks.extend(parsed)
            skipped_chunks.extend(skipped)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunked": True,
                    "chunks": chunks,
                    "skipped_chunks": skipped_chunks,
                    "total_methods": total_methods,
                    "parsed_methods": sum(int(chunk.get("method_count", 0)) for chunk in chunks),
                    "skipped_methods": sum(int(chunk.get("method_count", 0)) for chunk in skipped_chunks),
                    "requested_chunk_size": chunk_size,
                    "adaptive_min_chunk_size": min_chunk_size,
                    "parse_timeout_seconds": parse_timeout_seconds,
                    "parse_inactivity_timeout_seconds": parse_inactivity_timeout_seconds,
                },
                f,
                indent=2,
            )
        if not chunks:
            raise RuntimeError(f"Joern parse produced no usable CPG chunks. See logs in {logs_dir}.")
    else:
        parse_log = str(Path(abs_cpg).with_suffix(".parse.log"))
        try:
            _run_with_seconds_progress(
                _joern_parse_cmd(joern_parse_bat, abs_src, abs_cpg),
                desc="Joern parse",
                log_path=parse_log,
                output_path=abs_cpg,
                timeout_seconds=parse_timeout_seconds,
                inactivity_timeout_seconds=parse_inactivity_timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if _has_usable_cpg(abs_cpg):
                print(
                    f"[!] Joern parse finished with overlay/frontend warnings, but a usable CPG was produced "
                    f"at {abs_cpg}. Continuing."
                )
            else:
                raise

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


METHOD_MAP_SCRIPT = Path(__file__).resolve().parent / "joern_scripts" / "method_file_map.sc"


def method_map_path(batch_cpg_path: str) -> str:
    return f"{os.path.abspath(batch_cpg_path)}.method_map.json"


def run_joern_method_map(batch_cpg_path: str, joern_bat: str | None = None) -> dict[str, dict[str, list[str]]]:
    """Return ``{chunk_index: {method_node_id: [source_filename, full_name]}}``.

    joern-export names every DOT after its method, not after the file the method
    came from, so a DOT alone cannot say which submission it belongs to. This
    queries the CPG itself, which is the only authoritative answer.

    The map is keyed by parse chunk because each chunk is an independent CPG
    whose node ids restart from the same base: flattening them lets a later
    chunk's ids overwrite an earlier chunk's and silently drops half the
    attributions. It is cached next to the CPG so a resumed run does not pay
    for it twice.
    """
    if not METHOD_MAP_SCRIPT.is_file():
        raise FileNotFoundError(f"Joern method-map script is missing: {METHOD_MAP_SCRIPT}")

    cache_path = Path(method_map_path(batch_cpg_path))
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("format") == "joern_method_map_v3":
            chunks = cached["chunks"]
            total = sum(len(entries) for entries in chunks.values())
            print(f"[*] Reusing cached Joern method map: {total:,} methods over {len(chunks)} chunk(s).")
            return chunks
        print("[*] Ignoring a cached method map written in an older format.")

    joern_bat = _resolve_joern_cmd(joern_bat, "JOERN_BAT", "joern.bat", "joern")
    cpg_inputs = _cpg_inputs(batch_cpg_path)
    logs_dir = Path(batch_cpg_path).parent / "_method_map_logs"
    timeout_seconds = _optional_positive_int_env("JOERN_METHOD_MAP_TIMEOUT_SECONDS")

    method_map: dict[str, list[list[str]]] = {}
    print(f"[*] Building authoritative method->file map from {len(cpg_inputs)} CPG chunk(s)...")
    for cpg_info in tqdm(cpg_inputs, desc="Joern method map", unit="chunk"):
        chunk_idx = int(cpg_info.get("index", 0))
        chunk_entries = method_map.setdefault(str(chunk_idx), [])
        cpg_path = os.path.abspath(cpg_info["cpg_path"])
        if not os.path.exists(cpg_path):
            print(f"[!] Method map: CPG chunk {chunk_idx} is missing at {cpg_path}; skipping.")
            continue
        tsv_path = Path(f"{cpg_path}.method_map.tsv")
        environment = os.environ.copy()
        environment["SPECTRAL_METHOD_MAP_CPG"] = cpg_path
        environment["SPECTRAL_METHOD_MAP_OUT"] = str(tsv_path)
        try:
            _run_with_seconds_progress(
                [joern_bat, "--script", str(METHOD_MAP_SCRIPT)],
                desc=f"Method map chunk {chunk_idx + 1}/{len(cpg_inputs)}",
                log_path=str(logs_dir / f"method_map_{chunk_idx:04d}.log"),
                output_path=str(tsv_path),
                timeout_seconds=timeout_seconds,
                env=environment,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if not tsv_path.exists():
                raise RuntimeError(
                    f"Joern method map failed for chunk {chunk_idx} ({type(exc).__name__}). "
                    f"See {logs_dir / f'method_map_{chunk_idx:04d}.log'}."
                ) from exc
            print(f"[!] Method map chunk {chunk_idx} returned non-zero but produced output; continuing.")

        with tsv_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                # Row order is joern-export's method order, which is also the
                # ``<n>-<repr>.dot`` numbering. CFG/DDG exports start at a
                # statement rather than the METHOD node, so that ordinal is the
                # only way to attribute them; it is verified against the graph
                # name before use.
                chunk_entries.append(parts)
        tsv_path.unlink(missing_ok=True)

    total = sum(len(entries) for entries in method_map.values())
    if not total:
        raise RuntimeError("Joern method map is empty; DOT files cannot be attributed to source files.")

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({"format": "joern_method_map_v3", "chunks": method_map}, f)
    print(f"[+] Method map ready: {total:,} methods over {len(method_map)} chunk(s).")
    return method_map


def run_joern_export(joern_export_bat: str, batch_cpg_path: str, base_out_dir: str, graph_types: list[str], total_methods: int):
    joern_export_bat = _resolve_joern_cmd(joern_export_bat, "JOERN_EXPORT_BAT", "joern-export.bat", "joern-export")

    cpg_inputs = _cpg_inputs(batch_cpg_path)
    per_layer_export_time = {}
    layer_total_times = {}
    logs_dir = Path(base_out_dir) / "_logs"
    export_timeout_seconds = _optional_positive_int_env("JOERN_EXPORT_CHUNK_TIMEOUT_SECONDS")
    export_inactivity_timeout_seconds = _optional_positive_int_env("JOERN_EXPORT_INACTIVITY_TIMEOUT_SECONDS")

    print("[*] Running Joern-Export for base layers...")
    print(f"[*] Joern export chunk timeout: {str(export_timeout_seconds) + 's' if export_timeout_seconds else 'disabled'}")
    print(
        "[*] Joern export inactivity timeout: "
        + (str(export_inactivity_timeout_seconds) + "s" if export_inactivity_timeout_seconds else "disabled")
    )
    tasks = [(gtype, cpg_info) for gtype in graph_types for cpg_info in cpg_inputs]
    workers = min(
        max(1, int(os.getenv("JOERN_EXPORT_WORKERS", "2"))),
        max(1, len(tasks)),
    )
    print(f"[*] Joern export workers: {workers}")

    def export_one(task: tuple[str, dict]) -> tuple[str, float]:
        gtype, cpg_info = task
        started = time.perf_counter()
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
        try:
            desc = (
                f"Export {gtype.upper()} chunk {chunk_idx + 1}/{len(cpg_inputs)}"
                if len(cpg_inputs) > 1
                else f"Export {gtype.upper()}"
            )
            _run_with_seconds_progress(
                [joern_export_bat, cpg_path, f"--repr={gtype}", "--out", layer_out],
                desc=desc,
                log_path=str(logs_dir / log_name),
                timeout_seconds=export_timeout_seconds,
                inactivity_timeout_seconds=export_inactivity_timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            produced = _dot_count(layer_out)
            if produced > 0:
                print(
                    f"[!] Export {gtype.upper()} chunk {chunk_idx + 1}/{len(cpg_inputs)} returned non-zero, "
                    f"but produced {produced:,} DOT files. Continuing with partial output."
                )
            else:
                print(
                    f"[!] Export {gtype.upper()} chunk {chunk_idx + 1}/{len(cpg_inputs)} failed and produced no DOT files. "
                    f"This layer will remain empty."
                )
                layer_out_path.mkdir(parents=True, exist_ok=True)
        return gtype, time.perf_counter() - started

    if workers == 1:
        results = map(export_one, tasks)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="joern-export")
        results = executor.map(export_one, tasks)
    try:
        for gtype, duration in tqdm(
            results,
            total=len(tasks),
            desc="Exporting graph chunks",
            unit="chunk-layer",
        ):
            layer_total_times[f"raw_{gtype}_export_time"] = (
                layer_total_times.get(f"raw_{gtype}_export_time", 0.0) + duration
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    for gtype in graph_types:
        duration = layer_total_times.get(f"raw_{gtype}_export_time", 0.0)
        per_layer_export_time[gtype] = duration / total_methods if total_methods > 0 else 0

    return per_layer_export_time, layer_total_times
