import os
import json
import networkx as nx
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tqdm import tqdm

NODE_LINE_RE = re.compile(rb'^\s*"\d+"\s*\[label')
METHOD_ID_RE = re.compile(rb"m_(\d+)")
DOT_ORDINAL_RE = re.compile(r"(\d+)-")


def _ordered_dot_files(folder: str) -> list[str]:
    def _sort_key(path: Path):
        parts = tuple(part.lower() for part in path.relative_to(folder).parts[:-1])
        match = DOT_ORDINAL_RE.match(path.name)
        if match:
            return (*parts, 0, int(match.group(1)), path.name.lower())
        return (*parts, 1, path.name.lower())

    return [str(path) for path in sorted(Path(folder).rglob("*.dot"), key=_sort_key)]


def _dot_ordinal(path: str | Path) -> int | None:
    match = DOT_ORDINAL_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def _load_chunk_manifest(path: str | Path | None) -> dict | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("chunked"):
        return None
    return manifest


def _method_ids_from_chunk(chunk: dict) -> list[str]:
    if chunk.get("method_ids"):
        return [str(method_id) for method_id in chunk["method_ids"]]
    ids = []
    for file_name in chunk.get("files", []):
        stem = Path(file_name).stem
        ids.append(stem[2:] if stem.startswith("m_") else stem)
    return ids


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _dot_index_workers() -> int:
    default = min(32, max(4, (os.cpu_count() or 4) * 2))
    return _env_int("DOT_INDEX_WORKERS", default)


def threaded_bounded_map(func, items, max_workers: int, max_pending: int | None = None):
    """
    Yield thread-pool results as each task completes with bounded in-flight work.

    This avoids both Windows multiprocessing hangs and ordered-map progress stalls
    on very large Joern DOT exports.
    """
    max_pending = max_pending or max_workers * 4
    iterator = iter(items)
    pending = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while len(pending) < max_pending:
            try:
                pending.add(executor.submit(func, next(iterator)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    pending.add(executor.submit(func, next(iterator)))
                except StopIteration:
                    pass


def _quick_analyze_dot(fpath):
    """
    Streaming metadata scan for Joern DOT files.

    The old implementation read every DOT file fully inside a multiprocessing
    worker. On Windows, a few huge/malformed AST files can leave the pool
    waiting at 99.99%. This bounded scan only reads up to DOT_INDEX_MAX_BYTES
    per file and is safe to run in a thread pool.
    """
    max_bytes = _env_int("DOT_INDEX_MAX_BYTES", 32 * 1024 * 1024)
    try:
        found_idx = None
        node_count = 0
        bytes_seen = 0
        with open(fpath, "rb") as f:
            for raw_line in f:
                bytes_seen += len(raw_line)
                if found_idx is None:
                    match = METHOD_ID_RE.search(raw_line)
                    if match:
                        found_idx = match.group(1).decode("ascii", errors="ignore")
                if NODE_LINE_RE.match(raw_line):
                    node_count += 1
                if bytes_seen >= max_bytes:
                    break

        if found_idx is None:
            return None

        # Return tuple for build_dot_index: (id, path, score). Score is node
        # count; if multiple DOTs match the same ID, choose the richer method.
        return (found_idx, fpath, node_count)
    except Exception:
        return None

def build_dot_index(joern_base_out, graph_types, method_ids=None, chunk_manifest_path=None):
    """
    Scans the exported DOT files in parallel and builds a mapping.
    """
    index = {g: {} for g in graph_types}
    chunk_manifest = _load_chunk_manifest(chunk_manifest_path)
    
    for gtype in graph_types:
        folder = os.path.join(joern_base_out, gtype)
        if not os.path.exists(folder):
            continue
        
        files = _ordered_dot_files(folder)
        mapped_ids = set()
        used_paths = set()
        mapped_paths = set()

        workers = _dot_index_workers()
        print(f"[*] {gtype.upper()}: indexing {len(files):,} DOT files with {workers} IO workers.")
        for res in tqdm(
            threaded_bounded_map(_quick_analyze_dot, files, max_workers=workers),
            total=len(files),
            desc=f"Indexing {gtype.upper()}",
            unit="dot",
        ):
            if not res:
                continue
            idx, fpath, score = res
            mapped_ids.add(idx)
            mapped_paths.add(fpath)
            if idx not in index[gtype] or score > index[gtype][idx][1]:
                index[gtype][idx] = (fpath, score)
                used_paths.add(fpath)

        if chunk_manifest:
            fallback_count = 0
            mismatch_count = 0
            for chunk in chunk_manifest.get("chunks", []):
                chunk_idx = int(chunk.get("index", 0))
                expected_ids = _method_ids_from_chunk(chunk)
                if not expected_ids:
                    continue

                chunk_folder = Path(folder) / f"chunk_{chunk_idx:04d}"
                if not chunk_folder.exists():
                    continue

                chunk_files = _ordered_dot_files(str(chunk_folder))
                for fpath in chunk_files:
                    if fpath in mapped_paths:
                        continue
                    ordinal = _dot_ordinal(fpath)
                    if ordinal is None or ordinal >= len(expected_ids):
                        mismatch_count += 1
                        continue
                    idx = expected_ids[ordinal]
                    if idx in index[gtype]:
                        continue
                    index[gtype][idx] = (fpath, -1)
                    fallback_count += 1

            if fallback_count:
                print(
                    f"[*] {gtype.upper()}: chunk-aware positional DOT mapping fallback matched "
                    f"{fallback_count:,} methods."
                )
            if mismatch_count:
                print(
                    f"[!] {gtype.upper()}: skipped {mismatch_count:,} DOT files whose ordinal did not "
                    "fit the recorded Joern parse chunk."
                )
        elif method_ids:
            missing_ids = [str(mid) for mid in method_ids if str(mid) not in mapped_ids]
            if missing_ids and files:
                # Some frontends (notably C#/Python/C) export per-method DOTs as
                # ordinal filenames like `0-ddg.dot` without the original `m_<id>`
                # marker in either the filename or graph title. In that case we
                # fall back to the unpack order, which matches the generated source
                # file order that Joern processed.
                missing_paths = [path for path in files if path not in mapped_paths]
                fallback_count = min(len(missing_ids), len(missing_paths))
                if fallback_count:
                    print(
                        f"[*] {gtype.upper()}: positional DOT mapping fallback matched "
                        f"{fallback_count:,} methods."
                    )
                for idx, fpath in zip(missing_ids[:fallback_count], missing_paths[:fallback_count]):
                    index[gtype][idx] = (fpath, -1)
                
    return {g: {idx: val[0] for idx, val in items.items()} for g, items in index.items()}

def parse_joern_dot_fast(fpath):
    """
    Optimized DOT parser for Joern's specific format.
    Handles various label formats (with/without parentheses, with/without <BR/>).
    """
    G = nx.DiGraph()
    # Joern DOT labels appear in both HTML-like and quoted forms depending on
    # version/repr. Keep the parser permissive; missing nodes here silently
    # turns a valid layer into a tiny/empty graph.
    html_node_re = re.compile(
        r'^\s*"(\d+)"\s*\[\s*label\s*=\s*<(.*?)>\s*\]',
        re.MULTILINE | re.DOTALL,
    )
    quoted_node_re = re.compile(
        r'^\s*"(\d+)"\s*\[\s*label\s*=\s*"((?:\\.|[^"\\])*)".*?\]',
        re.MULTILINE | re.DOTALL,
    )
    edge_re = re.compile(r'^\s*"(\d+)"\s*->\s*"(\d+)"', re.MULTILINE)
    
    try:
        max_bytes = _env_int("DOT_PARSE_MAX_BYTES", 64 * 1024 * 1024)
        size = Path(fpath).stat().st_size
        if size > max_bytes:
            graph = nx.DiGraph()
            graph.graph["parse_error"] = (
                f"DOT file too large for safe parsing: {size:,} bytes "
                f"(limit {max_bytes:,}; set DOT_PARSE_MAX_BYTES to override)"
            )
            graph.graph["source_path"] = str(fpath)
            return graph

        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract nodes
        node_matches = [(nid, raw_label) for nid, raw_label in html_node_re.findall(content)]
        node_matches.extend(
            (nid, bytes(raw_label, "utf-8").decode("unicode_escape", errors="replace"))
            for nid, raw_label in quoted_node_re.findall(content)
            if nid not in {existing_nid for existing_nid, _ in node_matches}
        )
        for nid, raw_label in node_matches:
            # Clean up the label. Joern labels often look like:
            # (TYPE,VALUE) or TYPE, VALUE<BR/>CODE
            label_text = raw_label.strip()
            if label_text.startswith('(') and label_text.endswith(')'):
                label_text = label_text[1:-1]
            
            # Split by first comma to get type
            parts = label_text.split(',', 1)
            ntype = parts[0].strip()
            nlabel = parts[1].strip() if len(parts) > 1 else ntype
            
            # Remove <BR/> and everything after it for the display label
            nlabel = nlabel.split('<BR/>')[0].strip()
            
            G.add_node(nid, type=ntype, label=nlabel)
            
        # Extract edges
        for src, dst in edge_re.findall(content):
            G.add_edge(src, dst)
            
        return G
    except Exception as exc:
        graph = nx.DiGraph()
        graph.graph["parse_error"] = f"{type(exc).__name__}: {exc}"
        graph.graph["source_path"] = str(fpath)
        return graph

def process_single_method(args):
    idx, paths, output_dir, per_method_cpg_time, per_layer_export_time = args
    results = {}
    skipped_layers = []
    
    for gtype, fpath in paths.items():
        if fpath and os.path.exists(fpath):
            # Use the fast parser instead of networkx default
            graph = parse_joern_dot_fast(fpath)
            
            if graph.number_of_nodes() == 0:
                skipped_layers.append(
                    {
                        "idx": int(idx),
                        "graph_type": gtype,
                        "dot_path": str(fpath),
                        "reason": graph.graph.get("parse_error", "empty graph after parsing"),
                    }
                )
                results[gtype] = {"nodes": 0, "edges": 0, "graph_data": None}
            else:
                results[gtype] = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "graph_data": nx.node_link_data(graph)
                }
        else:
            skipped_layers.append(
                {
                    "idx": int(idx),
                    "graph_type": gtype,
                    "dot_path": str(fpath) if fpath else None,
                    "reason": "missing Joern DOT file",
                }
            )
            results[gtype] = {"nodes": 0, "edges": 0, "graph_data": None}
            
        results[gtype]["metrics"] = {
            "cpg_time": per_method_cpg_time,
            "export_time": per_layer_export_time.get(gtype, 0),
            "total_time": per_method_cpg_time + per_layer_export_time.get(gtype, 0)
        }
        results[gtype]["eigenvalues"] = None

    out_file = os.path.join(output_dir, f"{idx}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"idx": int(idx), "features": results}, f)
    return {"idx": int(idx), "skipped_layers": skipped_layers}
