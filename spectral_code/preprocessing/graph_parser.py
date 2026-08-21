import ast as py_ast
import os
import html
import json
import networkx as nx
import re
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from tqdm import tqdm

NODE_LINE_RE = re.compile(rb'^\s*"?\d+"?\s*\[label')
METHOD_ID_RE = re.compile(rb"m_(\d+)")
DOT_ORDINAL_RE = re.compile(r"(\d+)-")
LIB2TO3_TOOL = None
LIB2TO3_UNAVAILABLE = False


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

# Joern emits METHOD nodes for constructs that are not user code: one
# ``<global>``/``<module>`` pseudo-method per file, referenced operators, and
# class initialisers. Including them would bolt a fixed synthetic subtree onto
# every graph; ``is_user_method`` below filters them by terminal method name.
NON_SOURCE_FILENAMES = frozenset({"<empty>", "<includes>", "<unknown>", "", "N/A"})

FIRST_NODE_ID_RE = re.compile(rb'^\s*"?(\d+)"?\s*\[')


CHUNK_DIR_RE = re.compile(r"chunk_(\d+)$")


def _chunk_index_for(fpath: str, layer_root: str) -> str:
    """Which parse chunk exported this DOT.

    Each parse chunk is a separate CPG whose node ids restart from the same
    base, so an id is only meaningful together with its chunk. joern-export
    writes chunked layers to ``<layer>/chunk_XXXX/``; an unchunked run has a
    single CPG, which is chunk 0.
    """
    try:
        relative = Path(fpath).resolve().relative_to(Path(layer_root).resolve())
    except ValueError:
        relative = Path(fpath)
    for part in relative.parts:
        match = CHUNK_DIR_RE.fullmatch(part)
        if match:
            return str(int(match.group(1)))
    return "0"


GRAPH_NAME_RE = re.compile(rb'^\s*digraph\s+"(.*)"\s*\{')


def _dot_header(fpath: str, max_scan: int = 32) -> tuple[str | None, list[str]]:
    """Return the DOT's graph name and its first few node ids."""
    name = None
    node_ids: list[str] = []
    try:
        with open(fpath, "rb") as f:
            for raw_line in f:
                if name is None:
                    header = GRAPH_NAME_RE.match(raw_line)
                    if header:
                        name = html.unescape(header.group(1).decode("utf-8", errors="replace"))
                        continue
                match = FIRST_NODE_ID_RE.match(raw_line)
                if match:
                    node_ids.append(match.group(1).decode("ascii", errors="ignore"))
                    if len(node_ids) >= max_scan:
                        break
    except OSError:
        return None, []
    return name, node_ids


def _lookup_method_entry(fpath: str, chunk_methods: list, by_node_id: dict) -> list | None:
    """Find the method a Joern DOT belongs to, without reading the whole file.

    An AST or PDG export starts at the METHOD node, so its id resolves directly.
    A CFG or DDG export starts at a *statement*, whose id is not in the method
    table at all - those are attributed by the ``<n>-<repr>.dot`` ordinal, which
    is joern-export's method order and therefore the order of the method table.
    The ordinal is only trusted when the DOT's graph name matches the method it
    points at, so a frontend that ever skips a method cannot shift every
    subsequent record onto the wrong source file.
    """
    name, node_ids = _dot_header(fpath)
    for node_id in node_ids:
        entry = by_node_id.get(node_id)
        if entry:
            return entry

    ordinal = _dot_ordinal(fpath)
    if ordinal is None or not 0 <= ordinal < len(chunk_methods):
        return None
    candidate = chunk_methods[ordinal]
    return candidate if _names_match(candidate[2], name) else None


def method_id_from_source_filename(filename: str) -> str | None:
    """Map a Joern ``filename`` property back to the pipeline's method id."""
    if not filename or filename in NON_SOURCE_FILENAMES:
        return None
    stem = Path(filename.replace("\\", "/")).stem
    if not stem.startswith("m_"):
        return None
    candidate = stem[2:]
    return candidate if candidate.isdigit() else None


def is_user_method(full_name: str) -> bool:
    """Return whether a Joern method row represents source-authored code.

    Python qualifies real functions as ``<module>.function``.  The previous
    substring check therefore discarded every Python function merely because
    its namespace contained ``<module>``.  Only the method's own terminal name
    (or an explicit operator namespace) is synthetic.
    """
    qualified = full_name.split(":", 1)[0].strip()
    short_name = qualified.rsplit(".", 1)[-1]
    if short_name in {"<global>", "<init>", "<clinit>", "<module>", "<fakeNew>"}:
        return False
    return not (
        qualified.startswith("<operator>")
        or ".<operator>." in qualified
        or short_name.startswith("<operator>")
    )


DUPLICATE_SUFFIX_RE = re.compile(r"<duplicate>\d+$")


def method_short_name(full_name: str) -> str:
    """``Wrapper_2.m_2:java.lang.String()`` -> ``m_2``; ``helper`` -> ``helper``."""
    return full_name.split(":", 1)[0].rsplit(".", 1)[-1].strip()


def _names_match(full_name: str, dot_name: str | None) -> bool:
    """Does a DOT's graph name denote the method the ordinal points at?

    joern-export names the graph after the method's *display* name, which is not
    always the tail of its fullName: static-analysis artefacts such as
    ``main<duplicate>0`` and qualified operator stubs differ in spelling only.
    """
    if dot_name is None:
        return True
    qualified = full_name.split(":", 1)[0].strip()
    stripped = DUPLICATE_SUFFIX_RE.sub("", qualified)
    target = DUPLICATE_SUFFIX_RE.sub("", dot_name.strip())
    if not target:
        return False
    candidates = {qualified, stripped, qualified.rsplit(".", 1)[-1], stripped.rsplit(".", 1)[-1]}
    return target in candidates or stripped.endswith(target)


def _select_record_methods(idx: str, entries: list[tuple[str, str]]) -> list[str]:
    """Choose which of a file's exported methods represent the dataset record.

    Snippet datasets store one method per record and the unpacker marks it by
    renaming it to ``m_<idx>``; everything else Joern finds in that file is a
    wrapper artefact or a nested/anonymous body, so the record is exactly the
    marked method. Whole-program records (AtCoder submissions, C files) carry no
    marker, and there every user method is part of the record.
    """
    marker = f"m_{idx}"
    primary = [path for path, full_name in entries if method_short_name(full_name) == marker]
    return primary if primary else [path for path, _ in entries]


def build_dot_index_with_method_map(joern_base_out, graph_types, method_map, method_ids=None):
    """Map every DOT to its source file using the CPG's own method table.

    Returns ``{graph_type: {method_id: [dot_path, ...]}}``.  A source file that
    defines several functions (an AtCoder submission, a C file with helpers)
    yields several DOTs, and *all* of them belong to that submission; keeping
    only one silently reduced such a program to a single arbitrary function.
    Joern node ids are unique across the whole CPG, so the per-method DOTs of one
    file merge without any id remapping, and the AST/CFG/DDG layers stay aligned.
    """
    index = {gtype: {} for gtype in graph_types}
    wanted = {str(value) for value in method_ids} if method_ids is not None else None
    stats = {gtype: {"mapped": 0, "synthetic": 0, "unattributed": 0} for gtype in graph_types}

    for gtype in graph_types:
        folder = os.path.join(joern_base_out, gtype)
        if not os.path.exists(folder):
            continue

        files = _ordered_dot_files(folder)
        by_method: dict[str, list[tuple[str, str]]] = {}
        node_index: dict[str, dict[str, list[str]]] = {
            chunk: {entry[0]: entry for entry in entries} for chunk, entries in method_map.items()
        }
        print(f"[*] {gtype.upper()}: attributing {len(files):,} DOT files via the CPG method map.")
        for fpath in tqdm(files, desc=f"Mapping {gtype.upper()}", unit="dot"):
            chunk = _chunk_index_for(fpath, folder)
            entry = _lookup_method_entry(fpath, method_map.get(chunk) or [], node_index.get(chunk) or {})
            if not entry:
                stats[gtype]["unattributed"] += 1
                continue
            filename, full_name = entry[1], entry[2]
            if not is_user_method(full_name):
                stats[gtype]["synthetic"] += 1
                continue
            idx = method_id_from_source_filename(filename)
            if idx is None or (wanted is not None and idx not in wanted):
                stats[gtype]["unattributed"] += 1
                continue
            by_method.setdefault(idx, []).append((fpath, full_name))
            stats[gtype]["mapped"] += 1

        merged = 0
        for idx, entries in by_method.items():
            selected = _select_record_methods(idx, entries)
            index[gtype][idx] = selected
            merged += len(selected) > 1

        counts = stats[gtype]
        print(
            f"    {gtype.upper()}: {counts['mapped']:,} user-method DOTs over "
            f"{len(index[gtype]):,} records ({merged:,} multi-method); skipped "
            f"{counts['synthetic']:,} synthetic and {counts['unattributed']:,} unattributed."
        )

    return index


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
        r'^\s*"?(\d+)"?\s*\[\s*label\s*=\s*<(.*?)>\s*\]',
        re.MULTILINE | re.DOTALL,
    )
    quoted_node_re = re.compile(
        r'^\s*"?(\d+)"?\s*\[\s*label\s*=\s*"((?:\\.|[^"\\])*)".*?\]',
        re.MULTILINE | re.DOTALL,
    )
    edge_re = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?', re.MULTILINE)
    
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


def parse_joern_dot_files(paths) -> nx.DiGraph:
    """Parse one DOT, or merge every DOT belonging to the same source file.

    Joern node ids are globally unique inside a CPG, so a plain union keeps the
    per-function subtrees disjoint and leaves AST/CFG/DDG node ids aligned for
    the later CPG composition and DDG projection.
    """
    if isinstance(paths, (str, os.PathLike)):
        return parse_joern_dot_fast(paths)

    path_list = [path for path in paths if path]
    if not path_list:
        return nx.DiGraph()
    if len(path_list) == 1:
        return parse_joern_dot_fast(path_list[0])

    merged = nx.DiGraph()
    errors = []
    for path in path_list:
        graph = parse_joern_dot_fast(path)
        if graph.graph.get("parse_error"):
            errors.append(f"{Path(path).name}: {graph.graph['parse_error']}")
        merged.add_nodes_from(graph.nodes(data=True))
        merged.add_edges_from(graph.edges(data=True))
    merged.graph["merged_method_dots"] = len(path_list)
    if errors:
        merged.graph["parse_error"] = "; ".join(errors)
    return merged


def _python_ast_source_path(source_dir: str | None, idx: str | int) -> Path | None:
    if not source_dir:
        return None
    path = Path(source_dir) / f"m_{idx}.py"
    return path if path.exists() else None


def _csharp_source_path(source_dir: str | None, idx: str | int) -> Path | None:
    if not source_dir:
        return None
    path = Path(source_dir) / f"m_{idx}.cs"
    return path if path.exists() else None


def _parse_csharp_source(source_path: Path):
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_c_sharp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "C# graph fallbacks require tree-sitter and tree-sitter-c-sharp"
        ) from exc
    code = source_path.read_bytes()
    root = Parser(Language(tree_sitter_c_sharp.language())).parse(code).root_node
    return code, root


def _syntax_node_text(code: bytes, node, limit: int = 120) -> str:
    return code[node.start_byte:node.end_byte].decode(
        "utf-8", errors="replace"
    ).replace("\n", " ").strip()[:limit]


def _walk_named_syntax(node):
    if node is None:
        return
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        yield current
        stack.extend(reversed(current.named_children))


def build_csharp_ast_graph(source_path: Path) -> nx.DiGraph:
    """Build a real C# syntax tree when Joern's C# AST export is incomplete.

    Tree-sitter preserves the concrete syntax hierarchy even for incomplete
    snippets, which is exactly what GPTCloneBench C# contains.  We use only
    named syntax nodes; their standard grammar names are consumed by the
    notebook's canonical type mapping.
    """
    graph = nx.DiGraph()
    try:
        code, root = _parse_csharp_source(source_path)
        counter = 0
        stack: list[tuple[object, str | None]] = [(root, None)]
        while stack:
            node, parent_id = stack.pop()
            if not node.is_named:
                continue
            node_id = str(counter)
            counter += 1
            raw_text = _syntax_node_text(code, node, 96)
            graph.add_node(node_id, type=node.type, label=f"{node.type},{raw_text[:96]}" if raw_text else node.type)
            if parent_id is not None:
                graph.add_edge(parent_id, node_id, type="child")
            # Reversed push keeps the original source order under LIFO traversal.
            for child in reversed(node.children):
                if child.is_named:
                    stack.append((child, node_id))
        graph.graph["source_path"] = str(source_path)
        graph.graph["source"] = "csharp_tree_sitter_ast_fallback"
    except Exception as exc:
        graph.graph["parse_error"] = f"C# Tree-sitter fallback failed: {type(exc).__name__}: {exc}"
        graph.graph["source_path"] = str(source_path)
    return graph


CSHARP_METHOD_NODES = {
    "method_declaration",
    "constructor_declaration",
    "destructor_declaration",
    "local_function_statement",
    "accessor_declaration",
    "operator_declaration",
    "conversion_operator_declaration",
}
CSHARP_LOOP_NODES = {
    "for_statement",
    "for_each_statement",
    "while_statement",
    "do_statement",
}
CSHARP_TERMINAL_NODES = {
    "return_statement",
    "throw_statement",
    "break_statement",
    "continue_statement",
    "goto_statement",
}


def _is_csharp_statement(node) -> bool:
    return (
        node is not None
        and node.type.endswith("_statement")
        and node.type != "global_statement"
    )


def _csharp_method_nodes(root) -> list:
    return [node for node in _walk_named_syntax(root) if node.type in CSHARP_METHOD_NODES]


def _csharp_statement_label(code: bytes, node) -> str:
    text = _syntax_node_text(code, node, 100)
    return f"{node.type},{text}" if text else node.type


def build_csharp_cfg_graph(source_path: Path) -> nx.DiGraph:
    """Build a conservative statement-level CFG when C# overlays fail.

    Joern's C# frontend can leave CFG/DDG exports empty after producing a valid
    CPG.  This fallback retains branches, loop back-edges and terminal
    statements, providing a real control-flow graph instead of an empty layer.
    """
    graph = nx.DiGraph()
    try:
        code, root = _parse_csharp_source(source_path)
    except Exception as exc:
        graph.graph["parse_error"] = f"C# CFG fallback failed: {type(exc).__name__}: {exc}"
        graph.graph["source_path"] = str(source_path)
        return graph

    counter = 0

    def add_statement(node) -> str:
        nonlocal counter
        node_id = f"s{counter}"
        counter += 1
        graph.add_node(node_id, type=node.type, label=_csharp_statement_label(code, node))
        return node_id

    def connect(predecessors: list[str], target: str, edge_type: str = "next") -> None:
        for predecessor in predecessors:
            graph.add_edge(predecessor, target, type=edge_type)

    def statement_children(container) -> list:
        if container is None:
            return []
        if _is_csharp_statement(container):
            return [container]
        return [child for child in container.named_children if _is_csharp_statement(child)]

    def build_container(container, predecessors: list[str]) -> list[str]:
        exits = predecessors
        for statement in statement_children(container):
            exits = build_statement(statement, exits)
        return exits

    def build_statement(statement, predecessors: list[str]) -> list[str]:
        current = add_statement(statement)
        connect(predecessors, current)

        if statement.type == "if_statement":
            consequence = statement.child_by_field_name("consequence")
            alternative = statement.child_by_field_name("alternative")
            true_exits = build_container(consequence, [current]) if consequence else [current]
            false_exits = build_container(alternative, [current]) if alternative else [current]
            return list(dict.fromkeys([*true_exits, *false_exits]))

        if statement.type in CSHARP_LOOP_NODES:
            body = statement.child_by_field_name("body")
            body_exits = build_container(body, [current]) if body else [current]
            connect(body_exits, current, "loop")
            return [current]

        if statement.type == "switch_statement":
            body = statement.child_by_field_name("body")
            sections = [
                node for node in _walk_named_syntax(body)
                if node.type == "switch_section"
            ] if body else []
            exits = []
            for section in sections:
                exits.extend(build_container(section, [current]))
            return exits or [current]

        if statement.type in {"try_statement", "using_statement", "lock_statement", "fixed_statement"}:
            bodies = [
                child for child in statement.named_children
                if child.type in {"block", "catch_clause", "finally_clause"}
            ]
            exits = []
            for body in bodies:
                block = body.child_by_field_name("body") or body
                exits.extend(build_container(block, [current]))
            return exits or [current]

        if statement.type in CSHARP_TERMINAL_NODES:
            return []
        return [current]

    methods = _csharp_method_nodes(root)
    if methods:
        for method_index, method in enumerate(methods):
            body = method.child_by_field_name("body")
            if body is None:
                continue
            entry = f"entry:{method_index}"
            method_name = method.child_by_field_name("name")
            name = _syntax_node_text(code, method_name) if method_name else method.type
            graph.add_node(entry, type="Entry", label=f"Entry,{name}")
            build_container(body, [entry])
    else:
        entry = "entry:0"
        graph.add_node(entry, type="Entry", label="Entry,<top-level>")
        top_level = [
            node for node in _walk_named_syntax(root)
            if _is_csharp_statement(node)
        ]
        exits = [entry]
        for statement in sorted(top_level, key=lambda node: node.start_byte):
            exits = build_statement(statement, exits)

    graph.graph["source_path"] = str(source_path)
    graph.graph["source"] = "csharp_tree_sitter_cfg_fallback"
    return graph


def _csharp_local_syntax_nodes(statement):
    """Yield syntax under one statement without entering nested statements."""
    stack = [statement]
    first = True
    while stack:
        node = stack.pop()
        if not first and _is_csharp_statement(node):
            continue
        first = False
        yield node
        stack.extend(reversed(node.named_children))


def _identifier_texts(code: bytes, node) -> set[str]:
    if node is None:
        return set()
    return {
        _syntax_node_text(code, child)
        for child in _walk_named_syntax(node)
        if child.type == "identifier" and _syntax_node_text(code, child)
    }


def _identifier_nodes(node) -> list:
    if node is None:
        return []
    return [child for child in _walk_named_syntax(node) if child.type == "identifier"]


def _csharp_defs_and_uses(code: bytes, statement) -> tuple[set[str], set[str]]:
    local_nodes = list(_csharp_local_syntax_nodes(statement))
    identifier_nodes = [node for node in local_nodes if node.type == "identifier"]
    definitions: set[str] = set()
    definition_positions: set[tuple[int, int]] = set()
    compound_assignment_uses: set[str] = set()

    for node in local_nodes:
        if node.type == "variable_declarator":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                name_node = next((child for child in node.named_children if child.type == "identifier"), None)
            definitions.update(_identifier_texts(code, name_node))
            definition_positions.update(
                (identifier.start_byte, identifier.end_byte)
                for identifier in _identifier_nodes(name_node)
            )
        elif node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            left_names = _identifier_texts(code, left)
            definitions.update(left_names)
            definition_positions.update(
                (identifier.start_byte, identifier.end_byte)
                for identifier in _identifier_nodes(left)
            )
            right = node.child_by_field_name("right")
            operator_text = code[left.end_byte:right.start_byte].decode("utf-8", errors="ignore").strip() if left and right else ""
            if operator_text and operator_text != "=":
                compound_assignment_uses.update(left_names)
        elif node.type in {"postfix_unary_expression", "prefix_unary_expression"}:
            text = _syntax_node_text(code, node)
            if "++" in text or "--" in text:
                names = _identifier_texts(code, node)
                definitions.update(names)
                definition_positions.update(
                    (identifier.start_byte, identifier.end_byte)
                    for identifier in _identifier_nodes(node)
                )
                compound_assignment_uses.update(names)

    uses = {
        _syntax_node_text(code, identifier)
        for identifier in identifier_nodes
        if (identifier.start_byte, identifier.end_byte) not in definition_positions
        and _syntax_node_text(code, identifier)
    } | compound_assignment_uses
    return definitions, uses


def build_csharp_ddg_graph(source_path: Path) -> nx.DiGraph:
    """Build a deterministic def-use graph for C# as an overlay fallback."""
    graph = nx.DiGraph()
    try:
        code, root = _parse_csharp_source(source_path)
    except Exception as exc:
        graph.graph["parse_error"] = f"C# DDG fallback failed: {type(exc).__name__}: {exc}"
        graph.graph["source_path"] = str(source_path)
        return graph

    methods = _csharp_method_nodes(root) or [root]
    statement_index = 0
    for method_index, method in enumerate(methods):
        last_definition: dict[str, str] = {}
        external_nodes: dict[str, str] = {}
        entry = f"entry:{method_index}"
        method_name_node = method.child_by_field_name("name") if method is not root else None
        method_name = _syntax_node_text(code, method_name_node) if method_name_node else "<top-level>"
        graph.add_node(entry, type="Entry", label=f"Entry,{method_name}")

        parameters = method.child_by_field_name("parameters") if method is not root else None
        for parameter in _walk_named_syntax(parameters) if parameters else []:
            if parameter.type != "parameter":
                continue
            identifiers = [
                child for child in parameter.named_children if child.type == "identifier"
            ]
            if identifiers:
                last_definition[_syntax_node_text(code, identifiers[-1])] = entry

        body = method.child_by_field_name("body") if method is not root else root
        # Error recovery on malformed snippets (for example a spaced ``= >``
        # lambda arrow) can expose a synthetic method node without a body.
        # CFG already skips these nodes; DDG must do the same.
        if body is None:
            continue
        statements = sorted(
            (node for node in _walk_named_syntax(body) if _is_csharp_statement(node)),
            key=lambda node: (node.start_byte, node.end_byte),
        )
        for statement in statements:
            node_id = f"s{statement_index}"
            statement_index += 1
            graph.add_node(node_id, type=statement.type, label=_csharp_statement_label(code, statement))
            definitions, uses = _csharp_defs_and_uses(code, statement)
            for name in sorted(uses):
                if name not in last_definition:
                    external_id = external_nodes.setdefault(name, f"input:{method_index}:{name}")
                    graph.add_node(external_id, type="External", label=f"External,{name}")
                    source_id = external_id
                else:
                    source_id = last_definition[name]
                graph.add_edge(source_id, node_id, type=f"use:{name}")
            for name in sorted(definitions):
                last_definition[name] = node_id

    graph.graph["source_path"] = str(source_path)
    graph.graph["source"] = "csharp_tree_sitter_ddg_fallback"
    return graph


SPACED_PYTHON_OPERATOR_REPAIRS = (
    (re.compile(r"\*\s+\*\s+="), "**="),
    (re.compile(r"/\s+/\s+="), "//="),
    (re.compile(r"<<\s+="), "<<="),
    (re.compile(r">>\s+="), ">>="),
    (re.compile(r"(?<![<>=!])<\s+=(?!=)"), "<="),
    (re.compile(r"(?<![<>=!])>\s+=(?!=)"), ">="),
    (re.compile(r"(?<![<>=!])!\s+=(?!=)"), "!="),
    (re.compile(r"(?<![<>=!])=\s+=(?!=)"), "=="),
    (re.compile(r"(?<!/)/\s+/(?![=/])"), "//"),
    (re.compile(r"(?<!\*)\*\s+\*(?![=])"), "**"),
    (re.compile(r"(?<!<)<\s+<(?![=])"), "<<"),
    (re.compile(r"(?<!>)>\s+>(?![=])"), ">>"),
    (re.compile(r"-\s+>"), "->"),
    (re.compile(r":\s+="), ":="),
    (re.compile(r"\+\s+="), "+="),
    (re.compile(r"-\s+="), "-="),
    (re.compile(r"\*\s+="), "*="),
    (re.compile(r"/\s+="), "/="),
    (re.compile(r"%\s+="), "%="),
    (re.compile(r"&\s+="), "&="),
    (re.compile(r"\|\s+="), "|="),
    (re.compile(r"\^\s+="), "^="),
)


def _repair_spaced_python_operators(code: str) -> str:
    repaired = code
    for pattern, replacement in SPACED_PYTHON_OPERATOR_REPAIRS:
        repaired = pattern.sub(replacement, repaired)
    return repaired


def _parse_python_candidate(code: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return py_ast.parse(code)


def _get_lib2to3_tool():
    global LIB2TO3_TOOL, LIB2TO3_UNAVAILABLE
    if LIB2TO3_UNAVAILABLE:
        return None
    if LIB2TO3_TOOL is not None:
        return LIB2TO3_TOOL
    try:
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package

        LIB2TO3_TOOL = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
        return LIB2TO3_TOOL
    except Exception:
        LIB2TO3_UNAVAILABLE = True
        return None


def _parse_python_source(code: str):
    candidates = [code]
    repaired_code = _repair_spaced_python_operators(code)
    if repaired_code != code:
        candidates.append(repaired_code)

    for candidate in candidates:
        try:
            return _parse_python_candidate(candidate)
        except SyntaxError:
            pass

    tool = _get_lib2to3_tool()
    if tool is not None:
        for candidate in candidates:
            try:
                translated = str(tool.refactor_string(candidate, name="<semantic-python-snippet>"))
                return _parse_python_candidate(translated)
            except Exception:
                pass
    return None


def build_python_ast_graph(source_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    code = source_path.read_text(encoding="utf-8", errors="replace")
    tree = _parse_python_source(code)
    if tree is None:
        graph.graph["parse_error"] = "Python AST fallback failed: source is not parseable as Python 3 or lib2to3-translated Python 2"
        graph.graph["source_path"] = str(source_path)
        return graph

    counter = 0

    def add_node(node, parent_id: str | None = None, edge_label: str | None = None) -> str:
        nonlocal counter
        node_id = str(counter)
        counter += 1
        node_type = type(node).__name__
        label = node_type
        if isinstance(node, (py_ast.FunctionDef, py_ast.AsyncFunctionDef, py_ast.ClassDef)):
            label = f"{node_type},{node.name}"
        elif isinstance(node, py_ast.Name):
            label = f"{node_type},{node.id}"
        elif isinstance(node, py_ast.arg):
            label = f"{node_type},{node.arg}"
        elif isinstance(node, py_ast.Constant):
            label = f"{node_type},{repr(node.value)[:80]}"

        graph.add_node(node_id, type=node_type, label=label)
        if parent_id is not None:
            graph.add_edge(parent_id, node_id, type=edge_label or "child")

        for field_name, value in py_ast.iter_fields(node):
            if isinstance(value, py_ast.AST):
                add_node(value, node_id, field_name)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, py_ast.AST):
                        add_node(item, node_id, field_name)
        return node_id

    add_node(tree)
    graph.graph["source_path"] = str(source_path)
    graph.graph["source"] = "python_ast_fallback"
    return graph


def _python_source_graph(source_path: Path):
    code = source_path.read_text(encoding="utf-8", errors="replace")
    tree = _parse_python_source(code)
    return code, tree


def _python_statement_label(stmt) -> str:
    label = type(stmt).__name__
    line = getattr(stmt, "lineno", None)
    if isinstance(stmt, (py_ast.FunctionDef, py_ast.AsyncFunctionDef, py_ast.ClassDef)):
        label = f"{label},{stmt.name}"
    elif isinstance(stmt, py_ast.Assign):
        targets = [getattr(target, "id", type(target).__name__) for target in stmt.targets]
        label = f"{label},{'='.join(targets[:3])}"
    elif isinstance(stmt, py_ast.Name):
        label = f"{label},{stmt.id}"
    return f"{label}@{line}" if line else label


def _statement_children(stmt) -> list:
    children = []
    for field_name in ("body", "orelse", "finalbody"):
        value = getattr(stmt, field_name, None)
        if isinstance(value, list):
            children.extend(item for item in value if isinstance(item, py_ast.stmt))
    handlers = getattr(stmt, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            children.extend(item for item in getattr(handler, "body", []) if isinstance(item, py_ast.stmt))
    return children


def _ordered_statements(tree) -> list:
    statements = []

    def visit_block(block):
        for stmt in block:
            if not isinstance(stmt, py_ast.stmt):
                continue
            statements.append(stmt)
            visit_block(_statement_children(stmt))

    visit_block(getattr(tree, "body", []))
    return statements


def build_python_cfg_graph(source_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    _, tree = _python_source_graph(source_path)
    if tree is None:
        graph.graph["parse_error"] = "Python CFG fallback failed: source is not parseable"
        graph.graph["source_path"] = str(source_path)
        return graph

    counter = 0

    def add_stmt(stmt) -> str:
        nonlocal counter
        node_id = str(counter)
        counter += 1
        graph.add_node(node_id, type=type(stmt).__name__, label=_python_statement_label(stmt))
        return node_id

    def build_block(block, previous: list[str] | None = None) -> list[str]:
        exits = previous or []
        for stmt in block:
            if not isinstance(stmt, py_ast.stmt):
                continue
            current = add_stmt(stmt)
            for prev in exits:
                graph.add_edge(prev, current, type="next")

            if isinstance(stmt, (py_ast.If, py_ast.IfExp)):
                body_exits = build_block(getattr(stmt, "body", []), [current])
                else_block = getattr(stmt, "orelse", [])
                else_exits = build_block(else_block, [current]) if else_block else [current]
                exits = body_exits + else_exits
            elif isinstance(stmt, (py_ast.FunctionDef, py_ast.AsyncFunctionDef, py_ast.ClassDef)):
                body_exits = build_block(getattr(stmt, "body", []), [current])
                exits = body_exits or [current]
            elif isinstance(stmt, (py_ast.For, py_ast.AsyncFor, py_ast.While)):
                body_exits = build_block(getattr(stmt, "body", []), [current])
                for exit_id in body_exits:
                    graph.add_edge(exit_id, current, type="loop")
                else_exits = build_block(getattr(stmt, "orelse", []), [current]) if getattr(stmt, "orelse", []) else []
                exits = [current] + else_exits
            elif isinstance(stmt, py_ast.Try):
                exits = []
                exits.extend(build_block(getattr(stmt, "body", []), [current]))
                for handler in getattr(stmt, "handlers", []):
                    exits.extend(build_block(getattr(handler, "body", []), [current]))
                exits.extend(build_block(getattr(stmt, "orelse", []), exits or [current]))
                exits.extend(build_block(getattr(stmt, "finalbody", []), exits or [current]))
                if not exits:
                    exits = [current]
            elif isinstance(stmt, (py_ast.Return, py_ast.Raise, py_ast.Break, py_ast.Continue)):
                exits = []
            else:
                exits = [current]
        return exits

    build_block(getattr(tree, "body", []), [])
    graph.graph["source_path"] = str(source_path)
    graph.graph["source"] = "python_cfg_fallback"
    return graph


def _names_in_context(node, context_type) -> set[str]:
    names = set()
    for child in py_ast.walk(node):
        if isinstance(child, py_ast.Name) and isinstance(child.ctx, context_type):
            names.add(child.id)
        elif isinstance(child, py_ast.arg) and context_type is py_ast.Store:
            names.add(child.arg)
    return names


def build_python_ddg_graph(source_path: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    _, tree = _python_source_graph(source_path)
    if tree is None:
        graph.graph["parse_error"] = "Python DDG fallback failed: source is not parseable"
        graph.graph["source_path"] = str(source_path)
        return graph

    last_def: dict[str, str] = {}
    external_nodes: dict[str, str] = {}
    statements = _ordered_statements(tree)

    def external_node(name: str) -> str:
        if name not in external_nodes:
            node_id = f"input:{name}"
            external_nodes[name] = node_id
            graph.add_node(node_id, type="External", label=f"External,{name}")
        return external_nodes[name]

    for index, stmt in enumerate(statements):
        node_id = f"s{index}"
        graph.add_node(node_id, type=type(stmt).__name__, label=_python_statement_label(stmt))
        uses = _names_in_context(stmt, py_ast.Load)
        defs = _names_in_context(stmt, py_ast.Store)
        if isinstance(stmt, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
            defs.add(stmt.name)
            for arg in getattr(stmt.args, "args", []):
                last_def[arg.arg] = node_id

        for name in sorted(uses):
            graph.add_edge(last_def.get(name, external_node(name)), node_id, type=f"use:{name}")
        for name in sorted(defs):
            last_def[name] = node_id

    graph.graph["source_path"] = str(source_path)
    graph.graph["source"] = "python_ddg_fallback"
    return graph


def _python_fallback_enabled(gtype: str, source_dir: str | None) -> bool:
    language = os.getenv("JOERN_LANGUAGE", "").strip().lower()
    return gtype in {"ast", "cfg", "ddg"} and language == "pythonsrc" and bool(source_dir)


def _fallback_python_graph(gtype: str, idx: str | int, source_dir: str | None) -> nx.DiGraph | None:
    source_path = _python_ast_source_path(source_dir, idx)
    if source_path is None:
        return None
    if gtype == "ast":
        return build_python_ast_graph(source_path)
    if gtype == "cfg":
        return build_python_cfg_graph(source_path)
    if gtype == "ddg":
        return build_python_ddg_graph(source_path)
    return None


def _csharp_fallback_enabled(gtype: str, source_dir: str | None) -> bool:
    language = os.getenv("JOERN_LANGUAGE", "").strip().lower()
    return gtype in {"ast", "cfg", "ddg"} and language in {"csharpsrc", "csharp", "c#", "cs"} and bool(source_dir)


def _fallback_csharp_graph(gtype: str, idx: str | int, source_dir: str | None) -> nx.DiGraph | None:
    source_path = _csharp_source_path(source_dir, idx)
    if source_path is None:
        return None
    try:
        if gtype == "ast":
            return build_csharp_ast_graph(source_path)
        if gtype == "cfg":
            return build_csharp_cfg_graph(source_path)
        if gtype == "ddg":
            return build_csharp_ddg_graph(source_path)
    except Exception as exc:
        # One malformed source must become a reported missing layer, not abort
        # the entire concurrent dataset conversion.
        graph = nx.DiGraph()
        graph.graph["parse_error"] = (
            f"C# {gtype.upper()} fallback failed: {type(exc).__name__}: {exc}"
        )
        graph.graph["source_path"] = str(source_path)
        return graph
    return None


def _should_replace_with_csharp_fallback(gtype: str, graph: nx.DiGraph, idx: str | int, source_dir: str | None) -> bool:
    if not _csharp_fallback_enabled(gtype, source_dir):
        return False
    mode = os.getenv(
        "CSHARP_GRAPH_FALLBACK_MODE",
        os.getenv("CSHARP_AST_FALLBACK_MODE", "prefer"),
    ).strip().lower()
    return mode in {"prefer", "always", "source"} or graph.number_of_nodes() == 0


def _source_line_count(source_dir: str | None, idx: str | int) -> int:
    source_path = _python_ast_source_path(source_dir, idx)
    if source_path is None:
        return 0
    return sum(1 for line in source_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _should_replace_with_python_fallback(gtype: str, graph: nx.DiGraph, idx: str | int, source_dir: str | None) -> bool:
    if not _python_fallback_enabled(gtype, source_dir):
        return False
    mode = os.getenv("PYTHON_GRAPH_FALLBACK_MODE", "prefer").strip().lower()
    if mode in {"prefer", "always", "source"}:
        return True
    if gtype == "ast" and graph.number_of_nodes() == 0:
        return True
    if gtype not in {"cfg", "ddg"}:
        return False
    min_lines = _env_int("PYTHON_GRAPH_FALLBACK_MIN_SOURCE_LINES", 8)
    tiny_nodes = _env_int("PYTHON_GRAPH_FALLBACK_TINY_NODES", 8)
    line_count = _source_line_count(source_dir, idx)
    line_based_tiny_nodes = max(tiny_nodes, int(min(line_count, 40) * 0.75))
    return line_count >= min_lines and graph.number_of_nodes() <= line_based_tiny_nodes


def process_single_method(args):
    if len(args) == 6:
        idx, paths, output_dir, per_method_cpg_time, per_layer_export_time, source_dir = args
    else:
        idx, paths, output_dir, per_method_cpg_time, per_layer_export_time = args
        source_dir = None
    results = {}
    skipped_layers = []
    fallback_layers = []
    
    for gtype, fpath in paths.items():
        # A method-mapped entry is the list of every DOT exported for that
        # source file; a legacy entry is a single path.
        existing = (
            [path for path in fpath if path and os.path.exists(path)]
            if isinstance(fpath, (list, tuple))
            else fpath if fpath and os.path.exists(fpath) else None
        )
        if existing:
            graph = parse_joern_dot_files(existing)
            fallback_graph = None
            if _should_replace_with_python_fallback(gtype, graph, idx, source_dir):
                fallback_graph = _fallback_python_graph(gtype, idx, source_dir)
            elif _should_replace_with_csharp_fallback(gtype, graph, idx, source_dir):
                fallback_graph = _fallback_csharp_graph(gtype, idx, source_dir)
            if fallback_graph is not None and fallback_graph.number_of_nodes() > 0:
                graph = fallback_graph
                fallback_layers.append(
                    {
                        "idx": int(idx),
                        "graph_type": gtype,
                        "source": graph.graph.get("source"),
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                    }
                )
            
            if graph.number_of_nodes() == 0:
                skipped_layers.append(
                    {
                        "idx": int(idx),
                        "graph_type": gtype,
                        "dot_path": ", ".join(map(str, existing)) if isinstance(existing, list) else str(existing),
                        "reason": graph.graph.get("parse_error", "empty graph after parsing"),
                    }
                )
                results[gtype] = {"nodes": 0, "edges": 0, "graph_data": None}
            else:
                feature = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "graph_data": nx.node_link_data(graph)
                }
                if graph.graph.get("source"):
                    feature["source"] = graph.graph["source"]
                results[gtype] = feature
        else:
            if _python_fallback_enabled(gtype, source_dir):
                graph = _fallback_python_graph(gtype, idx, source_dir)
            elif _csharp_fallback_enabled(gtype, source_dir):
                graph = _fallback_csharp_graph(gtype, idx, source_dir)
            else:
                graph = None
            if graph is not None and graph.number_of_nodes() > 0:
                fallback_layers.append(
                    {
                        "idx": int(idx),
                        "graph_type": gtype,
                        "source": graph.graph.get("source"),
                        "nodes": graph.number_of_nodes(),
                        "edges": graph.number_of_edges(),
                    }
                )
                results[gtype] = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "graph_data": nx.node_link_data(graph),
                    "source": graph.graph.get("source"),
                }
            else:
                reason = "missing Joern DOT file"
                if graph is not None and graph.graph.get("parse_error"):
                    reason = f"{reason}; {graph.graph['parse_error']}"
                skipped_layers.append(
                    {
                        "idx": int(idx),
                        "graph_type": gtype,
                        "dot_path": str(fpath) if fpath else None,
                        "reason": reason,
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
    return {
        "idx": int(idx),
        "skipped_layers": skipped_layers,
        "fallback_layers": fallback_layers,
    }
