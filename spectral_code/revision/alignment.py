"""Recover fallback DDG->AST identity only after exact syntax-tree validation.

Fallback parsers number AST nodes and statement DDG nodes independently. Their
numeric/string IDs must never be joined heuristically. This local preparation
step reproduces the original AST traversal, checks exported node types AND
edges, and writes an explicit mapping sidecar for portable Kaggle consumers.
External dependency nodes without an AST counterpart remain unmapped.
"""
from __future__ import annotations

import argparse
import ast as py_ast
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def fallback_mapping(source: str, language: str, ast: dict, ddg: dict) -> tuple[list, str]:
    types, edges, objects = [], [], {}
    nodes = []
    if language == "python" and ast.get("node_types", [None])[0] == "Module":
        try:
            # Same Python-2/operator repair path as the existing exporter,
            # used only locally. The generated sidecar needs no parser on Kaggle.
            from spectral_code.preprocessing.graph_parser import _parse_python_source, _ordered_statements
            tree = _parse_python_source(source)
        except ImportError:
            tree = py_ast.parse(source)
        if tree is None:
            return [], "source_parse_failed"

        def visit(node, parent=None):
            i = len(types)
            objects[id(node)] = i
            types.append(type(node).__name__)
            if parent is not None:
                edges.append((parent, i))
            for _, value in py_ast.iter_fields(node):
                for child in value if isinstance(value, list) else [value]:
                    if isinstance(child, py_ast.AST):
                        visit(child, i)
        visit(tree)
        statements = _ordered_statements(tree)
        mapping = {f"s{i}": objects[id(node)] for i, node in enumerate(statements)}
    elif language == "csharp" and ast.get("node_types", [None])[0] == "compilation_unit":
        from tree_sitter import Language, Parser
        import tree_sitter_c_sharp
        from spectral_code.preprocessing.graph_parser import CSHARP_METHOD_NODES
        tree = Parser(Language(tree_sitter_c_sharp.language())).parse(source.encode("utf-8"))
        root = tree.root_node
        stack = [(root, None)]
        while stack:
            node, parent = stack.pop()
            i = len(types)
            objects[node.id] = i
            types.append(node.type); nodes.append(node)
            if parent is not None:
                edges.append((parent, i))
            stack.extend((child, i) for child in reversed(node.named_children))
        methods = [node for node in nodes if node.type in CSHARP_METHOD_NODES] or [root]
        mapping = {}
        statement_index = 0
        for method_index, method in enumerate(methods):
            mapping[f"entry:{method_index}"] = objects[method.id]
            body = method.child_by_field_name("body") if method is not root else root
            if body is None:
                continue
            local, stack = [], [body]
            while stack:
                node = stack.pop()
                if node.type.endswith("_statement") and node.type != "global_statement":
                    local.append(node)
                stack.extend(reversed(node.named_children))
            for node in sorted(local, key=lambda n: (n.start_byte, n.end_byte)):
                mapping[f"s{statement_index}"] = objects[node.id]
                statement_index += 1
    else:
        return [], "not_fallback_ast"
    expected_edges = set(zip(ast.get("row", []), ast.get("col", [])))
    if types != ast.get("node_types") or set(edges) != expected_edges:
        return [], "ast_reproduction_mismatch"
    ast_ids = ast.get("node_ids", [])
    result = []
    for i, node_id in enumerate(ddg.get("node_ids", [])):
        position = mapping.get(str(node_id))
        # Confirm statement type too; a stale DDG from another source is rejected.
        if position is not None and i < len(ddg.get("node_types", [])):
            ddg_type = ddg["node_types"][i]
            if not str(node_id).startswith("entry:") and ddg_type != types[position]:
                return [], "ddg_statement_type_mismatch"
        result.append(ast_ids[position] if position is not None else None)
    return result, "validated_ast_and_ddg_statement_types"


def _retained(ast, ddg, mapping, max_nodes):
    valid_ast = set(str(x) for x in ast.get("node_ids", [])[:max_nodes])
    count = 0
    for a, b in zip(ddg.get("row", []), ddg.get("col", [])):
        a, b = int(a), int(b)
        if 0 <= a < len(mapping) and 0 <= b < len(mapping):
            count += str(mapping[a]) in valid_ast and str(mapping[b]) in valid_ast
    return count


def build_alignment(root: Path, max_nodes: int = 256) -> dict:
    codes = {}
    with gzip.open(root / "codes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            code = json.loads(line)
            if code["language"] in {"python", "csharp"}:
                codes[str(code["code_id"])] = code
    output = root / "ddg_ast_alignment.jsonl.gz"
    stats = defaultdict(Counter)
    with gzip.open(root / "graph_spectra.jsonl.gz", "rt", encoding="utf-8") as stream, gzip.open(output, "wt", encoding="utf-8", compresslevel=1) as dest:
        for line in stream:
            record = json.loads(line)
            code = codes.get(str(record["code_id"]))
            if code is None:
                continue
            ast = record["graphs"].get("ast", {}).get("adjacency", {})
            ddg = record["graphs"].get("ddg", {}).get("adjacency", {})
            if not ast.get("node_types"):
                continue
            s = stats[code["language"]]
            s["codes"] += 1
            s["exported_edges"] += len(ddg.get("row", []))
            s["projectable_before"] += _retained(ast, ddg, ddg.get("node_ids", []), max_nodes)
            mapping, reason = fallback_mapping(code["code"], code["language"], ast, ddg)
            s[reason] += 1
            if mapping:
                dest.write(json.dumps({"code_id": str(record["code_id"]), "ast_node_ids": mapping}) + "\n")
                s["projectable_after"] += _retained(ast, ddg, mapping, max_nodes)
            else:
                s["projectable_after"] += _retained(ast, ddg, ddg.get("node_ids", []), max_nodes)
    report = {"method": "source reparse, exact exported AST types/edges, DDG statement type validation",
              "external_nodes": "unmapped; no synthetic AST correspondence", "max_nodes": max_nodes,
              "languages": {lang: dict(counts) for lang, counts in stats.items()}}
    (root / "ddg_alignment_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    path = root / "metadata.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = hashlib.sha256()
    with gzip.open(output, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            content.update(block)
    manifest["ddg_ast_alignment"] = {"file": output.name, "sha256": digest,
                                     "content_sha256": content.hexdigest(), "audit": report}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_alignment(args.root), indent=2))
