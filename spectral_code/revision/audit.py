"""Language-wise exported-graph coverage without claiming unavailable parser logs."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .canonical_types import get_canonical_id
from .data import CleanDataCorpus, _open_text, _project_ddg


def audit_graphs(root: Path, max_nodes: int = 256) -> dict:
    languages = {}
    with gzip.open(root / "codes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            languages[str(row["code_id"])] = row["language"]
    stats, types = defaultdict(Counter), defaultdict(Counter)
    with gzip.open(root / "graph_spectra.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            language = languages.get(str(row["code_id"]))
            if language is None:
                continue
            s = stats[language]
            s["codes"] += 1
            layers = row.get("graphs", {})
            ast = layers.get("ast", {}).get("adjacency", {})
            raw_types = ast.get("node_types", [])
            s["ast_backbone_available"] += bool(raw_types)
            s["truncated_codes"] += len(raw_types) > max_nodes
            s["retained_nodes"] += min(max_nodes, len(raw_types))
            types[language].update(str(t) for t in raw_types[:max_nodes])
            for layer in ("ast", "cfg", "ddg", "cpg"):
                info = layers.get(layer, {})
                s[f"{layer}_missing_spectra"] += not bool(info.get("eigenvalues"))
                status = str(info.get("spectral_status", "unreported"))
                s[f"{layer}_status_{status}"] += 1
            ddg = layers.get("ddg", {}).get("adjacency", {})
            ast_index = {str(node): i for i, node in enumerate(ast.get("node_ids", [])[:max_nodes])}
            ddg_ids = ddg.get("node_ids", [])
            edges = list(zip(ddg.get("row", []), ddg.get("col", [])))
            s["ddg_empty_export"] += not bool(edges)
            s["ddg_exported_edges"] += len(edges)
            for a, b in edges:
                a, b = int(a), int(b)
                if 0 <= a < len(ddg_ids) and 0 <= b < len(ddg_ids):
                    s["ddg_edges_projectable_at_cap"] += str(ddg_ids[a]) in ast_index and str(ddg_ids[b]) in ast_index
    unknown = get_canonical_id("Canonical_Unknown")
    report = {"max_nodes": max_nodes, "parser_success": "unavailable from exports alone; AST availability is a proxy",
              "ddg_warning": "An empty DDG is not necessarily a parser failure", "languages": {}}
    for language, counts in stats.items():
        counts["unknown_retained_nodes"] = sum(n for raw, n in types[language].items()
                                               if get_canonical_id(raw, language=language) == unknown)
        report["languages"][language] = {
            **dict(counts), "truncated_fraction": counts["truncated_codes"] / max(1, counts["codes"]),
            "unknown_retained_node_fraction": counts["unknown_retained_nodes"] / max(1, counts["retained_nodes"]),
            "ddg_projectable_edge_fraction": counts["ddg_edges_projectable_at_cap"] / max(1, counts["ddg_exported_edges"]),
        }
    (root / "coverage_by_language.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit_split_coverage(root: Path, max_nodes: int = 256) -> dict:
    """Coverage on actual retained pairs, with raw and repaired DDG counts."""
    corpus = CleanDataCorpus(root)
    membership = {}
    for split in ("train", "valid", "test"):
        frame = corpus.pairs(split)
        for key in set(frame.left_id) | set(frame.right_id):
            if key in membership and membership[key] != split:
                raise ValueError("Endpoint occurs in multiple splits")
            membership[key] = split
    stats, seen = defaultdict(Counter), set()
    unknown = get_canonical_id("Canonical_Unknown")
    with _open_text(corpus.graphs_path) as stream:
        for line in stream:
            row = json.loads(line)
            key = str(row.get("code_id"))
            if key not in membership:
                continue
            seen.add(key)
            language, split = corpus.code_languages[key], membership[key]
            counts = stats[(language, split)]
            counts["programs"] += 1
            layers = row.get("graphs", {})
            ast = layers.get("ast", {}).get("adjacency", {})
            types = ast.get("node_types", [])
            counts["ast_available"] += bool(types)
            counts["truncated_codes"] += len(types) > max_nodes
            retained = types[:max_nodes]
            counts["retained_nodes"] += len(retained)
            counts["unknown_retained_nodes"] += sum(get_canonical_id(t, language=language) == unknown for t in retained)
            ddg = layers.get("ddg", {}).get("adjacency", {})
            counts["empty_ddg_export"] += not bool(ddg.get("row"))
            counts["ddg_exported_edges"] += len(ddg.get("row", []))
            counts["ddg_projectable_raw"] += len(_project_ddg(ast, ddg, max_nodes))
            repaired = {**ddg, "ast_node_ids": corpus.ddg_alignment[key]} if key in corpus.ddg_alignment else ddg
            counts["ddg_projectable_after_alignment"] += len(_project_ddg(ast, repaired, max_nodes))
    for key in set(membership) - seen:
        stats[(corpus.code_languages[key], membership[key])]["missing_graph_export"] += 1
    rows = []
    for (language, split), counts in sorted(stats.items()):
        rows.append({"language": language, "split": split, **dict(counts),
                     "truncated_fraction": counts["truncated_codes"] / max(1, counts["programs"]),
                     "unknown_node_fraction": counts["unknown_retained_nodes"] / max(1, counts["retained_nodes"])})
    pd.DataFrame(rows).to_csv(root / "coverage_by_language_split.csv", index=False)
    result = {"max_nodes": max_nodes, "records": rows,
              "parser_success": "Unavailable from exports; AST availability is not a parser success log",
              "ddg_failure": "Empty DDGs are reported as empty exports, not extraction failures"}
    (root / "coverage_by_language_split.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--by-split", action="store_true")
    args = parser.parse_args()
    print(json.dumps((audit_split_coverage if args.by_split else audit_graphs)(args.root, args.max_nodes), indent=2))
