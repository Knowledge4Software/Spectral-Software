from __future__ import annotations

from pathlib import Path
import gzip
import hashlib

import pytest

from spectral_code.evaluation.clean_data_export import _graph_to_sparse_adjacency
from spectral_code.preprocessing.graph_parser import (build_python_ast_graph, build_python_ddg_graph,
                                                      build_csharp_ast_graph, build_csharp_ddg_graph)
from spectral_code.revision.alignment import fallback_mapping
from spectral_code.revision.data import _project_ddg, verify_alignment


@pytest.mark.parametrize("language,code,builders", [
    ("python", "def f(x):\n    y = x + 1\n    z = y * 2\n    return z\n",
     (build_python_ast_graph, build_python_ddg_graph)),
    ("csharp", "class Demo { static int F(int x) { int y = x + 1; int z = y * 2; return z; } }",
     (build_csharp_ast_graph, build_csharp_ddg_graph)),
])
def test_source_validated_fallback_mapping_restores_real_dependency_edges(tmp_path, language, code, builders):
    source = tmp_path / ("sample.py" if language == "python" else "sample.cs")
    source.write_text(code, encoding="utf-8")
    ast = _graph_to_sparse_adjacency(builders[0](source))
    ddg = _graph_to_sparse_adjacency(builders[1](source), include_node_labels=False)
    before = _project_ddg(ast, ddg, len(ast["node_ids"]))
    mapping, reason = fallback_mapping(source.read_text(), language, ast, ddg)
    assert reason == "validated_ast_and_ddg_statement_types"
    assert all(mapping[i] is None for i, name in enumerate(ddg["node_ids"]) if str(name).startswith("input:"))
    aligned = {**ddg, "ast_node_ids": mapping}
    after = _project_ddg(ast, aligned, len(ast["node_ids"]))
    assert len(after) > len(before)
    assert (after >= 0).all() and (after < len(ast["node_ids"])).all()
    damaged = {**ast, "row": [], "col": []}
    assert fallback_mapping(source.read_text(), language, damaged, ddg)[1] == "ast_reproduction_mismatch"


def test_negative_ddg_positions_are_rejected():
    ast = {"node_ids": ["a", "b"]}
    ddg = {"node_ids": ["a", "b"], "row": [-1], "col": [0]}
    assert len(_project_ddg(ast, ddg, 2)) == 0


@pytest.mark.parametrize("layout", ("gzip_tmp", "plain"))
def test_alignment_accepts_kaggle_layout_and_rejects_changed_payload(tmp_path, layout):
    payload = b'{"code_id":"1","ast_node_ids":["0",null]}\n'
    compressed = gzip.compress(payload)
    spec = {"file": "ddg_ast_alignment.jsonl.gz", "sha256": hashlib.sha256(compressed).hexdigest(),
            "content_sha256": hashlib.sha256(payload).hexdigest()}
    name = "ddg_ast_alignment.jsonl.gz.tmp" if layout == "gzip_tmp" else "ddg_ast_alignment.jsonl"
    nested = tmp_path / "extracted"
    nested.mkdir()
    path = nested / name
    path.write_bytes(compressed if layout == "gzip_tmp" else payload)
    assert verify_alignment(tmp_path, spec) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_alignment(tmp_path, spec)
