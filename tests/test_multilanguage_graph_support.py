from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectral_code.graph.factory import create_graph_builder
from spectral_code.graph.joern_graph import JoernGraphBuilder
from spectral_code.preprocessing.data_unpacker import unpack_jsonl_to_java
from spectral_code.preprocessing.graph_parser import (
    build_csharp_ast_graph,
    build_csharp_cfg_graph,
    build_csharp_ddg_graph,
    is_user_method,
)
from spectral_code.preprocessing.joern_runner import _joern_parse_cmd, _joern_parse_mode
from spectral_code.preprocessing.language_support import (
    joern_language,
    normalize_source_language,
    source_extension,
)


@pytest.mark.parametrize(
    ("label", "canonical", "extension", "frontend"),
    [
        ("Java", "java", ".java", "javasrc"),
        ("Python", "python", ".py", "pythonsrc"),
        ("py3", "python", ".py", "pythonsrc"),
        ("C", "c", ".c", "c"),
        ("C++", "cpp", ".cpp", "c"),
        ("cxx", "cpp", ".cpp", "c"),
        ("C#", "csharp", ".cs", "csharpsrc"),
        ("csharpsrc", "csharp", ".cs", "csharpsrc"),
    ],
)
def test_language_registry(label, canonical, extension, frontend):
    assert normalize_source_language(label) == canonical
    assert source_extension(label) == extension
    assert joern_language(label) == frontend


def test_unpacker_preserves_each_language_extension(tmp_path: Path):
    records = [
        {"idx": 1, "lang": "C++", "func": "#include <vector>\nint main(){std::vector<int> v; return 0;}"},
        {"idx": 2, "lang": "C#", "func": "int Add(int a, int b) { return a + b; }"},
        {"idx": 3, "lang": "Java", "func": "int add(int a, int b) { return a + b; }"},
        {"idx": 4, "lang": "Python", "func": "def add(a, b):\n    return a + b"},
        {"idx": 5, "lang": "C", "func": "int main(void){return 0;}"},
    ]
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sources"
    output_dir.mkdir()

    assert unpack_jsonl_to_java(str(data_file), str(output_dir)) == ["1", "2", "3", "4", "5"]
    assert (output_dir / "m_1.cpp").read_text(encoding="utf-8").startswith("#include <vector>")
    assert "class Wrapper_2" in (output_dir / "m_2.cs").read_text(encoding="utf-8")
    assert "class Wrapper_3" in (output_dir / "m_3.java").read_text(encoding="utf-8")
    assert (output_dir / "m_4.py").read_text(encoding="utf-8").startswith("def add")
    assert (output_dir / "m_5.c").read_text(encoding="utf-8").startswith("int main")


@pytest.mark.parametrize(
    ("configured", "frontend"),
    [
        ("C++", "c"),
        ("cpp", "c"),
        ("C#", "csharpsrc"),
        ("Python", "pythonsrc"),
        ("Java", "javasrc"),
    ],
)
def test_batch_joern_command_selects_correct_frontend(monkeypatch, configured, frontend):
    monkeypatch.setenv("JOERN_LANGUAGE", configured)
    monkeypatch.delenv("JOERN_USE_DIRECT_FRONTEND", raising=False)
    command = _joern_parse_cmd("joern-parse", "src", "cpg.bin")
    assert command == ["joern-parse", "src", "--language", frontend, "--output", "cpg.bin"]
    assert frontend in _joern_parse_mode() or configured.lower() in {"c++", "cpp"}


def test_single_record_builder_prepares_cpp_and_full_units():
    cpp_source, cpp_extension, cpp_frontend = JoernGraphBuilder._prepare_source(
        "#include <iostream>\nint main(){return 0;}", "C++"
    )
    assert cpp_extension == ".cpp"
    assert cpp_frontend == "c"
    assert cpp_source.startswith("#include <iostream>")

    java_source, java_extension, java_frontend = JoernGraphBuilder._prepare_source(
        "public class Demo { public static void main(String[] args) {} }", "java"
    )
    assert java_extension == ".java"
    assert java_frontend == "javasrc"
    assert java_source.count("class Demo") == 1

    assert create_graph_builder("cpg").repr_type == "cpg"


def test_python_module_qualified_functions_are_not_discarded():
    assert is_user_method("<module>.add:ANY(int,int)")
    assert not is_user_method("<module>:ANY()")
    assert not is_user_method("<operator>.addition:ANY(ANY,ANY)")


def test_csharp_fallbacks_produce_ast_cfg_and_ddg(tmp_path: Path):
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_c_sharp")
    source = tmp_path / "m_7.cs"
    source.write_text(
        """
        class Demo {
            static int Sum(int n) {
                int total = 0;
                for (int i = 0; i < n; i++) { total += i; }
                if (total > 2) { return total; }
                return -1;
            }
        }
        """,
        encoding="utf-8",
    )

    ast_graph = build_csharp_ast_graph(source)
    cfg_graph = build_csharp_cfg_graph(source)
    ddg_graph = build_csharp_ddg_graph(source)

    assert ast_graph.number_of_nodes() > 10
    assert cfg_graph.number_of_nodes() >= 5
    assert any(data.get("type") == "loop" for *_, data in cfg_graph.edges(data=True))
    assert ddg_graph.number_of_nodes() >= 5
    assert ddg_graph.number_of_edges() > 0
    assert cfg_graph.graph["source"] == "csharp_tree_sitter_cfg_fallback"
    assert ddg_graph.graph["source"] == "csharp_tree_sitter_ddg_fallback"


def test_csharp_ddg_tolerates_recovery_method_without_body(tmp_path: Path):
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_c_sharp")
    source = tmp_path / "m_1291.cs"
    source.write_text(
        """
        class Wrapper {
            static bool Verify(object input) {
                var values = items.Where(x = > x.Enabled);
                return values.Count() > 0;
            }
        }
        """,
        encoding="utf-8",
    )

    graph = build_csharp_ddg_graph(source)

    assert graph.number_of_nodes() > 0
    assert graph.graph["source"] == "csharp_tree_sitter_ddg_fallback"
