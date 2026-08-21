from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import networkx as nx

from spectral_code.graph.base import GraphBuilder
from spectral_code.preprocessing.language_support import (
    joern_language,
    normalize_source_language,
    source_extension,
)


def _joern_executable(env_name: str, windows_name: str, posix_name: str) -> str:
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
        existing_command(os.getenv(env_name)),
        *(str(path) for path in joern_home_candidates if path.exists()),
        shutil.which(windows_name),
        shutil.which(posix_name),
        *(str(path) for path in common_candidates if path.exists()),
        os.getenv(env_name),
        windows_name if os.name == "nt" else posix_name,
    ]
    return next(str(candidate) for candidate in candidates if candidate)


def _bat_safe_cmd(command: list[str]) -> list[str]:
    if os.name == "nt" and command and str(command[0]).lower().endswith((".bat", ".cmd")):
        return ["cmd.exe", "/d", "/c", *command]
    return command


JOERN_PARSE_BAT = _joern_executable("JOERN_PARSE_BAT", "joern-parse.bat", "joern-parse")
JOERN_EXPORT_BAT = _joern_executable("JOERN_EXPORT_BAT", "joern-export.bat", "joern-export")

JAVA_UNIT_RE = re.compile(
    r"(?m)^\s*(?:package\s+[\w.]+\s*;|import\s+[\w.*]+\s*;|"
    r"(?:public\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+\w+)"
)
CSHARP_UNIT_RE = re.compile(
    r"(?m)^\s*(?:using\s+[\w.]+\s*;|namespace\s+[\w.]+|"
    r"(?:public\s+)?(?:abstract\s+|sealed\s+|static\s+)?(?:class|interface|struct|enum|record)\s+\w+)"
)


class JoernGraphBuilder(GraphBuilder):
    """Build one or more Joern graph representations for a source record.

    ``lang`` accepts dataset labels and common aliases including ``C++``,
    ``cpp``, ``C#``, ``cs``, ``py`` and Joern's own frontend names.  C++ files
    retain a ``.cpp`` suffix and are routed to Joern's ``c2cpg`` frontend via
    ``--language c``.
    """

    REPRESENTATIONS = ("ast", "cfg", "ddg", "pdg", "cpg")
    BUILD_ALL_REPRESENTATIONS = ("ast", "cfg", "ddg", "pdg")

    def __init__(self, repr_type: str = "cfg", **kwargs):
        del kwargs
        self.repr_type = repr_type.strip().lower()
        if self.repr_type not in self.REPRESENTATIONS:
            supported = ", ".join(self.REPRESENTATIONS)
            raise ValueError(f"Unsupported Joern representation type: {repr_type!r}; expected {supported}")

    @staticmethod
    def _prepare_source(code: str, lang: str, source_mode: str | None = None) -> tuple[str, str, str]:
        language = normalize_source_language(lang)
        mode = (source_mode or "auto").strip().lower()
        is_unit = mode in {"compilation_unit", "program", "file", "full"}
        if mode == "auto":
            is_unit = (
                language == "java" and bool(JAVA_UNIT_RE.search(code))
            ) or (
                language == "csharp" and bool(CSHARP_UNIT_RE.search(code))
            )

        if language == "java" and not is_unit:
            class_name = f"Wrapper_{uuid.uuid4().hex[:8]}"
            code = f"public class {class_name} {{\n{code}\n}}\n"
        elif language == "csharp" and not is_unit:
            class_name = f"Wrapper_{uuid.uuid4().hex[:8]}"
            code = f"public class {class_name} {{\n{code}\n}}\n"
        else:
            code = code.rstrip() + "\n"
        return code, source_extension(language), joern_language(language)

    @staticmethod
    def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            _bat_safe_cmd(command),
            capture_output=True,
            text=True,
            check=check,
        )

    def _get_cpg(self, code: str, lang: str = "java", source_mode: str | None = None):
        """Generate a CPG and return its temporary directory and path."""
        source, extension, frontend = self._prepare_source(code, lang, source_mode)
        temp_dir = tempfile.mkdtemp(prefix="joern_bulk_")
        cpg_bin = os.path.join(temp_dir, "cpg.bin")
        src_file = os.path.join(temp_dir, f"source{extension}")
        Path(src_file).write_text(source, encoding="utf-8")
        try:
            self._run([JOERN_PARSE_BAT, src_file, "--language", frontend, "--output", cpg_bin])
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return temp_dir, cpg_bin

    def _export_from_cpg(self, cpg_bin: str, gtype: str, temp_dir: str) -> nx.DiGraph:
        out_dir = os.path.join(temp_dir, gtype)
        self._run([JOERN_EXPORT_BAT, cpg_bin, f"--repr={gtype}", "--out", out_dir])
        return self._read_dot_from_dir(out_dir)

    def build_all(
        self,
        code: str,
        lang: str = "java",
        source_mode: str | None = None,
    ) -> dict[str, nx.DiGraph]:
        """Generate AST/CFG/DDG/PDG from one parse call."""
        temp_dir = None
        try:
            temp_dir, cpg_bin = self._get_cpg(code, lang, source_mode)
            graphs = {}
            for gtype in self.BUILD_ALL_REPRESENTATIONS:
                try:
                    graphs[gtype] = self._export_from_cpg(cpg_bin, gtype, temp_dir)
                except (OSError, subprocess.SubprocessError) as exc:
                    graph = nx.DiGraph()
                    graph.graph["parse_error"] = f"{type(exc).__name__}: {exc}"
                    graphs[gtype] = graph
            return graphs
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return {
                gtype: self._error_graph(exc)
                for gtype in self.BUILD_ALL_REPRESENTATIONS
            }
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _error_graph(exc: Exception) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.graph["parse_error"] = f"{type(exc).__name__}: {exc}"
        return graph

    @staticmethod
    def _read_dot_from_dir(out_dir: str) -> nx.DiGraph:
        root = Path(out_dir)
        if not root.exists():
            return nx.DiGraph()
        valid_files = [path for path in root.rglob("*.dot") if path.stat().st_size > 0]
        if not valid_files:
            return nx.DiGraph()
        for selected_file in sorted(valid_files, key=lambda path: path.stat().st_size, reverse=True):
            try:
                graph = nx.drawing.nx_pydot.read_dot(str(selected_file))
            except Exception:
                continue
            if graph.number_of_nodes() > 0:
                return nx.DiGraph(graph)
        return nx.DiGraph()

    def build(
        self,
        code: str,
        lang: str = "java",
        source_mode: str | None = None,
    ) -> nx.DiGraph:
        temp_dir = None
        try:
            temp_dir, cpg_bin = self._get_cpg(code, lang, source_mode)
            return self._export_from_cpg(cpg_bin, self.repr_type, temp_dir)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return self._error_graph(exc)
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
