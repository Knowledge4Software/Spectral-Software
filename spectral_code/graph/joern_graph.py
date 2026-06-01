import os
import tempfile
import subprocess
import shutil
import uuid
import networkx as nx

from spectral_code.graph.base import GraphBuilder

JOERN_PARSE_BAT = r"C:\joern-cli\joern-parse.bat"
JOERN_EXPORT_BAT = r"C:\joern-cli\joern-export.bat"

class JoernGraphBuilder(GraphBuilder):
    def __init__(self, repr_type: str = "cfg", **kwargs):
        self.repr_type = repr_type
        # Mapping graph types to Joern internal representations
        self.joern_repr_map = {
            "cfg": "cfg",
            "ast": "ast",
            "ddg": "ddg",
            "pdg": "pdg",
        }
        if self.repr_type not in self.joern_repr_map:
            raise ValueError(f"Unsupported Joern representation type: {self.repr_type}")
        
    def _get_cpg(self, code: str, lang: str = "java"):
        """INTERNAL: Generates CPG and returns temp_dir and cpg_path."""
        if lang.lower() == "java":
            class_name = f"Wrapper_{uuid.uuid4().hex[:8]}"
            code = f"public class {class_name} {{\n{code}\n}}"
        
        temp_dir = tempfile.mkdtemp(prefix="joern_bulk_")
        cpg_bin = os.path.join(temp_dir, "cpg.bin")
        
        ext = ".java" if lang.lower() == "java" else ".py"
        src_file = os.path.join(temp_dir, f"source{ext}")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(code)
            
        parse_cmd = [JOERN_PARSE_BAT, src_file, "--output", cpg_bin]
        subprocess.run(parse_cmd, capture_output=True, text=True, shell=True, check=True)
        return temp_dir, cpg_bin

    def _export_from_cpg(self, cpg_bin: str, gtype: str, temp_dir: str) -> nx.Graph:
        """INTERNAL: Exports a specific graph type from an existing CPG."""
        out_dir = os.path.join(temp_dir, gtype)
        export_cmd = [JOERN_EXPORT_BAT, cpg_bin, f"--repr={gtype}", "--out", out_dir]
        subprocess.run(export_cmd, capture_output=True, text=True, shell=True)
        return self._read_dot_from_dir(out_dir)

    def build_all(self, code: str, lang: str = "java") -> dict:
        """
        Optimized method to generate ALL graph types (AST, CFG, DDG, PDG) 
        using only ONE joern-parse call.
        """
        if lang.lower() == "java":
            class_name = f"Wrapper_{uuid.uuid4().hex[:8]}"
            code = f"public class {class_name} {{\n{code}\n}}"
        
        temp_dir = tempfile.mkdtemp(prefix="joern_bulk_")
        cpg_bin = os.path.join(temp_dir, "cpg.bin")
        
        graphs = {}
        
        try:
            # 1. Write source
            ext = ".java" if lang.lower() == "java" else ".py"
            src_file = os.path.join(temp_dir, f"source{ext}")
            with open(src_file, "w", encoding="utf-8") as f:
                f.write(code)
                
            # 2. Generate CPG (The slow part - only do this ONCE)
            parse_cmd = [JOERN_PARSE_BAT, src_file, "--output", cpg_bin]
            subprocess.run(parse_cmd, capture_output=True, text=True, shell=True, check=True)

            # 3. Export each type (Faster because CPG is already there)
            for gtype in ["ast", "cfg", "ddg", "pdg"]:
                out_dir = os.path.join(temp_dir, gtype)
                export_cmd = [JOERN_EXPORT_BAT, cpg_bin, f"--repr={gtype}", "--out", out_dir]
                subprocess.run(export_cmd, capture_output=True, text=True, shell=True)
                
                graphs[gtype] = self._read_dot_from_dir(out_dir)

            return graphs
            
        except Exception as e:
            # Return empty graphs on failure
            return {gt: nx.DiGraph() for gt in ["ast", "cfg", "ddg", "pdg"]}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _read_dot_from_dir(self, out_dir) -> nx.Graph:
        """Helper to find and read the largest dot file in a directory."""
        if not os.path.exists(out_dir):
            return nx.DiGraph()
            
        dot_files = [f for f in os.listdir(out_dir) if f.endswith(".dot")]
        valid_files = [os.path.join(out_dir, f) for f in dot_files if os.path.getsize(os.path.join(out_dir, f)) > 0]
        
        if not valid_files:
            return nx.DiGraph()
            
        selected_file = max(valid_files, key=os.path.getsize)
        try:
            G = nx.drawing.nx_pydot.read_dot(selected_file)
            if len(G.nodes) == 0:
                for f in sorted(valid_files, key=os.path.getsize, reverse=True):
                    alt_G = nx.drawing.nx_pydot.read_dot(f)
                    if len(alt_G.nodes) > 0: return alt_G
            return G
        except:
            return nx.DiGraph()

    def build(self, code: str, lang: str = "java") -> nx.Graph:
        """
        Builds the graph by invoking Joern via subprocess and extracting
        the corresponding .dot files.
        """
        if lang.lower() == "java":
            class_name = f"Wrapper_{uuid.uuid4().hex[:8]}"
            code = f"public class {class_name} {{\n{code}\n}}"
        
        temp_dir = tempfile.mkdtemp(prefix="joern_temp_")
        cpg_bin = os.path.join(temp_dir, "cpg.bin")
        out_dir = os.path.join(temp_dir, "out")
        
        try:
            # write source
            ext = ".java" if lang.lower() == "java" else ".py"
            src_file = os.path.join(temp_dir, f"source{ext}")
            with open(src_file, "w", encoding="utf-8") as f:
                f.write(code)
                
            # Generate CPG
            parse_cmd = [JOERN_PARSE_BAT, src_file, "--output", cpg_bin]
            res_parse = subprocess.run(parse_cmd, capture_output=True, text=True, shell=True)
            if res_parse.returncode != 0:
                return nx.DiGraph()

            # Export to specific representation
            joern_rep = self.joern_repr_map[self.repr_type]
            export_cmd = [JOERN_EXPORT_BAT, cpg_bin, f"--repr={joern_rep}", "--out", out_dir]
            res_export = subprocess.run(export_cmd, capture_output=True, text=True, shell=True)
            
            return self._read_dot_from_dir(out_dir)
            
        except Exception as e:
            print(f"Exception during Joern generation: {e}")
            return nx.DiGraph()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
