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
        
    def build(self, code: str, lang: str = "java") -> nx.Graph:
        """
        Builds the graph by invoking Joern via subprocess and extracting
        the corresponding .dot files.
        """
        # Wrap snippet in a dummy class to enforce correct parsing by Joern
        if lang.lower() == "java":
            class_name = f"Wrapper_{uuid.uuid4().hex[:8]}"
            code = f"public class {class_name} {{\n{code}\n}}"
        
        temp_dir = tempfile.mkdtemp(prefix="joern_temp_")
        cpg_bin = os.path.join(temp_dir, "cpg.bin")
        out_dir = os.path.join(temp_dir, "out")
        
        try:
            # write source
            ext = ".java" if lang.lower() == "java" else ".py" # Adjust to needs
            src_file = os.path.join(temp_dir, f"source{ext}")
            with open(src_file, "w", encoding="utf-8") as f:
                f.write(code)
                
            # Generate CPG
            parse_cmd = [JOERN_PARSE_BAT, src_file, "--output", cpg_bin]
            res_parse = subprocess.run(parse_cmd, capture_output=True, text=True, shell=True)
            if res_parse.returncode != 0:
                print(f"Joern parse error:\n{res_parse.stdout}\n{res_parse.stderr}")
                return nx.DiGraph() # Return empty map if fails

            # Export to specific representation
            joern_rep = self.joern_repr_map[self.repr_type]
            export_cmd = [JOERN_EXPORT_BAT, cpg_bin, f"--repr={joern_rep}", "--out", out_dir]
            res_export = subprocess.run(export_cmd, capture_output=True, text=True, shell=True)
            if res_export.returncode != 0:
                print(f"Joern export error:\n{res_export.stdout}\n{res_export.stderr}")
                return nx.DiGraph()

            # Find output graphs inside exported directory
            if not os.path.exists(out_dir):
                return nx.DiGraph()
                
            dot_files = [f for f in os.listdir(out_dir) if f.endswith(".dot")]
            valid_files = [os.path.join(out_dir, f) for f in dot_files if os.path.getsize(os.path.join(out_dir, f)) > 0]
            
            if not valid_files:
                return nx.DiGraph()
            
            # Select largest dot graph (Main function body)
            # Joern can sometimes generate 0-cfg.dot (for Method), 1-cfg.dot (for sub-logic), etc.
            # We want to pick the file that contains the core logic of our function.
            # Usually sorting by size puts the wrapper or main logic first.
            
            # Print for debug when nodes size is suspicious:
            # print(f"Valid dot files found: {[os.path.basename(f) for f in valid_files]}")
            
            # Select max size dot file (this works for CFG/PDG/DDG but AST might sometimes be split weirdly)
            selected_file = max(valid_files, key=os.path.getsize)
            
            # Read via pydot to networkx structure
            G = nx.drawing.nx_pydot.read_dot(selected_file)
            
            # NetworkX sometimes fails to parse nodes if quotes are unescaped in networkx pydot reader, 
            # causing an empty graph silently. Let's explicitly check and fallback if needed.
            if len(G.nodes) == 0:
                print(f"Warning: Selected file {os.path.basename(selected_file)} parsed to 0 nodes. Trying others...")
                for f in sorted(valid_files, key=os.path.getsize, reverse=True):
                    if f == selected_file: continue
                    alt_G = nx.drawing.nx_pydot.read_dot(f)
                    if len(alt_G.nodes) > 0:
                        return alt_G

            return G
            
        except Exception as e:
            print(f"Exception during Joern generation: {e}")
            return nx.DiGraph()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
