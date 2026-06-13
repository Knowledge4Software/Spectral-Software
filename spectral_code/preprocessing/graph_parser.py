import os
import json
import networkx as nx
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

def _quick_analyze_dot(fpath):
    """
    Very fast regex-based analysis to determine method ID and node count 
    without full graph parsing.
    """
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            full_content = f.read()
            
            # Find m_{idx} in digraph name or label
            found_idx = re.search(r'm_(\d+)', full_content)
            if not found_idx: return None
            
            idx = found_idx.group(1)
            
            # Count nodes roughly (score)
            nodes = re.findall(r'^\s*"\d+"\s*\[label', full_content, re.MULTILINE)
            
            # Return tuple for build_dot_index: (id, path, score)
            # Score is node count; if multiple DOTs match the same ID, 
            # we pick the one with more nodes (usually the method vs a summary)
            return (idx, fpath, len(nodes))
    except Exception:
        return None

def build_dot_index(joern_base_out, graph_types):
    """
    Scans the exported DOT files in parallel and builds a mapping.
    """
    index = {g: {} for g in graph_types}
    
    for gtype in graph_types:
        folder = os.path.join(joern_base_out, gtype)
        if not os.path.exists(folder): continue
        
        files = [
            str(path)
            for path in tqdm(Path(folder).rglob("*.dot"), desc=f"Scanning {gtype.upper()} DOT files", unit="file")
        ]
        
        # Parallelize the metadata scan with tqdm
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(_quick_analyze_dot, files), 
                                total=len(files), 
                                desc=f"Indexing {gtype.upper()}", 
                                unit="dot"))
            
        for res in results:
            if res:
                idx, fpath, score = res
                if idx not in index[gtype] or score > index[gtype][idx][1]:
                    index[gtype][idx] = (fpath, score)
                
    return {g: {idx: val[0] for idx, val in items.items()} for g, items in index.items()}

def parse_joern_dot_fast(fpath):
    """
    Optimized DOT parser for Joern's specific format.
    Handles various label formats (with/without parentheses, with/without <BR/>).
    """
    G = nx.DiGraph()
    # Broadly match node ID and everything inside label = <...>
    node_re = re.compile(r'^\s*"(\d+)"\s*\[\s*label = <(.*?)>\s*\]', re.MULTILINE | re.DOTALL)
    edge_re = re.compile(r'^\s*"(\d+)"\s*->\s*"(\d+)"', re.MULTILINE)
    
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract nodes
        for nid, raw_label in node_re.findall(content):
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
    except Exception:
        return nx.DiGraph()

def process_single_method(args):
    idx, paths, output_dir, per_method_cpg_time, per_layer_export_time = args
    results = {}
    
    for gtype, fpath in paths.items():
        if fpath and os.path.exists(fpath):
            # Use the fast parser instead of networkx default
            graph = parse_joern_dot_fast(fpath)
            
            if graph.number_of_nodes() == 0:
                 results[gtype] = {"nodes": 0, "edges": 0, "graph_data": None}
            else:
                results[gtype] = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "graph_data": nx.node_link_data(graph)
                }
        else:
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
    return True
