import os
import json
import networkx as nx

def process_single_method(args):
    position_idx, idx, joern_base_out, output_dir, graph_types, per_method_cpg_time, per_layer_export_time = args
    results = {}
    
    for gtype in graph_types:
        layer_folder = os.path.join(joern_base_out, gtype)
        target_dot = f"{position_idx}-{gtype}.dot"
        file_path = os.path.join(layer_folder, target_dot)

        graph = nx.DiGraph()
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                raw_g = nx.drawing.nx_pydot.read_dot(file_path)
                graph = nx.DiGraph(raw_g)
            except Exception:
                pass
                    
        graph_data = nx.node_link_data(graph) if graph.number_of_nodes() > 0 else None
        results[gtype] = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "graph_data": graph_data,
            "eigenvalues": None,
            "metrics": {
                "cpg_time": per_method_cpg_time,
                "export_time": per_layer_export_time[gtype],
                "total_time": per_method_cpg_time + per_layer_export_time[gtype]
            }
        }
        
    out_file = os.path.join(output_dir, f"{idx}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"idx": int(idx), "features": results}, f)
    return True
