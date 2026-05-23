import os
import json
import networkx as nx

def load_graph_from_json(json_path, graph_type="cfg"):
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Force lowercase to match "ast", "cfg", "ddg", "pdg" keys in JSON
    layer_key = graph_type.lower()
    
    graph_data = data.get("features", {}).get(layer_key, {}).get("graph_data")
    return nx.node_link_graph(graph_data) if graph_data else None

def get_all_dataset_ids(features_dir):
    if not os.path.exists(features_dir):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(features_dir) if f.endswith(".json")]