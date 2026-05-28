import os
import json
import networkx as nx
from tqdm import tqdm
from spectral_code.preprocessing.simple import SimpleGraphPreprocessor
from spectral_code.utils.dataset import load_graph_from_json

def clean_and_compose_graphs(raw_features_dir: str, base_layers: list[str], limit: int = None):
    json_files = [f for f in os.listdir(raw_features_dir) if f.endswith('.json')]
    if limit is not None:
        json_files = json_files[:limit]
        
    preprocessor = SimpleGraphPreprocessor()
    cleaned_graphs_db = {}
    
    total_methods_cleaned = 0
    total_layers_cleaned = 0
    
    for file_name in tqdm(json_files, desc="Cleaning & Composing Directed Layers"):
        file_path = os.path.join(raw_features_dir, file_name)
        method_id = file_name.replace(".json", "")
        cleaned_graphs_db[method_id] = {}
        total_methods_cleaned += 1
        
        temp_layers = {}
        for gtype in base_layers:
            raw_graph = load_graph_from_json(file_path, graph_type=gtype)
            if raw_graph is not None and raw_graph.number_of_nodes() > 0:
                raw_graph = nx.DiGraph(raw_graph)
                cleaned = preprocessor.process(raw_graph)
                cleaned = nx.DiGraph(cleaned)
                
                if cleaned.number_of_nodes() == 0:
                    cleaned = raw_graph
                    
                temp_layers[gtype] = cleaned
                total_layers_cleaned += 1
            else:
                temp_layers[gtype] = None
                
            cleaned_graphs_db[method_id][gtype] = temp_layers[gtype]

        valid_graphs = [g for g in temp_layers.values() if g is not None]
        if valid_graphs:
            cpg = nx.DiGraph(valid_graphs[0].copy())
            for g in valid_graphs[1:]:
                cpg = nx.compose(cpg, g)
            cleaned_graphs_db[method_id]["cpg"] = nx.DiGraph(cpg)
        else:
            cleaned_graphs_db[method_id]["cpg"] = None
            
    return cleaned_graphs_db, total_methods_cleaned, total_layers_cleaned
