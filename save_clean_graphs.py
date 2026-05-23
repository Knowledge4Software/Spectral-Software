import os
import sys
import pickle
import json
import networkx as nx
from tqdm import tqdm

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from spectral_code.utils.dataset import load_graph_from_json

FEATURES_DIR = os.path.join(BASE_DIR, "outputs", "dataset_features")
GRAPH_OUT_DIR = os.path.join(BASE_DIR, "outputs", "clean_graphs")
os.makedirs(GRAPH_OUT_DIR, exist_ok=True)

GRAPH_TYPES = ["ast", "cfg", "ddg", "pdg"]

def main():
    if not os.path.exists(FEATURES_DIR):
        print(f"[-] Directory not found: {FEATURES_DIR}")
        return

    json_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(".json")]
    print(f"[*] Found {len(json_files)} JSON files on disk. Processing sequentially...")

    # Dictionary format: graph_db[method_id][gtype] = networkx_graph_object
    graph_db = {}
    total_graphs_saved = 0

    for file_name in tqdm(json_files, desc="Cleaning Graphs"):
        method_id = os.path.splitext(file_name)[0]
        json_path = os.path.join(FEATURES_DIR, file_name)
        
        for gtype in GRAPH_TYPES:
            try:
                graph = load_graph_from_json(json_path, graph_type=gtype)
                if graph is not None and graph.number_of_nodes() > 0:
                    # Convert to undirected graph
                    undirected_graph = graph.to_undirected()
                    
                    # Correct NetworkX function for isolated nodes
                    isolated_nodes = list(nx.isolates(undirected_graph))
                    undirected_graph.remove_nodes_from(isolated_nodes)
                    
                    # If graph still has valid structure, save it
                    if undirected_graph.number_of_nodes() > 0:
                        if method_id not in graph_db:
                            graph_db[method_id] = {}
                        graph_db[method_id][gtype] = undirected_graph
                        total_graphs_saved += 1
            except Exception as e:
                print(f"\n[-] Error processing Method {method_id} [{gtype}]: {e}")

    # Save the entire clean graph database into a pickle file
    output_path = os.path.join(GRAPH_OUT_DIR, "cleaned_graphs_db.pkl")
    print(f"\n[*] Writing {total_graphs_saved} graphs to high-performance binary file...")
    
    with open(output_path, "wb") as f:
        pickle.dump(graph_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[+] Success! Cleaned graphs database saved perfectly to '{output_path}'")
    print(f"[+] Final file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()