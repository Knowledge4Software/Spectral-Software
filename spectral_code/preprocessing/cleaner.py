import os
import json
import pickle
import networkx as nx
from tqdm import tqdm
from spectral_code.preprocessing.simple import SimpleGraphPreprocessor
from spectral_code.utils.dataset import load_graph_from_json


def compose_cpg(layers: dict[str, "nx.DiGraph | None"]) -> "nx.DiGraph | None":
    """Overlay the base layers into a CPG, merging only genuinely shared nodes.

    Joern's layers share one CPG node-id space, so overlaying them by node id is
    correct and is exactly what makes the CPG a true multi-layer graph. The
    locally built Python/C# fallback graphs do not share an id space: each
    builder numbers its own nodes from zero, so AST node "3" and CFG node "3"
    are unrelated program elements. Composing those by id silently dissolved the
    entire CFG into arbitrary AST nodes. When any layer was rebuilt locally, each
    layer therefore contributes namespaced nodes instead.
    """
    named = [(name, graph) for name, graph in layers.items() if graph is not None]
    if not named:
        return None

    # ``source`` is set only by the local fallback builders; Joern-parsed layers
    # never carry it, and it survives the node-link round trip through JSON.
    independent_id_spaces = any(graph.graph.get("source") for _, graph in named)

    cpg = nx.DiGraph()
    for name, graph in named:
        if independent_id_spaces:
            graph = nx.relabel_nodes(graph, {node: f"{name}:{node}" for node in graph.nodes()}, copy=True)
        cpg = nx.compose(cpg, graph)
    if independent_id_spaces:
        cpg.graph["cpg_layers_namespaced"] = True
    return nx.DiGraph(cpg)


def _clean_method_file(file_path: str, base_layers: list[str], preprocessor: SimpleGraphPreprocessor):
    method_id = os.path.basename(file_path).replace(".json", "")
    method_layers = {}
    layers_cleaned = 0

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
            layers_cleaned += 1
        else:
            temp_layers[gtype] = None

        method_layers[gtype] = temp_layers[gtype]

    method_layers["cpg"] = compose_cpg(temp_layers)

    return method_id, method_layers, layers_cleaned

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

        cleaned_graphs_db[method_id]["cpg"] = compose_cpg(temp_layers)

    return cleaned_graphs_db, total_methods_cleaned, total_layers_cleaned


def clean_and_compose_graphs_sharded(
    raw_features_dir: str,
    base_layers: list[str],
    output_dir: str,
    shard_size: int = 1000,
    limit: int = None,
):
    json_files = [f for f in os.listdir(raw_features_dir) if f.endswith(".json")]
    if limit is not None:
        json_files = json_files[:limit]

    os.makedirs(output_dir, exist_ok=True)
    preprocessor = SimpleGraphPreprocessor()

    total_methods_cleaned = 0
    total_layers_cleaned = 0
    shard = {}
    shard_paths = []
    shard_index = 0

    def flush_shard():
        nonlocal shard, shard_index
        if not shard:
            return
        shard_index += 1
        shard_path = os.path.join(output_dir, f"graphs_{shard_index:06d}.pkl")
        with open(shard_path, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_paths.append(shard_path)
        shard = {}

    for file_name in tqdm(json_files, desc="Cleaning & composing graph shards", unit="method"):
        file_path = os.path.join(raw_features_dir, file_name)
        method_id, method_layers, layers_cleaned = _clean_method_file(file_path, base_layers, preprocessor)
        shard[method_id] = method_layers
        total_methods_cleaned += 1
        total_layers_cleaned += layers_cleaned

        if len(shard) >= shard_size:
            flush_shard()

    flush_shard()

    manifest = {
        "format": "cleaned_graph_shards_v1",
        "shard_size": shard_size,
        "total_methods": total_methods_cleaned,
        "total_base_layers_cleaned": total_layers_cleaned,
        "shards": shard_paths,
    }
    manifest_path = os.path.join(os.path.dirname(output_dir), "graph_shards_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path, total_methods_cleaned, total_layers_cleaned
