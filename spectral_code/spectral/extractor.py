import time
import numpy as np
from tqdm import tqdm
from spectral_code.spectral.factory import create_spectral_analyzer

def extract_all_spectral_features(graph_db: dict, graph_types: list[str], mode: str = "normalized_laplacian"):
    analyzer = create_spectral_analyzer(
        mode=mode,
        solver_name="dense",
        k=0,
        solver_kwargs={},
        spectral_kwargs={}
    )
    
    features_db = {}
    layer_counts = {gtype: 0 for gtype in graph_types}
    layer_durations = {gtype: 0.0 for gtype in graph_types}
    layer_node_sums = {gtype: 0 for gtype in graph_types}

    for method_id, layers in tqdm(graph_db.items(), desc="Processing Methods"):
        features_db[method_id] = {}
        for gtype in graph_types:
            graph = layers.get(gtype)
            if graph is not None and graph.number_of_nodes() > 0:
                nodes_count = graph.number_of_nodes()
                layer_node_sums[gtype] += nodes_count
                
                t0 = time.perf_counter()
                try:
                    raw_eigs, _ = analyzer.analyze(graph)
                    elapsed_time = time.perf_counter() - t0
                    features_db[method_id][gtype] = {
                        "eigenvalues": raw_eigs,
                        "compute_time_seconds": elapsed_time
                    }
                    layer_counts[gtype] += 1
                    layer_durations[gtype] += elapsed_time
                except Exception:
                    features_db[method_id][gtype] = {
                        "eigenvalues": np.array([], dtype=np.float64),
                        "compute_time_seconds": 0.0
                    }
            else:
                features_db[method_id][gtype] = {
                    "eigenvalues": np.array([], dtype=np.float64),
                    "compute_time_seconds": 0.0
                }
                
    return features_db, layer_counts, layer_durations, layer_node_sums
