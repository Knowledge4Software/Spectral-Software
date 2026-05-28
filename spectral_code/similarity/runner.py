import os
import pickle
from spectral_code.similarity.pss import PSSSimilarity
from spectral_code.similarity.heat_kernel import HeatKernelSimilarity

def run_pairwise_similarity(features_db_path: str, layer: str = "cfg"):
    if not os.path.exists(features_db_path):
        print(f"[-] Database not found: {features_db_path}")
        return None

    with open(features_db_path, "rb") as f:
        features_db = pickle.load(f)

    pss_metric = PSSSimilarity()
    hk_metric = HeatKernelSimilarity()

    method_ids = list(features_db.keys())
    if len(method_ids) < 2:
        print("[-] Not enough methods for comparison.")
        return None

    id1, id2 = method_ids[0], method_ids[1]

    ev1 = features_db[id1][layer]["eigenvalues"]
    ev2 = features_db[id2][layer]["eigenvalues"]

    score_pss = pss_metric.compute(ev1, ev2)
    score_hk = hk_metric.compute(ev1, ev2)

    return {
        "id1": id1,
        "id2": id2,
        "nodes1": len(ev1),
        "nodes2": len(ev2),
        "score_pss": score_pss,
        "score_hk": score_hk
    }
