import os
from spectral_code.similarity.runner import run_pairwise_similarity

FEATURES_DB_PATH = r"C:\Users\koush\PyProjects\Spectral-Software\outputs\spectral_features\spectral_vectors_full.pkl"

def main():
    print(f"[*] Proceeding to calculate pairwise similarity scores...")
    results = run_pairwise_similarity(FEATURES_DB_PATH, layer="cfg")
    if results:
        print(f"[+] Method 1 Nodes: {results['nodes1']} | Method 2 Nodes: {results['nodes2']}")
        print(f"[+] PSS Similarity: {results['score_pss']:.6f}")
        print(f"[+] Heat Kernel Similarity: {results['score_hk']:.6f}")

if __name__ == "__main__":
    main()