import sys
from pathlib import Path
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.graph.joern_graph import JoernGraphBuilder
from spectral_code.similarity.ged import GEDSimilarity

def test_ged():
    # 1. Define Snippets
    # Snippet A: Simple loop
    code_a = """
    void foo(int n) {
        for(int i=0; i<n; i++) {
            System.out.println(i);
        }
    }
    """
    
    # Snippet B: Identical logic, different variable name
    code_b = """
    void foo(int max_val) {
        for(int counter=0; counter<max_val; counter++) {
            System.out.println(counter);
        }
    }
    """
    
    # Snippet C: Completely different logic
    code_c = """
    void bar(int x) {
        if (x > 10) {
            System.out.println("Big");
        } else {
            System.out.println("Small");
        }
    }
    """

    builder = JoernGraphBuilder(repr_type="cfg")
    ged_sim = GEDSimilarity(timeout=10) # 10 seconds timeout for safety

    print("[*] Generating graphs for snippets...")
    # build() method in JoernGraphBuilder returns a single nx.DiGraph
    # build_all() returns a dict of graphs. Let's use build() or build_all().
    # Looking at joern_graph.py, JoernGraphBuilder inherits from GraphBuilder.
    # Let's see JoernGraphBuilder.build implementation if it exists.
    # Actually I only saw build_all in my previous read. Let's check JoernGraphBuilder again.

    # I'll use build_all to be sure.
    graphs_a = builder.build_all(code_a)
    graphs_b = builder.build_all(code_b)
    graphs_c = builder.build_all(code_c)

    # We'll use CFG for comparison
    g_a = graphs_a["cfg"]
    g_b = graphs_b["cfg"]
    g_c = graphs_c["cfg"]

    print(f"Graph A: {g_a.number_of_nodes()} nodes, {g_a.number_of_edges()} edges")
    print(f"Graph B: {g_b.number_of_nodes()} nodes, {g_b.number_of_edges()} edges")
    print(f"Graph C: {g_c.number_of_nodes()} nodes, {g_c.number_of_edges()} edges")

    print("\n[*] Comparing Pairs:")
    
    # Pair 1: A and B (Should be very similar)
    sim_ab = ged_sim.compute_normalized(g_a, g_b)
    dist_ab = ged_sim.compute(g_a, g_b)
    print(f"A vs B (Similar): Distance = {dist_ab}, Similarity = {sim_ab:.4f}")

    # Pair 2: A and C (Should be different)
    sim_ac = ged_sim.compute_normalized(g_a, g_c)
    dist_ac = ged_sim.compute(g_a, g_c)
    print(f"A vs C (Different): Distance = {dist_ac}, Similarity = {sim_ac:.4f}")

    # Pair 3: Self comparison
    sim_aa = ged_sim.compute_normalized(g_a, g_a)
    print(f"A vs A (Identity): Similarity = {sim_aa:.4f}")

if __name__ == "__main__":
    test_ged()
