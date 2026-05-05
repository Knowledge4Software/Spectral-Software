from spectral_code.pipeline import Pipeline
from spectral_code.config import PipelineConfig


def pretty_print(result):
    profile = result["profile"]

    print("\n" + "="*50)
    print("PIPELINE PROFILE")
    print("="*50)

    print(f"Nodes: {profile['num_nodes']}")
    print(f"Edges: {profile['num_edges']}")

    print("\n--- Time (seconds) ---")
    print(f"Preprocessing : {profile['preprocessing_time']:.6f}")
    print(f"Graph Build   : {profile['graph_time']:.6f}")
    print(f"Spectral      : {profile['spectral_time']:.6f}")
    print(f"Total         : {profile['total_time']:.6f}")

    print("\n--- Memory (bytes) ---")
    print(f"Current: {profile['memory_current']}")
    print(f"Peak   : {profile['memory_peak']}")

    print("\n--- Eigenvalues (first 10) ---")
    print(result["eigenvalues"][:10])

    print("="*50 + "\n")


if __name__ == "__main__":
    code = """
        def add(a, b):
            return a + b
        """

    config = PipelineConfig(
        eigen_solver="dense",
        k_eigen=None,
    )

    pipeline = Pipeline(config)
    result = pipeline.run(code)

    pretty_print(result)