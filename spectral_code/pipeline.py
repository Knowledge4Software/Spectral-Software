import time
import tracemalloc

from spectral_code.config import PipelineConfig
from spectral_code.preprocessing.factory import create_preprocessor
from spectral_code.graph.factory import create_graph_builder
from spectral_code.spectral.factory import create_spectral_analyzer


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        self.preprocessor = create_preprocessor(
            config.preprocessor,
            **config.preprocessor_kwargs
        )

        self.graph_builder = create_graph_builder(
            config.graph_type,
            **config.graph_kwargs
        )

        self.spectral_analyzer = create_spectral_analyzer(
            config.spectral_mode,
            config.eigen_solver,
            config.k_eigen,
            config.solver_kwargs,
            config.spectral_kwargs,
        )

    def run(self, code: str, lang: str = "python"):
        profile = {}

        tracemalloc.start()
        total_start = time.perf_counter()

        # --- Preprocess ---
        t0 = time.perf_counter()
        if self.config.use_preprocessing and self.preprocessor:
            code = self.preprocessor.process(code)
        profile["preprocessing_time"] = time.perf_counter() - t0

        # --- Graph ---
        t0 = time.perf_counter()
        graph = self.graph_builder.build(code, lang=lang)
        profile["graph_time"] = time.perf_counter() - t0
        profile["num_nodes"] = graph.number_of_nodes()
        profile["num_edges"] = graph.number_of_edges()

        # --- Spectral ---
        t0 = time.perf_counter()
        eigvals, eigvecs = self.spectral_analyzer.analyze(graph)
        profile["spectral_time"] = time.perf_counter() - t0

        # --- Total ---
        profile["total_time"] = time.perf_counter() - total_start

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        profile["memory_current"] = current
        profile["memory_peak"] = peak

        return {
            "graph": graph,
            "eigenvalues": eigvals,
            "eigenvectors": eigvecs,
            "profile": profile,
        }