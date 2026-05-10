import time
import tracemalloc

from spectral_code.config import PipelineConfig
from spectral_code.preprocessing.factory import create_preprocessor
from spectral_code.graph.factory import create_graph_builder
from spectral_code.spectral.factory import create_spectral_analyzer
from spectral_code.visualization.factory import create_graph_visualizer


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

        self.visualizer = None
        if config.visualization_enabled:
            self.visualizer = create_graph_visualizer(
                config.visualization_backend,
                **config.visualization_kwargs,
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

        # --- Visualization (optional) ---
        if self.visualizer is not None:
            t0 = time.perf_counter()
            vis_info = self.visualizer.render(
                graph,
                output_dir=self.config.visualization_output_dir,
                graph_type=self.config.graph_type,
                lang=lang,
                title=self.config.visualization_title
                or f"{self.config.graph_type.upper()} graph ({lang})",
                fmt=self.config.visualization_format,
            )
            profile["visualization_time"] = time.perf_counter() - t0
            profile["visualization_image"] = str(vis_info.get("image_path")) if vis_info.get("image_path") else None
            profile["visualization_dot"] = str(vis_info.get("dot_path")) if vis_info.get("dot_path") else None
            profile["visualization_backend"] = str(vis_info.get("backend"))
        else:
            profile["visualization_time"] = 0.0
            profile["visualization_image"] = None
            profile["visualization_dot"] = None
            profile["visualization_backend"] = None

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
            "visualization": {
                "image": profile.get("visualization_image"),
                "dot": profile.get("visualization_dot"),
                "backend": profile.get("visualization_backend"),
            },
            "profile": profile,
        }
