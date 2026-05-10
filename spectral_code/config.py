from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    use_preprocessing: bool = True

    preprocessor: str | None = "simple"
    graph_type: str = "ast"

    spectral_mode: str = "laplacian"
    eigen_solver: str = "auto"

    k_eigen: int = 20

    # NEW: kwargs per stage
    preprocessor_kwargs: dict = field(default_factory=dict)
    graph_kwargs: dict = field(default_factory=dict)
    spectral_kwargs: dict = field(default_factory=dict)
    solver_kwargs: dict = field(default_factory=dict)

    # Visualization
    visualization_enabled: bool = False
    visualization_backend: str | None = "matplotlib"
    visualization_output_dir: str = "artifacts/graph_visualizations"
    visualization_format: str = "png"
    visualization_title: str | None = None
    visualization_kwargs: dict = field(default_factory=dict)
