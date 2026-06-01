import os
from dataclasses import dataclass, field
from typing import Any

@dataclass
class GridSearchConfig:
    graph_types: list[str] = field(default_factory=lambda: ["ast", "cfg", "ddg", "pdg"])
    k_values: list[Any] = field(default_factory=lambda: [10, 20, 50, "all"])
    similarity_metrics: dict[str, Any] = field(default_factory=dict)
    
    # Dataset limits
    train_samples: int = 300
    test_samples: int = 150
    seed: int = 42
    lang: str = "java"
    
    # Storage Paths
    outputs_dir: str = r"C:\Users\koush\PyProjects\outputs"
    cache_file: str = r"C:\Users\koush\PyProjects\outputs/spectral_features_cache.pkl"
    results_csv: str = r"C:\Users\koush\PyProjects\outputs/grid_search_results.csv"
    models_dir: str = r"C:\Users\koush\PyProjects\outputs/models"

    def __post_init__(self):
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)