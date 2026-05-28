import os
import pickle
import numpy as np
from tqdm import tqdm
from spectral_code.config import PipelineConfig
from spectral_code.pipeline import Pipeline

class FeatureCacheManager:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def extract_features(self, pairs: list, graph_types: list[str], lang: str = "java") -> dict:
        unique_codes = {}
        for p in pairs:
            unique_codes[p.left_id] = p.left_code
            unique_codes[p.right_id] = p.right_code

        is_updated = False
        for code_id, code_content in tqdm(unique_codes.items(), desc="Extracting Spectra (Cache)"):
            if code_id not in self.cache:
                self.cache[code_id] = {}
                
            for gtype in graph_types:
                if gtype in self.cache[code_id]:
                    continue
                    
                try:
                    config = PipelineConfig(
                        graph_type=gtype,
                        k_eigen=100,
                        use_preprocessing=False,
                        visualization_enabled=False
                    )
                    pipeline = Pipeline(config)
                    result = pipeline.run(code_content, lang=lang)
                    
                    eigvals = np.asarray(result["eigenvalues"], dtype=float)
                    self.cache[code_id][gtype] = np.sort(eigvals)
                except Exception:
                    self.cache[code_id][gtype] = np.array([], dtype=float)
                is_updated = True

        if is_updated:
            self.save_cache()
            
        return self.cache