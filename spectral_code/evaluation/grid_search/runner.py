import os
import pickle
import numpy as np
import pandas as pd
from typing import Any
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from spectral_code.evaluation.grid_search.config import GridSearchConfig
from spectral_code.evaluation.grid_search.cache_manager import FeatureCacheManager

class GridSearchRunner:
    def __init__(self, config: GridSearchConfig):
        self.config = config
        self.cache_manager = FeatureCacheManager(config.cache_file)

    def _prepare_vectors(self, pairs: list, gtype: str, k: Any, metric_obj: Any, features_cache: dict):
        X, y = [], []
        for p in pairs:
            ev1 = features_cache[p.left_id].get(gtype, np.array([]))
            ev2 = features_cache[p.right_id].get(gtype, np.array([]))
            
            if k != "all" and isinstance(k, int):
                ev1_k = ev1[:k]
                ev2_k = ev2[:k]
            else:
                ev1_k = ev1
                ev2_k = ev2
                
            score = metric_obj.compute(ev1_k, ev2_k)
            X.append([score])
            y.append(p.label)
        return np.array(X), np.array(y)

    def run(self, train_pairs: list, test_pairs: list) -> pd.DataFrame:
        all_pairs = train_pairs + test_pairs
        features_cache = self.cache_manager.extract_features(
            all_pairs, self.config.graph_types, lang=self.config.lang
        )

        grid_results = []
        total_combos = len(self.config.graph_types) * len(self.config.k_values) * len(self.config.similarity_metrics)
        pbar = tqdm(total=total_combos, desc="Evaluating Grid Configurations")

        for gtype in self.config.graph_types:
            for k in self.config.k_values:
                for metric_name, metric_obj in self.config.similarity_metrics.items():
                    
                    X_train, y_train = self._prepare_vectors(train_pairs, gtype, k, metric_obj, features_cache)
                    X_test, y_test = self._prepare_vectors(test_pairs, gtype, k, metric_obj, features_cache)

                    if len(X_train) == 0 or len(X_test) == 0:
                        pbar.update(1)
                        continue

                    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.config.seed, n_jobs=-1)
                    clf.fit(X_train, y_train)
                    
                    # Optional: Save model object to outputs/models
                    model_path = os.path.join(self.config.models_dir, f"model_{gtype}_{k}_{metric_name}.pkl")
                    with open(model_path, "wb") as f:
                        pickle.dump(clf, f)

                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average='binary', pos_label=1, zero_division=0
                    )
                    
                    grid_results.append({
                        "Graph Type": gtype.upper(),
                        "K Truncation": str(k),
                        "Similarity Metric": metric_name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1-Score": f1
                    })
                    pbar.update(1)
                    
        pbar.close()
        
        df_results = pd.DataFrame(grid_results)
        df_results.to_csv(self.config.results_csv, index=False)
        return df_results