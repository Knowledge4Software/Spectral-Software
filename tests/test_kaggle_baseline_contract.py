from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# One folder per benchmark under the RQ2 comparison.
KAGGLE = ROOT / "kaggle" / "rq2"
DATASETS = ("atcoder", "xglue", "codenet", "gptclonebench", "semanticclonebench")
BASELINES = (
    "astnn_baseline.ipynb",
    "cdlh_baseline.ipynb",
    "deckard_baseline.ipynb",
    "deepsim_baseline.ipynb",
    "fa_ast_ggnn_baseline.ipynb",
    "fa_ast_gmn_baseline.ipynb",
    "gnn_baselines.ipynb",
    "rtvnn_baseline.ipynb",
    "snn_baselines.ipynb",
)

# CDLH and the two FA-AST variants are far slower per pair than the other
# baselines, so on the three large datasets they run the bounded 50k profile
# instead of the full split; anything else exhausts a Kaggle session. The
# reduction is in the number of pairs, never in model capacity, and every
# method compared against them on that dataset uses the same pair budget. The
# two small benchmarks keep the full split. Every other clause of the contract
# below still applies to these notebooks.
SESSION_50K = {
    (dataset, baseline)
    for dataset in ("atcoder", "xglue", "codenet")
    for baseline in ("cdlh_baseline.ipynb", "fa_ast_ggnn_baseline.ipynb",
                     "fa_ast_gmn_baseline.ipynb")
}


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_all_baselines_keep_the_shared_full_split_and_result_contract():
    for dataset in DATASETS:
        for filename in BASELINES:
            path = KAGGLE / dataset / "baselines" / filename
            source = _source(path)
            expected = ("session_50k" if (dataset, filename) in SESSION_50K
                        else "final_full")
            assert f'RUN_PROFILE = "{expected}"' in source, path
            assert 'RUN_CONFIG["epochs"] = 4' in source, path
            assert "model = nn.DataParallel(" not in source, path
            assert "train_test_split" not in source, path
            assert "random_split(" not in source, path
            assert "pairs.csv" in source or "CleanDataCorpus" in source, path
            if filename not in {"gnn_baselines.ipynb", "snn_baselines.ipynb"}:
                assert "# Execute the configured baseline." in source, path
                assert "FAITHFUL_RESULTS = [run_faithful_baseline" in source, path
                # Folder, ZIP, and Kaggle's temporary-extension layouts are
                # all valid ways to attach the final clean-data bundle.
                assert "faithful_baseline_input_cache" in source, path
                assert "attached clean-data ZIP" in source, path
                assert '"graph_spectra.jsonl.gz.tmp"' in source, path
                assert "Shared faithful-baseline schedule" in source, path
                assert 'RUN_CONFIG["epochs"] = 4' in source, path
                assert 'RUN_CONFIG.setdefault("patience", 2)' in source, path
                assert 'PYTORCH_ALLOC_CONF", "expandable_segments:True"' in source, path
                assert "model = nn.DataParallel(" not in source, path
                assert '"data_parallel": False' in source, path
                assert '"used_gpu_count": 1' in source, path
                assert "auxiliary = auxiliary.mean()" in source, path
                assert "_adaptive_train_step" in source, path
                assert "_predict_rows_adaptive" in source, path
                assert "size-bucketed FA-AST batches" in source, path
                assert "def _flow(self, rows" in source, path
                assert "required_layers: tuple[str, ...] | None" in source, path
                assert "del raw" in source, path
                if filename == "cdlh_baseline.ipynb":
                    assert '"cdlh": {"name": "CDLH", "batch": 16' in source, path
                    assert "use_reentrant=False" in source, path
                    if dataset == "codenet":
                        assert '"cdlh": 4' in source, path
            if filename == "gnn_baselines.ipynb":
                assert 'BATCH_SIZE = int(globals().get("GNN_BATCH_SIZE", 64))' in source, path
                assert "predict_batch_adaptive" in source, path
                assert "gnn_is_cuda_oom" in source, path
                assert "training_microbatch = BATCH_SIZE" in source, path
                assert '"MinimumTrainingMicrobatch"' in source, path
                assert '"MacroF1": test_metrics["MacroF1"]' in source, path
                assert '"BalancedAccuracy": test_metrics["BalancedAccuracy"]' in source, path
                assert '"ROC_AUC": test_metrics["ROC_AUC"]' in source, path
                assert "reduce `BATCH_SIZE` from `2048`" not in source, path
                assert "MAX_GRAPH_NODES" in source, path
            if filename == "snn_baselines.ipynb":
                assert "batchnorm_singleton_guard" in source, path
                assert 'BATCH_SIZE = int(globals().get("SNN_BATCH_SIZE", 8192))' in source, path
            if dataset == "codenet":
                assert "CodeNet 50k resource guard" in source, path
                assert "MAX_NODES = 128" in source, path
            for field in (
                "P",
                "R",
                "F1",
                "Acc",
                "MacroF1",
                "BalancedAccuracy",
                "ROC_AUC",
                "TrainableParameters",
                "RuntimeSeconds",
                "RuntimeMinutes",
            ):
                assert f'"{field}"' in source, (path, field)


def test_all_baseline_code_cells_compile():
    for dataset in DATASETS:
        for filename in BASELINES:
            path = KAGGLE / dataset / "baselines" / filename
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "code":
                    compile("".join(cell.get("source", [])), f"{path}#cell-{index}", "exec")


def test_all_baselines_use_the_same_label_balance_selection_policy():
    for dataset in DATASETS:
        for filename in BASELINES:
            path = KAGGLE / dataset / "baselines" / filename
            source = _source(path)
            assert '"metric": "Accuracy" if balanced else "F1"' in source, path
            assert "best_f1_threshold" not in source, path
