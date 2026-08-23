from __future__ import annotations

import gzip
import io
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.kaggle.rq1.evaluate_pss import ALL_GRAPH_TYPES, common_support, evaluate, load_latent_export
from experiments.kaggle.rq1.run_table import (
    METHOD_ORDER,
    add_predict_all_clone,
    require_current_main_method_export,
    run_dataset,
    write_paper_files,
)
from spectral_code.similarity.pss import PSSSimilarity


ROOT = Path(__file__).resolve().parents[1]
RQ1 = ROOT / "experiments" / "kaggle" / "rq1"
NOTEBOOKS = (
    RQ1 / "01_atcoder_export_latent_graphs.ipynb",
    RQ1 / "02_xglue_export_latent_graphs.ipynb",
)


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_rq1_notebooks_use_latest_method_and_export_every_latent_graph():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = _source(notebook)
        assert 'RUN_PROFILE = "final_full"' in source, path
        assert "RQ1_EXPORT_LATENT_GRAPHS = True" in source, path
        assert "self.block_sizes = (config.density_bins, config.heat_samples)" in source, path
        assert "self.classifier(spectral_pair_features)" in source, path
        assert "left_embedding * right_embedding" not in source, path
        assert "embedding_contrastive_weight must remain zero" in source, path
        assert "USE_SOURCE_LEXICAL = False" in source, path
        assert 'validation_selection_metric = "Accuracy" if validation_is_balanced else "F1"' in source, path
        assert "choose_threshold(valid_labels, valid_probabilities, validation_selection_metric)" in source, path
        assert '"selection_metric": validation_selection_metric' in source, path
        assert 'encoder_auxiliary["latent_adjacency"] = adjacency.detach()' in source, path
        assert 'encoder_auxiliary["latent_eigenvalues"] = eigenvalues.detach()' in source, path
        assert "ordered_ids = sorted(graphs)" in source, path
        assert "np.savez_compressed(" in source, path
        assert 'compression=zipfile.ZIP_STORED' in source, path
        assert "model, frames, graphs, final_path, validation_selection_metric," in source, path
        assert '"validation_selection_metric": validation_selection_metric' in source, path
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{path}#cell-{index}", "exec")


def _write_synthetic_bundle(root: Path) -> tuple[Path, Path]:
    clean = root / "clean_data"
    clean.mkdir()
    code_ids = [f"c{index}" for index in range(8)]
    pairs = pd.DataFrame(
        [
            ("r1", "train", 1, "c0", "c1"),
            ("r2", "train", 1, "c2", "c3"),
            ("r3", "train", 0, "c0", "c4"),
            ("r4", "train", 0, "c2", "c6"),
            ("v1", "valid", 1, "c0", "c1"),
            ("v2", "valid", 1, "c2", "c3"),
            ("v3", "valid", 0, "c0", "c4"),
            ("v4", "valid", 0, "c2", "c6"),
            ("t1", "test", 1, "c0", "c1"),
            ("t2", "test", 1, "c2", "c3"),
            ("t3", "test", 0, "c1", "c5"),
            ("t4", "test", 0, "c3", "c7"),
        ],
        columns=["pair_id", "split", "label", "left_id", "right_id"],
    )
    records = []
    for index, code_id in enumerate(code_ids):
        base = np.asarray([0.0, 0.2 + 0.02 * (index // 2), 0.8 + 0.03 * (index // 2)])
        records.append(
            {
                "code_id": code_id,
                "graphs": {
                    graph: {"eigenvalues": (base + offset).tolist()}
                    for graph, offset in zip(("ast", "cfg", "ddg", "cpg"), (0.0, 0.01, 0.02, 0.03))
                },
            }
        )
    with gzip.open(clean / "graph_spectra.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    latent_path = root / "rq1_synthetic_latent_graphs.zip"
    pair_bytes = io.BytesIO()
    pairs.to_csv(pair_bytes, index=False, compression="gzip")
    shard_bytes = io.BytesIO()
    latent = np.asarray(
        [[0.0, 0.1 + 0.001 * (index // 2), 1.2 + 0.001 * (index // 2)] for index in range(8)],
        dtype=np.float32,
    )
    np.savez_compressed(
        shard_bytes,
        code_ids=np.asarray(code_ids),
        eigenvalues=latent,
        adjacency=np.zeros((8, 3, 3), dtype=np.float16),
    )
    manifest = {"format": "spectra-rq1-latent-graphs-v1", "dataset": "synthetic", "code_count": 8}
    with zipfile.ZipFile(latent_path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("rq1_pairs.csv.gz", pair_bytes.getvalue())
        archive.writestr("shards/latent_graphs_00000.npz", shard_bytes.getvalue())
    return clean, latent_path


def test_rq1_evaluator_uses_common_pairs_and_writes_all_graph_metrics(tmp_path: Path):
    clean, latent = _write_synthetic_bundle(tmp_path)
    output = tmp_path / "results"
    report = evaluate(
        Namespace(
            clean_data=clean,
            latent_export=latent,
            output_dir=output,
            bootstrap=10,
            seed=42,
        )
    )
    metrics = pd.read_csv(output / "rq1_pss_metrics.csv")
    assert set(metrics.graph_type) == set(ALL_GRAPH_TYPES)
    assert set(metrics.split) == {"valid", "test"}
    assert len(metrics) == len(ALL_GRAPH_TYPES) * 2
    assert report["coverage"]["excluded_pairs"] == 0
    assert report["validation_selection_metric"] == "accuracy"
    assert report["best_conventional_selected_on_validation_roc_auc"] in {"ast", "cfg", "ddg", "cpg"}
    assert (output / "rq1_pair_pss_scores.csv.gz").is_file()
    assert (output / "rq1_latent_effects.csv").is_file()
    assert (output / "rq1_paired_bootstrap.json").is_file()


def test_vectorized_pss_is_scalar_pss_exactly_pairwise_equivalent():
    rng = np.random.default_rng(42)
    left = [np.sort(rng.uniform(0, 2, size=size)) for size in (3, 5, 5, 9, 12, 12)]
    right = [np.sort(rng.uniform(0, 2, size=size)) for size in (4, 5, 8, 3, 12, 7)]
    # Include the normalized-Laplacian zero eigenvalue and artificial padding.
    for values in (*left, *right):
        values[0] = 0.0
    left[-1] = np.pad(left[-1], (0, 3))
    metric = PSSSimilarity(gamma=0.17, distance_power=1.3)
    scalar = np.asarray([metric.compute(a, b) for a, b in zip(left, right)])
    vectorized = metric.compute_many(left, right, batch_size=2)
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-12, atol=1e-12)


def test_rq1_table_runner_produces_all_learned_and_fixed_graph_rows(tmp_path: Path):
    clean, latent = _write_synthetic_bundle(tmp_path)
    args = Namespace(
        resume=False,
        seed=42,
        rf_trees=2,
        rf_max_depth=3,
        rf_jobs=1,
        snn_epochs=1,
        snn_patience=1,
        snn_batch_size=4,
        device="cpu",
    )
    results, common_pairs = run_dataset("atcoder", clean, latent, tmp_path / "table", args)
    expected = {
        f"{'SPECTRA-Siam Spectrum' if graph == 'latent' else graph.upper()} + {learner}"
        for graph in ALL_GRAPH_TYPES for learner in ("No Train", "RF", "LR", "SNN")
    }
    assert set(results.Method) == expected
    assert set(results.columns) >= {"P", "R", "F1", "Acc", "ValidationSelectionMetric"}
    assert set(results.ValidationSelectionMetric) == {"Accuracy"}
    table = add_predict_all_clone("atcoder", results, common_pairs)
    assert "Predict All Clone" in set(table.Method)
    write_paper_files(table, tmp_path / "paper")
    assert (tmp_path / "paper" / "rq1_all_table_rows.csv").is_file()
    assert (tmp_path / "paper" / "rq1_paper_table_values.csv").is_file()
    assert (tmp_path / "paper" / "rq1_paper_table_values.tex").is_file()
    assert (tmp_path / "paper" / "rq1_paper_table.png").is_file()
    assert "Predict All Clone" in METHOD_ORDER


def test_rq1_loader_accepts_complete_kaggle_output_zip(tmp_path: Path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    _, compact = _write_synthetic_bundle(source_root)
    outer = tmp_path / "complete_kaggle_output.zip"
    with zipfile.ZipFile(compact) as source, zipfile.ZipFile(outer, "w") as target:
        for name in source.namelist():
            target.writestr(f"rq1_synthetic_latent_graphs/{name}", source.read(name))
        target.writestr("spectra_siam_synthetic_final.pt", b"not needed by the latent loader")
    manifest, pairs, spectra = load_latent_export(outer)
    assert manifest["dataset"] == "synthetic"
    assert len(pairs) == 12
    assert len(spectra) == 8


def test_rq1_main_method_guard_rejects_stale_source_lexical_setting():
    require_current_main_method_export({
        "dataset": "atcoder_v3",
        "method_input_signature": {
            "input_ablation": "lex", "use_node_lexical": True, "use_source_lexical": False,
        },
    })
    with pytest.raises(RuntimeError, match="Stale/incompatible"):
        require_current_main_method_export({
            "dataset": "atcoder_v3",
            "model_config": {"use_node_lexical": True, "use_source_lexical": True},
        })


def test_rq1_refresh_latent_keeps_fixed_graph_rows(tmp_path: Path):
    clean, latent = _write_synthetic_bundle(tmp_path)
    args = Namespace(
        resume=True, seed=42, rf_trees=2, rf_max_depth=3, rf_jobs=1,
        snn_epochs=1, snn_patience=1, snn_batch_size=4, device="cpu",
        graphs=None, learners=None, skip_snn_graphs=None, refresh_latent=False,
    )
    output = tmp_path / "table"
    first, _ = run_dataset("atcoder", clean, latent, output, args)
    modified = first.copy()
    modified.loc[modified.Method.eq("AST + RF"), "F1"] = 0.123
    modified.to_csv(output / "rq1_table_rows.csv", index=False)
    args.refresh_latent = True
    refreshed, _ = run_dataset("atcoder", clean, latent, output, args)
    assert float(refreshed.loc[refreshed.Method.eq("AST + RF"), "F1"].iloc[0]) == 0.123
    assert set(refreshed.Method) == set(first.Method)


def test_predict_all_clone_is_resume_safe(tmp_path: Path):
    _, latent = _write_synthetic_bundle(tmp_path)
    _, pairs, _ = load_latent_export(latent)
    empty = pd.DataFrame(columns=["Method"])
    once = add_predict_all_clone("atcoder", empty, pairs)
    twice = add_predict_all_clone("atcoder", once, pairs)
    assert list(twice.Method).count("Predict All Clone") == 1


def test_rq1_default_launcher_compiles():
    """The per-dataset shims are gone; run_table.py's CLI covers that case."""
    path = RQ1 / "04_run_all.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_common_support_preserves_official_balanced_validation_policy():
    pairs = pd.DataFrame(
        [
            ("valid", "a", "b", 1), ("valid", "a", "c", 1),
            ("valid", "a", "d", 0), ("valid", "a", "e", 0),
            ("test", "a", "b", 1), ("test", "a", "d", 0),
        ],
        columns=["split", "left_id", "right_id", "label"],
    )
    ids = {"a", "b", "c", "d"}  # endpoint e is unavailable in every view
    conventional = {
        code_id: {graph: np.asarray([0.0, 1.0]) for graph in ("ast", "cfg", "ddg", "cpg")}
        for code_id in ids
    }
    latent = {code_id: np.asarray([0.0, 1.0]) for code_id in ids}
    selected, coverage = common_support(pairs, conventional, latent)
    counts = selected[selected.split.eq("valid")].label.value_counts()
    assert counts.to_dict() == {1: 1, 0: 1}
    assert coverage["validation_balance_dropped"] == 1
