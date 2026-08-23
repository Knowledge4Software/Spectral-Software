from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "kaggle/exploratory/06_codenet_nonclone_scopes"
BASE = ROOT / "experiments/kaggle/rq2/codenet/method/spectra_siam_lex.ipynb"
NOTEBOOKS = {
    "01_clone_vs_aw.ipynb": {
        "clone": 4_000,
        "hard_nonclone": 4_000,
    },
    "02_clone_vs_diff_problem.ipynb": {
        "clone": 4_000,
        "nonclone_diff_problem": 4_000,
    },
    "03_clone_vs_mixed_aw_diff.ipynb": {
        "clone": 4_000,
        "hard_nonclone": 2_000,
        "nonclone_diff_problem": 2_000,
    },
}
BASELINE_NOTEBOOKS = {
    filename.replace(".ipynb", f"_{method}.ipynb"): {**counts}
    for filename, counts in NOTEBOOKS.items()
    for method in ("astnn", "deepsim", "rtvnn")
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _selector_namespace(notebook: dict) -> dict:
    training = "".join(notebook["cells"][11]["source"])
    tree = ast.parse(training)
    names = {
        "EXPECTED_CONFIGURATIONS",
        "PAIR_KIND_TARGETS_PER_CONFIGURATION",
        "NONCLONE_KINDS",
        "FORBIDDEN_PAIR_KINDS",
        "PAIR_SCOPE_AUDIT",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "select_nonclone_scope_protocol":
            nodes.append(node)
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "scope-selector", "exec"), namespace)
    return namespace


def _baseline_selector_namespace(notebook: dict) -> dict:
    runtime = "".join(notebook["cells"][3]["source"])
    tree = ast.parse(runtime)
    names = {
        "EXPECTED_CONFIGURATIONS",
        "PAIR_KIND_TARGETS_PER_CONFIGURATION",
        "NONCLONE_KINDS",
        "FORBIDDEN_PAIR_KINDS",
        "PAIR_SCOPE_AUDIT",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "select_nonclone_scope_protocol":
            nodes.append(node)
    namespace = {"pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "baseline-scope-selector", "exec"), namespace)
    return namespace


def test_scope_notebooks_compile_and_keep_the_redesigned_method_2_core():
    base_model_cell = "".join(_load(BASE)["cells"][9]["source"])
    assert set(path.name for path in PACKAGE.glob("*.ipynb")) == set(NOTEBOOKS) | set(BASELINE_NOTEBOOKS)
    for filename in NOTEBOOKS:
        path = PACKAGE / filename
        notebook = _load(path)
        source = _source(notebook)
        config = "".join(notebook["cells"][1]["source"])
        final_profile = config[
            config.index('RUN_PRESETS["final_full"] = {'):
            config.index("# --- bounded run budget ---")
        ]
        assert 'READOUT_MODE = "graph_signal_spectral"' in source
        assert 'RUN_PROFILE = "final_full"' in source
        assert "SEED = 42" in config
        assert '"epochs": 8' in final_profile
        assert '"epochs": 4' not in final_profile
        assert "USE_SOURCE_LEXICAL = False" in source
        assert 'FORBIDDEN_PAIR_KINDS = ("nonclone_mutation",)' in source
        assert "select_nonclone_scope_protocol(raw, split)" in source
        assert "stratified_pair_cap(selected" not in source
        assert '"PairScopeAudit": PAIR_SCOPE_AUDIT' in source
        assert "MUTATION_INJECTION_PATTERN" not in source
        model_cell = "".join(notebook["cells"][9]["source"])
        slug = filename.removesuffix(".ipynb").removeprefix("01_").removeprefix("02_").removeprefix("03_")
        # The scope notebooks keep the same descriptor-only pair-head contract
        # as the maintained lexical-input method; only pair selection differs.
        assert "spectral_pair_features" in base_model_cell
        assert f'METHOD_VARIANT = "spectral_descriptor_pair_{slug}"' in config
        assert "EMBEDDING_CONTRASTIVE_WEIGHT = 0.0" in config
        assert "AUC_RANKING_WEIGHT = 0.10" in config
        assert "relation_priors = []" not in model_cell
        assert "joint_descriptor" not in model_cell
        assert "relation_descriptors" not in model_cell
        assert "spectral_pair_features" in model_cell
        assert "self.classifier(spectral_pair_features)" in model_cell
        assert "3 * config.readout_descriptor_dim + 3 * len(self.block_sizes)" in model_cell
        assert "pair_hidden_dim = config.hidden_dim + config.hidden_dim // 2" in model_cell
        assert "pair_bottleneck_dim = config.hidden_dim - config.hidden_dim // 8" in model_cell
        assert "block_cosines" in model_cell
        assert "auc_ranking_weight * auc_ranking" in model_cell
        assert "left_embedding,\n                right_embedding," not in model_cell
        assert "symmetric_density_heat_chebyshev_comparisons_only" in source
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{path}#cell-{index}", "exec")


def test_each_notebook_selects_the_exact_uniform_protocol_and_no_mutations():
    configurations = (
        "python", "java", "cpp", "csharp", "python_java", "python_cpp",
        "python_csharp", "java_cpp", "java_csharp", "cpp_csharp",
    )
    rows = []
    for configuration in configurations:
        for pair_kind, label in (
            ("clone", 1),
            ("hard_nonclone", 0),
            ("nonclone_diff_problem", 0),
            ("nonclone_mutation", 0),
        ):
            count = 280 if pair_kind != "nonclone_mutation" else 3
            for index in range(count):
                rows.append({
                    "configuration_id": configuration,
                    "pair_kind": pair_kind,
                    "label": label,
                    "pair_id": f"{configuration}/{pair_kind}/{index:04d}",
                    "left_id": f"l/{configuration}/{pair_kind}/{index}",
                    "right_id": f"r/{configuration}/{pair_kind}/{index}",
                })
    frame = pd.DataFrame(rows)

    for filename, expected_total in NOTEBOOKS.items():
        namespace = _selector_namespace(_load(PACKAGE / filename))
        selected, audit = namespace["select_nonclone_scope_protocol"](frame, "train")
        expected_train = {kind: int(total * 0.7) for kind, total in expected_total.items()}
        assert selected.pair_kind.value_counts().to_dict() == expected_train
        bucket_sizes = selected.groupby(["pair_kind", "configuration_id"]).size()
        assert all(count == 1 for count in bucket_sizes.groupby(level="pair_kind").nunique())
        assert "nonclone_mutation" not in set(selected.pair_kind)
        assert audit["excluded_mutation_pairs"] == 30

        targets = namespace["PAIR_KIND_TARGETS_PER_CONFIGURATION"]
        aggregate = {
            kind: sum(
                per_configuration * len(configurations)
                for split_targets in targets.values()
                for candidate, per_configuration in split_targets.items()
                if candidate == kind
            )
            for kind in expected_total
        }
        assert aggregate == expected_total


def test_astnn_deepsim_and_rtvnn_notebooks_use_the_identical_fixed_pair_protocol():
    configurations = (
        "python", "java", "cpp", "csharp", "python_java", "python_cpp",
        "python_csharp", "java_cpp", "java_csharp", "cpp_csharp",
    )
    rows = []
    for configuration in configurations:
        for pair_kind, label in (
            ("clone", 1),
            ("hard_nonclone", 0),
            ("nonclone_diff_problem", 0),
            ("nonclone_mutation", 0),
        ):
            count = 280 if pair_kind != "nonclone_mutation" else 3
            for index in range(count):
                rows.append({
                    "configuration_id": configuration,
                    "pair_kind": pair_kind,
                    "label": label,
                    "pair_id": f"{configuration}/{pair_kind}/{index:04d}",
                    "left_id": f"l/{configuration}/{pair_kind}/{index}",
                    "right_id": f"r/{configuration}/{pair_kind}/{index}",
                })
    frame = pd.DataFrame(rows)

    for filename, expected_total in BASELINE_NOTEBOOKS.items():
        notebook = _load(PACKAGE / filename)
        source = _source(notebook)
        method = next(
            candidate
            for candidate in ("astnn", "deepsim", "rtvnn")
            if filename.endswith(f"_{candidate}.ipynb")
        )
        assert 'RUN_PROFILE = "final_full"' in source
        assert 'DATASET_KEYS = ("codenet-4l-clean-data",)' in source
        assert f"run_faithful_baseline(key, '{method}')" in source
        assert "select_nonclone_scope_protocol" in source
        assert "pair_scope_audit" in source
        assert "lost pairs during graph joining" in source
        assert "nonclone_mutation" in source
        assert "codenet_4l_clean_data.zip" in source
        assert "bundle_root = _materialize_codenet_zip_input()" in source
        assert '"graph_spectra.jsonl.gz.tmp"' in source
        assert "with _open_text(pairs_path) as pair_stream:" in source
        assert "CfgFallbackCodes" in source
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{filename}#cell-{index}", "exec")

        namespace = _baseline_selector_namespace(notebook)
        selected, audit = namespace["select_nonclone_scope_protocol"](frame, "train")
        expected_train = {kind: int(total * 0.7) for kind, total in expected_total.items()}
        assert selected.pair_kind.value_counts().to_dict() == expected_train
        assert "nonclone_mutation" not in set(selected.pair_kind)
        assert audit["excluded_mutation_pairs"] == 30


def test_fixed_pipeline_runner_excludes_mutations_and_targets_12k():
    runner = (ROOT / "create_datasets_graphs/codenet_4l/nonclone_scope_study/run_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert 'PAIR_KINDS = "clone,hard_nonclone,nonclone_diff_problem"' in runner
    assert 'GRAPH_TYPES = "ast,cfg,ddg,cpg"' in runner
    assert "SAMPLE_SIZE = 12_000" in runner
    assert "MIN_PROGRAM_LINES = 20" in runner
    assert "MAX_PROGRAM_LINES = 50" in runner
    assignment = runner.index("PAIR_KINDS =")
    assert "nonclone_mutation" not in runner[assignment:assignment + 100]
