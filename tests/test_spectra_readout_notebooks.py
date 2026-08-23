from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "experiments" / "kaggle"
# The RQ2 comparison keeps one folder per benchmark.
RQ2 = KAGGLE / "rq2"
REFERENCE = KAGGLE / "exploratory/06_codenet_nonclone_scopes/02_clone_vs_diff_problem.ipynb"
TEMPLATE = ROOT / "spectral_code/templates/spectra_siam_kaggle_template.ipynb"
RQ1_NOTEBOOKS = (
    ROOT / "experiments/kaggle/rq1/01_atcoder_export_latent_graphs.ipynb",
    ROOT / "experiments/kaggle/rq1/02_xglue_export_latent_graphs.ipynb",
)
# Same model as the reference, plus detached latent tensors returned only under
# ``if not self.training`` so RQ1 can dump latent graphs. The extra tensors
# cannot reach a loss term or the pair classifier, so the run stays comparable.
# Notebooks that additionally dump the latent graph. They carry the same
# model as the reference plus detached tensors returned only under
# ``if not self.training``, so the trained model is unchanged.
LATENT_EXPORTS = {
    RQ2 / "xglue/method/spectra_siam_lex_export_latent_graphs.ipynb",
    *(KAGGLE / "rq1").glob("*_export_latent_graphs.ipynb"),
}
FEATURE_ABLATION_NOTEBOOKS = tuple(
    KAGGLE / "exploratory" / "04_feature_ablation" / name
    for name in (
        "atcoder_v3.ipynb",
        "codexglue_v3.ipynb",
        "codexglue_v3_feature_ablation_canonical.ipynb",
        "codexglue_v3_feature_ablation_lexical.ipynb",
        "gptclonebench_v3.ipynb",
        "semanticclonebench_v3.ipynb",
    )
)
DATASETS = ("atcoder", "xglue", "codenet")
VARIANTS = {
    "topo": (True, False, False, False),
    "label": (False, False, False, False),
    "lex": (False, False, True, False),
}
LEGACY = ("canonical", "lexical", "proposed", "topology_only", "mutation_injection")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _model_cell(notebook: dict) -> str:
    cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "class CanonicalSpectraConfig" in "".join(cell.get("source", []))
        and "class CanonicalSpectraEncoder" in "".join(cell.get("source", []))
        and "class CanonicalSpectraSiam" in "".join(cell.get("source", []))
        and "def canonical_spectra_loss" in "".join(cell.get("source", []))
    ]
    assert len(cells) == 1
    return cells[0]


def _without_latent_export(model: str) -> str:
    """Undo an export notebook's inert additions so it can be compared.

    An export notebook returns the same auxiliary dict as the reference plus
    detached tensors, guarded by ``if not self.training``. Detached tensors
    reach no loss term and the pair classifier never sees them, so the trained
    model is the reference model; only the return plumbing differs.

    The block is removed structurally rather than by matching its exact text,
    because each export notebook words its explanatory comment differently.
    """
    out, skipping = [], False
    for line in model.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "if not self.training:":
            skipping = True                       # drop the guarded block
            continue
        if skipping:
            # The block ends at the first line indented no deeper than 8.
            if stripped and (len(line) - len(line.lstrip())) <= 8:
                skipping = False
            else:
                continue
        if stripped == "return embedding, descriptor, encoder_auxiliary":
            continue
        out.append(line)
    return "".join(out).replace(
        "        encoder_auxiliary = {", "        return embedding, descriptor, {")


def test_every_spectra_notebook_uses_the_complete_latest_reference_model():
    reference_model = _model_cell(_load(REFERENCE))
    implementation_paths = []
    for path in KAGGLE.rglob("*.ipynb"):
        notebook = _load(path)
        if "class CanonicalSpectraSiam" in _source(notebook):
            implementation_paths.append(path)
    implementation_paths.append(TEMPLATE)

    for path in implementation_paths:
        notebook = _load(path)
        source = _source(notebook)
        model = _model_cell(notebook)
        if path in LATENT_EXPORTS:
            model = _without_latent_export(model)
        assert model == reference_model, path
        assert '"hybrid", "eigenvalue_only", "graph_signal_spectral"' in source, path
        assert "readout_mode must be 'hybrid' or 'eigenvalue_only'" not in source, path
        assert "self.block_sizes = (config.density_bins, config.heat_samples)" in source, path
        assert "self.classifier(spectral_pair_features)" in source, path
        assert "EMBEDDING_CONTRASTIVE_WEIGHT = 0.0" in source, path
        assert "embedding_contrastive_weight=EMBEDDING_CONTRASTIVE_WEIGHT" in source, path

    # RQ1 deliberately adds detached export tensors to the same encoder cell,
    # so exact-cell equality is inappropriate; its runnable architecture and
    # classifier must nevertheless expose the same current signatures.
    for path in RQ1_NOTEBOOKS:
        source = _source(_load(path))
        assert '"hybrid", "eigenvalue_only", "graph_signal_spectral"' in source, path
        assert "readout_mode must be 'hybrid' or 'eigenvalue_only'" not in source, path
        assert "self.block_sizes = (config.density_bins, config.heat_samples)" in source, path
        assert "self.classifier(spectral_pair_features)" in source, path


def test_method_directories_contain_only_the_three_input_only_variants():
    for dataset in DATASETS:
        method_dir = RQ2 / dataset / "method"
        names = {path.stem for path in method_dir.glob("spectra_siam_*.ipynb")
                 if path not in LATENT_EXPORTS}
        assert names == {f"spectra_siam_{name}" for name in VARIANTS}
        assert not any((method_dir / f"spectra_siam_{name}.ipynb").exists() for name in LEGACY)


def test_each_ablation_changes_only_encoder_inputs_and_keeps_the_spectral_pair_head():
    for dataset in DATASETS:
        for name, (strip_types, strip_topology, node_lexical, source_lexical) in VARIANTS.items():
            path = RQ2 / dataset / "method" / f"spectra_siam_{name}.ipynb"
            notebook = _load(path)
            source = _source(notebook)
            assert f"STRIP_NODE_TYPES = {strip_types}" in source, path
            assert f"STRIP_TOPOLOGY = {strip_topology}" in source, path
            assert f"USE_NODE_LEXICAL = {node_lexical}" in source, path
            assert f"USE_SOURCE_LEXICAL = {source_lexical}" in source, path
            assert f'METHOD_VARIANT = "input_only_{name}"' in source, path
            assert "The head sees only symmetric, block-wise spectral comparisons." in source, path
            assert "never receives an individual descriptor, code embedding, pooled node" in source, path
            assert "Input-only ablation: the encoder receives no observed graph edges." in source, path
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") == "code":
                    compile("".join(cell.get("source", [])), f"{path}#cell-{index}", "exec")


def test_input_ablation_descriptions_define_a_cumulative_hierarchy():
    expected = {
        "topo": "Only the exported topology enters the encoder",
        "label": "Exported topology and canonical node labels enter the encoder",
        "lex": "exported topology, canonical node labels, and hashed lexical node sketches enter only the encoder",
    }
    for dataset in DATASETS:
        for name, description in expected.items():
            path = RQ2 / dataset / "method" / f"spectra_siam_{name}.ipynb"
            assert description in _source(_load(path)), path


def test_feature_ablation_canonical_and_lexical_arms_are_cumulative():
    canonical = (
        '("canonical", dict(strip_node_types=False, use_node_lexical=False, '
        'use_source_lexical=False, readout_mode="graph_signal_spectral"))'
    )
    lexical = (
        '("lexical", dict(strip_node_types=False, use_node_lexical=True, '
        'use_source_lexical=False, readout_mode="graph_signal_spectral"))'
    )
    for path in FEATURE_ABLATION_NOTEBOOKS:
        text = _source(_load(path))
        if "canonical" in path.stem:
            assert canonical in text, path
        elif "lexical" in path.stem:
            assert lexical in text, path
        else:
            assert canonical in text, path
            assert lexical in text, path


def test_ablation_runs_are_final_and_write_variant_scoped_artifacts():
    """A Kaggle Run All must train and preserve outputs for every arm."""
    for dataset in DATASETS:
        model_cells = []
        for name in VARIANTS:
            path = RQ2 / dataset / "method" / f"spectra_siam_{name}.ipynb"
            notebook = _load(path)
            source = _source(notebook)
            assert 'RUN_PROFILE = "final_full"' in source, path
            assert "RUN_EXPERIMENT = True" in source, path
            assert f'INPUT_ABLATION = "{name}"' in source, path
            assert 'RUN_TAG = f"{DATASET_KEY}_{METHOD_VARIANT}"' in source, path
            # The train invocation and each paper-facing artifact are present.
            assert "RESULTS = run_experiment()" in source, path
            assert "spectra_siam_{RUN_TAG}_final.pt" in source, path
            assert "spectra_siam_{RUN_TAG}_metrics.csv" in source, path
            assert "spectra_siam_{RUN_TAG}_predictions.csv.gz" in source, path
            assert "{RUN_TAG}_spectra_siam_results.csv" in source, path
            assert "{RUN_TAG}_spectra_siam_run_metadata.json" in source, path
            assert "spectra_siam_{RUN_TAG}_result.json" in source, path
            assert "TrainableParameters" in source, path
            model_cells.append(next(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
                if "class CanonicalSpectraSiam" in "".join(cell.get("source", []))
            ))
        # Input flags change tensors before the encoder; they never alter the
        # architecture or the number of trainable parameters.
        assert len(set(model_cells)) == 1, dataset


def test_mutation_injection_package_is_absent():
    assert not (KAGGLE / "exploratory" / "05_mutation_injection").exists()
    assert not (KAGGLE / "MUTATION_INJECTION_PROTOCOL.md").exists()
