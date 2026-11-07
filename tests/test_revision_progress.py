import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.sync_notebook_progress import setup_source, update_notebook
from spectral_code.revision import data, progress, training
from spectral_code.revision.model import RevisionModelConfig
from spectral_code.revision.protocol import RunConfig
from spectral_code.revision.versioning import training_implementation_hash


@pytest.fixture(autouse=True)
def reset_adapters():
    progress.uninstall_progress()
    yield
    progress.uninstall_progress()


@pytest.mark.parametrize("compressed", (False, True))
def test_progress_reader_preserves_payload_and_counts_physical_bytes(tmp_path, compressed):
    path = tmp_path / ("codes.jsonl.gz" if compressed else "codes.jsonl")
    content = 'first\nsecond \u03bb\nthird\n'
    if compressed:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(content)
    else:
        path.write_text(content, encoding="utf-8")
    original = data._open_text
    original_hash = training_implementation_hash()
    progress.install_progress()
    progress.install_progress()  # repeated setup must not double-wrap loaders
    with data._open_text(path) as stream:
        assert "".join(stream) == content
        assert stream.bar.n == path.stat().st_size
    with data._open_text(path) as stream:
        assert stream.readline() + stream.read() == content
    assert training_implementation_hash() == original_hash
    progress.uninstall_progress()
    assert data._open_text is original


def _small_data():
    graphs = {}
    edges = np.array([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int32)
    for i in range(4):
        graphs[str(i)] = data.CanonicalGraph(str(i), "java", np.arange(3) + 1,
                                             np.full((3, 4), i + 1, dtype=np.int64),
                                             np.zeros(96, dtype=np.int64), (edges, edges, edges), {})
    frame = pd.DataFrame([{"left_id": str(i % 4), "right_id": str((i + 1) % 4),
                           "label": i % 2, "pair_id": str(i)} for i in range(8)])
    return {"train": frame, "valid": frame[:4], "test": frame[4:]}, graphs


def test_progress_preserves_training_exactly_and_resumes_existing_checkpoint(tmp_path):
    torch.set_num_threads(1)
    frames, graphs = _small_data()
    config = RunConfig(max_epochs=2, min_epochs=2, patience=3, batch_size=2,
                       effective_batch_size=4, max_nodes=8, amp=False)
    model = RevisionModelConfig(hidden_dim=16, latent_nodes=4, max_input_nodes=8,
                                 spectral_bands=3, spectral_signals=2, density_bins=4,
                                 heat_samples=3, chebyshev_degree=3)
    baseline = tmp_path / "plain"
    training.train_session(frames, graphs, config, model, training.LOSS_WEIGHTS, baseline, data_hash="same")
    progress.install_progress()
    from spectral_code.revision import benchmark
    assert benchmark._loader is not training._loader
    shown = tmp_path / "shown"
    training.train_session(frames, graphs, config, model, training.LOSS_WEIGHTS, shown, data_hash="same")
    a = torch.load(baseline / "progress.pt", weights_only=False)
    b = torch.load(shown / "progress.pt", weights_only=False)
    assert a["identity"] == b["identity"]
    assert a["history"] == b["history"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    # Reattach a checkpoint produced without progress; it remains compatible.
    continued = training.train_session(frames, graphs, config, model, training.LOSS_WEIGHTS,
                                       tmp_path / "continued", data_hash="same",
                                       resume=baseline / "progress.pt")
    assert continued["complete"] and continued["identity"] == a["identity"]
    with pytest.raises(ValueError, match="identity mismatch"):
        training.train_session(frames, graphs, replace(config, learning_rate=2e-4), model,
                               training.LOSS_WEIGHTS, tmp_path / "bad", data_hash="same",
                               resume=baseline / "progress.pt")


def test_state_adapters_only_report_persisted_token_and_embedding_counts(tmp_path):
    progress.install_progress()
    from spectral_code.revision import transformer, embeddings
    state = {"codes": 10, "next_index": 4}
    source = tmp_path / "tokens.tmp"; target = tmp_path / "tokens.json"
    source.write_text(json.dumps(state))
    transformer.replace_file(source, target)
    assert json.loads(target.read_text()) == state
    assert progress._state_bars[str(target)].n == 4
    embedding = tmp_path / "embeddings.json"
    embeddings._json_save(state, embedding)
    assert json.loads(embedding.read_text()) == state
    assert progress._state_bars[str(embedding)].n == 4
    progress.uninstall_progress()
    assert not progress._state_bars


def test_notebook_progress_sync_is_idempotent_and_preserves_code(tmp_path):
    path = tmp_path / "example.ipynb"
    path.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["\n\nvalue = 42\n"], "execution_count": 1, "outputs": [{"text": "old"}]},
        {"cell_type": "code", "source": ["value += 1\n"], "execution_count": None, "outputs": []}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    assert update_notebook(path)
    first = path.read_bytes()
    assert not update_notebook(path)
    assert path.read_bytes() == first
    notebook = json.loads(path.read_text())
    assert "value = 42" in "".join(notebook["cells"][0]["source"])
    assert "total=2" in "".join(notebook["cells"][0]["source"])
    assert all(not c["outputs"] and c["execution_count"] is None for c in notebook["cells"])
    compile(setup_source(2), "progress_setup", "exec")


def test_notebook_release_bytes_are_stable_across_windows_newlines(tmp_path):
    path = tmp_path / "windows.ipynb"
    path.write_bytes(json.dumps({"cells": [{"cell_type": "code", "source": ["value = 42\n"],
                                           "outputs": [], "execution_count": None}],
                                "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=1).replace("\n", "\r\n").encode())
    assert update_notebook(path)
    assert b"\r\n" not in path.read_bytes()
    assert not update_notebook(path)


def test_notebook_progress_preserves_future_import_and_cell_magic_positions(tmp_path):
    path = tmp_path / "future.ipynb"
    path.write_text(json.dumps({"cells": [
        {"cell_type": "code", "source": ["%%capture\nprint('setup')\n"], "outputs": [], "execution_count": None},
        {"cell_type": "code", "source": ["from __future__ import annotations\nvalue = 42\n"], "outputs": [], "execution_count": None}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    assert update_notebook(path)
    notebook = json.loads(path.read_text())
    assert "".join(notebook["cells"][0]["source"]).startswith("%%capture\n")
    source = "".join(notebook["cells"][1]["source"])
    assert source.startswith("from __future__ import annotations\n")
    compile(source, "future", "exec")
    first = path.read_bytes()
    assert not update_notebook(path)
    assert path.read_bytes() == first
