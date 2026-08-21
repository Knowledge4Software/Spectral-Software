"""Build the Section 6.5 timing notebooks: one per benchmark.

Each notebook reruns the canonical lexical method unchanged and records, for
every batch of every epoch, how long each stage of the methodology took.

Nothing about the model, data, optimizer, or seed changes, so the accuracy
these runs report should match the RQ2 runs for the same benchmark; only
measurement is added. Instrumentation wraps each module's ``forward`` rather
than editing the model, so the timed code is the code the paper ran.

GPU work is asynchronous, so device stages are timed with CUDA events recorded
on the stream. A wall-clock timer around a CUDA call would measure the kernel
launch, not the kernel.

Outputs per run, in /kaggle/working:
  d5_batch_timings_<dataset>.csv    one row per batch, one column per stage
  d5_epoch_timings_<dataset>.csv    one row per epoch
  d5_stage_summary_<dataset>.csv    per-stage mean/std/95% CI over batches
  d5_timing_summary_<dataset>.json  totals, throughput, device, run metadata
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BLOCKS = Path(__file__).resolve().parent / "_d5_blocks"

# (rq2 folder, paper label, expected clean-data attachment)
BENCHMARKS = (
    ("codenet", "CodeNet", "codenet_4l_clean_data.zip"),
    ("atcoder", "AtCoder", "atcoder_v3_clean_data.zip"),
    ("xglue", "BigCloneBench", "codexglue_v3_clean_data.zip"),
)


def _cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": source.splitlines(keepends=True)}


def training_edits() -> list[tuple[str, str]]:
    """The exact substitutions that turn the training loop into a timed one."""
    return [
        # Attach the timers the moment the model exists. Without this the
        # stage wrappers are defined but never installed, and every forward
        # stage silently reports zero while the run still looks successful.
        ("    model = CanonicalSpectraSiam(config).to(DEVICE)\n",
         "    model = CanonicalSpectraSiam(config).to(DEVICE)\n"
         "    d5_instrument(model)\n"),

        # Start the epoch and first-batch clocks.
        ("        total_loss = 0.0; seen = 0\n",
         "        total_loss = 0.0; seen = 0\n"
         "        globals()['D5_EPOCH_NOW'] = epoch  # predict() reads this\n"
         "        _d5_epoch_began = _time.perf_counter()\n"
         "        _d5_batch_began = _time.perf_counter()\n"),

        # Time waiting on the loader, then the host-to-device copy.
        ("            labels = labels.to(DEVICE)\n",
         "            _d5_current['h01_data_wait'] = (\n"
         "                _time.perf_counter() - _d5_batch_began) * 1000.0\n"
         "            _d5_batch_began = _time.perf_counter()\n"
         "            with d5_stage('h02_host_to_device'):\n"
         "                labels = labels.to(DEVICE)\n"
         "                left, right = left.to(DEVICE), right.to(DEVICE)\n"),

        # The copy already happened, so the model call must not repeat it.
        ("                logits, auxiliary = model(left.to(DEVICE), right.to(DEVICE), temperature)\n",
         "                logits, auxiliary = model(left, right, temperature)\n"),

        ("                loss, _ = canonical_spectra_loss(\n",
         "                _d5_loss = d5_stage('s11_loss'); _d5_loss.__enter__()\n"
         "                loss, _ = canonical_spectra_loss(\n"),

        ("            scaler.scale(loss / accumulation).backward()\n",
         "                _d5_loss.__exit__(None, None, None)\n"
         "            with d5_stage('s12_backward'):\n"
         "                scaler.scale(loss / accumulation).backward()\n"),

        ("                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)\n",
         "                with d5_stage('s13_optimizer_step'):\n"
         "                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)\n"),

        # Close the batch and restart the loader clock.
        ("            total_loss += float(loss.detach().cpu()) * len(labels); seen += len(labels)\n",
         "            total_loss += float(loss.detach().cpu()) * len(labels); seen += len(labels)\n"
         "            d5_flush(epoch, 'train', index, len(labels),\n"
         "                     (_time.perf_counter() - _d5_batch_began) * 1000.0)\n"
         "            _d5_batch_began = _time.perf_counter()\n"),

        # Time inference too. Without this, validation appears only as a
        # per-epoch total, and the paper cannot separate the cost of scoring a
        # pair from the cost of training on it.
        ('        for left, right, labels in tqdm(loader, desc="Evaluating", leave=False):\n'
         '            logits, _ = model(left.to(DEVICE), right.to(DEVICE), temperature)\n'
         '            labels_all.append(labels.numpy()); probabilities_all.append(torch.sigmoid(logits).cpu().numpy())\n',
         '        _d5_index = 0\n'
         '        _d5_began = _time.perf_counter()\n'
         '        for left, right, labels in tqdm(loader, desc="Evaluating", leave=False):\n'
         '            _d5_index += 1\n'
         '            _d5_current["h01_data_wait"] = (\n'
         '                _time.perf_counter() - _d5_began) * 1000.0\n'
         '            _d5_began = _time.perf_counter()\n'
         '            with d5_stage("h02_host_to_device"):\n'
         '                left, right = left.to(DEVICE), right.to(DEVICE)\n'
         '            logits, _ = model(left, right, temperature)\n'
         '            labels_all.append(labels.numpy()); probabilities_all.append(torch.sigmoid(logits).cpu().numpy())\n'
         '            if D5_TIME_INFERENCE:\n'
         '                d5_flush(D5_EPOCH_NOW, "valid", _d5_index, len(labels),\n'
         '                         (_time.perf_counter() - _d5_began) * 1000.0)\n'
         '            _d5_began = _time.perf_counter()\n'),

        # Split the epoch into its train and validation halves.
        ('        valid_labels, valid_probabilities = predict(model, loaders["valid"], temperature)\n',
         "        _d5_train_seconds = _time.perf_counter() - _d5_epoch_began\n"
         "        _d5_valid_began = _time.perf_counter()\n"
         '        valid_labels, valid_probabilities = predict(model, loaders["valid"], temperature)\n'
         "        _d5_valid_seconds = _time.perf_counter() - _d5_valid_began\n"
         "        D5_EPOCH_ROWS.append({\n"
         "            'Epoch': epoch,\n"
         "            'TrainSeconds': round(_d5_train_seconds, 4),\n"
         "            'ValidSeconds': round(_d5_valid_seconds, 4),\n"
         "            'EpochSeconds': round(_d5_train_seconds + _d5_valid_seconds, 4),\n"
         "            'TrainBatches': index,\n"
         "            'TrainPairs': seen,\n"
         "            'PairsPerSecond': round(seen / max(_d5_train_seconds, 1e-9), 2),\n"
         "            'MeanBatchMs': round(_d5_train_seconds * 1000.0 / max(index, 1), 4),\n"
         "            'TrainLoss': total_loss / max(seen, 1),\n"
         "        })\n"),
    ]


def patch_training(notebook: dict) -> None:
    """Insert per-batch and per-epoch timing into the training loop.

    Every replacement is asserted, so a change to the source notebook fails
    here rather than silently producing an un-instrumented run.
    """
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if "for epoch in range(1, EPOCHS + 1):" not in source:
            continue

        for old, new in training_edits():
            if old not in source:
                raise SystemExit(
                    "training-loop anchor not found; the source notebook "
                    f"changed:\n{old[:100]}")
            source = source.replace(old, new, 1)

        cell["source"] = source.splitlines(keepends=True)
        return
    raise SystemExit("training loop not found")


def build(source: Path, label: str, attachment: str,
          instrumentation: str, report: str) -> dict:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    header = {
        "cell_type": "markdown", "metadata": {},
        "source": (
            f"# Section 6.5: timing breakdown - {label}\n"
            "\n"
            "Reruns the canonical SPECTRA-Siam (lexical) configuration and "
            "records how long each stage of the methodology takes, per batch "
            "and per epoch.\n"
            "\n"
            "The model, data, optimizer, and seed are unchanged, so the "
            "accuracy reported here should match the RQ2 run for this "
            "benchmark; only measurement is added.\n"
            "\n"
            f"**Attach** `{attachment}`, select a T4 GPU, and Run All.\n"
            "\n"
            "Device stages are timed with CUDA events, because GPU work is "
            "asynchronous and a wall-clock timer would measure the kernel "
            "launch rather than the kernel itself.\n"
        ).splitlines(keepends=True),
    }

    # The instrumentation must exist after the model is defined and before the
    # training cell runs; the report comes last.
    model_index = next(
        index for index, cell in enumerate(cells)
        if "class CanonicalSpectraSiam" in "".join(cell.get("source", [])))

    cells = ([header]
             + cells[:model_index + 1]
             + [_cell(instrumentation.replace("__LABEL__", label))]
             + cells[model_index + 1:]
             + [_cell(report)])

    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook["cells"] = cells
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_ROOT / "kaggle" / "d5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    instrumentation = (_BLOCKS / "instrumentation.py").read_text(encoding="utf-8")
    report = (_BLOCKS / "report.py").read_text(encoding="utf-8")

    for folder, label, attachment in BENCHMARKS:
        source = _ROOT / "kaggle" / "rq2" / folder / "method" / "spectra_siam_lex.ipynb"
        if not source.is_file():
            raise SystemExit(f"missing source notebook: {source}")

        notebook = build(source, label, attachment, instrumentation, report)
        patch_training(notebook)

        # Fail here rather than 20 minutes into a Kaggle session.
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), f"{folder}#cell-{index}", "exec")

        path = args.output_dir / f"d5_timing_{folder}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
