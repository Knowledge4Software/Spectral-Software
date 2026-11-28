import ast
import json
import os
from pathlib import Path
import sys
import time

from scripts.run_revision_server import (ServerPaths, make_catalog, paired_inputs_ready,
                                         selected_jobs, watch_worker)


def test_inventory_spans_all_seeds_and_preserves_transfer_budget():
    catalog = make_catalog()
    assert len(selected_jobs(catalog, "remaining", [])) == 7
    focused = selected_jobs(catalog, "focused", [])
    assert len(focused) == 45
    lease = selected_jobs(catalog, "lease", [])
    assert len(lease) == 52 and len({j["id"] for j in lease}) == 52
    assert sum(j["group"] == "focused" for j in lease[:8]) == 4
    assert all(j["gpu"] for j in lease)
    assert {j["id"].rsplit("seed", 1)[-1] for j in focused} == {"42", "123", "231"}
    for dataset in ("codenet", "atcoder", "bigclonebench"):
        for seed in (42, 123, 231):
            for arm in ("full", "mean_max", "classification_only"):
                assert catalog[f"{dataset}_{arm}_seed{seed}"]["dataset"] == dataset
    assert catalog["qwen3_frozen_probe_seed231"]["dependency"] == "qwen3_frozen_encoder_and_probe"


def test_legacy_paths_are_mapped_without_changing_embedded_code():
    embedded = 'from pathlib import Path\nroot = Path("/kaggle/working")\n'
    source = f"_sources = {{'jobs.py': {embedded!r}}}\n"
    source += 'path = Path(f"/kaggle/working/run_{42}")\nresult = run_external()'
    tree = ServerPaths(Path("/tmp/job"), Path("/tmp/input")).visit(ast.parse(source))
    ns = {"Path": Path, "run_external": lambda **kwargs: kwargs}
    exec(compile(ast.fix_missing_locations(tree), "server_test", "exec"), ns)
    assert ns["_sources"]["jobs.py"] == embedded
    assert ns["path"] == Path("/tmp/job/run_42")
    assert Path(ns["result"]["output_root"]) == Path("/tmp/job")


def test_worker_cutoff_returns_without_erasing_checkpoint(tmp_path):
    marker = tmp_path / "progress.pt"
    marker.write_bytes(b"saved")
    with (tmp_path / "log").open("wb") as log:
        rc, reason = watch_worker(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            work=tmp_path, stream=log, env=os.environ.copy(),
            deadline=time.monotonic() + .2, idle_seconds=0,
        )
    assert rc != 0 and reason == "lease_cutoff"
    assert marker.read_bytes() == b"saved"


def test_cli_passes_fresh_jobs_and_catalog_to_runner(monkeypatch):
    from scripts import run_revision_server as runner

    monkeypatch.setattr(sys, "argv", ["runner", "run", "--profile", "fresh", "--lease-hours", "3"])
    def run(jobs, args, catalog):
        assert args.profile == "fresh"
        assert len(jobs) == 8
        assert sum(job["group"] == "focused" for job in jobs) == 5
        assert sum(job["group"] == "loss_experiment" for job in jobs) == 3
        assert all(job["id"] in catalog for job in jobs)
        return 0
    monkeypatch.setattr(runner, "run_queue", run)
    assert runner.cli() == 0


def test_paired_analysis_waits_for_both_methods_and_three_seeds(tmp_path):
    for method in ("full", "mean_max"):
        for seed in (42, 123, 231):
            run = tmp_path / f"{method}_{seed}" / "work"
            run.mkdir(parents=True)
            (run / "final_test.json").write_text(json.dumps({"arm": method, "seed": seed}))
            predictions = run / "test_predictions.csv.gz"
            predictions.touch()
    assert paired_inputs_ready(tmp_path, ("full", "mean_max"))
    predictions.unlink()
    assert not paired_inputs_ready(tmp_path, ("full", "mean_max"))
