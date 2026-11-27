import json
from pathlib import Path
import subprocess
import sys


def test_incremental_publisher_sends_only_validation_fields(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    output = tmp_path / "output"
    run = output / "seed42" / "revision_run"
    run.mkdir(parents=True)
    status = run / "status.json"
    status.write_text(json.dumps({"status": "training", "epoch": 2,
        "best_valid": {"F1": .8}, "hidden": "PRIVATE", "checkpoint": "/private/progress.pt"}))
    (run / "progress.pt").write_text("DO NOT PUBLISH")
    script = Path(__file__).resolve().parents[1] / "scripts" / "publish_server_results.py"
    command = [sys.executable, str(script), str(output), "--work", str(tmp_path / "work"),
               "--remote", str(remote), "--run-id", "test-lease", "--once"]
    subprocess.run(command, check=True)
    report = subprocess.check_output(
        ["git", "-C", str(remote), "show", "gpu-results/test-lease:validation-progress.json"], text=True)
    assert "PRIVATE" not in report and "progress.pt" not in report
    assert json.loads(report)["seed42"]["best_validation"]["F1"] == .8
    status.write_text(json.dumps({"status": "complete", "complete": True, "epoch": 3}))
    subprocess.run(command, check=True)
    subprocess.run(command, check=True)
    commits = subprocess.check_output(
        ["git", "-C", str(remote), "rev-list", "--count", "gpu-results/test-lease"], text=True)
    assert int(commits) == 2
