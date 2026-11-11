"""Exercise the Windows cleaner against isolated repositories, never real results."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell")
pytestmark = pytest.mark.skipif(os.name != "nt" or POWERSHELL is None, reason="Windows PowerShell cleanup")


def _workspace(tmp_path):
    repo = tmp_path / "workspace"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/clean_workspace.ps1", repo / "scripts/clean_workspace.ps1")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    files = {
        ".pytest_cache/entry": b"cache",
        "spectral_code/__pycache__/discard.pyc": b"cache",
        "tests/__pycache__/tracked.pyc": b"tracked cache: preserve",
        "output/revision/test_restart_1109/progress.pt": b"disposable CPU test checkpoint",
        "output/revision/validation/real_experiment/progress.pt": b"research checkpoint: preserve",
        "output/revision/test_real_evaluation/progress.pt": b"unknown test output: preserve",
        "output/revision/archive/releases/frozen.zip": b"immutable release: preserve",
        ".venv/lib/__pycache__/keep.pyc": b"environment: preserve",
        "spectral_code/.venv/lib/__pycache__/keep.pyc": b"nested environment: preserve",
        "revision/current.ipynb": b"notebook: preserve",
        "Spectral CCD Twocolumn.pdf": b"manuscript: preserve",
        ".env": b"local configuration: preserve",
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(["git", "-C", str(repo), "add", "tests/__pycache__/tracked.pyc"], check=True)
    return repo, files


def _run(repo, *args):
    result = subprocess.run([POWERSHELL, "-NoProfile", "-File", str(repo / "scripts/clean_workspace.ps1"), *args],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.lstrip("\ufeff"))


def test_dry_run_and_apply_preserve_research_tracked_files_and_environment(tmp_path):
    repo, files = _workspace(tmp_path)
    preview = _run(repo, "-DryRun")
    assert preview["dry_run"] and preview["total_files"] == 3
    assert all((repo / name).read_bytes() == content for name, content in files.items())
    assert not (repo / "output/revision/maintenance").exists()
    actual = _run(repo)
    assert not actual["dry_run"] and actual["total_files"] == preview["total_files"]
    disposable = {".pytest_cache/entry", "spectral_code/__pycache__/discard.pyc",
                  "output/revision/test_restart_1109/progress.pt"}
    for name, content in files.items():
        if name in disposable:assert not (repo / name).exists()
        else:assert (repo / name).read_bytes() == content
    assert any("tracked files" in entry["reason"] for entry in actual["skipped"])
    assert json.loads((repo / "output/revision/maintenance/cleanup-last.json").read_text(encoding="utf-8-sig"))["schema"] == "spectra_workspace_cleanup_v2"


def test_report_escape_is_rejected_before_any_deletion(tmp_path):
    repo, files = _workspace(tmp_path)
    result = subprocess.run([POWERSHELL, "-NoProfile", "-File", str(repo / "scripts/clean_workspace.ps1"),
                             "-ReportPath", "../outside.json"], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "escapes workspace" in result.stderr
    assert all((repo / name).read_bytes() == content for name, content in files.items())
    assert not (tmp_path / "outside.json").exists()


def test_junction_targets_and_fixture_trees_containing_links_are_preserved(tmp_path):
    repo, _ = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir(); (outside / "keep.txt").write_text("must survive")
    cache_link = repo / "revision/__pycache__"
    fixture_link = repo / "output/revision/test_restart_1109/outside_link"
    created = []
    try:
        for link in (cache_link, fixture_link):
            command = f"New-Item -ItemType Junction -Path '{link}' -Target '{outside}' | Out-Null"
            result = subprocess.run([POWERSHELL, "-NoProfile", "-Command", command], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            created.append(link)
        report = _run(repo)
        assert (outside / "keep.txt").read_text() == "must survive"
        assert cache_link.exists() and fixture_link.exists()
        assert (repo / "output/revision/test_restart_1109/progress.pt").exists()
        assert any("link" in entry["reason"] for entry in report["skipped"])
    finally:
        for link in reversed(created):
            # Non-recursive RemoveDirectory removes only the junction itself.
            os.rmdir(link)
