"""One command: rebuild every clean dataset with corrected graphs and repackage it.

Run this after the DOT-attribution fix. For each benchmark it re-extracts the
graphs from ``data/DataSets.zip``, recomputes the spectral features, exports the
portable ``clean_data`` bundle, verifies the graphs actually belong to their
records, and replaces the Kaggle ZIP in ``outputs/``.

What the fix changed
--------------------
joern-export names each DOT after its *method*, never after the file the method
came from, so the previous index had to guess the owner from an ``m_<id>``
marker or from export order. Two things went wrong:

* C submissions had no marker and the C frontend emits ~5x more DOT files than
  source files, so the ordinal fallback attached graphs to the wrong records
  (source-size/AST-size rank correlation ~0.0 instead of ~0.9).
* A file defining several functions kept only one of them, which reduced ~20% of
  AtCoder Java submissions to an empty ``METHOD/BLOCK`` skeleton.

Extraction now asks the CPG itself which file each method came from, and merges
every user method of a file. Single-method snippet records (BigCloneBench and
the Java/C# halves of the other benchmarks) still resolve to exactly the method
the record is about, so their graphs are unchanged.

Usage
-----
From the repository root, with the project venv active::

    python scripts/rebuild_datasets_for_kaggle.py

Only the benchmarks whose graphs were wrong::

    python scripts/rebuild_datasets_for_kaggle.py --datasets atcoder_v3 gptclonebench_v3 semanticclonebench_v3

Resume after an interruption without redoing finished languages::

    python scripts/rebuild_datasets_for_kaggle.py --resume

Each dataset is independent: a failure in one leaves the others' fresh ZIPs in
place, and the previous ZIP of the failed dataset is restored from its backup.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

OUTPUTS_ROOT = PROJECT_ROOT.parent / "outputs"
DATA_ROOT = PROJECT_ROOT.parent / "data"
PUBLISH_DIR = OUTPUTS_ROOT / "kaggle_datasets"

ALL_DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")
# Only these three had mis-attributed graphs; codexglue_v3 is byte-identical
# before and after the fix because every BigCloneBench record is one method.
BROKEN_DATASETS = ("atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")


def _print_header(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


def _format_duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes}m{secs:02d}s"


def _check_prerequisites(archive: Path) -> list[str]:
    problems = []
    if not archive.is_file():
        problems.append(f"Source archive not found: {archive}")

    joern_home = Path(r"C:/joern-cli") if sys.platform == "win32" else None
    joern_names = ("joern-parse", "joern-export", "joern")
    resolved = {name: shutil.which(name) or shutil.which(f"{name}.bat") for name in joern_names}
    if joern_home and joern_home.is_dir():
        for name in joern_names:
            if not resolved[name]:
                candidate = joern_home / (f"{name}.bat" if sys.platform == "win32" else name)
                resolved[name] = str(candidate) if candidate.exists() else None
    missing = [name for name, path in resolved.items() if not path]
    if missing:
        problems.append(
            f"Joern executables not found: {missing}. Install Joern and set JOERN_HOME, "
            "or put joern-cli on PATH. The rebuild needs joern-parse, joern-export and joern "
            "(the last one resolves which source file each exported method belongs to)."
        )

    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_c_sharp  # noqa: F401
    except ModuleNotFoundError:
        problems.append(
            "C# graphs need tree-sitter and tree-sitter-c-sharp: python -m pip install -r requirements.txt"
        )
    return problems


def _run(command: list[str], description: str) -> int:
    print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = _format_duration(time.perf_counter() - started)
    status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
    print(f"[{status}] {description} ({elapsed})", flush=True)
    return result.returncode


BACKUP_DIR = OUTPUTS_ROOT / "_previous_archives"


def _backup_existing_zip(dataset: str) -> Path | None:
    """Park the canonical archive outside the tree the rebuild is about to wipe."""
    zip_path = PUBLISH_DIR / f"{dataset}_clean_data.zip"
    if not zip_path.is_file():
        zip_path = OUTPUTS_ROOT / dataset / f"{dataset}_clean_data.zip"
    if not zip_path.is_file():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / zip_path.name
    shutil.move(str(zip_path), str(backup))
    print(f"[backup] previous archive kept at {backup}")
    return backup


def _restore_backup(dataset: str, backup: Path | None) -> None:
    if backup and backup.is_file():
        restored = PUBLISH_DIR / f"{dataset}_clean_data.zip"
        restored.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(backup), str(restored))
        print(f"[restore] put the previous archive back: {restored}")


def rebuild_dataset(dataset: str, archive: Path, resume: bool) -> tuple[bool, str]:
    _print_header(f"{dataset}")
    backup = _backup_existing_zip(dataset)

    build_command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_kaggle_benchmark_datasets.py"),
        "--datasets", dataset, "--archive", str(archive),
    ]
    if not resume:
        build_command.append("--force")
    if _run(build_command, f"extract graphs and spectra for {dataset}") != 0:
        _restore_backup(dataset, backup)
        return False, "extraction failed"

    if _run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "check_graph_health.py"), "--datasets", dataset],
        f"graph health check for {dataset}",
    ) != 0:
        _restore_backup(dataset, backup)
        return False, "graph health check failed - graphs do not match their records"

    zip_path = OUTPUTS_ROOT / dataset / f"{dataset}_clean_data.zip"
    if not zip_path.is_file():
        _restore_backup(dataset, backup)
        return False, "build reported success but produced no ZIP"

    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    target = PUBLISH_DIR / zip_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(zip_path), str(target))
    size_mb = target.stat().st_size / 1e6
    print(f"[+] centralized at {target} ({size_mb:.1f} MB)")

    if backup and backup.is_file():
        backup.unlink()
        print("[cleanup] removed the superseded previous archive")
    return True, f"{size_mb:.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument("--only-broken", action="store_true", help=f"Shorthand for --datasets {' '.join(BROKEN_DATASETS)}")
    parser.add_argument("--archive", type=Path, default=DATA_ROOT / "DataSets.zip")
    parser.add_argument("--resume", action="store_true", help="Keep finished languages and existing bundles instead of forcing a fresh build.")
    parser.add_argument("--skip-checks", action="store_true", help="Do not verify Joern and parser prerequisites first.")
    args = parser.parse_args()

    datasets = list(BROKEN_DATASETS) if args.only_broken else args.datasets
    archive = args.archive.resolve()

    _print_header("Rebuilding clean datasets with corrected graph attribution")
    print(f"archive : {archive}")
    print(f"outputs : {OUTPUTS_ROOT}")
    print(f"publish : {PUBLISH_DIR}")
    print(f"datasets: {', '.join(datasets)}")
    if "codexglue_v3" in datasets:
        print(
            "\nnote: codexglue_v3 (BigCloneBench) graphs are unchanged by this fix, because every\n"
            "      record is a single method. Rebuilding it is optional - pass --only-broken to\n"
            "      skip it and save roughly an hour."
        )

    if not args.skip_checks:
        problems = _check_prerequisites(archive)
        if problems:
            print("\n[-] Cannot start:")
            for problem in problems:
                print(f"    - {problem}")
            return 1
        print("\n[+] Prerequisites satisfied (Joern, tree-sitter, source archive).")

    started = time.perf_counter()
    results: dict[str, tuple[bool, str]] = {}
    for dataset in datasets:
        try:
            results[dataset] = rebuild_dataset(dataset, archive, args.resume)
        except KeyboardInterrupt:
            print("\n[!] Interrupted. Re-run with --resume to continue where this stopped.")
            return 130

    _print_header(f"Summary ({_format_duration(time.perf_counter() - started)})")
    for dataset in datasets:
        ok, detail = results.get(dataset, (False, "not attempted"))
        print(f"  {'OK  ' if ok else 'FAIL'}  {dataset:24s} {detail}")

    failed = [name for name, (ok, _) in results.items() if not ok]
    if failed:
        print(f"\n[-] {len(failed)} dataset(s) failed; their previous archives were restored.")
        return 1

    print(f"\n[+] Upload these to Kaggle:")
    for dataset in datasets:
        print(f"    {PUBLISH_DIR / f'{dataset}_clean_data.zip'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
