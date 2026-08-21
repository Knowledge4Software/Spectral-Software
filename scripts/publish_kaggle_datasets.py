"""Centralize validated clean-data archives in ``outputs/kaggle_datasets``.

The build writes each archive next to its dataset output; the Kaggle notebooks
attach them from one shared folder. This refuses to publish an archive whose
graphs failed the health check, so a mis-attributed extraction cannot silently
reach an experiment again.

Usage::

    python scripts/publish_kaggle_datasets.py
    python scripts/publish_kaggle_datasets.py --datasets atcoder_v3 --skip-health-check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.clean_data_export import create_clean_data_zip
OUTPUTS_ROOT = PROJECT_ROOT.parent / "outputs"
PUBLISH_DIR = OUTPUTS_ROOT / "kaggle_datasets"
DATASETS = (
    "codexglue_v3",
    "atcoder_v3",
    "gptclonebench_v3",
    "semanticclonebench_v3",
    "codenet_4l",
)
OUTPUT_DIRS = {
    "codenet_4l": "codenet_4l_nonclone_12k",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_dir(outputs_root: Path, dataset: str) -> Path:
    return outputs_root / OUTPUT_DIRS.get(dataset, dataset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--skip-health-check", action="store_true")
    args = parser.parse_args()

    publish_dir = args.outputs_root / "kaggle_datasets"
    publish_dir.mkdir(parents=True, exist_ok=True)

    available = [
        name
        for name in args.datasets
        if (_build_dir(args.outputs_root, name) / f"{name}_clean_data.zip").is_file()
        or (publish_dir / f"{name}_clean_data.zip").is_file()
    ]
    missing = sorted(set(args.datasets) - set(available))
    if missing:
        print(f"[!] Not built yet, skipping: {missing}")
    if not available:
        print("[-] Nothing to publish.")
        return 1

    new_standard_builds = [
        name
        for name in available
        if name != "codenet_4l" and (_build_dir(args.outputs_root, name) / f"{name}_clean_data.zip").is_file()
    ]
    if not args.skip_health_check and new_standard_builds:
        print("[*] Verifying graph health before publishing...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "check_graph_health.py"), "--datasets", *new_standard_builds],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print("[-] Refusing to publish: graph health check failed. Re-extract before publishing.")
            return 1

    for name in available:
        build_dir = _build_dir(args.outputs_root, name)
        source = build_dir / f"{name}_clean_data.zip"
        clean_dir = build_dir / "clean_data"
        target = publish_dir / f"{name}_clean_data.zip"

        if not source.is_file():
            print(f"[=] already centralized: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
            continue

        # A ZIP older than the clean_data it claims to package means the build
        # stopped before repackaging, and publishing it would ship the previous
        # extraction under the new name. Rebuild it from what is actually on disk.
        newest_source = max(
            (path.stat().st_mtime for path in clean_dir.iterdir() if path.is_file()),
            default=0.0,
        )
        if newest_source > source.stat().st_mtime:
            print(f"[!] {name}: ZIP is older than its clean_data; repackaging before publishing.")
            create_clean_data_zip(clean_dir, source)

        if target.is_file() and _sha256(source) == _sha256(target):
            source.unlink()
            print(f"[+] removed duplicate build ZIP; canonical copy is {target.name}")
            continue
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
        print(f"[+] centralized {target.name} ({target.stat().st_size / 1e6:.1f} MB)")

    print(f"\n[+] {len(available)} archive(s) in {publish_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
