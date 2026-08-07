"""Copy validated V3 clean-data archives into ``outputs/kaggle_datasets``.

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
DATASETS = ("codexglue_v3", "atcoder_v3", "gptclonebench_v3", "semanticclonebench_v3")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--skip-health-check", action="store_true")
    args = parser.parse_args()

    publish_dir = args.outputs_root / "kaggle_datasets"
    publish_dir.mkdir(parents=True, exist_ok=True)

    available = [name for name in args.datasets if (args.outputs_root / name / f"{name}_clean_data.zip").is_file()]
    missing = sorted(set(args.datasets) - set(available))
    if missing:
        print(f"[!] Not built yet, skipping: {missing}")
    if not available:
        print("[-] Nothing to publish.")
        return 1

    if not args.skip_health_check:
        print("[*] Verifying graph health before publishing...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "check_graph_health.py"), "--datasets", *available],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print("[-] Refusing to publish: graph health check failed. Re-extract before publishing.")
            return 1

    for name in available:
        source = args.outputs_root / name / f"{name}_clean_data.zip"
        clean_dir = args.outputs_root / name / "clean_data"

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

        target = publish_dir / f"{name}_clean_data.zip"
        shutil.copy2(source, target)
        print(f"[+] published {target.name} ({target.stat().st_size / 1e6:.1f} MB)")

    print(f"\n[+] {len(available)} archive(s) in {publish_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
