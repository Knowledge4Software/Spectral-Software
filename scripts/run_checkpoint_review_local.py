"""Run the pinned CodeNet validation review on an existing local CUDA setup."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from spectral_code.revision.checkpoint_review import run_checkpoint_review
from spectral_code.revision.progress import install_progress

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(42, 123, 231), default=[123, 231])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if not torch.cuda.is_available():
        parser.error("Use the project CUDA environment; local GPU inference requires CUDA")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    seeds = sorted(set(args.seeds))
    manifest = json.loads((ROOT / "revision/queues/checkpoint_review_manifest.json").read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e["dataset"] == "codenet" and e["seed"] in seeds]
    if len(entries) != 6 * len(seeds):
        raise ValueError("Expected all six prespecified conditions for every requested seed")
    output = args.output or ROOT.parent / "outputs/revesion/05" / ("local_codenet_f1_seeds" + "_".join(map(str, seeds)))
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                          seeds=seeds, checkpoints=len(entries), output=str(output), training_updates=0)), flush=True)
    install_progress()
    result = run_checkpoint_review(entries,
        ROOT.parent / "data/working/revision/codenet_revision_common_support", output,
        input_root=ROOT.parent / "outputs/revesion", started=started)
    print(json.dumps({k: v for k, v in result.items() if k != "completed"}, indent=2), flush=True)
    if result["status"] != "complete":
        raise SystemExit("Analysis is incomplete; inspect the saved status and small reports")


if __name__ == "__main__":
    main()
