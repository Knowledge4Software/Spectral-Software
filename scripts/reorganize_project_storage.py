"""Reorganise ``data/`` and ``outputs/`` into one predictable layout.

Three problems this fixes:

1. **Published datasets live under ``outputs/``.** A dataset that was uploaded
   to Kaggle is an input to every later experiment, not an output of one, so it
   belongs in ``data/``.

2. **Every clean-data export is stored twice**, once as ``clean_data/`` and
   once as the ``*_clean_data.zip`` built from it. The ZIP is what gets
   attached on Kaggle and what the local runners read, so the unpacked copy is
   redundant.

3. **Build scratch outlives the build.** Joern batch directories and per-record
   graph caches are only useful for resuming an interrupted extraction; once
   the dataset exists they are dead weight.

Nothing is deleted that cannot be rebuilt from a surviving artifact, and every
removal is reported with its reason. Run without ``--apply`` first: the default
only prints the plan.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SPECTRALS = _ROOT.parent
DATA = SPECTRALS / "data"
OUTPUTS = SPECTRALS / "outputs"


@dataclass
class Move:
    source: Path
    destination: Path
    reason: str


@dataclass
class Remove:
    target: Path
    reason: str
    keeps: str = ""          # what still holds this content


@dataclass
class Plan:
    moves: list[Move] = field(default_factory=list)
    removes: list[Remove] = field(default_factory=list)


# Clean-data exports: the ZIP is authoritative, the unpacked folder is not.
CLEAN_DATA_DATASETS = (
    "atcoder_v3",
    "codexglue_v3",
    "gptclonebench_v3",
    "semanticclonebench_v3",
    "codenet_4l_clone50k_diff50k",
    "codenet_4l_nonclone_12k",
)

# Extraction scratch. Each entry names the artifact that makes it redundant.
SCRATCH = {
    "codenet_4l_clone_50k":
        "Joern batch scratch only; holds no clean_data of its own",
    "codenet_4l_all_clones":
        "batch scratch and graph cache only; holds no clean_data of its own",
    "codenet_4l_distributed":
        "per-laptop extraction caches from the distributed build",
    "graph_record_cache_laptop2":
        "per-record graph cache from the same distributed build",
    "codenet_4l_clone50k_diff50k_nonclone_java_cpp_cache":
        "non-clone pairing cache",
    "codenet_4l_clone50k_diff50k_nonclone_java_cpp_work":
        "non-clone pairing scratch",
    "codenet_4l_clone50k_diff50k_nonclone_py_csharp_cache":
        "non-clone pairing cache",
    "codenet_4l_clone50k_diff50k_nonclone_py_csharp_work":
        "non-clone pairing scratch",
}

# Per-language Joern method-map logs left beside each export. They are
# extraction diagnostics, not results: the clean-data ZIP carries the graphs
# they describe. Small, so they are only removed with --include-stale.
METHOD_MAP_LOGS = (
    "atcoder_v3", "codexglue_v3", "gptclonebench_v3",
    "semanticclonebench_v3", "codenet_4l_nonclone_12k",
)

# Superseded result exports. Small, so removal is optional (--include-stale).
STALE = {
    "canonical_experiments": "August preflight, superseded by the V3 runs",
    "visualizations": "HTML previews of the non-clone scope study",
    "paper_tables_current": "superseded by kaggle/latex/",
    "rq1_table": "superseded by kaggle/latex/rq1/",
}


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(count: int) -> str:
    return f"{count / 2**30:6.2f} GB" if count >= 2**30 else f"{count / 2**20:6.1f} MB"


def build_plan(include_stale: bool) -> Plan:
    plan = Plan()

    # 1. Published Kaggle datasets are inputs; move them beside the other data.
    published = OUTPUTS / "kaggle_datasets"
    if published.is_dir():
        for archive in sorted(published.glob("*.zip")):
            plan.moves.append(Move(
                archive, DATA / "kaggle_datasets" / archive.name,
                "published Kaggle dataset: an input to every experiment"))

    # 2. Clean-data exports: keep the ZIP, drop the unpacked duplicate.
    for name in CLEAN_DATA_DATASETS:
        folder = OUTPUTS / name
        unpacked = folder / "clean_data"
        if not unpacked.is_dir():
            continue
        local_zip = next(folder.glob("*_clean_data.zip"), None)
        published_zip = published / f"{name}_clean_data.zip"
        archive = local_zip or (published_zip if published_zip.is_file() else None)
        if archive is None:
            # No ZIP anywhere: this copy is the only one, so keep it and say so.
            plan.moves.append(Move(
                unpacked, DATA / "clean_data" / name,
                "the only copy of this export; no ZIP exists, so it is kept"))
            continue
        if local_zip is not None:
            plan.moves.append(Move(
                local_zip, DATA / "clean_data" / local_zip.name,
                "clean-data export, the form Kaggle and the local runners read"))
        plan.removes.append(Remove(
            unpacked, "unpacked duplicate of the clean-data ZIP",
            keeps=archive.name))

    # 3. Extraction scratch.
    for name, reason in SCRATCH.items():
        target = OUTPUTS / name
        if target.exists():
            plan.removes.append(Remove(target, reason))

    if include_stale:
        for name in METHOD_MAP_LOGS:
            target = OUTPUTS / name
            if target.is_dir():
                plan.removes.append(Remove(
                    target, "Joern method-map logs from graph extraction",
                    keeps=f"{name} clean-data ZIP"))
        for name, reason in STALE.items():
            target = OUTPUTS / name
            if target.exists():
                plan.removes.append(Remove(target, reason))

    return plan


def report(plan: Plan) -> tuple[int, int]:
    moved = freed = 0
    if plan.moves:
        print("MOVE  (outputs -> data: these are inputs, not results)")
        for move in plan.moves:
            count = size_of(move.source)
            moved += count
            print(f"  {human(count)}  {move.source.relative_to(SPECTRALS)}")
            print(f"             -> {move.destination.relative_to(SPECTRALS)}")
            print(f"             {move.reason}")
    if plan.removes:
        print("\nREMOVE")
        for remove in plan.removes:
            count = size_of(remove.target)
            freed += count
            print(f"  {human(count)}  {remove.target.relative_to(SPECTRALS)}")
            print(f"             {remove.reason}")
            if remove.keeps:
                print(f"             kept instead: {remove.keeps}")
    return moved, freed


def apply(plan: Plan) -> None:
    for move in plan.moves:
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        if move.destination.exists():
            print(f"  skip (exists): {move.destination.relative_to(SPECTRALS)}")
            continue
        shutil.move(str(move.source), str(move.destination))
        print(f"  moved {move.destination.relative_to(SPECTRALS)}")
    for remove in plan.removes:
        if remove.target.is_dir():
            shutil.rmtree(remove.target, ignore_errors=True)
        elif remove.target.exists():
            remove.target.unlink()
        print(f"  removed {remove.target.relative_to(SPECTRALS)}")

    # Drop directories left empty by the moves.
    for folder in sorted(OUTPUTS.iterdir()) if OUTPUTS.is_dir() else []:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            print(f"  removed empty {folder.relative_to(SPECTRALS)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="perform the plan (default: print it only)")
    parser.add_argument("--include-stale", action="store_true",
                        help="also remove superseded result exports")
    args = parser.parse_args()

    plan = build_plan(args.include_stale)
    moved, freed = report(plan)

    print(f"\n{human(moved)} would move, {human(freed)} would be freed"
          if not args.apply else f"\n{human(moved)} moved, {human(freed)} freed")
    if not args.apply:
        print("\nre-run with --apply to perform this plan")
        return
    print()
    apply(plan)


if __name__ == "__main__":
    main()
