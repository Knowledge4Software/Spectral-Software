"""Give ``data/`` names that say what each file is, and drop what is contained.

Two problems this fixes.

**Names that mislead.** ``codenet_4l_clean_data.zip`` reads like the CodeNet
dataset, but it holds the 12,000-pair subset that no reported run uses; the
export the paper actually trains on is
``codenet_4l_clone50k_diff50k_clean_data.zip``. Two files whose names differ by
a suffix are 12k and 100k versions of different things, and one
``codenet_4l_clone50k_diff50k.zip`` is not an export at all but the parquet
source it was built from.

**Copies of copies.** Every ``*_prepared/`` corpus is a subset of
``codenet_4l_all_clones_prepared`` (verified by code id: 100% containment), and
a clean-data export already carries its own source code, pairs, splits, and
sampling seeds. So a subset's prepared corpus adds nothing that its export does
not already have.

The layout this produces::

    data/
      benchmarks/          one export per benchmark, named for what it holds
        atcoder.zip
        bigclonebench.zip
        codenet.zip                  <- the 100k export the paper uses
        codenet-12k.zip              <- the small subset, kept but labelled
        gptclonebench.zip
        semanticclonebench.zip
      sources/             what the exports were built from
        codenet-programs.parquet.zip
        codenet_4l_all_clones_prepared/
        v3_prepared/
        base datasets.zip
        codenet dataset.zip

Run without ``--apply`` first: the default only prints the plan.
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DATA = _ROOT.parent / "data"

# old path -> (new path, why the new name is clearer)
RENAMES = {
    "kaggle_datasets/atcoder_v3_clean_data.zip": (
        "benchmarks/atcoder.zip", "the AtCoder export"),
    "kaggle_datasets/codexglue_v3_clean_data.zip": (
        "benchmarks/bigclonebench.zip",
        "the paper calls this BigCloneBench, not CodeXGLUE"),
    "kaggle_datasets/gptclonebench_v3_clean_data.zip": (
        "benchmarks/gptclonebench.zip", "the GPTCloneBench export"),
    "kaggle_datasets/semanticclonebench_v3_clean_data.zip": (
        "benchmarks/semanticclonebench.zip", "the SemanticCloneBench export"),
    "clean_data/codenet_4l_clone50k_diff50k_clean_data.zip": (
        "benchmarks/codenet.zip",
        "the 100k CodeNet export every reported CodeNet run uses"),
    "kaggle_datasets/codenet_4l_clean_data.zip": (
        "benchmarks/codenet-12k.zip",
        "a 12k subset, not the CodeNet dataset its old name implied"),
    "kaggle_datasets/codenet_4l_clone50k_diff50k.zip": (
        "sources/codenet-programs.parquet.zip",
        "parquet source, not a graph export"),
    "codenet_4l_all_clones_prepared": (
        "sources/codenet_4l_all_clones_prepared",
        "the full corpus every CodeNet subset is drawn from"),
    "v3_prepared": ("sources/v3_prepared", "the corpora behind the V3 exports"),
    "base datasets.zip": ("sources/base datasets.zip", "upstream download"),
    "codenet dataset.zip": ("sources/codenet dataset.zip", "upstream download"),
}

# Prepared corpora whose content survives inside an export. Each maps to the
# export that makes it redundant.
CONTAINED = {
    "codenet_4l_clone50k_diff50k_prepared": "benchmarks/codenet.zip",
    "codenet_4l_nonclone_12k_prepared": "benchmarks/codenet-12k.zip",
}

# Subsets with no export of their own and no run that uses them. They are a
# strict subset of all_clones, which is kept.
UNUSED = {
    "codenet_4l_clone_50k_prepared":
        "a 50k subset of codenet_4l_all_clones_prepared with no export and no "
        "reported run",
    "clean_data/codenet_4l_nonclone_12k":
        "unpacked copy of the 12k export, byte for byte identical to its ZIP",
}


@dataclass
class Plan:
    moves: list[tuple[Path, Path, str]] = field(default_factory=list)
    removes: list[tuple[Path, str, str]] = field(default_factory=list)


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(count: int) -> str:
    return f"{count / 2**30:6.2f} GB" if count >= 2**30 else f"{count / 2**20:6.1f} MB"


def build_plan(drop_contained: bool) -> Plan:
    plan = Plan()
    for old, (new, reason) in RENAMES.items():
        source = DATA / old
        if source.exists():
            plan.moves.append((source, DATA / new, reason))
    for name, held_by in CONTAINED.items():
        target = DATA / name
        if target.exists() and drop_contained:
            plan.removes.append((
                target,
                "source code, pairs, splits, and sampling seeds all live in the "
                "export",
                held_by))
    for name, reason in UNUSED.items():
        target = DATA / name
        if target.exists():
            plan.removes.append((target, reason, ""))
    return plan


def report(plan: Plan) -> tuple[int, int]:
    moved = freed = 0
    if plan.moves:
        print("RENAME")
        for source, destination, reason in plan.moves:
            moved += size_of(source)
            print(f"  {source.relative_to(DATA)}")
            print(f"    -> {destination.relative_to(DATA)}")
            print(f"       {reason}")
    if plan.removes:
        print("\nREMOVE")
        for target, reason, held_by in plan.removes:
            count = size_of(target)
            freed += count
            print(f"  {human(count)}  {target.relative_to(DATA)}")
            print(f"             {reason}")
            if held_by:
                print(f"             content kept in: {held_by}")
    return moved, freed


def apply(plan: Plan) -> None:
    for source, destination, _ in plan.moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"  skip (exists): {destination.relative_to(DATA)}")
            continue
        shutil.move(str(source), str(destination))
        print(f"  {destination.relative_to(DATA)}")
    for target, _, _ in plan.removes:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
        print(f"  removed {target.relative_to(DATA)}")
    for folder in ("kaggle_datasets", "clean_data"):
        path = DATA / folder
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            print(f"  removed empty {folder}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="perform the plan (default: print it only)")
    parser.add_argument("--keep-contained", action="store_true",
                        help="keep prepared corpora whose content is in an export")
    args = parser.parse_args()

    plan = build_plan(drop_contained=not args.keep_contained)
    moved, freed = report(plan)
    print(f"\n{human(moved)} renamed, {human(freed)} "
          f"{'freed' if args.apply else 'would be freed'}")
    if not args.apply:
        print("\nre-run with --apply to perform this plan")
        return
    print()
    apply(plan)


if __name__ == "__main__":
    main()
