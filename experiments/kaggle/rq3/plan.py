"""Split the RQ3 matrix into Kaggle-sized notebook shards.

Every shard is a group of configurations for one model that should finish
inside one Kaggle session. Shard sizes come from measured RQ2 CodeNet
runtimes scaled by training-pair count, then bin-packed under a wall-clock
budget, so a slow model gets more shards than a fast one for the same work.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.kaggle.rq3.matrix import Configuration, all_configurations

# Minutes per training pair, from the finished RQ2 CodeNet runs
# (69,779 train pairs, 4 epochs, Kaggle T4).
MEASURED_MINUTES_AT_69779 = {
    "SPECTRA-Siam": 16.1,
    "ASTNN": 61.4,
    "RtvNN": 61.2,
    "DeepSim": 18.5,
}
REFERENCE_TRAIN_PAIRS = 69_779
MODELS = tuple(MEASURED_MINUTES_AT_69779)

# Kaggle allows ~12 h; budget well under it so a slow GPU or a queue delay
# cannot strand a shard mid-run.
DEFAULT_BUDGET_HOURS = 3.5


def minutes_per_pair(model: str) -> float:
    return MEASURED_MINUTES_AT_69779[model] / REFERENCE_TRAIN_PAIRS


@dataclass
class Shard:
    model: str
    index: int
    configurations: list[Configuration]
    estimated_hours: float

    @property
    def slug(self) -> str:
        model = self.model.lower().replace("-", "_")
        return f"rq3_{model}_{self.index:02d}"

    @property
    def tables(self) -> list[str]:
        return sorted({configuration.table for configuration in self.configurations})


def _group_key(configuration: Configuration) -> tuple[str, str, str]:
    """Group by table, target, and reinforcement.

    Table 10's 96 configurations are far too large for one session, and this
    granularity is the coarsest one whose biggest group still fits the budget.
    """
    return (configuration.table, configuration.target, configuration.reinforcement)


def build_shards(
    model: str,
    train_sizes: dict[str, int],
    *,
    budget_hours: float = DEFAULT_BUDGET_HOURS,
    max_train: int | None = None,
) -> list[Shard]:
    """Bin-pack this model's configurations into shards under the budget."""
    rate = minutes_per_pair(model)
    groups: dict[tuple[str, str, str], list[Configuration]] = {}
    for configuration in all_configurations():
        groups.setdefault(_group_key(configuration), []).append(configuration)

    def group_hours(members: list[Configuration]) -> float:
        pairs = sum(
            min(train_sizes[member.key], max_train) if max_train else train_sizes[member.key]
            for member in members
        )
        return rate * pairs / 60.0

    # Largest-first so big groups get their own shard before small ones fill gaps.
    ordered = sorted(groups.values(), key=group_hours, reverse=True)
    bins: list[tuple[float, list[Configuration]]] = []
    for members in ordered:
        hours = group_hours(members)
        for index, (used, bucket) in enumerate(bins):
            if used + hours <= budget_hours:
                bins[index] = (used + hours, bucket + members)
                break
        else:
            bins.append((hours, list(members)))

    shards = []
    for index, (hours, members) in enumerate(bins, start=1):
        members = sorted(members, key=lambda item: (item.table, item.target, item.name))
        shards.append(Shard(model=model, index=index, configurations=members, estimated_hours=hours))
    return shards


def load_train_sizes(clean_data: Path) -> dict[str, int]:
    """Training-pair count per configuration, used only for time estimates."""
    from experiments.kaggle.rq3.runner import load_pairs, split_frames

    pairs = load_pairs(clean_data)
    return {
        configuration.key: len(split_frames(pairs, configuration)["train"])
        for configuration in all_configurations()
    }


def build_all(
    clean_data: Path, *, budget_hours: float = DEFAULT_BUDGET_HOURS, max_train: int | None = None
) -> dict[str, list[Shard]]:
    train_sizes = load_train_sizes(clean_data)
    return {
        model: build_shards(model, train_sizes, budget_hours=budget_hours, max_train=max_train)
        for model in MODELS
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", type=Path, required=True)
    parser.add_argument("--budget-hours", type=float, default=DEFAULT_BUDGET_HOURS)
    parser.add_argument("--max-train", type=int, default=None)
    args = parser.parse_args()

    total_shards = total_hours = 0
    for model, shards in build_all(
        args.clean_data, budget_hours=args.budget_hours, max_train=args.max_train
    ).items():
        hours = sum(shard.estimated_hours for shard in shards)
        total_shards += len(shards)
        total_hours += hours
        print(f"{model:14s} {len(shards):3d} notebooks  {hours:6.1f} h  "
              f"(max shard {max(s.estimated_hours for s in shards):.1f} h)")
    print(f"{'TOTAL':14s} {total_shards:3d} notebooks  {total_hours:6.1f} h")
