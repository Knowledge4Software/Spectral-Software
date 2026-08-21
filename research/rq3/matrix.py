"""The RQ3 experiment matrix: every training configuration for Tables 7-10.

A configuration names the language buckets a model trains on and the bucket it
is evaluated on. Nothing here touches a model or a dataset, so the matrix can
be inspected, counted, and diffed without a GPU.

Language codes follow the paper: J=java, P=python, C=cpp, S=csharp.
Bucket ids match ``configuration_id`` in the CodeNet clean-data pairs file:
same-language buckets are the bare language ("java"), cross-language buckets
join two languages in the export's own fixed order ("python_java").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

# Paper symbol -> clean-data language name.
LANGUAGES: dict[str, str] = {"J": "java", "P": "python", "C": "cpp", "S": "csharp"}
SYMBOLS: tuple[str, ...] = ("J", "P", "C", "S")

# The export writes each cross-language bucket under one fixed ordering only,
# so the reverse spelling must map onto the same bucket id.
_CROSS_BUCKETS: dict[frozenset[str], str] = {
    frozenset({"java", "python"}): "python_java",
    frozenset({"java", "cpp"}): "java_cpp",
    frozenset({"java", "csharp"}): "java_csharp",
    frozenset({"python", "cpp"}): "python_cpp",
    frozenset({"python", "csharp"}): "python_csharp",
    frozenset({"cpp", "csharp"}): "cpp_csharp",
}


def bucket(left: str, right: str) -> str:
    """Return the clean-data ``configuration_id`` for a language pair."""
    first, second = LANGUAGES[left], LANGUAGES[right]
    if first == second:
        return first
    return _CROSS_BUCKETS[frozenset({first, second})]


@dataclass(frozen=True)
class Configuration:
    """One trainable setup: which buckets to train on, which bucket to test."""

    table: str
    name: str
    train_buckets: tuple[str, ...]
    test_bucket: str
    target: str
    path: tuple[str, ...] = ()
    reinforcement: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable identifier used for result rows and resume checkpoints."""
        return f"{self.table}|{self.name}"


def table7_within() -> list[Configuration]:
    """Within-language reference: train L-L, test L-L."""
    return [
        Configuration(
            table="table7_within",
            name=f"{symbol}-{symbol}",
            train_buckets=(bucket(symbol, symbol),),
            test_bucket=bucket(symbol, symbol),
            target=symbol,
            metadata={"setting": "within_language"},
        )
        for symbol in SYMBOLS
    ]


def table7_cross() -> list[Configuration]:
    """Cross-language generalization with no cross-language supervision.

    Training sees both endpoint languages only as same-language pairs; the
    unseen mixed-language bucket is the test target.
    """
    configurations = []
    for index, left in enumerate(SYMBOLS):
        for right in SYMBOLS[index + 1:]:
            configurations.append(
                Configuration(
                    table="table7_cross",
                    name=f"{left}-{right}",
                    train_buckets=(bucket(left, left), bucket(right, right)),
                    test_bucket=bucket(left, right),
                    target=f"{left}{right}",
                    metadata={"setting": "cross_language_no_supervision"},
                )
            )
    return configurations


def table8() -> list[Configuration]:
    """Length-1 bridges X1 -> Y: train X1-X1 and X1-Y, test Y-Y."""
    configurations = []
    for target in SYMBOLS:
        for source in SYMBOLS:
            if source == target:
                continue
            configurations.append(
                Configuration(
                    table="table8",
                    name=f"{source}->{target}",
                    train_buckets=(bucket(source, source), bucket(source, target)),
                    test_bucket=bucket(target, target),
                    target=target,
                    path=(source, target),
                    reinforcement="none",
                    metadata={"bridge_length": 1},
                )
            )
    return configurations


# Reinforcement variants add same-language buckets for the intermediate
# languages on top of the mandatory path.
_TABLE9_VARIANTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("none", ()),
    ("X2-X2", (1,)),
)
_TABLE10_VARIANTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("none", ()),
    ("X2-X2", (1,)),
    ("X3-X3", (2,)),
    ("X2-X2+X3-X3", (1, 2)),
)


def _bridge_configurations(
    table: str, path_length: int, variants: tuple[tuple[str, tuple[int, ...]], ...]
) -> list[Configuration]:
    """Build every distinct-language path of the requested length.

    A path (X1, ..., Xn, Y) trains on X1-X1 plus each consecutive hop, so the
    target language is reachable only through the chain.
    """
    configurations = []
    for target in SYMBOLS:
        others = [symbol for symbol in SYMBOLS if symbol != target]
        for intermediates in permutations(others, path_length):
            path = (*intermediates, target)
            mandatory = [bucket(path[0], path[0])]
            mandatory += [bucket(path[index], path[index + 1]) for index in range(len(path) - 1)]
            for label, reinforce_indices in variants:
                extra = [bucket(path[index], path[index]) for index in reinforce_indices]
                configurations.append(
                    Configuration(
                        table=table,
                        name=f"{'->'.join(path)}|{label}",
                        # dict.fromkeys keeps first-seen order while removing
                        # any bucket a reinforcement would duplicate.
                        train_buckets=tuple(dict.fromkeys(mandatory + extra)),
                        test_bucket=bucket(target, target),
                        target=target,
                        path=path,
                        reinforcement=label,
                        metadata={"bridge_length": path_length},
                    )
                )
    return configurations


def table9() -> list[Configuration]:
    """Length-2 bridges X1 -> X2 -> Y, with and without X2-X2."""
    return _bridge_configurations("table9", 2, _TABLE9_VARIANTS)


def table10() -> list[Configuration]:
    """Length-3 bridges X1 -> X2 -> X3 -> Y across four reinforcement modes."""
    return _bridge_configurations("table10", 3, _TABLE10_VARIANTS)


def all_configurations() -> list[Configuration]:
    return [*table7_within(), *table7_cross(), *table8(), *table9(), *table10()]


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for configuration in all_configurations():
        counts[configuration.table] = counts.get(configuration.table, 0) + 1
    counts["total"] = sum(counts.values())
    return counts


if __name__ == "__main__":
    for table, count in summary().items():
        print(f"{table:16s} {count:4d}")
