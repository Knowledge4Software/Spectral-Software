"""One declaration of which (dataset, language, layer) combinations are unusable.

Two different tools need to agree on this: the health gate, which must not fail a
build over a limitation we already know about, and the notebook patcher, which
must stop the baselines from scoring a layer that does not exist. Keeping the
table in two places is how they drift apart, so it lives here.

A limitation is a *declared* gap: a layer the toolchain genuinely cannot produce
today. Anything not declared here that fails the health check is a regression and
must break the build.
"""

from __future__ import annotations


CSHARP_OVERLAY_FAILURE = (
    "Joern's csharpsrc2cpg builds the CPG (its method table is complete) but crashes while "
    "applying the CFG/DDG overlays with 'AssertionError: ... trying to resolve astParent ... "
    "malformed cpg'. The AST is recovered from tree-sitter and is healthy; CFG is empty and "
    "DDG degenerates to 3-node stubs."
)

# (dataset, language, layer) -> why it cannot be produced.
KNOWN_LIMITATIONS: dict[tuple[str, str, str], str] = {
    ("gptclonebench_v3", "csharp", "cfg"): CSHARP_OVERLAY_FAILURE,
    ("gptclonebench_v3", "csharp", "ddg"): CSHARP_OVERLAY_FAILURE,
    ("semanticclonebench_v3", "csharp", "cfg"): CSHARP_OVERLAY_FAILURE,
    ("semanticclonebench_v3", "csharp", "ddg"): CSHARP_OVERLAY_FAILURE,
}


def limitation_for(dataset: str, language: str, layer: str) -> str | None:
    return KNOWN_LIMITATIONS.get((dataset, language, layer))


def unsupported_layers(dataset: str) -> list[str]:
    """Layers no baseline should score for this dataset, in a stable order."""
    layers = {layer for (name, _, layer) in KNOWN_LIMITATIONS if name == dataset}
    return sorted(layers)


def affected_languages(dataset: str, layer: str) -> list[str]:
    return sorted({language for (name, language, kind) in KNOWN_LIMITATIONS
                   if name == dataset and kind == layer})
