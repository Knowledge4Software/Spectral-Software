"""Evaluation helpers with lazy imports for lightweight dataset tooling.

Preparing a dataset must not import the model pipeline, NetworkX, or parser
stack.  The public baseline symbols remain backward-compatible and are loaded
only when a caller actually requests them.
"""

from __future__ import annotations

from importlib import import_module


__all__ = (
    "BigCloneBenchLoader",
    "ClonePair",
    "SpectralCloneBaseline",
    "evaluate_binary",
    "evaluate_by_type",
)

_SYMBOL_MODULES = {
    "BigCloneBenchLoader": ".bcb_dataset",
    "ClonePair": ".bcb_dataset",
    "SpectralCloneBaseline": ".baseline",
    "evaluate_binary": ".baseline",
    "evaluate_by_type": ".baseline",
}


def __getattr__(name: str):
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
