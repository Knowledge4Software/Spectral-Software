"""Canonical source-language and Joern frontend definitions.

Joern's public language names do not always match dataset labels.  In
particular, C and C++ are both parsed by ``c2cpg`` through ``--language c``.
Keeping that translation in one place prevents a dataset runner from silently
writing a C++ program as Python or selecting the wrong frontend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSupport:
    canonical_name: str
    extension: str
    joern_language: str


LANGUAGE_SUPPORT: dict[str, LanguageSupport] = {
    "java": LanguageSupport("java", ".java", "javasrc"),
    "python": LanguageSupport("python", ".py", "pythonsrc"),
    "c": LanguageSupport("c", ".c", "c"),
    "cpp": LanguageSupport("cpp", ".cpp", "c"),
    "csharp": LanguageSupport("csharp", ".cs", "csharpsrc"),
}

LANGUAGE_ALIASES = {
    "java": "java",
    "javasrc": "java",
    "python": "python",
    "python2": "python",
    "python3": "python",
    "py": "python",
    "py2": "python",
    "py3": "python",
    "pythonsrc": "python",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "gnu++": "cpp",
    "csharp": "csharp",
    "c#": "csharp",
    "cs": "csharp",
    "csharpsrc": "csharp",
}


def normalize_source_language(raw: object, *, default: str | None = None) -> str:
    """Return a canonical source language or raise a useful error."""
    key = str(raw or "").strip().lower()
    if not key and default is not None:
        key = default.strip().lower()
    if key.startswith("semantic_"):
        key = key.split("_", 1)[1]
    try:
        return LANGUAGE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_SUPPORT))
        raise ValueError(f"Unsupported source language {raw!r}; expected one of: {supported}") from exc


def source_extension(raw: object) -> str:
    return LANGUAGE_SUPPORT[normalize_source_language(raw)].extension


def joern_language(raw: object) -> str:
    """Map a dataset/source label to the Joern ``--language`` value."""
    return LANGUAGE_SUPPORT[normalize_source_language(raw)].joern_language


def canonical_language_from_joern(raw: object) -> str:
    """Normalize either a source label or an existing Joern frontend name."""
    return normalize_source_language(raw, default="java")

