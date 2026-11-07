"""The notebook must replace unsupported HF packages before model loading."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.revision_hf_setup import HF_PACKAGES, LORA_PACKAGES, notebook_setup


@pytest.fixture
def environment(monkeypatch):
    versions = {name: spec.removeprefix("==") for name, spec in HF_PACKAGES.items()}
    versions["torch"] = "2.11.0+cu130"
    calls = []

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    def install(command, *, check):
        assert check
        constraint = Path(command[command.index("--constraint") + 1]).read_text()
        assert constraint == "torch==2.11.0+cu130\n"
        assert not any(arg.startswith("torch==") for arg in command)
        for argument in command:
            if "==" in argument:
                name, release = argument.split("==")
                versions[name] = release
        calls.append(command)

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(subprocess, "run", install)
    # Model libraries are deliberately not imported by the notebook before setup.
    for name in list(sys.modules):
        if name.split(".")[0] in {"transformers", "huggingface_hub", "tokenizers", "peft", "accelerate"}:
            monkeypatch.delitem(sys.modules, name)
    return versions, calls


def test_supported_environment_needs_no_install(environment):
    _, calls = environment
    exec(notebook_setup(), {})
    assert calls == []


@pytest.mark.parametrize("release", ["4.50.0", "5.0.0", "5.22.2"])
def test_old_and_new_major_versions_are_replaced(environment, release):
    versions, calls = environment
    versions["transformers"] = release
    exec(notebook_setup(), {})
    assert len(calls) == 1
    assert versions["transformers"] == "4.57.3"


def test_lora_installs_both_optional_packages(environment):
    versions, calls = environment
    exec(notebook_setup(require_lora=True), {})
    assert len(calls) == 1
    for name, spec in LORA_PACKAGES.items():
        assert versions[name] == spec.removeprefix("==")


def test_loaded_unsupported_module_requires_a_fresh_kernel(environment, monkeypatch):
    versions, calls = environment
    versions["transformers"] = "5.0.0"
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.0.0"))
    with pytest.raises(RuntimeError, match="Restart the kernel/session"):
        exec(notebook_setup(), {})
    assert len(calls) == 1
