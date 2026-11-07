"""Optional tqdm adapters; leave scientific source/checkpoint identities unchanged.

Installed by notebook setup, rather than by training modules. Adapters preserve
loader order and file contents. The benchmark retains its unwrapped loader.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import math
from pathlib import Path

from tqdm.auto import tqdm

_patches = []
_state_bars = {}


def _patch(module, name, replacement):
    _patches.append((module, name, getattr(module, name)))
    setattr(module, name, replacement)


class _ProgressLoader:
    def __init__(self, loader, *, initial, total, description):
        self.loader, self.initial, self.total, self.description = loader, initial, total, description

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        with tqdm(self.loader, initial=self.initial, total=self.total,
                  desc=self.description, unit="batch", mininterval=2, leave=False) as bar:
            yield from bar


class _ProgressReader:
    def __init__(self, stream, bar):
        self.stream, self.bar = stream, bar

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def _update(self):
        # Gzip's underlying file position measures compressed bytes; plain text
        # uses the buffered binary stream. Do not seek or consume extra input.
        binary = getattr(self.stream, "buffer", self.stream)
        physical = getattr(binary, "fileobj", binary)
        current = min(self.bar.total, physical.tell())
        self.bar.update(max(0, current - self.bar.n))

    def __iter__(self):
        for line in self.stream:
            self._update()
            yield line

    def read(self, *args, **kwargs):
        result = self.stream.read(*args, **kwargs)
        self._update()
        return result

    def readline(self, *args, **kwargs):
        result = self.stream.readline(*args, **kwargs)
        self._update()
        return result


def _show_state(path, state):
    if "codes" not in state or "next_index" not in state:
        return
    key = str(path)
    bar = _state_bars.get(key)
    if bar is None:
        bar = _state_bars[key] = tqdm(total=state["codes"], initial=state["next_index"],
                                      desc="Tokenize" if Path(path).name == "tokens.json" else "Embed",
                                      unit="program", mininterval=2, leave=False)
    bar.update(max(0, state["next_index"] - bar.n))


def _close_states():
    for bar in _state_bars.values():
        bar.close()
    _state_bars.clear()


def _close_after(function):
    @wraps(function)
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        finally:
            _close_states()
    return run


def install_progress():
    """Enable notebook progress once; no optimizer, RNG, sampler or config edits."""
    if _patches:
        return
    from . import data, training, transformer, embeddings
    # Import before wrapping training._loader: measured benchmark loops keep the
    # original alias and exclude display overhead, regardless of import order.
    from . import benchmark
    original_loader = training._loader
    if isinstance(benchmark._loader, type(original_loader)):
        benchmark._loader = original_loader

    @wraps(original_loader)
    def loader(frame, graphs, config, batches=None):
        result = original_loader(frame, graphs, config, batches)
        total = math.ceil(len(frame) / config.batch_size)
        initial = total - len(result) if batches is not None else 0
        return _ProgressLoader(result, initial=initial, total=total,
                               description=f"Train seed {config.seed}" if batches is not None else "Evaluate")

    original_open = data._open_text

    @contextmanager
    def open_text(path):
        path = Path(path)
        with original_open(path) as stream:
            with tqdm(total=path.stat().st_size, desc=f"Read {path.name}", unit="B",
                      unit_scale=True, mininterval=2, leave=False) as bar:
                yield _ProgressReader(stream, bar)

    original_replace = transformer.replace_file

    @wraps(original_replace)
    def replace_file(source, target):
        result = original_replace(source, target)
        if Path(target).name == "tokens.json":
            _show_state(target, json.loads(Path(target).read_text(encoding="utf-8")))
        return result

    original_save = embeddings._json_save

    @wraps(original_save)
    def json_save(state, path):
        result = original_save(state, path)
        _show_state(path, state)
        return result

    _patch(training, "_loader", loader)
    _patch(data, "_open_text", open_text)
    _patch(transformer, "replace_file", replace_file)
    _patch(transformer, "prepare_tokens", _close_after(transformer.prepare_tokens))
    _patch(embeddings, "_json_save", json_save)
    _patch(embeddings, "extract_embeddings", _close_after(embeddings.extract_embeddings))


def uninstall_progress():
    """Restore ordinary loaders/readers, useful for local tests and scripts."""
    _close_states()
    while _patches:
        module, name, original = _patches.pop()
        setattr(module, name, original)
