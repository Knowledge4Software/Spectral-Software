"""Training dependency fingerprint, independent of reporting utilities.

Keep this list explicit: changing a model, data loader, sampler or training
policy must invalidate continuation. Reporting/plotting additions must not.
The versioning module itself is included so changes to this policy are visible.
"""
from pathlib import Path

from .protocol import fingerprint


TRAINING_SOURCES = (
    "__init__.py", "versioning.py", "canonical_types.py", "data.py",
    "alignment.py", "batching.py", "model.py", "protocol.py", "training.py",
    "io_utils.py", "legacy.py", "transformer.py", "jobs.py", "transfer.py",
    "transfer_matrix.py",
)


def training_implementation_hash(root=None):
    root = Path(root) if root is not None else Path(__file__).parent
    return fingerprint({name: (root / name).read_text(encoding="utf-8")
                        for name in TRAINING_SOURCES})
