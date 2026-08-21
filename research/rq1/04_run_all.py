"""Run both RQ1 datasets and produce the complete paper table and PNG."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.rq1.run_table import run_default_all


if __name__ == "__main__":
    run_default_all()
