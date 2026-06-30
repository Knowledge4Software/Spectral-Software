from pathlib import Path
import sys


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "spectral_code").exists() and (candidate / "pipelines").exists():
            return candidate
    raise RuntimeError("Project root not found.")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from spectral_code.evaluation.pipeline_section_runner import SectionConfig, run_full_pipeline_section


if __name__ == "__main__":
    run_full_pipeline_section(
        SectionConfig(dataset="semantic", variant="cs", run_dir=Path(__file__).resolve().parent)
    )

