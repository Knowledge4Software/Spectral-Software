# Project maintenance

Use [the current queue](queues/RUN_NEXT_INDEPENDENT_BATCH.md) for independent
controls/repeats and [the baseline roster](queues/LAUNCH_24.md) for older jobs.
The CSV trackers under `results/` are authoritative; templates alone are not
instructions to launch or repeat an experiment.

## Disposable files

`scripts/clean_workspace.ps1` removes Python/Jupyter/tool caches and explicitly
identified CPU test fixtures, including their generated notebooks, synthetic
checkpoints and packaging smoke tests. It excludes environments, tracked files,
directory links, unknown experiment folders and research artifacts.

```powershell
powershell -NoProfile -File scripts/clean_workspace.ps1 -DryRun
powershell -NoProfile -File scripts/clean_workspace.ps1
```

The default report is `output/revision/maintenance/cleanup-last.json`. For a
dated report, pass `-ReportPath output/revision/maintenance/<date>/cleanup.json`.
The report lists removed directories, file/byte counts and skipped paths.
Inaccessible paths remain intact; the script does not change their permissions.

## Preserved research material

- `spectral_code/`, dataset construction code, pipelines, scripts and tests.
- All 193 frozen revision notebooks, their manifests, numerical-fix copies,
  original continuation notebooks and historical paper experiments.
- `../data/benchmarks/`, `../outputs/revesion/`, checkpoints and every reported
  result, including incomplete and unfavorable attempts.
- Immutable release ZIPs, notebook-output backups, original reviewer notes,
  manuscript files and local environment/configuration files.

Frozen notebook hashes and the scientific training fingerprint protect ongoing
continuation and comparisons. Refresh only the approved independent copies with
`python -m scripts.prepare_revision_remaining_batch`; the general notebook
builder is for a separately versioned release. Preserve output versions when
continuing Kaggle jobs.

## Cleanup record: 2026-11-11

The cleanup removed regenerable test artifacts and caches, and consolidated
the active startup instructions. The obsolete `FIRST_RUN.md` and `RUN_ORDER.md`
were removed from the active documentation; their contents remain in Git history.
The Persian guide now shows current queues instead of accumulated historical
instructions to rerun already completed pilots.

The main cleanup removed about 308 MiB from 22 temporary/cache directories.
The old `.test_tmp` directory was skipped because Windows denied access.

Detailed before/after hashes and deletion records are local files under
`output/revision/maintenance/2026-11-11/`. The initial inventory covers 698 tracked
files, 47 research ZIPs, seven immutable release ZIPs and the manuscript.

The [file-organizer skill](https://github.com/Ayman-AbdelMohaimen/codex-skills/blob/14667b4850a3d99b23f8fbc0cf73c879ef219c66/file-organizer/SKILL.md)
was installed at `C:/Users/koush/.codex/skills/file-organizer/`, pinned to commit
`14667b4850a3d99b23f8fbc0cf73c879ef219c66`. Its inventory, duplicate detection,
active/archive separation and operation logging were used within the user's
explicit project-cleanup authorization. It contains Markdown instructions only;
no third-party executable cleanup code was run.
