# Distributed CodeNet 50k + 50k build

These scripts are the canonical two-laptop workflow for the fixed release:

- 50,000 clone pairs;
- 50,000 different-problem non-clone pairs;
- 135,068 unique endpoints;
- AST, CFG, DDG, and CPG adjacency plus up to 128 eigenvalues per layer.

## Canonical outputs

- laptop 1 cache: `outputs\codenet_4l_distributed\laptop1_nonclone_python_java_cpp_cache`
- laptop 1 resumable work: `outputs\codenet_4l_distributed\laptop1_nonclone_python_java_cpp_work`
- laptop 2/final cache: `outputs\codenet_4l_all_clones\graph_record_cache`
- laptop 2 C# work: `outputs\codenet_4l_distributed\laptop2_csharp_nonclone_work`
- final Kaggle ZIP: `outputs\codenet_4l_clone50k_diff50k\codenet_4l_clone50k_diff50k_clean_data.zip`

The older `nonclone_java_cpp_*` and `nonclone_py_csharp_*` directories are
legacy inputs only. They are preserved to avoid data loss but are not active
destinations in this workflow.

## Commands

Laptop 1 (Joern machine):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  "C:\PyProjects\spectrals\Spectral-Software\scripts\codenet_4l_distributed\01_laptop1_nonclone_python_java_cpp.ps1"
```

Laptop 2 (C# source-parser machine):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  "C:\PyProjects\spectrals\Spectral-Software\scripts\codenet_4l_distributed\02_laptop2_clone_and_csharp_nonclone.ps1"
```

After both commands finish, copy laptop 1's canonical cache directory to the
same canonical path on laptop 2 and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  "C:\PyProjects\spectrals\Spectral-Software\scripts\codenet_4l_distributed\03_laptop2_merge_and_package.ps1"
```

Run `01_laptop1_nonclone_python_java_cpp.ps1` on the machine with Joern at
`C:\joern-cli`. Run `02_laptop2_clone_and_csharp_nonclone.ps1` on the second
machine; C# uses the source parser and does not need Joern. After both finish,
copy laptop 1's cache directory to laptop 2 as
`outputs\codenet_4l_distributed\laptop1_nonclone_python_java_cpp_cache`, then run
`03_laptop2_merge_and_package.ps1` on laptop 2.

All scripts are resumable. Re-run the same script after an interruption. Do
not copy a live SQLite cache: stop the corresponding pipeline first. If laptop
2 already contains useful C# progress, update its code and data but do not
overwrite its `outputs` directory. If replacing the whole tree is unavoidable,
first preserve its existing `outputs\codenet_4l_all_clones\graph_record_cache`
under a different directory and pass that backup to script 02 with
`-PreviousLaptop2CacheDir`. Preserving the existing output tree is preferable
because it also retains incomplete but resumable spectral shards.
