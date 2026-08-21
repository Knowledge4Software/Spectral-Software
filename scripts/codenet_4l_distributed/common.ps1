Set-StrictMode -Version Latest

function Get-CodeNetContext {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
    $spectralsRoot = Split-Path $projectRoot -Parent
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Project Python was not found at $python"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $spectralsRoot "data\codenet_4l_clone50k_diff50k_prepared\metadata.json") -PathType Leaf)) {
        throw "Prepared CodeNet 50k+50k data is missing below $spectralsRoot\data"
    }
    [pscustomobject]@{
        ProjectRoot = $projectRoot
        SpectralsRoot = $spectralsRoot
        Python = $python
        Outputs = Join-Path $spectralsRoot "outputs"
        CombinedRunner = Join-Path $projectRoot "notebooks\datasets\codenet_4l\clone50k_diff50k\run_pipeline.py"
        CloneRunner = Join-Path $projectRoot "notebooks\datasets\codenet_4l\clone_50k\run_pipeline.py"
        Audit = Join-Path $projectRoot "scripts\codenet_4l_distributed\audit.py"
    }
}

function Set-CodeNetPerformanceEnvironment {
    param([string]$JoernHome = "")
    if ($JoernHome) { $env:JOERN_HOME = $JoernHome }
    $env:PYTHONUTF8 = "1"
    $env:SPECTRAL_APPROX_TOPK = "128"
    $env:PACKAGE_EIGENVALUE_LIMIT = "128"
    $env:SPECTRAL_SPARSE_SOLVER = "shift_invert"
    $env:SPECTRAL_SHARD_WORKERS = "4"
    $env:SPECTRAL_WORKERS = "1"
    $env:SPECTRAL_BLAS_THREADS = "1"
    $env:OPENBLAS_NUM_THREADS = "1"
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    $env:JOERN_PARSE_CHUNK_SIZE = "500"
    $env:JOERN_EXPORT_WORKERS = "2"
    # The direct Java frontend can omit the overlays needed for CFG/DDG DOT
    # export; use Joern's wrapper for the complete baseline graph contract.
    $env:JOERN_USE_DIRECT_FRONTEND = "0"
    $env:GRAPH_SHARD_SIZE = "500"
}

function Invoke-CodeNetPython {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "`n>>> $Python $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}
