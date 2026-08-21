param(
    [string]$IncomingLaptop1Cache = "C:\PyProjects\spectrals\outputs\codenet_4l_distributed\laptop1_nonclone_python_java_cpp_cache"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$ctx = Get-CodeNetContext
Set-CodeNetPerformanceEnvironment

$incoming = (Resolve-Path -LiteralPath $IncomingLaptop1Cache).Path
if (-not (Test-Path -LiteralPath (Join-Path $incoming "index.sqlite3") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $incoming "shards") -PathType Container)) {
    throw "Incoming laptop-1 cache must contain index.sqlite3 and shards: $incoming"
}
$cloneCache = Join-Path $ctx.Outputs "codenet_4l_all_clones\graph_record_cache"
$finalOutput = Join-Path $ctx.Outputs "codenet_4l_clone50k_diff50k"
$finalZip = Join-Path $finalOutput "codenet_4l_clone50k_diff50k_clean_data.zip"

Write-Host "Importing laptop-1 records into the laptop-2 cache." -ForegroundColor Green
Invoke-CodeNetPython $ctx.Python @(
    $ctx.CombinedRunner,
    "--shared-cache-dir", $cloneCache,
    "--output-dir", $finalOutput,
    "--import-cache-dir", $incoming,
    "--start-at", "reuse",
    "--stop-after", "recover",
    "--no-zip"
)

Write-Host "Auditing all 135,068 endpoints and the 128-value sparse contract before packaging." -ForegroundColor Green
Invoke-CodeNetPython $ctx.Python @(
    $ctx.Audit,
    "--cache-dir", $cloneCache,
    "--scope", "combined",
    "--require-complete",
    "--check-spectra"
)

Write-Host "Writing the final 50k-clone + 50k-non-clone Kaggle archive." -ForegroundColor Green
Invoke-CodeNetPython $ctx.Python @(
    $ctx.CombinedRunner,
    "--shared-cache-dir", $cloneCache,
    "--output-dir", $finalOutput,
    "--start-at", "package",
    "--stop-after", "package"
)

Invoke-CodeNetPython $ctx.Python @(
    $ctx.Audit,
    "--cache-dir", $cloneCache,
    "--scope", "combined",
    "--require-complete",
    "--zip", $finalZip
)
Write-Host "`nFINAL KAGGLE DATASET:" -ForegroundColor Green
Write-Host $finalZip -ForegroundColor Yellow
