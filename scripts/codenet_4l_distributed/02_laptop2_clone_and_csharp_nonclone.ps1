param(
    [string]$PreviousLaptop2CacheDir = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$ctx = Get-CodeNetContext
Set-CodeNetPerformanceEnvironment

$cloneCache = Join-Path $ctx.Outputs "codenet_4l_all_clones\graph_record_cache"
$cloneOutput = Join-Path $ctx.Outputs "codenet_4l_clone_50k"
$csharpWork = Join-Path $ctx.Outputs "codenet_4l_distributed\laptop2_csharp_nonclone_work"

if ($PreviousLaptop2CacheDir) {
    $previous = (Resolve-Path -LiteralPath $PreviousLaptop2CacheDir).Path
    if ($previous -eq (Resolve-Path -LiteralPath $cloneCache).Path) {
        throw "PreviousLaptop2CacheDir must be a backup cache, not the active destination cache."
    }
    Invoke-CodeNetPython $ctx.Python @(
        $ctx.CombinedRunner,
        "--shared-cache-dir", $cloneCache,
        "--output-dir", (Join-Path $ctx.Outputs "codenet_4l_clone50k_diff50k"),
        "--import-cache-dir", $previous,
        "--start-at", "reuse",
        "--stop-after", "reuse",
        "--no-zip"
    )
}

Write-Host "Finishing the remaining C# clone endpoints (source parser; Joern is not required)." -ForegroundColor Green
Invoke-CodeNetPython $ctx.Python @(
    $ctx.CloneRunner,
    "--shared-cache-dir", $cloneCache,
    "--output-dir", $cloneOutput,
    "--recover-work-from", (Join-Path $cloneOutput "_batch_work"),
    "--languages", "csharp",
    "--start-at", "recover",
    "--stop-after", "graphs"
)

Write-Host "Building the C# non-clone-only endpoints into the same durable cache." -ForegroundColor Green
Invoke-CodeNetPython $ctx.Python @(
    $ctx.CombinedRunner,
    "--shared-cache-dir", $cloneCache,
    "--output-dir", $csharpWork,
    "--only-nonclone-endpoints",
    "--languages", "csharp",
    "--start-at", "recover",
    "--stop-after", "graphs"
)

Invoke-CodeNetPython $ctx.Python @(
    $ctx.Audit,
    "--cache-dir", $cloneCache,
    "--scope", "combined"
)
Write-Host "`nLaptop 2 is ready for the final merge. Keep this cache in place:" -ForegroundColor Green
Write-Host $cloneCache -ForegroundColor Yellow
