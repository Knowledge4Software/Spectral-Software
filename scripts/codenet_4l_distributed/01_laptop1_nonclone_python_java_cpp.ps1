param(
    [string]$JoernHome = "C:\joern-cli"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$ctx = Get-CodeNetContext
Set-CodeNetPerformanceEnvironment -JoernHome $JoernHome

$joernParse = Join-Path $JoernHome "joern-parse.bat"
if (-not (Test-Path -LiteralPath $joernParse -PathType Leaf)) {
    throw "Joern was not found at $JoernHome"
}
& java -version
if ($LASTEXITCODE -ne 0) { throw "Java is not available on PATH" }

$cloneCache = Join-Path $ctx.Outputs "codenet_4l_all_clones\graph_record_cache"
$legacyPartialCache = Join-Path $ctx.Outputs "codenet_4l_clone50k_diff50k_nonclone_java_cpp_cache"
$partialCache = Join-Path $ctx.Outputs "codenet_4l_distributed\laptop1_nonclone_python_java_cpp_cache"
$workOutput = Join-Path $ctx.Outputs "codenet_4l_distributed\laptop1_nonclone_python_java_cpp_work"

Write-Host "Building non-clone-only Python + Java + C++ endpoints." -ForegroundColor Green
Write-Host "Durable cache: $partialCache"
Invoke-CodeNetPython $ctx.Python @(
    $ctx.CombinedRunner,
    "--shared-cache-dir", $partialCache,
    "--output-dir", $workOutput,
    "--import-cache-dir", $legacyPartialCache,
    "--import-cache-dir", $cloneCache,
    "--only-nonclone-endpoints",
    "--languages", "python,java,cpp",
    "--start-at", "reuse",
    "--stop-after", "graphs"
)

Invoke-CodeNetPython $ctx.Python @(
    $ctx.Audit,
    "--cache-dir", $partialCache,
    "--scope", "nonclone"
)
Write-Host "`nLaptop 1 is complete. Copy this directory to laptop 2 without changing its contents:" -ForegroundColor Green
Write-Host $partialCache -ForegroundColor Yellow
