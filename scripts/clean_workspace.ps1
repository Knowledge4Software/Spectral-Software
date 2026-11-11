param(
    [switch]$DryRun,
    [string]$ReportPath = 'output/revision/maintenance/cleanup-last.json'
)

$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$cleanupPrefix = $cleanupRoot + '\'
$cleanupRecords = [Collections.Generic.List[object]]::new()
$cleanupSkipped = [Collections.Generic.List[object]]::new()
$cleanupTargets = [Collections.Generic.List[string]]::new()

function Get-CleanupPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($cleanupPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup path escapes workspace: $absolute"
    }
    # Reject links in every existing ancestor as well as the target itself.
    $ancestor = $absolute
    while ($ancestor.Length -ge $cleanupRoot.Length) {
        $info = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
        if ($null -ne $info) {
            if ($info.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Cleanup path contains a directory/file link: $ancestor"
            }
        }
        if ($ancestor -eq $cleanupRoot) { break }
        $ancestor = [IO.Path]::GetDirectoryName($ancestor)
    }
    return $absolute
}

function Get-CleanupRelative([string]$Path) {
    return $Path.Substring($cleanupPrefix.Length).Replace('\', '/')
}

function Add-CleanupTarget([string]$Relative) {
    $candidate = Join-Path $cleanupRoot $Relative
    if (Test-Path -LiteralPath $candidate) { $cleanupTargets.Add($candidate) }
}

# Validate the report destination before any mutation.
$cleanupReportPath = Get-CleanupPath (Join-Path $cleanupRoot $ReportPath)
$cleanupReportPrefix = (Join-Path $cleanupRoot 'output/revision/maintenance').TrimEnd('\') + '\'
if (-not $cleanupReportPath.StartsWith($cleanupReportPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetExtension($cleanupReportPath) -ne '.json') {
    throw 'Cleanup reports must be JSON files under output/revision/maintenance/'
}
if (Test-Path -LiteralPath $cleanupReportPath) {
    $priorReport = Get-Content -LiteralPath $cleanupReportPath -Raw | ConvertFrom-Json
    if ($priorReport.schema -ne 'spectra_workspace_cleanup_v2') { throw 'Refusing to overwrite an unrelated report' }
}
$cleanupGitRoot = (& git -C $cleanupRoot rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0 -or [IO.Path]::GetFullPath($cleanupGitRoot).TrimEnd('\') -ne $cleanupRoot) {
    throw 'The cleaner must run inside its own Git workspace'
}

# Only cache directories are discovered recursively. Never traverse links or
# environments, and never classify research files by age/extension/size.
foreach ($source in @('create_datasets_graphs', 'experiments', 'revision', 'pipelines',
                      'scripts', 'spectral_code', 'tests', 'output/revision')) {
    $sourcePath = Join-Path $cleanupRoot $source
    if (-not (Test-Path -LiteralPath $sourcePath)) { continue }
    try { $sourcePath = Get-CleanupPath $sourcePath } catch {
        $cleanupSkipped.Add([ordered]@{path = $source; reason = $_.Exception.Message}); continue
    }
    $stack = [Collections.Generic.Stack[string]]::new()
    $stack.Push($sourcePath)
    while ($stack.Count -gt 0) {
        $directory = $stack.Pop()
        foreach ($child in (Get-ChildItem -LiteralPath $directory -Directory -Force)) {
            if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
            if ($child.Name -in @('.git', '.venv', 'venv', 'env', 'ENV')) { continue }
            if ($child.Name -in @('__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.ruff_cache', '.mypy_cache')) {
                $cleanupTargets.Add($child.FullName)
            } else { $stack.Push($child.FullName) }
        }
    }
}
foreach ($relative in @('.pytest_cache', '.ruff_cache', '.mypy_cache',
        'output/revision/input_v2/pytest_checks_retry',
        'output/revision/core_controls_review/pytest_queues',
        'output/revision/v2_results_review/pytest_queue',
        'output/revision/validation/generator_check', 'output/revision/validation/pytest_layout',
        'output/revision/validation/hf_setup_generated', 'output/revision/validation/hf_setup_pytest',
        'output/revision/validation/hf_setup_dependencies', 'output/revision/validation/hf_real_smoke',
        'output/revision/validation/packaging_precheck', 'output/revision/validation/packaging_precheck_after_dedup',
        'output/revision/validation/small_report_generator_tests', 'output/revision/validation/small_report_smoke',
        'output/revision/validation/text_resume_snippet_smoke',
        'output/revision/maintenance/cleanup-tests')) { Add-CleanupTarget $relative }

$cleanupOutput = Join-Path $cleanupRoot 'output/revision'
if (Test-Path -LiteralPath $cleanupOutput) {
    foreach ($child in (Get-ChildItem -LiteralPath $cleanupOutput -Directory)) {
        # Known CPU pytest fixtures, not arbitrary test_* experiment outputs.
        if ($child.Name -match '^test_(validation_recovery|recovery_queue|restart|repeat_running|independent_batch|architecture_update|core_completed)_\d{4}$') {
            $cleanupTargets.Add($child.FullName)
        }
    }
}

# A parent fixture subsumes any cache targets inside it.
$cleanupSelected = [Collections.Generic.List[string]]::new()
foreach ($candidate in ($cleanupTargets | Sort-Object -Unique | Sort-Object Length)) {
    $absolute = [IO.Path]::GetFullPath($candidate)
    if (@($cleanupSelected | Where-Object { $absolute -eq $_ -or $absolute.StartsWith($_ + '\', [StringComparison]::OrdinalIgnoreCase) }).Count -eq 0) {
        $cleanupSelected.Add($absolute)
    }
}
foreach ($target in $cleanupSelected) {
    try {
        $absolute = Get-CleanupPath (Resolve-Path -LiteralPath $target).Path
        $relative = Get-CleanupRelative $absolute
        $children = @(Get-ChildItem -LiteralPath $absolute -Recurse -Force)
        if (@($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -gt 0) {
            throw 'Target contains a directory/file link'
        }
        $tracked = @(& git -C $cleanupRoot ls-files -- $relative)
        if ($LASTEXITCODE -ne 0 -or $tracked.Count -gt 0) { throw 'Target contains tracked files' }
        $files = @($children | Where-Object { -not $_.PSIsContainer })
        $bytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
        if (-not $DryRun) { Remove-Item -LiteralPath $absolute -Recurse -Force }
        $cleanupRecords.Add([pscustomobject][ordered]@{path = $relative; files = $files.Count; bytes = $bytes;
            action = $(if ($DryRun) {'would_remove'} else {'removed'})})
    } catch {
        $cleanupSkipped.Add([ordered]@{path = $target; reason = $_.Exception.Message})
    }
}

# Empty legacy containers only. Inaccessible/nonempty folders remain intact;
# the cleaner never changes ownership or permissions to force a deletion.
$cleanupEmpty = [Collections.Generic.List[string]]::new()
foreach ($relative in @('.test_tmp', 'experiments/kaggle/revision', 'docs/revision', 'docs',
        'output/review', 'tmp', 'output/revision/core_controls_review', 'output/revision/v2_results_review')) {
    $candidate = Join-Path $cleanupRoot $relative
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    try {
        $absolute = Get-CleanupPath $candidate
        if (@(Get-ChildItem -LiteralPath $absolute -Force).Count -eq 0) {
            if (-not $DryRun) { Remove-Item -LiteralPath $absolute -Force }
            $cleanupEmpty.Add($relative)
        }
    } catch { $cleanupSkipped.Add([ordered]@{path = $relative; reason = $_.Exception.Message}) }
}
$cleanupReport = [ordered]@{schema = 'spectra_workspace_cleanup_v2'; dry_run = [bool]$DryRun;
    workspace = $cleanupRoot; records = @($cleanupRecords.ToArray()); empty_directories = @($cleanupEmpty.ToArray());
    skipped = @($cleanupSkipped.ToArray()); total_files = [long](($cleanupRecords | Measure-Object -Property files -Sum).Sum);
    total_bytes = [long](($cleanupRecords | Measure-Object -Property bytes -Sum).Sum)}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($cleanupReportPath)) | Out-Null
    $cleanupReport | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $cleanupReportPath -Encoding UTF8
}
$cleanupReport | ConvertTo-Json -Depth 6
