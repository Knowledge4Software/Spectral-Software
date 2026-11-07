param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$revisionWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$revisionPrefix = $revisionWorkspace.TrimEnd('\') + '\'
$revisionTargets = [System.Collections.Generic.List[string]]::new()
$revisionSkipped = [System.Collections.Generic.List[string]]::new()

# Visit only project code. Never descend into a directory link or the venv.
foreach ($revisionSource in @('create_datasets_graphs', 'experiments', 'revision', 'pipelines', 'scripts', 'spectral_code', 'tests')) {
    $revisionStack = [System.Collections.Generic.Stack[string]]::new()
    $revisionStack.Push((Join-Path $revisionWorkspace $revisionSource))
    while ($revisionStack.Count -gt 0) {
        $revisionDirectory = $revisionStack.Pop()
        foreach ($revisionChild in (Get-ChildItem -LiteralPath $revisionDirectory -Directory -Force)) {
            if ($revisionChild.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { continue }
            if ($revisionChild.Name -in @('__pycache__', '.ipynb_checkpoints')) {
                $revisionTargets.Add($revisionChild.FullName)
            } else {
                $revisionStack.Push($revisionChild.FullName)
            }
        }
    }
}

$revisionPytest = Join-Path $revisionWorkspace '.pytest_cache'
if (Test-Path -LiteralPath $revisionPytest) { $revisionTargets.Add($revisionPytest) }
foreach ($revisionCheckScratch in @('output/revision/validation/generator_check', 'output/revision/validation/pytest_layout',
        'output/revision/validation/hf_setup_generated', 'output/revision/validation/hf_setup_pytest',
        'output/revision/validation/hf_setup_dependencies', 'output/revision/validation/hf_real_smoke')) {
    $revisionCheckPath = Join-Path $revisionWorkspace $revisionCheckScratch
    if (Test-Path -LiteralPath $revisionCheckPath) { $revisionTargets.Add($revisionCheckPath) }
}
$revisionScratch = Join-Path $revisionWorkspace 'tmp'
if (Test-Path -LiteralPath $revisionScratch) {
    foreach ($revisionChild in (Get-ChildItem -LiteralPath $revisionScratch -Directory)) {
        if ($revisionChild.Name -match '^(review_pytest|revision_.+)$') {
            $revisionTargets.Add($revisionChild.FullName)
        }
    }
}

$revisionRemoved = [System.Collections.Generic.List[string]]::new()
$revisionEmptyRemoved = [System.Collections.Generic.List[string]]::new()
foreach ($revisionTarget in $revisionTargets) {
    $revisionAbsolute = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $revisionTarget).Path)
    if (-not $revisionAbsolute.StartsWith($revisionPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup target escapes workspace: $revisionAbsolute"
    }
    $revisionInfo = Get-Item -LiteralPath $revisionAbsolute -Force
    if ($revisionInfo.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        $revisionSkipped.Add($revisionAbsolute); continue
    }
    $revisionLinks = @(Get-ChildItem -LiteralPath $revisionAbsolute -Recurse -Force |
        Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint })
    if ($revisionLinks.Count -gt 0) { $revisionSkipped.Add($revisionAbsolute); continue }
    $revisionRelative = $revisionAbsolute.Substring($revisionPrefix.Length).Replace('\', '/')
    $revisionTracked = @(& git -C $revisionWorkspace ls-files -- $revisionRelative)
    if ($LASTEXITCODE -ne 0 -or $revisionTracked.Count -gt 0) {
        throw "Refusing to clean a tracked directory: $revisionAbsolute"
    }
    if (-not $DryRun) {
        try {
            Remove-Item -LiteralPath $revisionAbsolute -Recurse -Force
        } catch {
            $revisionSkipped.Add($revisionAbsolute); continue
        }
    }
    $revisionRemoved.Add($revisionRelative)
}

# Retired revision paths contain only empty directories after the layout move.
# Remove empty directories only; never recurse-delete an old source tree.
foreach ($revisionRetired in @('experiments/kaggle/revision', 'docs/revision', 'docs', 'output/review', 'tmp')) {
    $revisionRetiredAbsolute = [System.IO.Path]::GetFullPath((Join-Path $revisionWorkspace $revisionRetired))
    if (-not $revisionRetiredAbsolute.StartsWith($revisionPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Retired path escapes workspace: $revisionRetiredAbsolute"
    }
    if (-not (Test-Path -LiteralPath $revisionRetiredAbsolute -PathType Container)) { continue }
    $revisionRetiredInfo = Get-Item -LiteralPath $revisionRetiredAbsolute -Force
    if ($revisionRetiredInfo.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { continue }
    $revisionRetiredChildren = @(Get-ChildItem -LiteralPath $revisionRetiredAbsolute -Recurse -Force)
    if (@($revisionRetiredChildren | Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint }).Count -gt 0) { continue }
    $revisionRetiredDirectories = @($revisionRetiredChildren | Where-Object { $_.PSIsContainer } | Sort-Object { $_.FullName.Length } -Descending)
    $revisionRetiredDirectories += $revisionRetiredInfo
    foreach ($revisionEmpty in $revisionRetiredDirectories) {
        if (@(Get-ChildItem -LiteralPath $revisionEmpty.FullName -Force).Count -ne 0) { continue }
        if (-not $DryRun) { Remove-Item -LiteralPath $revisionEmpty.FullName }
        $revisionEmptyRemoved.Add($revisionEmpty.FullName.Substring($revisionPrefix.Length).Replace('\', '/'))
    }
}

$revisionReport = [ordered]@{dry_run = [bool]$DryRun; targets = @($revisionRemoved.ToArray()); empty_retired_directories = @($revisionEmptyRemoved.ToArray()); skipped = @($revisionSkipped.ToArray())}
if (-not $DryRun) {
    $revisionReportDirectory = Join-Path $revisionWorkspace 'output/revision/validation'
    New-Item -ItemType Directory -Force -Path $revisionReportDirectory | Out-Null
    $revisionReport | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $revisionReportDirectory 'workspace_cleanup.json') -Encoding UTF8
}
$revisionReport | ConvertTo-Json -Depth 4
