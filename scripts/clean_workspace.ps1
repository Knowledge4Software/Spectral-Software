param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$revisionWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$revisionPrefix = $revisionWorkspace.TrimEnd('\') + '\'
$revisionTargets = [System.Collections.Generic.List[string]]::new()
$revisionSkipped = [System.Collections.Generic.List[string]]::new()

# Visit only project code. Never descend into a directory link or the venv.
foreach ($revisionSource in @('create_datasets_graphs', 'experiments', 'pipelines', 'scripts', 'spectral_code', 'tests')) {
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
$revisionScratch = Join-Path $revisionWorkspace 'tmp'
if (Test-Path -LiteralPath $revisionScratch) {
    foreach ($revisionChild in (Get-ChildItem -LiteralPath $revisionScratch -Directory)) {
        if ($revisionChild.Name -match '^(review_pytest|revision_.+)$') {
            $revisionTargets.Add($revisionChild.FullName)
        }
    }
}

$revisionRemoved = [System.Collections.Generic.List[string]]::new()
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

$revisionReport = [ordered]@{dry_run = [bool]$DryRun; targets = @($revisionRemoved.ToArray()); skipped = @($revisionSkipped.ToArray())}
if (-not $DryRun) {
    $revisionReportDirectory = Join-Path $revisionWorkspace 'output/review'
    New-Item -ItemType Directory -Force -Path $revisionReportDirectory | Out-Null
    $revisionReport | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $revisionReportDirectory 'workspace_cleanup.json') -Encoding UTF8
}
$revisionReport | ConvertTo-Json -Depth 4
