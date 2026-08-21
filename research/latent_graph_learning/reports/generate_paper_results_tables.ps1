param(
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$reportDirectory = Split-Path -Parent $PSCommandPath
$repositoryRoot = (Resolve-Path (Join-Path $reportDirectory '..\..\..')).Path
$artifactDirectory = Join-Path $reportDirectory 'artifacts'
$repositoryCsv = Join-Path $artifactDirectory 'wide_test_metrics.csv'
$pretrainedCsv = Join-Path $artifactDirectory 'pretrained_collaborator_results.csv'
$paperFiles = @(
    (Join-Path $repositoryRoot 'paper\spectra.tex'),
    (Join-Path $repositoryRoot 'paper\spectra_manuscript.tex')
)

$benchmarks = @(
    'BigCloneBench',
    'SemanticCloneBench',
    'GPTCloneBench',
    'ATCoder'
)
$metrics = @('P', 'R', 'F1', 'Acc')

function Format-Metric([object]$value) {
    return ([double]::Parse([string]$value, $culture)).ToString('0.0000', $culture)
}

function Get-RepositoryMetric([object]$row, [string]$benchmark, [string]$metric) {
    $propertyName = "$benchmark`n$metric"
    $property = $row.PSObject.Properties[$propertyName]
    if ($null -eq $property) {
        throw "Missing column '$propertyName' in $repositoryCsv"
    }
    return Format-Metric $property.Value
}

function Add-CommonTableHeader(
    [System.Collections.Generic.List[string]]$lines,
    [string]$caption,
    [string]$label
) {
    $lines.Add('\begin{table*}[t]')
    $lines.Add('\centering')
    $lines.Add("\caption{$caption}")
    $lines.Add("\label{$label}")
    $lines.Add('\scriptsize')
    $lines.Add('\setlength{\tabcolsep}{2.2pt}')
    $lines.Add('\renewcommand{\arraystretch}{1.08}')
    $lines.Add('\resizebox{\textwidth}{!}{%')
    $lines.Add('\begin{tabular}{l*{16}{c}}')
    $lines.Add('\toprule')
    $lines.Add('Method & \multicolumn{4}{c}{\shortstack{BigCloneBench\\(CodeXGLUE)}} & \multicolumn{4}{c}{SemanticCloneBench} & \multicolumn{4}{c}{GPTCloneBench} & \multicolumn{4}{c}{ATCoder} \\')
    $lines.Add('\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}')
    $lines.Add(' & P & R & F1 & Acc & P & R & F1 & Acc & P & R & F1 & Acc & P & R & F1 & Acc \\')
    $lines.Add('\midrule')
}

function Add-TableFooter([System.Collections.Generic.List[string]]$lines) {
    $lines.Add('\bottomrule')
    $lines.Add('\end{tabular}%')
    $lines.Add('}')
    $lines.Add('\end{table*}')
}

function Get-RepositoryTable([object[]]$rows) {
    $lines = [System.Collections.Generic.List[string]]::new()
    Add-CommonTableHeader $lines 'Test performance of methods executed by the repository pipeline. All values are read from the archived final benchmark artifact; P, R, and Acc denote precision, recall, and accuracy.' 'tab:results_baselines_overall'

    $familyNames = @{
        'GNN Baselines' = 'Observed-graph GNN baselines'
        'Non-graph Code Baselines' = 'Token/tree code baselines'
        'Other Graph-based Learning Methods' = 'Other graph-based learning methods'
        'Our Method' = 'Proposed method'
        'Spectral Representation Baselines' = 'Fixed spectral-representation baselines'
    }
    $previousFamily = $null
    foreach ($row in $rows) {
        if ($row.Family -ne $previousFamily) {
            if ($null -ne $previousFamily) {
                $lines.Add('\addlinespace[2pt]')
            }
            $displayFamily = $familyNames[$row.Family]
            $lines.Add("\multicolumn{17}{l}{\textit{$displayFamily}} \\")
            $previousFamily = $row.Family
        }

        $values = [System.Collections.Generic.List[string]]::new()
        foreach ($benchmark in $benchmarks) {
            foreach ($metric in $metrics) {
                $values.Add((Get-RepositoryMetric $row $benchmark $metric))
            }
        }
        $method = if ($row.Method -eq 'SPECTRA-Siam') { '\textbf{SPECTRA-Siam}' } else { $row.Method }
        $lines.Add(('{0} & {1} \\' -f $method, ($values -join ' & ')))
    }
    Add-TableFooter $lines
    return $lines
}

function Get-PretrainedTable([object[]]$rows) {
    $lines = [System.Collections.Generic.List[string]]::new()
    Add-CommonTableHeader $lines 'Test performance of independently implemented pretrained baselines supplied in the collaborator benchmark report. These values are retained separately because the corresponding training code and run metadata are not stored in this repository.' 'tab:pretrained_results_overall'

    $methodOrder = @(
        'CodeBERT + No Train', 'CodeBERT + RF', 'CodeBERT + SNN',
        'CodeBERT + PCA + RF', 'CodeBERT + PCA + SNN',
        'GraphCodeBERT + No Train', 'GraphCodeBERT + RF', 'GraphCodeBERT + SNN',
        'GraphCodeBERT + PCA + RF', 'GraphCodeBERT + PCA + SNN',
        'UniXcoder + No Train', 'UniXcoder + RF', 'UniXcoder + SNN',
        'UniXcoder + PCA + RF', 'UniXcoder + PCA + SNN',
        'CodeT5 + No Train', 'CodeT5 + RF', 'CodeT5 + SNN',
        'CodeT5 + PCA + RF', 'CodeT5 + PCA + SNN'
    )
    $encoderNames = @('CodeBERT', 'GraphCodeBERT', 'UniXcoder', 'CodeT5')

    for ($methodIndex = 0; $methodIndex -lt $methodOrder.Count; $methodIndex++) {
        if (($methodIndex % 5) -eq 0) {
            if ($methodIndex -gt 0) {
                $lines.Add('\addlinespace[2pt]')
            }
            $encoder = $encoderNames[[int]($methodIndex / 5)]
            $lines.Add("\multicolumn{17}{l}{\textit{$encoder}} \\")
        }

        $method = $methodOrder[$methodIndex]
        $values = [System.Collections.Generic.List[string]]::new()
        foreach ($benchmark in $benchmarks) {
            $sourceRow = $rows | Where-Object {
                $_.Benchmark -eq $benchmark -and $_.Method -eq $method
            }
            if (@($sourceRow).Count -ne 1) {
                throw "Expected exactly one pretrained row for '$benchmark' / '$method'."
            }
            foreach ($metric in $metrics) {
                $values.Add((Format-Metric $sourceRow.$metric))
            }
        }
        $lines.Add(('{0} & {1} \\' -f $method, ($values -join ' & ')))
    }
    Add-TableFooter $lines
    return $lines
}

function Get-BestRepositoryRow([object[]]$rows, [string]$benchmark) {
    $key = "$benchmark`nF1"
    return $rows | Sort-Object { [double]$_.PSObject.Properties[$key].Value } -Descending | Select-Object -First 1
}

function Get-BestPretrainedRow([object[]]$rows, [string]$benchmark) {
    return $rows | Where-Object Benchmark -eq $benchmark | Sort-Object { [double]$_.F1 } -Descending | Select-Object -First 1
}

function Get-ResultsBlock([object[]]$repositoryRows, [object[]]$pretrainedRows) {
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add('\section{Experimental Results}')
    $lines.Add('\label{sec:exp_results}')
    $lines.Add('')
    $lines.Add('The result tables distinguish two provenance classes. Controlled repository')
    $lines.Add('runs are read from \texttt{wide\_test\_metrics.csv}; independently implemented')
    $lines.Add('pretrained baselines are transcribed from the collaborator benchmark report and')
    $lines.Add('archived in \texttt{pretrained\_collaborator\_results.csv}. BigCloneBench in')
    $lines.Add('the tables denotes the stored CodeXGLUE binary-clone split. Separating the two')
    $lines.Add('sources preserves the pretrained results without implying that their missing')
    $lines.Add('runtime and training metadata were reproduced by this repository.')
    $lines.Add('')
    $lines.Add('\subsection{RQ1: Controlled Repository Results}')
    $lines.Add('\label{sec:rq}')
    $lines.Add('')
    $lines.Add('Table~\ref{tab:results_baselines_overall} reports the test metrics available in')
    $lines.Add('the final repository artifact. The fixed-spectrum rows directly test whether a')
    $lines.Add('learned latent graph improves over spectra of predefined AST, CFG, DDG, and CPG')
    $lines.Add('views; the remaining rows provide controlled code-representation baselines.')
    $lines.Add('')
    foreach ($line in (Get-RepositoryTable $repositoryRows)) { $lines.Add($line) }
    $lines.Add('')
    $lines.Add('\subsection{RQ2: Pretrained and Overall Comparison}')
    $lines.Add('\label{sec:rq2}')
    $lines.Add('')
    $lines.Add('Table~\ref{tab:pretrained_results_overall} retains all pretrained configurations')
    $lines.Add('reported by the collaborator, including direct similarity, RF, SNN, and')
    $lines.Add('PCA-controlled variants. No missing pretrained score is replaced by a zero or')
    $lines.Add('inferred from another configuration.')
    $lines.Add('')
    foreach ($line in (Get-PretrainedTable $pretrainedRows)) { $lines.Add($line) }
    $lines.Add('')
    $lines.Add('\begin{table*}[t]')
    $lines.Add('\centering')
    $lines.Add('\caption{Highest observed test F1 within each result source, with \method{} shown separately. ``Repository best'' is selected only from Table~\ref{tab:results_baselines_overall}; ``pretrained best'' is selected only from Table~\ref{tab:pretrained_results_overall}.}')
    $lines.Add('\label{tab:best_results_by_source}')
    $lines.Add('\small')
    $lines.Add('\begin{tabularx}{\textwidth}{lXcXcc}')
    $lines.Add('\toprule')
    $lines.Add('Dataset & Repository best & F1 & Pretrained best & F1 & \method{} F1 \\')
    $lines.Add('\midrule')

    $spectraRow = $repositoryRows | Where-Object Method -eq 'SPECTRA-Siam'
    foreach ($benchmark in $benchmarks) {
        $repositoryBest = Get-BestRepositoryRow $repositoryRows $benchmark
        $pretrainedBest = Get-BestPretrainedRow $pretrainedRows $benchmark
        $spectraF1 = Get-RepositoryMetric $spectraRow $benchmark 'F1'
        $repositoryF1 = Get-RepositoryMetric $repositoryBest $benchmark 'F1'
        $displayBenchmark = if ($benchmark -eq 'BigCloneBench') { 'BigCloneBench (CodeXGLUE)' } else { $benchmark }
        $lines.Add(('{0} & {1} & {2} & {3} & {4} & {5} \\' -f
            $displayBenchmark,
            $repositoryBest.Method,
            $repositoryF1,
            $pretrainedBest.Method,
            (Format-Metric $pretrainedBest.F1),
            $spectraF1))
    }
    $lines.Add('\bottomrule')
    $lines.Add('\end{tabularx}')
    $lines.Add('\end{table*}')
    $lines.Add('')
    $lines.Add('These are descriptive single-run comparisons. The current artifacts do not')
    $lines.Add('contain repeated-seed distributions or paired significance tests, so the table')
    $lines.Add('supports observed-score comparisons rather than statistical-superiority claims.')
    $lines.Add('')
    return ($lines -join "`r`n") + "`r`n`r`n"
}

function Get-DatasetBlock {
    return @'
\subsection{Datasets and Split Integrity}
\label{subsec:datasets}

The experiments use four prepared V3 benchmark bundles. The counts in
Table~\ref{tab:datasets} are read from the metadata consumed by the graph
pipeline; they are therefore the counts of the actual experimental splits,
not headline totals from a different release of a benchmark. In each split,
``all / clone'' gives the number of labelled pairs and the number of positive
clone pairs; the number of non-clone pairs is their difference.

\begin{table*}[t]
\centering
\caption{Prepared benchmark composition before graph-coverage filtering. Pair
counts preserve the supplied train/validation/test partitions.}
\label{tab:datasets}
\small
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lllrrr}
\toprule
Dataset & Languages & Fragments & \shortstack{Train pairs\\all / clone} &
\shortstack{Validation pairs\\all / clone} & \shortstack{Test pairs\\all / clone} \\
\midrule
BigCloneBench (CodeXGLUE) & Java & $9{,}126$ & $901{,}028 / 450{,}862$ & $415{,}416 / 53{,}839$ & $415{,}416 / 56{,}820$ \\
AtCoder & Java, Python & $43{,}143$ & $544{,}054 / 272{,}027$ & $116{,}704 / 58{,}352$ & $116{,}736 / 58{,}368$ \\
GPTCloneBench & C, C\#, Java, Python & $5{,}924$ & $12{,}432 / 6{,}216$ & $2{,}658 / 1{,}329$ & $2{,}682 / 1{,}341$ \\
SemanticCloneBench & C, C\#, Java, Python & $8{,}000$ & $5{,}836 / 2{,}918$ & $1{,}084 / 542$ & $1{,}080 / 540$ \\
\bottomrule
\end{tabular}
\end{table*}

\paragraph{BigCloneBench (CodeXGLUE).}
The Java benchmark uses the stored official CodeXGLUE split without resampling
or relabelling. The final \method{} run retains
$900{,}850/415{,}416/415{,}416$ graph-evaluable train/validation/test pairs.

\paragraph{AtCoder.}
The prepared problem-disjoint split contains accepted Java and Python
solutions. Positive Java--Python pairs solve the same problem, and negative
pairs solve different problems. The final \method{} run retains
$544{,}054/116{,}704/116{,}736$ graph-evaluable pairs.

\paragraph{GPTCloneBench.}
The multilingual prepared split is author-group safe. It contains
$17{,}772$ labelled pairs over $5{,}924$ unique endpoints and is class
balanced in each partition. The final \method{} run retains
$12{,}335/2{,}619/2{,}663$ graph-evaluable pairs.

\paragraph{SemanticCloneBench.}
The semantic-group-disjoint benchmark contains $2{,}000$ fragments in each
of C, C\#, Java, and Python. Its $8{,}000$ labelled pairs are class balanced
within each supplied partition. The final \method{} run retains
$5{,}825/1{,}082/1{,}078$ graph-evaluable pairs.

Pairs are retained for a method only when both endpoints have every required
representation. Each result artifact records the resulting split sizes. When
coverage differs between methods, a strict pairwise claim requires their common
evaluable pair IDs or an explicit disclosure of the differing sample sizes.

'@
}

function Test-PaperStructure([string]$paperFile, [string]$text, [string]$expectedResultsBlock) {
    if (-not $text.Contains($expectedResultsBlock.TrimEnd())) {
        throw "Generated result block was not inserted exactly in $paperFile"
    }

    $withoutComments = [regex]::Replace($text, '(?m)(?<!\\)%.*$', '')
    $labels = @([regex]::Matches($withoutComments, '\\label\{([^}]+)\}') | ForEach-Object {
        $_.Groups[1].Value
    })
    $duplicateLabels = @($labels | Group-Object | Where-Object Count -gt 1 | ForEach-Object Name)
    if ($duplicateLabels.Count -gt 0) {
        throw "Duplicate labels in $paperFile`: $($duplicateLabels -join ', ')"
    }

    $definedLabels = [System.Collections.Generic.HashSet[string]]::new([string[]]$labels)
    $references = @([regex]::Matches($withoutComments, '\\(?:auto|eq|page|v)?ref\{([^}]+)\}') | ForEach-Object {
        $_.Groups[1].Value
    } | Sort-Object -Unique)
    $missingReferences = @($references | Where-Object { -not $definedLabels.Contains($_) })
    if ($missingReferences.Count -gt 0) {
        throw "Undefined references in $paperFile`: $($missingReferences -join ', ')"
    }

    $environmentStack = [System.Collections.Generic.Stack[string]]::new()
    foreach ($match in [regex]::Matches($withoutComments, '\\(begin|end)\{([^}]+)\}')) {
        $kind = $match.Groups[1].Value
        $environment = $match.Groups[2].Value
        if ($kind -eq 'begin') {
            $environmentStack.Push($environment)
        }
        else {
            if ($environmentStack.Count -eq 0) {
                throw "Unexpected \\end{$environment} in $paperFile"
            }
            $opened = $environmentStack.Pop()
            if ($opened -ne $environment) {
                throw "Environment mismatch in $paperFile`: opened '$opened', closed '$environment'"
            }
        }
    }
    if ($environmentStack.Count -gt 0) {
        throw "Unclosed environments in $paperFile`: $($environmentStack.ToArray() -join ', ')"
    }

    $resultsStart = $text.IndexOf('\section{Experimental Results}')
    $resultsEnd = $text.IndexOf('\subsection{RQ3}', $resultsStart)
    if ($resultsStart -lt 0 -or $resultsEnd -lt 0) {
        throw "Could not isolate the generated results in $paperFile"
    }
    $resultsText = $text.Substring($resultsStart, $resultsEnd - $resultsStart)
    if ($resultsText.Contains('XXXXX') -or $resultsText.Contains('& 0.00 &')) {
        throw "A placeholder remains in the generated result tables in $paperFile"
    }
}

$repositoryRows = @(Import-Csv -LiteralPath $repositoryCsv)
$pretrainedRows = @(Import-Csv -LiteralPath $pretrainedCsv)
if ($repositoryRows.Count -ne 28) {
    throw "Expected 28 repository result rows, found $($repositoryRows.Count)."
}
if ($pretrainedRows.Count -ne 80) {
    throw "Expected 80 pretrained result rows, found $($pretrainedRows.Count)."
}

$resultsBlock = Get-ResultsBlock $repositoryRows $pretrainedRows
$resultsPattern = '(?s)\\section\{Experimental Results\}\r?\n\\label\{sec:exp_results\}.*?(?=\\subsection\{RQ3\})'
$datasetPattern = '(?s)\\subsection\{Datasets(?: and Split Integrity)?\}\r?\n\\label\{subsec:datasets\}.*?(?=\\subsection\{Baselines\})'

foreach ($paperFile in $paperFiles) {
    $text = [System.IO.File]::ReadAllText($paperFile)
    if (-not [regex]::IsMatch($text, $resultsPattern)) {
        throw "Could not locate the replaceable results block in $paperFile"
    }
    $updated = [regex]::Replace($text, $resultsPattern, { param($match) $resultsBlock }, 1)

    if ((Split-Path -Leaf $paperFile) -eq 'spectra.tex') {
        if (-not [regex]::IsMatch($updated, $datasetPattern)) {
            throw "Could not locate the replaceable dataset block in $paperFile"
        }
        $datasetBlock = (Get-DatasetBlock) + "`r`n"
        $updated = [regex]::Replace($updated, $datasetPattern, { param($match) $datasetBlock }, 1)
    }

    Test-PaperStructure $paperFile $updated $resultsBlock

    if ($Check) {
        Write-Output "Validated replacement targets for $paperFile"
    }
    else {
        [System.IO.File]::WriteAllText(
            $paperFile,
            $updated,
            [System.Text.UTF8Encoding]::new($false)
        )
        Write-Output "Updated $paperFile"
    }
}
