param(
    [string[]]$Papers = @(
        "paper0_position",
        "paper1_standalone_pra"
    ),
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PapersRoot = Join-Path $RepoRoot "docs\papers"

foreach ($paper in $Papers) {
    $paperDir = Join-Path $PapersRoot $paper
    $texPath = Join-Path $paperDir "paper.tex"
    if (-not (Test-Path $texPath)) {
        throw "Missing paper.tex for $paper at $texPath"
    }

    Push-Location $paperDir
    try {
        if ($Clean) {
            latexmk -C paper.tex | Out-Host
        }
        latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex | Out-Host
    }
    finally {
        Pop-Location
    }
}
