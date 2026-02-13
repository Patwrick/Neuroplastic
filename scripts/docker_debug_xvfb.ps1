Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker\compose.minestudio.yml"

if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
    throw "Compose file not found. Expected path: $ComposeFile"
}

$Inner = "bash docker/run_with_xvfb.sh /bin/echo OK_XVFB_DIRECT"

Write-Host "RepoRoot: $RepoRoot"
Write-Host "ComposeFile: $ComposeFile"
Write-Host "Inner command: $Inner"

docker compose -f $ComposeFile run --rm minestudio bash -lc "$Inner"
