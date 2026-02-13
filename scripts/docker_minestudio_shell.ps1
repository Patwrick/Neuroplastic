Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker\compose.minestudio.yml"

if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
    throw "Compose file not found. Expected path: $ComposeFile"
}

Write-Host "RepoRoot: $RepoRoot"
Write-Host "ComposeFile: $ComposeFile"

docker compose -f $ComposeFile run --rm minestudio bash
