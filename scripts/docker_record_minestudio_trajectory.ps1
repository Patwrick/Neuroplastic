param(
    [int]$Steps = 500,
    [int]$Seed = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker\compose.minestudio.yml"

if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
    throw "Compose file not found. Expected path: $ComposeFile"
}

$Inner = "bash docker/run_with_xvfb.sh python -u scripts/record_trajectory.py --backend minestudio --steps $Steps --seed $Seed --log_every 25 --out_dir /workspace/data/trajectories"

Write-Host "RepoRoot: $RepoRoot"
Write-Host "ComposeFile: $ComposeFile"
Write-Host "Inner command: $Inner"

docker compose -f $ComposeFile build
docker compose -f $ComposeFile run --rm minestudio bash -lc "$Inner"
