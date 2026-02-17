param(
    [int]$Steps = 200,
    [int]$Seed = 0,
    [ValidateSet("plastic", "static", "slot_static", "slot_plastic", "slot_hash_static", "slot_hash_plastic")]
    [string]$Mode = "plastic",
    [string]$OutDir = "/workspace/data/rollouts",
    [string]$ExtraArgs = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker\compose.minestudio.yml"

if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
    throw "Compose file not found. Expected path: $ComposeFile"
}

$Inner = "bash docker/run_with_xvfb.sh python -u scripts/rollout_cue_plastic.py --backend minestudio --steps $Steps --seed $Seed --mode $Mode --out_dir '$OutDir'"
if (-not [string]::IsNullOrWhiteSpace($ExtraArgs)) {
    $Inner = "$Inner $ExtraArgs"
}

Write-Host "RepoRoot: $RepoRoot"
Write-Host "ComposeFile: $ComposeFile"
Write-Host "Inner command: $Inner"

docker compose -f $ComposeFile build
docker compose -f $ComposeFile run --rm minestudio bash -lc "$Inner"
