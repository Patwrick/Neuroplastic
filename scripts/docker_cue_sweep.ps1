param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [int]$Steps = 2000,
    [string]$Seeds = "0,1,2,3,4",
    [int]$K = 32,
    [ValidateSet("uniform", "zipf")]
    [string]$CueDist = "zipf",
    [double]$ZipfAlpha = 1.2,
    [string]$SlotsList = "8",
    [string]$Modes = "static,plastic,slot_static,slot_plastic",
    [int]$ChangeEvery = 0,
    [int]$EpisodeLen = 0,
    [string]$ExtraArgs = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-IntList {
    param(
        [string]$Value,
        [bool]$AllowEmpty = $false
    )
    $items = @()
    foreach ($token in ($Value -split ",")) {
        $trimmed = $token.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        $parsed = 0
        if (-not [int]::TryParse($trimmed, [ref]$parsed)) {
            throw "Invalid integer list value: '$trimmed'"
        }
        $items += $parsed
    }

    if (-not $AllowEmpty -and $items.Count -eq 0) {
        throw "Expected at least one integer value in: '$Value'"
    }
    return $items
}

function Parse-StringList {
    param(
        [string]$Value,
        [bool]$AllowEmpty = $false
    )
    $items = @()
    foreach ($token in ($Value -split ",")) {
        $trimmed = $token.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        $items += $trimmed
    }
    if (-not $AllowEmpty -and $items.Count -eq 0) {
        throw "Expected at least one mode in: '$Value'"
    }
    return $items
}

$AllowedModes = @("static", "plastic", "slot_static", "slot_plastic", "slot_hash_static", "slot_hash_plastic")
$SeedValues = @(Parse-IntList -Value $Seeds -AllowEmpty $false)
$ModeValues = @(Parse-StringList -Value $Modes -AllowEmpty $false)
$SlotValues = @(Parse-IntList -Value $SlotsList -AllowEmpty $true)

foreach ($mode in $ModeValues) {
    if ($AllowedModes -notcontains $mode) {
        throw "Unsupported mode '$mode'. Allowed: $($AllowedModes -join ', ')"
    }
}

if ($Steps -le 0) {
    throw "Steps must be > 0, got $Steps"
}
if ($K -le 0) {
    throw "K must be > 0, got $K"
}
if ($ChangeEvery -lt 0) {
    throw "ChangeEvery must be >= 0, got $ChangeEvery"
}
if ($EpisodeLen -lt 0) {
    throw "EpisodeLen must be >= 0, got $EpisodeLen"
}
foreach ($slot in $SlotValues) {
    if ($slot -le 0) {
        throw "Slots values must be > 0, got $slot"
    }
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ComposeFile = Join-Path $RepoRoot "docker\compose.minestudio.yml"
if (-not (Test-Path -LiteralPath $ComposeFile -PathType Leaf)) {
    throw "Compose file not found. Expected path: $ComposeFile"
}

$OutDir = "/workspace/data/rollouts/$Tag"
$ZipfAlphaText = $ZipfAlpha.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$ResolvedEpisodeLen = $EpisodeLen
if ($ResolvedEpisodeLen -eq 0) {
    $ResolvedEpisodeLen = $Steps
}

$GitCommit = "unknown"
$GitDirty = "unknown"
try {
    $commitRaw = & git -C $RepoRoot rev-parse --short HEAD 2>$null
    if ($LASTEXITCODE -eq 0) {
        $commitText = (($commitRaw | Out-String).Trim())
        if (-not [string]::IsNullOrWhiteSpace($commitText)) {
            $GitCommit = $commitText
        }
    }
} catch {
}

try {
    $statusRaw = & git -C $RepoRoot status --porcelain 2>$null
    if ($LASTEXITCODE -eq 0) {
        $statusText = (($statusRaw | Out-String).Trim())
        if ([string]::IsNullOrWhiteSpace($statusText)) {
            $GitDirty = "0"
        } else {
            $GitDirty = "1"
        }
    }
} catch {
}

$Jobs = @()
foreach ($seed in $SeedValues) {
    foreach ($mode in $ModeValues) {
        if ($mode.StartsWith("slot_")) {
            if ($SlotValues.Count -eq 0) {
                continue
            }
            foreach ($slot in $SlotValues) {
                $Jobs += [PSCustomObject]@{
                    Seed  = $seed
                    Mode  = $mode
                    Slots = $slot
                }
            }
        } else {
            $Jobs += [PSCustomObject]@{
                Seed  = $seed
                Mode  = $mode
                Slots = $null
            }
        }
    }
}

if ($Jobs.Count -eq 0) {
    throw "No runs to execute. Check Modes/SlotsList. Slot modes require a non-empty SlotsList."
}

Write-Host "RepoRoot: $RepoRoot"
Write-Host "ComposeFile: $ComposeFile"
Write-Host "Tag: $Tag"
Write-Host "OutDir: $OutDir"
Write-Host "Total runs: $($Jobs.Count)"
Write-Host "ChangeEvery: $ChangeEvery"
Write-Host "EpisodeLen: $ResolvedEpisodeLen"
Write-Host "GitCommit: $GitCommit"
Write-Host "GitDirty: $GitDirty"
Write-Host "Building Docker image once..."

& docker compose -f $ComposeFile build
if ($LASTEXITCODE -ne 0) {
    throw "docker compose build failed with exit code $LASTEXITCODE"
}

$Index = 0
foreach ($job in $Jobs) {
    $Index += 1
    $cmd = "bash docker/run_with_xvfb.sh python -u scripts/rollout_cue_plastic.py --backend minestudio --steps $Steps --seed $($job.Seed) --mode $($job.Mode) --K $K --cue_dist $CueDist --zipf_alpha $ZipfAlphaText --change_every $ChangeEvery --episode_len $ResolvedEpisodeLen --out_dir '$OutDir'"
    $slotText = "-"
    if ($null -ne $job.Slots) {
        $cmd = "$cmd --slots $($job.Slots)"
        $slotText = [string]$job.Slots
    }
    if (-not [string]::IsNullOrWhiteSpace($ExtraArgs)) {
        $cmd = "$cmd $ExtraArgs"
    }

    Write-Host ("[{0}/{1}] seed={2} mode={3} slots={4}" -f $Index, $Jobs.Count, $job.Seed, $job.Mode, $slotText)
    & docker compose -f $ComposeFile run --rm -e "CSGN_GIT_COMMIT=$GitCommit" -e "CSGN_GIT_DIRTY=$GitDirty" minestudio bash -lc $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Rollout failed at run $Index/$($Jobs.Count) (seed=$($job.Seed), mode=$($job.Mode), slots=$slotText) with exit code $LASTEXITCODE"
    }
}

Write-Host "Sweep complete. Rollouts written to: $OutDir"
