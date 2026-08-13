<#
.SYNOPSIS
    Compiles the FEATURE COMBINATIONS that no other gate builds.
    (GAP-173; found as a GAP-097 residual)

.DESCRIPTION
    Every gate this project runs builds features ONE AT A TIME:

        CI  rust-ci.yml   cargo check --workspace
                          cargo check --workspace --features metal
                          cargo test  --workspace
        CI  ci_cuda.yaml  cargo test --features cuda
        local             aarch64-cross-check.ps1  (default features)

    Each feature separately. NEVER two at once — and `telemetry` appears in no
    CI job at all. That is a structural hole, not an oversight in one sweep, and
    `fuel-dispatch/src/telemetry/baracuda_provider.rs` was the first thing to
    fall through it: `mod telemetry` is `#[cfg(feature = "telemetry")]`, and
    `mod baracuda_provider` is `#[cfg(feature = "cuda")]` INSIDE it, so the file
    is parsed ONLY under the COMBINATION. It sat with a missing `DType::F8E5M2`
    arm — a hard E0004 — through an entire dtype sweep, because
    `--all-targets` is not `--all-features`, and one feature is not two.

    ── WHAT THIS GATE'S CORRECTNESS DEPENDS ON, AND WHAT IT DOES NOT ──

    It depends on the COMBINATION LIST below being right. It does NOT depend on
    knowing which match sites exist.

    That distinction is the whole design, so do not undo it by maintaining a
    site inventory alongside this script. A site list decays with every commit;
    a combination list changes only when someone edits a `[features]` table.

    It also means this gate SUBSUMES the blind spot of the scan that found the
    combinations. That scan was regex-and-brace based, so it could not see
    matches generated inside macros (this repo has `cpu_cast_wrapper!`-style
    dtype macros, so that blind spot is real and probably non-empty). THE
    COMPILER EXPANDS MACROS. Compiling the combination catches an E0004 inside
    an expansion without anything ever having to find it by reading source.

    ── ASK THE CHEAP QUESTION FIRST: WHICH *SINGLE* NON-DEFAULT FEATURES ARE
       BUILT BY NOTHING? ──

    This script was motivated by an exotic case (`cuda`+`telemetry`) and named
    for combinations, and that framing buried a simpler and more productive
    question. `telemetry` is a SINGLE non-default feature that appears in NO CI
    job, so the entire telemetry module — eight-plus files — was compiled by
    nothing at all. That is a far more ordinary hiding place than a two-feature
    intersection, and it is cheaper to check: the `telemetry` leg needs no
    CUDA, no SDK, no slot, and no forge, and runs in seconds anywhere.

    So in any future audit: enumerate the SINGLE non-default features first and
    ask which have no gate. Only then go looking for intersections.

    (Note on what that leg does NOT defend, so nobody over-credits it: Fuel's
    wire-boundary dtype exhaustiveness — a new `DType` being a compile error
    before it can reach a structure_key — is NOT enforced here. It used to live
    in `telemetry/structure_key_derive.rs::dtype_token` behind this very gate;
    consolidating the sk4 token table into `fuel_ir::sk4_token` moved it into
    an UNGATED file, so a plain `cargo check -p fuel-ir` now enforces it. That
    was a side effect of deduplication, not a goal, and it is worth knowing
    because it is the reason no wildcard-free match requires `telemetry` alone
    today. Measured: zero such sites.)

    ⚠️ THE LIST IS HAND-MAINTAINED, AND THAT MEANS IT ROTS.
    Adding a cargo feature does NOT add a combination here. When it isn't
    added, this gate keeps passing while silently covering less of the
    workspace — the same failure mode as an allowlist key that outlived the
    file it named. The non-rotting version derives the combination set from the
    workspace `[features]` tables; that is a deliberate later increment, and
    the declared-vs-derived tension is GAP-161's question one level up.

    ⚠️ AND A FEATURE BEING NON-DEFAULT IN ITS OWN CRATE DOES NOT MAKE IT
    UNBUILT. `fuel-core/Cargo.toml` enables `capture` on
    `fuel-correctness-fixtures`, so that crate's gated code compiles under a
    plain `-p fuel-core`. Anything that later derives this list must resolve
    the feature GRAPH, not per-crate `default` tables — an earlier version of
    the scan got exactly this wrong and over-reported a site as invisible.

.NOTES
    ── WHY THE CHECKS ARE ORDERED THE WAY THEY ARE ──

    Exit code FIRST, then artifacts. A failed compile and a never-attempted
    compile BOTH produce no artifact, so an artifact-first check reports a real
    E0004 as "the compiler never ran" — a true statement that points the reader
    at the harness instead of at their code. aarch64-cross-check.ps1 carries the
    same note; it was learned there by sabotage.

    Artifacts come from `--message-format json` (`compiler-artifact`), NOT the
    human-readable "Checking <crate>" lines, because cargo does not reprint
    those when every unit is fresh. A warm cache would otherwise turn this into
    a false RED, and a standing gate that reds falsely gets disabled.

    For the CUDA leg, an exit code of EITHER polarity is evidence about the
    harness until an artifact proves the compiler ran: this box has produced a
    `--features cuda` run that returned 0 having compiled nothing, and red runs
    that never reached cargo at all. So that leg additionally requires
    vcvarsall's own `Environment initialized for:` banner.
#>
[CmdletBinding()]
param(
    # Skip the CUDA leg (needs MSVC + the machine-wide cuda-build slot).
    [switch]$SkipCuda
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

# ===========================================================================
# THE COMBINATION LIST. This, not a site list, is what must be kept correct.
# ===========================================================================
#   crate       : the -p target
#   features    : the combination no other gate forms
#   cuda        : needs MSVC env + the machine-wide cuda-build slot
#   why         : reported on failure so a red is self-explaining
$combinations = @(
    @{
        crate    = 'fuel-dispatch'
        features = 'telemetry'
        cuda     = $false
        why      = '`telemetry` appears in NO CI job, so the whole telemetry module (8+ files) is compiled by nothing. Seconds, no CUDA, machine-independent.'
    },
    @{
        crate    = 'fuel-ir'
        features = 'dlpack'
        cuda     = $false
        why      = 'fuel-ir `default = []` and nothing enables `dlpack`, so no CI job compiles dlpack/convert.rs.'
    },
    @{
        crate    = 'fuel-dispatch'
        features = 'telemetry,baracuda-types'
        cuda     = $false
        why      = '`mod telemetry` is cfg(telemetry) and `mod baracuda_provider` is cfg(baracuda-types) INSIDE it. CI builds each feature separately, so this file is never parsed.'
    }
)

function Get-VcVarsAll {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $root = & $vswhere -latest -products * -property installationPath 2>$null | Select-Object -First 1
        if ($root) {
            $p = Join-Path $root 'VC\Auxiliary\Build\vcvarsall.bat'
            if (Test-Path $p) { return $p }
        }
    }
    # Do NOT hand-set NVCC_CCBIN as a fallback: it sets the compiler without a
    # matching INCLUDE/LIB, giving a mixed-toolset environment whose failures
    # surface deep in the stdlib and read like a CUDA bug (CLAUDE.md).
    return $null
}

function Invoke-PlainLeg($combo) {
    $out = & cargo check -p $combo.crate --features $combo.features --all-targets --message-format json 2>$null
    return @{ exit = $LASTEXITCODE; json = $out; banner = $true }
}

function Invoke-CudaLeg($combo) {
    $vc = Get-VcVarsAll
    if (-not $vc) { throw 'feature-combination-check: vcvarsall.bat not found (vswhere).' }

    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) "fuel-featcombo-$PID"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $bat = Join-Path $tmp 'leg.bat'
    $log = Join-Path $tmp 'leg.log'

    # The vcvarsall path is quoted INSIDE the .bat so no quotes ride on the
    # invoking command line. No `if errorlevel` guard and no internal
    # redirection: both have been observed to kill a run silently between
    # vcvarsall and cargo on this box.
    @"
@echo off
cd /d "$repo"
call "$vc" amd64
cargo check -p $($combo.crate) --features $($combo.features) --all-targets --message-format json
echo FEATCOMBO_EXIT=%ERRORLEVEL%
"@ | Set-Content -Path $bat -Encoding ASCII

    # cuda-build.ps1 owns BARACUDA_FORGE_THREADS (protocol K) and the
    # machine-wide slot; do not set threads here or the N x K budget breaks.
    & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'cuda-build.ps1') -Project fuel -- cmd /c "$bat" *> $log

    $lines = Get-Content $log -ErrorAction SilentlyContinue
    $exit = 1
    $m = $lines | Select-String -Pattern 'FEATCOMBO_EXIT=(\d+)' | Select-Object -Last 1
    if ($m) { $exit = [int]$m.Matches[0].Groups[1].Value }
    $banner = [bool]($lines | Select-String -Pattern 'Environment initialized for:')
    return @{ exit = $exit; json = $lines; banner = $banner; log = $log }
}

$failed = @()
foreach ($combo in $combinations) {
    $label = "$($combo.crate) --features $($combo.features)"
    if ($combo.cuda -and $SkipCuda) {
        Write-Host "SKIP: $label (-SkipCuda)" -ForegroundColor Yellow
        Write-Host '      NOTE: skipping is not passing. This leg is the one the gate exists for.'
        continue
    }
    Write-Host "=== $label ===" -ForegroundColor Cyan
    $r = if ($combo.cuda) { Invoke-CudaLeg $combo } else { Invoke-PlainLeg $combo }

    $diagnostics = @()
    $sawArtifact = $false
    foreach ($line in $r.json) {
        if (-not $line -or $line[0] -ne '{') { continue }
        try { $msg = $line | ConvertFrom-Json } catch { continue }
        if ($msg.reason -eq 'compiler-artifact' -and $msg.package_id -match [regex]::Escape($combo.crate)) {
            $sawArtifact = $true
        }
        if ($msg.reason -eq 'compiler-message' -and $msg.message.rendered) {
            if ($msg.message.level -eq 'error') { $diagnostics += $msg.message.rendered }
        }
    }
    foreach ($d in $diagnostics) { Write-Host $d }

    # ORDER: exit code first (see .NOTES).
    if ($r.exit -ne 0) {
        Write-Host "FAIL: $label exited $($r.exit)." -ForegroundColor Red
        Write-Host "      why this combination is gated: $($combo.why)" -ForegroundColor Yellow
        $failed += $label
        continue
    }
    if ($combo.cuda -and -not $r.banner) {
        Write-Host "FAIL: $label returned 0 but vcvarsall never initialised." -ForegroundColor Red
        Write-Host '      An exit code of EITHER polarity is evidence about the harness' -ForegroundColor Red
        Write-Host '      until an artifact proves the compiler ran.' -ForegroundColor Red
        $failed += $label
        continue
    }
    if (-not $sawArtifact) {
        Write-Host "FAIL: $label succeeded but produced no artifact for $($combo.crate)." -ForegroundColor Red
        Write-Host '      A clean exit over a crate that was never compiled says nothing.' -ForegroundColor Red
        $failed += $label
        continue
    }
    Write-Host "PASS: $label" -ForegroundColor Green
}

Write-Host ''
if ($failed.Count -gt 0) {
    Write-Host "FAIL: $($failed.Count) combination(s): $($failed -join '; ')" -ForegroundColor Red
    exit 1
}
Write-Host 'PASS: every listed feature combination compiles.' -ForegroundColor Green
Write-Host '      This proves the COMBINATIONS LISTED compile — not that the list is'
Write-Host '      complete. Adding a cargo feature requires adding a combination here,'
Write-Host '      or this gate silently covers less of the workspace.' -ForegroundColor Yellow
exit 0
