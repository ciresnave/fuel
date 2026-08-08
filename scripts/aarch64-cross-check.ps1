<#
.SYNOPSIS
    Type-checks Fuel's aarch64 code paths from any host, without a Mac and
    without a GPU. (GAP-162)

.DESCRIPTION
    `cargo check --target aarch64-apple-darwin` compiles for aarch64 WITHOUT
    LINKING. That is the whole trick: the Apple-only parts of the dependency
    graph (objc2, the Metal frameworks) are LINK-time dependencies, so a
    non-linking check gets all the way through type-checking on a Windows or
    Linux box.

    It exists because of what it would have caught. `DType::I8` was added
    2026-05-19 and `fuel-metal-backend` never got its match arms — silently
    non-exhaustive for roughly three months, because NOTHING on any developer
    box compiles that code. A gate that compiles it would have failed on day
    one. The same invisibility later hid an F8E5M2 gap in the same file
    (GAP-160). Code nothing compiles is code nothing checks.

.NOTES
    ── WHICH PHASES THIS REACHES, and why that sentence is here ──

    A compile line proves the compiler reached the CRATE, not that it reached
    the PHASE that answers your question. This gate was built after exactly
    that mistake: a run of `-p fuel-metal-backend` reported `0 x E0004` while
    dying in NAME RESOLUTION, which runs BEFORE exhaustiveness checking — the
    zero meant "never asked", not "nothing wrong". A second layer of the same:
    a PARSE error (`gen`, a reserved keyword in edition 2024) fires even inside
    a `#[cfg(...)]`-disabled block, because parsing precedes cfg-stripping.

    REACHES:        parse . macro expansion . name resolution . type-check .
                    MATCH EXHAUSTIVENESS (E0004) . borrow-check
    DOES NOT REACH: monomorphization of uninstantiated generics . codegen .
                    LINKING . anything at runtime

    So this gate CANNOT tell you:
      * that the aarch64 build LINKS (it never links);
      * that NEON kernels are numerically CORRECT — compiling a NEON path and
        that path computing the right answer are different claims, and only
        real hardware settles the second;
      * that a crate which fails EARLIER is exhaustiveness-clean. If this
        reports errors, an `E0004: 0` alongside them is a LOWER BOUND, not a
        pass.

    ── TWO IMPLEMENTATION NOTES THAT ARE NOT COSMETIC ──

    1. FRESHNESS. The artifact check parses `--message-format json` for
       `compiler-artifact` entries, NOT the human-readable "Checking <crate>"
       lines. A first draft of this gate used those lines and FAILED on a warm
       cache, because cargo does not reprint them when every unit is fresh —
       a false RED, which is how a standing gate gets disabled. Cargo does
       still emit `compiler-artifact` (with `fresh: true`), so JSON is the
       cache-robust form.

    2. RUSTFLAGS. `.cargo/config.toml` sets `[build] rustflags =
       ["-C", "target-cpu=native"]`, which cargo applies to CROSS-COMPILES too.
       `native` resolves to the HOST cpu (e.g. znver4) and is then handed to an
       aarch64 compilation, emitting thousands of "not a recognized processor /
       feature" lines. They are ignored by LLVM and harmless to correctness,
       but they bury real diagnostics — so this script overrides rustflags for
       this target only. A host cpu-tuning flag is meaningless for a
       cross-target regardless of the noise.

    Scope is the crates that are GREEN. `fuel-metal-backend` was excluded while
    it was ~26 errors deep against a stale `Layout` strides API, and ADDED once
    repaired (GAP-160/164) — a standing gate that is RED ON ARRIVAL gets
    disabled within a week, which is worse than having no gate at all.

    ── AND ONE LIMIT SPECIFIC TO METAL ──

    `fuel-metal-backend` compiling is NOT `fuel-metal-backend` working. There is
    no Metal hardware on this box and no Metal runtime test anywhere in the
    repo, so a green here says its types line up — nothing about whether a
    single shader computes the right answer. The crate went years without
    compiling; it has never been executed by this project's CI at all. Treat a
    PASS as "the rot is gone", never as "Metal is supported".
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$target = 'aarch64-apple-darwin'
$crates = @('fuel-cpu-kernels', 'fuel-quantized', 'fuel-cpu-backend', 'fuel-metal-backend')

$installed = & rustup target list --installed
if ($installed -notcontains $target) {
    Write-Host "FAIL: rust target '$target' is not installed." -ForegroundColor Red
    Write-Host "      rustup target add $target"
    exit 2
}

# See NOTE 2: keep the host's `target-cpu=native` out of a cross compile.
$env:CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS = ''

$cargoArgs = @('check')
foreach ($c in $crates) { $cargoArgs += @('-p', $c) }
$cargoArgs += @('--target', $target, '--all-targets', '--message-format', 'json')

Write-Host "cargo $($cargoArgs -join ' ')"
Write-Host ''

# stdout carries the JSON stream; LLVM's target-cpu chatter goes to stderr and
# is dropped deliberately (see NOTE 2).
$json = & cargo @cargoArgs 2>$null
$cargoExit = $LASTEXITCODE

$seen = @{}
$diagnostics = @()
foreach ($line in $json) {
    if (-not $line) { continue }
    try { $m = $line | ConvertFrom-Json } catch { continue }
    switch ($m.reason) {
        'compiler-artifact' {
            # package_id shapes vary across cargo versions; match by substring.
            foreach ($c in $crates) { if ($m.package_id -match [regex]::Escape($c)) { $seen[$c] = $true } }
        }
        'compiler-message' {
            if ($m.message.rendered) { $diagnostics += $m.message.rendered }
        }
    }
}

foreach ($d in $diagnostics) { Write-Host $d }

# ORDER MATTERS HERE, and a sabotage run is what proved it.
#
# A failed compile and a never-attempted compile BOTH produce no
# `compiler-artifact`. A first version checked artifacts first and therefore
# reported a real E0004 in fuel-quantized as "the compiler did not reach these
# crates" — a true statement about artifacts, and a wrong diagnosis pointing
# the reader at the harness instead of at their code.
#
# So: a nonzero cargo exit is a COMPILE FAILURE and is reported as one. Missing
# artifacts on a SUCCESSFUL run is the silent-skip case this gate exists for,
# and only that combination earns the "did not reach" message.
$e0004 = ($diagnostics | Where-Object { $_ -match 'E0004' }).Count
if ($cargoExit -ne 0) {
    Write-Host ''
    Write-Host "FAIL: cargo exited $cargoExit for $target (E0004 seen: $e0004)." -ForegroundColor Red
    Write-Host '      Treat that E0004 count as a LOWER BOUND: a fn that fails to' -ForegroundColor Yellow
    Write-Host '      type-check may never reach exhaustiveness checking at all.' -ForegroundColor Yellow
    $unbuilt = $crates | Where-Object { -not $seen.ContainsKey($_) }
    if ($unbuilt.Count -gt 0) {
        Write-Host "      Not built (failed, or blocked by a failed dependency): $($unbuilt -join ', ')" -ForegroundColor Yellow
    }
    exit $cargoExit
}

# ARTIFACT CHECK: cargo says success — so each target crate must actually have
# produced an artifact. "A build happened" is not "THESE crates were checked".
$missing = $crates | Where-Object { -not $seen.ContainsKey($_) }
if ($missing.Count -gt 0) {
    Write-Host ''
    Write-Host "FAIL: cargo succeeded but produced no artifact for: $($missing -join ', ')" -ForegroundColor Red
    Write-Host '      The compiler did not reach these crates, so a clean exit says' -ForegroundColor Red
    Write-Host '      nothing about them. This is the failure this gate exists for.' -ForegroundColor Red
    exit 3
}

Write-Host "PASS: $($crates -join ', ') type-check for $target." -ForegroundColor Green
Write-Host '      Reached: parse, resolution, type-check, exhaustiveness, borrow-check.'
Write-Host '      NOT reached: linking, codegen, runtime. NEON correctness is UNVERIFIED'
Write-Host '      by this gate and needs real hardware.'
Write-Host '      fuel-metal-backend: COMPILE-correct is not KERNEL-correct. No Metal'
Write-Host '      hardware here and no Metal runtime test exists, so this says its types'
Write-Host '      line up — NOT that any shader computes the right answer.' -ForegroundColor Yellow
exit 0
