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

# ─── SECOND PASS: the baseline this gate was BLIND TO ───────────────────────
#
# NOTE 3, added after this script reported PASS on fuel-quantized while CI
# failed on the same crate, same target, same rustc version.
#
# The pass above runs under `.cargo/config.toml`'s
# `[target.aarch64-apple-darwin] rustflags = ["-C", "target-cpu=generic"]`.
# `generic` is ARMv8.0-A and does NOT include `dotprod`, so anything behind
# `#[cfg(target_feature = "dotprod")]` is NEVER COMPILED by that pass.
#
# Apple Silicon enables `dotprod` in its BASELINE, and the macOS CI job DELETES
# `.cargo/config.toml` (the ring-crate workaround) — so CI compiles the OPPOSITE
# arm from the one this gate was checking. `fuel-quantized/src/neon.rs`'s
# hardware SDOT path reached main that way and broke both macOS jobs with E0658
# while this script said PASS.
#
# The config comment justifying `generic` argues NEON is baseline-mandatory in
# ARMv8 so the NEON paths still compile. That is TRUE of baseline NEON and
# FALSE of dotprod-gated NEON: a correct reason covering strictly less than the
# claim it was supporting.
#
# So run it again with the Apple-Silicon feature set — same crates, same
# target, the other arm.
# ⚠️ AND THE SECOND BLIND SPOT, which a sabotage run exposed: the pass above
# runs on the DEFAULT toolchain, which on this box is nightly. CI runs STABLE.
# `core::arch::aarch64::vdotq_s32` is an unstable library feature, so it
# compiles clean on nightly and fails E0658 on stable — meaning a +dotprod pass
# on nightly STILL reports PASS on the exact defect that broke CI. Re-adding
# the defect and re-running proved it: PASS, exit 0.
#
# A gate that does not match CI's TOOLCHAIN is as blind as one that does not
# match its FEATURE SET. This pass must pin both.
$stableOk = $true
# NOTE: rustup's EXIT CODE is deliberately NOT read. It answers a different
# question from the one this gate asks: rustup can exit non-zero for reasons
# that say nothing about the toolchain -- notably a failed SELF-UPDATE when
# another session on this box holds rustup.exe, which with many concurrent
# sessions is the steady state rather than a race (diagnosed in vulkane). The
# check below asserts the PROPERTY this gate depends on -- is the target
# present -- and subsumes a genuine failure, because a failed rustup emits no
# target list and 2>$null keeps stderr out of the variable. Restoring an
# exit-code test would add no detection and one false-red path.
$stableTargets = & rustup +stable target list --installed 2>$null
if ($stableTargets -notcontains $target) { $stableOk = $false }
if (-not $stableOk) {
    Write-Host ''
    Write-Host "FAIL: the stable toolchain with target '$target' is required." -ForegroundColor Red
    Write-Host '      CI compiles this crate on STABLE, and unstable-feature errors' -ForegroundColor Red
    Write-Host '      (E0658) are INVISIBLE on nightly. Without it this gate cannot' -ForegroundColor Red
    Write-Host '      see the class of defect it exists to catch.' -ForegroundColor Red
    Write-Host "      rustup toolchain install stable; rustup +stable target add $target" -ForegroundColor Yellow
    exit 2
}

$env:CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS = '-C target-feature=+dotprod'
Write-Host ''
Write-Host 'second pass (STABLE toolchain, +dotprod — exactly what the macOS runner compiles)'
$json2 = & cargo +stable @cargoArgs 2>$null
$exit2 = $LASTEXITCODE
$diag2 = @()
$seen2 = @{}
foreach ($line in $json2) {
    if (-not $line) { continue }
    try { $m = $line | ConvertFrom-Json } catch { continue }
    if ($m.reason -eq 'compiler-message' -and $m.message.rendered) { $diag2 += $m.message.rendered }
    if ($m.reason -eq 'compiler-artifact') {
        foreach ($c in $crates) { if ($m.package_id -match [regex]::Escape($c)) { $seen2[$c] = $true } }
    }
}
foreach ($d in $diag2) { Write-Host $d }
# Same ordering rule as the first pass: EXIT CODE before artifacts, because a
# failed compile and a never-attempted one both produce no artifact.
if ($exit2 -ne 0) {
    Write-Host ''
    Write-Host "FAIL: cargo exited $exit2 for $target with +dotprod." -ForegroundColor Red
    Write-Host '      This is the arm the macOS runner compiles; a PASS from the' -ForegroundColor Red
    Write-Host '      first pass alone does NOT cover it.' -ForegroundColor Red
    exit $exit2
}
$missing2 = $crates | Where-Object { -not $seen2.ContainsKey($_) }
if ($missing2.Count -gt 0) {
    Write-Host ''
    Write-Host "FAIL: +dotprod pass produced no artifact for: $($missing2 -join ', ')" -ForegroundColor Red
    exit 3
}

Write-Host "PASS: $($crates -join ', ') type-check for $target." -ForegroundColor Green
Write-Host '      BOTH baselines: target-cpu=generic AND +dotprod (Apple Silicon).'
Write-Host '      Reached: parse, resolution, type-check, exhaustiveness, borrow-check.'
Write-Host '      NOT reached: linking, codegen, runtime. NEON correctness is UNVERIFIED'
Write-Host '      by this gate and needs real hardware.'
Write-Host '      fuel-metal-backend: COMPILE-correct is not KERNEL-correct. No Metal'
Write-Host '      hardware here and no Metal runtime test exists, so this says its types'
Write-Host '      line up — NOT that any shader computes the right answer.' -ForegroundColor Yellow
exit 0
