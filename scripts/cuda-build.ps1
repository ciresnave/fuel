#Requires -Version 5.1
<#
.SYNOPSIS
  cuda-build — machine-wide COUNTED ADMISSION for CUDA-compiling builds (N at a time).

.DESCRIPTION
  Sibling to gpu-run.ps1, deliberately SEPARATE from it. gpu-run serializes
  GPU-TOUCHING RUNS (one at a time, exclusive). This wrapper admits a BOUNDED
  NUMBER of concurrent CUDA BUILDS, which is a different resource with a
  different shape.

  WHY THIS EXISTS — gpu-run.ps1's scope note argues itself wrong
  (co-designed 2026-08-08 across fuel + baracuda; see docs/gaps.md GAP-151):
    gpu-run says "do NOT wrap compile-only (cargo check / clippy / build) —
    those never touch the device and serializing them only wastes wall-clock."
    The PREMISE IS TRUE: nvcc/ptxas never touch the device. The CONCLUSION IS
    FALSE: the binding resource was never the device. It is HOST ADDRESS SPACE
    consumed by K concurrent ptxas, and that resource is contended by BUILDS
    ONLY — exactly the population the rule excludes. A coverage hole with a
    stated, plausible, wrong justification is worse than an unstated one:
    it argues the next reader out of noticing.

  WHY COUNTED ADMISSION AND NOT THE gpu-run MUTEX:
    The resource is DIVISIBLE. Measured on this box: ~16 concurrent nvcc
    survive; an allocation failed near 22. That is a BUDGET, not an exclusive.
    CUDA builds run 10-30 minutes here, so folding them into Global\gpu-run
    would turn a 3-lane project into a 1-lane project for hours in order to
    protect a resource that demonstrably admits more than one holder.

  WHY N MUTEXES AND NOT A Semaphore(N) — THE TRAP:
    The obvious implementation is a named Semaphore. It is WRONG, and it would
    silently destroy the one property gpu-run's robustness rests on.
    Windows Mutex has ABANDONED-HOLDER RECOVERY (AbandonedMutexException).
    Windows Semaphore has NONE. A hard-killed process holding a semaphore count
    leaks that count PERMANENTLY: after N kills the semaphore is permanently
    full and every CUDA build on the machine blocks forever, with no stale
    state to detect and nothing to reclaim. That fails CLOSED and SILENT --
    strictly worse than no coordination at all, since today's failure mode is
    at least a loud ptxas error. N named mutexes give counted admission while
    KEEPING per-slot abandonment recovery.

  ============================ PROTOCOL CONSTANTS ============================
  EVERY participant across every project MUST agree on ALL THREE of these.
  They are a PROTOCOL, not local tuning knobs.

    (1) the slot NAME PATTERN   Global\cuda-build-slot-<i>
    (2) the slot COUNT          $SlotCount
    (3) the per-build THREADS   $ForgeThreads

  *** THE COUNT IS PART OF THE PROTOCOL, AND THIS IS THE NON-OBVIOUS PART. ***
  gpu-run already teaches that a different NAME means no mutual exclusion.
  A shared name with a DIFFERENT COUNT is the same trap in a new dress, and it
  is quieter. If Fuel uses N=2 (slots 0,1) and a peer uses N=3 (slots 0,1,2),
  the peer can always take slot 2 -- which Fuel never contests -- so the
  effective budget silently becomes 3. THE MOST PERMISSIVE PARTICIPANT SETS
  THE BUDGET FOR EVERYONE, and no participant can detect this locally: each
  one's own slots behave exactly as designed. Raising N is a CROSS-PROJECT
  change, never a local one.

  Likewise the real footprint is the PAIR: total concurrent nvcc = N x K.
  A participant that holds one slot but sets BARACUDA_FORGE_THREADS=32 blows
  the budget single-handedly while looking perfectly well-behaved. So this
  wrapper SETS the variable rather than defaulting it, and logs when it is
  overriding a pre-existing value.

  *** THE THIRD MULTIPLIER: C, CONCURRENT FORGE INVOCATIONS PER BUILD. ***
  Found by baracuda 2026-08-08, after N and K were already agreed. The budget
  is not N x K, it is N x C x K.
    ONE `cargo build` can run SEVERAL -sys build scripts CONCURRENTLY:
    baracuda-kernels-sys and baracuda-cutlass-kernels-sys are independent
    crates, so cargo runs both build.rs's at once up to -j, and EACH invokes
    forge at K. Under a SINGLE wrapper slot that is C x K nvcc, not K.
  C is invisible to BOTH parties, which is what makes it dangerous: the
  WRAPPER sees one build, and a per-invocation forge acquire sees each
  invocation as lone. It is the "most permissive participant wins, silently"
  shape one level deeper — nobody is misbehaving and the budget is still
  breached.

  With C=2 (baracuda's two heavy -sys crates), N=2 and K=8 would give 32
  concurrent nvcc against a measured ~22 ceiling. So K WAS LOWERED 8 -> 4:
  N x C x K = 2 x 2 x 4 = 16, which is the measured survivable footprint.
  The alternative on offer was to keep K=8 and DOCUMENT the C=2 overshoot as
  a known limitation. That was refused: this project's own GAP-157 records
  that DOCUMENTATION OF A HAZARD IS NOT A CONTROL FOR IT, and shipping a
  constant known to exceed the ceiling while writing a note about it is
  precisely that defect.

  K RETURNS TO 8 — in lockstep, not unilaterally — once baracuda-forge lands
  its per-slot aggregate cap (concurrent forge invocations sharing one
  CUDA_BUILD_SLOT together honour K regardless of C), since only forge can
  observe C. At that point N x C x K collapses back to N x K.

  *** $SlotCount IS A CONSERVATIVE DEFAULT, NOT A GUARANTEE. ***
  Confirmed by baracuda 2026-08-08: ptxas footprint is KERNEL-DEPENDENT.
  A fused flash-attention or heavily-templated CUTLASS GEMM produces
  intermediate .cudafe1.gpu files of 100-180 MB EACH, so 16 concurrent SMALL
  kernels are fine while 16 concurrent FLASH ptxas can blow past the ceiling.
  A fixed slot count is therefore a FICTION for footprint purposes -- it is a
  pragmatic v1, defensible only because baracuda's ptxas-fatal surfacing
  (62182fce) makes an overflow SELF-DIAGNOSING rather than silent. The
  structural fix is memory-aware admission inside baracuda-forge, which is the
  only party that can see the kernel set; forge participates in THIS SAME slot
  namespace. See GAP-151.

  SCOPE — wrap builds that COMPILE CUDA: cargo build/check/test with
  --features cuda, direct nvcc, anything that pulls baracuda-kernels-sys.
  Do NOT wrap CPU-only builds (they contend for nothing this protects), and do
  NOT wrap GPU-touching RUNS -- those want gpu-run.ps1's exclusive lock. A run
  that both compiles CUDA and then executes on the device wants BOTH, outer
  cuda-build, inner gpu-run (they are different namespaces and do not deadlock).

.PARAMETER Project
  Label recorded in the slot lockfile: fuel | baracuda | vulkane | ...

.PARAMETER Command
  The command + args to run under the slot (everything after `--`).

.EXAMPLE
  pwsh scripts/cuda-build.ps1 -Project fuel -- cargo check -p fuel-cuda-backend --features cuda

.EXAMPLE
  pwsh scripts/cuda-build.ps1 -Project baracuda -- cargo build --features cuda
#>
# No param() block AT ALL — deliberate; same reasoning as gpu-run.ps1. Under
# `pwsh -File`, an advanced binder chokes on the `--` separator and a plain
# param([string]$Project) mis-binds the command's first token positionally.
# Convention: $Project is the FIRST argument.

$ErrorActionPreference = 'Stop'

# ----------------------------- PROTOCOL ------------------------------------
# NEVER change these unilaterally. See the PROTOCOL CONSTANTS block above:
# a divergent count silently widens the budget for everyone.
$SlotPrefix      = 'Global\cuda-build-slot-'   # MUST be Global\ (machine-wide), not Local\
$SlotCount       = 2                           # N — cross-project constant; raising it is a CROSS-PROJECT change
$ForgeThreads    = 4                           # K — see THE THIRD MULTIPLIER below; was 8, lowered 2026-08-08
$CudaBuildVersion = 2                          # bump on behavioral change; logged, never encoded in the name

$LockDir = Join-Path $env:TEMP 'cuda-build'
$LogPath = Join-Path $env:TEMP 'cuda-build.log'

function Write-CbLog([string] $msg) {
    $ts   = (Get-Date).ToString('o')
    $line = "$ts [$Project v$CudaBuildVersion pid=$PID] $msg"
    # RETRY, do not just swallow. A bare try/catch here loses lines EXACTLY when
    # several participants are logging at once — i.e. precisely under the
    # contention this wrapper exists to record. Measured during the 2026-08-08
    # shakedown: two concurrent acquires, two log lines silently dropped. An
    # instrument that fails in the only case it exists for is the defect class
    # this whole protocol is about, so it must not be in the protocol's own log.
    for ($i = 0; $i -lt 12; $i++) {
        try { Add-Content -LiteralPath $LogPath -Value $line -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 25 }
    }
    # Still best-effort at the end: a logging failure must never abort a build.
}

# --- Parse the argument vector by hand (see note above) ---------------------
$argv    = @($args)
$Project = $null
if ($argv.Count -ge 2 -and $argv[0] -eq '-Project') {
    $Project = $argv[1]
    $argv    = @($argv | Select-Object -Skip 2)
}
if ($argv.Count -gt 0 -and $argv[0] -eq '--') {
    $argv = @($argv | Select-Object -Skip 1)
}
if ([string]::IsNullOrWhiteSpace($Project)) {
    throw 'cuda-build: -Project <name> must be the first argument. Usage: cuda-build.ps1 -Project <name> -- <cmd> <args...>'
}
if ($argv.Count -eq 0) {
    throw 'cuda-build: no command given. Usage: cuda-build.ps1 -Project <name> -- <cmd> <args...>'
}
$exe    = $argv[0]
$rest   = @($argv | Select-Object -Skip 1)
$cmdStr = ($argv -join ' ')

# --- Nesting pass-through ---------------------------------------------------
# A nested cuda-build inside a child already holding a slot is a DIFFERENT
# process; without this it would queue behind its own parent and, once all
# slots are held by such pairs, deadlock the whole machine.
if ($env:CUDA_BUILD_HELD -eq '1') {
    Write-CbLog "nested cuda-build — passing through (slot $($env:CUDA_BUILD_SLOT) held by pid=$($env:CUDA_BUILD_HELD_PID)): $cmdStr"
    & $exe @rest
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $LockDir)) {
    New-Item -ItemType Directory -Path $LockDir -Force | Out-Null
}

# --- Acquire ANY free slot --------------------------------------------------
# WaitAny over all slot mutexes: the kernel picks a free one atomically, so
# there is no try-in-turn race and no polling. AbandonedMutexException carries
# .MutexIndex, so a slot whose holder was hard-killed is reclaimed loudly and
# deterministically — the property a Semaphore cannot provide at all.
$mutexes  = @(0..($SlotCount - 1) | ForEach-Object { New-Object System.Threading.Mutex($false, "$SlotPrefix$_") })
$slot     = -1
$t0       = Get-Date
$reclaimed = $false

try {
    $timeoutMin = if ($env:CUDA_BUILD_TIMEOUT_MIN) { [int]$env:CUDA_BUILD_TIMEOUT_MIN } else { 60 }
    $announced  = $false
    while ($slot -lt 0) {
        try {
            $idx = [System.Threading.WaitHandle]::WaitAny($mutexes, 60000)
            if ($idx -ne [System.Threading.WaitHandle]::WaitTimeout) {
                $slot = $idx
                break
            }
        }
        catch [System.Threading.AbandonedMutexException] {
            # Previous holder died mid-build (hard kill / bugcheck). The wait
            # STILL succeeded and we own that slot.
            $slot      = $_.Exception.MutexIndex
            $reclaimed = $true
            break
        }

        $mins = [int]((Get-Date) - $t0).TotalMinutes
        if (-not $announced) {
            $held = @(0..($SlotCount - 1) | ForEach-Object {
                $f = Join-Path $LockDir "slot-$_.json"
                if (Test-Path -LiteralPath $f) { (Get-Content -Raw -LiteralPath $f -ErrorAction SilentlyContinue) } else { '(free?)' }
            }) -join ' | '
            $warn = "cuda-build: all $SlotCount CUDA-build slots busy — WAITING. Holders: $held"
            [Console]::Error.WriteLine($warn); Write-CbLog $warn
            $announced = $true
        }
        else {
            [Console]::Error.WriteLine("cuda-build: still waiting ${mins}m for a CUDA-build slot")
        }
        if ($mins -ge $timeoutMin) {
            # A HUNG holder never abandons its mutex, so an unbounded wait would
            # wedge every CUDA build silently. Abort loud instead.
            $abort = "cuda-build: ABORT after ${mins}m (>= ${timeoutMin}m) — a slot holder appears WEDGED " +
                     "(a hung holder never abandons its mutex). Investigate/kill it, then retry; raise the " +
                     "bound with `$env:CUDA_BUILD_TIMEOUT_MIN."
            [Console]::Error.WriteLine($abort); Write-CbLog $abort
            exit 75   # EX_TEMPFAIL — distinct 'try again later', never a child code
        }
    }

    $staleFile = Join-Path $LockDir "slot-$slot.json"
    $stale = if (Test-Path -LiteralPath $staleFile) {
                 (Get-Content -Raw -LiteralPath $staleFile -ErrorAction SilentlyContinue)
             } else { $null }

    if ($reclaimed) {
        $msg = "RECLAIMED abandoned slot $slot (previous holder died mid-build). Stale: $stale"
        [Console]::Error.WriteLine("cuda-build: $msg"); Write-CbLog $msg
    }
    elseif ($stale) {
        # UNCONTENDED stale-holder detection — ported from gpu-run.ps1 v3, which
        # learned it from baracuda's shakedown (2026-08-05). The
        # AbandonedMutexException path above is loud, but it can ONLY fire when
        # the mutex object OUTLIVES the dead holder — i.e. someone was already
        # waiting. If a holder dies with NOBODY waiting, the last handle closes
        # with it, the kernel object is DESTROYED, and a later New-Object Mutex
        # creates a BRAND-NEW mutex carrying no abandoned state: WaitAny succeeds
        # cleanly and the crash is invisible. Once the object is gone the mutex
        # structurally CANNOT carry the signal, so a surviving lockfile naming a
        # dead pid is the only remaining evidence. Pid reuse can mask it — a
        # false NEGATIVE, never a false alarm.
        $stalePid = $null
        try { $stalePid = ([regex]::Match($stale, '"pid"\s*:\s*(\d+)')).Groups[1].Value } catch {}
        if ($stalePid) {
            $alive = $null
            try { $alive = Get-Process -Id ([int]$stalePid) -ErrorAction Stop } catch { $alive = $null }
            if (-not $alive) {
                $msg = "RECLAIMED stale slot $slot from DEAD pid=$stalePid (uncontended crash — no " +
                       "AbandonedMutexException is raised when the mutex object dies with its last handle). " +
                       "Stale: $stale"
                [Console]::Error.WriteLine("cuda-build: $msg"); Write-CbLog $msg
            }
        }
    }
    $waited = [int]((Get-Date) - $t0).TotalSeconds
    Write-CbLog "acquired slot $slot after ${waited}s"

    # --- Pin the per-build thread count ------------------------------------
    # SET, not default: N x K is the real footprint, so a participant holding
    # one slot with an unbounded pool blows the budget single-handedly while
    # looking well-behaved. Confirmed by baracuda: BARACUDA_FORGE_THREADS is
    # checked FIRST in forge's thread_count() and is an ABSOLUTE early return
    # (`return n.max(1)`), taken BEFORE the calculated.min(NUM_JOBS) fallback —
    # so this yields exactly K per build regardless of -j, and does NOT fight
    # that cap. RAYON_NUM_THREADS is deliberately NOT used: it perturbs every
    # other rayon consumer in the process, not just forge's nvcc pool.
    if ($env:BARACUDA_FORGE_THREADS -and $env:BARACUDA_FORGE_THREADS -ne "$ForgeThreads") {
        $m = "OVERRIDING pre-existing BARACUDA_FORGE_THREADS=$($env:BARACUDA_FORGE_THREADS) with protocol value $ForgeThreads"
        [Console]::Error.WriteLine("cuda-build: $m"); Write-CbLog $m
    }
    $env:BARACUDA_FORGE_THREADS = "$ForgeThreads"

    $meta = [ordered]@{
        v       = 1
        sv      = $CudaBuildVersion
        slot    = $slot
        pid     = $PID
        project = $Project
        threads = $ForgeThreads
        cmd     = $cmdStr
        start   = (Get-Date).ToString('o')
    } | ConvertTo-Json -Compress
    Set-Content -LiteralPath (Join-Path $LockDir "slot-$slot.json") -Value $meta -Encoding UTF8
    Write-CbLog "RUN (slot $slot, $ForgeThreads threads): $cmdStr"

    # --- Run the child HOLDING the slot ------------------------------------
    $env:CUDA_BUILD_HELD     = '1'
    $env:CUDA_BUILD_HELD_PID = "$PID"
    $env:CUDA_BUILD_SLOT     = "$slot"
    # PS 7.4+ with EAP=Stop makes a failing native child throw instead of
    # setting $LASTEXITCODE; disable locally so we capture the REAL exit code.
    $PSNativeCommandUseErrorActionPreference = $false
    & $exe @rest
    $code = $LASTEXITCODE
    Write-CbLog "DONE exit=$code (slot $slot): $cmdStr"
    exit $code
}
finally {
    $env:CUDA_BUILD_HELD     = $null
    $env:CUDA_BUILD_HELD_PID = $null
    $env:CUDA_BUILD_SLOT     = $null
    if ($slot -ge 0) {
        Remove-Item -LiteralPath (Join-Path $LockDir "slot-$slot.json") -Force -ErrorAction SilentlyContinue
        try { $mutexes[$slot].ReleaseMutex() } catch {}
        Write-CbLog "released slot $slot"
    }
    foreach ($m in $mutexes) { $m.Dispose() }
}
