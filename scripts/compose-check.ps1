<#
.SYNOPSIS
  Does YOUR branch still pass its gates once the PRs ahead of you land?

.DESCRIPTION
  A rebase answers "do these MERGE". This answers "do these COMPOSE".

  Two PRs can each be green, share no files, produce no textual conflict, and
  still interact: #49 asserted that no allowlist entry outlives its edge, #53
  removed the edge. Neither touched the other's files, so nothing in review,
  rebase or merge could surface it -- it appeared only after one landed and
  reddened the other's CI. Cost: a full CI cycle plus a wrong diagnosis.

  This script is that check as one command: merge the other heads into a
  throwaway worktree off your branch, run a named gate set, tear everything
  down unconditionally.

.PARAMETER Mine
  Your branch or ref. Never modified -- the scratch worktree is detached.

.PARAMETER Against
  Heads landing before you. A bare number N is fetched as `pull/N/head`;
  anything else is used as a git ref verbatim.

.PARAMETER Gate
  Commands to run after merging, in order. Defaults to the dependency-direction
  gate, which is the fast structural one this check exists for.

  It runs the gates UNCONDITIONALLY. It deliberately does not pre-filter on
  whether the branches share files: a file-overlap matrix answers "will git
  conflict", which is what a rebase already answers -- #49 and #53 touched NO
  file in common and still broke each other. Any "no shared files, skip the
  gates" short-circuit removes the only thing this tool adds.

  It also reports no file counts. If you add any, diff against
  `git merge-base main <head>`, never against `main`: the main-relative form
  attributes MAIN's changes to the PR, and gets worse the further behind the PR
  is -- exactly when you are running this.

  !! THIS IS NOT CI. It runs the gates you name and NOTHING else -- no clippy,
  no fmt, no full test suite, no other platform. A green compose-check means
  "these compose under the named gates", never "this PR is green".

.OUTPUTS
  Exit 0  COMPOSES        merged cleanly, every gate passed
  Exit 2  CONFLICT        does not compose TEXTUALLY -- a result, not an error
  Exit 3  GATE FAILED     merged cleanly, a gate went red -- the semantic case
  Exit 1  HARNESS         the check itself could not run; verdict is UNKNOWN

  2 and 3 are reported separately on purpose: they look identical from outside
  and demand different responses.

.EXAMPLE
  pwsh scripts/compose-check.ps1 -Mine my-branch -Against 53
  pwsh scripts/compose-check.ps1 -Mine my-branch -Against 53,54 -Gate 'cargo test -p fuel-ir'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]   $Mine,
    [Parameter(Mandatory)][string[]] $Against,
    [string[]] $Gate = @('cargo test -q -p fuel-ir --test crate_dependency_direction'),
    [string]   $Remote = 'origin'
)

$ErrorActionPreference = 'Stop'

# The body lives in a function DELIBERATELY. `return` inside a `try` at SCRIPT
# scope ends the SCRIPT -- skipping the trailing `exit`, so the process exits 0
# while the console shows "GATE FAILED". The born-red caught exactly that: a
# correct diagnosis with a success exit code, which is the failure this tool
# exists to help people avoid. Inside a function, `return` yields a value.
function Invoke-ComposeCheck {
    param($Mine, $Against, $Gate, $Remote)

$repo = (git rev-parse --show-toplevel 2>$null)
if (-not $repo) { Write-Host 'HARNESS: not inside a git repository'; return 1 }

$scratch = Join-Path ([System.IO.Path]::GetTempPath()) ("compose-check-" + [guid]::NewGuid().ToString('N').Substring(0, 12))
$made = $false
$verdict = 1
$detail = 'did not run'

try {
    # Resolve each target. A bare number is a PR; fetch its head.
    $refs = @()
    foreach ($a in $Against) {
        if ($a -match '^#?(\d+)$') {
            $n = $Matches[1]
            git fetch -q $Remote "pull/$n/head" 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) { Write-Host "HARNESS: cannot fetch pull/$n/head"; return 1 }
            # Resolve to a FULL SHA here, in the main worktree. FETCH_HEAD is
            # PER-WORKTREE: merging 'FETCH_HEAD' inside the scratch worktree
            # fails with "could not open ... FETCH_HEAD", which this script
            # previously reported as a textual CONFLICT -- a confident wrong
            # verdict. Objects are shared; FETCH_HEAD is not.
            $full = (git rev-parse FETCH_HEAD)
            if (-not $full) { Write-Host "HARNESS: pull/$n/head fetched but did not resolve"; return 1 }
            $refs += @{ Name = "#$n"; Ref = $full; Sha = $full.Substring(0, 8) }
        }
        else {
            $sha = (git rev-parse --short=8 $a 2>$null)
            if (-not $sha) { Write-Host "HARNESS: cannot resolve ref '$a'"; return 1 }
            $refs += @{ Name = $a; Ref = $a; Sha = $sha }
        }
    }

    # Detached worktree: no branch is created, so nothing can be left holding one.
    git worktree add -q --detach $scratch $Mine 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "HARNESS: worktree add failed for '$Mine'"; return 1 }
    $made = $true
    Write-Host ("compose-check: {0} @ {1}" -f $Mine, (git -C $scratch rev-parse --short=8 HEAD))

    foreach ($r in $refs) {
        git -C $scratch merge --no-edit -q $r.Ref 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            # A failed merge is not automatically a conflict. Conflicted paths
            # are the discriminator: none means the merge never ran (bad ref,
            # unrelated histories, dirty tree) and the verdict is UNKNOWN, not
            # "does not compose". Reporting those alike is the same error as
            # reading an exit code without an artifact.
            $conflicted = @(git -C $scratch diff --name-only --diff-filter=U 2>$null)
            git -C $scratch merge --abort 2>&1 | Out-Null
            if ($conflicted.Count -eq 0) {
                $detail = "HARNESS: merge of $($r.Name) ($($r.Sha)) failed with NO conflicted paths -- the merge did not run. Verdict UNKNOWN."
                $verdict = 1; return 1
            }
            $conflicted | ForEach-Object { Write-Host "    conflict: $_" }
            $detail = "CONFLICT with $($r.Name) ($($r.Sha)) in $($conflicted.Count) file(s) -- these do not compose TEXTUALLY"
            # `return 2`, never a bare `return`: a bare return inside try/finally
            # exits the function with NO value, so `exit $null` becomes exit 0 --
            # a correct diagnosis printed alongside a success exit code. That is
            # the exact failure this tool exists to catch, and it happened here
            # twice before the born-red pinned it.
            $verdict = 2; return 2
        }
        Write-Host ("  merged {0} ({1})" -f $r.Name, $r.Sha)
    }

    foreach ($g in $Gate) {
        Write-Host "  gate: $g"
        Push-Location $scratch
        try { $out = & cmd /c "$g 2>&1"; $code = $LASTEXITCODE } finally { Pop-Location }
        if ($code -ne 0) {
            $out | Select-Object -Last 15 | ForEach-Object { Write-Host "    $_" }
            $detail = "GATE FAILED: $g (exit $code) -- they merge, and the result is red"
            $verdict = 3; return 3
        }
        $out | Where-Object { $_ -match 'test result|dep-direction' } | ForEach-Object { Write-Host "    $_" }
    }
    $verdict = 0; $detail = "COMPOSES: merged " + (($refs | ForEach-Object { $_.Name }) -join ', ') + ", all gates passed"
}
finally {
    # UNCONDITIONAL. A leaked worktree holds a branch and makes
    # `gh pr merge --delete-branch` fail later, blamed on something else.
    if ($made) {
        git worktree remove --force $scratch 2>&1 | Out-Null
        git worktree prune 2>&1 | Out-Null
    }
    if (Test-Path $scratch) { Remove-Item -Recurse -Force $scratch -ErrorAction SilentlyContinue }
    Write-Host ""
    Write-Host $detail
    # Say which of the three actually happened. "removed" when none was ever
    # created is a false statement in an artifact, and this tool exists to stop
    # exactly that.
    $cleanup = if (-not $made) { 'none created' }
               elseif (Test-Path $scratch) { 'LEAKED -- investigate' }
               else { 'removed' }
    Write-Host "cleanup: worktree $cleanup"
    }
    return $verdict
}

exit (Invoke-ComposeCheck -Mine $Mine -Against $Against -Gate $Gate -Remote $Remote)
