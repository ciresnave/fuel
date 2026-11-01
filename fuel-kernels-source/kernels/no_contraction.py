#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Post-process a SPIR-V file from slangc to decorate every OpFAdd, OpFSub and OpFMul result with
NoContraction, so a driver may not fuse `a * b + c` into one fused multiply-add.

Why: slangc has no way to emit NoContraction (`-fp-mode precise` and the `precise` qualifier were
measured to emit none on the pinned Vulkan SDK). The double-double kernels (`unary_f64.slang`) rely on
`a * b` and `a + b` rounding separately; llvmpipe (Lavapipe) contracts them, which moved sin, cos and
sin(1e10) off the reference (ROADMAP row 134 follow-up; fuel#566 measured the three tests passing once
the decoration is present).

A kernel opts in with a `// no-contraction` comment in its first 40 lines (see compile.sh). The
script is idempotent: ids already decorated are skipped, and a module with nothing to add is left
untouched.

Usage:
    python no_contraction.py path/to/file.spv

Modifies in place. Requires spirv-dis, spirv-as and spirv-val on PATH (Vulkan SDK).
"""

import re
import shutil
import subprocess  # nosec B404 - runs three fixed Vulkan SDK tools by argv list, no shell; developer-run regeneration script
import sys
from pathlib import Path

ARITH = re.compile(r"^\s*(%\S+)\s*=\s*Op(FAdd|FSub|FMul)\b")
DECORATED = re.compile(r"^\s*OpDecorate\s+(%\S+)\s+NoContraction\b")
TYPE_DECL = re.compile(r"^\s*%\S+\s*=\s*OpType")


def _first_group(pattern: re.Pattern, lines: list[str]) -> list[str]:
    return [m.group(1) for m in map(pattern.match, lines) if m]


def decorate(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    already = set(_first_group(DECORATED, lines))
    ids = [i for i in _first_group(ARITH, lines) if i not in already]
    if not ids:
        return text, 0
    first_type = next(i for i, ln in enumerate(lines) if TYPE_DECL.match(ln))
    dec = [f"               OpDecorate {i} NoContraction" for i in ids]
    return "\n".join(lines[:first_type] + dec + lines[first_type:]) + "\n", len(ids)


def _tool(name: str) -> str:
    """Full path of an SDK tool on PATH (so subprocess never gets a bare name)."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"error: {name} not found on PATH (install the Vulkan SDK)")
    return path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _checked_spv(arg: str) -> Path:
    """The one path this script touches: an existing `.spv` file inside this repository.

    Resolved first (so `..` and symlinks cannot walk out), then required to end in `.spv` and to lie
    under the repository root. Anything else is refused, so the path handed to the SDK tools below is
    never an arbitrary caller-chosen file.
    """
    path = Path(arg).resolve()
    if path.suffix != ".spv" or not path.is_file():
        raise SystemExit(f"error: {arg} is not an existing .spv file")
    if REPO_ROOT not in path.parents:
        raise SystemExit(f"error: {path} is outside the repository ({REPO_ROOT})")
    return path


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run one SDK tool. argv is a list built from `_tool(<fixed name>)`, fixed flags and the
    validated `.spv` path from `_checked_spv`; no shell is involved."""
    # nosec B603 - argv list, no shell; executable resolved by shutil.which from a fixed name
    # nosemgrep - the only variable argument is the path validated by _checked_spv
    return subprocess.run(argv, **kwargs)  # nosec B603  # nosemgrep


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: no_contraction.py path/to/file.spv", file=sys.stderr)
        return 1
    spv = _checked_spv(sys.argv[1])
    dis = _run([_tool("spirv-dis"), "--no-color", str(spv)], check=True, capture_output=True, text=True)
    fixed, n = decorate(dis.stdout)
    if n == 0:
        return 0
    _run(
        [_tool("spirv-as"), "--target-env", "vulkan1.1", "-o", str(spv), "-"],
        check=True,
        input=fixed,
        text=True,
        capture_output=True,
    )
    val = _run([_tool("spirv-val"), "--target-env", "vulkan1.1", str(spv)], capture_output=True, text=True)
    if val.returncode != 0:
        print("spirv-val failed after no_contraction:", val.stderr, file=sys.stderr)
        return val.returncode
    print(f"    no-contraction: decorated {n} results in {spv.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
