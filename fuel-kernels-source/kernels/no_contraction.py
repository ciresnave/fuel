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
import subprocess
import sys
from pathlib import Path

ARITH = re.compile(r"^\s*(%\S+)\s*=\s*Op(FAdd|FSub|FMul)\b")
DECORATED = re.compile(r"^\s*OpDecorate\s+(%\S+)\s+NoContraction\b")
TYPE_DECL = re.compile(r"^\s*%\S+\s*=\s*OpType")


def decorate(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    already = {m.group(1) for ln in lines if (m := DECORATED.match(ln))}
    ids = [m.group(1) for ln in lines if (m := ARITH.match(ln)) and m.group(1) not in already]
    if not ids:
        return text, 0
    first_type = next(i for i, ln in enumerate(lines) if TYPE_DECL.match(ln))
    dec = [f"               OpDecorate {i} NoContraction" for i in ids]
    return "\n".join(lines[:first_type] + dec + lines[first_type:]) + "\n", len(ids)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: no_contraction.py path/to/file.spv", file=sys.stderr)
        return 1
    spv = Path(sys.argv[1])
    if not spv.is_file():
        print(f"error: {spv} not found", file=sys.stderr)
        return 1
    dis = subprocess.run(
        ["spirv-dis", "--no-color", str(spv)], check=True, capture_output=True, text=True
    )
    fixed, n = decorate(dis.stdout)
    if n == 0:
        return 0
    subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.1", "-o", str(spv), "-"],
        check=True,
        input=fixed,
        text=True,
        capture_output=True,
    )
    val = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.1", str(spv)], capture_output=True, text=True
    )
    if val.returncode != 0:
        print("spirv-val failed after no_contraction:", val.stderr, file=sys.stderr)
        return val.returncode
    print(f"    no-contraction: decorated {n} results in {spv.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
