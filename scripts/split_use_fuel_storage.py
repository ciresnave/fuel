#!/usr/bin/env python3
"""Split `use fuel_storage::{ ... };` blocks across fuel_storage and fuel_dispatch.

After the dispatch move, items like baracuda_dispatch::, dispatch::,
kernel::, plan::, fused::, compiled::, pipelined::, cast_fusion::,
cost::, vulkan_dispatch:: live in fuel_dispatch; BackendStorage,
Storage, alloc_cpu_zeroed, from_slice_cpu, dispatch_storage stay in
fuel_storage.

Rewrites each `use fuel_storage::{...};` block, separating items by
their new home.
"""
import re
import sys
from pathlib import Path

# Items that stay in fuel_storage (top-level re-exports).
STORAGE_ITEMS = {
    "BackendStorage",
    "Storage",
    "alloc_cpu_zeroed",
    "from_slice_cpu",
    "dispatch_storage",
    "CudaStorage",
    "VulkanStorage",
    "MetalStorage",
}

# Module paths that moved to fuel_dispatch.
DISPATCH_PREFIXES = (
    "baracuda_dispatch::",
    "vulkan_dispatch::",
    "dispatch::",
    "kernel::",
    "plan::",
    "fused::",
    "compiled::",
    "pipelined::",
    "cast_fusion::",
    "cost::",
)

# Top-level re-exports from fuel_dispatch (also moved).
DISPATCH_TOPLEVEL = {
    "PipelinedExecutor",
    "compile_node",
    "execute_compiled",
    "CompiledNode",
    "KernelBindingTable",
    "KernelDTypes",
    "KernelRef",
    "OpParams",
    "compile_plan",
    "resolve_kernel",
    "ExecutionPlan",
    "NodeKernelBinding",
    "TolerancePolicy",
}


def classify(item: str) -> str:
    """Return 'storage' or 'dispatch' for a use-tree item."""
    stripped = item.strip().rstrip(",").strip()
    if stripped.startswith(DISPATCH_PREFIXES):
        return "dispatch"
    head = stripped.split("::", 1)[0].split("{", 1)[0].strip()
    if head in STORAGE_ITEMS:
        return "storage"
    if head in DISPATCH_TOPLEVEL:
        return "dispatch"
    # Default — unknown items get classified by remaining context.
    return "storage"


def split_inner(inner: str) -> tuple[list[str], list[str]]:
    """Split the inner content of a use-block (without the `use foo::{...};`
    wrapper) into (storage_items, dispatch_items).

    Handles nested braces (e.g. `kernel::{KernelBindingTable, OpParams}`).
    """
    storage = []
    dispatch = []
    depth = 0
    buf = []
    for ch in inner:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                bucket = classify(item)
                (dispatch if bucket == "dispatch" else storage).append(item)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        bucket = classify(tail)
        (dispatch if bucket == "dispatch" else storage).append(tail)
    return storage, dispatch


BLOCK_RE = re.compile(
    r"use fuel_storage::\{([^}]*(?:\{[^}]*\}[^}]*)*)\};",
    re.DOTALL,
)


def rewrite(text: str) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        storage, dispatch = split_inner(inner)
        out = []
        if dispatch:
            joined = ", ".join(dispatch)
            out.append(f"use fuel_dispatch::{{{joined}}};")
        if storage:
            joined = ", ".join(storage)
            out.append(f"use fuel_storage::{{{joined}}};")
        return "\n".join(out)
    return BLOCK_RE.sub(repl, text)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: split_use_fuel_storage.py FILE ...", file=sys.stderr)
        return 2
    changed = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        original = p.read_text(encoding="utf-8")
        new = rewrite(original)
        if new != original:
            p.write_text(new, encoding="utf-8")
            changed += 1
            print(f"rewrote {p}")
    print(f"changed {changed} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
