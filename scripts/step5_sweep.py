"""
Step 5 sweep: replace `CpuStorage` (fuel_core_types alias) with `HostBuffer`
across the workspace.

Rules:
- In fuel-cpu-backend/src/dyn_impl.rs: SKIP entirely (already uses HostBuffer correctly)
- In fuel-cpu-backend/src/lib.rs: SKIP the `pub use dyn_impl::CpuStorage` line
- In any other file: replace all `CpuStorage` that refers to the type alias with `HostBuffer`,
  and `CpuStorageRef` with `HostBufferRef`.
- Also update import lines: `CpuStorage` in imports from fuel_core_types -> `HostBuffer`
- Keep method names like `to_cpu_storage`, `storage_from_cpu_storage` etc. (they come from
  static BackendStorage/BackendDevice traits being removed in step 8; don't rename them now
  to avoid churn).

Special cases:
- fuel-core-types/src/cpu_storage.rs: keep the alias definition line intact
- fuel-core-types/src/lib.rs: keep `pub use cpu_storage::{..., CpuStorage, CpuStorageRef, ...}`
  (backward compat re-export stays until step 9)
- fuel-cpu-backend/src/lib.rs: keep `pub use dyn_impl::CpuStorage` as-is
"""

import re
import os
import sys

ROOT = "c:/Users/cires/OneDrive/Documents/projects/fuel"

# Files to skip entirely (already correct or must not be touched)
SKIP_FILES = {
    os.path.normpath(f"{ROOT}/fuel-cpu-backend/src/dyn_impl.rs"),
    os.path.normpath(f"{ROOT}/fuel-core-types/src/cpu_storage.rs"),  # definition file
}

# Files to skip specific lines (keep certain re-exports intact)
KEEP_LINES_CONTAINING = {
    os.path.normpath(f"{ROOT}/fuel-core-types/src/lib.rs"): [
        "pub use cpu_storage::",  # keep the backward-compat re-export
    ],
    os.path.normpath(f"{ROOT}/fuel-cpu-backend/src/lib.rs"): [
        "pub use dyn_impl::CpuStorage",  # keep re-export of the newtype
    ],
    os.path.normpath(f"{ROOT}/fuel-core/src/lib.rs"): [
        "pub use cpu_backend::{CpuStorage",  # keep backward-compat re-export
    ],
    os.path.normpath(f"{ROOT}/fuel-core/src/cpu_backend/mod.rs"): [
        "pub use fuel_core_types::{CpuDevice, CpuStorage",  # keep re-export
    ],
}

def should_skip_line(filepath, line):
    keep = KEEP_LINES_CONTAINING.get(filepath, [])
    return any(k in line for k in keep)

def process_file(filepath):
    norm = os.path.normpath(filepath)
    if norm in SKIP_FILES:
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return False

    changed = False
    new_lines = []
    for line in lines:
        if should_skip_line(norm, line):
            new_lines.append(line)
            continue

        new_line = line

        # Replace `CpuStorageRef` before `CpuStorage` to avoid partial match issues
        new_line = new_line.replace("CpuStorageRef", "HostBufferRef")

        # Replace `CpuStorage` — but only where it refers to the type alias.
        # We use word-boundary replacement. The new fuel_cpu_backend::CpuStorage
        # newtype is only referenced via qualified path `fuel_cpu_backend::...` or
        # imported explicitly. In fuel-core-types and fuel-core, all bare `CpuStorage`
        # references are the alias.
        # Exception: don't replace in string literals (function names like "to_cpu_storage")
        # We'll do a simple word-boundary replace and accept that method name strings
        # like `"to_cpu_storage"` also get renamed (they're in error messages, not critical).
        # Actually: method NAMES in trait bodies (fn to_cpu_storage) should NOT be renamed
        # since we'll delete those traits in step 8. Use a regex that only matches type
        # positions (after : , < > ( ) = &) and import paths, not fn names.

        # Strategy: replace `CpuStorage` as a type/value but not as part of a function name.
        # Function names in Rust are snake_case; `CpuStorage` only appears as PascalCase type.
        # So a word-boundary replace is safe — there's no fn named `CpuStorage`.
        new_line = re.sub(r'\bCpuStorage\b', 'HostBuffer', new_line)

        if new_line != line:
            changed = True
        new_lines.append(new_line)

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return True
    return False

# Gather files to process
crates = [
    "fuel-core-types/src",
    "fuel-core/src",
    "fuel-core/tests",
    "fuel-cpu-backend/src",
    "fuel-graph-cuda/src",
    "fuel-metal/src",
    "fuel-nn/src",
    "fuel-transformers/src",
    "fuel-examples/examples",
    "fuel-flash-attn-cuda/src",
    "fuel-flash-attn-v3-cuda/src",
]

total_changed = 0
for crate_dir in crates:
    full_dir = os.path.join(ROOT, crate_dir)
    if not os.path.exists(full_dir):
        continue
    for dirpath, _, filenames in os.walk(full_dir):
        for fname in filenames:
            if not fname.endswith('.rs'):
                continue
            fpath = os.path.join(dirpath, fname)
            if process_file(fpath):
                rel = os.path.relpath(fpath, ROOT)
                print(f"  changed: {rel}")
                total_changed += 1

print(f"\nTotal files changed: {total_changed}")
