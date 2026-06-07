#!/usr/bin/env python3
"""Add the cpu_dev() helper near the start of a test module / test file."""

import re
import sys

HELPER = '''
    /// Phase 7.5 G2: tests need a real device for slot-populating
    /// constructors. Singleton CpuBackendDevice via OnceLock.
    fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_core_types::DynBackendDevice> {
        static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_core_types::DynBackendDevice>>
            = std::sync::OnceLock::new();
        D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }
'''

HELPER_TOPLEVEL = '''
/// Phase 7.5 G2: tests need a real device for slot-populating
/// constructors. Singleton CpuBackendDevice via OnceLock.
fn cpu_dev() -> &'static std::sync::Arc<dyn fuel_core_types::DynBackendDevice> {
    static D: std::sync::OnceLock<std::sync::Arc<dyn fuel_core_types::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| std::sync::Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}
'''


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    if 'fn cpu_dev' in src:
        print(f"{path}: cpu_dev already present, skipping")
        return
    # Heuristic: if there's a `mod tests {` and cpu_dev is used inside it,
    # insert the helper just after `mod tests {` opening brace (and `use super::*;`
    # if present).
    m = re.search(r'mod\s+tests\s*\{', src)
    if m:
        # Insert HELPER right after the opening `{`. Walk forward to skip any
        # `use ...;` lines that come right after.
        insert_pos = m.end()
        # If the next line is `use super::*;`, skip past it.
        # Actually let's just insert right after the brace; super::*; is fine.
        new = src[:insert_pos] + HELPER + src[insert_pos:]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
        print(f"{path}: helper inserted into mod tests")
        return
    # Otherwise — top-level test file. Insert at the top after any
    # `//!` doc comments and `use` lines.
    # Find the last `use` line; insert after it (and any blank line).
    # Simpler: find first occurrence of cpu_dev() and insert before its
    # enclosing function.
    # Even simpler: insert at top of file after any leading docblock.
    lines = src.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('use ') or line.startswith('//!') or line.strip() == '' or line.startswith('//'):
            insert_idx = i + 1
        else:
            break
    new = ''.join(lines[:insert_idx]) + HELPER_TOPLEVEL + ''.join(lines[insert_idx:])
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new)
    print(f"{path}: top-level helper inserted")


if __name__ == '__main__':
    main()
