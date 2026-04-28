//! Best-effort Windows DLL search path extension for oneMKL.
//!
//! Mirrors `fuel-aocl-cpu-backend`'s `dll_path` module. The Intel
//! oneAPI installer drops `mkl_rt.2.dll` at
//! `C:\Program Files (x86)\Intel\oneAPI\mkl\<ver>\bin\` and offers a
//! `setvars.bat` users can run before invoking — but apps that
//! launch without that prep crash with `STATUS_DLL_NOT_FOUND` at
//! the first MKL call.
//!
//! [`ensure_loadable`] is called at the top of `MklBackend::try_new`
//! and `probe_mkl_loadable` to prepend a discovered MKL bin dir to
//! the process's `PATH` env var before any `LoadLibrary` attempt.
//!
//! On non-Windows platforms this is a no-op — Linux's `ld.so` reads
//! its config files at process start.
//!
//! # Discovery order (Windows)
//!
//! 1. `MKLROOT` env var (typically set by `setvars.bat`):
//!    `<MKLROOT>\bin\` (oneAPI 2024+) or `<MKLROOT>\redist\intel64\`
//!    (older releases).
//! 2. `ONEAPI_ROOT` env var: `<ONEAPI_ROOT>\mkl\latest\bin\`.
//! 3. Standard install layout: `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\`.
//! 4. Versioned subdirs under the same root, newest-first.
//!
//! First match containing `mkl_rt*.dll` wins.

#[cfg(windows)]
pub(crate) fn ensure_loadable() {
    let candidates: Vec<std::path::PathBuf> = candidate_dirs();
    for dir in candidates {
        if dir.is_dir() && contains_mkl_rt_dll(&dir) {
            prepend_to_path(&dir);
            return;
        }
    }
}

#[cfg(not(windows))]
pub(crate) fn ensure_loadable() {
    // No-op on non-Windows.
}

#[cfg(windows)]
fn candidate_dirs() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();

    // 1. MKLROOT (set by Intel's setvars.bat). Modern oneAPI uses
    // `<MKLROOT>\bin\`; older releases use `<MKLROOT>\redist\intel64`.
    if let Ok(root) = std::env::var("MKLROOT") {
        let p = std::path::PathBuf::from(&root);
        out.push(p.join("bin"));
        let mut redist = p;
        redist.push("redist");
        redist.push("intel64");
        out.push(redist);
    }

    // 2. ONEAPI_ROOT pointing at the oneAPI install root.
    if let Ok(root) = std::env::var("ONEAPI_ROOT") {
        let mut p = std::path::PathBuf::from(&root);
        p.push("mkl");
        p.push("latest");
        p.push("bin");
        out.push(p);
    }

    // 3. Standard Windows install — `latest` symlink first.
    out.push(std::path::PathBuf::from(
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
    ));

    // 4. Versioned subdirs under the standard root, newest-first.
    let standard_root = std::path::Path::new(r"C:\Program Files (x86)\Intel\oneAPI\mkl");
    if let Ok(entries) = std::fs::read_dir(standard_root) {
        let mut versioned: Vec<std::path::PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s != "latest" && s.chars().next().is_some_and(|c| c.is_ascii_digit()))
                    .unwrap_or(false)
            })
            .map(|p| p.join("bin"))
            .collect();
        // Newest-first by lexicographic name (works for "2025.3" > "2024.2").
        versioned.sort_by(|a, b| b.cmp(a));
        out.extend(versioned);
    }

    out
}

#[cfg(windows)]
fn contains_mkl_rt_dll(dir: &std::path::Path) -> bool {
    // Versioned `mkl_rt.<n>.dll` filenames vary across releases —
    // 2024+ uses `mkl_rt.2.dll`, older releases used `mkl_rt.dll`.
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("mkl_rt") && name.ends_with(".dll") {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(windows)]
fn prepend_to_path(dir: &std::path::Path) {
    let dir_str = match dir.to_str() {
        Some(s) => s,
        None => return,
    };
    let current = std::env::var_os("PATH").unwrap_or_default();
    let current_str = current.to_string_lossy();
    if current_str
        .split(';')
        .any(|p| p.eq_ignore_ascii_case(dir_str))
    {
        return;
    }
    let new = format!("{dir_str};{}", current_str);
    // SAFETY: env::set_var is unsafe in edition 2024. Called at
    // backend construction time — typically once at startup before
    // worker threads — so the practical race is nil.
    unsafe { std::env::set_var("PATH", new) };
}
