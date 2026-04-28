//! Best-effort Windows DLL search path extension for AOCL.
//!
//! The AMD AOCL installer drops `AOCL-LibBlis-Win-dll.dll` at
//! `C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\LP64\` (or the
//! `ILP64` variant) but **does not add that directory to system
//! PATH**. Without intervention, the first `aocl_blas::gemm` call
//! fails the dynamic loader with `STATUS_DLL_NOT_FOUND` (Windows
//! error 0xc0000135) and the user sees a startup crash.
//!
//! This module's [`ensure_loadable`] is called at the top of
//! `AoclBackend::try_new` (and the standalone `probe_aocl_loadable`)
//! to prepend a discovered AOCL bin dir to the process's `PATH`
//! environment variable before any `LoadLibrary` attempt happens.
//!
//! On non-Windows platforms this is a no-op — the dynamic loader
//! reads `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` at process start, so
//! mid-process changes wouldn't take effect anyway, and AOCL on
//! Linux typically lives in `ld.so.conf`-managed paths.
//!
//! # Discovery order (Windows)
//!
//! 1. `AOCL_ROOT` env var, if set: `<AOCL_ROOT>\amd-blis\lib\LP64`
//! 2. `C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\LP64`
//! 3. `C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\ILP64`
//!
//! First match wins. If none exist, the function returns silently —
//! the subsequent gemm probe will fail and `try_new` returns an Err
//! that the caller surfaces.

#[cfg(windows)]
pub(crate) fn ensure_loadable() {
    let candidates: Vec<std::path::PathBuf> = candidate_dirs();
    for dir in candidates {
        if dir.is_dir() && contains_blis_dll(&dir) {
            prepend_to_path(&dir);
            return;
        }
    }
}

#[cfg(not(windows))]
pub(crate) fn ensure_loadable() {
    // No-op on non-Windows. The dynamic loader's search path is
    // process-launch-fixed there; mid-process PATH manipulation
    // wouldn't help.
}

#[cfg(windows)]
fn candidate_dirs() -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(root) = std::env::var("AOCL_ROOT") {
        let mut p = std::path::PathBuf::from(&root);
        p.push("amd-blis");
        p.push("lib");
        let mut lp64 = p.clone();
        lp64.push("LP64");
        out.push(lp64);
        let mut ilp64 = p;
        ilp64.push("ILP64");
        out.push(ilp64);
    }
    out.push(std::path::PathBuf::from(
        r"C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\LP64",
    ));
    out.push(std::path::PathBuf::from(
        r"C:\Program Files\AMD\AOCL-Windows\amd-blis\lib\ILP64",
    ));
    out
}

#[cfg(windows)]
fn contains_blis_dll(dir: &std::path::Path) -> bool {
    // Either the single-threaded or multi-threaded BLIS DLL is fine —
    // AOCL ships both, the loader picks via the import lib.
    dir.join("AOCL-LibBlis-Win-dll.dll").is_file()
        || dir.join("AOCL-LibBlis-Win-MT-dll.dll").is_file()
}

#[cfg(windows)]
fn prepend_to_path(dir: &std::path::Path) {
    let dir_str = match dir.to_str() {
        Some(s) => s,
        None => return,
    };
    let current = std::env::var_os("PATH").unwrap_or_default();
    let current_str = current.to_string_lossy();
    // Idempotent — don't keep prepending across multiple try_new
    // calls in the same process.
    if current_str
        .split(';')
        .any(|p| p.eq_ignore_ascii_case(dir_str))
    {
        return;
    }
    let new = format!("{dir_str};{}", current_str);
    // SAFETY: env::set_var is unsafe in edition 2024 because it isn't
    // thread-safe with concurrent reads. AoclBackend::try_new runs at
    // backend construction time — typically once at startup before
    // any threads are spawned — so the practical race is nil.
    unsafe { std::env::set_var("PATH", new) };
}
