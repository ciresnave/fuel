//! CUDA-graph capture of a "run" — Phase C PR-C2a.
//!
//! A *run* (see `fuel-graph`'s `extract_runs`) is a single-device,
//! contiguous span of kernel launches between branch points. Re-issuing
//! those launches one-by-one every realize costs per-launch CPU
//! overhead; CUDA graphs let us record the launch sequence once and
//! replay it as a single, cheap `cuGraphLaunch`.
//!
//! This module is a **capability**, not yet a wired-in optimization. The
//! pipelined executor does not drive capture itself, because a captured
//! graph only pays off when the *same* run is replayed many times — and
//! the only repeated-replay point (the decode loop) builds a fresh graph
//! per token until Phase D gives runs a stable, cross-realize identity.
//! So the win lands with Phase D's persistent graph; this primitive is
//! built + GPU-tested now so Phase D has a proven thing to consume.
//!
//! Hard rule the API enforces by contract: **no device allocation and no
//! synchronization may happen inside a capture scope** — both are
//! illegal during stream capture. Allocate every operand buffer first,
//! then capture only the launches.
//!
//! Operand rebasing is supported two ways:
//!   - replay against the *same* buffers (fixed captured pointers, fresh
//!     contents) — the cheap decode-loop path; and
//!   - [`CapturedRun::rebind`] onto *new* buffers via `cuGraphExecUpdate`
//!     (topology-invariant, only kernel arguments change), with a
//!     re-instantiation fallback if CUDA refuses the in-place update.

use baracuda_driver::{CaptureMode, Graph, GraphExec, Stream};

use crate::{CudaDevice, Error, Result, WrapErr};

/// A captured, instantiated run: a span of kernel launches recorded into
/// a CUDA graph and instantiated into an executable that replays with far
/// lower per-launch CPU overhead than re-issuing the launches by hand.
pub struct CapturedRun {
    // The template graph must outlive the executable derived from it, and
    // is replaced by the new template on a successful `rebind`. Held for
    // ownership; not read directly after instantiation.
    #[allow(dead_code)]
    graph: Graph,
    exec: GraphExec,
}

impl CapturedRun {
    /// Replay the captured launches on `device`'s stream.
    ///
    /// The captured operand *pointers* are fixed, so this reads whatever
    /// data currently lives at those addresses. Writing fresh inputs into
    /// the same buffers and replaying recomputes the run — the cheap
    /// decode-loop path (capture once, replay per token).
    pub fn replay(&self, device: &CudaDevice) -> Result<()> {
        self.exec.launch(device.stream()).w()
    }

    /// Rebind the captured run onto *new* operand buffers, leaving it
    /// ready to [`replay`](CapturedRun::replay).
    ///
    /// `relaunch` must re-issue the SAME launch sequence (same kernels,
    /// same order) against the new pointers. The freshly captured
    /// template updates the executable in place via `cuGraphExecUpdate`
    /// (topology-invariant — only kernel arguments differ). If CUDA
    /// refuses the in-place update (or it errors), this falls back to
    /// re-instantiating the new template — still correct, just not the
    /// cheap path.
    pub fn rebind<F>(&mut self, device: &CudaDevice, relaunch: F) -> Result<()>
    where
        F: FnOnce(&Stream) -> Result<()>,
    {
        let new_template = device.capture_graph(relaunch)?;
        match self.exec.update(&new_template) {
            // `result == 0` is `CUgraphExecUpdateResult::SUCCESS`.
            Ok(res) if res.result == 0 => {}
            _ => {
                self.exec = new_template.instantiate().w()?;
            }
        }
        self.graph = new_template;
        Ok(())
    }
}

impl CudaDevice {
    /// Whether this backend can capture a run's kernel launches into a
    /// replayable CUDA graph. Always `true` for CUDA.
    ///
    /// The dispatch layer queries a backend's capabilities before
    /// attempting capture. The generic, executor-facing capability
    /// surface (a backend-trait hook the backend-agnostic executor can
    /// call) is Phase D's job; for now this device-level method is the
    /// advertisement.
    pub fn supports_graph_capture(&self) -> bool {
        true
    }

    /// Capture the kernel launches issued by `launch` into a replayable,
    /// instantiated [`CapturedRun`].
    ///
    /// `launch` is handed the device's stream (now in capture mode) and
    /// must enqueue its work on it. It **must not allocate device memory
    /// or synchronize** — both are forbidden during stream capture;
    /// allocate every operand buffer before calling.
    pub fn capture_run<F>(&self, launch: F) -> Result<CapturedRun>
    where
        F: FnOnce(&Stream) -> Result<()>,
    {
        let graph = self.capture_graph(launch)?;
        let exec = graph.instantiate().w()?;
        Ok(CapturedRun { graph, exec })
    }

    /// Capture `launch` into a (non-instantiated) template graph. Shared
    /// by [`capture_run`](CudaDevice::capture_run) and
    /// [`CapturedRun::rebind`]; bridges a fuel-`Result` closure across
    /// baracuda's capture scope (baracuda's closure must return its own
    /// `Result`, so a user error is stashed and re-raised after the scope
    /// ends — capture is always closed, never leaked).
    fn capture_graph<F>(&self, launch: F) -> Result<Graph>
    where
        F: FnOnce(&Stream) -> Result<()>,
    {
        let mut user_err: Option<Error> = None;
        let graph = self
            .stream()
            .capture(CaptureMode::ThreadLocal, |s| {
                if let Err(e) = launch(s) {
                    user_err = Some(e);
                }
                Ok(())
            })
            .w()?;
        if let Some(e) = user_err {
            return Err(e);
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use crate::CudaDevice;
    use crate::baracuda::scratch::Workspace;
    use crate::baracuda_kernels_sys as sys;

    fn dev_or_skip() -> Option<CudaDevice> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no CUDA device; skipping: {e:?}");
                None
            }
        }
    }

    /// One contiguous `affine_f32` launch (`y = mul*x + add`) onto the
    /// given (capturing) stream. Returns the raw status (0 = ok).
    /// Capture-safe: no alloc, no sync; scratch is the null/0 "no
    /// scratch" workspace the contig affine path uses.
    ///
    /// # Safety
    /// `x`/`y` must point to `n` valid f32s on the device; `stream` must
    /// be a valid CUstream.
    unsafe fn affine(
        n: usize,
        x: *const c_void,
        y: *mut c_void,
        mul: f32,
        add: f32,
        scratch: &Workspace,
        stream: *mut c_void,
    ) -> i32 {
        unsafe {
            sys::baracuda_kernels_affine_f32_run(
                n as i64,
                x,
                y,
                mul,
                add,
                scratch.as_raw(),
                scratch.bytes(),
                stream,
            )
        }
    }

    /// Capture an `affine` chain (`z = 3*(2*x + 1) = 6x + 3`), then prove
    /// the three reuse modes a captured run must support:
    ///   1. replay after capture — correct on the original operands;
    ///   2. replay after overwriting the SAME input buffer with new data
    ///      (the cheap decode-loop path — fixed pointers, fresh contents);
    ///   3. rebind onto entirely NEW operand buffers (true operand
    ///      rebasing via `cuGraphExecUpdate`), then replay.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn capture_replay_rebind_affine_chain() {
        let Some(dev) = dev_or_skip() else { return };
        let n = 4usize;
        let expect = |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 6.0 * v + 3.0).collect() };

        // ---- pre-allocate ALL buffers before capture (no malloc in capture) ----
        let xa = [1.0f32, 2.0, 3.0, 4.0];
        let mut xbuf = dev.clone_htod(&xa).unwrap();
        let ybuf = dev.alloc_zeros::<f32>(n).unwrap();
        let zbuf = dev.alloc_zeros::<f32>(n).unwrap();
        let scratch = Workspace::alloc(&dev, 0).unwrap();

        let xptr = xbuf.as_raw().0 as *const c_void;
        let yptr = ybuf.as_raw().0 as *mut c_void;
        let yptr_in = ybuf.as_raw().0 as *const c_void;
        let zptr = zbuf.as_raw().0 as *mut c_void;

        // ---- capture the two-launch run ----
        let mut captured = dev
            .capture_run(|s| {
                let sr = s.as_raw() as *mut c_void;
                let st1 = unsafe { affine(n, xptr, yptr, 2.0, 1.0, &scratch, sr) };
                assert_eq!(st1, 0, "affine #1 capture status");
                let st2 = unsafe { affine(n, yptr_in, zptr, 3.0, 0.0, &scratch, sr) };
                assert_eq!(st2, 0, "affine #2 capture status");
                Ok(())
            })
            .expect("capture_run");

        // 1) replay on original operands
        captured.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        let got = dev.clone_dtoh(&zbuf.slice(0..n)).unwrap();
        assert_eq!(got, expect(&xa), "replay on original operands");

        // 2) same buffer, new data -> replay recomputes (decode-loop path)
        let xb = [10.0f32, 20.0, 30.0, 40.0];
        dev.memcpy_htod(&xb, &mut xbuf.slice_mut(0..n)).unwrap();
        captured.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        let got = dev.clone_dtoh(&zbuf.slice(0..n)).unwrap();
        assert_eq!(got, expect(&xb), "replay after overwriting same input buffer");

        // 3) rebind onto NEW buffers (operand rebasing) -> replay
        let xc = [5.0f32, 6.0, 7.0, 8.0];
        let x2 = dev.clone_htod(&xc).unwrap();
        let y2 = dev.alloc_zeros::<f32>(n).unwrap();
        let z2 = dev.alloc_zeros::<f32>(n).unwrap();
        let x2p = x2.as_raw().0 as *const c_void;
        let y2p = y2.as_raw().0 as *mut c_void;
        let y2p_in = y2.as_raw().0 as *const c_void;
        let z2p = z2.as_raw().0 as *mut c_void;
        captured
            .rebind(&dev, |s| {
                let sr = s.as_raw() as *mut c_void;
                let st1 = unsafe { affine(n, x2p, y2p, 2.0, 1.0, &scratch, sr) };
                assert_eq!(st1, 0, "rebind affine #1 status");
                let st2 = unsafe { affine(n, y2p_in, z2p, 3.0, 0.0, &scratch, sr) };
                assert_eq!(st2, 0, "rebind affine #2 status");
                Ok(())
            })
            .expect("rebind");
        captured.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        let got = dev.clone_dtoh(&z2.slice(0..n)).unwrap();
        assert_eq!(got, expect(&xc), "replay after rebinding to new operand buffers");

        assert!(dev.supports_graph_capture());
    }

    /// DIAGNOSTIC (index_select capture bisection, 2026-07-10): capture the
    /// `index_select` kernel DIRECTLY via `capture_run` — bypassing the
    /// pipelined executor's `capture_decode` / `execute_work_item` path — on
    /// the same box/params where `capture_decode` mis-replays output element 0.
    /// Baracuda's isolated harness replays this exact call clean; if THIS
    /// Fuel-direct capture also replays clean, the bug is in the executor's
    /// capture path, not the kernel or `capture_run` itself (the bisection).
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn capture_index_select_direct_replays_clean() {
        use crate::CudaStorageBytes;
        use crate::baracuda::indexing::index_select_f32_into;

        let Some(dev) = dev_or_skip() else { return };
        let read_f32 = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        // wte [3,4] f32, row v = 1 + 10v + c → rows clearly distinct.
        let wte_f32: Vec<f32> = (0..3u32)
            .flat_map(|v| (0..4u32).map(move |c| 1.0 + 10.0 * v as f32 + c as f32))
            .collect();
        let wte_bytes: Vec<u8> = wte_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let wte = CudaStorageBytes::from_cpu_bytes(&dev, &wte_bytes).unwrap();
        let tok = CudaStorageBytes::from_cpu_bytes(&dev, &0u32.to_le_bytes()).unwrap();
        let emb = CudaStorageBytes::alloc(&dev, 4 * 4).unwrap(); // [1,4], zeroed

        // Eager warm on the zeroed output (exactly baracuda's harness shape),
        // with Fuel's exact params: outer=1, src_dim=3, n_idx=1, inner=4.
        index_select_f32_into(&wte, &tok, 1, 3, 1, 4, &emb).unwrap();
        dev.synchronize().unwrap();
        let warm = read_f32(&emb.to_cpu_bytes().unwrap());
        assert_eq!(warm, vec![1.0, 2.0, 3.0, 4.0], "eager warm gathers row 0");

        // Capture the SAME call directly (NO executor), then replay unchanged.
        let captured = dev
            .capture_run(|_s| index_select_f32_into(&wte, &tok, 1, 3, 1, 4, &emb))
            .expect("capture_run");
        captured.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        let replay = read_f32(&emb.to_cpu_bytes().unwrap());
        assert_eq!(
            replay, warm,
            "direct capture_run replays clean ⇒ index_select mis-replay is in the \
             executor capture path, not the kernel/capture_run",
        );
    }
}
