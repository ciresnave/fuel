//! Vulkan command-buffer capture of a "run" — Phase C PR-C2b.
//!
//! The Vulkan analogue of CUDA-graph capture (see fuel-cuda-backend's
//! `capture` module). A *run* — a single-device, contiguous span of
//! compute dispatches between branch points — is recorded once into a
//! **reusable** `VkCommandBuffer` (vulkane's `begin()` uses default begin
//! flags, i.e. NOT `ONE_TIME_SUBMIT`, so the buffer is re-submittable),
//! then replayed by re-submitting it. That trades per-dispatch CPU
//! recording overhead for a single `vkQueueSubmit`.
//!
//! Reuse modes:
//!   - [`replay`](CapturedRun::replay) re-submits the recorded buffer. It
//!     reads whatever the bound descriptor sets currently point at, so
//!     writing fresh data into the *same* operand buffers and replaying
//!     recomputes the run — the cheap decode-loop path (capture once,
//!     re-submit per token against stable storage).
//!   - [`rebind`](CapturedRun::rebind) re-records the command buffer
//!     against *new* operand buffers (a fresh descriptor set bound to the
//!     new buffers). This is the Vulkan counterpart of CUDA's rebind
//!     (which re-captures a template and `cuGraphExecUpdate`s it): it
//!     re-issues the dispatch sequence, not the whole pipeline/descriptor
//!     setup, and matches how this backend uses descriptor sets
//!     elsewhere — a fresh set per dispatch rather than in-place
//!     `vkUpdateDescriptorSets` reuse, which NVIDIA does not reflect into
//!     an already-recorded, re-submitted command buffer.
//!
//! Like the CUDA module this is a **capability**, not a wired-in
//! optimization: capture amortizes only over repeated replay of the same
//! run, which arrives with Phase D's persistent cross-realize graph. When
//! Phase D wires it into the executor it will record the real run-batches
//! through the `Recorder`; this primitive proves the approach now.
//!
//! Contract: the caller must keep the bound pipeline and descriptor
//! set(s) (and the buffers they reference) alive for as long as the
//! `CapturedRun` may be replayed.

use vulkane::safe::*;

use crate::{VulkanBackend, vk_err};

/// A captured, reusable command buffer: a run's compute dispatches
/// recorded once and replayable via a single `vkQueueSubmit`.
pub struct CapturedRun {
    // Drop order matters: `CommandBuffer::drop` calls `vkFreeCommandBuffers`
    // against `pool`, so the buffer must drop BEFORE the pool is destroyed.
    // Rust drops fields in declaration order — keep `cb` first.
    cb: CommandBuffer,
    pool: CommandPool,
}

impl VulkanBackend {
    /// Whether this backend can capture a run's dispatches into a
    /// reusable command buffer. Always `true` for Vulkan.
    ///
    /// The generic, executor-facing capability surface is Phase D's job;
    /// for now this backend-level method is the advertisement.
    pub fn supports_command_buffer_capture(&self) -> bool {
        true
    }

    /// Record the compute dispatches issued by `record` into a reusable
    /// command buffer and return a [`CapturedRun`].
    ///
    /// `record` is handed a fresh `CommandBufferRecording` and must
    /// enqueue its `bind_compute_pipeline` / `bind_compute_descriptor_sets`
    /// / `dispatch` calls onto it. A dedicated command pool is allocated
    /// for the captured buffer so its lifetime is independent of the
    /// backend's batching `Recorder`.
    pub fn capture_run<F>(&self, record: F) -> fuel_core_types::Result<CapturedRun>
    where
        F: FnOnce(&mut CommandBufferRecording<'_>) -> fuel_core_types::Result<()>,
    {
        let pool = CommandPool::new(&self.device, self.queue_family).map_err(vk_err)?;
        let cb = record_into(&pool, record)?;
        Ok(CapturedRun { cb, pool })
    }
}

/// Allocate a primary command buffer from `pool` and record `f` into it.
fn record_into<F>(pool: &CommandPool, f: F) -> fuel_core_types::Result<CommandBuffer>
where
    F: FnOnce(&mut CommandBufferRecording<'_>) -> fuel_core_types::Result<()>,
{
    let mut cb = pool.allocate_primary().map_err(vk_err)?;
    {
        let mut rec = cb.begin().map_err(vk_err)?;
        f(&mut rec)?;
        rec.end().map_err(vk_err)?;
    }
    Ok(cb)
}

impl CapturedRun {
    /// Replay the captured dispatches: submit the command buffer and block
    /// until it completes. Reads whatever buffers the bound descriptor
    /// sets currently point at.
    pub fn replay(&self, backend: &VulkanBackend) -> fuel_core_types::Result<()> {
        let fence = Fence::new(&backend.device).map_err(vk_err)?;
        backend
            .queue
            .submit(&[&self.cb], Some(&fence))
            .map_err(vk_err)?;
        fence.wait(u64::MAX).map_err(vk_err)?;
        Ok(())
    }

    /// Rebind the captured run onto new operand buffers by re-recording
    /// the command buffer. `record` must re-issue the SAME dispatch
    /// sequence (same pipeline + workgroups), binding a descriptor set
    /// that points at the new buffers. After this returns the run is ready
    /// to [`replay`](CapturedRun::replay) against the new operands.
    pub fn rebind<F>(&mut self, _backend: &VulkanBackend, record: F) -> fuel_core_types::Result<()>
    where
        F: FnOnce(&mut CommandBufferRecording<'_>) -> fuel_core_types::Result<()>,
    {
        // Allocate a fresh command buffer from the existing pool and
        // record the re-targeted dispatches; the previous buffer drops
        // (freed back to the pool) when the assignment overwrites it.
        self.cb = record_into(&self.pool, record)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::{Layout, Shape};

    fn backend_or_skip() -> Option<VulkanBackend> {
        match VulkanBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("no Vulkan device; skipping: {e:?}");
                None
            }
        }
    }

    fn as_bytes(v: &[f32]) -> &[u8] {
        // SAFETY: f32 has no padding/invalid bit patterns; the view lives
        // only for the duration of the upload call.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn from_bytes(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Capture an `affine_f32` dispatch (`out = 2*x + 1`) into a reusable
    /// command buffer, then prove two reuse modes:
    ///   1. replay after capture — correct on the original operands;
    ///   2. rebind onto NEW operand buffers (re-record with a fresh
    ///      descriptor set), then replay — operand rebasing.
    /// Mode 2's output buffer is freshly allocated (zeroed) and only holds
    /// the correct values if the rebound command buffer genuinely
    /// re-executes against the new operands, so the test fails if rebind
    /// is a no-op.
    #[test]
    #[ignore = "requires a live Vulkan device"]
    fn capture_replay_rebind_affine() {
        let Some(backend) = backend_or_skip() else { return };
        let n = 4usize;
        let bytes = n * std::mem::size_of::<f32>();
        let layout = Layout::contiguous(Shape::from_dims(&[n]));
        let expect = |x: &[f32]| -> Vec<f32> { x.iter().map(|v| 2.0 * v + 1.0).collect() };

        let xa = [1.0f32, 2.0, 3.0, 4.0];
        let input = backend.upload_bytes(as_bytes(&xa)).unwrap();
        let out = backend.alloc_bytes(bytes).unwrap();

        let pipeline = &backend.pipelines.affine_pipeline;
        let pipe_layout = &backend.pipelines.affine_layout;

        // Shared helper builds the real affine descriptor + params UBO +
        // workgroup count (no duplicated param layout). The transient
        // params buffer must outlive the replays that use it — bind it for
        // the whole test scope.
        let (desc, _pbuf, _pmem, wg) = backend
            .build_affine_f32_dispatch(&input, &out, 2.0, 1.0, &layout)
            .unwrap();

        let mut captured = backend
            .capture_run(|rec| {
                rec.bind_compute_pipeline(pipeline);
                rec.bind_compute_descriptor_sets(pipe_layout, 0, &[&desc]);
                rec.dispatch(wg, 1, 1);
                Ok(())
            })
            .expect("capture_run");

        // 1) replay on original operands
        captured.replay(&backend).unwrap();
        let got = from_bytes(&backend.download_bytes(&out).unwrap());
        assert_eq!(&got[..n], &expect(&xa)[..], "replay on original operands");

        // 2) rebind onto NEW operand buffers (re-record with a fresh
        //    descriptor set), then replay
        let xc = [5.0f32, 6.0, 7.0, 8.0];
        let input2 = backend.upload_bytes(as_bytes(&xc)).unwrap();
        let out2 = backend.alloc_bytes(bytes).unwrap();
        let (desc2, _p2buf, _p2mem, wg2) = backend
            .build_affine_f32_dispatch(&input2, &out2, 2.0, 1.0, &layout)
            .unwrap();
        captured
            .rebind(&backend, |rec| {
                rec.bind_compute_pipeline(pipeline);
                rec.bind_compute_descriptor_sets(pipe_layout, 0, &[&desc2]);
                rec.dispatch(wg2, 1, 1);
                Ok(())
            })
            .expect("rebind");
        captured.replay(&backend).unwrap();
        let got2 = from_bytes(&backend.download_bytes(&out2).unwrap());
        assert_eq!(
            &got2[..n],
            &expect(&xc)[..],
            "replay after rebinding to new operand buffers"
        );

        // `input`/`out`/`desc` must outlive the original capture; still in
        // scope here, so the borrows the command buffer relies on stay
        // sound across both replays.
        let _ = (&input, &out, &desc);
        assert!(backend.supports_command_buffer_capture());
    }
}
