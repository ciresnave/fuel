//! Device-resident `KVCache<B>` infrastructure extracted from `lazy.rs`.
//!
//! This module hosts the backend-agnostic KV cache that `LlamaModel`
//! and `PhiModel` consume during decode. Splitting it out (a) makes
//! `lazy.rs` smaller, (b) clarifies the dependency boundary —
//! `lazy_kv_cache_device.rs` depends on `fuel_graph_executor` +
//! optional Vulkan/CUDA but does NOT depend on any specific model
//! shell, and (c) prepares the standalone lazy_llama2c / lazy_phi
//! ports to consume the same KV cache without copy-paste once they
//! grow KV-aware forward methods.
//!
//! The original code stays bit-identical; `lazy.rs` re-exports the
//! public surface (`KVCacheEntry`, `KVCache`, `GpuKVCache`) so
//! external consumers (binaries in `fuel-lazy-examples`) compile
//! unchanged.

use crate::lazy::LlamaConfig;
use crate::Shape;
use fuel_graph_executor::GraphBackend;

/// Device-resident KV cache, generic over `GraphBackend`. Keys and
/// values stay on the device that owns `B::Storage` across decode
/// steps, eliminating the D2H readback + H2D re-upload round-trip
/// that the host-resident `LlamaKVCache` path requires.
///
/// For `B = CpuBackend`, `B::Storage = AnyRefTensor` which is already
/// host-resident, so this type collapses gracefully to a host cache
/// for CPU users. For `B = CudaBackend` / `VulkanBackend` / future
/// GPU backends, storage lives on the device and concat / update
/// happens via the backend's native ops.
/// Per-layer KV storage. `F32` is the default (full precision, 4 bytes
/// per element). `Q8` stores the GGML Q8_0 block stream (34 bytes per
/// 32 elements = 1.0625 bytes/elem — roughly 4× the cache capacity at
/// ~1% quality loss). The Q8 variant is opt-in via
/// `KVCache::enable_q8_cache()`.
pub enum KVCacheEntry<S> {
    F32 { k: S, v: S },
    /// `k_blocks` / `v_blocks` are U32-typed storages holding the raw
    /// Q8_0 block byte stream (via `GraphBackend::quantize_q8_0`).
    Q8 { k_blocks: S, v_blocks: S },
}

pub struct KVCache<B: fuel_graph_executor::GraphBackend> {
    /// Per-layer cache entry. `None` until the layer's first forward
    /// populates it. Logical shape: `[1, n_kv_heads, cached_len, head_dim]`.
    pub(crate) layers: Vec<Option<KVCacheEntry<B::Storage>>>,
    pub cached_len: usize,
    // Shape metadata held for future save/restore and cross-device
    // migration methods. Not currently read on the decode hot path.
    #[allow(dead_code)]
    pub(crate) n_kv_heads: usize,
    #[allow(dead_code)]
    pub(crate) head_dim: usize,
    /// When true, fresh K/V are quantized to Q8_0 after each forward
    /// and dequantized on the next read. Requires the backend to
    /// implement `GraphBackend::{quantize,dequantize}_q8_0`.
    pub q8_enabled: bool,
    /// When true, the cache's layers have been spilled to host via a
    /// backend-specific `park` method. Ops against a parked cache
    /// must `unpark` first; the cache's `forward_with_cache_*` entry
    /// points would see host-backed storages and panic cleanly.
    pub parked: bool,
}

impl<B: fuel_graph_executor::GraphBackend> KVCache<B> {
    pub fn new(config: &LlamaConfig) -> Self {
        Self::with_dims(config.n_layers, config.n_kv_heads, config.head_dim)
    }

    /// Constructor for models that don't use `LlamaConfig` (e.g. PhiModel).
    pub fn with_dims(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| None).collect(),
            cached_len: 0,
            n_kv_heads,
            head_dim,
            q8_enabled: false,
            parked: false,
        }
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Read access to a layer's entry. Returns `None` if the layer
    /// hasn't been populated yet (fresh cache) or has been cleared.
    /// Used by tiered-residency paths and tests.
    pub fn layer(&self, li: usize) -> Option<&KVCacheEntry<B::Storage>> {
        self.layers.get(li).and_then(|o| o.as_ref())
    }

    /// Mutable access. Rarely needed from outside; mainly for
    /// residency management code that needs to swap entries in place.
    pub fn layer_mut(&mut self, li: usize) -> Option<&mut KVCacheEntry<B::Storage>> {
        self.layers.get_mut(li).and_then(|o| o.as_mut())
    }

    /// Install a layer's entry directly. Used by tests and by the
    /// tiered-residency park/unpark paths when they need to swap
    /// in a rebuilt entry.
    pub fn set_layer(&mut self, li: usize, entry: KVCacheEntry<B::Storage>) {
        self.layers[li] = Some(entry);
    }

    /// Enable Q8_0 quantization of the KV cache. Fresh K/V will be
    /// quantized after each forward pass and dequantized on the next
    /// read. Cuts KV-cache memory ~4× at ~1% quality loss.
    pub fn enable_q8_cache(&mut self) {
        self.q8_enabled = true;
    }

    /// Shrink the cache back to the first `new_len` positions along the
    /// seq dim. Used by speculative decoding's reject path to roll back
    /// after drafted tokens are rejected by the target model.
    ///
    /// No-op if `new_len >= cached_len`. For `new_len == 0` all layer
    /// entries are cleared (same state as a fresh cache).
    ///
    /// Q8-cached entries are not yet supported — bails with an error.
    /// Q8 blocks are 32-element aligned and an arbitrary `new_len`
    /// would require re-quantizing the trailing partial block; needs
    /// a separate kernel. Tracked as follow-up.
    pub fn truncate_to(&mut self, new_len: usize, backend: &B) -> crate::Result<()> {
        if new_len >= self.cached_len {
            return Ok(());
        }
        if self.q8_enabled {
            fuel_core_types::bail!(
                "KVCache::truncate_to: Q8 cache truncation not yet implemented"
            );
        }

        let batch = 1;
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let old_seq = self.cached_len;

        for layer in &mut self.layers {
            let entry = match layer.take() {
                Some(e) => e,
                None => continue,
            };
            let (k, v) = match entry {
                KVCacheEntry::F32 { k, v } => (k, v),
                KVCacheEntry::Q8 { .. } => unreachable!("guarded above"),
            };
            // Early-return cleanly: if new_len == 0, drop the storage.
            if new_len == 0 {
                continue;
            }
            let new_k = truncate_kv_seq(backend, &k, batch, n_kv, old_seq, new_len, hd)?;
            let new_v = truncate_kv_seq(backend, &v, batch, n_kv, old_seq, new_len, hd)?;
            *layer = Some(KVCacheEntry::F32 { k: new_k, v: new_v });
        }
        self.cached_len = new_len;
        Ok(())
    }
}

/// Shrink an F32 K/V storage of shape `[batch, n_kv, old_seq, head_dim]`
/// (row-major contiguous) to `[batch, n_kv, new_seq, head_dim]`. Uses
/// `copy_strided_src` — one dispatch per tensor, all on-device.
fn truncate_kv_seq<B: fuel_graph_executor::GraphBackend>(
    backend: &B,
    src: &B::Storage,
    batch: usize,
    n_kv: usize,
    old_seq: usize,
    new_seq: usize,
    head_dim: usize,
) -> crate::Result<B::Storage> {
    // Source is contiguous with the OLD seq length; we want to read
    // only the first new_seq rows along dim 2. That's a strided read
    // where dim-2 stride stays head_dim but the gap between heads
    // skips the trailing old_seq-new_seq rows' worth of data.
    let src_shape = Shape::from_dims(&[batch, n_kv, new_seq, head_dim]);
    let src_strides: fuel_core_types::StrideVec = smallvec::smallvec![
        (n_kv * old_seq * head_dim) as isize,
        (old_seq * head_dim) as isize,
        head_dim as isize,
        1_isize,
    ];
    let src_layout = fuel_core_types::Layout::new(src_shape.clone(), src_strides, 0);

    let dtype = backend.storage_dtype(src);
    let dst_shape = Shape::from_dims(&[batch, n_kv, new_seq, head_dim]);
    let mut dst = backend.alloc_zeros(&dst_shape, dtype)?;
    backend.copy_strided_src(src, &mut dst, 0, &src_layout)?;
    Ok(dst)
}

/// CUDA-only alias kept for backward compatibility with existing
/// callers. Prefer `KVCache<CudaBackend>` directly in new code.
#[cfg(feature = "cuda")]
pub type GpuKVCache = KVCache<fuel_cuda_backend::CudaBackend>;

// ---- Tiered residency: KVCache park / unpark (Vulkan-only) ------------
//
// An idle `KVCache<VulkanBackend>` can be spilled to a host-side
// `ResidencyFile` via `park`, reclaiming its VRAM. When the caller
// needs the cache again (e.g., the next turn of a paused
// conversation), `unpark` faults each layer back to VRAM.
//
// First consumer of the P5 tiered-residency API. Other consumers
// (weight-layer offloading, long-context KV windowing) will come
// later; they reuse the same `ResidencyFile` + evict/fault_back
// primitives.

#[cfg(feature = "vulkan")]
impl KVCache<fuel_vulkan_backend::VulkanBackend> {
    /// Evict all layer K/V storage to the given residency file,
    /// freeing VRAM. `cached_len`, `parked` flag, and layer metadata
    /// are preserved so `unpark` can bring it back faithfully.
    ///
    /// Fails cleanly if:
    /// - the cache is already parked (guard against double-park),
    /// - any layer uses the Q8 variant (Q8 park is a follow-up —
    ///   the bytes-to-host path for Q8-backed layers needs its
    ///   own kernel path to preserve block structure).
    pub fn park(
        &mut self,
        backend: &fuel_vulkan_backend::VulkanBackend,
        file: &std::sync::Arc<fuel_vulkan_backend::residency::ResidencyFile>,
    ) -> crate::Result<()> {
        if self.parked {
            fuel_core_types::bail!("KVCache::park: cache is already parked");
        }
        if self.q8_enabled {
            fuel_core_types::bail!(
                "KVCache::park: Q8-enabled caches are not yet supported"
            );
        }
        // Evict each layer's K and V. Replace the entries in-place
        // so callers holding `&mut cache` see the updated tiers.
        for li in 0..self.layers.len() {
            let entry = match self.layers[li].take() {
                Some(e) => e,
                None => continue, // layer hasn't been populated yet
            };
            let (k, v) = match entry {
                KVCacheEntry::F32 { k, v } => (k, v),
                KVCacheEntry::Q8 { .. } => unreachable!("guarded above"),
            };
            let k_host = backend.evict(&k, file)?;
            let v_host = backend.evict(&v, file)?;
            // Drop the old device-backed handles so the Arc<VulkanBuffer>
            // refcount drops to zero and the VRAM sub-allocation is
            // returned to the buffer pool.
            drop(k);
            drop(v);
            self.layers[li] = Some(KVCacheEntry::F32 { k: k_host, v: v_host });
        }
        self.parked = true;
        Ok(())
    }

    /// Bring a parked cache's layers back into VRAM. Reverses
    /// [`Self::park`]. Fails if the cache isn't parked.
    pub fn unpark(
        &mut self,
        backend: &fuel_vulkan_backend::VulkanBackend,
    ) -> crate::Result<()> {
        if !self.parked {
            fuel_core_types::bail!("KVCache::unpark: cache is not parked");
        }
        for li in 0..self.layers.len() {
            let entry = match self.layers[li].take() {
                Some(e) => e,
                None => continue,
            };
            let (k, v) = match entry {
                KVCacheEntry::F32 { k, v } => (k, v),
                KVCacheEntry::Q8 { .. } => unreachable!(
                    "park bailed on Q8; we shouldn't see it on unpark"
                ),
            };
            let k_dev = backend.fault_back(&k)?;
            let v_dev = backend.fault_back(&v)?;
            drop(k);
            drop(v);
            self.layers[li] = Some(KVCacheEntry::F32 { k: k_dev, v: v_dev });
        }
        self.parked = false;
        Ok(())
    }
}

