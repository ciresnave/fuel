//! KV-Cache implementations for autoregressive transformer inference.
//!
//! This module provides several cache strategies for storing key/value tensors during
//! autoregressive generation:
//!
//! - [`Cache`] / [`KvCache`] -- pre-allocated rolling buffer using `slice_set` for updates.
//!   Good for CPU inference and when you want a fixed memory ceiling.
//! - [`RotatingCache`] / [`RotatingKvCache`] -- sliding-window cache for models with bounded
//!   attention context (e.g. Mistral). Older entries are overwritten in a ring buffer.
//! - [`ConcatKvCache`] -- concatenation-based cache that grows dynamically. Preferred for
//!   GPU inference where the `cat` kernel is highly optimized.
//! - [`ScatteredCacheBuilder`] / [`ScatteredKvCache`] -- index-scatter-based cache for
//!   batched inference with heterogeneous sequence lengths and masking.
use crate::{DType, Device, Result, Tensor};

/// A single pre-allocated cache buffer for one of K or V.
///
/// The buffer is lazily allocated on the first [`append`](Cache::append) call (so that the
/// shape is inferred from the input) and grows in chunks of `max_seq_len` if the sequence
/// exceeds the initial capacity. Updates use `slice_set` for in-place writes.
#[derive(Debug, Clone)]
pub struct Cache {
    all_data: Option<Tensor>,
    dim: usize,
    current_seq_len: usize,
    grow_by: usize,
    max_seq_len: usize,
}

impl Cache {
    /// Creates a new empty `Cache`.
    ///
    /// * `dim` -- the tensor dimension along which sequence tokens are stored.
    /// * `max_seq_len` -- initial capacity (in tokens) of the pre-allocated buffer.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            current_seq_len: 0,
            grow_by: max_seq_len,
            max_seq_len,
        }
    }

    /// Returns the dimension along which tokens are cached.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of tokens currently stored in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Returns the current buffer capacity in tokens.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Returns a reference to the underlying pre-allocated tensor, if any.
    pub fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    /// Returns a narrow view of the buffer containing only the valid (filled) portion,
    /// or `None` if no data has been appended yet.
    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => Some(d.narrow(self.dim, 0, self.current_seq_len)?),
        };
        Ok(data)
    }

    /// Resets the cache, clearing all stored data. The buffer will be re-allocated on the
    /// next [`append`](Cache::append) call.
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        self.all_data = None;
    }

    /// Appends `src` to the cache along the configured dimension. The buffer grows
    /// automatically if needed.
    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad)
        };
        let ad = self.all_data.as_mut().unwrap();
        while self.current_seq_len + seq_len > self.max_seq_len {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.grow_by;
            let next_ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            *ad = Tensor::cat(&[&*ad, &next_ad], self.dim)?;
            self.max_seq_len += self.grow_by;
        }
        ad.slice_set(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

/// A pre-allocated rolling buffer for efficient KV caching in autoregressive transformers.
///
/// Wraps two [`Cache`] instances (one for keys, one for values) and exposes a unified
/// API for appending new tokens and retrieving the full cached sequence. The buffer is
/// lazily allocated on the first call to [`append`](KvCache::append) and grows
/// automatically if the sequence exceeds the initial `max_seq_len`.
///
/// # Example
///
/// ```rust
/// use candle_core::{DType, Device, Tensor};
/// use candle_core::kv_cache::KvCache;
///
/// // Create a cache for the sequence dimension (dim=2) with capacity for 32 tokens.
/// let mut cache = KvCache::new(2, 32);
/// assert_eq!(cache.current_seq_len(), 0);
///
/// // Simulate a prefill step with 5 tokens: shape [batch=1, heads=4, seq=5, head_dim=8].
/// let k = Tensor::zeros((1, 4, 5, 8), DType::F32, &Device::Cpu)?;
/// let v = Tensor::zeros((1, 4, 5, 8), DType::F32, &Device::Cpu)?;
/// let (full_k, full_v) = cache.append(&k, &v)?;
/// assert_eq!(full_k.dims(), &[1, 4, 5, 8]);
/// assert_eq!(cache.current_seq_len(), 5);
///
/// // Simulate a single decode step.
/// let k1 = Tensor::zeros((1, 4, 1, 8), DType::F32, &Device::Cpu)?;
/// let v1 = Tensor::zeros((1, 4, 1, 8), DType::F32, &Device::Cpu)?;
/// let (full_k, full_v) = cache.append(&k1, &v1)?;
/// assert_eq!(full_k.dims(), &[1, 4, 6, 8]);
/// assert_eq!(cache.current_seq_len(), 6);
///
/// // Reset for a new generation.
/// cache.reset();
/// assert_eq!(cache.current_seq_len(), 0);
/// # Ok::<(), candle_core::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    /// Creates a new empty `KvCache`.
    ///
    /// * `dim` -- the tensor dimension along which sequence tokens are concatenated
    ///   (typically `2` for `[batch, heads, seq, head_dim]` layouts).
    /// * `max_seq_len` -- initial buffer capacity in tokens.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = Cache::new(dim, max_seq_len);
        let v = Cache::new(dim, max_seq_len);
        Self { k, v }
    }

    /// Returns a reference to the key cache.
    pub fn k_cache(&self) -> &Cache {
        &self.k
    }

    /// Returns a reference to the value cache.
    pub fn v_cache(&self) -> &Cache {
        &self.v
    }

    /// Returns a mutable reference to the key cache.
    pub fn k_cache_mut(&mut self) -> &mut Cache {
        &mut self.k
    }

    /// Returns a mutable reference to the value cache.
    pub fn v_cache_mut(&mut self) -> &mut Cache {
        &mut self.v
    }

    /// Returns the currently cached key tensor, or `None` if the cache is empty.
    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    /// Returns the currently cached value tensor, or `None` if the cache is empty.
    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    /// Appends new key and value tensors to the cache and returns the full accumulated
    /// `(key, value)` pair covering all tokens seen so far.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let out_k = self.k.current_data()?;
        let out_v = self.v.current_data()?;
        let k = match out_k {
            None => {
                let mut shape = k.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, k.dtype(), k.device())?
            }
            Some(k) => k,
        };
        let v = match out_v {
            None => {
                let mut shape = v.dims().to_vec();
                shape[self.k.dim] = 0;
                Tensor::zeros(shape, v.dtype(), v.device())?
            }
            Some(v) => v,
        };
        Ok((k, v))
    }

    /// Returns the number of tokens currently stored in the cache.
    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    /// Clears both the key and value caches. The buffers will be re-allocated on the
    /// next [`append`](KvCache::append) call.
    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

/// A sliding-window cache that overwrites the oldest entries in a ring buffer.
///
/// Useful for models with bounded attention context (e.g. Mistral). The buffer has a
/// fixed size of `max_seq_len` tokens; once full, new tokens overwrite the oldest ones.
#[derive(Debug, Clone)]
pub struct RotatingCache {
    all_data: Option<Tensor>,
    dim: usize,
    // `offset` is the current write index in the buffer
    offset: usize,
    // The total size of the sequence seen so far.
    current_seq_len: usize,
    // max_seq_len is the size of the rotating buffer, it is actually allowed for the full
    // sequence to grow past this limit.
    max_seq_len: usize,
}

impl RotatingCache {
    /// Creates a new empty `RotatingCache` with the given window size.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self {
            all_data: None,
            dim,
            offset: 0,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    /// Returns the current write offset in the ring buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the dimension along which tokens are cached.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the total number of tokens seen so far (may exceed `max_seq_len`).
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Returns the fixed window size of the rotating buffer.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Returns a reference to the underlying tensor, if allocated.
    pub fn all_data(&self) -> &Option<Tensor> {
        &self.all_data
    }

    /// Returns the valid portion of the cache, or the full buffer if the window is full.
    pub fn current_data(&self) -> Result<Option<Tensor>> {
        let data = match self.all_data.as_ref() {
            None => None,
            Some(d) => {
                if self.current_seq_len >= self.max_seq_len {
                    Some(d.clone())
                } else {
                    Some(d.narrow(self.dim, 0, self.current_seq_len)?)
                }
            }
        };
        Ok(data)
    }

    /// Resets the cache, clearing all stored data.
    pub fn reset(&mut self) {
        self.offset = 0;
        self.current_seq_len = 0;
        self.all_data = None;
    }

    /// Appends `src` to the rotating cache and returns the resulting cache view.
    pub fn append(&mut self, src: &Tensor) -> Result<Tensor> {
        let seq_len = src.dim(self.dim)?;
        // This doesn't seem very idiomatic but because the creation can fail, it's tricky to use
        // self.all_data.get_or_insert_with.
        if self.all_data.is_none() {
            let mut shape = src.dims().to_vec();
            shape[self.dim] = self.max_seq_len;
            let ad = Tensor::zeros(shape, src.dtype(), src.device())?;
            self.all_data = Some(ad)
        };
        let ad = self.all_data.as_mut().unwrap();

        self.current_seq_len += seq_len;
        if seq_len >= self.max_seq_len {
            let to_copy = src
                .narrow(self.dim, seq_len - self.max_seq_len, self.max_seq_len)?
                .contiguous()?;
            ad.slice_set(&to_copy, self.dim, 0)?;
            self.offset = 0;
            // Here we return `src` rather than `ad` so that all the past can be used.
            Ok(src.clone())
        } else {
            let rem_len = self.max_seq_len - self.offset;
            if seq_len <= rem_len {
                ad.slice_set(&src.contiguous()?, self.dim, self.offset)?;
                self.offset = (self.offset + seq_len) % self.max_seq_len;
            } else {
                // We have to make two copies here as we go over the boundary of the cache.
                if rem_len > 0 {
                    let src1 = src.narrow(self.dim, 0, rem_len)?.contiguous()?;
                    ad.slice_set(&src1, self.dim, self.offset)?;
                }
                let src2 = src
                    .narrow(self.dim, rem_len, seq_len - rem_len)?
                    .contiguous()?;
                ad.slice_set(&src2, self.dim, 0)?;
                self.offset = seq_len - rem_len;
            }
            if self.current_seq_len >= self.max_seq_len {
                Ok(ad.clone())
            } else {
                Ok(ad.narrow(self.dim, 0, self.current_seq_len)?)
            }
        }
    }

    fn get_mask_abs(&self, size1: usize, size2: usize, device: &Device) -> Result<Tensor> {
        let context = self.max_seq_len;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    u8::from(size1 + j > size2 + i || size1 + j + context < size2 + i)
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), device)
    }

    fn get_mask_rel(&self, size1: usize, size2: usize, device: &Device) -> Result<Tensor> {
        let context = self.max_seq_len;
        let upd_offset = (self.offset + size1) % self.max_seq_len;
        let mask: Vec<_> = (0..size1)
            .flat_map(|pos_src| {
                // The absolute position of the elements that will get added to the cache.
                let pos_src = self.current_seq_len + pos_src;
                (0..size2).map(move |pos_cache_rel| {
                    // The absolute position of the cache elements after the addition.
                    let pos_cache = self.current_seq_len + size1 + pos_cache_rel - upd_offset;
                    let pos_cache = if pos_cache_rel < upd_offset {
                        pos_cache
                    } else {
                        pos_cache - self.max_seq_len
                    };
                    u8::from(pos_cache > pos_src || pos_cache + context < pos_src)
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        if seq_len <= self.max_seq_len {
            let upd_offset = (self.offset + seq_len) % self.max_seq_len;
            let cache_out_len = (self.current_seq_len + seq_len).min(self.max_seq_len);
            (0..cache_out_len)
                .map(|i| {
                    let pos_cache = self.current_seq_len + seq_len + i - upd_offset;
                    if i < upd_offset {
                        pos_cache
                    } else {
                        pos_cache - self.max_seq_len
                    }
                })
                .collect()
        } else {
            (self.current_seq_len..(self.current_seq_len + seq_len)).collect()
        }
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &Device) -> Result<Option<Tensor>> {
        let mask = if seq_len == 1 {
            None
        } else {
            let mask = if seq_len < self.max_seq_len {
                let cache_out_len = (self.current_seq_len + seq_len).min(self.max_seq_len);
                self.get_mask_rel(seq_len, cache_out_len, device)?
            } else {
                self.get_mask_abs(seq_len, seq_len, device)?
            };
            Some(mask)
        };
        Ok(mask)
    }
}

/// A paired sliding-window KV cache wrapping two [`RotatingCache`] instances.
///
/// Provides attention mask and position computation helpers for models that use
/// sliding-window attention.
#[derive(Debug, Clone)]
pub struct RotatingKvCache {
    k: RotatingCache,
    v: RotatingCache,
}

impl RotatingKvCache {
    /// Creates a new empty `RotatingKvCache` with the given window size.
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        let k = RotatingCache::new(dim, max_seq_len);
        let v = RotatingCache::new(dim, max_seq_len);
        Self { k, v }
    }

    /// Returns a reference to the key cache.
    pub fn k_cache(&self) -> &RotatingCache {
        &self.k
    }

    /// Returns a reference to the value cache.
    pub fn v_cache(&self) -> &RotatingCache {
        &self.v
    }

    /// Returns a mutable reference to the key cache.
    pub fn k_cache_mut(&mut self) -> &mut RotatingCache {
        &mut self.k
    }

    /// Returns a mutable reference to the value cache.
    pub fn v_cache_mut(&mut self) -> &mut RotatingCache {
        &mut self.v
    }

    /// Returns the currently cached key tensor.
    pub fn k(&self) -> Result<Option<Tensor>> {
        self.k.current_data()
    }

    /// Returns the currently cached value tensor.
    pub fn v(&self) -> Result<Option<Tensor>> {
        self.v.current_data()
    }

    /// Appends new key/value tensors and returns the full cache contents.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let out_k = self.k.append(k)?;
        let out_v = self.v.append(v)?;
        Ok((out_k, out_v))
    }

    /// Returns the current write offset in the ring buffer.
    pub fn offset(&self) -> usize {
        self.k.offset()
    }

    /// Returns the total number of tokens seen so far.
    pub fn current_seq_len(&self) -> usize {
        self.k.current_seq_len()
    }

    /// Returns the attn_mask to be applied *after* adding `seq_len` to the cache.
    pub fn attn_mask(&self, seq_len: usize, device: &Device) -> Result<Option<Tensor>> {
        self.k.attn_mask(seq_len, device)
    }

    /// Returns the positions corresponding to all the elements that will be returned
    /// *after* adding `seq_len` to the cache.
    pub fn positions(&self, seq_len: usize) -> Vec<usize> {
        self.k.positions(seq_len)
    }

    /// Resets both the key and value caches.
    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

/// Pre-computed indices and attention mask for a [`ScatteredKvCache`] update.
#[derive(Debug, Clone)]
pub struct IndicesAndMask {
    indices: Tensor,
    mask: Tensor,
}

impl IndicesAndMask {
    /// Returns the attention mask tensor.
    pub fn mask(&self) -> &Tensor {
        &self.mask
    }
}

/// Index-scatter-based KV cache for batched inference with heterogeneous sequence lengths.
///
/// Updates are performed via `scatter_set`, which allows each batch element to write to
/// a different cache position.
#[derive(Debug, Clone)]
pub struct ScatteredKvCache {
    k: Tensor,
    v: Tensor,
    context: usize,
}

impl ScatteredKvCache {
    /// Scatters new key/value entries into the cache at positions given by `iam` and
    /// returns the full cache tensors.
    pub fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        iam: &IndicesAndMask,
    ) -> Result<(Tensor, Tensor)> {
        if self.context <= k.dim(2)? {
            return Ok((k.clone(), v.clone()));
        }
        let indices = iam.indices.unsqueeze(2)?.unsqueeze(1)?;
        let indices = indices.broadcast_as(k.shape())?.contiguous()?;
        self.k.scatter_set(&indices, k, 2)?;
        self.v.scatter_set(&indices, v, 2)?;
        Ok((self.k.clone(), self.v.clone()))
    }

    /// Returns a reference to the full key cache tensor.
    pub fn k(&self) -> &Tensor {
        &self.k
    }

    /// Returns a reference to the full value cache tensor.
    pub fn v(&self) -> &Tensor {
        &self.v
    }
}

/// Builder for [`ScatteredKvCache`] instances and their per-step indices/masks.
///
/// Tracks per-batch-element positions and ring-buffer indices, and generates the
/// [`IndicesAndMask`] needed for each forward step.
#[derive(Debug, Clone)]
pub struct ScatteredCacheBuilder {
    context: usize,
    // The current position in the stream, this can be larger than context.
    positions: Vec<usize>,
    // The index where the next element will be stored.
    indices: Vec<usize>,
    dtype: DType,
    device: Device,
}

impl ScatteredCacheBuilder {
    /// Creates a new builder for the given batch size, context window, dtype, and device.
    pub fn new(batch_size: usize, context: usize, dtype: DType, device: &Device) -> Result<Self> {
        let positions = vec![0; batch_size];
        let indices = vec![0; batch_size];
        Ok(Self {
            positions,
            indices,
            context,
            dtype,
            device: device.clone(),
        })
    }

    /// Allocates a new [`ScatteredKvCache`] with zero-initialized tensors.
    pub fn make_cache(&self, num_heads: usize, head_dim: usize) -> Result<ScatteredKvCache> {
        let batch_size = self.batch_size();
        let shape = (batch_size, num_heads, self.context, head_dim);
        let k = Tensor::zeros(shape, self.dtype, self.device())?;
        let v = Tensor::zeros(shape, self.dtype, self.device())?;
        Ok(ScatteredKvCache {
            k,
            v,
            context: self.context,
        })
    }

    /// Returns the current absolute position for each batch element.
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// Resets all batch elements to position 0.
    pub fn reset(&mut self) {
        self.positions.fill(0);
        self.indices.fill(0);
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.positions.len()
    }

    /// Resets a single batch element to position 0.
    pub fn reset_batch_index(&mut self, batch_index: usize) {
        self.positions[batch_index] = 0;
        self.indices[batch_index] = 0;
    }

    /// Computes cache write indices and the causal attention mask for the next `seq_len`
    /// tokens. `batch_mask` indicates which batch elements are active.
    #[allow(clippy::needless_range_loop)]
    pub fn indices_and_mask(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask> {
        // mask shape is (b, h, t, k)
        let context = self.context;
        if self.context <= seq_len {
            return self.indices_and_mask_abs(seq_len, batch_mask);
        }
        let mut attention_masks = Vec::with_capacity(self.batch_size());
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let masks: Vec<Vec<f32>> = vec![vec![0.0; context]; seq_len];
                let indices = vec![self.indices[batch_i] as u32; seq_len];
                attention_masks.push(masks);
                cache_indices.push(indices);
            } else {
                let start_index = self.indices[batch_i];
                let start_pos = self.positions[batch_i];
                let mut masks: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
                let mut indices = Vec::with_capacity(seq_len);
                let mut all_pos = vec![usize::MAX; context];
                if start_pos < context {
                    for i in 0..start_pos {
                        all_pos[i] = i;
                    }
                } else {
                    let offset = start_pos - start_index;
                    for i in 0..context {
                        all_pos[i] = if i < start_index {
                            i + offset
                        } else {
                            i + offset - context
                        };
                    }
                }
                for seq_i in 0..seq_len {
                    let index = self.indices[batch_i];
                    all_pos[index] = seq_i + start_pos;
                    indices.push(index as u32);
                    self.indices[batch_i] += 1;
                    self.positions[batch_i] += 1;
                    if self.indices[batch_i] >= self.context {
                        self.indices[batch_i] = 0;
                    }
                }

                for seq_i in 0..seq_len {
                    let my_pos = seq_i + start_pos;
                    let mask = all_pos
                        .iter()
                        .map(|&pos| {
                            if pos <= my_pos {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f32>>();
                    masks.push(mask);
                }

                attention_masks.push(masks);
                cache_indices.push(indices);
            }
        }
        // Flattening the attention mask then using Tensor::from_vec rather using Tensor::new ends
        // up being almost 10x faster with candle 0.9.0. This has been fixed in candle 0.9.1.
        let attention_masks = attention_masks
            .into_iter()
            .flat_map(|m| m.into_iter().flatten())
            .collect::<Vec<f32>>();
        let mask = Tensor::from_vec(attention_masks, ((), 1, seq_len, context), self.device())?
            .to_dtype(self.dtype)?;
        let indices = Tensor::new(cache_indices, self.device())?;
        Ok(IndicesAndMask { indices, mask })
    }

    /// Returns the device used for cache tensors.
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(clippy::needless_range_loop)]
    fn indices_and_mask_abs(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask> {
        let mask = self.get_mask_abs(seq_len, seq_len)?;
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let indices = vec![self.indices[batch_i] as u32; seq_len];
                cache_indices.push(indices);
            } else {
                let mut indices = Vec::with_capacity(seq_len);
                for _ in 0..seq_len {
                    let index = self.indices[batch_i];
                    indices.push(index as u32);
                    self.indices[batch_i] += 1;
                    self.positions[batch_i] += 1;
                    if self.indices[batch_i] >= self.context {
                        self.indices[batch_i] = 0;
                    }
                }
                cache_indices.push(indices);
            }
        }
        let indices = Tensor::new(cache_indices, self.device())?;
        Ok(IndicesAndMask { indices, mask })
    }

    fn get_mask_abs(&self, size1: usize, size2: usize) -> Result<Tensor> {
        let context = self.context;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    if size1 + j > size2 + i || size1 + j + context < size2 + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), self.device())
    }
}

/// KV-Cache using concatenation for append operations
///
/// This implementation uses `Tensor::cat` instead of `slice_set` for updates,
/// providing significant GPU performance improvements for autoregressive generation.
///
/// # When to Use
///
/// **Recommended for:**
/// - GPU inference (CUDA, Metal)
/// - Autoregressive generation (token-by-token decoding)
///
/// **Use `KvCache` instead for:**
/// - CPU-only inference
/// - When you need fixed memory allocation upfront
///
/// # Example
///
/// ```ignore
/// use candle_nn::kv_cache::ConcatKvCache;
///
/// let mut cache = ConcatKvCache::new(2); // dim=2 for sequence dimension
///
/// // First token (prefill)
/// let k1 = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let v1 = Tensor::randn(0f32, 1., (1, 8, 10, 64), &device)?;
/// let (k, v) = cache.append(&k1, &v1)?;
///
/// // Subsequent tokens (decode)
/// let k_new = Tensor::randn(0f32, 1., (1, 8, 1, 64), &device)?;
/// let v_new = Tensor::randn(0f32, 1., (1, 8, 1, 64), &device)?;
/// let (k, v) = cache.append(&k_new, &v_new)?;
/// ```
#[derive(Debug, Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl ConcatKvCache {
    /// Create a new empty concatenation-based KV-cache
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to concatenate
    ///   - For attention with shape `[batch, heads, seq, head_dim]`, use `dim=2`
    ///   - For attention with shape `[batch, seq, heads, head_dim]`, use `dim=1`
    ///
    /// # Example
    /// ```ignore
    /// // For standard transformer attention: [B, H, S, D]
    /// let cache = ConcatKvCache::new(2);
    /// ```
    pub fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,
        }
    }

    /// Get current sequence length in the cache
    ///
    /// Returns 0 if the cache is empty.
    pub fn current_seq_len(&self) -> usize {
        self.k
            .as_ref()
            .and_then(|k| k.dims().get(self.dim).copied())
            .unwrap_or(0)
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }

    /// Get the concatenation dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Append key and value tensors to the cache
    ///
    /// This is the core operation that uses optimized concatenation kernels.
    ///
    /// # Arguments
    /// * `k` - Key tensor to append (shape: [..., seq_len, ...])
    /// * `v` - Value tensor to append (shape: [..., seq_len, ...])
    ///
    /// # Returns
    /// Tuple of `(full_k, full_v)` containing all cached keys and values,
    /// including the newly appended data.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        // Ensure inputs are contiguous for optimal concatenation performance
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // Update K cache using concatenation
        self.k = Some(match &self.k {
            None => k.clone(),
            Some(k_cache) => {
                // Concatenate along the sequence dimension
                // GPU kernel for cat is highly optimized:
                // - Fused allocation + copy
                // - Coalesced memory access
                // - Single kernel launch
                Tensor::cat(&[k_cache, &k], self.dim)?
            }
        });

        // Update V cache using concatenation
        self.v = Some(match &self.v {
            None => v.clone(),
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?,
        });

        Ok((
            self.k.as_ref().unwrap().clone(),
            self.v.as_ref().unwrap().clone(),
        ))
    }

    /// Reset the cache (clear all stored keys and values)
    ///
    /// After calling this, `is_empty()` will return `true` and
    /// `current_seq_len()` will return 0.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Get reference to current K cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn k(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    /// Get reference to current V cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }

    /// Get mutable reference to K cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn k_mut(&mut self) -> Option<&mut Tensor> {
        self.k.as_mut()
    }

    /// Get mutable reference to V cache data
    ///
    /// Returns `None` if the cache is empty.
    pub fn v_mut(&mut self) -> Option<&mut Tensor> {
        self.v.as_mut()
    }

    /// Get owned K and V tensors, consuming the cache
    ///
    /// Returns `None` if the cache is empty.
    pub fn into_inner(self) -> Option<(Tensor, Tensor)> {
        match (self.k, self.v) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IndexOp;

    #[test]
    fn test_scattered_kv_cache() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ScatteredCacheBuilder::new(2, 5, DType::F32, &device)?;
        let inf = f32::INFINITY;

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[0], [0]]);
        assert_eq!(
            mask,
            [[[0.0, -inf, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[1], [0]]);
        assert_eq!(
            mask,
            [[[0.0, 0.0, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        let iam = cache.indices_and_mask(3, &[false, true])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[2, 2, 2], [0, 1, 2]]);
        assert_eq!(
            mask,
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
                [
                    [0.0, -inf, -inf, -inf, -inf],
                    [0.0, 0.0, -inf, -inf, -inf],
                    [0.0, 0.0, 0.0, -inf, -inf]
                ]
            ]
        );

        let iam = cache.indices_and_mask(3, &[true, true])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[2, 3, 4], [3, 4, 0]]);
        assert_eq!(
            mask,
            [
                [
                    [0.0, 0.0, 0.0, -inf, -inf],
                    [0.0, 0.0, 0.0, 0.0, -inf],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ],
                [
                    [-inf, 0.0, 0.0, 0.0, -inf],
                    [-inf, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            ]
        );

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[0], [1]]);
        assert_eq!(
            mask,
            [[[0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        let iam = cache.indices_and_mask(2, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[1, 2], [1, 1]]);
        assert_eq!(
            mask,
            [
                [[0.0, 0.0, -inf, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
            ]
        );

        Ok(())
    }

    #[test]
    fn test_concat_cache_basic() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(2);

        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);

        // First append
        let k1 = Tensor::zeros((1, 8, 3, 64), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 8, 3, 64), DType::F32, &device)?;
        let (k, v) = cache.append(&k1, &v1)?;

        assert_eq!(k.dims(), &[1, 8, 3, 64]);
        assert_eq!(v.dims(), &[1, 8, 3, 64]);
        assert_eq!(cache.current_seq_len(), 3);
        assert!(!cache.is_empty());

        // Second append
        let k2 = Tensor::zeros((1, 8, 2, 64), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 8, 2, 64), DType::F32, &device)?;
        let (k, v) = cache.append(&k2, &v2)?;

        assert_eq!(k.dims(), &[1, 8, 5, 64]); // 3 + 2
        assert_eq!(v.dims(), &[1, 8, 5, 64]);
        assert_eq!(cache.current_seq_len(), 5);

        Ok(())
    }

    #[test]
    fn test_concat_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(2);

        let k = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        let v = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        cache.append(&k, &v)?;

        assert_eq!(cache.current_seq_len(), 10);

        cache.reset();

        assert!(cache.is_empty());
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.k().is_none());
        assert!(cache.v().is_none());

        Ok(())
    }

    #[test]
    fn test_concat_cache_multiple_appends() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(2);

        // Simulate autoregressive generation
        let k_prefill = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        let v_prefill = Tensor::zeros((1, 8, 10, 64), DType::F32, &device)?;
        cache.append(&k_prefill, &v_prefill)?;

        assert_eq!(cache.current_seq_len(), 10);

        // Decode phase: append one token at a time
        for i in 1..=5 {
            let k_token = Tensor::zeros((1, 8, 1, 64), DType::F32, &device)?;
            let v_token = Tensor::zeros((1, 8, 1, 64), DType::F32, &device)?;
            let (k, v) = cache.append(&k_token, &v_token)?;
            assert_eq!(k.dims()[2], 10 + i);
            assert_eq!(v.dims()[2], 10 + i);
        }

        assert_eq!(cache.current_seq_len(), 15);

        Ok(())
    }

    #[test]
    fn test_concat_cache_different_dim() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ConcatKvCache::new(1); // Concatenate on dim 1 instead of 2

        let k1 = Tensor::zeros((1, 3, 8, 64), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 3, 8, 64), DType::F32, &device)?;
        let (k, _v) = cache.append(&k1, &v1)?;

        assert_eq!(k.dims(), &[1, 3, 8, 64]);

        let k2 = Tensor::zeros((1, 2, 8, 64), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 2, 8, 64), DType::F32, &device)?;
        let (k, _v) = cache.append(&k2, &v2)?;

        assert_eq!(k.dims(), &[1, 5, 8, 64]); // Concatenated on dim 1
        assert_eq!(cache.current_seq_len(), 5);

        Ok(())
    }
}
