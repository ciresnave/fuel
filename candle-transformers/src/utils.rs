//! Shared utilities: repeat_kv, repeat_penalty, causal mask.

use candle::{Device, Result, Tensor};

/// Build a causal attention mask of shape `(seq_len, kv_len)` where
/// `kv_len = index_pos + seq_len`.
///
/// `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
///
/// - `index_pos == 0`: classic square `(seq_len, seq_len)` mask.
/// - `index_pos > 0`: rectangular mask for prefix KV caching — the first
///   `index_pos` columns are all-zero (every query attends to all cached prefix
///   keys) and the last `seq_len` columns form the standard causal triangle.
///
/// All models that maintain a KV cache should use this function so that
/// batched user-turn prefill works correctly after prefix restoration.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

/// Fill `on_false` with `on_true` where `mask` is true (non-zero).
///
/// Equivalent to PyTorch's `Tensor.masked_fill_(mask, value)`.
/// The `on_true` scalar is automatically cast to the dtype of `on_false`.
pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let logits = logits.to_dtype(candle::DType::F32)?;
    let vocab_size = logits.elem_count();

    // Build a penalty mask on the CPU: 1.0 at context token positions, 0.0 elsewhere.
    // This is much smaller to transfer than pulling the entire logits tensor to CPU
    // and avoids per-token GPU-CPU synchronization.
    let mut penalty_mask = vec![0f32; vocab_size];
    let mut already_seen = std::collections::HashSet::new();
    for &token_id in context {
        if already_seen.insert(token_id) {
            let idx = token_id as usize;
            if idx < vocab_size {
                penalty_mask[idx] = 1.0;
            }
        }
    }
    let penalty_mask =
        Tensor::from_vec(penalty_mask, vocab_size, device)?;

    // For tokens in the context:
    //   positive logits  -> divide by penalty (multiply by 1/penalty)
    //   negative logits  -> multiply by penalty
    // For tokens NOT in the context: keep the original value (multiply by 1.0).
    let inv_penalty = Tensor::full(1.0f32 / penalty, vocab_size, device)?;
    let mul_penalty = Tensor::full(penalty, vocab_size, device)?;
    let ones = Tensor::ones(vocab_size, candle::DType::F32, device)?;

    // sign_mask: 1 where logit >= 0, 0 where logit < 0
    let sign_mask = logits.ge(0f32)?;
    // per-element factor for tokens in context: 1/penalty if positive, penalty if negative
    let context_factor = sign_mask.where_cond(&inv_penalty, &mul_penalty)?;
    // Only apply the factor where the penalty mask is set; use 1.0 elsewhere.
    let mask_bool = penalty_mask.ge(1f32)?;
    let final_factor = mask_bool.where_cond(&context_factor, &ones)?;

    logits.mul(&final_factor)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
