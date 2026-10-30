# Composed-ops Qwen3.5 forward (ROADMAP row 115): formulas, op mapping, oracle, before any code

Facts read from the HF reference `transformers 5.22.1` `models/qwen3_5/modeling_qwen3_5.py` (installed in a throwaway venv,
CPU torch 2.15) and from Fuel at `origin/main@c3150e596`. Tags: **READ** (source opened), **RAN** (executed, output seen),
**NOT MEASURED**.

## 1. The reference exists and agrees with itself

**RAN** `Qwen/Qwen3.5-0.8B` (hidden 1024, 24 layers: 18 `linear_attention` + 6 `full_attention`, vocab 248320, tied
embeddings) in f32 on CPU, 8 tokens `[9707, 11, 1879, 0, 358, 1079, 264, 1273]`: the whole-sequence forward and the
token-at-a-time cached forward give the same last-token argmax (token 11) and `max |logits difference| = 1.76e-5` on the
last position. So the oracle is self-consistent between prefill and decode, which is the property row 117 needs. The
captured activations (embedding, layers 0, 1, 3, 23, final norm, logits, per-step logits) are saved to a safetensors file
outside the repo; the Rust test must not depend on a multi-hundred-MB download in CI (section 6).

## 2. Gated DeltaNet layer (`linear_attn.*`), exactly as the reference computes it

All math in f32. Per layer, input `x [T, hidden]` (already `input_layernorm`-ed by the decoder layer):

1. `mixed = in_proj_qkv(x)` `[T, 2*key_dim + value_dim]`; `z = in_proj_z(x)` `[T, value_dim]` reshaped `[T, nv, dv]`;
   `b = in_proj_b(x)` `[T, nv]`; `a = in_proj_a(x)` `[T, nv]`. No biases anywhere in this layer (**READ**: `bias=False`).
2. Depthwise **causal** conv over the channel axis of `mixed` (kernel `K = linear_conv_kernel_dim`, 4), **no bias**, then
   **SiLU** (`hidden_act`). Causal means the first `K-1` positions see zeros before the sequence (or the carried conv state).
3. Split into `q [T, nk, dk]`, `k [T, nk, dk]`, `v [T, nv, dv]` (widths `key_dim, key_dim, value_dim`).
4. `beta = sigmoid(b)`; `g = -exp(A_log) * softplus(a + dt_bias)` (per value head, f32). `g <= 0`, so `exp(g)` decays.
5. If `nv > nk`: repeat each q/k head `nv/nk` times with **`repeat_interleave`** (head `h` of q/k serves value heads
   `h*r .. h*r + r-1`), not a tiled repeat. (For the 0.8B `nv == nk == 16`; for the 27B the ratio differs, so this is
   exactly the kind of convention an oracle on the 0.8B cannot catch.)
6. **L2-normalise q and k** over the head dim, `eps = 1e-6` (`x / sqrt(sum(x^2) + eps)`), then scale `q` by `1/sqrt(dk)`.
7. Recurrence, per value head, state `S [dk, dv]` starting at zero (or the carried state), for each token `t`:
   `S <- S * exp(g_t)`;  `kv = S^T k_t` (`[dv]`);  `delta = (v_t - kv) * beta_t`;  `S <- S + k_t delta^T`;  `o_t = S^T q_t`.
8. Gated RMS norm per value head over `dv`: `o <- weight * (o / sqrt(mean(o^2) + eps)) * silu(z)` with `eps = rms_norm_eps`
   (1e-6). The norm weight here is **not** zero-centred (`linear_attn.norm.weight`, the loader already stores it as is).
9. `out = out_proj(o reshaped [T, value_dim])`.

State carried between calls: `S` per layer (`[nv, dk, dv]`, f32, **fixed size**) and the conv window (the last `K-1`
inputs to the conv, `[conv_dim, K-1]`). This is the state row 117's `Fixed` variant describes.

## 3. Full-attention layer (`self_attn.*`)

1. `qg = q_proj(x)` is `[T, nh * 2 * hd]`, viewed `[T, nh, 2*hd]` and **chunked in two along the last axis**: first half is
   the query, second half the gate (this is a per-head split, not the first half of the flat vector).
2. `q = q_norm(q)`, `k = k_norm(k_proj(x))` are RMS norms over `hd` (zero-centred: weight `1 + w`, the loader already stores
   `1 + w`); `v = v_proj(x)`.
3. Partial rotary: `rotary_dim = int(hd * partial_rotary_factor)` (256 * 0.25 = 64); the **first `rotary_dim`** of each head is
   rotated with the **rotate-half** convention (`cat(-x2, x1)`, NOT interleaved pairs), the rest passes through. `inv_freq =
   1 / theta^(2i/rotary_dim)`, `theta = 1e7`. Text-only: the three mrope position streams are identical, so the
   recomposition is a no-op and this is ordinary 1-D RoPE.
4. GQA attention, scale `hd^-0.5`, causal; `attn * sigmoid(gate)`; `o_proj`.

## 4. Mapping to Fuel ops (what exists, what has to be composed)

From the audit note section 2 (**READ** against the kernel contracts): `CausalConv1d` exists on CPU and CUDA, not Vulkan;
`SiLU`, `sigmoid`, `exp`, elementwise, `matmul`, `reduce`, `rsqrt` exist everywhere; `softplus` and `L2Norm` are composed
(`log(1 + exp(x))`, `x * rsqrt(sum(x^2) + eps)`). The recurrence has **no kernel and no single op**: the first
implementation unrolls the token loop into graph nodes (about 25 per token per layer, audit-note arithmetic), which is
correct for the 8-token oracle (about 3.6k nodes for 18 layers) and impractical past a few hundred tokens. That limit is
real and is what rows 116 (chunkwise) and 122 (fused) exist for. Row 115 must **not** be sold as a usable long-prompt path.

`softplus` needs a numerically safe form: `log(1 + exp(x))` overflows for large `x`; the reference (`F.softplus`) switches to
the identity above a threshold. The composed form is `max(x, 0) + log(1 + exp(-|x|))`, which is exact in f32 and has no
branch.

## 5. Steps (red-first; each step's gate named)

1. Per-primitive oracles in Rust against numbers computed by the Python reference on **seeded random small inputs** (not
   the real checkpoint): conv + SiLU, L2-norm + scaling, the recurrence over 6 tokens, the gated norm, partial rotary,
   the gated attention. Each comes with a sabotage sibling (a wrong head-repeat order, an interleaved rotary, a tiled
   instead of interleaved repeat, a flat instead of per-head q/gate split must fail the matching test).
2. One GDN layer and one full-attention layer against the reference layer outputs on random weights.
3. The whole model on the real 0.8B checkpoint against the saved activations (layers 0, 1, 3, 23, final norm, logits).
   Gate: `cargo test -p fuel-transformers --test qwen3_5_forward -- --ignored` style, run locally; CI cannot hold a
   1.7 GB download, so it runs only when the checkpoint path env var is set and is otherwise reported as ignored, never ok.
4. Decode: the same layer with carried `(S, conv window)` stepping one token at a time must equal the whole-sequence forward
   on the last position within a sabotage-calibrated tolerance (the reference shows 1.8e-5 on logits).
5. `Qwen3_5Model::forward` stops returning the typed refusal; GAP-368 narrowed (prefill-correct, slow), not closed.

## 6. What this note does not settle

- Tolerances: not chosen. They are calibrated by sabotage per step 1, not inherited from the 1.8e-5 above.
- bf16: the checkpoint is bf16; the oracle is f32 (the reference upcast the weights). A bf16 run is a separate row.
- The 27B's `nv/nk` ratio and head dims (config not read); step 1's repeat-order sabotage exists because of that.
- The Vulkan path (no conv contract): row 115 targets CPU first, then CUDA; Vulkan needs the conv composed from shifted
  multiply-adds (audit note, **NOT MEASURED**).
- Where the real-checkpoint fixture lives so the test is reproducible without the venv: a small committed fixture of the
  layer-0 and layer-3 weights is not decided (licence and size).
