# rung-2 shifted-prefix donation — Increment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Place an already-computed KV prefix at a non-zero, block-aligned position offset `M` in a sharer session, by materializing fresh blocks whose keys are the cached keys uniformly delta-rotated by θ·M (values copied verbatim). No new numeric Op, no attention-kernel changes.

> ## STATUS (2026-08-05) — C1–C3 SHIPPED (exact); C4/C5 outcome changed
>
> Tasks 1–3 are done and committed on `feat/rope-rung2` (`e4856895`, `cafe3f57`, `92e4e948`, `72cb70b1`): the delta-rotation primitive, the block-aligned bookkeeping, and `DeviceKvPool::splice_prefix_shifted` — all exact, born-red, mutation-checked where applicable.
>
> **Task 4 was reshaped and Task 5 dropped.** Implementation revealed (see the spec's CORRECTION) that exact mid-prompt reuse is impossible for multi-layer models — the primitive is POSITION-exact but the reused prefix loses the preamble's context at layers > 0 (`n_layers=1` maxdiff 0, `n_layers=2` ~2.1e-3). So Task 4 is NOT the `maxdiff==0` anchor below; it is a **characterization test** (`shifted_prefix_reuse_is_exact_at_depth_1_and_lossy_deeper`) that pins the exact-at-depth-1 / lossy-deeper boundary. The **scheduler reference caller (C5) is parked** — it would be an approximate feature with no consumer (Lightbulb serves the rung-1 start-prefix case). The `maxdiff==0` Task 4 text below is retained only as the record of what was originally planned.

**Architecture:** Three layers, bottom-up. (C1) a RoPE **delta-rotation primitive** in `lazy.rs` — constant-position tables + a per-block f32 rotation that reuses the model's exact `rope_with_tables_decomposed`. (C3+core) pure-core **bookkeeping** in `kv_block_pool.rs` that validates the block-aligned offset and allocates fresh dst blocks, returning `(M, src→dst physical pairs)`. (C2) the **product** `DeviceKvPool::splice_prefix_shifted` that drives the two: rotate each src K block → fresh dst K block, copy each src V block → fresh dst V block. (C4) a model-level **logit-parity anchor**. Spec: `docs/superpowers/specs/2026-08-05-rope-rung2-shifted-prefix-design.md`.

**Tech stack:** Rust, fuel-core lazy DAG. Reuses `LazyTensor::{from_f32, const_f32_like, rope_with_tables_decomposed, realize_f32}`, `fuel_graph::build_rope_tables`, `KvBlockPool::{append, resident_block, session_block_table, filled_tokens, prefixes registry, take_free}`, `DeviceKvPool::{read_block, write_block, read_block_bytes, write_block_bytes, geometry, device, core/core_mut}`, `LlamaModel::forward_paged_step`.

## Global Constraints

- One cargo invocation at a time; `-p fuel-core` only; never workspace-wide.
- TDD: born-red observed before green. Never-panic (`Result`). f32/CPU-verifiable throughout.
- Commit-producing work on branch `feat/rope-rung2` (worktree `../fuel-crash-vmm`), branched from PUSHED `origin/main`; re-fetch right before any push.
- Exactness via REUSE: the rotation goes through the model's `rope_with_tables_decomposed` + the canonical `fuel_graph::build_rope_tables`, never a hand-rolled rotation (the interleaved-vs-rotate-half trap).
- Block-aligned offset only (`M % block_size == 0`); materialize (copy, not refcount-share); f32 pool. Non-aligned offsets, bf16, and the scheduler reference caller are out of this increment.
- Oracle discipline: KV-state correctness is asserted on LOGITS (`maxdiff == 0`), never sampled tokens (`tiny_model`'s greedy argmax is a fixed point — a real ~1e-3 perturbation hides under it).

---

### Task 1 — RoPE delta-rotation primitive (C1)

**Files:**
- Modify: `fuel-core/src/lazy.rs` — add `LazyTensor::rope_delta_tables_const` beside `rope_tables_const` (~5356); add free fn `rope_delta_rotate_block_f32` beside it.
- Test: `fuel-core/src/lazy.rs` `#[cfg(test)]` (beside the existing rope tests, ~1825).

**Interfaces:**
- Produces:
  - `LazyTensor::rope_delta_tables_const(&self, theta: f64, delta: usize, rows: usize, head_dim: usize) -> (Self, Self)` — cos/sin `[rows, head_dim]`, **every row identical** = the position-`delta` row.
  - `rope_delta_rotate_block_f32(dev: &Device, k_block: &[f32], theta: f64, delta: usize, block_size: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32>` — rotate a K block (post-RoPE at its original positions) by a uniform θ·`delta`, returning the shifted K block, same length/layout.

- [ ] **Step 1: Write the failing test.** Exactness: a K block rotated at `start_pos = p0` then delta-rotated by `M` equals the same raw K rotated directly at `start_pos = p0 + M`.

```rust
#[test]
fn rope_delta_rotate_equals_direct_shift() {
    let dev = Device::cpu();
    let (n_kv_heads, head_dim, bs) = (2usize, 4usize, 4usize);
    let theta = 10000.0;
    let (p0, m) = (0usize, 8usize); // block at original positions 0..4, shift by 8 → 8..12
    // raw K for one block: [1, n_kv_heads, bs, head_dim]
    let raw: Vec<f32> = (0..n_kv_heads * bs * head_dim).map(|i| (i as f32 * 0.017).sin()).collect();
    let shape = Shape::from_dims(&[1, n_kv_heads, bs, head_dim]);
    let k = LazyTensor::from_f32(std::sync::Arc::from(raw.clone()), shape.clone(), &dev);
    // direct: rope at positions p0+m .. p0+m+bs
    let (c_dir, s_dir) = k.rope_tables_const(theta, p0 + m, bs, head_dim);
    let direct = k.rope_with_tables_decomposed(&c_dir, &s_dir).unwrap().realize_f32();
    // delta path: rope at p0, then uniform delta m
    let (c0, s0) = k.rope_tables_const(theta, p0, bs, head_dim);
    let at_p0 = k.rope_with_tables_decomposed(&c0, &s0).unwrap().realize_f32();
    let shifted = rope_delta_rotate_block_f32(&dev, &at_p0, theta, m, bs, n_kv_heads, head_dim);
    let maxdiff = direct.iter().zip(&shifted).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(maxdiff < 1e-5, "delta-rotate == direct shift (maxdiff {maxdiff})");
}
```

- [ ] **Step 2: Run → RED** (`rope_delta_tables_const` / `rope_delta_rotate_block_f32` missing).

Run: `cargo test -p fuel-core rope_delta_rotate_equals_direct_shift`

- [ ] **Step 3: Implement.**

```rust
// beside rope_tables_const (~5370):
pub fn rope_delta_tables_const(
    &self, theta: f64, delta: usize, rows: usize, head_dim: usize,
) -> (Self, Self) {
    // The one position-`delta` row (NOT build_rope_tables(.., rows, ..), which
    // would give the incrementing progression delta, delta+1, …).
    let (c1, s1) = fuel_graph::build_rope_tables(theta, delta, 1, head_dim);
    let mut cos = Vec::with_capacity(rows * head_dim);
    let mut sin = Vec::with_capacity(rows * head_dim);
    for _ in 0..rows { cos.extend_from_slice(&c1); sin.extend_from_slice(&s1); }
    let shape = Shape::from_dims(&[rows, head_dim]);
    (self.const_f32_like(cos, shape.clone()), self.const_f32_like(sin, shape))
}

// free fn beside it:
pub fn rope_delta_rotate_block_f32(
    dev: &Device, k_block: &[f32], theta: f64, delta: usize,
    block_size: usize, n_kv_heads: usize, head_dim: usize,
) -> Vec<f32> {
    let shape = Shape::from_dims(&[1, n_kv_heads, block_size, head_dim]);
    let k = LazyTensor::from_f32(std::sync::Arc::from(k_block.to_vec()), shape, dev);
    let (cos, sin) = k.rope_delta_tables_const(theta, delta, block_size, head_dim);
    k.rope_with_tables_decomposed(&cos, &sin)
        .expect("rope_delta_rotate_block_f32: rope")
        .realize_f32()
}
```

- [ ] **Step 4: Run → GREEN.** Regress the existing rope tests: `cargo test -p fuel-core rope`.
- [ ] **Step 5: Commit** `feat(rope): uniform delta-rotation primitive (constant-M tables) for shifted prefixes`.

---

### Task 2 — Core bookkeeping: block-aligned validate + fresh-block allocation (C3 + core)

**Files:**
- Modify: `fuel-core/src/kv_block_pool.rs` — add `KvAllocError::OffsetNotBlockAligned`; add `KvBlockPool::alloc_shifted_prefix_slots`.
- Test: same file `#[cfg(test)]` (beside the `splice_prefix` tests).

**Interfaces:**
- Consumes: `PrefixId`, the `prefixes` registry (`PrefixOwner { owner, prefix_blocks }`), `filled_tokens`, `take_free`, `refcount`, `table_mut`, `resident_block`.
- Produces:
  - `KvAllocError::OffsetNotBlockAligned { filled: usize, block_size: usize }`
  - `KvBlockPool::alloc_shifted_prefix_slots(&mut self, prefix: PrefixId, dst: SessionHandle) -> Result<(usize, Vec<(PhysBlockId, PhysBlockId)>), KvAllocError>` — returns `(m_offset, pairs)` where `m_offset = filled_tokens(dst)` and each pair is `(src_owner_phys, fresh_dst_phys)` in prefix-block order. Validates BEFORE mutating: dst open; `m_offset % block_size == 0` (else `OffsetNotBlockAligned`); prefix registered (`UnknownPrefix`); owner fully filled (`PrefixNotFullyFilled`); `prefix_blocks` free blocks available (`OutOfBlocks`). On success: allocates `prefix_blocks` fresh blocks (refcount 1) appended to `dst.slots`, bumps `dst.filled_tokens` by `prefix_blocks * block_size`. A refusal mutates nothing.

- [ ] **Step 1: Write the failing test.**

```rust
#[test]
fn alloc_shifted_prefix_slots_validates_and_allocates() {
    let mut pool = KvBlockPool::new(test_geom(/*bs*/4, /*blocks*/64));
    // donor: 2 full blocks (8 tokens); register a prefix
    let donor = pool.open();
    pool.append(donor, 8).unwrap();
    let pid = pool.register_prefix(donor, 2).unwrap();
    // dst prefilled to a NON-aligned offset → refusal, pool untouched
    let dst = pool.open();
    pool.append(dst, 5).unwrap();
    let free0 = pool.free_blocks();
    assert!(matches!(pool.alloc_shifted_prefix_slots(pid, dst),
        Err(KvAllocError::OffsetNotBlockAligned { filled: 5, block_size: 4 })));
    assert_eq!(pool.free_blocks(), free0, "refusal allocates nothing");
    assert_eq!(pool.filled_tokens(dst), Some(5), "refusal does not bump fill");
    // aligned dst (8 tokens): allocates 2 fresh blocks, returns pairs
    let dst2 = pool.open();
    pool.append(dst2, 8).unwrap();
    let (m, pairs) = pool.alloc_shifted_prefix_slots(pid, dst2).unwrap();
    assert_eq!(m, 8);
    assert_eq!(pairs.len(), 2);
    assert_eq!(pool.filled_tokens(dst2), Some(16), "fill bumped by 2*4");
    // fresh dst blocks are exclusive (refcount 1), distinct from the owner's
    for (src, fresh) in &pairs {
        assert_eq!(pool.block_refcount(*fresh), 1, "fresh dst block is exclusive");
        assert_ne!(src, fresh, "dst block is a COPY target, not the shared original");
    }
}
```
(Use the file's existing test helpers for `test_geom`; mirror the `splice_prefix` tests' setup.)

- [ ] **Step 2: Run → RED.**
- [ ] **Step 3: Implement** (validate-before-mutate; look up owner from `prefixes`; read owner block table; `take_free` per block).

```rust
pub fn alloc_shifted_prefix_slots(
    &mut self, prefix: PrefixId, dst: SessionHandle,
) -> Result<(usize, Vec<(PhysBlockId, PhysBlockId)>), KvAllocError> {
    let bs = self.geom.block_size;
    let m = *self.tables.get(&dst).ok_or(KvAllocError::UnknownSession)?
        .filled_tokens_ref(); // or: .filled_tokens
    if m % bs != 0 {
        return Err(KvAllocError::OffsetNotBlockAligned { filled: m, block_size: bs });
    }
    let (owner, prefix_blocks) = {
        let o = self.prefixes.get(&prefix).ok_or(KvAllocError::UnknownPrefix)?;
        (o.owner, o.prefix_blocks)
    };
    // owner is fully-filled by construction (register_prefix enforced it); re-check.
    let owner_filled = self.tables.get(&owner).ok_or(KvAllocError::UnknownSession)?.filled_tokens;
    if prefix_blocks * bs > owner_filled {
        return Err(KvAllocError::PrefixNotFullyFilled { prefix_blocks, donor_filled: owner_filled });
    }
    if prefix_blocks > self.free.len() {
        return Err(KvAllocError::OutOfBlocks { need: prefix_blocks, have: self.free.len() });
    }
    let src: Vec<PhysBlockId> = (0..prefix_blocks)
        .map(|i| self.resident_block(owner, i).expect("owner block resident"))
        .collect();
    let mut pairs = Vec::with_capacity(prefix_blocks);
    for &s in &src {
        let fresh = self.take_free()?; // pre-checked; cannot fail
        self.refcount[fresh as usize] = 1;
        self.table_mut(dst)?.slots.push(Slot::Resident(fresh));
        pairs.push((s, fresh));
    }
    self.table_mut(dst)?.filled_tokens += prefix_blocks * bs;
    Ok((m, pairs))
}
```
(Note: read `filled_tokens` via the existing field access used elsewhere in this file, e.g. `self.tables.get(&dst)?.filled_tokens`; the `filled_tokens_ref` above is shorthand — use the real field.)

- [ ] **Step 4: Run → GREEN.** Regress: `cargo test -p fuel-core kv_block_pool`.
- [ ] **Step 5: MUTATION-CHECK (confirmed recompile):** delete the `m % bs != 0` guard → rebuild → the `OffsetNotBlockAligned` assertion must go RED; revert.
- [ ] **Step 6: Commit** `feat(kv-pool): alloc_shifted_prefix_slots — block-aligned fresh-block allocation for shifted prefixes`.

---

### Task 3 — The product: `DeviceKvPool::splice_prefix_shifted` (C2)

**Files:**
- Modify: `fuel-core/src/kv_block_pool_device.rs` — add `DeviceKvPool::splice_prefix_shifted`.
- Test: same file `#[cfg(test)]`.

**Interfaces:**
- Consumes: Task 2's `alloc_shifted_prefix_slots`; Task 1's `rope_delta_rotate_block_f32`; `read_block(layer, K, phys)`, `write_block(layer, K, phys, &[f32])`, `read_block_bytes(layer, V, phys)`, `write_block_bytes(layer, V, phys, &bytes)`, `geometry()`, `device()`, `core_mut()`.
- Produces: `DeviceKvPool::splice_prefix_shifted(&mut self, prefix: PrefixId, dst: SessionHandle, rope_base: f64) -> Result<usize, KvAllocError>` — returns `shared_tokens = prefix_blocks * block_size`. For each `(src, fresh)` pair × each layer: rotate the K block by θ·M into `fresh`; copy the V block verbatim into `fresh`.

- [ ] **Step 1: Write the failing test** — the rotated dst K block equals a direct rope-at-shifted-position of the same raw content, and V is byte-identical. Build the donor by writing KNOWN raw K rotated at pos 0..8 (via `forward_paged_step` on a tiny model, or by direct `write_block` of a rope'd reference), register, splice-shifted into an aligned dst, then compare `read_block(K, fresh)` to `rope_delta_rotate_block_f32(read_block(K, src), M)` and `read_block_bytes(V, fresh) == read_block_bytes(V, src)`.

```rust
#[test]
fn splice_prefix_shifted_rotates_k_copies_v() {
    let geom = KvGeometry { n_layers: 2, n_kv_heads: 2, head_dim: 4,
        num_blocks: 64, block_size: 4, elem_size: DType::F32.size_in_bytes() };
    let mut pool = DeviceKvPool::new(geom, DType::F32, &Device::cpu()).unwrap();
    let rope_base = 10000.0;
    // donor: 2 full blocks of arbitrary K/V bytes
    let donor = pool.core_mut().open();
    pool.core_mut().append(donor, 8).unwrap();
    for (l, blk) in [(0usize, 0usize)] { let _ = (l, blk); } // (fill via write_block below)
    // ... write deterministic K,V into donor's 2 blocks for each layer ...
    let pid = pool.core_mut().register_prefix(donor, 2).unwrap();
    let dst = pool.core_mut().open();
    pool.core_mut().append(dst, 8).unwrap(); // aligned M=8
    let n = pool.splice_prefix_shifted(pid, dst, rope_base).unwrap();
    assert_eq!(n, 8);
    // dst's shifted blocks are slots 2,3 (after the 2 preamble blocks)
    for (i, src_i) in [(2usize, 0usize), (3, 1)] {
        let src = pool.core().resident_block(donor, src_i).unwrap();
        let fresh = pool.core().resident_block(dst, i).unwrap();
        for l in 0..2 {
            let want = rope_delta_rotate_block_f32(
                &Device::cpu(), &pool.read_block(l, BlockKind::K, src).unwrap(),
                rope_base, 8, 4, 2, 4);
            let got = pool.read_block(l, BlockKind::K, fresh).unwrap();
            let md = want.iter().zip(&got).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            assert!(md < 1e-5, "K rotated (layer {l}, maxdiff {md})");
            assert_eq!(pool.read_block_bytes(l, BlockKind::V, fresh).unwrap(),
                       pool.read_block_bytes(l, BlockKind::V, src).unwrap(), "V copied verbatim");
        }
    }
}
```

- [ ] **Step 2: Run → RED.**
- [ ] **Step 3: Implement.**

```rust
pub fn splice_prefix_shifted(
    &mut self, prefix: crate::kv_block_pool::PrefixId,
    dst: crate::kv_block_pool::SessionHandle, rope_base: f64,
) -> Result<usize, crate::kv_block_pool::KvAllocError> {
    let g = self.geometry();
    let (m, pairs) = self.core_mut().alloc_shifted_prefix_slots(prefix, dst)?;
    for (src, fresh) in &pairs {
        for l in 0..g.n_layers {
            let k = self.read_block(l, BlockKind::K, *src)?;
            let rot = crate::lazy::rope_delta_rotate_block_f32(
                self.device(), &k, rope_base, m, g.block_size, g.n_kv_heads, g.head_dim);
            self.write_block(l, BlockKind::K, *fresh, &rot)?;
            let v = self.read_block_bytes(l, BlockKind::V, *src)?;
            self.write_block_bytes(l, BlockKind::V, *fresh, &v)?;
        }
    }
    Ok(pairs.len() * g.block_size)
}
```
(If `alloc_shifted_prefix_slots` succeeds but a later block op errors, the dst slots are already appended — acceptable for increment 1 since read/write_block on freshly-allocated resident blocks cannot fail under the pre-checks; document that the numeric phase is infallible given valid geometry.)

- [ ] **Step 4: Run → GREEN.** Regress: `cargo test -p fuel-core kv_block_pool_device`.
- [ ] **Step 5: Commit** `feat(kv-pool): DeviceKvPool::splice_prefix_shifted — materialize a delta-rotated prefix at a block-aligned offset`.

---

### Task 4 — Model-level correctness anchor (C4, the serving-value proof)

**Files:**
- Test: `fuel-core/tests/paged_decode_parity.rs` (mirrors rung-1's `prefix_shared_session_decodes_like_from_scratch`).

**Interfaces:**
- Consumes: `LlamaModel::forward_paged_step`, `DeviceKvPool`, `register_prefix`, `splice_prefix_shifted`. Reuse the file's `tiny_cfg`/`tiny_weights`/`tiny_model` + block helpers.

- [ ] **Step 1: Write the failing test** — `shifted_prefix_session_decodes_like_from_scratch`. Full prompt = `[preamble (block-aligned M tokens)][shared prefix (N tokens)][suffix]`. Reference: a from-scratch session prefills the WHOLE prompt via `forward_paged_step`, collecting per-step logits. Shared path: donor prefills the shared prefix (N tokens, whole blocks), `register_prefix`; a sharer prefills the preamble (M tokens, block-aligned), `splice_prefix_shifted(pid, sharer, rope_base)`, prefills the suffix, decodes — collecting per-step logits. Assert **maxdiff == 0** across all collected logits (rung-1 measured the shifted-vs-direct primitive exact at f32; the end-to-end should be exact too — if a tiny nonzero drift appears from realize reassociation, calibrate a `< 1e-6` bound against a sabotage run, per [[sabotage-test-calibration]], and document it).

```rust
// sketch — mirror prefix_shared_session_decodes_like_from_scratch:
let model = tiny_model(9999);
let bs = 4; let rope_base = model.config.rope_base;
let preamble = [1u32,2,3,4];            // M=4 (block-aligned)
let shared   = [5u32,6,7,8,9,10,11,12]; // N=8 (2 whole blocks)
let suffix   = [13u32,14,15];
let full: Vec<u32> = preamble.iter().chain(&shared).chain(&suffix).copied().collect();
// reference logits: whole prompt from scratch, then K decode steps ...
// shared: donor prefills `shared` @ pos 0..8, register 2 blocks;
//         sharer prefills `preamble` (M=4) → splice_prefix_shifted → prefills `suffix` → decode;
// assert logits_maxdiff(shared_steps, scratch_steps) == 0
```

- [ ] **Step 2: Run → RED** (either the anchor fails because `splice_prefix_shifted` isn't wired end-to-end, or — once wired — it is the first green proof).
- [ ] **Step 3: GREEN** after Tasks 1–3.
- [ ] **Step 4: MUTATION-CHECK (confirmed recompile):** in `splice_prefix_shifted`, pass `m + 1` (or `0`) as the delta → rebuild → the anchor's `maxdiff == 0` must go RED (the shifted prefix is now mis-positioned); revert. This proves the anchor has teeth that token equality would not.
- [ ] **Step 5: Commit** `test(rung-2): shifted-prefix decode parity anchor (logit oracle, mutation-verified)`.

---

## Self-Review

- **Spec coverage:** C1 (Task 1), C3+core bookkeeping (Task 2), C2 product (Task 3), C4 anchor (Task 4). C5 (scheduler reference caller) is increment 2, out of scope here — noted in the spec.
- **Type consistency:** `alloc_shifted_prefix_slots` returns `(usize, Vec<(PhysBlockId, PhysBlockId)>)` and `splice_prefix_shifted` consumes exactly that; `rope_delta_rotate_block_f32`'s signature is identical at its definition (Task 1) and both call sites (Task 3 impl + Task 3 test). `rope_delta_tables_const` produces `[rows, head_dim]` matching `rope_with_tables_decomposed`'s table expectation.
- **Placeholder scan:** the two `filled_tokens` shorthands in Task 2's code are flagged inline to use the real field access; the Task 3/4 test bodies are sketches that reuse existing file helpers (do not invent new scaffolding — mirror rung-1's tests).
- **Teeth:** Tasks 2 and 4 carry explicit mutation-checks with confirmed recompilation; Task 4 asserts on logits, never tokens.
