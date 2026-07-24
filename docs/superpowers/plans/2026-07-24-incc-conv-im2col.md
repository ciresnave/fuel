# Increment C — Thread `im2col-basis`: migrate the conv family (`conv2d`, `conv_transpose_2d`) to total `PatternNode` recipes **without a new primitive**

**Verdict: `BUILD_AS_DESIGNED` — there is NO basis extension.** The premise that the conv family needs an `Op::Im2Col` / `Op::Col2Im` primitive is **wrong**. im2col is an overlapping-window strided gather (`Op::IndexSelect`/`Op::Gather`); col2im is its scatter-add adjoint (`Op::IndexAdd`/`Op::ScatterAdd`). Both, plus the window→index map (`Op::Iota` + arithmetic + `Op::Cast`), the `Op::Pad`, and the batched/grouped matmul (`Op::MatMul`), are **already in the closed `Op` basis and already re-emittable by `tag_to_op`**. conv2d / conv_transpose_2d migrate like any other Increment-C op. **Flag this loudly:** the four "BASIS-GAP" self-returns in the plan-of-record drop to **two** (only `qmatmul` and `inplace_affine` remain genuine basis questions).

## 1. Why the module docs are stale (the load-bearing correction)

`registry/conv2d.rs:8-44,110-129` and `conv_transpose_2d.rs:8-22,106-119` argue that no clean decomposition exists because "synthesizing Conv2D from `Slice`+`MatMul`+`Concat` … creates `N·Hout·Wout` slice operations — astronomical node count." **That is the only alternative those docs considered.** They pre-date (or overlook) the **index-gather idiom**, which has *constant* node count independent of spatial size:

- **im2col = `IndexSelect`** over the flattened padded-spatial axis. The same sliding-window pattern applies to every `(n, c)`, so one **1-D index** of length `Kh·Kw·Hout·Wout` gathers the whole patch matrix. Overlapping windows just repeat index values — a gather reads a source position as many times as needed.
- The window→flat-index map is built from **`Op::Iota`** (its own doc, `lib.rs:230-238`, says it exists to build mask/index tensors) + `MulScalar`/`Add` + **`Op::Cast(U32)`**. Integer-valued, exact in F32 for padded-spatial extent `< 2^24`.
- Padding is **`Op::Pad`** (Constant, `0.0`) — matching the kernel's zero-fill.

The CPU kernel itself is im2col + batched GEMM (`fuel-cpu-backend/src/conv2d.rs:373-414`; `col [b,m,k]`, kernel `broadcast_as((b,k,n))`, `MatMul((b,m,n,k))`). The recipe **mirrors that algorithm at the graph level**.

## 2. Grammar + shape coverage is already complete (no `tag_to_op` / `primitive_shape` work)

| Op needed | `tag_to_op` re-emit | `primitive_shape` rule |
|---|---|---|
| `Pad` | `runtime_fused.rs:382` | `shape.rs` (Pad test :363) |
| `Iota` (leaf) | `:404` (len ← `target_shape`) | `:370` (`&[], &[]`) |
| `Cast(U32)` | `:339` (dtype ← `cast_dtype`) | ✓ |
| `IndexSelect` / `Gather` | `:406` / `:407` (dim ← `axis`) | `:215` (dim ← index len) / `:356` |
| `IndexAdd` / `ScatterAdd` | `:408` / `:409` | `:232` (out = base shape) |
| `MatMul` (batched) | `:348` (empty roles ⇒ implicit accept) | `:340` (`[7,2,3]×[7,3,5]`) |
| `Reshape`/`Permute`/`BroadcastTo`/`Slice`/`MulScalar`/`Add` | `:366-379,315-316` | ✓ |

`emit` (`runtime_fused.rs:1052`) emits children first, then computes each node's shape via `primitive_shape` (the single source of truth). `Op::Iota`'s 0-operand leaf is handled (the `Op::ScanPlaceholder` 0-operand precedent). **MatMul-in-recipe already ships** in `fused_linear.rs:124`. Batched `Op::MatMul` is validated in the builder (`lib.rs:3934-3986`, GQA-divisible batch prefix). The recipe emits an **explicit `BroadcastTo`** of the weight to a same-rank operand before `Op::MatMul` (mirrors the CPU kernel + `fused_linear`).

## 3. conv2d recipe (per-call baked, like `selective_scan`'s `recipe(seqlen, …)`)

`decompose` reads `stride/padding/groups` from params and `Cin/Hin/Win/Cout/Kh/Kw` from the node's input shapes, computes concrete `Hpad,Wpad,Hout,Wout,K`, and builds a `PatternNode` with **concrete `OpAttrs`** (this is the established per-call pattern — `selective_scan` bakes `seqlen`, `powi_backward` constant-folds param-derived values). No open scalar slots ⇒ `decompose_via_recipe(graph, id, &recipe, Some(vec![]))` (`registry.rs:900`). Bind space `0=x, 1=weight, [2=bias]`.

Emitted subgraph (groups=1):
```
xp   = Pad[(0,0),(0,0),(ph,ph),(pw,pw), Constant 0](x)      # [N,Cin,Hpad,Wpad]
xf   = Reshape([N,Cin,Hpad*Wpad])(xp)
idx  = Cast(U32)( build_flat_index() )                       # 1-D [Kh*Kw*Hout*Wout], Iota+arith
cols = IndexSelect(dim=2)(xf, idx)                           # [N,Cin,Kh*Kw*Hout*Wout]
P    = Permute/Reshape(cols) -> [N, Cin*Kh*Kw, Hout*Wout]    # (Cin,Kh,Kw) contraction order = weight order
Wm   = Reshape([Cout, Cin*Kh*Kw])(weight)
Wb   = BroadcastTo([N, Cout, Cin*Kh*Kw])(Wm)
Y    = MatMul(Wb, P)                                          # [N, Cout, Hout*Wout]
out  = Reshape([N, Cout, Hout, Wout])(Y)
[ +Add(BroadcastTo(bias)) if 3 inputs ]
```
`build_flat_index()` = per-axis `Iota(Kh)`, `Iota(Kw)`, `Iota(Hout)`, `Iota(Wout)` reshaped to broadcastable ranks, `MulScalar` by `stride*·Wpad` factors, `Add`-combined, `Reshape` to 1-D. `idx(ky,kx,oh,ow) = (oh·sh+ky)·Wpad + (ow·sw+kx)` (dilation is always 1 for Conv2D — `FusedOpParams::Conv2D` has no dilation field, `registry.rs:199-203`). Contraction ordering `(Cin,Kh,Kw)` matches the weight reshape (`fuel-conv/src/lib.rs:217-220`).

**Groups (slice 2):** reshape `xf → [N, groups, Cin/g, Hpad*Wpad]`, gather, reshape `P → [N, groups, K/g, Hout*Wout]`, weight `→ [groups, Cout/g, K/g]` broadcast to `[N, groups, Cout/g, K/g]`, batched `MatMul → [N, groups, Cout/g, Hout*Wout]`, reshape `[N,Cout,Hout,Wout]`. Depthwise (`groups=Cin`) is the same path. **No Slice/Concat soup.**

## 4. conv_transpose_2d recipe (col2im = scatter-add adjoint)

`ConvTranspose2D` carries `stride/padding/output_padding/dilation/groups` (`registry.rs:225-231`). Transposed conv = matmul then **col2im (overlap-add)**:
```
xf    = Reshape([N,Cin,Hin*Win])(x)
Wm    = Reshape([Cin, Cout*Kh*Kw])(weight)                   # weight is [Cin, Cout/g, Kh, Kw]
cols  = MatMul(Wm^T-arranged broadcast, xf) -> [N, Cout*Kh*Kw, Hin*Win]
csrc  = Reshape/Permute(cols) -> [N, Cout, L]  (L = Kh*Kw*Hin*Win)
base0 = MulScalar(0.0)(BroadcastTo([N,Cout,Sout_pad])(...))  # zero base, no Const in a recipe
sidx  = Cast(U32)( build_scatter_index() )                   # l=(ky,kx,ih,iw) -> (ih*sh+ky*dh)*Wpad_o + (iw*sw+kx*dw)
acc   = IndexAdd(dim=2)(base0, sidx, csrc)                   # overlapping cols accumulate (+=)
out   = Slice-crop(acc by padding) -> Reshape([N,Cout,Hout,Wout])
```
`IndexAdd`'s `+=` semantics ARE the overlap-add. Groups: batched matmul + per-group base, same as conv2d. `output_padding`/`dilation`/`stride` fold into `build_scatter_index()` + the output-buffer size + the crop.

## 5. Parity gate — RISK-B applies fully (net-new decompose, no frozen-legacy to preserve)

conv currently **self-returns**, so there is nothing to preserve bit-exactly and the toy f64 interpreter (`eval_rope`/`eval_norm`) does **not** implement `IndexSelect/Iota/MatMul/Pad/IndexAdd`. The gate is a **real-backend CPU-realize numerical test** (per RISK-B, plan-of-record:44-45):
- conv2d reference = `fuel_conv::conv2d_direct` (`fuel-conv/src/lib.rs:137`); tolerance `rel<1e-5` (the same bound `via_gemm_matches_direct`, `:422-462`, already meets). MatMul reorders summation ⇒ not bit-exact ⇒ calibrated + **sabotage-calibrated** (corrupt the index or swap matmul operands ⇒ oracle RED).
- conv_transpose reference = the **native CPU conv_transpose2d kernel** (`fuel-cpu-backend`), first pinned against a PyTorch fixture.
Tests live in `fuel-core/tests` (need CPU-backend realize; model on `phase7b_conv2d_oracle.rs`). Gate `-p fuel-core` + `-p fuel-graph`, never workspace-wide.

## 6. Why im2col is NOT a basis element (the principled distinction from `Op::Scan`/G3)

A new primitive is justified only when the operation is **inexpressible** in the existing basis. `Op::Scan` (G3) qualified: the SSM recurrence has no finite closed form in the elementwise/gather basis (unbounded unroll; the CumSum closed form overflows for `a<0`). **im2col has no such obstruction — it is ordinary indexing.** A hypothetical `Op::Im2Col` would itself decompose to `Iota`+`IndexSelect`, so it fails the "irreducible base-map terminal" test; adding it would be sugar, not a basis element, and would carry the full cost of a basis extension (03-ir MAJOR bump, shape/dtype rules, CPU realize, GPU kernels-or-decline, `tag_to_op` arm + `OpAttrs` carriers, `base_map_hash` `op_key`) for zero expressive gain.

## 7. Draft `docs/architecture/10-decisions-log.md` addendum (a CORRECTION, not a basis change)

> ### 2026-07-24 — conv2d / conv_transpose_2d are NOT a primitive-basis gap (correction to the 2026-06-20 "recipe principle / total decompose" G2 example)
> The 2026-06-20 entry named **conv2d** as a canonical fused op that self-returns for want of an `Op::Im2Col`. **That characterization was wrong.** im2col is a strided gather of overlapping windows, fully expressible in the existing closed `Op` basis: **im2col = `Op::IndexSelect`/`Op::Gather`** over the `Op::Pad`-padded, flattened spatial axis, with the window→flat-index map built from **`Op::Iota` + scalar arithmetic + `Op::Cast(U32)`** (bounded, constant node count — NOT the `N·Hout·Wout` `Slice`/`Concat` soup the original entry correctly rejected, which was the only alternative it weighed). **col2im (conv_transpose_2d) = the adjoint = a scatter-add = `Op::IndexAdd`/`Op::ScatterAdd`** into a zero-init base (`MulScalar(0)` of a broadcast), then `Op::Slice` crop. The grouped/batched conv matmul = `Op::MatMul`'s batched (rank≥2, GQA-divisible batch-prefix) form. **No `Op` variant is added; the build-time-closed primitive basis is UNCHANGED.** Contrast **G3 / `Op::Scan`**, which genuinely required a new primitive (the SSM recurrence is inexpressible; the CumSum closed form overflows). conv2d/conv_transpose_2d migrate to total `PatternNode` recipes like any other op (Increment C), gated by a real-backend numerical-parity test (`fuel-conv::conv2d_direct` / the native CPU conv_transpose kernel; the toy f64 parity interpreter does not implement these ops). `qmatmul` (sub-byte bit-unpack) and `inplace_affine` (destructive-affine aliasing) remain genuinely-open basis questions, separate from this correction.

Also reclassify these two in `docs/superpowers/plans/2026-07-24-increment-c-decompose-migration.md` (lines 38-42, 147, 262-264) from BASIS-GAP to migratable, and rewrite the "no primitive decomposition (yet)" module docs.

## 8. Sequence
`im2col-1` (conv2d groups=1 + the index-gather helper + numerical harness + docs/decisions-log correction) → `im2col-2` (conv2d groups+bias; extends slice-1 machinery, sequential on `conv2d.rs`) → `im2col-3` (conv_transpose_2d col2im; reuses slice-1's `build_flat_index` helper, own file `conv_transpose_2d.rs`). Worktree isolation: slices 1&2 both touch `conv2d.rs` (serialize); slice 3 is a separate file but should follow slice 1 to reuse the index helper.

