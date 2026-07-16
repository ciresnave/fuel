# Convergence Increment A — Full-parity `emit` via a shared `primitive_shape` Implementation Plan

> **For agentic workers:** execute task-by-task, TDD, checkbox steps. Write the failing test FIRST, run it RED, then implement to GREEN, then commit. Do not batch tasks. Each task ends with an independently-testable deliverable that is the oracle for later tasks.

Design spec: `docs/superpowers/specs/2026-07-15-convergence-A-full-parity-emit-design.md` (read it first).

## Goal

Make the runtime `emit` (`fuel-graph/src/runtime_fused.rs`) re-emit the **full first-order op set** — every op except the 4 basis-gap ops (`conv2d`/`conv_transpose_2d`/`qmatmul`/`inplace_affine`) and the higher-order body-carrying `Op::Scan` — with correct per-node shape **and** dtype, so the Increment C decompose-migration becomes pure data-movement. Achieve it by extracting **one** `primitive_shape(op, &[Shape], &[DType]) -> Result<(Shape, DType)>` function that is the single source of truth for primitive shape+dtype inference, called by BOTH the `Tensor` builders and `emit` (no drift). Extend `OpAttrs` to carry the params the full set needs and to conform to the pinned KISS §6.19 canonical positional-blob grammar. Validate byte-for-byte against the hand-written `registry::{rope,softmax_last_dim,layer_norm_last_dim}::decompose` oracles.

## Architecture

- `emit` today is **elementwise-only**: every re-emitted node takes `operand[0]`'s shape+dtype (`runtime_fused.rs:429-431`), and `tag_to_op` (`runtime_fused.rs:256-295`) covers 32 of ~72 `OpTag`s (binary arith / unary / activations / `AddScalar`/`MulScalar`), returning `None` for everything shape-changing, reducing, dtype-changing, or structural. `validate_representable` (`runtime_fused.rs:320-347`) rejects any region containing an op `tag_to_op` can't build.
- Shape math today lives **inline in each `Tensor` builder** (`try_permute` lib.rs:4622, `cast` :5468, `try_broadcast_to` :5507, `unsqueeze` :5528, `squeeze` :5582, `try_reshape` :5665, `reduce_sum_to` :5688, `reduce_max_to` :5712, `sum_dim`/`mean_dim` via `axis_reduction`, `index_reduction` :5790, `matmul` :3908, `concat` :6797, `slice` :6841, `flip` :5044, `roll` :5071, `masked_fill` :5285, `pad` :5334, `where_cond` :5420, `index_select` :6719, `gather` :6761). Each validates args, computes `out_dims`+dtype, and pushes a `Node`. There is **no** reusable per-op shape-inference fn — that drift hazard is what Task 1 closes.
- The graph `Op` enum (`lib.rs:217`, `#[derive(Debug, Clone, PartialEq)]`) carries op params **on the variant** (`Op::Slice{dim,start,len}`, `Op::Reshape(Shape)`, `Op::Cast(DType)`, `Op::Concat{dim}`, …). So `primitive_shape` reads params off the `Op`, not off `OpAttrs`. `OpAttrs` matters only for `tag_to_op` — reconstructing an `Op` from an `OpTag` + attrs.
- `OpAttrs` (`fuel-kernel-seam-types/src/lib.rs:71`) is an intentionally **dependency-free std-only POD** wire type (its Cargo.toml forbids pulling `fuel_ir`/`fuel-graph`). Today: `scalars: Vec<f64>`, `axis: Option<i64>`, `perm: Vec<u8>`, `target_shape: Vec<i64>`, `dims: Vec<u8>`. It cannot reference `fuel_ir::DType` or `PadMode` — new dtype/mode carriers must be dependency-free integer/string codes.
- `op_to_attrs` (`jit.rs:127-150`) is the forward projection (graph `Op` → `OpAttrs`) used by the matcher; `tag_to_op` (`runtime_fused.rs:256`) is the inverse used by `emit`. Task 2/3 extend the pair in lockstep.
- Structural comparison tool: `crate::opt::base_map_hash(&Graph, NodeId) -> u64` (`opt.rs:399`) — `NodeId`-independent content hash folding `op_key` + shape/dtype, commutative-operand-canonicalized. `crate::opt::lower_to_base_map(&SharedGraph, &[NodeId])` (`opt.rs:364`) drives decompose to fixpoint. Both are already used by `runtime_fused::region_base_map_hash`.

## Tech Stack

Rust (edition 2024). Crates touched: `fuel-graph` (primitive_shape, tag_to_op, emit, builders, oracle tests), `fuel-kernel-seam-types` (OpAttrs fields + canonical serialization). `fuel_ir` provides `Shape`, `DType`, `Error`. Pure graph-layer — **no GPU**.

## Global Constraints (binding — copy discipline exactly)

- **Build per-crate, NEVER workspace-wide.** `tensor-tools` has a standing `Device::Cpu` break and is a default-member, so bare `cargo check`/`cargo test` at the root fails. Always `-p fuel-graph` / `-p fuel-kernel-seam-types`.
- **ONE cargo invocation at a time.** The build-dir lock serializes; parallel invocations thrash.
- **Run cargo in the FOREGROUND.** A subagent deadlocks waiting on its own backgrounded cargo job. Do not background the test commands in this plan.
- **TDD, born-red.** Write the failing test first, run it and SEE it fail for the stated reason, then implement to green. A "born-red" test is the goal, not an accident.
- **Never panic on production paths.** `primitive_shape` returns `Result`; a malformed op/shape is an `Err`, never a panic. No new `.unwrap()`/`.expect()` on non-test paths. (The one existing `emit` `.expect("region validated re-emittable at registration")` stays — its consumers wrap `emit` in `catch_unwind`; see spec "Error handling".)
- **Validate at graph-build time.** Every check that can run at build time must. The `Tensor` builders keep their existing argument validation + `Result`/panic behavior unchanged after routing through `primitive_shape` (Task 6 is behavior-preserving).
- **Docs are part of the change.** Task 2 + Task 7 record the `OpAttrs` additions + canonical encoding in `docs/architecture/`-adjacent `kernel-seam-interop.md` (the wire-type home) and flag Baracuda to mirror (like F1).

## Boundaries (NOT in Increment A)

- The decompose **migration** (Increment C — moving the ~16 migratable `registry::*::decompose` fns to `PatternNode` data). A only makes `emit` *capable*.
- Unify internal+external registry + wire the 18 stubbed matchers (Increment D); KISC framing (Increment E).
- The 4 basis-gap ops (`conv2d`/`conv_transpose_2d`/`qmatmul`/`inplace_affine`) and **`Op::Scan`** (higher-order, body-carrying — its shape depends on the body sub-DAG, it gets its own shape rule later). `Op::NonZeroIndices` (multi-output, data-dependent) is likewise out — treat as an honest `Err` from `primitive_shape`.

---

## Task 1 — `primitive_shape`: the single source of truth

Extract one function that answers "what shape+dtype does this primitive `Op` produce from these input shapes+dtypes." It reads params off the `Op` variant (not `OpAttrs`). This is the oracle every later task leans on.

**Files:**
- Create `fuel-graph/src/shape.rs`.
- Modify `fuel-graph/src/lib.rs` — add `mod shape;` and `pub use shape::primitive_shape;` near the other module declarations / re-exports (grep `pub use` / `mod opt;` region at the top of `lib.rs`).

**Interfaces:**
- Produces: `pub fn primitive_shape(op: &crate::Op, input_shapes: &[fuel_ir::Shape], input_dtypes: &[fuel_ir::DType]) -> std::result::Result<(fuel_ir::Shape, fuel_ir::DType), fuel_ir::Error>`.
- Consumes: `crate::Op` variants + `fuel_ir::{Shape, DType, Error}`. `Shape::from_dims(&[usize])`, `shape.dims() -> &[usize]`, `shape.rank()`, `shape.elem_count()`, `Error::Msg(String).bt()`.

**Shape/dtype rules (full first-order coverage):**
- Elementwise unary (`shape=in[0]`, `dtype=in[0]`): `Neg,Abs,Sqr,Sqrt,Rsqrt,Recip,Exp,Log,Sin,Cos,Tanh,Sigmoid,Silu,Gelu,GeluErf,Relu,Erf,Step,Floor,Ceil,Round,Sign,Contiguize,LogSoftmaxLastDim,LogSoftmaxLastDimBackward,AddScalar(_),MulScalar(_),PowI(_),Clamp{..},CumSum{..},Flip{..},Roll{..},Triu{..},Tril{..}`.
- Elementwise binary (`shape=in[0]`, `dtype=in[0]`): `Add,Sub,Mul,Div,Maximum,Minimum,Pow,Rem`.
- Comparison (`shape=in[0]`, `dtype=DType::U8`): `Equal,Ne,Lt,Le,Gt,Ge`.
- `Where` (inputs `(cond,a,b)`): `shape=in[0]` (cond), `dtype=in[1]` (a). `MaskedFill{..}`: `shape=in[0]`, `dtype=in[0]`.
- `Cast(dt)`: `shape=in[0]`, `dtype=dt`.
- Shape/layout: `Transpose` (swap last two, rank≥2), `Permute(axes)` (`out[i]=in[0].dims[axes[i]]`), `Reshape(s)`/`BroadcastTo(s)`/`ReduceSumTo(s)`/`ReduceMaxTo(s)` (→ `s`), `Unsqueeze{dim}` (insert 1 at dim), `Squeeze{dim}` (remove dim), `Slice{dim,start,len}` (dim→len), `Concat{dim}` (`in[0].dims[dim]+in[1].dims[dim]`), `Pad{padding,..}` (`out[i]=in[0][i]+before+after`). dtype=in[0] for all.
- Rank-reducing reductions (remove `dim`, dtype=in[0]): `SumDim(d),MaxDim(d),MinDim(d),MeanDim(d)`. `ArgMaxDim(d)`/`ArgMinDim(d)`: remove dim, `dtype=DType::U32`.
- Scalar reductions (→ rank-0 `Shape::from_dims(&[])`, dtype=in[0]): `SumAll,MaxAll,MinAll,MeanAll`.
- `MatMul` (same-rank operands — the builder inserts BroadcastTo before this; batch prefix carries from `in[0]`, `m=in[0][-2]`, `k=in[0][-1]`, `k2=in[1][-2]`, `n=in[1][-1]`, assert `k==k2`, out = batch_prefix ++ `[m,n]`, dtype=in[0]).
- Indexing (data=in[0], index=in[1], dtype=in[0]): `IndexSelect{dim}` (dim → `in[1].elem_count()`), `Gather{dim}` (out = `in[1].shape` = index shape), `IndexAdd{dim}`/`ScatterAdd{dim}` (out = in[0] shape).
- `Iota{len}` (leaf, no inputs): `shape=Shape::from_dims(&[len])`, `dtype=DType::F32`.
- **`Err`** (honest miss, never panic) for: leaves/bookkeeping with no pure inference (`Const`, `Copy`, `Release`, `Move`, `WriteSlice`, `WriteSliceRotating`, in-place `*Inplace` variants), `NonZeroIndices`, `Op::Fused(..)`, `Op::Scan`/`Op::ScanPlaceholder`, and the basis-gap-only paths. Message form: `Error::Msg(format!("primitive_shape: {op:?} is not a first-order shape-inferable primitive")).bt()`.
- Malformed inputs (e.g. an elementwise op with `input_shapes.is_empty()`, a `dim` out of range, a `MatMul` inner-dim mismatch) → `Err`, never a panic or index-out-of-bounds.

**Steps:**
- [ ] Write the failing test module in `shape.rs` (COMPLETE):
```rust
#[cfg(test)]
mod tests {
    use super::primitive_shape;
    use crate::{Op, PadMode};
    use fuel_ir::{DType, Shape};

    fn s(d: &[usize]) -> Shape { Shape::from_dims(d) }

    #[test]
    fn elementwise_preserves_shape_and_dtype() {
        let (sh, dt) = primitive_shape(&Op::Add, &[s(&[2, 3]), s(&[2, 3])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[2, 3]), DType::F32));
        let (sh, dt) = primitive_shape(&Op::Neg, &[s(&[4])], &[DType::BF16]).unwrap();
        assert_eq!((sh, dt), (s(&[4]), DType::BF16));
    }

    #[test]
    fn comparison_forces_u8() {
        let (sh, dt) = primitive_shape(&Op::Lt, &[s(&[5]), s(&[5])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[5]), DType::U8));
    }

    #[test]
    fn cast_takes_target_dtype() {
        let (sh, dt) = primitive_shape(&Op::Cast(DType::F16), &[s(&[2, 2])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[2, 2]), DType::F16));
    }

    #[test]
    fn slice_shrinks_named_dim() {
        let (sh, dt) = primitive_shape(&Op::Slice { dim: 1, start: 2, len: 3 }, &[s(&[4, 8])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[4, 3]), DType::F32));
    }

    #[test]
    fn concat_sums_the_join_dim() {
        let (sh, _) = primitive_shape(&Op::Concat { dim: 0 }, &[s(&[2, 4]), s(&[5, 4])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!(sh, s(&[7, 4]));
    }

    #[test]
    fn permute_reorders_axes() {
        let (sh, _) = primitive_shape(&Op::Permute(vec![2, 0, 1]), &[s(&[2, 3, 4])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[4, 2, 3]));
    }

    #[test]
    fn reshape_and_broadcast_take_target() {
        let (sh, _) = primitive_shape(&Op::Reshape(s(&[6])), &[s(&[2, 3])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[6]));
        let (sh, _) = primitive_shape(&Op::BroadcastTo(s(&[2, 3])), &[s(&[1, 3])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[2, 3]));
    }

    #[test]
    fn mean_dim_removes_the_dim() {
        let (sh, dt) = primitive_shape(&Op::MeanDim(1), &[s(&[2, 3, 4])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[2, 4]), DType::F32));
    }

    #[test]
    fn reduce_to_and_scalar_reductions() {
        let (sh, _) = primitive_shape(&Op::ReduceMaxTo(s(&[2, 1])), &[s(&[2, 5])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[2, 1]));
        let (sh, _) = primitive_shape(&Op::SumAll, &[s(&[3, 3])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[]));
    }

    #[test]
    fn argmax_produces_u32() {
        let (sh, dt) = primitive_shape(&Op::ArgMaxDim(0), &[s(&[4, 6])], &[DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[6]), DType::U32));
    }

    #[test]
    fn matmul_contracts_inner_dim() {
        let (sh, _) = primitive_shape(&Op::MatMul, &[s(&[2, 3]), s(&[3, 5])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!(sh, s(&[2, 5]));
        // batched, same-rank
        let (sh, _) = primitive_shape(&Op::MatMul, &[s(&[7, 2, 3]), s(&[7, 3, 5])], &[DType::F32, DType::F32]).unwrap();
        assert_eq!(sh, s(&[7, 2, 5]));
    }

    #[test]
    fn where_is_cond_shape_a_dtype() {
        let (sh, dt) = primitive_shape(&Op::Where, &[s(&[4]), s(&[4]), s(&[4])], &[DType::U8, DType::F32, DType::F32]).unwrap();
        assert_eq!((sh, dt), (s(&[4]), DType::F32));
    }

    #[test]
    fn index_ops() {
        // IndexSelect: dim replaced by index length (index is in[1], 1-D)
        let (sh, _) = primitive_shape(&Op::IndexSelect { dim: 0 }, &[s(&[8, 4]), s(&[3])], &[DType::F32, DType::U32]).unwrap();
        assert_eq!(sh, s(&[3, 4]));
        // Gather: out == index shape
        let (sh, _) = primitive_shape(&Op::Gather { dim: 1 }, &[s(&[2, 8]), s(&[2, 3])], &[DType::F32, DType::U32]).unwrap();
        assert_eq!(sh, s(&[2, 3]));
    }

    #[test]
    fn pad_extends_each_axis() {
        let (sh, _) = primitive_shape(
            &Op::Pad { padding: vec![(1, 1), (0, 2)], mode: PadMode::Constant, value: 0.0 },
            &[s(&[3, 4])], &[DType::F32]).unwrap();
        assert_eq!(sh, s(&[5, 6]));
    }

    #[test]
    fn iota_is_a_len_vector() {
        let (sh, dt) = primitive_shape(&Op::Iota { len: 9 }, &[], &[]).unwrap();
        assert_eq!((sh, dt), (s(&[9]), DType::F32));
    }

    #[test]
    fn out_of_scope_ops_error_not_panic() {
        assert!(primitive_shape(&Op::Const, &[], &[]).is_err());
        assert!(primitive_shape(&Op::Release, &[s(&[1])], &[DType::F32]).is_err());
    }

    #[test]
    fn malformed_input_is_err_not_panic() {
        // Elementwise with no inputs.
        assert!(primitive_shape(&Op::Add, &[], &[]).is_err());
        // Slice dim out of range.
        assert!(primitive_shape(&Op::Slice { dim: 5, start: 0, len: 1 }, &[s(&[2, 2])], &[DType::F32]).is_err());
        // MatMul inner-dim mismatch.
        assert!(primitive_shape(&Op::MatMul, &[s(&[2, 3]), s(&[4, 5])], &[DType::F32, DType::F32]).is_err());
    }
}
```
- [ ] Run RED: `cargo test -p fuel-graph --lib shape::tests` — fails to compile (`primitive_shape` / `shape` module do not exist yet). That is the expected red.
- [ ] Implement `primitive_shape` in `shape.rs`. Head:
```rust
//! `primitive_shape` — the single source of truth for primitive-Op shape+dtype
//! inference (Convergence Increment A). Called by BOTH the `Tensor` builders
//! (lib.rs) and the runtime `emit` re-emitter (runtime_fused.rs) so there is
//! exactly one place that answers "what does this primitive Op produce". Reads
//! params off the `Op` variant; never panics — a malformed op/shape is an `Err`.
use crate::Op;
use fuel_ir::{DType, Error, Shape};

fn need<'a>(shapes: &'a [Shape], n: usize, op: &Op) -> Result<&'a [Shape], Error> {
    if shapes.len() < n {
        return Err(Error::Msg(format!("primitive_shape: {op:?} needs {n} input shape(s), got {}", shapes.len())).bt());
    }
    Ok(shapes)
}

pub fn primitive_shape(
    op: &Op,
    input_shapes: &[Shape],
    input_dtypes: &[DType],
) -> Result<(Shape, DType), Error> {
    use Op::*;
    // Helper closures reused below.
    let elem0 = || -> Result<(Shape, DType), Error> {
        need(input_shapes, 1, op)?;
        Ok((input_shapes[0].clone(), input_dtypes[0]))
    };
    Ok(match op {
        // --- elementwise unary + binary + scalar-param (shape=in[0], dtype=in[0]) ---
        Add | Sub | Mul | Div | Maximum | Minimum | Pow | Rem
        | Neg | Abs | Sqr | Sqrt | Rsqrt | Recip | Exp | Log | Sin | Cos
        | Tanh | Sigmoid | Silu | Gelu | GeluErf | Relu | Erf | Step
        | Floor | Ceil | Round | Sign | Contiguize
        | AddScalar(_) | MulScalar(_) | PowI(_) | Clamp { .. }
        | CumSum { .. } | Flip { .. } | Roll { .. } | Triu { .. } | Tril { .. }
        | LogSoftmaxLastDim | LogSoftmaxLastDimBackward | MaskedFill { .. } => elem0()?,

        // --- comparison → U8 ---
        Equal | Ne | Lt | Le | Gt | Ge => {
            need(input_shapes, 1, op)?;
            (input_shapes[0].clone(), DType::U8)
        }

        // --- Where: shape = cond (in[0]), dtype = a (in[1]) ---
        Where => {
            need(input_shapes, 2, op)?;
            (input_shapes[0].clone(), input_dtypes[1])
        }

        // --- dtype-changing ---
        Cast(dt) => { need(input_shapes, 1, op)?; (input_shapes[0].clone(), *dt) }

        // --- shape/layout carrying an explicit target shape on the variant ---
        Reshape(sh) | BroadcastTo(sh) | ReduceSumTo(sh) | ReduceMaxTo(sh) => {
            need(input_shapes, 1, op)?;
            (sh.clone(), input_dtypes[0])
        }

        Transpose => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if d.len() < 2 {
                return Err(Error::Msg(format!("primitive_shape: Transpose needs rank>=2, got {d:?}")).bt());
            }
            let mut out = d.to_vec();
            let r = out.len();
            out.swap(r - 2, r - 1);
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Permute(axes) => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if axes.len() != d.len() || axes.iter().any(|&a| a >= d.len()) {
                return Err(Error::Msg(format!("primitive_shape: Permute axes {axes:?} invalid for {d:?}")).bt());
            }
            let out: Vec<usize> = axes.iter().map(|&a| d[a]).collect();
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Unsqueeze { dim } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if *dim > d.len() {
                return Err(Error::Msg(format!("primitive_shape: Unsqueeze dim {dim} > rank {}", d.len())).bt());
            }
            let mut out = d.to_vec();
            out.insert(*dim, 1);
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Squeeze { dim } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if *dim >= d.len() {
                return Err(Error::Msg(format!("primitive_shape: Squeeze dim {dim} >= rank {}", d.len())).bt());
            }
            let out: Vec<usize> = d.iter().enumerate().filter(|(i, _)| *i != *dim).map(|(_, &v)| v).collect();
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Slice { dim, start, len } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if *dim >= d.len() || start + len > d[*dim] {
                return Err(Error::Msg(format!("primitive_shape: Slice{{dim:{dim},start:{start},len:{len}}} invalid for {d:?}")).bt());
            }
            let mut out = d.to_vec();
            out[*dim] = *len;
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Concat { dim } => {
            need(input_shapes, 2, op)?;
            let a = input_shapes[0].dims();
            let b = input_shapes[1].dims();
            if *dim >= a.len() || a.len() != b.len() {
                return Err(Error::Msg(format!("primitive_shape: Concat dim {dim} invalid for {a:?} / {b:?}")).bt());
            }
            let mut out = a.to_vec();
            out[*dim] = a[*dim] + b[*dim];
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Pad { padding, .. } => {
            need(input_shapes, 1, op)?;
            let d = input_shapes[0].dims();
            if padding.len() != d.len() {
                return Err(Error::Msg(format!("primitive_shape: Pad padding {padding:?} rank != {d:?}")).bt());
            }
            let out: Vec<usize> = d.iter().zip(padding).map(|(&x, (b, a))| x + b + a).collect();
            (Shape::from_dims(&out), input_dtypes[0])
        }

        // --- rank-reducing reductions ---
        SumDim(dim) | MaxDim(dim) | MinDim(dim) | MeanDim(dim) => {
            need(input_shapes, 1, op)?;
            (reduce_remove(input_shapes[0].dims(), *dim, op)?, input_dtypes[0])
        }
        ArgMaxDim(dim) | ArgMinDim(dim) => {
            need(input_shapes, 1, op)?;
            (reduce_remove(input_shapes[0].dims(), *dim, op)?, DType::U32)
        }
        SumAll | MaxAll | MinAll | MeanAll => {
            need(input_shapes, 1, op)?;
            (Shape::from_dims(&[]), input_dtypes[0])
        }

        // --- matmul (same-rank operands; builder pre-broadcasts) ---
        MatMul => {
            need(input_shapes, 2, op)?;
            let l = input_shapes[0].dims();
            let r = input_shapes[1].dims();
            if l.len() < 2 || r.len() != l.len() {
                return Err(Error::Msg(format!("primitive_shape: MatMul needs same-rank>=2 operands, got {l:?} / {r:?}")).bt());
            }
            let rank = l.len();
            let (m, k) = (l[rank - 2], l[rank - 1]);
            let (k2, n) = (r[rank - 2], r[rank - 1]);
            if k != k2 {
                return Err(Error::Msg(format!("primitive_shape: MatMul inner-dim mismatch {k} vs {k2}")).bt());
            }
            let mut out = l[..rank - 2].to_vec();
            out.push(m);
            out.push(n);
            (Shape::from_dims(&out), input_dtypes[0])
        }

        // --- indexing (data=in[0], index=in[1]) ---
        IndexSelect { dim } => {
            need(input_shapes, 2, op)?;
            let d = input_shapes[0].dims();
            if *dim >= d.len() {
                return Err(Error::Msg(format!("primitive_shape: IndexSelect dim {dim} invalid for {d:?}")).bt());
            }
            let mut out = d.to_vec();
            out[*dim] = input_shapes[1].elem_count();
            (Shape::from_dims(&out), input_dtypes[0])
        }
        Gather { .. } => {
            need(input_shapes, 2, op)?;
            (input_shapes[1].clone(), input_dtypes[0])
        }
        IndexAdd { .. } | ScatterAdd { .. } => {
            need(input_shapes, 1, op)?;
            (input_shapes[0].clone(), input_dtypes[0])
        }

        // --- value source leaf ---
        Iota { len } => (Shape::from_dims(&[*len]), DType::F32),

        // --- everything else: honest miss, never a crash ---
        _ => {
            return Err(Error::Msg(format!(
                "primitive_shape: {op:?} is not a first-order shape-inferable primitive"
            )).bt());
        }
    })
}

fn reduce_remove(dims: &[usize], dim: usize, op: &Op) -> Result<Shape, Error> {
    if dim >= dims.len() {
        return Err(Error::Msg(format!("primitive_shape: {op:?} dim {dim} out of range for {dims:?}")).bt());
    }
    let out: Vec<usize> = dims.iter().enumerate().filter(|(i, _)| *i != dim).map(|(_, &v)| v).collect();
    Ok(Shape::from_dims(&out))
}
```
  Wire it in `lib.rs`: add `mod shape;` beside the other `mod` lines and `pub use shape::primitive_shape;` beside the other `pub use`. Confirm `PadMode` is importable in the test (`use crate::PadMode;` — it is `pub`, lib.rs:198).
- [ ] Run GREEN: `cargo test -p fuel-graph --lib shape::tests`. All pass.
- [ ] Commit: `feat(graph): primitive_shape — single source of truth for first-order op shape+dtype (Convergence A, Task 1)`.

---

## Task 2 — Extend `OpAttrs` for the full first-order set (+ `op_to_attrs`)

Add the dependency-free carriers `tag_to_op` (Task 3) needs but `OpAttrs` can't express today, and extend the forward projection `op_to_attrs` to populate them, so a `Op → OpAttrs → Op` round-trip is faithful.

**Files:**
- Modify `fuel-kernel-seam-types/src/lib.rs:71` (`struct OpAttrs`) — additive optional fields.
- Modify `fuel-graph/src/jit.rs:127-150` (`op_to_attrs`) — populate the new fields.
- Modify `docs/architecture/`-adjacent `kernel-seam-interop.md` (find with `grep -rl "OpAttrs" docs/`) — record the additive fields + flag Baracuda to mirror.

**Interfaces:**
- Produces (added to `OpAttrs`, all `#[derive(Default)]`-compatible):
  - `pub cast_dtype: Option<String>` — `Cast` target dtype as `DType::as_str()` name (dep-free; fuel-graph maps back via `DType::from_str`). Also carries `MaskedFill`'s value dtype.
  - `pub slice_start: Option<u64>`, `pub slice_len: Option<u64>` — `Slice` (its `dim` rides `axis`).
  - `pub roll_shift: Option<i64>` — `Roll` (its `dim` rides `axis`).
  - `pub pad_amounts: Vec<(u64, u64)>` — `Pad` per-axis `(before, after)`.
  - `pub pad_mode: Option<u8>` — `Pad` mode code: `0=Constant, 1=Reflect, 2=Replicate` (mirrors `PadMode` order, lib.rs:198; dep-free).
  - `pub pad_value: Option<f64>` — `Pad` constant fill value.
  - `pub keepdim: Option<bool>` — §6.19 reduce-schema conformance (serialized in Task 7; not consumed by `tag_to_op` — Fuel's reduce Ops encode keepdim structurally via `ReduceSumTo`/`ReduceMaxTo`'s target shape or via rank-reducing `SumDim`).
- Consumes: `MaskedFill` value rides the existing `scalars[0]`; `Iota`'s `len` rides the existing `target_shape` as a single-element `[len]` (its output shape). No new field for those two.

**Steps:**
- [ ] Write the failing round-trip test in `fuel-graph/src/jit.rs` `#[cfg(test)] mod tests` (COMPLETE):
```rust
    #[test]
    fn op_to_attrs_projects_new_first_order_params() {
        use fuel_ir::DType;
        // Cast → cast_dtype name.
        let a = op_to_attrs(&Op::Cast(DType::F16));
        assert_eq!(a.cast_dtype.as_deref(), Some("f16"));
        // Slice → axis(dim) + start + len.
        let a = op_to_attrs(&Op::Slice { dim: 2, start: 3, len: 5 });
        assert_eq!((a.axis, a.slice_start, a.slice_len), (Some(2), Some(3), Some(5)));
        // Concat → axis(dim).
        assert_eq!(op_to_attrs(&Op::Concat { dim: 1 }).axis, Some(1));
        // Roll → axis(dim) + roll_shift.
        let a = op_to_attrs(&Op::Roll { dim: 0, shift: -2 });
        assert_eq!((a.axis, a.roll_shift), (Some(0), Some(-2)));
        // Flip → axis(dim).
        assert_eq!(op_to_attrs(&Op::Flip { dim: 1 }).axis, Some(1));
        // Pad → amounts + mode + value.
        let a = op_to_attrs(&Op::Pad { padding: vec![(1, 1), (0, 2)], mode: crate::PadMode::Constant, value: 0.5 });
        assert_eq!(a.pad_amounts, vec![(1, 1), (0, 2)]);
        assert_eq!((a.pad_mode, a.pad_value), (Some(0), Some(0.5)));
        // Iota len rides target_shape.
        assert_eq!(op_to_attrs(&Op::Iota { len: 7 }).target_shape, vec![7]);
    }
```
- [ ] Run RED: `cargo test -p fuel-graph --lib jit::tests::op_to_attrs_projects_new_first_order_params` — fails to compile (`cast_dtype`/`slice_start`/… fields do not exist).
- [ ] Implement:
  - In `fuel-kernel-seam-types/src/lib.rs`, add the fields to `OpAttrs` with doc comments; the `#[derive(Default)]` already covers them (all `Option`/`Vec`). Keep `#[derive(Clone, Debug, Default, PartialEq)]`.
  - In `fuel-graph/src/jit.rs` `op_to_attrs`, extend the `match op`:
```rust
        Op::Slice { dim, start, len } => {
            a.axis = Some(*dim as i64);
            a.slice_start = Some(*start as u64);
            a.slice_len = Some(*len as u64);
        }
        Op::Concat { dim } | Op::Flip { dim } => a.axis = Some(*dim as i64),
        Op::Roll { dim, shift } => { a.axis = Some(*dim as i64); a.roll_shift = Some(*shift); }
        Op::Cast(dt) => a.cast_dtype = Some(dt.as_str().to_string()),
        Op::Iota { len } => a.target_shape = vec![*len as i64],
        Op::Pad { padding, mode, value } => {
            a.pad_amounts = padding.iter().map(|&(b, e)| (b as u64, e as u64)).collect();
            a.pad_mode = Some(match mode {
                crate::PadMode::Constant => 0,
                crate::PadMode::Reflect => 1,
                crate::PadMode::Replicate => 2,
            });
            a.pad_value = Some(*value);
        }
        Op::MaskedFill { value } => {
            a.scalars = vec![value.to_f64()];            // value rides scalars[0]
            a.cast_dtype = Some(value.dtype().as_str().to_string());
        }
```
    Confirm the `Scalar` accessor names against `lib.rs` (`Scalar::from_f64` is at lib.rs:3549; use whatever `to_f64()`/`dtype()` the type exposes — grep `impl Scalar`). If `Scalar` lacks a public `to_f64`/`dtype`, defer the `MaskedFill` arm and note it (see Ambiguities); it is not exercised by the Task 5 oracle.
- [ ] Run GREEN: `cargo test -p fuel-graph --lib jit::tests` (whole jit suite — the existing `match_node_discriminates_on_perm_attr` etc. must still pass; the additive fields are wildcard-empty so `attrs_match` is unaffected). Also `cargo test -p fuel-kernel-seam-types` (its own suite must still pass).
- [ ] Update `kernel-seam-interop.md`: add the additive field list under the F1 record, note "Fuel-led additive, Baracuda to mirror; conforms to KISS §6.19 (see Task 7)".
- [ ] Commit: `feat(seam-types): OpAttrs carries Slice/Cast/Roll/Pad/keepdim params for full first-order emit (Convergence A, Task 2)`.

---

## Task 3 — Grow `tag_to_op` + `validate_representable` to the full first-order set

Make `tag_to_op(OpTag, &OpAttrs) -> Option<Op>` reconstruct every non-basis-gap, non-Scan `Op` from its tag + attrs. `validate_representable`'s accept set is exactly `tag_to_op`'s coverage (unchanged mechanism), so this widens what regions register.

**Files:** Modify `fuel-graph/src/runtime_fused.rs` (`tag_to_op` :256, `scalar_slot_arity` :301, `validate_representable` :320).

**Interfaces:**
- Consumes: `OpTag` (fuel-kernel-seam-types) + the Task-2 `OpAttrs` fields; `fuel_ir::{DType, Shape}` (needs a `use fuel_ir::{DType, Shape};` at the fn — currently only in tests). `std::str::FromStr` for `DType::from_str`.
- Produces: a `tag_to_op` total over the in-scope set; `Op::Fused`-emitting fused ops and the 4 basis-gap tags + Scan tags still return `None` (documented honest miss).

**Reconstruction table (add to the `match tag`):**
- Shape/layout: `Transpose→Op::Transpose`; `Permute→Op::Permute(attrs.perm.iter().map(|&x| x as usize).collect())`; `Reshape→Op::Reshape(Shape::from_dims(&target_usize))`; `BroadcastTo→Op::BroadcastTo(...)`; `ReduceSumTo→Op::ReduceSumTo(...)`; `ReduceMaxTo→Op::ReduceMaxTo(...)` (all from `attrs.target_shape` via `i64→usize`); `Unsqueeze→Op::Unsqueeze{dim: attrs.dims.first()? as usize}`; `Squeeze→Op::Squeeze{...}`; `Slice→Op::Slice{dim: attrs.axis? as usize, start: attrs.slice_start? as usize, len: attrs.slice_len? as usize}`; `Concat→Op::Concat{dim: attrs.axis? as usize}`; `Flip→Op::Flip{dim: attrs.axis? as usize}`; `Roll→Op::Roll{dim: attrs.axis? as usize, shift: attrs.roll_shift?}`; `Pad→Op::Pad{padding: attrs.pad_amounts.iter().map(|&(b,e)|(b as usize,e as usize)).collect(), mode: <u8→PadMode>, value: attrs.pad_value.unwrap_or(0.0)}`; `Triu→Op::Triu{diagonal: attrs.axis?}`; `Tril→Op::Tril{diagonal: attrs.axis?}`.
- Dtype: `Cast→Op::Cast(DType::from_str(attrs.cast_dtype.as_deref()?).ok()?)`; comparisons `Equal/Ne/Lt/Le/Gt/Ge→Op::{Equal,…}`.
- Reductions: `SumDim→Op::SumDim(attrs.axis? as usize)`, `MeanDim→Op::MeanDim(...)`, `SumAll/MaxAll/MinAll/MeanAll→Op::{…}`, `CumSum→Op::CumSum{dim: attrs.axis? as usize}`.
- `MatMul→Op::MatMul`; `Where→Op::Where`; `LogSoftmaxLastDim→Op::LogSoftmaxLastDim`; `Iota→Op::Iota{len: *attrs.target_shape.first()? as usize}`.
- Indexing: `IndexSelect→Op::IndexSelect{dim: attrs.axis? as usize}`, `Gather→Op::Gather{...}`, `IndexAdd→Op::IndexAdd{...}`, `ScatterAdd→Op::ScatterAdd{...}`. (`attrs.axis` carries dim; extend `op_to_attrs` in Task 2 only if a matcher needs it — for `tag_to_op`/emit the region author supplies `axis`.)
- Scalar-param (unchanged): `AddScalar/MulScalar` via `attrs.scalars.first()?`. `PowI`/`Clamp` remain `None` (they need i32/two-scalar carriers not in scope — keep as honest miss unless a region needs them; the oracle does not). `MaskedFill` → `Op::MaskedFill{value: Scalar::from_f64(attrs.scalars.first().copied()?, DType::from_str(attrs.cast_dtype.as_deref()?).ok()?)}` **only if** `Scalar::from_f64(f64, DType)` matches (lib.rs:3549); else keep `None` + note.
- Still `None` (documented): all `*Inplace`, `Const`, `Copy/Release/Move/WriteSlice*`, `Op::Fused`, `Scan`/`ScanPlaceholder`, `NonZeroIndices`, the 4 basis-gap tags, `PadBackward`.

Add helper for the target-shape decode (used 4×):
```rust
fn shape_from_attr(attrs: &OpAttrs) -> Option<fuel_ir::Shape> {
    if attrs.target_shape.is_empty() { return None; }
    let dims: Vec<usize> = attrs.target_shape.iter().map(|&d| d as usize).collect();
    Some(fuel_ir::Shape::from_dims(&dims))
}
```

**Steps:**
- [ ] Write the failing tests in `runtime_fused.rs` `#[cfg(test)] mod tests` (COMPLETE):
```rust
    #[test]
    fn tag_to_op_reconstructs_shape_changing_ops() {
        use fuel_ir::Shape;
        // Slice{dim:1,start:2,len:3}
        let attrs = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
        assert!(matches!(super::tag_to_op(OpTag::Slice, &attrs), Some(Op::Slice { dim: 1, start: 2, len: 3 })));
        // Concat{dim:0}
        let attrs = OpAttrs { axis: Some(0), ..OpAttrs::default() };
        assert!(matches!(super::tag_to_op(OpTag::Concat, &attrs), Some(Op::Concat { dim: 0 })));
        // Reshape([6])
        let attrs = OpAttrs { target_shape: vec![6], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::Reshape, &attrs), Some(Op::Reshape(Shape::from_dims(&[6]))));
        // BroadcastTo([2,3])
        let attrs = OpAttrs { target_shape: vec![2, 3], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::BroadcastTo, &attrs), Some(Op::BroadcastTo(Shape::from_dims(&[2, 3]))));
        // ReduceMaxTo([2,1])
        let attrs = OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::ReduceMaxTo, &attrs), Some(Op::ReduceMaxTo(Shape::from_dims(&[2, 1]))));
    }

    #[test]
    fn tag_to_op_reconstructs_reductions_dtype_and_matmul() {
        use fuel_ir::DType;
        assert!(matches!(super::tag_to_op(OpTag::MeanDim, &OpAttrs { axis: Some(1), ..OpAttrs::default() }), Some(Op::MeanDim(1))));
        assert!(matches!(super::tag_to_op(OpTag::MatMul, &OpAttrs::default()), Some(Op::MatMul)));
        // Cast target dtype via name.
        let attrs = OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() };
        assert_eq!(super::tag_to_op(OpTag::Cast, &attrs), Some(Op::Cast(DType::F16)));
        // Comparison.
        assert!(matches!(super::tag_to_op(OpTag::Lt, &OpAttrs::default()), Some(Op::Lt)));
    }

    #[test]
    fn tag_to_op_still_rejects_basis_gap_and_scan() {
        // qmatmul/conv flow through Op::Fused (no OpTag); Scan is higher-order.
        assert_eq!(super::tag_to_op(OpTag::Iota, &OpAttrs::default()), None, "Iota needs a len (target_shape) — empty attrs is a miss");
    }

    #[test]
    fn validate_representable_now_accepts_a_slice_region() {
        // Region: Concat{0}(Neg(Slice{...}(bind0)), bind0) — the rope rotate-half shape.
        let region = PatternNode::Op {
            op: OpTag::Concat,
            attrs: OpAttrs { axis: Some(0), ..OpAttrs::default() },
            operands: vec![
                PatternNode::Op {
                    op: OpTag::Neg,
                    attrs: OpAttrs::default(),
                    operands: vec![PatternNode::Op {
                        op: OpTag::Slice,
                        attrs: OpAttrs { axis: Some(0), slice_start: Some(0), slice_len: Some(1), ..OpAttrs::default() },
                        operands: vec![PatternNode::Bind { index: 0 }],
                    }],
                },
                PatternNode::Bind { index: 0 },
            ],
        };
        assert!(super::validate_representable(&region).is_ok(), "slice/concat region must now validate");
    }
```
- [ ] Run RED: `cargo test -p fuel-graph --lib runtime_fused::tests::tag_to_op_reconstructs_shape_changing_ops` — fails (arms return `None` → `unwrap`/`matches!` mismatch, or won't compile if `super::tag_to_op` visibility needs widening: `tag_to_op`/`validate_representable` are private module fns, accessible from the in-module `tests` via `super::`).
- [ ] Implement the `tag_to_op` arms per the table above; add `use std::str::FromStr;` locally for `DType::from_str`. Leave `scalar_slot_arity` unchanged unless you wire `MaskedFill` as a slot (not required). `validate_representable` needs no code change — its accept set follows `tag_to_op` automatically.
- [ ] Run GREEN: `cargo test -p fuel-graph --lib runtime_fused::tests`. All pass, including the pre-existing `register_rejects_unrepresentable_region` (MatMul was the rejection example — **update that test**: MatMul is now representable. Change its op to a genuinely still-unrepresentable one, e.g. `OpTag::Iota` with empty attrs, or `OpTag::PowI`, and assert `UnRepresentable(that_tag)`). Note the existing test at :527 must be adjusted; do it in this task.
- [ ] Commit: `feat(graph): tag_to_op + validate_representable cover the full first-order op set (Convergence A, Task 3)`.

---

## Task 4 — `emit` computes (shape, dtype) via `primitive_shape`

Replace the `operand[0]` shortcut (`runtime_fused.rs:429-431`) so re-emitted nodes get correct shape+dtype for shape-changing and dtype-changing ops.

**Files:** Modify `fuel-graph/src/runtime_fused.rs` (`emit` :402-437).

**Interfaces:**
- Consumes: `crate::shape::primitive_shape` (Task 1); the grown `tag_to_op` (Task 3).
- Produces: `emit` unchanged signature; interior node `(shape, dtype)` now from `primitive_shape(&prim, &child_shapes, &child_dtypes)`.

**Approach:** after building `child_ids`, gather `child_shapes: Vec<Shape>` and `child_dtypes: Vec<DType>` from `graph.node(id)`, call `primitive_shape`. On `Err` (a validation-bug region), fall back to the old `child_ids[0]` shape/dtype so `emit` stays panic-free and total (the region was validated re-emittable at registration, so an `Err` here means a malformed authored region — a fixpoint-ish safe default, not a crash). Replace lines 429-431:
```rust
            let child_shapes: Vec<fuel_ir::Shape> = child_ids.iter().map(|&c| graph.node(c).shape.clone()).collect();
            let child_dtypes: Vec<fuel_ir::DType> = child_ids.iter().map(|&c| graph.node(c).dtype).collect();
            let (s, d) = crate::shape::primitive_shape(&prim, &child_shapes, &child_dtypes)
                .unwrap_or_else(|_| (child_shapes[0].clone(), child_dtypes[0]));
            graph.push(Node { op: prim, inputs: child_ids, shape: s, dtype: d })
```
Add `use fuel_ir::{DType, Shape};` if not already imported in the fn scope (it is imported in tests only — add to the fn or module). Delete the now-stale "v1 same-shape elementwise" comment at :428.

**Steps:**
- [ ] Write the failing tests in `runtime_fused.rs` tests (COMPLETE):
```rust
    #[test]
    fn emit_gets_shape_right_for_a_reduction_region() {
        use fuel_ir::{DType, Shape};
        // Region: ReduceSumTo([2,1])(bind0). Input [2,5] → output [2,1].
        let region = PatternNode::Op {
            op: OpTag::ReduceSumTo,
            attrs: OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[2, 5]), dtype: DType::F32 });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(matches!(g.node(root).op, Op::ReduceSumTo(_)));
        assert_eq!(g.node(root).shape, Shape::from_dims(&[2, 1]), "emit must use the reduced shape, not operand[0]");
        assert_eq!(g.node(root).dtype, DType::F32);
    }

    #[test]
    fn emit_gets_dtype_right_for_a_cast_region() {
        use fuel_ir::{DType, Shape};
        // Region: Cast(F16)(bind0). Input F32 → output F16, same shape.
        let region = PatternNode::Op {
            op: OpTag::Cast,
            attrs: OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() },
            operands: vec![PatternNode::Bind { index: 0 }],
        };
        let mut g = Graph::new();
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(&[3, 3]), dtype: DType::F32 });
        let root = emit_region(&mut g, &region, &[x], &[]);
        assert!(matches!(g.node(root).op, Op::Cast(DType::F16)));
        assert_eq!(g.node(root).dtype, DType::F16, "emit must take Cast's target dtype, not operand[0]'s");
        assert_eq!(g.node(root).shape, Shape::from_dims(&[3, 3]));
    }
```
- [ ] Run RED: `cargo test -p fuel-graph --lib runtime_fused::tests::emit_gets_shape_right_for_a_reduction_region` — fails: current `emit` copies `operand[0]`'s shape `[2,5]`, so the assert on `[2,1]` fails (born red on real behavior). The cast test fails on dtype `F32 != F16`.
- [ ] Implement the `emit` change above.
- [ ] Run GREEN: `cargo test -p fuel-graph --lib runtime_fused::tests`. All pass, including the existing elementwise `decompose_region_re_emits_relu_add` / `slot_template_*` (elementwise shapes still resolve identically through `primitive_shape`).
- [ ] Commit: `feat(graph): emit re-emits full-parity shape+dtype via primitive_shape (Convergence A, Task 4)`.

---

## Task 5 — Byte-for-byte oracle acceptance (rope / softmax / layer_norm)

The A.4 acceptance gate + Increment-C de-risking: express each region as a `PatternNode`, re-emit via the grown `emit`, and assert the result is structurally identical (op + shape + dtype at every node) to the hand-written `registry::*::decompose` output. An extraction/reconstruction error surfaces here as a decompose mismatch, not a silent wrong shape.

**Files:** Modify `fuel-graph/src/runtime_fused.rs` tests (or a new `#[cfg(test)]` file `fuel-graph/tests/emit_decompose_parity.rs` — but keeping it in-module gives access to the private `emit`; use the in-module `tests`). Reads `crate::registry::{rope,softmax_last_dim,layer_norm_last_dim}::decompose` (all `pub`, `pub mod`, registry.rs:45/53/54).

**Interfaces:**
- Consumes: `emit_region` (pub), `crate::registry::*::decompose(graph, node_id, &FusedOpParams)`, `crate::registry::FusedOpParams`, `crate::opt::base_map_hash`.
- Produces: a reusable structural-equality helper `assert_structural_eq(&Graph, NodeId, NodeId)`.

**Helper (put in the tests module):**
```rust
    /// Recursively assert two subgraphs are identical: same Op, shape, dtype,
    /// arity, and recursively-equal inputs. Shared leaves (same NodeId) match by
    /// identity. This is the "byte-for-byte" node-structure check.
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b { return; } // shared leaf (bound external input)
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(na.shape, nb.shape, "shape mismatch at {:?} vs {:?}", na.op, nb.op);
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(na.inputs.len(), nb.inputs.len(), "arity mismatch at {:?}", na.op);
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }
```

**Steps:**
- [ ] Write the failing tests (COMPLETE). Softmax first (no params to thread), then rope, then layer_norm (params + baked `AddScalar(eps)`):
```rust
    #[test]
    fn emit_matches_softmax_last_dim_decompose() {
        use fuel_ir::{DType, Shape};
        use crate::registry::{FusedOpParams, softmax_last_dim};
        let mut g = Graph::new();
        let sh = Shape::from_dims(&[2, 4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: sh.clone(), dtype: DType::F32 });
        // Oracle: registry decompose reads inputs[0] + shape + dtype off the node.
        let fused = g.push(Node { op: Op::Const, inputs: vec![x], shape: sh.clone(), dtype: DType::F32 });
        let oracle = softmax_last_dim::decompose(&mut g, fused, &FusedOpParams::SoftmaxLastDim);

        // Region mirroring the 7-node softmax subgraph (last=1, keepdim shape [2,1]):
        let kd = OpAttrs { target_shape: vec![2, 1], ..OpAttrs::default() };
        let full = OpAttrs { target_shape: vec![2, 4], ..OpAttrs::default() };
        let region = PatternNode::Op { // Div(e, db)
            op: OpTag::Div, attrs: OpAttrs::default(),
            operands: vec![
                exp_sub_region(&kd, &full),                     // e
                bcast(&full, reduce_sum(&kd, exp_sub_region(&kd, &full))), // db = BroadcastTo(ReduceSumTo(e))
            ],
        };
        let emitted = emit_region(&mut g, &region, &[x], &[]);
        assert_structural_eq(&g, oracle, emitted);
    }
```
  Because the softmax region shares the `e = Exp(Sub(x, BroadcastTo(ReduceMaxTo(x))))` subterm twice (numerator + denominator), build it with small local helper fns so both occurrences are identical `PatternNode`s (the emitted graph will CSE-share by construction only if `emit` dedups — it does not, but `assert_structural_eq` compares structurally, so two identical subtrees are fine). Provide these region-builder helpers in the tests module:
```rust
    fn bind0() -> PatternNode { PatternNode::Bind { index: 0 } }
    fn reduce_max(kd: &OpAttrs, child: PatternNode) -> PatternNode {
        PatternNode::Op { op: OpTag::ReduceMaxTo, attrs: kd.clone(), operands: vec![child] }
    }
    fn reduce_sum(kd: &OpAttrs, child: PatternNode) -> PatternNode {
        PatternNode::Op { op: OpTag::ReduceSumTo, attrs: kd.clone(), operands: vec![child] }
    }
    fn bcast(full: &OpAttrs, child: PatternNode) -> PatternNode {
        PatternNode::Op { op: OpTag::BroadcastTo, attrs: full.clone(), operands: vec![child] }
    }
    fn exp_sub_region(kd: &OpAttrs, full: &OpAttrs) -> PatternNode {
        // Exp(Sub(x, BroadcastTo(ReduceMaxTo(x))))
        PatternNode::Op { op: OpTag::Exp, attrs: OpAttrs::default(), operands: vec![
            PatternNode::Op { op: OpTag::Sub, attrs: OpAttrs::default(), operands: vec![
                bind0(),
                bcast(full, reduce_max(kd, bind0())),
            ]},
        ]}
    }
```
  NOTE the decompose emits `Sub(x, mb)` and `Div(e, db)` — verify operand order in `softmax_last_dim::decompose` (lines 103-133: `Sub{inputs:[x_id, mb_id]}`, `Div{inputs:[e_id, db_id]}`) and mirror it exactly (do NOT rely on commutativity — `assert_structural_eq` is order-sensitive).
- [ ] Add the rope test (mirror `registry/rope.rs:83-177` — inputs `(x,cos,sin)` = binds `0,1,2`; ops Reshape/BroadcastTo/Slice/Neg/Concat/Mul/Add with the exact `broadcast_shape` `[1,seq,d]` and `half` slices). Use a shape like `[2, 4]` (seq=2, d=4, half=2). Oracle: build a node with `inputs:[x,cos,sin]`, `shape:[2,4]`, call `rope::decompose(&mut g, node, &FusedOpParams::Rope)` (verify the `FusedOpParams` variant name for rope in `registry.rs`; decompose ignores it, so any valid variant compiles). Region attrs: Reshape `target_shape: vec![1,2,4]`?? — **match the decompose's `broadcast_shape`**: for rank-2 `[2,4]`, `rope::decompose` builds `broadcast_shape_dims = [seq, d] = [2, 4]` (rank 2: `vec![1;2]` then `[rank-2]=seq`, `[rank-1]=d` → `[2,4]`). So Reshape target = `[2,4]`, BroadcastTo target = `[2,4]`, Slice `{dim:1,start:0,len:2}` and `{dim:1,start:2,len:2}`, Concat `{dim:1}`. Bind cos=1, sin=2, x=0. Mirror node order: `Mul(x, cos_bcast)`, `Mul(rotated_half, sin_bcast)`, `Add(left, right)`.
- [ ] Add the layer_norm test (mirror `registry/layer_norm_last_dim.rs:76-166`; params `FusedOpParams::LayerNormLastDim { eps: 1e-5 }`, region bakes `AddScalar(1e-5)` via `attrs.scalars = vec![1e-5]` on the `OpTag::AddScalar` node so it is a pattern-constant, not a slot; MeanDim uses `axis: Some(last)`; Reshape keepdim shape). Input `[2, 4]`, last=1, reduced shape `[2]`, keepdim `[2,1]`.
- [ ] (Optional, exercises the dtype path end-to-end) add a tiny Cast-bearing region test asserting `assert_structural_eq` against a hand-built two-node `Cast(F16)(Add(a,b))` reference graph.
- [ ] Run RED first (before Tasks 1-4 land they'd fail; here they're already in — so this task's RED is: write the test, run, and if any `primitive_shape`/`tag_to_op` arm is subtly wrong the structural-eq assert fails at the offending node with a precise `op/shape/dtype mismatch` message). If all three pass immediately, that is the acceptance — but first temporarily perturb one region attr (e.g. wrong `slice_len`) to confirm the harness actually catches a mismatch, then revert.
- [ ] Run GREEN: `cargo test -p fuel-graph --lib runtime_fused::tests`. rope + softmax + layer_norm parity all pass.
- [ ] Commit: `test(graph): byte-for-byte emit==decompose parity for rope/softmax/layer_norm (Convergence A, Task 5 — A.4 acceptance)`.

---

## Task 6 — Route the `Tensor` builders through `primitive_shape` (drift removal)

Make `primitive_shape` the single source of truth in fact, not just in principle: the ~20 builders compute their output shape+dtype by *calling* `primitive_shape` instead of inline math. Behavior-preserving; the existing builder suite is the gate.

**Files:** Modify `fuel-graph/src/lib.rs` builders: `try_permute` (:4622) + `transpose`/`try_transpose` (:4661/:4687), `cast` (:5468), `broadcast_to`/`try_broadcast_to` (:5486/:5507), `unsqueeze`/`try_unsqueeze` (:5528/:5553), `squeeze` (:5582), `reshape`/`try_reshape` (:5641/:5665), `reduce_sum_to`/`reduce_max_to` (:5688/:5712), `axis_reduction` (sum/max/min/mean_dim), `index_reduction` (:5790), `scalar_reduction` (sum/max/min/mean_all), `matmul` (:3908, after its broadcast/GQA prep — call `primitive_shape` on the final same-rank `lhs/rhs`), `concat` (:6797), `slice` (:6841), `flip` (:5044), `roll` (:5071), `masked_fill` (:5285), `pad` (:5334), `where_cond` (:5420), `index_select` (:6719), `gather` (:6761).

**Pattern (apply per builder):** keep every argument-validation check and the `Node` push; replace the inline `out_dims`/`dtype` computation with:
```rust
let (out_shape, out_dtype) = crate::shape::primitive_shape(&the_op, &input_shapes, &input_dtypes)
    .expect("builder args already validated"); // safe: the builder validated first
```
where `the_op` is the exact `Op` value being pushed, `input_shapes`/`input_dtypes` are this builder's operand shapes/dtypes. The `.expect` is justified because the builder's own validation runs first (so `primitive_shape` cannot error) — **but** to honor never-panic on the `try_*` Result-returning siblings, in those map the `Err` to the builder's typed `Error` rather than `.expect`. For the panicking siblings (`transpose`, `reshape`, `broadcast_to`, `matmul`, `concat`, `slice`, …) the `.expect` preserves the existing panic-on-bad-args contract.

**Steps:**
- [ ] Write the no-drift equivalence test in `lib.rs` tests (COMPLETE — a genuine regression guard that fails if a builder and `primitive_shape` disagree):
```rust
    #[test]
    fn builders_agree_with_primitive_shape() {
        use fuel_ir::{DType, Shape};
        let g = Graph::new_shared();
        let x = Tensor::from_shape_dtype(&g, Shape::from_dims(&[2, 3, 4]), DType::F32); // use the crate's const/leaf ctor
        // Each builder's output shape/dtype must equal primitive_shape's answer.
        let checks: Vec<(Tensor, Op)> = vec![
            (x.slice(1, 0, 2), Op::Slice { dim: 1, start: 0, len: 2 }),
            (x.mean_dim(2), Op::MeanDim(2)),
            (x.try_permute(&[2, 0, 1]).unwrap(), Op::Permute(vec![2, 0, 1])),
            (x.cast(DType::F16), Op::Cast(DType::F16)),
        ];
        for (t, op) in checks {
            let (ps_shape, ps_dtype) = crate::shape::primitive_shape(&op, &[x.shape().clone()], &[x.dtype()]).unwrap();
            assert_eq!(t.shape(), &ps_shape, "shape drift for {op:?}");
            assert_eq!(t.dtype(), ps_dtype, "dtype drift for {op:?}");
        }
    }
```
  Adjust the leaf-tensor constructor to the crate's actual test idiom (grep existing `lib.rs` tests for how they build a `Tensor` from a shape — e.g. a `const_tensor`/`zeros`/`from_shape` helper; `Graph::new_shared` may be named differently — match what the file uses).
- [ ] Run RED: `cargo test -p fuel-graph --lib builders_agree_with_primitive_shape` — before routing, this passes trivially (both compute the same thing). To make it a real red, first route ONE builder incorrectly is not desired; instead treat this task's safety net as **the full existing builder suite** plus this equivalence guard. The honest red step: run the full suite BEFORE editing to capture the green baseline, then after each builder edit re-run — any regression is the red-to-fix signal.
- [ ] Implement the routing builder-by-builder. After each, run `cargo test -p fuel-graph --lib` (foreground) and keep it green. Do `matmul` carefully — call `primitive_shape` only after the existing broadcast/GQA prep produces same-rank `lhs`/`rhs`.
- [ ] Run GREEN: full `cargo test -p fuel-graph --lib` green + `builders_agree_with_primitive_shape` green.
- [ ] Commit: `refactor(graph): Tensor builders derive shape+dtype from primitive_shape (no-drift single source, Convergence A, Task 6)`.

---

## Task 7 — `OpAttrs` §6.19 canonical positional-blob serialization

Give `OpAttrs` the pinned KISS §6.19 canonical, no-elision positional little-endian blob so a Fuel recipe is byte-comparable with a Baracuda-emitted one (the §2.A conformance-gap fix). An empty-schema op serializes as a **zero-length** length-prefixed blob (one canonical byte form). Fuel's internal `OpAttrs` stays a struct; this adds the canonical serialization onto it.

**Files:**
- Modify `fuel-kernel-seam-types/src/lib.rs` — add `impl OpAttrs { pub fn to_canonical_bytes(&self, op: OpTag) -> Vec<u8> }`.
- Modify `kernel-seam-interop.md` — record the per-op positional schema + the zero-length-blob rule + `SEAM_CAP_RECIPE_IMPORT = FEAT bit 35` cross-reference (per `baracuda-recipe-grammar-codesign-reply-2.md`).

**Interfaces:**
- Produces: `pub fn to_canonical_bytes(&self, op: OpTag) -> Vec<u8>` — the §6.19.3 positional blob for `op`, length-prefixed (`u32` LE length, then the positional fields LE). Per the confirmed schemas in `baracuda-recipe-grammar-codesign-reply-2.md`:
  - `reduce{monoid, reduce_axes, keepdim}` (SumDim/MeanDim/ReduceSumTo/ReduceMaxTo): axes from `axis`/`target_shape`, `keepdim` from the `keepdim` field.
  - `gather{axis, oob, index_operand, index_dtype}` / `scatter{axis, scatter_combine, oob, index_operand, index_dtype}`.
  - shape/layout ops: their positional params (`perm`, `target_shape`, `slice_start`/`slice_len`, `pad_amounts`+`pad_mode`+`pad_value`, `cast_dtype`).
  - **empty-schema op** (e.g. `Add`, `Neg`, `MatMul`, `Where`): `[0,0,0,0]` (u32 LE zero length) — the single canonical byte form.
- Consumes: `OpTag` + all `OpAttrs` fields.

**Steps:**
- [ ] Write the failing golden-bytes tests in `fuel-kernel-seam-types/src/lib.rs` tests (COMPLETE):
```rust
    #[test]
    fn empty_schema_op_serializes_zero_length() {
        // Add carries no attrs → one canonical byte form: u32 LE length 0.
        assert_eq!(OpAttrs::default().to_canonical_bytes(OpTag::Add), vec![0, 0, 0, 0]);
        assert_eq!(OpAttrs::default().to_canonical_bytes(OpTag::MatMul), vec![0, 0, 0, 0]);
    }

    #[test]
    fn slice_serializes_positionally() {
        // Slice schema (positional): axis(u32), start(u64), len(u64) — see kernel-seam-interop.md.
        let a = OpAttrs { axis: Some(1), slice_start: Some(2), slice_len: Some(3), ..OpAttrs::default() };
        let mut expect = Vec::new();
        let body = {
            let mut b = Vec::new();
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&2u64.to_le_bytes());
            b.extend_from_slice(&3u64.to_le_bytes());
            b
        };
        expect.extend_from_slice(&(body.len() as u32).to_le_bytes());
        expect.extend_from_slice(&body);
        assert_eq!(a.to_canonical_bytes(OpTag::Slice), expect);
    }

    #[test]
    fn cast_serializes_dtype_name_length_prefixed() {
        let a = OpAttrs { cast_dtype: Some("f16".into()), ..OpAttrs::default() };
        let mut body = Vec::new();
        body.extend_from_slice(&(3u32.to_le_bytes())); // name length
        body.extend_from_slice(b"f16");
        let mut expect = (body.len() as u32).to_le_bytes().to_vec();
        expect.extend_from_slice(&body);
        assert_eq!(a.to_canonical_bytes(OpTag::Cast), expect);
    }

    #[test]
    fn canonical_bytes_are_deterministic() {
        let a = OpAttrs { target_shape: vec![2, 3], ..OpAttrs::default() };
        assert_eq!(a.to_canonical_bytes(OpTag::Reshape), a.to_canonical_bytes(OpTag::Reshape));
    }
```
- [ ] Run RED: `cargo test -p fuel-kernel-seam-types` — fails to compile (`to_canonical_bytes` missing).
- [ ] Implement `to_canonical_bytes`: build the per-`OpTag` positional `body: Vec<u8>` (LE), then return `(body.len() as u32).to_le_bytes()` ++ `body`. Define one match arm per schema group; the fallthrough (empty-schema) yields `body = []` → `[0,0,0,0]`. Pin the exact field order per op in a doc comment and mirror it in `kernel-seam-interop.md`. Keep it self-contained (std-only — no `fuel_ir`).
- [ ] Run GREEN: `cargo test -p fuel-kernel-seam-types`.
- [ ] Update `kernel-seam-interop.md`: the §6.19 positional schema table (field order per op), the zero-length-blob rule, and `SEAM_CAP_RECIPE_IMPORT = FEAT bit 35`. Flag Baracuda to mirror the encoding.
- [ ] Commit: `feat(seam-types): OpAttrs canonical §6.19 positional-blob serialization (Convergence A, Task 7 — §2.A conformance fix)`.

---

## Self-review

**Spec-component → task coverage:**
- A.1 `primitive_shape` single source of truth → **Task 1** (define + full first-order coverage) + **Task 6** (builders actually call it — the no-drift half).
- A.2 part 1 `OpAttrs` field coverage → **Task 2**. A.2 part 2 §6.19 canonical serialization / §2.A fix → **Task 7**.
- A.3 `tag_to_op` + `validate_representable` growth → **Task 3**; `emit` via `primitive_shape` → **Task 4**.
- A.4 byte-for-byte rope/softmax/layer_norm (+ Cast dtype) oracle → **Task 5**.
- A.5 Baracuda flat-DAG reply → DONE (spec), no task.
- Error handling / never-panic → Task 1 (`Result`), Task 4 (`emit` `Err`→safe fallback, existing `.expect` retained), Task 6 (`try_*` map `Err`→typed error).
- Boundaries (basis-gap ops, `Op::Scan`, `NonZeroIndices`, migration/D/E) → excluded in Tasks 1/3 (honest `Err`/`None`), no migration task.

**Type-name consistency:** `primitive_shape(op: &Op, &[Shape], &[DType]) -> Result<(Shape, DType), fuel_ir::Error>` (Task 1) is called identically in Tasks 4 & 6. `OpAttrs` fields `cast_dtype/slice_start/slice_len/roll_shift/pad_amounts/pad_mode/pad_value/keepdim` are introduced in Task 2 and consumed by Tasks 3 & 7 with the same names/types. `to_canonical_bytes(&self, op: OpTag) -> Vec<u8>` (Task 7). `assert_structural_eq` (Task 5) is test-local.

**Sequencing:** Task 1 (oracle) → Task 2 (attrs) → Task 3 (tag_to_op, needs attrs) → Task 4 (emit, needs primitive_shape + tag_to_op) → Task 5 (acceptance, needs 1-4) → Task 6 (builder routing, needs 1; sequenced after the acceptance so the emit-capability keystone lands first) → Task 7 (serialization, needs 2; independent of emit).

## Ambiguities / judgment calls (flagged for the build agent)

1. **`Cast` dtype encoding.** `OpAttrs` is dependency-free (its Cargo.toml forbids `fuel_ir`), so the target dtype cannot be `fuel_ir::DType`. Chose `Option<String>` using the existing stable `DType::as_str()`/`FromStr` names (dtype.rs:61-83). A numeric code table was the alternative but there is no stable `#[repr]` on `DType` today; the string is the safe dep-free choice.
2. **`Iota` len / `MaskedFill` value reuse existing fields** (`target_shape` = `[len]`; value on `scalars[0]` + dtype on `cast_dtype`) rather than minting `iota_len`/`masked_fill_value` fields — mirrors how `target_shape` already serves both `BroadcastTo` and `Reshape` (OpTag disambiguates). If the build agent finds `Scalar` lacks public `to_f64()`/`dtype()` accessors (grep `impl Scalar` near lib.rs:3549), **defer the `MaskedFill` arm** in Tasks 2/3 (keep `tag_to_op(MaskedFill)→None`); it is not exercised by the Task 5 oracle and is an honest miss, not a blocker.
3. **`keepdim` field is forward-looking** — added for §6.19 reduce-schema conformance (Task 7 serialization) but not consumed by `tag_to_op`, because Fuel's reduce Ops already encode keepdim structurally (`ReduceSumTo`/`ReduceMaxTo` carry the kept-shape; `SumDim`/`MeanDim` are rank-reducing). Documented as such rather than left as an unexplained unused field.
4. **Task 6 red step is non-idiomatic** — the builder routing is behavior-preserving, so there is no natural born-red test. The safety net is the full existing `fuel-graph` builder suite (capture green baseline first, keep green after each builder edit) plus the `builders_agree_with_primitive_shape` equivalence guard. Called out explicitly per the spec's "if extraction proves too invasive, fall back to the emit-only shape table" escape hatch — if a builder's inline math turns out to diverge from `primitive_shape` (a latent existing bug), surface it rather than paper over it.
5. **`matmul` builder** keeps its broadcast/GQA-divisibility prep (lib.rs:3935-3976) — that is a graph *rewrite* (it inserts `BroadcastTo` nodes), not single-op shape inference. `primitive_shape(MatMul)` only handles the final same-rank operands; the builder calls it after its prep. Verified against lib.rs:3977-3993.
6. **Oracle operand order is load-bearing.** `assert_structural_eq` is order-sensitive (no commutative canonicalization), so the Task 5 regions must mirror `decompose`'s exact input order (`Sub{[x, mb]}`, `Div{[e, db]}`, rope `Add{[left, right]}`, `Concat{[neg_second, first_half]}`). This is deliberate — it is a stricter check than `base_map_hash` (which sorts commutative operands) and catches an operand-swap bug that the hash would mask.
