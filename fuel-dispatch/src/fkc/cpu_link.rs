//! The built-in CPU backend's FKC `link_registry` (kernel-seam-interop §3.5,
//! §4.3; FKC §12.6). Maps each CPU kernel contract's `entry_point` symbol to
//! the production dispatch wrapper — the real, non-stub resolution the importer
//! uses so an imported contract binds the **actual** kernel (no raw pointers in
//! the serialized contract, FKC P9).
//!
//! For the built-in CPU backend the wrappers and this table co-locate in
//! fuel-dispatch — the dispatch layer that adapts raw byte-kernels to
//! [`KernelRef`]. An *external* provider (e.g. Baracuda) instead exports its own
//! link registry across the FFI; this is Fuel's internal-provider analogue, and
//! the first FKC conformance reference.

use crate::fkc::lower::LinkRegistry;
use crate::kernel::KernelRef;

/// One `(contract entry_point symbol, production wrapper)` pair. The symbol
/// matches the contract's `entry_point: "fuel_cpu_backend::byte_kernels::<op>_<dt>"`.
macro_rules! ep {
    ($op:literal, $dt:literal, $wrapper:ident) => {
        (
            concat!("fuel_cpu_backend::byte_kernels::", $op, "_", $dt),
            crate::dispatch::$wrapper as KernelRef,
        )
    };
}

/// Like [`ep!`] but for a wrapper whose contract `entry_point` symbol is its
/// FULLY-QUALIFIED `fuel_dispatch::dispatch::<wrapper>` path. The FlashAttn
/// BACKWARD wrappers live in the dispatch layer (they select which gradient the
/// shared byte-kernel writes via a `FaBackwardWhich` selector), so their
/// contract `entry_point` carries the `fuel_dispatch::dispatch::` prefix — NOT
/// `ep!`'s `fuel_cpu_backend::byte_kernels::<op>_<dt>` byte-kernel shape.
macro_rules! ep_dispatch {
    ($wrapper:ident) => {
        (
            concat!("fuel_dispatch::dispatch::", stringify!($wrapper)),
            crate::dispatch::$wrapper as KernelRef,
        )
    };
}

/// One `(FUSED-contract entry_point symbol, production wrapper)` pair for a
/// `fused_op` section. Unlike the primitive [`ep!`] / [`ep_dispatch!`] pairs, a
/// fused contract declares a **dtype-agnostic BASE** `entry_point`
/// `fuel_dispatch::dispatch::<op>_cpu` (e.g. `…::softmax_last_dim_cpu`) plus a
/// per-input `dtypes: [F32, F64, BF16, F16]` list, so [`crate::fkc::lower::lower_fused`]
/// **dtype-fans** it (§3.4, the fused analogue of the primitive fan-out): per
/// fanned `dt` it resolves `<base>_<dt>` (e.g. `…::softmax_last_dim_cpu_f32`)
/// through this table. Each such fanned symbol binds the `<op>_<dt>_cpu_wrapper`
/// — the exact kernel fn the hand-written `register_default_fused_kernels` seam
/// registers for that op's `dt` dtype tuple. Note the naming skew: the FANNED
/// symbol is `<op>_cpu_<dt>` (base `…_cpu` + `_<dt>` suffix), while the wrapper
/// fn is `<op>_<dt>_cpu_wrapper` (`_cpu` after `_<dt>`), so the mapping is
/// spelled out per (op, dtype) row.
macro_rules! fep {
    ($symbol:literal, $wrapper:ident) => {
        (
            concat!("fuel_dispatch::dispatch::", $symbol),
            crate::dispatch::$wrapper as KernelRef,
        )
    };
}

/// The CPU elementwise-binary family's `symbol → production wrapper` map
/// (8 ops × 4 dtypes). The chassis umbrella section is `registrable: false`
/// (§3.10 describe-only), so it never reaches resolution and is absent here.
pub static CPU_BINARY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("add", "f32", add_elementwise_f32_cpu_wrapper),
    ep!("add", "f64", add_elementwise_f64_cpu_wrapper),
    ep!("add", "f16", add_elementwise_f16_cpu_wrapper),
    ep!("add", "bf16", add_elementwise_bf16_cpu_wrapper),
    ep!("sub", "f32", sub_elementwise_f32_cpu_wrapper),
    ep!("sub", "f64", sub_elementwise_f64_cpu_wrapper),
    ep!("sub", "f16", sub_elementwise_f16_cpu_wrapper),
    ep!("sub", "bf16", sub_elementwise_bf16_cpu_wrapper),
    ep!("mul", "f32", mul_elementwise_f32_cpu_wrapper),
    ep!("mul", "f64", mul_elementwise_f64_cpu_wrapper),
    ep!("mul", "f16", mul_elementwise_f16_cpu_wrapper),
    ep!("mul", "bf16", mul_elementwise_bf16_cpu_wrapper),
    ep!("div", "f32", div_elementwise_f32_cpu_wrapper),
    ep!("div", "f64", div_elementwise_f64_cpu_wrapper),
    ep!("div", "f16", div_elementwise_f16_cpu_wrapper),
    ep!("div", "bf16", div_elementwise_bf16_cpu_wrapper),
    ep!("maximum", "f32", maximum_elementwise_f32_cpu_wrapper),
    ep!("maximum", "f64", maximum_elementwise_f64_cpu_wrapper),
    ep!("maximum", "f16", maximum_elementwise_f16_cpu_wrapper),
    ep!("maximum", "bf16", maximum_elementwise_bf16_cpu_wrapper),
    ep!("minimum", "f32", minimum_elementwise_f32_cpu_wrapper),
    ep!("minimum", "f64", minimum_elementwise_f64_cpu_wrapper),
    ep!("minimum", "f16", minimum_elementwise_f16_cpu_wrapper),
    ep!("minimum", "bf16", minimum_elementwise_bf16_cpu_wrapper),
    ep!("pow", "f32", pow_elementwise_f32_cpu_wrapper),
    ep!("pow", "f64", pow_elementwise_f64_cpu_wrapper),
    ep!("pow", "f16", pow_elementwise_f16_cpu_wrapper),
    ep!("pow", "bf16", pow_elementwise_bf16_cpu_wrapper),
    ep!("rem", "f32", rem_elementwise_f32_cpu_wrapper),
    ep!("rem", "f64", rem_elementwise_f64_cpu_wrapper),
    ep!("rem", "f16", rem_elementwise_f16_cpu_wrapper),
    ep!("rem", "bf16", rem_elementwise_bf16_cpu_wrapper),
];

/// The CPU out-of-place scalar-param family's `symbol → production wrapper`
/// map (affine / clamp / powi × 4 dtypes + powi_backward × 4 = 16 kernels).
/// Contract: `docs/kernel-contracts/cpu/affine-clamp-powi.fkc.md`. The scalar
/// params (affine mul/add, clamp min/max, powi exp) ride in `OpParams`, NOT the
/// dtype-list, so the binding keys stay `[t, t]` for the single-input forward
/// ops and `[t, t, t]` for the two-input `powi_backward`. The `ep!` symbol is
/// built from `$op`/`$dt`, so the three f32 hand-written wrappers whose fn-name
/// differs from the symbol (`clamp_elementwise_f32`, `powi_elementwise_f32`)
/// still map to the contract's `clamp_f32` / `powi_f32` entry points.
pub static CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // affine (y = mul*x + add)
    ep!("affine", "f32",  affine_f32_cpu_wrapper),
    ep!("affine", "f64",  affine_f64_cpu_wrapper),
    ep!("affine", "bf16", affine_bf16_cpu_wrapper),
    ep!("affine", "f16",  affine_f16_cpu_wrapper),
    // clamp (y = clamp(x, min, max))
    ep!("clamp",  "f32",  clamp_elementwise_f32_cpu_wrapper),
    ep!("clamp",  "f64",  clamp_f64_cpu_wrapper),
    ep!("clamp",  "bf16", clamp_bf16_cpu_wrapper),
    ep!("clamp",  "f16",  clamp_f16_cpu_wrapper),
    // powi (y = x.powi(exp))
    ep!("powi",   "f32",  powi_elementwise_f32_cpu_wrapper),
    ep!("powi",   "f64",  powi_f64_cpu_wrapper),
    ep!("powi",   "bf16", powi_bf16_cpu_wrapper),
    ep!("powi",   "f16",  powi_f16_cpu_wrapper),
    // powi_backward (grad_x = exp*x^(exp-1)*upstream) — TWO inputs (x, upstream)
    ep!("powi_backward", "f32",  powi_backward_f32_cpu_wrapper),
    ep!("powi_backward", "f64",  powi_backward_f64_cpu_wrapper),
    ep!("powi_backward", "bf16", powi_backward_bf16_cpu_wrapper),
    ep!("powi_backward", "f16",  powi_backward_f16_cpu_wrapper),
];

/// The CPU elementwise-unary family's `symbol → production wrapper` map
/// (22 ops × 4 dtypes = 88 kernels). Contract:
/// `docs/kernel-contracts/cpu/elementwise-unary.fkc.md`. Each per-op section
/// declares a BASE `entry_point` (e.g. `…::relu`) and enumerates
/// `dtypes: [F32,F64,BF16,F16]`; the importer's §3.4 multi-dtype fan-out then
/// resolves `<base>_<dtype>` (e.g. `relu_f32`) against this table — so the
/// `$op` literals below are the byte-kernel BASES, NOT the OpKind names. The
/// two GELU flavors stay distinct: `gelu_tanh` (`OpKind::GeluElementwise`) has
/// base `gelu` (wrapper `gelu_elementwise_<dt>`), while `gelu_erf`
/// (`OpKind::GeluErfElementwise`) has base `gelu_erf`. The `unary` chassis
/// umbrella is `registrable: false` (§3.10 describe-only) and never resolves,
/// so it is absent here.
pub static CPU_UNARY_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("relu", "f32",  relu_elementwise_f32_cpu_wrapper),
    ep!("relu", "f64",  relu_elementwise_f64_cpu_wrapper),
    ep!("relu", "bf16", relu_elementwise_bf16_cpu_wrapper),
    ep!("relu", "f16",  relu_elementwise_f16_cpu_wrapper),
    ep!("neg", "f32",  neg_elementwise_f32_cpu_wrapper),
    ep!("neg", "f64",  neg_elementwise_f64_cpu_wrapper),
    ep!("neg", "bf16", neg_elementwise_bf16_cpu_wrapper),
    ep!("neg", "f16",  neg_elementwise_f16_cpu_wrapper),
    ep!("sqr", "f32",  sqr_elementwise_f32_cpu_wrapper),
    ep!("sqr", "f64",  sqr_elementwise_f64_cpu_wrapper),
    ep!("sqr", "bf16", sqr_elementwise_bf16_cpu_wrapper),
    ep!("sqr", "f16",  sqr_elementwise_f16_cpu_wrapper),
    ep!("sqrt", "f32",  sqrt_elementwise_f32_cpu_wrapper),
    ep!("sqrt", "f64",  sqrt_elementwise_f64_cpu_wrapper),
    ep!("sqrt", "bf16", sqrt_elementwise_bf16_cpu_wrapper),
    ep!("sqrt", "f16",  sqrt_elementwise_f16_cpu_wrapper),
    ep!("recip", "f32",  recip_elementwise_f32_cpu_wrapper),
    ep!("recip", "f64",  recip_elementwise_f64_cpu_wrapper),
    ep!("recip", "bf16", recip_elementwise_bf16_cpu_wrapper),
    ep!("recip", "f16",  recip_elementwise_f16_cpu_wrapper),
    ep!("abs", "f32",  abs_elementwise_f32_cpu_wrapper),
    ep!("abs", "f64",  abs_elementwise_f64_cpu_wrapper),
    ep!("abs", "bf16", abs_elementwise_bf16_cpu_wrapper),
    ep!("abs", "f16",  abs_elementwise_f16_cpu_wrapper),
    ep!("tanh", "f32",  tanh_elementwise_f32_cpu_wrapper),
    ep!("tanh", "f64",  tanh_elementwise_f64_cpu_wrapper),
    ep!("tanh", "bf16", tanh_elementwise_bf16_cpu_wrapper),
    ep!("tanh", "f16",  tanh_elementwise_f16_cpu_wrapper),
    ep!("exp", "f32",  exp_elementwise_f32_cpu_wrapper),
    ep!("exp", "f64",  exp_elementwise_f64_cpu_wrapper),
    ep!("exp", "bf16", exp_elementwise_bf16_cpu_wrapper),
    ep!("exp", "f16",  exp_elementwise_f16_cpu_wrapper),
    ep!("log", "f32",  log_elementwise_f32_cpu_wrapper),
    ep!("log", "f64",  log_elementwise_f64_cpu_wrapper),
    ep!("log", "bf16", log_elementwise_bf16_cpu_wrapper),
    ep!("log", "f16",  log_elementwise_f16_cpu_wrapper),
    ep!("sin", "f32",  sin_elementwise_f32_cpu_wrapper),
    ep!("sin", "f64",  sin_elementwise_f64_cpu_wrapper),
    ep!("sin", "bf16", sin_elementwise_bf16_cpu_wrapper),
    ep!("sin", "f16",  sin_elementwise_f16_cpu_wrapper),
    ep!("cos", "f32",  cos_elementwise_f32_cpu_wrapper),
    ep!("cos", "f64",  cos_elementwise_f64_cpu_wrapper),
    ep!("cos", "bf16", cos_elementwise_bf16_cpu_wrapper),
    ep!("cos", "f16",  cos_elementwise_f16_cpu_wrapper),
    ep!("sigmoid", "f32",  sigmoid_elementwise_f32_cpu_wrapper),
    ep!("sigmoid", "f64",  sigmoid_elementwise_f64_cpu_wrapper),
    ep!("sigmoid", "bf16", sigmoid_elementwise_bf16_cpu_wrapper),
    ep!("sigmoid", "f16",  sigmoid_elementwise_f16_cpu_wrapper),
    ep!("silu", "f32",  silu_elementwise_f32_cpu_wrapper),
    ep!("silu", "f64",  silu_elementwise_f64_cpu_wrapper),
    ep!("silu", "bf16", silu_elementwise_bf16_cpu_wrapper),
    ep!("silu", "f16",  silu_elementwise_f16_cpu_wrapper),
    ep!("step", "f32",  step_elementwise_f32_cpu_wrapper),
    ep!("step", "f64",  step_elementwise_f64_cpu_wrapper),
    ep!("step", "bf16", step_elementwise_bf16_cpu_wrapper),
    ep!("step", "f16",  step_elementwise_f16_cpu_wrapper),
    // gelu_tanh (the canonical Gelu): base `gelu`, wrapper `gelu_elementwise_*`.
    ep!("gelu", "f32",  gelu_elementwise_f32_cpu_wrapper),
    ep!("gelu", "f64",  gelu_elementwise_f64_cpu_wrapper),
    ep!("gelu", "bf16", gelu_elementwise_bf16_cpu_wrapper),
    ep!("gelu", "f16",  gelu_elementwise_f16_cpu_wrapper),
    ep!("floor", "f32",  floor_elementwise_f32_cpu_wrapper),
    ep!("floor", "f64",  floor_elementwise_f64_cpu_wrapper),
    ep!("floor", "bf16", floor_elementwise_bf16_cpu_wrapper),
    ep!("floor", "f16",  floor_elementwise_f16_cpu_wrapper),
    ep!("ceil", "f32",  ceil_elementwise_f32_cpu_wrapper),
    ep!("ceil", "f64",  ceil_elementwise_f64_cpu_wrapper),
    ep!("ceil", "bf16", ceil_elementwise_bf16_cpu_wrapper),
    ep!("ceil", "f16",  ceil_elementwise_f16_cpu_wrapper),
    ep!("round", "f32",  round_elementwise_f32_cpu_wrapper),
    ep!("round", "f64",  round_elementwise_f64_cpu_wrapper),
    ep!("round", "bf16", round_elementwise_bf16_cpu_wrapper),
    ep!("round", "f16",  round_elementwise_f16_cpu_wrapper),
    ep!("sign", "f32",  sign_elementwise_f32_cpu_wrapper),
    ep!("sign", "f64",  sign_elementwise_f64_cpu_wrapper),
    ep!("sign", "bf16", sign_elementwise_bf16_cpu_wrapper),
    ep!("sign", "f16",  sign_elementwise_f16_cpu_wrapper),
    ep!("erf", "f32",  erf_elementwise_f32_cpu_wrapper),
    ep!("erf", "f64",  erf_elementwise_f64_cpu_wrapper),
    ep!("erf", "bf16", erf_elementwise_bf16_cpu_wrapper),
    ep!("erf", "f16",  erf_elementwise_f16_cpu_wrapper),
    // gelu_erf (exact-erf GELU): base `gelu_erf`, DISTINCT from `gelu` above.
    ep!("gelu_erf", "f32",  gelu_erf_elementwise_f32_cpu_wrapper),
    ep!("gelu_erf", "f64",  gelu_erf_elementwise_f64_cpu_wrapper),
    ep!("gelu_erf", "bf16", gelu_erf_elementwise_bf16_cpu_wrapper),
    ep!("gelu_erf", "f16",  gelu_erf_elementwise_f16_cpu_wrapper),
    ep!("rsqrt", "f32",  rsqrt_elementwise_f32_cpu_wrapper),
    ep!("rsqrt", "f64",  rsqrt_elementwise_f64_cpu_wrapper),
    ep!("rsqrt", "bf16", rsqrt_elementwise_bf16_cpu_wrapper),
    ep!("rsqrt", "f16",  rsqrt_elementwise_f16_cpu_wrapper),
];

/// The CPU elementwise-COMPARE family's `symbol → production wrapper` map
/// (6 ops × 4 dtypes = 24). Contract:
/// `docs/kernel-contracts/cpu/compare-where.fkc.md`. Each thunk is a
/// single-(op,dtype) section (its `lhs`/`rhs` enumerate ONE dtype), so it does
/// NOT fan — the importer resolves its declared symbol AS-IS. That symbol
/// carries the `_u8` output-mask suffix (`T × T → U8` mask, `return.out:
/// fixed(U8)`), so the `ep!` dtype slot is `<dt>_u8` (e.g. `eq_f32_u8`), NOT
/// the plain `<dt>` the binary/unary families use. The `## compare` chassis
/// umbrella is `registrable: false` (§3.10 describe-only) and never resolves,
/// so it is absent here.
pub static CPU_COMPARE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("eq", "f32_u8",  eq_elementwise_f32_cpu_wrapper),
    ep!("eq", "f64_u8",  eq_elementwise_f64_cpu_wrapper),
    ep!("eq", "bf16_u8", eq_elementwise_bf16_cpu_wrapper),
    ep!("eq", "f16_u8",  eq_elementwise_f16_cpu_wrapper),
    ep!("ne", "f32_u8",  ne_elementwise_f32_cpu_wrapper),
    ep!("ne", "f64_u8",  ne_elementwise_f64_cpu_wrapper),
    ep!("ne", "bf16_u8", ne_elementwise_bf16_cpu_wrapper),
    ep!("ne", "f16_u8",  ne_elementwise_f16_cpu_wrapper),
    ep!("lt", "f32_u8",  lt_elementwise_f32_cpu_wrapper),
    ep!("lt", "f64_u8",  lt_elementwise_f64_cpu_wrapper),
    ep!("lt", "bf16_u8", lt_elementwise_bf16_cpu_wrapper),
    ep!("lt", "f16_u8",  lt_elementwise_f16_cpu_wrapper),
    ep!("le", "f32_u8",  le_elementwise_f32_cpu_wrapper),
    ep!("le", "f64_u8",  le_elementwise_f64_cpu_wrapper),
    ep!("le", "bf16_u8", le_elementwise_bf16_cpu_wrapper),
    ep!("le", "f16_u8",  le_elementwise_f16_cpu_wrapper),
    ep!("gt", "f32_u8",  gt_elementwise_f32_cpu_wrapper),
    ep!("gt", "f64_u8",  gt_elementwise_f64_cpu_wrapper),
    ep!("gt", "bf16_u8", gt_elementwise_bf16_cpu_wrapper),
    ep!("gt", "f16_u8",  gt_elementwise_f16_cpu_wrapper),
    ep!("ge", "f32_u8",  ge_elementwise_f32_cpu_wrapper),
    ep!("ge", "f64_u8",  ge_elementwise_f64_cpu_wrapper),
    ep!("ge", "bf16_u8", ge_elementwise_bf16_cpu_wrapper),
    ep!("ge", "f16_u8",  ge_elementwise_f16_cpu_wrapper),
];

/// The CPU ternary-select (`where`) family's `symbol → production wrapper` map
/// (1 op × 4 dtypes = 4). Contract:
/// `docs/kernel-contracts/cpu/compare-where.fkc.md`. The single `where_kernel`
/// section enumerates `a`/`b` `dtypes: [F32,F64,BF16,F16]`, so it FANS (§3.4):
/// its declared BASE `entry_point` `…::where` resolves `<base>_<dtype>` =
/// `where_{f32,f64,bf16,f16}` against this table. The binding key is
/// `[U8, T, T, T]` (cond U8 + a/b/out share T; `out: passthrough(a)` → T).
pub static CPU_WHERE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("where", "f32",  where_f32_cpu_wrapper),
    ep!("where", "f64",  where_f64_cpu_wrapper),
    ep!("where", "bf16", where_bf16_cpu_wrapper),
    ep!("where", "f16",  where_f16_cpu_wrapper),
];

/// The CPU per-axis REDUCE family's `symbol → production wrapper` map
/// (4 ops × 4 dtypes = 16). Contract: `docs/kernel-contracts/cpu/reduce.fkc.md`.
/// Each per-(op, dtype) section (`## sum_reduce_f32`, …) declares a SPECIFIC
/// single-dtype `entry_point` (`…::sum_reduce_f32`), so it does NOT fan — the
/// importer resolves that symbol AS-IS. The binding key is `[T, T]` (input +
/// `passthrough(input)` output; the reduce axes + keepdim ride in
/// `OpParams::Reduce`, NOT the dtype-list). The `## reduce` chassis umbrella is
/// `registrable: false` (§3.10 describe-only) and never resolves, so it is
/// absent here; the f32-only `argmax_dim_f32` / `argmin_dim_f32` sections are
/// `registrable: false` (deferred — production registers Arg{Max,Min}Dim for all
/// input dtypes via a hand-written dispatch) and are likewise absent.
pub static CPU_REDUCE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("sum_reduce",  "f32",  sum_reduce_f32_cpu_wrapper),
    ep!("sum_reduce",  "f64",  sum_reduce_f64_cpu_wrapper),
    ep!("sum_reduce",  "bf16", sum_reduce_bf16_cpu_wrapper),
    ep!("sum_reduce",  "f16",  sum_reduce_f16_cpu_wrapper),
    ep!("mean_reduce", "f32",  mean_reduce_f32_cpu_wrapper),
    ep!("mean_reduce", "f64",  mean_reduce_f64_cpu_wrapper),
    ep!("mean_reduce", "bf16", mean_reduce_bf16_cpu_wrapper),
    ep!("mean_reduce", "f16",  mean_reduce_f16_cpu_wrapper),
    ep!("max_reduce",  "f32",  max_reduce_f32_cpu_wrapper),
    ep!("max_reduce",  "f64",  max_reduce_f64_cpu_wrapper),
    ep!("max_reduce",  "bf16", max_reduce_bf16_cpu_wrapper),
    ep!("max_reduce",  "f16",  max_reduce_f16_cpu_wrapper),
    ep!("min_reduce",  "f32",  min_reduce_f32_cpu_wrapper),
    ep!("min_reduce",  "f64",  min_reduce_f64_cpu_wrapper),
    ep!("min_reduce",  "bf16", min_reduce_bf16_cpu_wrapper),
    ep!("min_reduce",  "f16",  min_reduce_f16_cpu_wrapper),
];

/// The CPU broadcast-target REDUCE-TO family's `symbol → production wrapper`
/// map (ReduceSumTo / ReduceMaxTo × 4 dtypes = 8, key `[T, T]`, +
/// ReduceMaxToBackward × 4 dtypes = 4, key `[T, T, T]` = 12). Contract:
/// `docs/kernel-contracts/cpu/reduce-to.fkc.md`. Each per-(op, dtype) section
/// (`## reduce_sum_to_f32`, …) declares a SPECIFIC single-dtype `entry_point`
/// (`…::reduce_sum_to_f32`), so it does NOT fan — the importer resolves that
/// symbol AS-IS. The forward keys are `[T, T]` (input + `passthrough(input)`
/// output; the target `input_shape`/`output_shape` ride in
/// `OpParams::ReduceSumTo` / `OpParams::ReduceMaxTo`, NOT the dtype-list); the
/// backward key is `[T, T, T]` (x, upstream + `passthrough(x)` output). The
/// `## reduce_to` chassis umbrella is `registrable: false` (§3.10 describe-only)
/// and never resolves, so it is absent here (without it the chassis would
/// double-register `ReduceSumTo`/`[F32]` → `DuplicateKernelRef` at init).
pub static CPU_REDUCE_TO_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("reduce_sum_to", "f32",  reduce_sum_to_f32_cpu_wrapper),
    ep!("reduce_sum_to", "f64",  reduce_sum_to_f64_cpu_wrapper),
    ep!("reduce_sum_to", "bf16", reduce_sum_to_bf16_cpu_wrapper),
    ep!("reduce_sum_to", "f16",  reduce_sum_to_f16_cpu_wrapper),
    ep!("reduce_max_to", "f32",  reduce_max_to_f32_cpu_wrapper),
    ep!("reduce_max_to", "f64",  reduce_max_to_f64_cpu_wrapper),
    ep!("reduce_max_to", "bf16", reduce_max_to_bf16_cpu_wrapper),
    ep!("reduce_max_to", "f16",  reduce_max_to_f16_cpu_wrapper),
    ep!("reduce_max_to_backward", "f32",  reduce_max_to_backward_f32_cpu_wrapper),
    ep!("reduce_max_to_backward", "f64",  reduce_max_to_backward_f64_cpu_wrapper),
    ep!("reduce_max_to_backward", "bf16", reduce_max_to_backward_bf16_cpu_wrapper),
    ep!("reduce_max_to_backward", "f16",  reduce_max_to_backward_f16_cpu_wrapper),
];

/// The CPU last-dim NORM (forward) family's `symbol → production wrapper` map
/// (Softmax / LogSoftmax / RmsNorm / LayerNorm last-dim × 4 dtypes = 16). Contract:
/// `docs/kernel-contracts/cpu/norm.fkc.md`. Each per-(op, dtype) section
/// (`## softmax_last_dim_f32`, `## rms_norm_last_dim_f32`, …) declares a SPECIFIC
/// single-dtype `entry_point` (`…::softmax_last_dim_f32`), so none of them fan —
/// the importer resolves that symbol AS-IS. The binding key is `[T, T]` (a SINGLE
/// input + `passthrough(input)` output; the RMS/LayerNorm kernels carry NO affine
/// gamma/beta operand — they are the bare normalization — and `outer_count` /
/// `last_dim` / `eps` ride in `OpParams::{SoftmaxLastDim,LogSoftmaxLastDim,
/// NormLastDim}`, NOT the dtype-list), identical to the deleted `&unary(t)` regs.
/// The `log_softmax` wrapper fn-names (`log_softmax_<dt>_cpu_wrapper`) differ from
/// their `log_softmax_last_dim_<dt>` symbol — the `ep!` symbol is built from the
/// `$op`/`$dt` literals, so the mapping still binds the correct contract symbol
/// (mirrors the clamp/powi/where fn-vs-symbol cases). This contract has NO `##`
/// chassis umbrella section, so there is no `registrable: false` describe-only
/// entry to omit; the BACKWARD forms live in a separate norm-backward contract.
pub static CPU_NORM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("softmax_last_dim", "f32",  softmax_last_dim_f32_cpu_wrapper),
    ep!("softmax_last_dim", "f64",  softmax_last_dim_f64_cpu_wrapper),
    ep!("softmax_last_dim", "bf16", softmax_last_dim_bf16_cpu_wrapper),
    ep!("softmax_last_dim", "f16",  softmax_last_dim_f16_cpu_wrapper),
    ep!("log_softmax_last_dim", "f32",  log_softmax_f32_cpu_wrapper),
    ep!("log_softmax_last_dim", "f64",  log_softmax_f64_cpu_wrapper),
    ep!("log_softmax_last_dim", "bf16", log_softmax_bf16_cpu_wrapper),
    ep!("log_softmax_last_dim", "f16",  log_softmax_f16_cpu_wrapper),
    ep!("rms_norm_last_dim", "f32",  rms_norm_last_dim_f32_cpu_wrapper),
    ep!("rms_norm_last_dim", "f64",  rms_norm_last_dim_f64_cpu_wrapper),
    ep!("rms_norm_last_dim", "bf16", rms_norm_last_dim_bf16_cpu_wrapper),
    ep!("rms_norm_last_dim", "f16",  rms_norm_last_dim_f16_cpu_wrapper),
    ep!("layer_norm_last_dim", "f32",  layer_norm_last_dim_f32_cpu_wrapper),
    ep!("layer_norm_last_dim", "f64",  layer_norm_last_dim_f64_cpu_wrapper),
    ep!("layer_norm_last_dim", "bf16", layer_norm_last_dim_bf16_cpu_wrapper),
    ep!("layer_norm_last_dim", "f16",  layer_norm_last_dim_f16_cpu_wrapper),
];

/// The CPU last-dim NORM-BACKWARD family's `symbol → production wrapper` map
/// (Softmax / LogSoftmax / RmsNorm / LayerNorm last-dim BACKWARD × 4 dtypes = 16).
/// Contract: `docs/kernel-contracts/cpu/norm-backward.fkc.md`. Each per-(op, dtype)
/// section (`## softmax_last_dim_backward_f32`, `## rms_norm_last_dim_backward_f32`,
/// …) declares a SPECIFIC single-dtype `entry_point`
/// (`…::softmax_last_dim_backward_f32`), so none of them fan — the importer
/// resolves that symbol AS-IS. The binding key is `[T, T, T]` — the BARE backward
/// takes TWO inputs (softmax/log-softmax: the forward output `y` + the upstream
/// gradient `g`; layer/rms-norm: the forward input `x` + `g`, stats recomputed) and
/// writes ONE `passthrough(y|x)` output, and outer_count / last_dim / eps ride in
/// `OpParams::{SoftmaxLastDim,LogSoftmaxLastDim,NormLastDim}`, NOT the dtype-list —
/// identical to the deleted `&binary(t)` regs. The `log_softmax` backward wrapper
/// fn-names (`log_softmax_backward_<dt>_cpu_wrapper`) differ from their
/// `log_softmax_last_dim_backward_<dt>` symbol — the `ep!` symbol is built from the
/// `$op`/`$dt` literals, so the mapping still binds the correct contract symbol
/// (mirrors the forward `log_softmax` fn-vs-symbol case). This contract has NO `##`
/// chassis umbrella section, so there is no `registrable: false` describe-only entry
/// to omit; the FORWARD forms live in the separate norm contract.
pub static CPU_NORM_BACKWARD_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("softmax_last_dim_backward", "f32",  softmax_last_dim_backward_f32_cpu_wrapper),
    ep!("softmax_last_dim_backward", "f64",  softmax_last_dim_backward_f64_cpu_wrapper),
    ep!("softmax_last_dim_backward", "bf16", softmax_last_dim_backward_bf16_cpu_wrapper),
    ep!("softmax_last_dim_backward", "f16",  softmax_last_dim_backward_f16_cpu_wrapper),
    ep!("log_softmax_last_dim_backward", "f32",  log_softmax_backward_f32_cpu_wrapper),
    ep!("log_softmax_last_dim_backward", "f64",  log_softmax_backward_f64_cpu_wrapper),
    ep!("log_softmax_last_dim_backward", "bf16", log_softmax_backward_bf16_cpu_wrapper),
    ep!("log_softmax_last_dim_backward", "f16",  log_softmax_backward_f16_cpu_wrapper),
    ep!("rms_norm_last_dim_backward", "f32",  rms_norm_last_dim_backward_f32_cpu_wrapper),
    ep!("rms_norm_last_dim_backward", "f64",  rms_norm_last_dim_backward_f64_cpu_wrapper),
    ep!("rms_norm_last_dim_backward", "bf16", rms_norm_last_dim_backward_bf16_cpu_wrapper),
    ep!("rms_norm_last_dim_backward", "f16",  rms_norm_last_dim_backward_f16_cpu_wrapper),
    ep!("layer_norm_last_dim_backward", "f32",  layer_norm_last_dim_backward_f32_cpu_wrapper),
    ep!("layer_norm_last_dim_backward", "f64",  layer_norm_last_dim_backward_f64_cpu_wrapper),
    ep!("layer_norm_last_dim_backward", "bf16", layer_norm_last_dim_backward_bf16_cpu_wrapper),
    ep!("layer_norm_last_dim_backward", "f16",  layer_norm_last_dim_backward_f16_cpu_wrapper),
];

/// The CPU RoPE (rotary position embedding) family's `symbol → production
/// wrapper` map (1 op × 4 dtypes = 4). Contract:
/// `docs/kernel-contracts/cpu/rope.fkc.md`. Each per-dtype section
/// (`## rope_f32`, …) declares a SPECIFIC single-dtype `entry_point`
/// (`…::rope_f32`), so none of them fan — the importer resolves that symbol
/// AS-IS. The binding key is `[T, T, T, T]` — RoPE takes THREE inputs (`x` +
/// the precomputed `cos`/`sin` tables, all one dtype; the `[seq, head_dim]`
/// tables broadcast over the `outer_count` axis by the kernel re-indexing them
/// per outer, NOT a stride-0 view) and writes ONE `passthrough(x)` output;
/// outer_count / seq / head_dim ride in `OpParams::Rope`, NOT the dtype-list —
/// identical to the deleted `rope_dts(t)` regs. This contract has NO `##`
/// chassis umbrella section, so there is no `registrable: false` describe-only
/// entry to omit. RoPE is ALSO registered in the `FusedKernelRegistry`
/// (`register_default_fused_kernels`, `FusedOps::ROPE`) — that is a SEPARATE
/// registry seam and stays untouched; this map only serves the
/// `KernelBindingTable` primitive path.
pub static CPU_ROPE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("rope", "f32",  rope_f32_cpu_wrapper),
    ep!("rope", "f64",  rope_f64_cpu_wrapper),
    ep!("rope", "bf16", rope_bf16_cpu_wrapper),
    ep!("rope", "f16",  rope_f16_cpu_wrapper),
];

/// The CPU SSM / Mamba family's `symbol → production wrapper` map — the FULL
/// family: FusedSoftmaxCrossEntropy + CausalConv1d + SelectiveScan +
/// SsdChunkScan (4 ops × 4 dtypes = 16). Contract:
/// `docs/kernel-contracts/cpu/ssm.fkc.md`. Each per-(op, dtype) section
/// (`## fused_softmax_cross_entropy_f32`, `## selective_scan_f32`, …) declares a
/// SPECIFIC single-dtype `entry_point` (`…::selective_scan_f32`), so none of
/// them fan — the importer resolves that symbol AS-IS. Binding keys:
/// - FSCE: `[T, I64, F32]` (logits T + I64 targets → `fixed(F32)` output;
///   n_rows / vocab / reduction / ignore_index ride in
///   `OpParams::FusedSoftmaxCrossEntropy`, NOT the dtype-list).
/// - CausalConv1d: `[T, T, T, T]` (x, weight, bias + `passthrough(x)` output;
///   batch / channels / seq / kernel / use_silu ride in `OpParams::CausalConv1d`).
/// - SelectiveScan / SsdChunkScan: `[T; 6]` — 5 inputs (u,delta,a,b,c /
///   x,dt,a,b,c) + the ONE bundled output slot. These sections return a
///   `return.bundle` multi-output (Option C, one packed buffer `[y ; last_state]`);
///   the importer's key-builder now appends the bundle's PRIMARY-slot dtype
///   (`passthrough(u)` / `passthrough(x)` → T) to the key tail (`fkc/lower.rs`
///   `assemble_dtype_variants`), so a bundle section keys `[T; 6]` byte-for-byte
///   the deleted hand-written reg. The geometry / delta_softplus / chunk_size
///   ride in `OpParams::{SelectiveScan, SsdChunkScan}`, NOT the dtype-list.
///
/// These sections have NO `##` chassis umbrella, so there is no
/// `registrable: false` describe-only entry to omit.
pub static CPU_SSM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("fused_softmax_cross_entropy", "f32",  fused_softmax_cross_entropy_f32_cpu_wrapper),
    ep!("fused_softmax_cross_entropy", "f64",  fused_softmax_cross_entropy_f64_cpu_wrapper),
    ep!("fused_softmax_cross_entropy", "bf16", fused_softmax_cross_entropy_bf16_cpu_wrapper),
    ep!("fused_softmax_cross_entropy", "f16",  fused_softmax_cross_entropy_f16_cpu_wrapper),
    ep!("causal_conv1d", "f32",  causal_conv1d_f32_cpu_wrapper),
    ep!("causal_conv1d", "f64",  causal_conv1d_f64_cpu_wrapper),
    ep!("causal_conv1d", "bf16", causal_conv1d_bf16_cpu_wrapper),
    ep!("causal_conv1d", "f16",  causal_conv1d_f16_cpu_wrapper),
    ep!("selective_scan", "f32",  selective_scan_f32_cpu_wrapper),
    ep!("selective_scan", "f64",  selective_scan_f64_cpu_wrapper),
    ep!("selective_scan", "bf16", selective_scan_bf16_cpu_wrapper),
    ep!("selective_scan", "f16",  selective_scan_f16_cpu_wrapper),
    ep!("ssd_chunk_scan", "f32",  ssd_chunk_scan_f32_cpu_wrapper),
    ep!("ssd_chunk_scan", "f64",  ssd_chunk_scan_f64_cpu_wrapper),
    ep!("ssd_chunk_scan", "bf16", ssd_chunk_scan_bf16_cpu_wrapper),
    ep!("ssd_chunk_scan", "f16",  ssd_chunk_scan_f16_cpu_wrapper),
];

/// The CPU 2D-convolution family's `symbol → production wrapper` map — the FULL
/// family: Conv2D + ConvTranspose2D × 4 dtypes = 8 symbols (each symbol serves
/// BOTH operand-count keys). Contract: `docs/kernel-contracts/cpu/conv.fkc.md`.
/// Each per-(op, dtype) section (`## conv2d_f32`, `## conv_transpose2d_f32`, …)
/// declares a SPECIFIC single-dtype `entry_point` (`…::conv2d_f32`), so none of
/// them dtype-fan — the importer resolves that symbol AS-IS. Because the contract
/// marks `bias` as `optional: true`, the importer's key-builder fans each section
/// into BOTH the no-bias key `[T, T, T]` (x, weight + out) and the with-bias key
/// `[T, T, T, T]` (x, weight, bias + `passthrough(x)` output) — both resolving
/// this SAME symbol/wrapper (the CPU wrapper handles 2 or 3 inputs). The spatial
/// geometry (x_shape/w_shape/out_shape, stride/padding/[output_padding]/dilation,
/// groups) rides in `OpParams::{Conv2D, ConvTranspose2D}`, NOT the dtype-list.
/// This map is the SOLE registration path for the whole family: every
/// hand-written `table.register(Conv2D/ConvTranspose2D, …)` reg (both operand
/// counts) is DELETED.
///
/// The conv ops' SEPARATE `FusedKernelRegistry` seam
/// (`register_default_fused_kernels`, `FusedOps::{CONV2D, CONV_TRANSPOSE2D}`,
/// both no-bias + with-bias) is untouched; this map only serves the
/// `KernelBindingTable` primitive path. These sections have NO `##` chassis
/// umbrella, so there is no `registrable: false` describe-only entry to omit.
pub static CPU_CONV_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("conv2d", "f32",  conv2d_f32_cpu_wrapper),
    ep!("conv2d", "f64",  conv2d_f64_cpu_wrapper),
    ep!("conv2d", "bf16", conv2d_bf16_cpu_wrapper),
    ep!("conv2d", "f16",  conv2d_f16_cpu_wrapper),
    ep!("conv_transpose2d", "f32",  conv_transpose2d_f32_cpu_wrapper),
    ep!("conv_transpose2d", "f64",  conv_transpose2d_f64_cpu_wrapper),
    ep!("conv_transpose2d", "bf16", conv_transpose2d_bf16_cpu_wrapper),
    ep!("conv_transpose2d", "f16",  conv_transpose2d_f16_cpu_wrapper),
];

/// The CPU **padding** family's `symbol → production wrapper` map — the FULL
/// family: mode-unified forward `Pad` × 6 dtypes = 6 + `PadBackward` × 4 dtypes
/// = 4 (all key `[T, T]`). Contract: `docs/kernel-contracts/cpu/padding.fkc.md`.
///
/// - **Forward `Pad`** is a dtype-agnostic byte kernel with a SINGLE production
///   wrapper — the mode-dispatching `pad_cpu_wrapper` (dispatch layer), which
///   selects `pad_const_cpu` / `pad_reflect_cpu` / `pad_replicate_cpu` at runtime
///   via `mode_tag` (incl. the reflect `before/after <= n-1` validation). The
///   contract's ONE registrable `## pad` section declares a BASE `entry_point`
///   (`…::pad_cpu`, a SYNTHETIC umbrella — there is no real `pad_cpu` byte kernel;
///   the three real mode kernels are `pad_{const,reflect,replicate}_cpu`) + an
///   enumerated `dtypes` list trimmed to production's wired 6
///   (`U8/U32/BF16/F16/F32/F64`), so the importer's §3.4 multi-dtype fan-out
///   resolves `pad_cpu_<dtype>` (a FABRICATED per-dtype symbol) — every dtype
///   variant mapping to the ONE `pad_cpu_wrapper`, exactly the Flip/Roll/Concat
///   pattern. The fan emits BYTE-FOR-BYTE the deleted hand-written
///   `table.register(Pad, &unary(t), …)` regs (key `[T, T]`, `out:
///   passthrough(input)`; the in_shape/out_shape/padding/mode_tag/fill_bytes ride
///   in `OpParams::Pad`, NOT the dtype-list). The three per-mode sections
///   (`pad_const_cpu` / `pad_reflect_cpu` / `pad_replicate_cpu`) and the
///   `pad_walk_cpu` helper are `registrable: false` (§3.10 describe-only mode
///   documentation) and never resolve, so they are ABSENT here.
/// - **`PadBackward`** is per-dtype (typed gradient accumulation — bf16/f16/f32
///   widen the scratch accumulator to f64, unlike the dtype-agnostic forward
///   copy). Each per-dtype section (`## pad_backward_f32`, …) declares a SPECIFIC
///   single-dtype `entry_point` (`…::pad_backward_f32`) resolved AS-IS (no fan),
///   key `[T, T]` (grad_out + `passthrough(grad_out)` grad_in; in_shape/out_shape/
///   padding/mode_tag ride in `OpParams::PadBackward`).
pub static CPU_PADDING_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Forward Pad — mode-unified dtype-agnostic byte kernel; the fabricated
    // `pad_cpu_<dt>` symbol maps to the ONE mode-dispatching `pad_cpu_wrapper`.
    ep!("pad_cpu", "u8",   pad_cpu_wrapper),
    ep!("pad_cpu", "u32",  pad_cpu_wrapper),
    ep!("pad_cpu", "bf16", pad_cpu_wrapper),
    ep!("pad_cpu", "f16",  pad_cpu_wrapper),
    ep!("pad_cpu", "f32",  pad_cpu_wrapper),
    ep!("pad_cpu", "f64",  pad_cpu_wrapper),
    // PadBackward — per-dtype typed accumulation; resolved AS-IS (no fan).
    ep!("pad_backward", "f32",  pad_backward_f32_cpu_wrapper),
    ep!("pad_backward", "f64",  pad_backward_f64_cpu_wrapper),
    ep!("pad_backward", "bf16", pad_backward_bf16_cpu_wrapper),
    ep!("pad_backward", "f16",  pad_backward_f16_cpu_wrapper),
];

/// The CPU **shape-ops** family's `symbol → production wrapper` map — the
/// MIGRATED subset (Flip + Roll + Concat + MaskedFill + WriteSlice +
/// WriteSliceRotating dtype-fanned + CumSum per-dtype). Contract:
/// `docs/kernel-contracts/cpu/shape-ops.fkc.md`.
///
/// - **Flip** / **Roll** / **Concat** / **MaskedFill** / **WriteSlice** /
///   **WriteSliceRotating** are **dtype-agnostic byte kernels** with a SINGLE
///   production wrapper each. Their contract sections declare a BASE
///   `entry_point` (`…::flip_cpu`, `…::roll_cpu`, `…::concat_cpu`,
///   `…::masked_fill_cpu`, `…::write_slice_cpu`, `…::write_slice_rotating_cpu`) +
///   an enumerated `dtypes` list, so the importer's §3.4 multi-dtype fan-out
///   resolves `<base>_<dtype>` (`flip_cpu_f32`, …) — a FABRICATED per-dtype
///   symbol (there is no real `flip_cpu_f32` fn) that every dtype variant maps to
///   the ONE dtype-agnostic wrapper. The contract's `dtypes` list was trimmed to
///   production's wired set (Flip/Roll/MaskedFill/WriteSlice/WriteSliceRotating 6:
///   F32/F64/BF16/F16/U32/U8; Concat 9: +I16/I32/I64) so the fan emits
///   BYTE-FOR-BYTE the deleted hand-written regs — Flip/Roll/Concat/WriteSlice/
///   WriteSliceRotating key `[T, T]`, MaskedFill key `[T, U8, T]` (`input` is the
///   sole varying operand; `mask` stays the fixed U8 slot; `out:
///   passthrough(input)`). **WriteSlice / WriteSliceRotating** model `dest` as the
///   in-place OUTPUT slot (not a key input) and — for the rotating op — the U32
///   `position` as a NON-KEY runtime operand, so the importer keys `[T_src, T_out]`
///   = `[T, T]`, matching `build_lookup_dtypes`' canonicalization exactly.
/// - **CumSum** is per-dtype (typed accumulation, NOT a byte copy — f32/f64
///   native, bf16/f16 widen to an f32 accumulator), so each `cumsum_<dt>`
///   section declares a SPECIFIC single-dtype `entry_point` resolved AS-IS (no
///   fan), key `[T, T]`, mapping to its OWN typed wrapper.
///
/// DEFERRED (absent here — never resolved): `contiguize` and `triangular` are
/// `registrable: false` chassis/describe-only sections (no `OpKind::Contiguize`;
/// `triangular` is the umbrella backing the two hand-written `Triu`/`Tril`
/// OpKinds, not a keyable section).
pub static CPU_SHAPE_OPS_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Flip — dtype-agnostic byte reorder; one wrapper fanned per dtype.
    ep!("flip_cpu", "f32",  flip_cpu_wrapper),
    ep!("flip_cpu", "f64",  flip_cpu_wrapper),
    ep!("flip_cpu", "bf16", flip_cpu_wrapper),
    ep!("flip_cpu", "f16",  flip_cpu_wrapper),
    ep!("flip_cpu", "u32",  flip_cpu_wrapper),
    ep!("flip_cpu", "u8",   flip_cpu_wrapper),
    // Roll — dtype-agnostic cyclic shift.
    ep!("roll_cpu", "f32",  roll_cpu_wrapper),
    ep!("roll_cpu", "f64",  roll_cpu_wrapper),
    ep!("roll_cpu", "bf16", roll_cpu_wrapper),
    ep!("roll_cpu", "f16",  roll_cpu_wrapper),
    ep!("roll_cpu", "u32",  roll_cpu_wrapper),
    ep!("roll_cpu", "u8",   roll_cpu_wrapper),
    // Concat — variadic uniform-dtype join collapsed to the [T, T] shorthand.
    ep!("concat_cpu", "f32",  concat_cpu_wrapper),
    ep!("concat_cpu", "f64",  concat_cpu_wrapper),
    ep!("concat_cpu", "bf16", concat_cpu_wrapper),
    ep!("concat_cpu", "f16",  concat_cpu_wrapper),
    ep!("concat_cpu", "u32",  concat_cpu_wrapper),
    ep!("concat_cpu", "u8",   concat_cpu_wrapper),
    ep!("concat_cpu", "i16",  concat_cpu_wrapper),
    ep!("concat_cpu", "i32",  concat_cpu_wrapper),
    ep!("concat_cpu", "i64",  concat_cpu_wrapper),
    // MaskedFill — dtype-agnostic data + fixed U8 mask; key [T, U8, T].
    ep!("masked_fill_cpu", "f32",  masked_fill_cpu_wrapper),
    ep!("masked_fill_cpu", "f64",  masked_fill_cpu_wrapper),
    ep!("masked_fill_cpu", "bf16", masked_fill_cpu_wrapper),
    ep!("masked_fill_cpu", "f16",  masked_fill_cpu_wrapper),
    ep!("masked_fill_cpu", "u32",  masked_fill_cpu_wrapper),
    ep!("masked_fill_cpu", "u8",   masked_fill_cpu_wrapper),
    // CumSum — per-dtype typed kernels, resolved AS-IS (no fan).
    ep!("cumsum", "f32",  cumsum_f32_cpu_wrapper),
    ep!("cumsum", "f64",  cumsum_f64_cpu_wrapper),
    ep!("cumsum", "bf16", cumsum_bf16_cpu_wrapper),
    ep!("cumsum", "f16",  cumsum_f16_cpu_wrapper),
    // WriteSlice — in-place rectangular scatter; dtype-agnostic byte copy, one
    // wrapper fanned per dtype. `dest` is the in-place OUTPUT slot, so the key is
    // `[T_src, T_out]` = [T, T]; the fabricated `write_slice_cpu_<dt>` symbol maps
    // to the ONE dtype-agnostic wrapper.
    ep!("write_slice_cpu", "f32",  write_slice_cpu_wrapper),
    ep!("write_slice_cpu", "f64",  write_slice_cpu_wrapper),
    ep!("write_slice_cpu", "bf16", write_slice_cpu_wrapper),
    ep!("write_slice_cpu", "f16",  write_slice_cpu_wrapper),
    ep!("write_slice_cpu", "u32",  write_slice_cpu_wrapper),
    ep!("write_slice_cpu", "u8",   write_slice_cpu_wrapper),
    // WriteSliceRotating — in-place ring-buffer scatter; same [T, T] key (dest =
    // output slot; the runtime `position` U32 operand is NOT a key slot). One
    // dtype-agnostic wrapper fanned per dtype.
    ep!("write_slice_rotating_cpu", "f32",  write_slice_rotating_cpu_wrapper),
    ep!("write_slice_rotating_cpu", "f64",  write_slice_rotating_cpu_wrapper),
    ep!("write_slice_rotating_cpu", "bf16", write_slice_rotating_cpu_wrapper),
    ep!("write_slice_rotating_cpu", "f16",  write_slice_rotating_cpu_wrapper),
    ep!("write_slice_rotating_cpu", "u32",  write_slice_rotating_cpu_wrapper),
    ep!("write_slice_rotating_cpu", "u8",   write_slice_rotating_cpu_wrapper),
    // WriteSliceDoff — in-place scatter with a device-resident I64 start on one
    // axis (no wrap). Same [T, T] key (dest = output slot; the runtime I64
    // `offset` operand is NOT a key slot). One dtype-agnostic wrapper per dtype.
    ep!("write_slice_doff_cpu", "f32",  write_slice_doff_cpu_wrapper),
    ep!("write_slice_doff_cpu", "f64",  write_slice_doff_cpu_wrapper),
    ep!("write_slice_doff_cpu", "bf16", write_slice_doff_cpu_wrapper),
    ep!("write_slice_doff_cpu", "f16",  write_slice_doff_cpu_wrapper),
    ep!("write_slice_doff_cpu", "u32",  write_slice_doff_cpu_wrapper),
    ep!("write_slice_doff_cpu", "u8",   write_slice_doff_cpu_wrapper),
];

/// The CPU **indexing / gather / scatter** family's `symbol → production
/// wrapper` map — the FULL family (IndexSelect + Gather dtype-fanned, IndexAdd +
/// ScatterAdd per-dtype). Contract: `docs/kernel-contracts/cpu/indexing.fkc.md`.
///
/// - **IndexSelect** / **Gather** are **dtype-agnostic byte copies** (parameter-
///   ized `dtype_size`, one copy per selected/gathered element) with a SINGLE
///   production wrapper each. Their contract sections declare a BASE `entry_point`
///   (`…::index_select_cpu` / `…::gather_cpu`) + an enumerated `dtypes` list, so
///   the importer's §3.4 fan-out resolves `<base>_<dtype>` (`index_select_cpu_f32`,
///   …) — a FABRICATED per-dtype symbol (there is no real `index_select_cpu_f32`
///   fn) that every dtype variant maps to the ONE dtype-agnostic wrapper. The
///   `indices` operand is the FIXED U32 slot and `out: passthrough(source)`, so
///   the fan emits key `[T, U32, T]`. The contract's `dtypes` list was trimmed to
///   production's 9 wired dtypes (F32/F64/BF16/F16/U32/U8/I16/I32/I64) — I8 is
///   describable (byte-agnostic) but NOT wired — so the fan emits BYTE-FOR-BYTE
///   the deleted hand-written `table.register(IndexSelect/Gather, …)` regs.
/// - **IndexAdd** / **ScatterAdd** are **per-dtype** (typed accumulation — f32/f64
///   native, bf16/f16 widen to an f32 accumulator; out seeded from `base` then
///   `+= src`), so each `index_add_<dt>` / `scatter_add_<dt>` section declares a
///   SPECIFIC single-dtype `entry_point` resolved AS-IS (no fan), key
///   `[T, U32, T, T]` (`base`, U32 `indices`, `src`, `passthrough(base)` output),
///   mapping to its OWN typed wrapper — 4 dtypes each (F32/F64/BF16/F16).
pub static CPU_INDEXING_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // IndexSelect — dtype-agnostic byte copy; the fabricated
    // `index_select_cpu_<dt>` symbol maps to the ONE `index_select_cpu_wrapper`.
    ep!("index_select_cpu", "f32",  index_select_cpu_wrapper),
    ep!("index_select_cpu", "f64",  index_select_cpu_wrapper),
    ep!("index_select_cpu", "bf16", index_select_cpu_wrapper),
    ep!("index_select_cpu", "f16",  index_select_cpu_wrapper),
    ep!("index_select_cpu", "u32",  index_select_cpu_wrapper),
    ep!("index_select_cpu", "u8",   index_select_cpu_wrapper),
    ep!("index_select_cpu", "i16",  index_select_cpu_wrapper),
    ep!("index_select_cpu", "i32",  index_select_cpu_wrapper),
    ep!("index_select_cpu", "i64",  index_select_cpu_wrapper),
    // Gather — dtype-agnostic byte copy.
    ep!("gather_cpu", "f32",  gather_cpu_wrapper),
    ep!("gather_cpu", "f64",  gather_cpu_wrapper),
    ep!("gather_cpu", "bf16", gather_cpu_wrapper),
    ep!("gather_cpu", "f16",  gather_cpu_wrapper),
    ep!("gather_cpu", "u32",  gather_cpu_wrapper),
    ep!("gather_cpu", "u8",   gather_cpu_wrapper),
    ep!("gather_cpu", "i16",  gather_cpu_wrapper),
    ep!("gather_cpu", "i32",  gather_cpu_wrapper),
    ep!("gather_cpu", "i64",  gather_cpu_wrapper),
    // IndexAdd — per-dtype typed accumulation; resolved AS-IS (no fan).
    ep!("index_add", "f32",  index_add_f32_cpu_wrapper),
    ep!("index_add", "f64",  index_add_f64_cpu_wrapper),
    ep!("index_add", "bf16", index_add_bf16_cpu_wrapper),
    ep!("index_add", "f16",  index_add_f16_cpu_wrapper),
    // ScatterAdd — per-dtype typed accumulation; resolved AS-IS (no fan).
    ep!("scatter_add", "f32",  scatter_add_f32_cpu_wrapper),
    ep!("scatter_add", "f64",  scatter_add_f64_cpu_wrapper),
    ep!("scatter_add", "bf16", scatter_add_bf16_cpu_wrapper),
    ep!("scatter_add", "f16",  scatter_add_f16_cpu_wrapper),
];

/// The CPU **matmul** family's `symbol → production wrapper` map — the FULL
/// portable family: bare batched `MatMul` (6 dtypes) + fused `FusedLinear`
/// (matmul + bias-add, 4 dtypes) = 10 entry points. Contract:
/// `docs/kernel-contracts/cpu/matmul.fkc.md`.
///
/// Each per-(op, dtype) section (`## matmul_f32`, `## fused_linear_f32`, …)
/// declares a SPECIFIC single-dtype `entry_point` (`…::matmul_f32`), so none of
/// them dtype-fan — the importer resolves that symbol AS-IS. Binding keys:
/// - **MatMul** `[T, T, T]` (lhs, rhs + `passthrough(lhs)` output); the batch
///   geometry (lhs_batch_dims/rhs_batch_dims/m/n/k) rides in `OpParams::Matmul`,
///   NOT the dtype-list. Float variants (F32/F64/BF16/F16) accumulate in
///   f32/native and narrow on store; integer variants (I8/U8) accumulate in i32
///   and SATURATE on store.
/// - **FusedLinear** `[T, T, T, T]` (a, b, bias + `passthrough(a)` output);
///   `bias` is a REQUIRED 1-D `[N]` operand (NOT `optional`, so no optional
///   {absent, present} fan — a single 4-slot key per dtype). Reuses
///   `OpParams::Matmul` for shape; the kernel seeds the accumulator with the
///   bias then accumulates over `k`.
///
/// This map covers ONLY the portable (`kernel_source: "portable-cpu"`) kernels
/// this contract declares. The MKL / AOCL BLAS siblings live in SEPARATE
/// external backend crates (`fuel-mkl-cpu-backend` / `fuel-aocl-cpu-backend`),
/// register through the exported dispatch helpers with their own
/// `"mkl"`/`"aocl"` `kernel_source` tags as ranked ALTERNATIVES at the SAME
/// keys, and are out of scope here (untouched). The quant `QMatMul` /
/// `Nf4Matmul` OpKinds have their own contracts and stay hand-written. These
/// sections have NO `##` chassis umbrella, so there is no `registrable: false`
/// describe-only entry to omit; the SEPARATE `FusedKernelRegistry`
/// `FusedOps::FUSED_LINEAR` seam (`register_default_fused_kernels`) is
/// hand-written (the matmul contract declares no `fused_op`), and stays
/// untouched — this map only serves the `KernelBindingTable` primitive path.
pub static CPU_MATMUL_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Bare batched MatMul — key [T, T, T].
    ep!("matmul", "f32",  matmul_f32_cpu_wrapper),
    ep!("matmul", "f64",  matmul_f64_cpu_wrapper),
    ep!("matmul", "bf16", matmul_bf16_cpu_wrapper),
    ep!("matmul", "f16",  matmul_f16_cpu_wrapper),
    ep!("matmul", "i8",   matmul_i8_cpu_wrapper),
    ep!("matmul", "u8",   matmul_u8_cpu_wrapper),
    // Fused matmul + bias-add — key [T, T, T, T].
    ep!("fused_linear", "f32",  fused_linear_f32_cpu_wrapper),
    ep!("fused_linear", "f64",  fused_linear_f64_cpu_wrapper),
    ep!("fused_linear", "bf16", fused_linear_bf16_cpu_wrapper),
    ep!("fused_linear", "f16",  fused_linear_f16_cpu_wrapper),
];

/// The CPU **attention** family's `symbol → production wrapper` map — the
/// MIGRATED (`KernelBindingTable`) subset: forward `FlashAttn` (4 dtypes) +
/// `FlashAttnBackward{Q,K,V}` (3 selectors × 4 dtypes) = 16 entry points.
/// Contract: `docs/kernel-contracts/cpu/attention.fkc.md`.
///
/// Each per-(op, dtype) section (`## flash_attn_f32`, `## flash_attn_backward_q_f32`,
/// …) declares a SPECIFIC single-dtype `entry_point`, so none of them dtype-fan
/// — the importer resolves that symbol AS-IS. Because the contract marks
/// `alibi_slopes` as `optional: true` (the LAST input), the importer's
/// key-builder fans each section into BOTH the no-alibi key (`[q,k,v,out]` /
/// `[q,k,v,do,out]`) and the with-alibi key (`+alibi`) — both resolving this
/// SAME symbol/wrapper (the CPU wrapper handles the presence/absence of the
/// alibi operand). The softmax/mask geometry rides in `OpParams::FlashAttn`
/// (shared by the forward op AND the three backward selectors — there is no
/// dedicated backward `OpParams` variant), NOT the dtype-list.
///
/// Two symbol shapes:
/// - **Forward FlashAttn** kernels are byte-kernel symbols
///   (`fuel_cpu_backend::byte_kernels::flash_attn_<dt>`, the `ep!` shape).
/// - **FlashAttnBackward{Q,K,V}** wrappers live in the dispatch layer (they pin
///   which gradient the shared byte-kernel writes via a `FaBackwardWhich`
///   selector), so their symbol is the fully-qualified
///   `fuel_dispatch::dispatch::<wrapper>` path (the `ep_dispatch!` shape).
///
/// **PagedAttn is REGISTRABLE** (`registrable: true` in the contract). Its
/// `fdx.gather: paged_blocks` pool is import METADATA (§3.9.1: the block_table /
/// context_lens ride as ordinary U32 operands + `OpParams::PagedAttn`, so
/// registration does not depend on the FDX gather VIEW, which stays
/// [consumer-ahead]), so the importer validates the gather block for coherence
/// and resolves the four paged `byte_kernels::paged_attn_<dt>` symbols here. The
/// contract marks `alibi_slopes` as `optional: true` (last input), so the
/// key-builder fans each paged section into BOTH the no-alibi key
/// `[q,kc,vc,bt:U32,cl:U32,out]` and the with-alibi key `+alibi` — both resolving
/// the SAME wrapper (the CPU `paged_attn_<dt>_cpu_wrapper` handles 5 or 6 inputs).
/// The SEPARATE `FusedKernelRegistry` FLASH_ATTN* / PAGED_ATTN seam
/// (`register_default_fused_kernels`) is a distinct registry and stays untouched;
/// this map only serves the `KernelBindingTable` primitive path.
pub static CPU_ATTENTION_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Forward FlashAttn — byte-kernel symbols; keys [T,T,T,T] / [T,T,T,T,T].
    ep!("flash_attn", "f32",  flash_attn_f32_cpu_wrapper),
    ep!("flash_attn", "f64",  flash_attn_f64_cpu_wrapper),
    ep!("flash_attn", "bf16", flash_attn_bf16_cpu_wrapper),
    ep!("flash_attn", "f16",  flash_attn_f16_cpu_wrapper),
    // FlashAttnBackward{Q,K,V} — dispatch-layer wrapper symbols;
    // keys [T,T,T,T,T] / [T,T,T,T,T,T].
    ep_dispatch!(flash_attn_backward_q_f32_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_q_f64_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_q_bf16_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_q_f16_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_k_f32_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_k_f64_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_k_bf16_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_k_f16_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_v_f32_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_v_f64_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_v_bf16_cpu_wrapper),
    ep_dispatch!(flash_attn_backward_v_f16_cpu_wrapper),
    // PagedAttn — byte-kernel symbols; keys [T,T,T,U32,U32,T] (+[T] with-alibi).
    // The optional-operand fan resolves both keys to the same per-dtype wrapper.
    ep!("paged_attn", "f32",  paged_attn_f32_cpu_wrapper),
    ep!("paged_attn", "f64",  paged_attn_f64_cpu_wrapper),
    ep!("paged_attn", "bf16", paged_attn_bf16_cpu_wrapper),
    ep!("paged_attn", "f16",  paged_attn_f16_cpu_wrapper),
];

/// The CPU **in-place scalar-param** family's `symbol → production wrapper`
/// map — the FULL family: 21 in-place unary ops (`<op>_inplace`, each fanned
/// ×4 dtypes = 84) + `InplaceAffine` / `ClampInplace` / `PowIInplace`
/// (`affine_inplace` / `clamp_inplace` / `powi_inplace`, 4 dtypes each = 12) =
/// 96 entry points. Contract:
/// `docs/kernel-contracts/cpu/inplace-unary-affine.fkc.md`.
///
/// - **In-place unary** (`relu_inplace` … `gelu_erf_inplace`): each per-op
///   section declares a BASE `entry_point` (`…::relu_inplace`) + enumerates
///   `dtypes: [F32,F64,BF16,F16]`, so the importer's §3.4 multi-dtype fan-out
///   resolves `<base>_<dtype>` (`relu_inplace_f32`) against this table — the
///   `$op` literals below are the byte-kernel BASES (`<op>_inplace`), NOT the
///   OpKind names. The two GELU flavors stay distinct: `gelu_inplace`
///   (`OpKind::GeluInplace`, the canonical tanh GELU) vs `gelu_erf_inplace`
///   (`OpKind::GeluErfInplace`).
/// - **InplaceAffine / ClampInplace / PowIInplace** are per-dtype SINGLE
///   sections (one enumerated dtype each), so they do NOT fan — the importer
///   resolves their specific `<op>_inplace_<dt>` symbol AS-IS. The affine rows
///   carry the THREE-WAY naming skew: the byte-kernel/entry_point suffix is
///   `affine_inplace_<dt>` while the production wrapper fn is
///   `inplace_affine_<dt>_cpu_wrapper` (words swapped) — the `ep!` symbol is
///   built from the `$op`/`$dt` literals (`affine_inplace`), so the mapping
///   still binds the correct contract symbol to the swapped-name wrapper
///   (mirrors the clamp/powi/log_softmax fn-vs-symbol cases). Clamp/powi
///   wrapper fn-names match their symbols directly.
///
/// Every binding keys `[T, T]` (the single `out` operand + its
/// `passthrough(out)` mirror; the executor's `WorkItemKind::InplaceKernel` arm
/// passes the target as `outputs[0]`, so the wrapper takes 0 inputs + 1 output,
/// but the binding-table KEY is `[T, T]`). Scalar params (affine `mul`/`add`,
/// clamp `min`/`max`, powi `exp`) ride in `OpParams::{Affine, Clamp, PowI}`, NOT
/// the dtype-list. The `## unary_inplace` chassis umbrella is `registrable: false`
/// (§3.10 describe-only) and never resolves, so it is absent here.
pub static CPU_INPLACE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // In-place unary — 21 ops × 4 dtypes, base `<op>_inplace` fanned per dtype.
    ep!("relu_inplace", "f32",  relu_inplace_f32_cpu_wrapper),
    ep!("relu_inplace", "f64",  relu_inplace_f64_cpu_wrapper),
    ep!("relu_inplace", "bf16", relu_inplace_bf16_cpu_wrapper),
    ep!("relu_inplace", "f16",  relu_inplace_f16_cpu_wrapper),
    ep!("silu_inplace", "f32",  silu_inplace_f32_cpu_wrapper),
    ep!("silu_inplace", "f64",  silu_inplace_f64_cpu_wrapper),
    ep!("silu_inplace", "bf16", silu_inplace_bf16_cpu_wrapper),
    ep!("silu_inplace", "f16",  silu_inplace_f16_cpu_wrapper),
    ep!("gelu_inplace", "f32",  gelu_inplace_f32_cpu_wrapper),
    ep!("gelu_inplace", "f64",  gelu_inplace_f64_cpu_wrapper),
    ep!("gelu_inplace", "bf16", gelu_inplace_bf16_cpu_wrapper),
    ep!("gelu_inplace", "f16",  gelu_inplace_f16_cpu_wrapper),
    ep!("tanh_inplace", "f32",  tanh_inplace_f32_cpu_wrapper),
    ep!("tanh_inplace", "f64",  tanh_inplace_f64_cpu_wrapper),
    ep!("tanh_inplace", "bf16", tanh_inplace_bf16_cpu_wrapper),
    ep!("tanh_inplace", "f16",  tanh_inplace_f16_cpu_wrapper),
    ep!("sigmoid_inplace", "f32",  sigmoid_inplace_f32_cpu_wrapper),
    ep!("sigmoid_inplace", "f64",  sigmoid_inplace_f64_cpu_wrapper),
    ep!("sigmoid_inplace", "bf16", sigmoid_inplace_bf16_cpu_wrapper),
    ep!("sigmoid_inplace", "f16",  sigmoid_inplace_f16_cpu_wrapper),
    ep!("neg_inplace", "f32",  neg_inplace_f32_cpu_wrapper),
    ep!("neg_inplace", "f64",  neg_inplace_f64_cpu_wrapper),
    ep!("neg_inplace", "bf16", neg_inplace_bf16_cpu_wrapper),
    ep!("neg_inplace", "f16",  neg_inplace_f16_cpu_wrapper),
    ep!("abs_inplace", "f32",  abs_inplace_f32_cpu_wrapper),
    ep!("abs_inplace", "f64",  abs_inplace_f64_cpu_wrapper),
    ep!("abs_inplace", "bf16", abs_inplace_bf16_cpu_wrapper),
    ep!("abs_inplace", "f16",  abs_inplace_f16_cpu_wrapper),
    ep!("sqr_inplace", "f32",  sqr_inplace_f32_cpu_wrapper),
    ep!("sqr_inplace", "f64",  sqr_inplace_f64_cpu_wrapper),
    ep!("sqr_inplace", "bf16", sqr_inplace_bf16_cpu_wrapper),
    ep!("sqr_inplace", "f16",  sqr_inplace_f16_cpu_wrapper),
    ep!("sqrt_inplace", "f32",  sqrt_inplace_f32_cpu_wrapper),
    ep!("sqrt_inplace", "f64",  sqrt_inplace_f64_cpu_wrapper),
    ep!("sqrt_inplace", "bf16", sqrt_inplace_bf16_cpu_wrapper),
    ep!("sqrt_inplace", "f16",  sqrt_inplace_f16_cpu_wrapper),
    ep!("rsqrt_inplace", "f32",  rsqrt_inplace_f32_cpu_wrapper),
    ep!("rsqrt_inplace", "f64",  rsqrt_inplace_f64_cpu_wrapper),
    ep!("rsqrt_inplace", "bf16", rsqrt_inplace_bf16_cpu_wrapper),
    ep!("rsqrt_inplace", "f16",  rsqrt_inplace_f16_cpu_wrapper),
    ep!("recip_inplace", "f32",  recip_inplace_f32_cpu_wrapper),
    ep!("recip_inplace", "f64",  recip_inplace_f64_cpu_wrapper),
    ep!("recip_inplace", "bf16", recip_inplace_bf16_cpu_wrapper),
    ep!("recip_inplace", "f16",  recip_inplace_f16_cpu_wrapper),
    ep!("exp_inplace", "f32",  exp_inplace_f32_cpu_wrapper),
    ep!("exp_inplace", "f64",  exp_inplace_f64_cpu_wrapper),
    ep!("exp_inplace", "bf16", exp_inplace_bf16_cpu_wrapper),
    ep!("exp_inplace", "f16",  exp_inplace_f16_cpu_wrapper),
    ep!("log_inplace", "f32",  log_inplace_f32_cpu_wrapper),
    ep!("log_inplace", "f64",  log_inplace_f64_cpu_wrapper),
    ep!("log_inplace", "bf16", log_inplace_bf16_cpu_wrapper),
    ep!("log_inplace", "f16",  log_inplace_f16_cpu_wrapper),
    ep!("sin_inplace", "f32",  sin_inplace_f32_cpu_wrapper),
    ep!("sin_inplace", "f64",  sin_inplace_f64_cpu_wrapper),
    ep!("sin_inplace", "bf16", sin_inplace_bf16_cpu_wrapper),
    ep!("sin_inplace", "f16",  sin_inplace_f16_cpu_wrapper),
    ep!("cos_inplace", "f32",  cos_inplace_f32_cpu_wrapper),
    ep!("cos_inplace", "f64",  cos_inplace_f64_cpu_wrapper),
    ep!("cos_inplace", "bf16", cos_inplace_bf16_cpu_wrapper),
    ep!("cos_inplace", "f16",  cos_inplace_f16_cpu_wrapper),
    ep!("sign_inplace", "f32",  sign_inplace_f32_cpu_wrapper),
    ep!("sign_inplace", "f64",  sign_inplace_f64_cpu_wrapper),
    ep!("sign_inplace", "bf16", sign_inplace_bf16_cpu_wrapper),
    ep!("sign_inplace", "f16",  sign_inplace_f16_cpu_wrapper),
    ep!("floor_inplace", "f32",  floor_inplace_f32_cpu_wrapper),
    ep!("floor_inplace", "f64",  floor_inplace_f64_cpu_wrapper),
    ep!("floor_inplace", "bf16", floor_inplace_bf16_cpu_wrapper),
    ep!("floor_inplace", "f16",  floor_inplace_f16_cpu_wrapper),
    ep!("ceil_inplace", "f32",  ceil_inplace_f32_cpu_wrapper),
    ep!("ceil_inplace", "f64",  ceil_inplace_f64_cpu_wrapper),
    ep!("ceil_inplace", "bf16", ceil_inplace_bf16_cpu_wrapper),
    ep!("ceil_inplace", "f16",  ceil_inplace_f16_cpu_wrapper),
    ep!("round_inplace", "f32",  round_inplace_f32_cpu_wrapper),
    ep!("round_inplace", "f64",  round_inplace_f64_cpu_wrapper),
    ep!("round_inplace", "bf16", round_inplace_bf16_cpu_wrapper),
    ep!("round_inplace", "f16",  round_inplace_f16_cpu_wrapper),
    ep!("erf_inplace", "f32",  erf_inplace_f32_cpu_wrapper),
    ep!("erf_inplace", "f64",  erf_inplace_f64_cpu_wrapper),
    ep!("erf_inplace", "bf16", erf_inplace_bf16_cpu_wrapper),
    ep!("erf_inplace", "f16",  erf_inplace_f16_cpu_wrapper),
    ep!("gelu_erf_inplace", "f32",  gelu_erf_inplace_f32_cpu_wrapper),
    ep!("gelu_erf_inplace", "f64",  gelu_erf_inplace_f64_cpu_wrapper),
    ep!("gelu_erf_inplace", "bf16", gelu_erf_inplace_bf16_cpu_wrapper),
    ep!("gelu_erf_inplace", "f16",  gelu_erf_inplace_f16_cpu_wrapper),
    // InplaceAffine — symbol `affine_inplace_<dt>`, wrapper `inplace_affine_<dt>_cpu_wrapper`
    // (words swapped); resolved AS-IS (single-dtype sections).
    ep!("affine_inplace", "f32",  inplace_affine_f32_cpu_wrapper),
    ep!("affine_inplace", "f64",  inplace_affine_f64_cpu_wrapper),
    ep!("affine_inplace", "bf16", inplace_affine_bf16_cpu_wrapper),
    ep!("affine_inplace", "f16",  inplace_affine_f16_cpu_wrapper),
    // ClampInplace — symbol == wrapper base; resolved AS-IS.
    ep!("clamp_inplace", "f32",  clamp_inplace_f32_cpu_wrapper),
    ep!("clamp_inplace", "f64",  clamp_inplace_f64_cpu_wrapper),
    ep!("clamp_inplace", "bf16", clamp_inplace_bf16_cpu_wrapper),
    ep!("clamp_inplace", "f16",  clamp_inplace_f16_cpu_wrapper),
    // PowIInplace — symbol == wrapper base; resolved AS-IS.
    ep!("powi_inplace", "f32",  powi_inplace_f32_cpu_wrapper),
    ep!("powi_inplace", "f64",  powi_inplace_f64_cpu_wrapper),
    ep!("powi_inplace", "bf16", powi_inplace_bf16_cpu_wrapper),
    ep!("powi_inplace", "f16",  powi_inplace_f16_cpu_wrapper),
];

/// The CPU **cast** family's `symbol → production wrapper` map — the FULL
/// directed-pair matrix: every ordered pair of the 11 real numeric dtypes
/// {F32,F64,F16,BF16,F8E4M3,U8,I8,U32,I16,I32,I64}, identity excluded =
/// 11 × 10 = 110 entry points. Contract: `docs/kernel-contracts/cpu/cast.fkc.md`.
///
/// Each per-pair section (`## cast_f64_to_f32`, …) declares a SPECIFIC
/// single-dtype `src` input + a `fixed(DST)` output, so it does NOT dtype-fan —
/// the importer keys `[SRC, DST]` (byte-for-byte the deleted hand-written
/// `table.register(Cast, &[SRC, DST], …)` regs) and resolves the section's
/// SPECIFIC `cast_<src>_to_<dst>` byte-kernel `entry_point` AS-IS. Because the
/// binding lookup is keyed on the **target** dtype (the Node's dtype), all 10 of
/// a target's source pairs bind the SAME per-target `cast_to_<dst>_cpu_wrapper`
/// (which `match`es on the source dtype internally to pick the right byte
/// kernel) — the synthetic-umbrella precedent (10 distinct real byte-kernel
/// entry_points → 1 wrapper, like `pad_cpu`'s fabricated per-dtype symbols → the
/// one `pad_cpu_wrapper`, except here every entry_point IS a real distinct byte
/// kernel). This map is the SOLE registration path for the whole family: every
/// hand-written `table.register(Cast, …)` reg is DELETED. Identity pairs
/// (`[T, T]`) are never registered — the optimizer elides identity `Cast` before
/// dispatch. These sections have NO `##` chassis umbrella, so there is no
/// `registrable: false` describe-only entry to omit.
pub static CPU_CAST_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // -> F32
    ep!("cast", "f64_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "f16_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "bf16_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "f8e4m3_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "u8_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "i8_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "u32_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "i16_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "i32_to_f32", cast_to_f32_cpu_wrapper),
    ep!("cast", "i64_to_f32", cast_to_f32_cpu_wrapper),
    // -> F64
    ep!("cast", "f32_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "f16_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "bf16_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "f8e4m3_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "u8_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "i8_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "u32_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "i16_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "i32_to_f64", cast_to_f64_cpu_wrapper),
    ep!("cast", "i64_to_f64", cast_to_f64_cpu_wrapper),
    // -> F16
    ep!("cast", "f32_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "f64_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "bf16_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "f8e4m3_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "u8_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "i8_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "u32_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "i16_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "i32_to_f16", cast_to_f16_cpu_wrapper),
    ep!("cast", "i64_to_f16", cast_to_f16_cpu_wrapper),
    // -> BF16
    ep!("cast", "f32_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "f64_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "f16_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "f8e4m3_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "u8_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "i8_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "u32_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "i16_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "i32_to_bf16", cast_to_bf16_cpu_wrapper),
    ep!("cast", "i64_to_bf16", cast_to_bf16_cpu_wrapper),
    // -> F8E4M3
    ep!("cast", "f32_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "f64_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "f16_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "bf16_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "u8_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "i8_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "u32_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "i16_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "i32_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    ep!("cast", "i64_to_f8e4m3", cast_to_f8e4m3_cpu_wrapper),
    // -> U8
    ep!("cast", "f32_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "f64_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "f16_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "bf16_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "f8e4m3_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "i8_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "u32_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "i16_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "i32_to_u8", cast_to_u8_cpu_wrapper),
    ep!("cast", "i64_to_u8", cast_to_u8_cpu_wrapper),
    // -> I8
    ep!("cast", "f32_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "f64_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "f16_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "bf16_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "f8e4m3_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "u8_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "u32_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "i16_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "i32_to_i8", cast_to_i8_cpu_wrapper),
    ep!("cast", "i64_to_i8", cast_to_i8_cpu_wrapper),
    // -> U32
    ep!("cast", "f32_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "f64_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "f16_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "bf16_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "f8e4m3_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "u8_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "i8_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "i16_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "i32_to_u32", cast_to_u32_cpu_wrapper),
    ep!("cast", "i64_to_u32", cast_to_u32_cpu_wrapper),
    // -> I16
    ep!("cast", "f32_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "f64_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "f16_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "bf16_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "f8e4m3_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "u8_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "i8_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "u32_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "i32_to_i16", cast_to_i16_cpu_wrapper),
    ep!("cast", "i64_to_i16", cast_to_i16_cpu_wrapper),
    // -> I32
    ep!("cast", "f32_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "f64_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "f16_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "bf16_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "f8e4m3_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "u8_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "i8_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "u32_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "i16_to_i32", cast_to_i32_cpu_wrapper),
    ep!("cast", "i64_to_i32", cast_to_i32_cpu_wrapper),
    // -> I64
    ep!("cast", "f32_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "f64_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "f16_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "bf16_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "f8e4m3_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "u8_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "i8_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "u32_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "i16_to_i64", cast_to_i64_cpu_wrapper),
    ep!("cast", "i32_to_i64", cast_to_i64_cpu_wrapper),
];

/// The CPU norm/softmax **FUSED** family's `symbol → production wrapper` map —
/// the FIRST fused-registry link table (the `fused_op` analogue of the
/// primitive `CPU_*_ENTRY_POINTS`). Contract:
/// `docs/kernel-contracts/fused/norm-softmax.fkc.md`.
///
/// Eight `fused_op` sections — three forward (`SOFTMAX_LAST_DIM` /
/// `RMS_NORM_LAST_DIM` / `LAYER_NORM_LAST_DIM`, key `[T, T]`) plus five backward
/// helpers (`SOFTMAX_LAST_DIM_BACKWARD` / `LAYER_NORM_LAST_DIM_BACKWARD` /
/// `RMS_NORM_LAST_DIM_BACKWARD` / `REDUCE_MAX_TO_BACKWARD` / `POWI_BACKWARD`,
/// key `[T, T, T]`). Each declares a dtype-agnostic BASE `entry_point`
/// `…::<op>_cpu` **plus a per-input `dtypes: [F32, F64, BF16, F16]` list**, so
/// [`crate::fkc::lower::lower_fused`] **dtype-fans** it (§3.4, see [`fep!`]) into
/// four per-dtype impls: per fanned `dt` it resolves `<base>_<dt>`
/// (`…::<op>_cpu_f32`, `…_f64`, `…_bf16`, `…_f16`) against this table (8 ops ×
/// 4 dtypes = 32 rows). Each row binds the `<op>_<dt>_cpu_wrapper` — the exact
/// kernel fn the hand-written `register_default_fused_kernels` seam registers
/// for that op's `dt` dtype tuple — so an imported impl binds the SAME
/// executable kernel per dtype (a 1:1 replacement for the hand-written per-dtype
/// registrations, not just the F32 representative).
///
/// This map serves the SEPARATE [`crate::fused::FusedKernelRegistry`] seam (the
/// join target `register_default_fused_kernels` populates), NOT the primitive
/// `KernelBindingTable`. It is the live resolution behind the fused import seam:
/// an authored `fused_op` bundle imported through [`CpuLinkRegistry`] binds the
/// real CPU fused kernels (FKC P9 — no raw pointers in the serialized contract).
pub static CPU_FUSED_NORM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // Forward (key [T, T]) — SoftmaxLastDim / RmsNormLastDim / LayerNormLastDim.
    fep!("softmax_last_dim_cpu_f32",  softmax_last_dim_f32_cpu_wrapper),
    fep!("softmax_last_dim_cpu_f64",  softmax_last_dim_f64_cpu_wrapper),
    fep!("softmax_last_dim_cpu_bf16", softmax_last_dim_bf16_cpu_wrapper),
    fep!("softmax_last_dim_cpu_f16",  softmax_last_dim_f16_cpu_wrapper),
    fep!("rms_norm_last_dim_cpu_f32",  rms_norm_last_dim_f32_cpu_wrapper),
    fep!("rms_norm_last_dim_cpu_f64",  rms_norm_last_dim_f64_cpu_wrapper),
    fep!("rms_norm_last_dim_cpu_bf16", rms_norm_last_dim_bf16_cpu_wrapper),
    fep!("rms_norm_last_dim_cpu_f16",  rms_norm_last_dim_f16_cpu_wrapper),
    fep!("layer_norm_last_dim_cpu_f32",  layer_norm_last_dim_f32_cpu_wrapper),
    fep!("layer_norm_last_dim_cpu_f64",  layer_norm_last_dim_f64_cpu_wrapper),
    fep!("layer_norm_last_dim_cpu_bf16", layer_norm_last_dim_bf16_cpu_wrapper),
    fep!("layer_norm_last_dim_cpu_f16",  layer_norm_last_dim_f16_cpu_wrapper),
    // Backward (key [T, T, T]) — Softmax / LayerNorm / RmsNorm backward.
    fep!("softmax_last_dim_backward_cpu_f32",  softmax_last_dim_backward_f32_cpu_wrapper),
    fep!("softmax_last_dim_backward_cpu_f64",  softmax_last_dim_backward_f64_cpu_wrapper),
    fep!("softmax_last_dim_backward_cpu_bf16", softmax_last_dim_backward_bf16_cpu_wrapper),
    fep!("softmax_last_dim_backward_cpu_f16",  softmax_last_dim_backward_f16_cpu_wrapper),
    fep!("layer_norm_last_dim_backward_cpu_f32",  layer_norm_last_dim_backward_f32_cpu_wrapper),
    fep!("layer_norm_last_dim_backward_cpu_f64",  layer_norm_last_dim_backward_f64_cpu_wrapper),
    fep!("layer_norm_last_dim_backward_cpu_bf16", layer_norm_last_dim_backward_bf16_cpu_wrapper),
    fep!("layer_norm_last_dim_backward_cpu_f16",  layer_norm_last_dim_backward_f16_cpu_wrapper),
    fep!("rms_norm_last_dim_backward_cpu_f32",  rms_norm_last_dim_backward_f32_cpu_wrapper),
    fep!("rms_norm_last_dim_backward_cpu_f64",  rms_norm_last_dim_backward_f64_cpu_wrapper),
    fep!("rms_norm_last_dim_backward_cpu_bf16", rms_norm_last_dim_backward_bf16_cpu_wrapper),
    fep!("rms_norm_last_dim_backward_cpu_f16",  rms_norm_last_dim_backward_f16_cpu_wrapper),
    // Backward of primitives (key [T, T, T]) — ReduceMaxTo / PowI backward.
    fep!("reduce_max_to_backward_cpu_f32",  reduce_max_to_backward_f32_cpu_wrapper),
    fep!("reduce_max_to_backward_cpu_f64",  reduce_max_to_backward_f64_cpu_wrapper),
    fep!("reduce_max_to_backward_cpu_bf16", reduce_max_to_backward_bf16_cpu_wrapper),
    fep!("reduce_max_to_backward_cpu_f16",  reduce_max_to_backward_f16_cpu_wrapper),
    fep!("powi_backward_cpu_f32",  powi_backward_f32_cpu_wrapper),
    fep!("powi_backward_cpu_f64",  powi_backward_f64_cpu_wrapper),
    fep!("powi_backward_cpu_bf16", powi_backward_bf16_cpu_wrapper),
    fep!("powi_backward_cpu_f16",  powi_backward_f16_cpu_wrapper),
];

/// The CPU **linear / quantized-matmul** FUSED family's `symbol → production
/// wrapper` map — the MIGRATED subset of the `audited: true` linear-quant
/// bundle. Contract: `docs/kernel-contracts/fused/linear-quant.fkc.md`.
///
/// Four of the bundle's five `fused_op` sections migrate here (the fifth,
/// `nf4_matmul`, is `registrable: false` — its `fdx.quant.family: AFFINE_BLOCK`
/// is consumer-ahead, §6 — so it never lowers/resolves and its hand-written
/// `FusedOps::NF4_MATMUL` regs stay authoritative; it is absent here):
///
/// - **FUSED_LINEAR** — a multi-dtype section (`a`/`b`/`bias`
///   `dtypes: [F32, F64, BF16, F16]`) on a dtype-agnostic BASE `entry_point`
///   `…::fused_linear_cpu`, so [`crate::fkc::lower::lower_fused`] **dtype-fans**
///   it (§3.4, see [`fep!`]) into 4 per-dtype impls resolving
///   `…::fused_linear_cpu_<dt>`, key `[T, T, T, T]` (a, b, bias +
///   `passthrough(a)` out).
/// - **QMATMUL** — a NON-fanning section (all operands single-dtype: `a` F32,
///   `w_q_bytes` U32 (the LOGICAL dispatch dtype — the physical GGML block
///   byte-honesty rides `fdx.quant: GGML_BLOCK`, §storage logical/physical
///   split), output F32), so its BASE `entry_point` `…::qmatmul_cpu` resolves
///   AS-IS (no `_<dt>` suffix), key `[F32, U32, F32]` byte-for-byte the deleted
///   hand-written `QM_F32` reg.
/// - **INPLACE_AFFINE** — a multi-dtype section (`x`
///   `dtypes: [F32, F64, BF16, F16]`) fanned into 4 per-dtype impls resolving
///   `…::inplace_affine_cpu_<dt>`, key `[T, T]` (out aliases x). NOTE the
///   fn-vs-symbol skew: the fanned symbol is `inplace_affine_cpu_<dt>` (base
///   `…_cpu` + `_<dt>`) while the wrapper fn is `inplace_affine_<dt>_cpu_wrapper`
///   (`_cpu` after `_<dt>`) — spelled out per row.
/// - **FUSED_SOFTMAX_CROSS_ENTROPY** — a multi-dtype section (`logits`
///   `dtypes: [F32, F64, BF16, F16]`; `targets` fixed I64; output `fixed(F32)`)
///   fanned into 4 per-dtype impls resolving
///   `…::fused_softmax_cross_entropy_cpu_<dt>`, key `[T, I64, F32]` byte-for-byte
///   the deleted hand-written `FSCE_<dt>` regs (the fan varies ONLY `logits`;
///   `targets` I64 + `fixed(F32)` output are constant across variants).
///
/// Each fanned row binds the `<op>_<dt>_cpu_wrapper` — the exact kernel fn the
/// deleted hand-written `register_default_fused_kernels` seam registered for
/// that op's `dt` dtype tuple — so an imported impl binds the SAME executable
/// kernel per dtype (a 1:1 replacement, real revision hash vs the hand-written
/// UNTRACKED sentinel). Serves the [`crate::fused::FusedKernelRegistry`] seam
/// (the join target `register_default_fused_kernels` populates), NOT the
/// primitive `KernelBindingTable`.
pub static CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // FUSED_LINEAR — key [T, T, T, T]; multi-dtype fan over {F32, F64, BF16, F16}.
    fep!("fused_linear_cpu_f32",  fused_linear_f32_cpu_wrapper),
    fep!("fused_linear_cpu_f64",  fused_linear_f64_cpu_wrapper),
    fep!("fused_linear_cpu_bf16", fused_linear_bf16_cpu_wrapper),
    fep!("fused_linear_cpu_f16",  fused_linear_f16_cpu_wrapper),
    // QMATMUL — key [F32, U32, F32]; NON-fanning, BASE symbol resolved AS-IS.
    fep!("qmatmul_cpu", qmatmul_f32_cpu_wrapper),
    // INPLACE_AFFINE — key [T, T]; multi-dtype fan. Symbol `inplace_affine_cpu_<dt>`
    // → wrapper `inplace_affine_<dt>_cpu_wrapper` (the `_cpu`/`_<dt>` skew).
    fep!("inplace_affine_cpu_f32",  inplace_affine_f32_cpu_wrapper),
    fep!("inplace_affine_cpu_f64",  inplace_affine_f64_cpu_wrapper),
    fep!("inplace_affine_cpu_bf16", inplace_affine_bf16_cpu_wrapper),
    fep!("inplace_affine_cpu_f16",  inplace_affine_f16_cpu_wrapper),
    // FUSED_SOFTMAX_CROSS_ENTROPY — key [T, I64, F32]; fan varies ONLY logits.
    fep!("fused_softmax_cross_entropy_cpu_f32",  fused_softmax_cross_entropy_f32_cpu_wrapper),
    fep!("fused_softmax_cross_entropy_cpu_f64",  fused_softmax_cross_entropy_f64_cpu_wrapper),
    fep!("fused_softmax_cross_entropy_cpu_bf16", fused_softmax_cross_entropy_bf16_cpu_wrapper),
    fep!("fused_softmax_cross_entropy_cpu_f16",  fused_softmax_cross_entropy_f16_cpu_wrapper),
];

/// The CPU **conv / RoPE / SSM** FUSED family's `symbol → production wrapper`
/// map — the FULL `audited: true` conv-rope bundle. Contract:
/// `docs/kernel-contracts/fused/conv-rope.fkc.md`.
///
/// All SIX `fused_op` sections migrate here; each declares a dtype-agnostic
/// BASE `entry_point` `…::<op>_cpu` **plus a per-input `dtypes: [F32, F64, BF16,
/// F16]` list**, so [`crate::fkc::lower::lower_fused`] **dtype-fans** it (§3.4,
/// see [`fep!`]) into four per-dtype impls resolving `<base>_<dt>` — one row per
/// (op, dtype) below (6 ops × 4 dtypes = 24 rows). The naming skew is the norm
/// family's: the FANNED symbol is `<op>_cpu_<dt>` (base `…_cpu` + `_<dt>`) while
/// the wrapper fn is `<op>_<dt>_cpu_wrapper`.
///
/// The 24 rows fan into **32 registered impls** because two sections
/// multiply their key count without adding a symbol row:
/// - **CONV2D** / **CONV_TRANSPOSE2D** mark `bias` `optional: true`, so the
///   importer's key-builder fans EACH per-dtype section into BOTH the no-bias
///   key `[T, T, T]` and the with-bias key `[T, T, T, T]` — both resolving the
///   SAME fanned symbol/wrapper (`conv2d_cpu_<dt>` → `conv2d_<dt>_cpu_wrapper`;
///   the CPU wrapper handles 2 or 3 inputs, the optional operand riding op-params
///   not a distinct symbol). So conv2d = 4 rows → 8 impls, conv_transpose2d =
///   4 rows → 8 impls (the CPU transposed-conv scatter kernel seeds the output
///   with `bias[co]` or `0`).
/// - **SELECTIVE_SCAN** / **SSD_CHUNK_SCAN** declare a `return.bundle`
///   multi-output (Option C, one packed `[y ; last_state]` buffer); the key-
///   builder appends the bundle's PRIMARY-slot dtype (`passthrough(u)` /
///   `passthrough(x)` → T) to the 5-input tail, so each keys `[T; 6]`
///   byte-for-byte the deleted hand-written reg. 4 rows → 4 impls each.
///
/// Net: ROPE 4 + CONV2D 8 + CONV_TRANSPOSE2D 8 + CAUSAL_CONV1D 4 +
/// SELECTIVE_SCAN 4 + SSD_CHUNK_SCAN 4 = 32 impls. Each row binds the exact
/// per-dtype kernel fn the deleted hand-written `register_default_fused_kernels`
/// seam registered (a 1:1 replacement, real revision hash vs the hand-written
/// UNTRACKED sentinel). Serves the [`crate::fused::FusedKernelRegistry`] seam
/// (the join target `register_default_fused_kernels` populates), NOT the
/// primitive `KernelBindingTable` (the `cpu/conv.fkc.md` / `cpu/ssm.fkc.md` /
/// `cpu/rope.fkc.md` primitive tables are separate and stay untouched).
pub static CPU_FUSED_CONV_ROPE_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    // ROPE — key [T, T, T, T] (x, cos, sin + passthrough(x) out); 4 dtypes.
    fep!("rope_cpu_f32",  rope_f32_cpu_wrapper),
    fep!("rope_cpu_f64",  rope_f64_cpu_wrapper),
    fep!("rope_cpu_bf16", rope_bf16_cpu_wrapper),
    fep!("rope_cpu_f16",  rope_f16_cpu_wrapper),
    // CONV2D — optional bias fans no-bias [T,T,T] + with-bias [T,T,T,T]; 4 rows → 8 impls.
    fep!("conv2d_cpu_f32",  conv2d_f32_cpu_wrapper),
    fep!("conv2d_cpu_f64",  conv2d_f64_cpu_wrapper),
    fep!("conv2d_cpu_bf16", conv2d_bf16_cpu_wrapper),
    fep!("conv2d_cpu_f16",  conv2d_f16_cpu_wrapper),
    // CONV_TRANSPOSE2D — optional bias fans no-bias + with-bias; 4 rows → 8 impls.
    fep!("conv_transpose2d_cpu_f32",  conv_transpose2d_f32_cpu_wrapper),
    fep!("conv_transpose2d_cpu_f64",  conv_transpose2d_f64_cpu_wrapper),
    fep!("conv_transpose2d_cpu_bf16", conv_transpose2d_bf16_cpu_wrapper),
    fep!("conv_transpose2d_cpu_f16",  conv_transpose2d_f16_cpu_wrapper),
    // CAUSAL_CONV1D — key [T, T, T, T] (x, weight, bias + passthrough(x) out); 4 dtypes.
    fep!("causal_conv1d_cpu_f32",  causal_conv1d_f32_cpu_wrapper),
    fep!("causal_conv1d_cpu_f64",  causal_conv1d_f64_cpu_wrapper),
    fep!("causal_conv1d_cpu_bf16", causal_conv1d_bf16_cpu_wrapper),
    fep!("causal_conv1d_cpu_f16",  causal_conv1d_f16_cpu_wrapper),
    // SELECTIVE_SCAN — return.bundle: 5 inputs + primary-slot dtype → key [T; 6]; 4 dtypes.
    fep!("selective_scan_cpu_f32",  selective_scan_f32_cpu_wrapper),
    fep!("selective_scan_cpu_f64",  selective_scan_f64_cpu_wrapper),
    fep!("selective_scan_cpu_bf16", selective_scan_bf16_cpu_wrapper),
    fep!("selective_scan_cpu_f16",  selective_scan_f16_cpu_wrapper),
    // SSD_CHUNK_SCAN — return.bundle: 5 inputs + primary-slot dtype → key [T; 6]; 4 dtypes.
    fep!("ssd_chunk_scan_cpu_f32",  ssd_chunk_scan_f32_cpu_wrapper),
    fep!("ssd_chunk_scan_cpu_f64",  ssd_chunk_scan_f64_cpu_wrapper),
    fep!("ssd_chunk_scan_cpu_bf16", ssd_chunk_scan_bf16_cpu_wrapper),
    fep!("ssd_chunk_scan_cpu_f16",  ssd_chunk_scan_f16_cpu_wrapper),
];

/// The built-in CPU backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`CPU_BINARY_ENTRY_POINTS`],
/// [`CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS`], [`CPU_UNARY_ENTRY_POINTS`],
/// [`CPU_COMPARE_ENTRY_POINTS`], [`CPU_WHERE_ENTRY_POINTS`],
/// [`CPU_REDUCE_ENTRY_POINTS`], [`CPU_REDUCE_TO_ENTRY_POINTS`],
/// [`CPU_NORM_ENTRY_POINTS`], [`CPU_NORM_BACKWARD_ENTRY_POINTS`],
/// [`CPU_ROPE_ENTRY_POINTS`], [`CPU_SSM_ENTRY_POINTS`],
/// [`CPU_CONV_ENTRY_POINTS`], [`CPU_PADDING_ENTRY_POINTS`],
/// [`CPU_SHAPE_OPS_ENTRY_POINTS`], [`CPU_INDEXING_ENTRY_POINTS`],
/// [`CPU_MATMUL_ENTRY_POINTS`],
/// [`CPU_ATTENTION_ENTRY_POINTS`], [`CPU_INPLACE_ENTRY_POINTS`], and
/// [`CPU_CAST_ENTRY_POINTS`].
/// Unresolved → `None`, which the importer turns into a typed
/// `UnknownEntryPoint` error (never a panic, never a fabricated pointer).
pub struct CpuLinkRegistry;

impl LinkRegistry for CpuLinkRegistry {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        CPU_BINARY_ENTRY_POINTS
            .iter()
            .chain(CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS.iter())
            .chain(CPU_UNARY_ENTRY_POINTS.iter())
            .chain(CPU_COMPARE_ENTRY_POINTS.iter())
            .chain(CPU_WHERE_ENTRY_POINTS.iter())
            .chain(CPU_REDUCE_ENTRY_POINTS.iter())
            .chain(CPU_REDUCE_TO_ENTRY_POINTS.iter())
            .chain(CPU_NORM_ENTRY_POINTS.iter())
            .chain(CPU_NORM_BACKWARD_ENTRY_POINTS.iter())
            .chain(CPU_ROPE_ENTRY_POINTS.iter())
            .chain(CPU_SSM_ENTRY_POINTS.iter())
            .chain(CPU_CONV_ENTRY_POINTS.iter())
            .chain(CPU_PADDING_ENTRY_POINTS.iter())
            .chain(CPU_SHAPE_OPS_ENTRY_POINTS.iter())
            .chain(CPU_INDEXING_ENTRY_POINTS.iter())
            .chain(CPU_MATMUL_ENTRY_POINTS.iter())
            .chain(CPU_ATTENTION_ENTRY_POINTS.iter())
            .chain(CPU_INPLACE_ENTRY_POINTS.iter())
            .chain(CPU_CAST_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, symbol: &str) -> Option<KernelRef> {
        // FUSED (`fused_op`) resolution — the live fused import seam. Chains the
        // per-family fused entry-point tables (the norm/softmax family,
        // `CPU_FUSED_NORM_ENTRY_POINTS`; the linear/quant-matmul family,
        // `CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS`; and the conv/RoPE/SSM family,
        // `CPU_FUSED_CONV_ROPE_ENTRY_POINTS`). Unresolved → `None`, which the
        // importer turns into a typed `UnknownEntryPoint` (never a panic, never a
        // fabricated pointer — FKC P9).
        //
        // All three checked-in fused bundles are now `audited: true` and WIRED,
        // PRODUCTION-migrated in `register_default_fused_kernels`:
        //   - `fused/norm-softmax.fkc.md` — 8 sections → 32 impls (SOFTMAX /
        //     RMS_NORM / LAYER_NORM_LAST_DIM (+backward), REDUCE_MAX_TO_BACKWARD,
        //     POWI_BACKWARD) via `CPU_FUSED_NORM_ENTRY_POINTS`.
        //   - `fused/linear-quant.fkc.md` — FUSED_LINEAR / QMATMUL /
        //     INPLACE_AFFINE / FUSED_SOFTMAX_CROSS_ENTROPY (its fifth section
        //     `nf4_matmul` is `registrable: false` — `fdx.quant.family:
        //     AFFINE_BLOCK` consumer-ahead, §6 — so NF4's hand-written regs stay
        //     authoritative) via `CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS`.
        //   - `fused/conv-rope.fkc.md` — all 6 sections → 32 impls (ROPE, CONV2D,
        //     CONV_TRANSPOSE2D, CAUSAL_CONV1D, SELECTIVE_SCAN, SSD_CHUNK_SCAN) via
        //     `CPU_FUSED_CONV_ROPE_ENTRY_POINTS`.
        //
        // Still hand-written in `register_default_fused_kernels` (no fused
        // contract yet): FLASH_ATTN / FLASH_ATTN_BACKWARD_{Q,K,V} / PAGED_ATTN,
        // plus the deferred NF4_MATMUL. The `cpu/*.fkc.md` corpora are all
        // primitive `op_kind` contracts (the "fused" in FusedLinear /
        // FusedSoftmaxCrossEntropy names an intra-op fusion, NOT a graph
        // `FusedOpId`); their separate primitive `KernelBindingTable` seams stay
        // untouched.
        CPU_FUSED_NORM_ENTRY_POINTS
            .iter()
            .chain(CPU_FUSED_LINEAR_QUANT_ENTRY_POINTS.iter())
            .chain(CPU_FUSED_CONV_ROPE_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }
}
