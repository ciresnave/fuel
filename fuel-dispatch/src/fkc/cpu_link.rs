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

/// The CPU SSM / Mamba family's `symbol → production wrapper` map — the
/// MIGRATED subset only: FusedSoftmaxCrossEntropy + CausalConv1d
/// (2 ops × 4 dtypes = 8). Contract: `docs/kernel-contracts/cpu/ssm.fkc.md`.
/// Each per-(op, dtype) section (`## fused_softmax_cross_entropy_f32`,
/// `## causal_conv1d_f32`, …) declares a SPECIFIC single-dtype `entry_point`
/// (`…::fused_softmax_cross_entropy_f32`), so none of them fan — the importer
/// resolves that symbol AS-IS. FSCE's binding key is `[T, I64, F32]` (logits T
/// + I64 targets → `fixed(F32)` output; n_rows / vocab / reduction /
/// ignore_index ride in `OpParams::FusedSoftmaxCrossEntropy`, NOT the
/// dtype-list); CausalConv1d's is `[T, T, T, T]` (x, weight, bias +
/// `passthrough(x)` output; batch / channels / seq / kernel / use_silu ride in
/// `OpParams::CausalConv1d`).
///
/// The two SCAN ops (`selective_scan`, `ssd_chunk_scan`) are DEFERRED and are
/// ABSENT here: their `return.bundle` multi-output (Option C, one buffer
/// `[y ; last_state]`) is not yet key-buildable by the importer, which reads
/// `return.outputs` ONLY (`fkc/lower.rs` `assemble_dtype_variants`). A bundle
/// section would therefore key on 5 input dtypes (missing the bundled output
/// slot) whereas production registers a 6-dtype `[T; 6]` key. Those sections
/// are `registrable: false` in the contract (so the importer skips them) and
/// keep their hand-written `table.register(...)` regs. The migrated ops have NO
/// `##` chassis umbrella section, so there is no `registrable: false`
/// describe-only entry to omit for them.
pub static CPU_SSM_ENTRY_POINTS: &[(&str, KernelRef)] = &[
    ep!("fused_softmax_cross_entropy", "f32",  fused_softmax_cross_entropy_f32_cpu_wrapper),
    ep!("fused_softmax_cross_entropy", "f64",  fused_softmax_cross_entropy_f64_cpu_wrapper),
    ep!("fused_softmax_cross_entropy", "bf16", fused_softmax_cross_entropy_bf16_cpu_wrapper),
    ep!("fused_softmax_cross_entropy", "f16",  fused_softmax_cross_entropy_f16_cpu_wrapper),
    ep!("causal_conv1d", "f32",  causal_conv1d_f32_cpu_wrapper),
    ep!("causal_conv1d", "f64",  causal_conv1d_f64_cpu_wrapper),
    ep!("causal_conv1d", "bf16", causal_conv1d_bf16_cpu_wrapper),
    ep!("causal_conv1d", "f16",  causal_conv1d_f16_cpu_wrapper),
];

/// The built-in CPU backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`CPU_BINARY_ENTRY_POINTS`],
/// [`CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS`], [`CPU_UNARY_ENTRY_POINTS`],
/// [`CPU_COMPARE_ENTRY_POINTS`], [`CPU_WHERE_ENTRY_POINTS`],
/// [`CPU_REDUCE_ENTRY_POINTS`], [`CPU_REDUCE_TO_ENTRY_POINTS`],
/// [`CPU_NORM_ENTRY_POINTS`], [`CPU_NORM_BACKWARD_ENTRY_POINTS`],
/// [`CPU_ROPE_ENTRY_POINTS`], and [`CPU_SSM_ENTRY_POINTS`].
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
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the elementwise-binary, affine/clamp/powi,
        // elementwise-unary, compare/where, reduce, reduce-to, norm,
        // norm-backward, rope, or ssm corpora (the ssm ops are all primitive
        // `op_kind` contracts — "fused" in FusedSoftmaxCrossEntropy names an
        // intra-op softmax+NLL fusion, NOT a graph `FusedOpId`).
        None
    }
}
