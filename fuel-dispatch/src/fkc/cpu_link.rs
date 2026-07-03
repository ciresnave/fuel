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

/// The CPU **padding** family's `symbol → production wrapper` map — the
/// MIGRATED subset: `PadBackward` × 4 dtypes = 4 kernels (key `[T, T]`).
/// Contract: `docs/kernel-contracts/cpu/padding.fkc.md`. Each per-dtype section
/// (`## pad_backward_f32`, …) declares a SPECIFIC single-dtype `entry_point`
/// (`…::pad_backward_f32`), so none of them fan — the importer resolves that
/// symbol AS-IS. The binding key is `[T, T]` (grad_out + `passthrough(grad_out)`
/// grad_in; the in_shape/out_shape/padding/mode_tag ride in
/// `OpParams::PadBackward`, NOT the dtype-list), identical to the deleted
/// `&unary(t)` regs. `PadBackward` is per-dtype (unlike the dtype-agnostic
/// forward `Pad`) because gradient accumulation is typed (bf16/f16/f32 widen the
/// scratch accumulator to f64).
///
/// **The FORWARD `Pad` half is DEFERRED** (its four multi-dtype sections —
/// `pad_const_cpu` / `pad_reflect_cpu` / `pad_replicate_cpu` / `pad_walk_cpu` —
/// are `registrable: false`, §3.10 describe-only, so they never resolve and are
/// absent here): the three forward modes (Constant/Reflect/Replicate) collapse
/// to ONE `(Pad, [t,t])` binding per dtype served by the SINGLE mode-dispatching
/// `pad_cpu_wrapper` (mode chosen at runtime via `mode_tag`), so the contract's
/// three independent forward sections would collide on the shared key
/// (`DuplicateKernelRef`), and production wires only 6 dtypes
/// (`U8/U32/BF16/F16/F32/F64`) vs the contract's dtype-agnostic 10. The
/// hand-written forward `Pad` regs stay authoritative until the contract models
/// the unified wrapper as a single registrable section.
pub static CPU_PADDING_ENTRY_POINTS: &[(&str, KernelRef)] = &[
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
/// **PagedAttn is DESCRIBE-ONLY** (`registrable: false` in the contract — its
/// `fdx.gather: paged_blocks` pool is [consumer-ahead], §3.9.1), so its four
/// sections never lower/resolve and are ABSENT here; the production `PagedAttn`
/// binding stays hand-written. The SEPARATE `FusedKernelRegistry` FLASH_ATTN*
/// seam (`register_default_fused_kernels`) is likewise a distinct registry and
/// stays untouched; this map only serves the `KernelBindingTable` primitive path.
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

/// The built-in CPU backend's [`LinkRegistry`] — resolves a contract's
/// `entry_point` symbols against [`CPU_BINARY_ENTRY_POINTS`],
/// [`CPU_AFFINE_CLAMP_POWI_ENTRY_POINTS`], [`CPU_UNARY_ENTRY_POINTS`],
/// [`CPU_COMPARE_ENTRY_POINTS`], [`CPU_WHERE_ENTRY_POINTS`],
/// [`CPU_REDUCE_ENTRY_POINTS`], [`CPU_REDUCE_TO_ENTRY_POINTS`],
/// [`CPU_NORM_ENTRY_POINTS`], [`CPU_NORM_BACKWARD_ENTRY_POINTS`],
/// [`CPU_ROPE_ENTRY_POINTS`], [`CPU_SSM_ENTRY_POINTS`],
/// [`CPU_CONV_ENTRY_POINTS`], [`CPU_PADDING_ENTRY_POINTS`],
/// [`CPU_SHAPE_OPS_ENTRY_POINTS`], [`CPU_MATMUL_ENTRY_POINTS`],
/// [`CPU_ATTENTION_ENTRY_POINTS`], and [`CPU_INPLACE_ENTRY_POINTS`].
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
            .chain(CPU_MATMUL_ENTRY_POINTS.iter())
            .chain(CPU_ATTENTION_ENTRY_POINTS.iter())
            .chain(CPU_INPLACE_ENTRY_POINTS.iter())
            .find(|(s, _)| *s == symbol)
            .map(|(_, k)| *k)
    }

    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        // No fused-op contracts in the elementwise-binary, affine/clamp/powi,
        // elementwise-unary, compare/where, reduce, reduce-to, norm,
        // norm-backward, rope, ssm, conv, padding, shape-ops, or matmul corpora.
        // The padding ops
        // are all primitive `op_kind: Pad`/`PadBackward` contracts (no
        // `fused_op`). The ssm ops are all
        // primitive `op_kind` contracts ("fused" in FusedSoftmaxCrossEntropy
        // names an intra-op softmax+NLL fusion, NOT a graph `FusedOpId`); the
        // conv contract's sections are all `op_kind: Conv2D/ConvTranspose2D`
        // primitives (the SEPARATE FusedOps::{CONV2D, CONV_TRANSPOSE2D} registry
        // seam is hand-written, not FKC-imported). The matmul contract's
        // sections are all `op_kind: MatMul`/`FusedLinear` primitives ("fused" in
        // FusedLinear names an intra-op matmul+bias fusion, NOT a graph
        // `FusedOpId`; the SEPARATE FusedOps::FUSED_LINEAR registry seam is
        // hand-written, not FKC-imported). The attention contract's sections are
        // all `op_kind: FlashAttn`/`FlashAttnBackward{Q,K,V}` primitives (bound on
        // the KernelBindingTable) plus describe-only PagedAttn; the SEPARATE
        // FusedOps::{FLASH_ATTN, FLASH_ATTN_BACKWARD_*, PAGED_ATTN} registry seam
        // (`register_default_fused_kernels`) is hand-written, not FKC-imported.
        None
    }
}
