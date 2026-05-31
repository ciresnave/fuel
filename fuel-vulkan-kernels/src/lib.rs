//! # fuel-vulkan-kernels
//!
//! Precompiled SPIR-V kernels for the Fuel Vulkan backend.
//!
//! Shader sources (`.wgsl`, `.glsl`, `.slang`) live in the
//! `fuel-kernels-source` crate at `kernels/`. They are compiled to
//! SPIR-V **ahead of time** using
//! `fuel-kernels-source/kernels/compile.sh` (requires Vulkan SDK +
//! naga-cli) and the resulting `.spv` files are committed to `spv/`
//! in this crate.
//!
//! This crate only exposes the embedded byte table and the
//! environment-variable name for the dev-time disk-override
//! mechanism. Backends are responsible for handing these to whatever
//! shader registry their graphics API uses
//! (`vulkane::safe::ShaderRegistry` for Vulkan).
//!
//! # Overriding at runtime
//!
//! Shader developers iterating on a kernel can avoid rebuilding Fuel
//! by setting [`OVERRIDE_ENV`] (`FUEL_SHADER_OVERRIDE_DIR`) to a
//! directory containing `.spv` files. Any backend that wires
//! [`OVERRIDE_ENV`] into its registry's lookup chain will pick up
//! those overrides instead of the embedded defaults — see the Vulkan
//! backend's `Pipelines::new` for an example.
//!
//! # Adding a new shader
//!
//! 1. Drop the source file into `fuel-kernels-source/kernels/` (any of
//!    `.wgsl`, `.glsl`, `.slang`).
//! 2. Run `./compile.sh` from that directory — writes SPIR-V into
//!    `../fuel-vulkan-kernels/spv/`.
//! 3. Add a `(name, include_bytes!(...))` entry to [`EMBEDDED`] below.
//! 4. Rebuild Fuel.

/// All shaders baked into the Fuel binary, as `(name, spirv_bytes)`
/// pairs. Backends populate their own shader registry from this
/// table.
pub static EMBEDDED: &[(&str, &[u8])] = &[
    ("add_assign_scaled",         include_bytes!("../spv/add_assign_scaled.spv")),
    ("affine",                    include_bytes!("../spv/affine.spv")),
    ("affine_f64",                include_bytes!("../spv/affine_f64.spv")),
    ("affine_f16",                include_bytes!("../spv/affine_f16.spv")),
    ("affine_bf16",               include_bytes!("../spv/affine_bf16.spv")),
    ("unary_bf16",                include_bytes!("../spv/unary_bf16.spv")),
    ("binary",                    include_bytes!("../spv/binary.spv")),
    ("binary_bf16",               include_bytes!("../spv/binary_bf16.spv")),
    ("binary_f16",                include_bytes!("../spv/binary_f16.spv")),
    ("binary_f64",                include_bytes!("../spv/binary_f64.spv")),
    ("clamp",                     include_bytes!("../spv/clamp.spv")),
    ("powi",                      include_bytes!("../spv/powi.spv")),
    ("cast_f32_to_f16",           include_bytes!("../spv/cast_f32_to_f16.spv")),
    ("cast_f16_to_f32",           include_bytes!("../spv/cast_f16_to_f32.spv")),
    ("cast_f32_to_bf16",          include_bytes!("../spv/cast_f32_to_bf16.spv")),
    ("cast_bf16_to_f32",          include_bytes!("../spv/cast_bf16_to_f32.spv")),
    ("cast_f32_to_f64",           include_bytes!("../spv/cast_f32_to_f64.spv")),
    ("cast_f64_to_f32",           include_bytes!("../spv/cast_f64_to_f32.spv")),
    ("cast_f32_to_f8e4m3",        include_bytes!("../spv/cast_f32_to_f8e4m3.spv")),
    ("cast_f8e4m3_to_f32",        include_bytes!("../spv/cast_f8e4m3_to_f32.spv")),
    ("cast_f16_to_f8e4m3",        include_bytes!("../spv/cast_f16_to_f8e4m3.spv")),
    ("cast_f8e4m3_to_f16",        include_bytes!("../spv/cast_f8e4m3_to_f16.spv")),
    ("cast_bf16_to_f8e4m3",       include_bytes!("../spv/cast_bf16_to_f8e4m3.spv")),
    ("cast_f8e4m3_to_bf16",       include_bytes!("../spv/cast_f8e4m3_to_bf16.spv")),
    ("flip_b2",                   include_bytes!("../spv/flip_b2.spv")),
    ("flip_b4",                   include_bytes!("../spv/flip_b4.spv")),
    ("flip_b8",                   include_bytes!("../spv/flip_b8.spv")),
    ("roll_b2",                   include_bytes!("../spv/roll_b2.spv")),
    ("roll_b4",                   include_bytes!("../spv/roll_b4.spv")),
    ("roll_b8",                   include_bytes!("../spv/roll_b8.spv")),
    ("cumsum_f32",                include_bytes!("../spv/cumsum_f32.spv")),
    ("cumsum_f64",                include_bytes!("../spv/cumsum_f64.spv")),
    ("cumsum_f16",                include_bytes!("../spv/cumsum_f16.spv")),
    ("cumsum_bf16",               include_bytes!("../spv/cumsum_bf16.spv")),
    ("triu_b2",                   include_bytes!("../spv/triu_b2.spv")),
    ("triu_b4",                   include_bytes!("../spv/triu_b4.spv")),
    ("triu_b8",                   include_bytes!("../spv/triu_b8.spv")),
    ("tril_b2",                   include_bytes!("../spv/tril_b2.spv")),
    ("tril_b4",                   include_bytes!("../spv/tril_b4.spv")),
    ("tril_b8",                   include_bytes!("../spv/tril_b8.spv")),
    ("strided_copy_signed_b2",    include_bytes!("../spv/strided_copy_signed_b2.spv")),
    ("strided_copy_signed_b4",    include_bytes!("../spv/strided_copy_signed_b4.spv")),
    ("strided_copy_signed_b8",    include_bytes!("../spv/strided_copy_signed_b8.spv")),
    ("gather_b1",                 include_bytes!("../spv/gather_b1.spv")),
    ("gather_b2",                 include_bytes!("../spv/gather_b2.spv")),
    ("gather_b4",                 include_bytes!("../spv/gather_b4.spv")),
    ("gather_b8",                 include_bytes!("../spv/gather_b8.spv")),
    ("masked_fill_b1",            include_bytes!("../spv/masked_fill_b1.spv")),
    ("masked_fill_b2",            include_bytes!("../spv/masked_fill_b2.spv")),
    ("masked_fill_b4",            include_bytes!("../spv/masked_fill_b4.spv")),
    ("masked_fill_b8",            include_bytes!("../spv/masked_fill_b8.spv")),
    ("pad_const_b1",              include_bytes!("../spv/pad_const_b1.spv")),
    ("pad_const_b2",              include_bytes!("../spv/pad_const_b2.spv")),
    ("pad_const_b4",              include_bytes!("../spv/pad_const_b4.spv")),
    ("pad_const_b8",              include_bytes!("../spv/pad_const_b8.spv")),
    ("pad_reflect_b1",            include_bytes!("../spv/pad_reflect_b1.spv")),
    ("pad_reflect_b2",            include_bytes!("../spv/pad_reflect_b2.spv")),
    ("pad_reflect_b4",            include_bytes!("../spv/pad_reflect_b4.spv")),
    ("pad_reflect_b8",            include_bytes!("../spv/pad_reflect_b8.spv")),
    ("pad_replicate_b1",          include_bytes!("../spv/pad_replicate_b1.spv")),
    ("pad_replicate_b2",          include_bytes!("../spv/pad_replicate_b2.spv")),
    ("pad_replicate_b4",          include_bytes!("../spv/pad_replicate_b4.spv")),
    ("pad_replicate_b8",          include_bytes!("../spv/pad_replicate_b8.spv")),
    ("pad_backward_const_b1",     include_bytes!("../spv/pad_backward_const_b1.spv")),
    ("pad_backward_const_b2",     include_bytes!("../spv/pad_backward_const_b2.spv")),
    ("pad_backward_const_b4",     include_bytes!("../spv/pad_backward_const_b4.spv")),
    ("pad_backward_const_b8",     include_bytes!("../spv/pad_backward_const_b8.spv")),
    ("pad_backward_reflect_f32",   include_bytes!("../spv/pad_backward_reflect_f32.spv")),
    ("pad_backward_replicate_f32", include_bytes!("../spv/pad_backward_replicate_f32.spv")),
    ("pad_backward_reflect_f64",   include_bytes!("../spv/pad_backward_reflect_f64.spv")),
    ("pad_backward_replicate_f64", include_bytes!("../spv/pad_backward_replicate_f64.spv")),
    ("pad_backward_reflect_bf16",  include_bytes!("../spv/pad_backward_reflect_bf16.spv")),
    ("pad_backward_replicate_bf16",include_bytes!("../spv/pad_backward_replicate_bf16.spv")),
    ("pad_backward_reflect_f16",   include_bytes!("../spv/pad_backward_reflect_f16.spv")),
    ("pad_backward_replicate_f16", include_bytes!("../spv/pad_backward_replicate_f16.spv")),
    ("write_slice_b1",            include_bytes!("../spv/write_slice_b1.spv")),
    ("write_slice_b2",            include_bytes!("../spv/write_slice_b2.spv")),
    ("write_slice_b4",            include_bytes!("../spv/write_slice_b4.spv")),
    ("write_slice_b8",            include_bytes!("../spv/write_slice_b8.spv")),
    ("concat_along_dim",          include_bytes!("../spv/concat_along_dim.spv")),
    ("concat_along_dim_f16",      include_bytes!("../spv/concat_along_dim_f16.spv")),
    ("concat_along_dim_bf16",     include_bytes!("../spv/concat_along_dim_bf16.spv")),
    ("concat_along_dim_f64",      include_bytes!("../spv/concat_along_dim_f64.spv")),
    ("conv2d_im2col",             include_bytes!("../spv/conv2d_im2col.spv")),
    ("conv2d_im2col_bf16",        include_bytes!("../spv/conv2d_im2col_bf16.spv")),
    ("flash_attention",           include_bytes!("../spv/flash_attention.spv")),
    ("dequant_q4_0",              include_bytes!("../spv/dequant_q4_0.spv")),
    ("dequant_q4_km",             include_bytes!("../spv/dequant_q4_km.spv")),
    ("dequant_q8_0",              include_bytes!("../spv/dequant_q8_0.spv")),
    ("index_select",              include_bytes!("../spv/index_select.spv")),
    ("index_select_f16",          include_bytes!("../spv/index_select_f16.spv")),
    ("index_select_bf16",         include_bytes!("../spv/index_select_bf16.spv")),
    ("index_select_f64",          include_bytes!("../spv/index_select_f64.spv")),
    ("matmul_q4_0_tiled",         include_bytes!("../spv/matmul_q4_0_tiled.spv")),
    ("qmatvec_q4_0",              include_bytes!("../spv/qmatvec_q4_0.spv")),
    ("quantize_q8_0",             include_bytes!("../spv/quantize_q8_0.spv")),
    ("matmul",                    include_bytes!("../spv/matmul.spv")),
    ("matmul_tiled",              include_bytes!("../spv/matmul_tiled.spv")),
    ("matmul_tiled_bf16_b",       include_bytes!("../spv/matmul_tiled_bf16_b.spv")),
    ("matmul_coop",               include_bytes!("../spv/matmul_coop.spv")),
    ("matmul_coop_bf16_bf16",     include_bytes!("../spv/matmul_coop_bf16_bf16.spv")),
    ("matmul_coop_f16_f16",       include_bytes!("../spv/matmul_coop_f16_f16.spv")),
    ("matmul_coop_bf16_bf16_bf16",include_bytes!("../spv/matmul_coop_bf16_bf16_bf16.spv")),
    ("matmul_small_bf16_bf16_f32",include_bytes!("../spv/matmul_small_bf16_bf16_f32.spv")),
    ("matmul_small_bf16_bf16_bf16",include_bytes!("../spv/matmul_small_bf16_bf16_bf16.spv")),
    ("matmul_coop_f16_f16_f16",   include_bytes!("../spv/matmul_coop_f16_f16_f16.spv")),
    ("matvec",                    include_bytes!("../spv/matvec.spv")),
    ("matvec_bf16_b",             include_bytes!("../spv/matvec_bf16_b.spv")),
    ("reduce",                    include_bytes!("../spv/reduce.spv")),
    ("reduce_f16",                include_bytes!("../spv/reduce_f16.spv")),
    ("reduce_bf16",               include_bytes!("../spv/reduce_bf16.spv")),
    ("reduce_f64",                include_bytes!("../spv/reduce_f64.spv")),
    ("arg_reduce_last_dim_f32",   include_bytes!("../spv/arg_reduce_last_dim_f32.spv")),
    ("scatter_add_f32",           include_bytes!("../spv/scatter_add_f32.spv")),
    ("scatter_add_f64",           include_bytes!("../spv/scatter_add_f64.spv")),
    ("scatter_add_bf16",          include_bytes!("../spv/scatter_add_bf16.spv")),
    ("scatter_add_f16",           include_bytes!("../spv/scatter_add_f16.spv")),
    ("arg_reduce_any_dim_f32",    include_bytes!("../spv/arg_reduce_any_dim_f32.spv")),
    ("arg_reduce_any_dim_f64",    include_bytes!("../spv/arg_reduce_any_dim_f64.spv")),
    ("arg_reduce_any_dim_bf16",   include_bytes!("../spv/arg_reduce_any_dim_bf16.spv")),
    ("arg_reduce_any_dim_f16",    include_bytes!("../spv/arg_reduce_any_dim_f16.spv")),
    ("index_add_f32",             include_bytes!("../spv/index_add_f32.spv")),
    ("index_add_f64",             include_bytes!("../spv/index_add_f64.spv")),
    ("index_add_bf16",            include_bytes!("../spv/index_add_bf16.spv")),
    ("index_add_f16",             include_bytes!("../spv/index_add_f16.spv")),
    ("arg_reduce_last_dim_f16",   include_bytes!("../spv/arg_reduce_last_dim_f16.spv")),
    ("arg_reduce_last_dim_bf16",  include_bytes!("../spv/arg_reduce_last_dim_bf16.spv")),
    ("arg_reduce_last_dim_f64",   include_bytes!("../spv/arg_reduce_last_dim_f64.spv")),
    ("reduce_last_dim",           include_bytes!("../spv/reduce_last_dim.spv")),
    ("reduce_last_dim_f16",       include_bytes!("../spv/reduce_last_dim_f16.spv")),
    ("reduce_last_dim_bf16",      include_bytes!("../spv/reduce_last_dim_bf16.spv")),
    ("reduce_last_dim_f64",       include_bytes!("../spv/reduce_last_dim_f64.spv")),
    ("rms_norm_last_dim",         include_bytes!("../spv/rms_norm_last_dim.spv")),
    ("rms_norm_last_dim_f16",     include_bytes!("../spv/rms_norm_last_dim_f16.spv")),
    ("rms_norm_last_dim_bf16",    include_bytes!("../spv/rms_norm_last_dim_bf16.spv")),
    ("rms_norm_last_dim_f64",     include_bytes!("../spv/rms_norm_last_dim_f64.spv")),
    ("rms_norm_last_dim_backward", include_bytes!("../spv/rms_norm_last_dim_backward.spv")),
    ("rope",                      include_bytes!("../spv/rope.spv")),
    ("rope_f16",                  include_bytes!("../spv/rope_f16.spv")),
    ("rope_bf16",                 include_bytes!("../spv/rope_bf16.spv")),
    ("rope_f64",                  include_bytes!("../spv/rope_f64.spv")),
    ("softmax",                   include_bytes!("../spv/softmax.spv")),
    ("softmax_f16",               include_bytes!("../spv/softmax_f16.spv")),
    ("softmax_bf16",              include_bytes!("../spv/softmax_bf16.spv")),
    ("softmax_f64",               include_bytes!("../spv/softmax_f64.spv")),
    ("softmax_last_dim_backward", include_bytes!("../spv/softmax_last_dim_backward.spv")),
    ("softmax_last_dim_backward_f16",  include_bytes!("../spv/softmax_last_dim_backward_f16.spv")),
    ("softmax_last_dim_backward_bf16", include_bytes!("../spv/softmax_last_dim_backward_bf16.spv")),
    ("softmax_last_dim_backward_f64",  include_bytes!("../spv/softmax_last_dim_backward_f64.spv")),
    ("layer_norm_last_dim_backward", include_bytes!("../spv/layer_norm_last_dim_backward.spv")),
    ("layer_norm_last_dim_backward_f16",  include_bytes!("../spv/layer_norm_last_dim_backward_f16.spv")),
    ("layer_norm_last_dim_backward_bf16", include_bytes!("../spv/layer_norm_last_dim_backward_bf16.spv")),
    ("layer_norm_last_dim_backward_f64",  include_bytes!("../spv/layer_norm_last_dim_backward_f64.spv")),
    ("layer_norm_last_dim",       include_bytes!("../spv/layer_norm_last_dim.spv")),
    ("layer_norm_last_dim_f16",   include_bytes!("../spv/layer_norm_last_dim_f16.spv")),
    ("layer_norm_last_dim_bf16",  include_bytes!("../spv/layer_norm_last_dim_bf16.spv")),
    ("layer_norm_last_dim_f64",   include_bytes!("../spv/layer_norm_last_dim_f64.spv")),
    ("strided_copy",              include_bytes!("../spv/strided_copy.spv")),
    ("unary",                     include_bytes!("../spv/unary.spv")),
    ("unary_f16",                 include_bytes!("../spv/unary_f16.spv")),
    ("unary_f64",                 include_bytes!("../spv/unary_f64.spv")),
];

/// Environment variable backends consult for an optional disk-override
/// directory. Set this during shader development to hot-swap individual
/// `.spv` files without rebuilding Fuel.
pub const OVERRIDE_ENV: &str = "FUEL_SHADER_OVERRIDE_DIR";

// ---- Shader name constants ---------------------------------------------
//
// Public string identifiers used by call sites that reference
// shaders by symbolic name rather than typing the literal. The names
// here match the keys in [`EMBEDDED`] above.

/// Element-wise unary ops (13 ops via uniform selector).
pub const UNARY: &str = "unary";
/// Element-wise unary ops, f16. Same 13-op surface as UNARY but
/// operates on native float16_t (needs shaderFloat16 + 16BitStorage).
pub const UNARY_F16: &str = "unary_f16";
/// Element-wise unary ops, f64 (needs shaderFloat64).
pub const UNARY_F64: &str = "unary_f64";
/// Element-wise unary ops, bf16. Same 13-op surface as UNARY but
/// stores bf16 packed two-per-u32 and does math at f32 with manual
/// round-trip conversion (no native bfloat16_t in Slang).
pub const UNARY_BF16: &str = "unary_bf16";
/// Element-wise binary ops (6 ops via uniform selector).
pub const BINARY: &str = "binary";
/// Element-wise binary ops, f16. Same 6-op surface as BINARY but
/// operates on native float16_t.
pub const BINARY_F16: &str = "binary_f16";
/// Element-wise binary ops, f64.
pub const BINARY_F64: &str = "binary_f64";
/// Element-wise binary ops, bf16. Stride-aware via the same Params
/// layout as BINARY_F16; bf16<->f32 round-trip via manual bit shifts.
pub const BINARY_BF16: &str = "binary_bf16";
/// Affine transform: y = x * mul + add.
pub const AFFINE: &str = "affine";
/// Affine f64 (native double; stride-aware).
pub const AFFINE_F64: &str = "affine_f64";
/// Affine f16 (float16_t native; stride-aware; f32 math).
pub const AFFINE_F16: &str = "affine_f16";
/// Affine bf16 (packed-u32 pair-thread; contiguous-only; f32 math).
pub const AFFINE_BF16: &str = "affine_bf16";
/// Element-wise clamp: y = clamp(x, lo, hi).
pub const CLAMP: &str = "clamp";
/// Element-wise integer power: y = x ^ exp.
pub const POWI: &str = "powi";
/// Cast f32 → f16 (rounded to nearest-even via f32tof16).
pub const CAST_F32_TO_F16: &str = "cast_f32_to_f16";
/// Cast f16 → f32 (exact, via f16tof32).
pub const CAST_F16_TO_F32: &str = "cast_f16_to_f32";
/// Cast f32 → bf16 (truncate-toward-zero: bits >> 16).
pub const CAST_F32_TO_BF16: &str = "cast_f32_to_bf16";
/// Cast bf16 → f32 (exact: bits << 16).
pub const CAST_BF16_TO_F32: &str = "cast_bf16_to_f32";
/// Cast f32 → f64 (widening, lossless). One thread per element.
pub const CAST_F32_TO_F64: &str = "cast_f32_to_f64";
/// Cast f64 → f32 (narrowing, round-to-nearest-even).
pub const CAST_F64_TO_F32: &str = "cast_f64_to_f32";
/// Cast f32 → F8E4M3 (round-to-nearest-even, saturate to ±448).
pub const CAST_F32_TO_F8E4M3: &str = "cast_f32_to_f8e4m3";
/// Cast F8E4M3 → f32 (exact reverse).
pub const CAST_F8E4M3_TO_F32: &str = "cast_f8e4m3_to_f32";
/// Cast f16 → F8E4M3 (via f32; same round-to-nearest-even + saturate).
pub const CAST_F16_TO_F8E4M3: &str = "cast_f16_to_f8e4m3";
/// Cast F8E4M3 → f16 (via f32; uses f32tof16 final round).
pub const CAST_F8E4M3_TO_F16: &str = "cast_f8e4m3_to_f16";
/// Cast bf16 → F8E4M3 (via f32).
pub const CAST_BF16_TO_F8E4M3: &str = "cast_bf16_to_f8e4m3";
/// Cast F8E4M3 → bf16 (via f32).
pub const CAST_F8E4M3_TO_BF16: &str = "cast_f8e4m3_to_bf16";
/// Pad with constant fill, byte-width-keyed (b1 = u8/i8, b2 = f16/bf16/i16/u16,
/// b4 = f32/i32/u32, b8 = f64/i64). One workgroup processes 256 output
/// elements (b4/b8) or 256 pairs/quads (b2/b1). Caller passes the fill
/// value as a bit pattern in the Params struct.
pub const PAD_CONST_B1: &str = "pad_const_b1";
pub const PAD_CONST_B2: &str = "pad_const_b2";
pub const PAD_CONST_B4: &str = "pad_const_b4";
pub const PAD_CONST_B8: &str = "pad_const_b8";
/// Pad with reflect mode (no-repeat), byte-width-keyed. Same shape
/// as pad_const_* but each thread maps out-of-range coords via the
/// reflect formula. PRECONDITION: per-axis left_pad and right_pad
/// must both be <= in_dim - 1.
pub const PAD_REFLECT_B1: &str = "pad_reflect_b1";
pub const PAD_REFLECT_B2: &str = "pad_reflect_b2";
pub const PAD_REFLECT_B4: &str = "pad_reflect_b4";
pub const PAD_REFLECT_B8: &str = "pad_reflect_b8";
/// Pad with replicate (edge-repeat) mode, byte-width-keyed. Each
/// out-of-range coord clamps to [0, in_dim - 1]. No precondition on
/// left_pad/right_pad sizes.
pub const PAD_REPLICATE_B1: &str = "pad_replicate_b1";
pub const PAD_REPLICATE_B2: &str = "pad_replicate_b2";
pub const PAD_REPLICATE_B4: &str = "pad_replicate_b4";
pub const PAD_REPLICATE_B8: &str = "pad_replicate_b8";
/// PadBackward constant mode: one thread per input element reads
/// the corresponding (offset by left_pad) output element. No
/// accumulation needed for constant mode.
pub const PAD_BACKWARD_CONST_B1: &str = "pad_backward_const_b1";
pub const PAD_BACKWARD_CONST_B2: &str = "pad_backward_const_b2";
pub const PAD_BACKWARD_CONST_B4: &str = "pad_backward_const_b4";
pub const PAD_BACKWARD_CONST_B8: &str = "pad_backward_const_b8";
/// PadBackward reflect/replicate mode — f32 only. Atomic-add via
/// uint CAS into the (zero-initialized) grad_in buffer. f16/bf16/
/// f64 follow once the sub-word / u64 CAS variants are wired.
pub const PAD_BACKWARD_REFLECT_F32: &str = "pad_backward_reflect_f32";
pub const PAD_BACKWARD_REPLICATE_F32: &str = "pad_backward_replicate_f32";
pub const PAD_BACKWARD_REFLECT_F64: &str = "pad_backward_reflect_f64";
pub const PAD_BACKWARD_REPLICATE_F64: &str = "pad_backward_replicate_f64";
pub const PAD_BACKWARD_REFLECT_BF16: &str = "pad_backward_reflect_bf16";
pub const PAD_BACKWARD_REPLICATE_BF16: &str = "pad_backward_replicate_bf16";
pub const PAD_BACKWARD_REFLECT_F16: &str = "pad_backward_reflect_f16";
pub const PAD_BACKWARD_REPLICATE_F16: &str = "pad_backward_replicate_f16";
/// MaskedFill: for each element, if mask byte != 0 → fill_value, else
/// copy input. Mask is always U8. Byte-width-keyed by element size.
pub const MASKED_FILL_B1: &str = "masked_fill_b1";
pub const MASKED_FILL_B2: &str = "masked_fill_b2";
pub const MASKED_FILL_B4: &str = "masked_fill_b4";
pub const MASKED_FILL_B8: &str = "masked_fill_b8";
/// Gather along `dim`: each output position's source coord at `dim`
/// is read from a U32 indices tensor (same shape as output). All
/// other coords are shared between source and output. Byte-width-
/// keyed by element size.
pub const GATHER_B1: &str = "gather_b1";
pub const GATHER_B2: &str = "gather_b2";
pub const GATHER_B4: &str = "gather_b4";
pub const GATHER_B8: &str = "gather_b8";
/// In-place rectangular slab write for 1-byte elements (u8/i8).
/// `range_start[last]` and `src_shape[last]` must both be multiples
/// of 4 — wrapper falls back to CPU otherwise.
pub const WRITE_SLICE_B1: &str = "write_slice_b1";
/// In-place rectangular slab write for 2-byte elements (f16/bf16).
pub const WRITE_SLICE_B2: &str = "write_slice_b2";
/// In-place rectangular slab write for 4-byte elements (f32/i32/u32).
pub const WRITE_SLICE_B4: &str = "write_slice_b4";
/// In-place rectangular slab write for 8-byte elements (f64/i64).
pub const WRITE_SLICE_B8: &str = "write_slice_b8";
/// Triu mask along the last two dims (4-byte elements).
pub const TRIU_B2: &str = "triu_b2";
pub const TRIU_B4: &str = "triu_b4";
pub const TRIU_B8: &str = "triu_b8";
/// Tril mask along the last two dims.
pub const TRIL_B2: &str = "tril_b2";
pub const TRIL_B4: &str = "tril_b4";
pub const TRIL_B8: &str = "tril_b8";
/// Flip along one dim (flat 3-tuple view: outer × dim_size × inner).
pub const FLIP_B2: &str = "flip_b2";
pub const FLIP_B4: &str = "flip_b4";
pub const FLIP_B8: &str = "flip_b8";
/// Cyclic shift along one dim.
pub const ROLL_B2: &str = "roll_b2";
pub const ROLL_B4: &str = "roll_b4";
pub const ROLL_B8: &str = "roll_b8";
/// Inclusive prefix sum (cumulative sum) along one dim. Per-dtype
/// because the accumulator needs typed addition (unlike flip/roll
/// which are pure data movement). Sequential per-slice walk;
/// stride-aware (rank-N + per-input strides + axis from OpParams).
pub const CUMSUM_F32: &str = "cumsum_f32";
pub const CUMSUM_F64: &str = "cumsum_f64";
pub const CUMSUM_F16: &str = "cumsum_f16";
pub const CUMSUM_BF16: &str = "cumsum_bf16";
/// Strided copy with signed strides (Contiguize on negative-stride
/// views from Flip / Roll / layout-on-Node).
pub const STRIDED_COPY_SIGNED_B2: &str = "strided_copy_signed_b2";
pub const STRIDED_COPY_SIGNED_B4: &str = "strided_copy_signed_b4";
pub const STRIDED_COPY_SIGNED_B8: &str = "strided_copy_signed_b8";
/// Tiled matrix multiply with 4x4 register tiling (WGSL).
pub const MATMUL: &str = "matmul";
/// GLSL matmul with shared-memory blocking.
pub const MATMUL_TILED_GLSL: &str = "matmul_tiled";
/// GLSL tiled matmul with bf16 weights: f32 A × bf16 B → f32 C.
/// Same tiling as MATMUL_TILED_GLSL; bf16 unpack on the B load.
pub const MATMUL_TILED_BF16_B_GLSL: &str = "matmul_tiled_bf16_b";
/// Cooperative-matrix (tensor-core) matmul: f32 A × bf16 B → f32 C.
/// Uses VK_KHR_cooperative_matrix with f16 inputs + f32 accumulation.
/// Only dispatched when the extension is available at runtime.
pub const MATMUL_COOP: &str = "matmul_coop";
/// Cooperative-matrix tiled matmul, bf16 × bf16 → f32. Sibling of
/// `MATMUL_COOP` (f32 × bf16). Both operands stored as bf16 (u16);
/// the kernel downcasts bf16→f16 on shared-mem load and uses the
/// coop[3] tile shape (A=f16, B=f16, C=f32, R=f32) for the matmul,
/// keeping the f32 accumulator.
pub const MATMUL_COOP_BF16_BF16: &str = "matmul_coop_bf16_bf16";
/// Cooperative-matrix tiled matmul, f16 × f16 → f32. Native f16
/// inputs (no downcast); same coop[3] tile + f32 accumulator as the
/// bf16 sibling.
pub const MATMUL_COOP_F16_F16: &str = "matmul_coop_f16_f16";
/// Cooperative-matrix tiled matmul, bf16 × bf16 → bf16 (downcast
/// store). f32 accumulator staged to shared mem, then 128 threads
/// convert+pack to packed-u32 bf16 output. Closes the bf16 inference
/// chain (next layer can consume bf16 directly).
pub const MATMUL_COOP_BF16_BF16_BF16: &str = "matmul_coop_bf16_bf16_bf16";
/// Small-shape bf16 × bf16 → f32 fallback. One thread per output
/// element with an f32 accumulator. Handles any shape; the picker
/// routes here when the coop-matrix kernel's 16-tile constraints
/// don't apply.
pub const MATMUL_SMALL_BF16_BF16_F32: &str = "matmul_small_bf16_bf16_f32";
/// Small-shape bf16 × bf16 → bf16 fallback. Same as the →f32
/// variant but with f32→bf16 downcast on the final store.
pub const MATMUL_SMALL_BF16_BF16_BF16: &str = "matmul_small_bf16_bf16_bf16";
/// Cooperative-matrix tiled matmul, f16 × f16 → f16 (downcast
/// store). Same staging pattern as the bf16→bf16 sibling but uses
/// `float16BitsToUint16` to pack the f32 accumulator into f16 lanes.
pub const MATMUL_COOP_F16_F16_F16: &str = "matmul_coop_f16_f16_f16";
/// GLSL gemv (M == 1 matmul specialization), all-f32.
pub const MATVEC_GLSL: &str = "matvec";
/// GLSL gemv (M == 1) with bf16 weight matrix (B), f32 activations
/// (A) and f32 output (C). Decode-phase path for bf16-quantized LLM
/// weights on GPU.
pub const MATVEC_BF16_B_GLSL: &str = "matvec_bf16_b";
/// Fused softmax along the last dimension (f32).
pub const SOFTMAX: &str = "softmax";
/// Softmax last-dim, f16 storage with f32 intermediate (mixed precision).
pub const SOFTMAX_F16: &str = "softmax_f16";
/// Softmax last-dim, bf16 storage (packed u32) with f32 intermediate.
/// Lane-pair scheme: each lane processes one u32 = two bf16 lanes;
/// `n_cols` must be even.
pub const SOFTMAX_BF16: &str = "softmax_bf16";
/// Softmax last-dim, native f64 end-to-end. Needs shaderFloat64 +
/// GroupNonUniformArithmetic; uses GLSL.std.450 Exp (NOT OpenCL.std).
pub const SOFTMAX_F64: &str = "softmax_f64";
/// Fused softmax backward: dx = y * (g - dot(y, g)) (f32).
pub const SOFTMAX_LAST_DIM_BACKWARD: &str = "softmax_last_dim_backward";
/// Softmax backward, f16 storage, f32 dot reduction.
pub const SOFTMAX_LAST_DIM_BACKWARD_F16: &str = "softmax_last_dim_backward_f16";
/// Softmax backward, bf16 packed-u32 storage with pair-thread layout
/// in Phase 2 (no race). Requires `n_cols % 2 == 0`.
pub const SOFTMAX_LAST_DIM_BACKWARD_BF16: &str = "softmax_last_dim_backward_bf16";
/// Softmax backward, native f64 end-to-end.
pub const SOFTMAX_LAST_DIM_BACKWARD_F64: &str = "softmax_last_dim_backward_f64";
/// Fused layer-norm backward (4 reductions: sum_x, sum_x², sum_g, sum_gx, f32).
pub const LAYER_NORM_LAST_DIM_BACKWARD: &str = "layer_norm_last_dim_backward";
/// LayerNorm backward, f16 storage with f32 reductions.
pub const LAYER_NORM_LAST_DIM_BACKWARD_F16: &str = "layer_norm_last_dim_backward_f16";
/// LayerNorm backward, bf16 packed-u32 storage with f32 reductions.
pub const LAYER_NORM_LAST_DIM_BACKWARD_BF16: &str = "layer_norm_last_dim_backward_bf16";
/// LayerNorm backward, native f64.
pub const LAYER_NORM_LAST_DIM_BACKWARD_F64: &str = "layer_norm_last_dim_backward_f64";
/// Fused layer-norm forward: y = (x - mean) / sqrt(var + eps).
pub const LAYER_NORM_LAST_DIM: &str = "layer_norm_last_dim";
/// LayerNorm forward, f16 mixed precision.
pub const LAYER_NORM_LAST_DIM_F16: &str = "layer_norm_last_dim_f16";
/// LayerNorm forward, bf16 packed-u32 mixed precision.
pub const LAYER_NORM_LAST_DIM_BF16: &str = "layer_norm_last_dim_bf16";
/// LayerNorm forward, native f64.
pub const LAYER_NORM_LAST_DIM_F64: &str = "layer_norm_last_dim_f64";
/// Parallel reduction over all elements (f32).
pub const REDUCE: &str = "reduce";
/// Full-tensor reduction, f16 storage with f32 accumulator.
pub const REDUCE_F16: &str = "reduce_f16";
/// Full-tensor reduction, bf16 storage (packed u32, lane-pair input)
/// with f32 accumulator. Single output bf16 in low 16 bits of
/// output[0]; n MUST be even.
pub const REDUCE_BF16: &str = "reduce_bf16";
/// Full-tensor reduction, native f64.
pub const REDUCE_F64: &str = "reduce_f64";
/// Argmax / argmin along the last dimension. Per-row workgroup with
/// tree-reduction tracking (val, idx) pairs in shared memory.
/// op_id: 0 = argmax, 1 = argmin. Output is U32 indices.
pub const ARG_REDUCE_LAST_DIM_F32: &str = "arg_reduce_last_dim_f32";
/// ScatterAdd along `dim` — f32. Output starts initialized to base
/// (copied by the wrapper); kernel atomically accumulates src into
/// out at the indexed positions via uint CAS (stock Vulkan, no
/// VK_EXT_shader_atomic_float required).
pub const SCATTER_ADD_F32: &str = "scatter_add_f32";
/// ScatterAdd f64 via u64 CAS. Requires shaderInt64 + shaderInt64
/// atomics + shaderFloat64 (Vulkan 1.2 core; available on every
/// device that supports f64).
pub const SCATTER_ADD_F64: &str = "scatter_add_f64";
/// ScatterAdd bf16 via sub-word CAS. Each u32 holds two bf16 lanes;
/// the kernel CAS-replaces only the target half. Math is at f32.
pub const SCATTER_ADD_BF16: &str = "scatter_add_bf16";
/// ScatterAdd f16 via sub-word CAS. Same pattern as bf16 but uses
/// Slang's `f16tof32`/`f32tof16` half-word builtins.
pub const SCATTER_ADD_F16: &str = "scatter_add_f16";
/// Argmax/Argmin along an arbitrary dim (f32). Slow-path counterpart
/// to `arg_reduce_last_dim_f32`: one thread per output element,
/// serial scan over the reduction dim. Used when dim != last.
pub const ARG_REDUCE_ANY_DIM_F32: &str = "arg_reduce_any_dim_f32";
/// Argmax/Argmin along an arbitrary dim (f64). Same shape as
/// `arg_reduce_any_dim_f32` with native double values.
pub const ARG_REDUCE_ANY_DIM_F64: &str = "arg_reduce_any_dim_f64";
/// Argmax/Argmin along an arbitrary dim (bf16). Lane-select reads
/// from packed-u32 input; reduction direction is strided.
pub const ARG_REDUCE_ANY_DIM_BF16: &str = "arg_reduce_any_dim_bf16";
/// Argmax/Argmin along an arbitrary dim (f16). Same as bf16 path
/// with `f16tof32` half-word conversion.
pub const ARG_REDUCE_ANY_DIM_F16: &str = "arg_reduce_any_dim_f16";
/// IndexAdd f32 via uint CAS atomic-add. Same CAS primitive as
/// ScatterAdd; differs only in dst_off (rank-1 indices instead of
/// rank-N).
pub const INDEX_ADD_F32: &str = "index_add_f32";
/// IndexAdd f64 via u64 CAS atomic double-add.
pub const INDEX_ADD_F64: &str = "index_add_f64";
/// IndexAdd bf16 via sub-word CAS atomic add.
pub const INDEX_ADD_BF16: &str = "index_add_bf16";
/// IndexAdd f16 via sub-word CAS atomic add (`f16tof32`/`f32tof16`).
pub const INDEX_ADD_F16: &str = "index_add_f16";
pub const ARG_REDUCE_LAST_DIM_F16: &str = "arg_reduce_last_dim_f16";
pub const ARG_REDUCE_LAST_DIM_BF16: &str = "arg_reduce_last_dim_bf16";
pub const ARG_REDUCE_LAST_DIM_F64: &str = "arg_reduce_last_dim_f64";
/// Per-row reduction along the last dimension (f32).
pub const REDUCE_LAST_DIM: &str = "reduce_last_dim";
/// Per-row reduction along last dim, f16 storage with f32 accumulator.
pub const REDUCE_LAST_DIM_F16: &str = "reduce_last_dim_f16";
/// Per-row reduction along last dim, bf16 storage (packed u32, lane-
/// pair input) with f32 accumulator. Output buffer MUST be
/// zero-initialized by the wrapper; the kernel uses InterlockedOr to
/// write a single bf16 half-word per row without racing the other
/// workgroup writing to the same u32.
pub const REDUCE_LAST_DIM_BF16: &str = "reduce_last_dim_bf16";
/// Per-row reduction along last dim, native f64 end-to-end.
pub const REDUCE_LAST_DIM_F64: &str = "reduce_last_dim_f64";
/// Fused root-mean-square normalization along the last dimension (f32).
pub const RMS_NORM_LAST_DIM: &str = "rms_norm_last_dim";
/// RMSNorm last-dim, f16 storage with f32 accumulation.
pub const RMS_NORM_LAST_DIM_F16: &str = "rms_norm_last_dim_f16";
/// RMSNorm last-dim, bf16 storage (packed u32) with f32 accumulation.
/// Lane-pair scheme: each lane processes one u32 = two bf16 lanes;
/// `n_cols` must be even (LLM hidden_dim always is).
pub const RMS_NORM_LAST_DIM_BF16: &str = "rms_norm_last_dim_bf16";
/// RMSNorm last-dim, native f64 end-to-end. Needs shaderFloat64 +
/// GroupNonUniformArithmetic; uses GLSL.std.450 Sqrt (NOT OpenCL.std).
pub const RMS_NORM_LAST_DIM_F64: &str = "rms_norm_last_dim_f64";
/// Fused RMSNorm backward (grad_x from x + upstream).
pub const RMS_NORM_LAST_DIM_BACKWARD: &str = "rms_norm_last_dim_backward";
/// Strided copy (permute / broadcast / concat / slice).
pub const STRIDED_COPY: &str = "strided_copy";
/// Row-wise gather along a specified dim (embedding lookup, f32).
pub const INDEX_SELECT: &str = "index_select";
/// Index-select f16, pure data movement.
pub const INDEX_SELECT_F16: &str = "index_select_f16";
/// Index-select bf16, packed-u32 storage with pair-thread layout
/// (each thread copies a full u32 = 2 bf16 lanes). Requires
/// `inner % 2 == 0` — embedding-style workloads always satisfy this.
pub const INDEX_SELECT_BF16: &str = "index_select_bf16";
/// Index-select f64, pure data movement.
pub const INDEX_SELECT_F64: &str = "index_select_f64";
/// In-place scaled accumulate (`dst += src * scale`).
pub const ADD_ASSIGN_SCALED: &str = "add_assign_scaled";
/// Fused rotary position embedding.
pub const ROPE: &str = "rope";
/// Fused RoPE, f16 storage with f32 rotation arithmetic.
pub const ROPE_F16: &str = "rope_f16";
/// Fused RoPE, bf16 packed-u32 storage with f32 rotation. Pair-thread
/// layout (each thread writes 2 full u32 words covering 4 bf16
/// positions); requires `head_dim % 4 == 0`.
pub const ROPE_BF16: &str = "rope_bf16";
/// Fused RoPE, native f64 end-to-end.
pub const ROPE_F64: &str = "rope_f64";
/// Single-dispatch concat along an arbitrary dim (f32).
pub const CONCAT_ALONG_DIM: &str = "concat_along_dim";
/// Concat f16, native float16_t storage.
pub const CONCAT_ALONG_DIM_F16: &str = "concat_along_dim_f16";
/// Concat bf16, packed-u32 storage. Single-thread-per-bf16 layout
/// with InterlockedOr half-word output writes — handles the case
/// where adjacent output positions come from different sources at
/// the (a, b) boundary. Wrapper zero-fills output first.
pub const CONCAT_ALONG_DIM_BF16: &str = "concat_along_dim_bf16";
/// Concat f64, native double storage.
pub const CONCAT_ALONG_DIM_F64: &str = "concat_along_dim_f64";
/// Conv2D im2col patches rearrangement. Output of this kernel feeds
/// the existing matmul shaders as the right-hand operand to compute
/// conv2d (one matmul per (batch, group) sub-block).
pub const CONV2D_IM2COL: &str = "conv2d_im2col";
/// Conv2D im2col rearrangement for bf16 input/output. Pairs with
/// `MATMUL_COOP_BF16_BF16_BF16` to complete the Conv2D bf16 pipeline
/// (im2col + half-precision coop matmul).
pub const CONV2D_IM2COL_BF16: &str = "conv2d_im2col_bf16";
/// FlashAttention v2 forward (Phase 8 Tier 2). Tile-based scaled-dot-
/// product attention with online softmax. Handles GQA (Hq > Hkv),
/// causal mask, sliding window, ALiBi, and softcap. Limited to
/// head_dim ≤ 128 by D_MAX in the shader.
pub const FLASH_ATTENTION: &str = "flash_attention";
/// GGML Q4_0 block dequantization to f32.
pub const DEQUANT_Q4_0: &str = "dequant_q4_0";
/// GGML Q4_K_M super-block dequantization to f32.
pub const DEQUANT_Q4_KM: &str = "dequant_q4_km";
/// GGML Q8_0 block dequantization to f32.
pub const DEQUANT_Q8_0: &str = "dequant_q8_0";
/// Fused Q4_0 × F32 gemv (decode hot path for quantized models).
pub const QMATVEC_Q4_0: &str = "qmatvec_q4_0";
/// Fused Q4_0 × F32 tiled matmul for M>1 (prefill hot path).
pub const MATMUL_Q4_0_TILED: &str = "matmul_q4_0_tiled";
/// F32 → Q8_0 quantization (for KV-cache compression).
pub const QUANTIZE_Q8_0: &str = "quantize_q8_0";
