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
    ("binary",                    include_bytes!("../spv/binary.spv")),
    ("concat_along_dim",          include_bytes!("../spv/concat_along_dim.spv")),
    ("dequant_q4_0",              include_bytes!("../spv/dequant_q4_0.spv")),
    ("dequant_q4_km",             include_bytes!("../spv/dequant_q4_km.spv")),
    ("dequant_q8_0",              include_bytes!("../spv/dequant_q8_0.spv")),
    ("index_select",              include_bytes!("../spv/index_select.spv")),
    ("matmul_q4_0_tiled",         include_bytes!("../spv/matmul_q4_0_tiled.spv")),
    ("qmatvec_q4_0",              include_bytes!("../spv/qmatvec_q4_0.spv")),
    ("quantize_q8_0",             include_bytes!("../spv/quantize_q8_0.spv")),
    ("matmul",                    include_bytes!("../spv/matmul.spv")),
    ("matmul_tiled",              include_bytes!("../spv/matmul_tiled.spv")),
    ("matmul_tiled_bf16_b",       include_bytes!("../spv/matmul_tiled_bf16_b.spv")),
    ("matmul_coop",               include_bytes!("../spv/matmul_coop.spv")),
    ("matvec",                    include_bytes!("../spv/matvec.spv")),
    ("matvec_bf16_b",             include_bytes!("../spv/matvec_bf16_b.spv")),
    ("reduce",                    include_bytes!("../spv/reduce.spv")),
    ("reduce_last_dim",           include_bytes!("../spv/reduce_last_dim.spv")),
    ("rms_norm_last_dim",         include_bytes!("../spv/rms_norm_last_dim.spv")),
    ("rms_norm_last_dim_backward", include_bytes!("../spv/rms_norm_last_dim_backward.spv")),
    ("rope",                      include_bytes!("../spv/rope.spv")),
    ("softmax",                   include_bytes!("../spv/softmax.spv")),
    ("softmax_last_dim_backward", include_bytes!("../spv/softmax_last_dim_backward.spv")),
    ("layer_norm_last_dim_backward", include_bytes!("../spv/layer_norm_last_dim_backward.spv")),
    ("strided_copy",              include_bytes!("../spv/strided_copy.spv")),
    ("unary",                     include_bytes!("../spv/unary.spv")),
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
/// Element-wise binary ops (6 ops via uniform selector).
pub const BINARY: &str = "binary";
/// Affine transform: y = x * mul + add.
pub const AFFINE: &str = "affine";
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
/// GLSL gemv (M == 1 matmul specialization), all-f32.
pub const MATVEC_GLSL: &str = "matvec";
/// GLSL gemv (M == 1) with bf16 weight matrix (B), f32 activations
/// (A) and f32 output (C). Decode-phase path for bf16-quantized LLM
/// weights on GPU.
pub const MATVEC_BF16_B_GLSL: &str = "matvec_bf16_b";
/// Fused softmax along the last dimension.
pub const SOFTMAX: &str = "softmax";
/// Fused softmax backward: dx = y * (g - dot(y, g)).
pub const SOFTMAX_LAST_DIM_BACKWARD: &str = "softmax_last_dim_backward";
/// Fused layer-norm backward (4 reductions: sum_x, sum_x², sum_g, sum_gx).
pub const LAYER_NORM_LAST_DIM_BACKWARD: &str = "layer_norm_last_dim_backward";
/// Parallel reduction over all elements.
pub const REDUCE: &str = "reduce";
/// Per-row reduction along the last dimension.
pub const REDUCE_LAST_DIM: &str = "reduce_last_dim";
/// Fused root-mean-square normalization along the last dimension.
pub const RMS_NORM_LAST_DIM: &str = "rms_norm_last_dim";
/// Fused RMSNorm backward (grad_x from x + upstream).
pub const RMS_NORM_LAST_DIM_BACKWARD: &str = "rms_norm_last_dim_backward";
/// Strided copy (permute / broadcast / concat / slice).
pub const STRIDED_COPY: &str = "strided_copy";
/// Row-wise gather along a specified dim (embedding lookup).
pub const INDEX_SELECT: &str = "index_select";
/// In-place scaled accumulate (`dst += src * scale`).
pub const ADD_ASSIGN_SCALED: &str = "add_assign_scaled";
/// Fused rotary position embedding.
pub const ROPE: &str = "rope";
/// Single-dispatch concat along an arbitrary dim.
pub const CONCAT_ALONG_DIM: &str = "concat_along_dim";
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
