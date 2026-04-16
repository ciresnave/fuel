//! Precompiled SPIR-V kernels, shared across GPU backends.
//!
//! Shader sources (`.wgsl`, `.glsl`, `.slang`) live in `src/shaders/`.
//! They are compiled to SPIR-V **ahead of time** using
//! `src/shaders/compile.sh` (requires Vulkan SDK + naga-cli) and the
//! resulting `.spv` files are committed to `src/shaders_spirv/`.
//!
//! This module only exposes the embedded byte table and the
//! environment-variable name for the dev-time disk-override
//! mechanism. Backends are responsible for handing these to whatever
//! shader registry their graphics API uses
//! (`vulkane::safe::ShaderRegistry` for Vulkan, future equivalents
//! for other backends). Keeping the registry construction out of
//! this crate avoids coupling a Vulkan-specific dependency into
//! the backend-agnostic graph executor.
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
//! 1. Drop the source file into `src/shaders/` (any of `.wgsl`,
//!    `.glsl`, `.slang`).
//! 2. Run `./compile.sh` from that directory.
//! 3. Add a `(name, include_bytes!(...))` entry to [`EMBEDDED`] below.
//! 4. Rebuild Fuel.

/// All shaders baked into the Fuel binary, as `(name, spirv_bytes)`
/// pairs. Backends populate their own shader registry from this
/// table.
pub static EMBEDDED: &[(&str, &[u8])] = &[
    ("add_assign_scaled",         include_bytes!("shaders_spirv/add_assign_scaled.spv")),
    ("affine",                    include_bytes!("shaders_spirv/affine.spv")),
    ("binary",                    include_bytes!("shaders_spirv/binary.spv")),
    ("concat_along_dim",          include_bytes!("shaders_spirv/concat_along_dim.spv")),
    ("index_select",              include_bytes!("shaders_spirv/index_select.spv")),
    ("matmul",                    include_bytes!("shaders_spirv/matmul.spv")),
    ("matmul_tiled",              include_bytes!("shaders_spirv/matmul_tiled.spv")),
    ("matmul_tiled_bf16_b",       include_bytes!("shaders_spirv/matmul_tiled_bf16_b.spv")),
    ("matvec",                    include_bytes!("shaders_spirv/matvec.spv")),
    ("matvec_bf16_b",             include_bytes!("shaders_spirv/matvec_bf16_b.spv")),
    ("reduce",                    include_bytes!("shaders_spirv/reduce.spv")),
    ("reduce_last_dim",           include_bytes!("shaders_spirv/reduce_last_dim.spv")),
    ("rms_norm_last_dim",         include_bytes!("shaders_spirv/rms_norm_last_dim.spv")),
    ("rms_norm_last_dim_backward", include_bytes!("shaders_spirv/rms_norm_last_dim_backward.spv")),
    ("rope",                      include_bytes!("shaders_spirv/rope.spv")),
    ("softmax",                   include_bytes!("shaders_spirv/softmax.spv")),
    ("strided_copy",              include_bytes!("shaders_spirv/strided_copy.spv")),
    ("unary",                     include_bytes!("shaders_spirv/unary.spv")),
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
/// GLSL gemv (M == 1 matmul specialization), all-f32.
pub const MATVEC_GLSL: &str = "matvec";
/// GLSL gemv (M == 1) with bf16 weight matrix (B), f32 activations
/// (A) and f32 output (C). Decode-phase path for bf16-quantized LLM
/// weights on GPU.
pub const MATVEC_BF16_B_GLSL: &str = "matvec_bf16_b";
/// Fused softmax along the last dimension.
pub const SOFTMAX: &str = "softmax";
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
