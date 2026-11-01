// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compute pipeline management for the precompiled SPIR-V shaders
//! shipped in `fuel-graph-executor`.

use std::sync::Mutex;
use std::sync::OnceLock;
use vulkane::safe::*;

use crate::OptionalFeatureSupport;
use crate::spirv_caps::{KernelUnavailable, required_features};

/// Shader-registry contents, lazily materialized as `&'static
/// [ShaderSource]` so it satisfies `ShaderRegistry::with_embedded`'s
/// `'static` bound. Built once on first access from
/// `fuel_vulkan_kernels::EMBEDDED`.
fn embedded_shader_sources() -> &'static [ShaderSource] {
    static SOURCES: OnceLock<Vec<ShaderSource>> = OnceLock::new();
    SOURCES
        .get_or_init(|| {
            fuel_vulkan_kernels::EMBEDDED
                .iter()
                .map(|(name, spv)| ShaderSource { name, spv })
                .collect()
        })
        .as_slice()
}

/// Loads shader modules, skipping any that need an optional device feature the device
/// lacks. The check reads the module's own `OpCapability` words, so no hand-kept table
/// can drift; a skipped kernel is recorded in `unavailable`.
struct Gate<'a> {
    registry: &'a ShaderRegistry,
    device: &'a Device,
    support: OptionalFeatureSupport,
    unavailable: Vec<KernelUnavailable>,
}

impl Gate<'_> {
    fn module(&mut self, name: &'static str) -> Result<Option<ShaderModule>> {
        let bytes = self.registry.load(name)?;
        if let Some(missing_feature) = self.support.first_missing(&required_features(&bytes)) {
            self.unavailable.push(KernelUnavailable {
                kernel: name,
                missing_feature,
            });
            return Ok(None);
        }
        Ok(Some(ShaderModule::from_spirv_bytes(self.device, &bytes)?))
    }

    fn pipeline(
        &mut self,
        name: &'static str,
        layout: &PipelineLayout,
    ) -> Result<Option<ComputePipeline>> {
        match self.module(name)? {
            Some(m) => Ok(Some(ComputePipeline::new(self.device, layout, &m, "main")?)),
            None => Ok(None),
        }
    }
}

/// Construct Fuel's Vulkan shader registry — the embedded byte
/// table from `fuel-graph-executor` plus the dev-time disk override
/// at `FUEL_SHADER_OVERRIDE_DIR`.
fn shader_registry() -> ShaderRegistry {
    ShaderRegistry::new()
        .with_embedded(embedded_shader_sources())
        .with_env_override(fuel_vulkan_kernels::OVERRIDE_ENV)
}

/// The five descriptor-set layouts the kernels share, by storage-buffer count (each also
/// carries one uniform buffer, bound last).
struct Layouts {
    layout_2s1u: DescriptorSetLayout,
    layout_3s1u: DescriptorSetLayout,
    layout_4s1u: DescriptorSetLayout,
    layout_5s1u: DescriptorSetLayout,
    layout_6s1u: DescriptorSetLayout,
}

impl Layouts {
    fn new(device: &Device) -> Result<Self> {
        let with_storage = |storage: u32| -> Result<DescriptorSetLayout> {
            let mut bindings: Vec<_> = (0..storage).map(storage_binding).collect();
            bindings.push(uniform_binding(storage));
            DescriptorSetLayout::new(device, &bindings)
        };
        Ok(Self {
            layout_2s1u: with_storage(2)?,
            layout_3s1u: with_storage(3)?,
            layout_4s1u: with_storage(4)?,
            layout_5s1u: with_storage(5)?,
            layout_6s1u: with_storage(6)?,
        })
    }
}

/// Builds each row of the [`pipelines!`] table: one method per row kind, each returning
/// that row's `(pipeline, layout)` pair.
struct Build<'a> {
    gate: Gate<'a>,
    device: &'a Device,
    coop_shape_ok: bool,
}

impl Build<'_> {
    /// A kernel every device runs: its shader is loaded unconditionally.
    fn required(
        &mut self,
        shader: &'static str,
        set: &DescriptorSetLayout,
    ) -> Result<(ComputePipeline, PipelineLayout)> {
        let layout = PipelineLayout::new(self.device, &[set])?;
        let module = self.gate.registry.load_module(self.device, shader)?;
        let pipeline = ComputePipeline::new(self.device, &layout, &module, "main")?;
        Ok((pipeline, layout))
    }

    /// A kernel whose shader may need an optional device feature: `None` (and a
    /// `KernelUnavailable` record) when the device lacks it.
    fn gated(
        &mut self,
        shader: &'static str,
        set: &DescriptorSetLayout,
    ) -> Result<(Option<ComputePipeline>, PipelineLayout)> {
        let layout = PipelineLayout::new(self.device, &[set])?;
        let pipeline = self.gate.pipeline(shader, &layout)?;
        Ok((pipeline, layout))
    }

    /// A cooperative-matrix kernel: absent (both halves `None`) unless the device
    /// advertises the exact tile, then gated like [`Build::gated`].
    fn coop(
        &mut self,
        shader: &'static str,
        set: &DescriptorSetLayout,
    ) -> Result<(Option<ComputePipeline>, Option<PipelineLayout>)> {
        if !self.coop_shape_ok {
            return Ok((None, None));
        }
        let (pipeline, layout) = self.gated(shader, set)?;
        Ok((pipeline, Some(layout)))
    }
}

macro_rules! pipe_ty {
    (required) => { ComputePipeline };
    ($other:ident) => { Option<ComputePipeline> };
}

macro_rules! layout_ty {
    (coop) => { Option<PipelineLayout> };
    ($other:ident) => { PipelineLayout };
}

/// Declares `Pipelines` and `Pipelines::new` from one table. A row is
/// `pipeline_field, layout_field, SHADER, descriptor_layout, kind;` where `kind` is a
/// [`Build`] method (`required`, `gated` or `coop`). Adding a kernel is one row.
macro_rules! pipelines {
    ($($pipe:ident, $lay:ident, $shader:ident, $set:ident, $kind:ident;)*) => {
        /// All pre-compiled compute pipelines.
        pub struct Pipelines {
            /// Kernels whose shader needs a device feature this device lacks; their pipeline
            /// fields are `None`. Use [`Pipelines::pipeline`] to reach one with a typed error.
            pub unavailable: Vec<KernelUnavailable>,
            /// 2 storage + 1 uniform (unary, affine, softmax, reduce).
            pub layout_2s1u: DescriptorSetLayout,
            /// 3 storage + 1 uniform (binary, matmul).
            pub layout_3s1u: DescriptorSetLayout,
            /// 4 storage + 1 uniform (rope: x, cos, sin, out, params).
            pub layout_4s1u: DescriptorSetLayout,
            /// 5 storage buffers + 1 uniform — used by flash_attention
            /// (q, k, v, alibi, o + params).
            pub layout_5s1u: DescriptorSetLayout,
            pub layout_6s1u: DescriptorSetLayout,

            $(
                pub $pipe: pipe_ty!($kind),
                pub $lay: layout_ty!($kind),
            )*

            /// Active descriptor pool — the one new allocations come from.
            /// `Mutex` (not `RefCell`) so `Pipelines: Send + Sync` and the
            /// owning `VulkanBackend` can flow through `Arc<VulkanBackend>`
            /// in the pipelined-executor binding-table dispatch (V.1 of
            /// the Vulkan catch-up).
            pub desc_pool: Mutex<DescriptorPool>,

            /// Pools that have been retired (filled up, replaced by a fresh
            /// one) but whose descriptors may still be referenced by
            /// in-flight command buffers on the GPU. We MUST keep these
            /// alive until the GPU is confirmed idle; otherwise
            /// `vkDestroyDescriptorPool` would invalidate handles the GPU
            /// is still reading → `ERROR_DEVICE_LOST`.
            ///
            /// `vulkane::DescriptorSet` holds no back-reference to its
            /// parent pool (confirmed by reading vulkane 0.4.2 source), so
            /// Rust-side Drop ordering won't save us; we have to explicitly
            /// retire the pools and only drop them after a sync point.
            ///
            /// Cleared by `VulkanBackend::drain_recorder` which runs after
            /// the D2H fence has signaled.
            pub retired_desc_pools: Mutex<Vec<DescriptorPool>>,

            pub device: Device,
        }
        impl Pipelines {
            /// `coop_shape_ok` means the device advertises the exact 16x16x16
            /// f16xf16->f32 cooperative-matrix tile the `matmul_coop*` shaders need
            /// (see `CoopMatrixShape::is_matmul_coop_tile`), NOT merely that
            /// `VK_KHR_cooperative_matrix` is present. The coop pipelines below are
            /// built eagerly with `?`, so gating them on extension-presence alone would
            /// fail backend init on a device that has the extension but not this tile.
            pub fn new(
                device: &Device,
                coop_shape_ok: bool,
                support: OptionalFeatureSupport,
            ) -> Result<Self> {
                let layouts = Layouts::new(device)?;

                let desc_pool = Mutex::new(make_desc_pool(device)?);

                // One registry for every shader: disk override, then embedded fallback.
                let registry = shader_registry();
                let mut build = Build {
                    gate: Gate {
                        registry: &registry,
                        device,
                        support,
                        unavailable: Vec::new(),
                    },
                    device,
                    coop_shape_ok,
                };
                $(
                    let ($pipe, $lay) = build.$kind(fuel_vulkan_kernels::$shader, &layouts.$set)?;
                )*
                let unavailable = build.gate.unavailable;

                Ok(Self {
                    unavailable,
                    layout_2s1u: layouts.layout_2s1u,
                    layout_3s1u: layouts.layout_3s1u,
                    layout_4s1u: layouts.layout_4s1u,
                    layout_5s1u: layouts.layout_5s1u,
                    layout_6s1u: layouts.layout_6s1u,
                    $($pipe, $lay,)*
                    desc_pool,
                    retired_desc_pools: Mutex::new(Vec::new()),
                    device: device.clone(),
                })
            }
        }
    };
}

pipelines! {
    unary_pipeline, unary_layout, UNARY, layout_2s1u, required;
    unary_f16_pipeline, unary_f16_layout, UNARY_F16, layout_2s1u, gated;
    unary_f64_pipeline, unary_f64_layout, UNARY_F64, layout_2s1u, required;
    unary_bf16_pipeline, unary_bf16_layout, UNARY_BF16, layout_2s1u, required;
    binary_pipeline, binary_layout, BINARY, layout_3s1u, required;
    binary_f16_pipeline, binary_f16_layout, BINARY_F16, layout_3s1u, gated;
    binary_f64_pipeline, binary_f64_layout, BINARY_F64, layout_3s1u, required;
    binary_bf16_pipeline, binary_bf16_layout, BINARY_BF16, layout_3s1u, required;
    affine_pipeline, affine_layout, AFFINE, layout_2s1u, required;
    affine_f64_pipeline, affine_f64_layout, AFFINE_F64, layout_2s1u, required;
    affine_f16_pipeline, affine_f16_layout, AFFINE_F16, layout_2s1u, gated;
    affine_bf16_pipeline, affine_bf16_layout, AFFINE_BF16, layout_2s1u, required;
    clamp_pipeline, clamp_layout, CLAMP, layout_2s1u, required;
    powi_pipeline, powi_layout, POWI, layout_2s1u, required;
    cast_f32_to_f16_pipeline, cast_f32_to_f16_layout, CAST_F32_TO_F16, layout_2s1u, gated;
    cast_f16_to_f32_pipeline, cast_f16_to_f32_layout, CAST_F16_TO_F32, layout_2s1u, gated;
    cast_f32_to_bf16_pipeline, cast_f32_to_bf16_layout, CAST_F32_TO_BF16, layout_2s1u, required;
    cast_bf16_to_f32_pipeline, cast_bf16_to_f32_layout, CAST_BF16_TO_F32, layout_2s1u, required;
    cast_f32_to_f8e4m3_pipeline, cast_f32_to_f8e4m3_layout, CAST_F32_TO_F8E4M3, layout_2s1u, required;
    cast_f8e4m3_to_f32_pipeline, cast_f8e4m3_to_f32_layout, CAST_F8E4M3_TO_F32, layout_2s1u, required;
    cast_f16_to_f8e4m3_pipeline, cast_f16_to_f8e4m3_layout, CAST_F16_TO_F8E4M3, layout_2s1u, gated;
    cast_f8e4m3_to_f16_pipeline, cast_f8e4m3_to_f16_layout, CAST_F8E4M3_TO_F16, layout_2s1u, gated;
    cast_bf16_to_f8e4m3_pipeline, cast_bf16_to_f8e4m3_layout, CAST_BF16_TO_F8E4M3, layout_2s1u, required;
    cast_f8e4m3_to_bf16_pipeline, cast_f8e4m3_to_bf16_layout, CAST_F8E4M3_TO_BF16, layout_2s1u, required;
    write_slice_b1_pipeline, write_slice_b1_layout, WRITE_SLICE_B1, layout_3s1u, required;
    write_slice_b2_pipeline, write_slice_b2_layout, WRITE_SLICE_B2, layout_3s1u, required;
    write_slice_b4_pipeline, write_slice_b4_layout, WRITE_SLICE_B4, layout_3s1u, required;
    pad_const_b1_pipeline, pad_const_b1_layout, PAD_CONST_B1, layout_3s1u, required;
    pad_const_b2_pipeline, pad_const_b2_layout, PAD_CONST_B2, layout_3s1u, required;
    pad_const_b4_pipeline, pad_const_b4_layout, PAD_CONST_B4, layout_3s1u, required;
    pad_const_b8_pipeline, pad_const_b8_layout, PAD_CONST_B8, layout_3s1u, required;
    pad_reflect_b1_pipeline, pad_reflect_b1_layout, PAD_REFLECT_B1, layout_3s1u, required;
    pad_reflect_b2_pipeline, pad_reflect_b2_layout, PAD_REFLECT_B2, layout_3s1u, required;
    pad_reflect_b4_pipeline, pad_reflect_b4_layout, PAD_REFLECT_B4, layout_3s1u, required;
    pad_reflect_b8_pipeline, pad_reflect_b8_layout, PAD_REFLECT_B8, layout_3s1u, required;
    pad_replicate_b1_pipeline, pad_replicate_b1_layout, PAD_REPLICATE_B1, layout_3s1u, required;
    pad_replicate_b2_pipeline, pad_replicate_b2_layout, PAD_REPLICATE_B2, layout_3s1u, required;
    pad_replicate_b4_pipeline, pad_replicate_b4_layout, PAD_REPLICATE_B4, layout_3s1u, required;
    pad_replicate_b8_pipeline, pad_replicate_b8_layout, PAD_REPLICATE_B8, layout_3s1u, required;
    pad_backward_const_b1_pipeline, pad_backward_const_b1_layout, PAD_BACKWARD_CONST_B1, layout_3s1u, required;
    pad_backward_const_b2_pipeline, pad_backward_const_b2_layout, PAD_BACKWARD_CONST_B2, layout_3s1u, required;
    pad_backward_const_b4_pipeline, pad_backward_const_b4_layout, PAD_BACKWARD_CONST_B4, layout_3s1u, required;
    pad_backward_const_b8_pipeline, pad_backward_const_b8_layout, PAD_BACKWARD_CONST_B8, layout_3s1u, required;
    pad_backward_reflect_f32_pipeline, pad_backward_reflect_f32_layout, PAD_BACKWARD_REFLECT_F32, layout_3s1u, required;
    pad_backward_replicate_f32_pipeline, pad_backward_replicate_f32_layout, PAD_BACKWARD_REPLICATE_F32, layout_3s1u, required;
    pad_backward_reflect_f64_pipeline, pad_backward_reflect_f64_layout, PAD_BACKWARD_REFLECT_F64, layout_3s1u, gated;
    pad_backward_replicate_f64_pipeline, pad_backward_replicate_f64_layout, PAD_BACKWARD_REPLICATE_F64, layout_3s1u, gated;
    pad_backward_reflect_bf16_pipeline, pad_backward_reflect_bf16_layout, PAD_BACKWARD_REFLECT_BF16, layout_3s1u, required;
    pad_backward_replicate_bf16_pipeline, pad_backward_replicate_bf16_layout, PAD_BACKWARD_REPLICATE_BF16, layout_3s1u, required;
    pad_backward_reflect_f16_pipeline, pad_backward_reflect_f16_layout, PAD_BACKWARD_REFLECT_F16, layout_3s1u, gated;
    pad_backward_replicate_f16_pipeline, pad_backward_replicate_f16_layout, PAD_BACKWARD_REPLICATE_F16, layout_3s1u, gated;
    masked_fill_b1_pipeline, masked_fill_b1_layout, MASKED_FILL_B1, layout_3s1u, required;
    masked_fill_b2_pipeline, masked_fill_b2_layout, MASKED_FILL_B2, layout_3s1u, required;
    masked_fill_b4_pipeline, masked_fill_b4_layout, MASKED_FILL_B4, layout_3s1u, required;
    masked_fill_b8_pipeline, masked_fill_b8_layout, MASKED_FILL_B8, layout_3s1u, required;
    gather_b1_pipeline, gather_b1_layout, GATHER_B1, layout_4s1u, required;
    gather_b2_pipeline, gather_b2_layout, GATHER_B2, layout_4s1u, required;
    gather_b4_pipeline, gather_b4_layout, GATHER_B4, layout_4s1u, required;
    gather_b8_pipeline, gather_b8_layout, GATHER_B8, layout_4s1u, required;
    write_slice_b8_pipeline, write_slice_b8_layout, WRITE_SLICE_B8, layout_3s1u, required;
    strided_copy_signed_b2_pipeline, strided_copy_signed_b2_layout, STRIDED_COPY_SIGNED_B2, layout_3s1u, gated;
    strided_copy_signed_b4_pipeline, strided_copy_signed_b4_layout, STRIDED_COPY_SIGNED_B4, layout_3s1u, required;
    strided_copy_signed_b8_pipeline, strided_copy_signed_b8_layout, STRIDED_COPY_SIGNED_B8, layout_3s1u, required;
    triu_b2_pipeline, triu_b2_layout, TRIU_B2, layout_2s1u, required;
    triu_b4_pipeline, triu_b4_layout, TRIU_B4, layout_2s1u, required;
    triu_b8_pipeline, triu_b8_layout, TRIU_B8, layout_2s1u, required;
    tril_b2_pipeline, tril_b2_layout, TRIL_B2, layout_2s1u, required;
    tril_b4_pipeline, tril_b4_layout, TRIL_B4, layout_2s1u, required;
    tril_b8_pipeline, tril_b8_layout, TRIL_B8, layout_2s1u, required;
    flip_b2_pipeline, flip_b2_layout, FLIP_B2, layout_2s1u, gated;
    flip_b4_pipeline, flip_b4_layout, FLIP_B4, layout_2s1u, required;
    flip_b8_pipeline, flip_b8_layout, FLIP_B8, layout_2s1u, required;
    roll_b2_pipeline, roll_b2_layout, ROLL_B2, layout_2s1u, gated;
    roll_b4_pipeline, roll_b4_layout, ROLL_B4, layout_2s1u, required;
    roll_b8_pipeline, roll_b8_layout, ROLL_B8, layout_2s1u, required;
    cumsum_f32_pipeline, cumsum_f32_layout, CUMSUM_F32, layout_2s1u, required;
    cumsum_f64_pipeline, cumsum_f64_layout, CUMSUM_F64, layout_2s1u, required;
    cumsum_f16_pipeline, cumsum_f16_layout, CUMSUM_F16, layout_2s1u, gated;
    cumsum_bf16_pipeline, cumsum_bf16_layout, CUMSUM_BF16, layout_2s1u, gated;
    matmul_pipeline, matmul_layout, MATMUL, layout_3s1u, required;
    matmul_tiled_pipeline, matmul_tiled_layout, MATMUL_TILED_GLSL, layout_3s1u, required;
    matvec_pipeline, matvec_layout, MATVEC_GLSL, layout_3s1u, required;
    matvec_cols_pipeline, matvec_cols_layout, MATVEC_COLS_GLSL, layout_3s1u, required;
    matvec_bf16_b_pipeline, matvec_bf16_b_layout, MATVEC_BF16_B_GLSL, layout_3s1u, required;
    matmul_tiled_bf16_b_pipeline, matmul_tiled_bf16_b_layout, MATMUL_TILED_BF16_B_GLSL, layout_3s1u, required;
    matmul_coop_pipeline, matmul_coop_layout, MATMUL_COOP, layout_3s1u, coop;
    matmul_coop_bf16_bf16_pipeline, matmul_coop_bf16_bf16_layout, MATMUL_COOP_BF16_BF16, layout_3s1u, coop;
    matmul_coop_f16_f16_pipeline, matmul_coop_f16_f16_layout, MATMUL_COOP_F16_F16, layout_3s1u, coop;
    matmul_coop_bf16_bf16_bf16_pipeline, matmul_coop_bf16_bf16_bf16_layout, MATMUL_COOP_BF16_BF16_BF16, layout_3s1u, coop;
    matmul_small_bf16_bf16_f32_pipeline, matmul_small_bf16_bf16_f32_layout, MATMUL_SMALL_BF16_BF16_F32, layout_3s1u, gated;
    matmul_small_bf16_bf16_bf16_pipeline, matmul_small_bf16_bf16_bf16_layout, MATMUL_SMALL_BF16_BF16_BF16, layout_3s1u, gated;
    matmul_small_f16_f16_f32_pipeline, matmul_small_f16_f16_f32_layout, MATMUL_SMALL_F16_F16_F32, layout_3s1u, required;
    matmul_small_f16_f16_f16_pipeline, matmul_small_f16_f16_f16_layout, MATMUL_SMALL_F16_F16_F16, layout_3s1u, required;
    flash_attn_f32_pipeline, flash_attn_f32_layout, FLASH_ATTN_F32, layout_5s1u, required;
    flash_attn_split_f32_pipeline, flash_attn_split_f32_layout, FLASH_ATTN_SPLIT_F32, layout_5s1u, required;
    flash_attn_combine_f32_pipeline, flash_attn_combine_f32_layout, FLASH_ATTN_COMBINE_F32, layout_2s1u, required;
    flash_attn_split_bf16_pipeline, flash_attn_split_bf16_layout, FLASH_ATTN_SPLIT_BF16, layout_5s1u, gated;
    flash_attn_combine_bf16_pipeline, flash_attn_combine_bf16_layout, FLASH_ATTN_COMBINE_BF16, layout_2s1u, gated;
    flash_attn_split_f16_pipeline, flash_attn_split_f16_layout, FLASH_ATTN_SPLIT_F16, layout_5s1u, required;
    flash_attn_combine_f16_pipeline, flash_attn_combine_f16_layout, FLASH_ATTN_COMBINE_F16, layout_2s1u, required;
    flash_attn_bf16_pipeline, flash_attn_bf16_layout, FLASH_ATTN_BF16, layout_5s1u, gated;
    flash_attn_f16_pipeline, flash_attn_f16_layout, FLASH_ATTN_F16, layout_5s1u, required;
    flash_attn_backward_q_f32_pipeline, flash_attn_backward_q_f32_layout, FLASH_ATTN_BACKWARD_Q_F32, layout_6s1u, required;
    flash_attn_backward_k_f32_pipeline, flash_attn_backward_k_f32_layout, FLASH_ATTN_BACKWARD_K_F32, layout_6s1u, required;
    flash_attn_backward_v_f32_pipeline, flash_attn_backward_v_f32_layout, FLASH_ATTN_BACKWARD_V_F32, layout_6s1u, required;
    matmul_coop_f16_f16_f16_pipeline, matmul_coop_f16_f16_f16_layout, MATMUL_COOP_F16_F16_F16, layout_3s1u, coop;
    softmax_pipeline, softmax_layout, SOFTMAX, layout_2s1u, required;
    softmax_f16_pipeline, softmax_f16_layout, SOFTMAX_F16, layout_2s1u, gated;
    softmax_bf16_pipeline, softmax_bf16_layout, SOFTMAX_BF16, layout_2s1u, required;
    softmax_f64_pipeline, softmax_f64_layout, SOFTMAX_F64, layout_2s1u, required;
    reduce_pipeline, reduce_layout, REDUCE, layout_2s1u, required;
    reduce_f16_pipeline, reduce_f16_layout, REDUCE_F16, layout_2s1u, gated;
    reduce_bf16_pipeline, reduce_bf16_layout, REDUCE_BF16, layout_2s1u, required;
    reduce_f64_pipeline, reduce_f64_layout, REDUCE_F64, layout_2s1u, required;
    cast_f32_to_f64_pipeline, cast_f32_to_f64_layout, CAST_F32_TO_F64, layout_2s1u, required;
    cast_f64_to_f32_pipeline, cast_f64_to_f32_layout, CAST_F64_TO_F32, layout_2s1u, required;
    reduce_last_dim_pipeline, reduce_last_dim_layout, REDUCE_LAST_DIM, layout_2s1u, required;
    arg_reduce_last_dim_f32_pipeline, arg_reduce_last_dim_f32_layout, ARG_REDUCE_LAST_DIM_F32, layout_2s1u, required;
    scatter_add_f32_pipeline, scatter_add_f32_layout, SCATTER_ADD_F32, layout_4s1u, required;
    scatter_add_f64_pipeline, scatter_add_f64_layout, SCATTER_ADD_F64, layout_4s1u, gated;
    scatter_add_bf16_pipeline, scatter_add_bf16_layout, SCATTER_ADD_BF16, layout_4s1u, required;
    scatter_add_f16_pipeline, scatter_add_f16_layout, SCATTER_ADD_F16, layout_4s1u, gated;
    arg_reduce_last_dim_f16_pipeline, arg_reduce_last_dim_f16_layout, ARG_REDUCE_LAST_DIM_F16, layout_2s1u, gated;
    arg_reduce_last_dim_bf16_pipeline, arg_reduce_last_dim_bf16_layout, ARG_REDUCE_LAST_DIM_BF16, layout_2s1u, required;
    arg_reduce_last_dim_f64_pipeline, arg_reduce_last_dim_f64_layout, ARG_REDUCE_LAST_DIM_F64, layout_2s1u, required;
    arg_reduce_any_dim_f32_pipeline, arg_reduce_any_dim_f32_layout, ARG_REDUCE_ANY_DIM_F32, layout_2s1u, required;
    arg_reduce_any_dim_f64_pipeline, arg_reduce_any_dim_f64_layout, ARG_REDUCE_ANY_DIM_F64, layout_2s1u, required;
    arg_reduce_any_dim_bf16_pipeline, arg_reduce_any_dim_bf16_layout, ARG_REDUCE_ANY_DIM_BF16, layout_2s1u, required;
    arg_reduce_any_dim_f16_pipeline, arg_reduce_any_dim_f16_layout, ARG_REDUCE_ANY_DIM_F16, layout_2s1u, gated;
    index_add_f32_pipeline, index_add_f32_layout, INDEX_ADD_F32, layout_3s1u, required;
    index_add_f64_pipeline, index_add_f64_layout, INDEX_ADD_F64, layout_3s1u, gated;
    index_add_bf16_pipeline, index_add_bf16_layout, INDEX_ADD_BF16, layout_3s1u, required;
    index_add_f16_pipeline, index_add_f16_layout, INDEX_ADD_F16, layout_3s1u, gated;
    reduce_last_dim_f16_pipeline, reduce_last_dim_f16_layout, REDUCE_LAST_DIM_F16, layout_2s1u, gated;
    reduce_last_dim_bf16_pipeline, reduce_last_dim_bf16_layout, REDUCE_LAST_DIM_BF16, layout_2s1u, required;
    reduce_last_dim_f64_pipeline, reduce_last_dim_f64_layout, REDUCE_LAST_DIM_F64, layout_2s1u, required;
    rms_norm_last_dim_pipeline, rms_norm_last_dim_layout, RMS_NORM_LAST_DIM, layout_2s1u, required;
    rms_norm_last_dim_f16_pipeline, rms_norm_last_dim_f16_layout, RMS_NORM_LAST_DIM_F16, layout_2s1u, gated;
    rms_norm_last_dim_bf16_pipeline, rms_norm_last_dim_bf16_layout, RMS_NORM_LAST_DIM_BF16, layout_2s1u, required;
    rms_norm_last_dim_f64_pipeline, rms_norm_last_dim_f64_layout, RMS_NORM_LAST_DIM_F64, layout_2s1u, required;
    rms_norm_last_dim_backward_pipeline, rms_norm_last_dim_backward_layout, RMS_NORM_LAST_DIM_BACKWARD, layout_3s1u, required;
    softmax_last_dim_backward_pipeline, softmax_last_dim_backward_layout, SOFTMAX_LAST_DIM_BACKWARD, layout_3s1u, required;
    softmax_last_dim_backward_f16_pipeline, softmax_last_dim_backward_f16_layout, SOFTMAX_LAST_DIM_BACKWARD_F16, layout_3s1u, gated;
    softmax_last_dim_backward_bf16_pipeline, softmax_last_dim_backward_bf16_layout, SOFTMAX_LAST_DIM_BACKWARD_BF16, layout_3s1u, required;
    softmax_last_dim_backward_f64_pipeline, softmax_last_dim_backward_f64_layout, SOFTMAX_LAST_DIM_BACKWARD_F64, layout_3s1u, required;
    layer_norm_last_dim_backward_pipeline, layer_norm_last_dim_backward_layout, LAYER_NORM_LAST_DIM_BACKWARD, layout_3s1u, required;
    layer_norm_last_dim_backward_f16_pipeline, layer_norm_last_dim_backward_f16_layout, LAYER_NORM_LAST_DIM_BACKWARD_F16, layout_3s1u, gated;
    layer_norm_last_dim_backward_bf16_pipeline, layer_norm_last_dim_backward_bf16_layout, LAYER_NORM_LAST_DIM_BACKWARD_BF16, layout_3s1u, required;
    layer_norm_last_dim_backward_f64_pipeline, layer_norm_last_dim_backward_f64_layout, LAYER_NORM_LAST_DIM_BACKWARD_F64, layout_3s1u, required;
    layer_norm_last_dim_pipeline, layer_norm_last_dim_layout, LAYER_NORM_LAST_DIM, layout_2s1u, required;
    layer_norm_last_dim_f16_pipeline, layer_norm_last_dim_f16_layout, LAYER_NORM_LAST_DIM_F16, layout_2s1u, gated;
    layer_norm_last_dim_bf16_pipeline, layer_norm_last_dim_bf16_layout, LAYER_NORM_LAST_DIM_BF16, layout_2s1u, required;
    layer_norm_last_dim_f64_pipeline, layer_norm_last_dim_f64_layout, LAYER_NORM_LAST_DIM_F64, layout_2s1u, required;
    strided_copy_pipeline, strided_copy_layout, STRIDED_COPY, layout_3s1u, required;
    index_select_pipeline, index_select_layout, INDEX_SELECT, layout_3s1u, required;
    index_select_f16_pipeline, index_select_f16_layout, INDEX_SELECT_F16, layout_3s1u, gated;
    index_select_bf16_pipeline, index_select_bf16_layout, INDEX_SELECT_BF16, layout_3s1u, required;
    index_select_f64_pipeline, index_select_f64_layout, INDEX_SELECT_F64, layout_3s1u, required;
    add_assign_scaled_pipeline, add_assign_scaled_layout, ADD_ASSIGN_SCALED, layout_2s1u, required;
    rope_pipeline, rope_layout, ROPE, layout_4s1u, required;
    rope_f16_pipeline, rope_f16_layout, ROPE_F16, layout_4s1u, gated;
    rope_bf16_pipeline, rope_bf16_layout, ROPE_BF16, layout_4s1u, required;
    rope_f64_pipeline, rope_f64_layout, ROPE_F64, layout_4s1u, required;
    concat_along_dim_pipeline, concat_along_dim_layout, CONCAT_ALONG_DIM, layout_3s1u, required;
    concat_along_dim_f16_pipeline, concat_along_dim_f16_layout, CONCAT_ALONG_DIM_F16, layout_3s1u, gated;
    concat_along_dim_bf16_pipeline, concat_along_dim_bf16_layout, CONCAT_ALONG_DIM_BF16, layout_3s1u, required;
    concat_along_dim_f64_pipeline, concat_along_dim_f64_layout, CONCAT_ALONG_DIM_F64, layout_3s1u, required;
    conv2d_im2col_pipeline, conv2d_im2col_layout, CONV2D_IM2COL, layout_2s1u, required;
    conv2d_im2col_bf16_pipeline, conv2d_im2col_bf16_layout, CONV2D_IM2COL_BF16, layout_2s1u, gated;
    flash_attention_pipeline, flash_attention_layout, FLASH_ATTENTION, layout_5s1u, required;
    dequant_q4_0_pipeline, dequant_q4_0_layout, DEQUANT_Q4_0, layout_2s1u, gated;
    dequant_q8_0_pipeline, dequant_q8_0_layout, DEQUANT_Q8_0, layout_2s1u, gated;
    dequant_q4_km_pipeline, dequant_q4_km_layout, DEQUANT_Q4_KM, layout_2s1u, gated;
    qmatvec_q4_0_pipeline, qmatvec_q4_0_layout, QMATVEC_Q4_0, layout_3s1u, gated;
    qmatvec_q8_0_pipeline, qmatvec_q8_0_layout, QMATVEC_Q8_0, layout_3s1u, gated;
    qmatvec_q6_k_pipeline, qmatvec_q6_k_layout, QMATVEC_Q6_K, layout_3s1u, gated;
    matmul_q4_0_tiled_pipeline, matmul_q4_0_tiled_layout, MATMUL_Q4_0_TILED, layout_3s1u, gated;
    quantize_q8_0_pipeline, quantize_q8_0_layout, QUANTIZE_Q8_0, layout_2s1u, gated;
}

impl Pipelines {
    pub fn allocate_desc(&self, layout: &DescriptorSetLayout) -> Result<DescriptorSet> {
        let _span = tracing::debug_span!("vk_alloc_desc").entered();
        // Try allocating from the current pool.
        {
            let pool = self.desc_pool.lock().expect("desc_pool poisoned");
            match pool.allocate(layout) {
                Ok(d) => return Ok(d),
                Err(Error::Vk(code)) if is_pool_oom(code) => { /* fall through to retire + recreate */
                }
                Err(e) => return Err(e),
            }
        }
        // Pool full. Retire the current pool (do NOT drop — its
        // descriptors are still being used by in-flight GPU work)
        // and swap in a fresh one. Retired pools get destroyed later
        // by `retire_pools_post_drain()` after the GPU is confirmed
        // idle.
        let _r = tracing::info_span!("vk_alloc_desc_retire_pool").entered();
        let fresh = make_desc_pool(&self.device)?;
        let old = std::mem::replace(
            &mut *self.desc_pool.lock().expect("desc_pool poisoned"),
            fresh,
        );
        self.retired_desc_pools
            .lock()
            .expect("retired_desc_pools poisoned")
            .push(old);
        self.desc_pool
            .lock()
            .expect("desc_pool poisoned")
            .allocate(layout)
    }

    /// Drop every retired descriptor pool. Safe to call only AFTER a
    /// sync point that guarantees the GPU is done with every command
    /// buffer ever recorded against those pools. `VulkanBackend`
    /// calls this from `drain_recorder`, which itself runs after the
    /// D2H copy's fence has signaled.
    pub fn retire_pools_post_drain(&self) {
        let mut retired = self
            .retired_desc_pools
            .lock()
            .expect("retired_desc_pools poisoned");
        if !retired.is_empty() {
            let _s = tracing::info_span!("vk_retired_pools_drop", n = retired.len()).entered();
            retired.clear();
        }
    }
}

fn is_pool_oom(code: vulkane::raw::bindings::VkResult) -> bool {
    use vulkane::raw::bindings::VkResult;
    matches!(
        code,
        VkResult::ERROR_OUT_OF_POOL_MEMORY | VkResult::ERROR_FRAGMENTED_POOL
    )
}

fn make_desc_pool(device: &Device) -> Result<DescriptorPool> {
    DescriptorPool::new(
        device,
        4096,
        &[
            DescriptorPoolSize {
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 16384,
            },
            DescriptorPoolSize {
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 4096,
            },
        ],
    )
}

impl Pipelines {
    /// The pipeline for `kernel`, or the typed reason this device does not offer it.
    pub fn pipeline<'a>(
        &self,
        pipeline: &'a Option<ComputePipeline>,
        kernel: &'static str,
    ) -> std::result::Result<&'a ComputePipeline, KernelUnavailable> {
        pipeline.as_ref().ok_or_else(|| {
            self.unavailable
                .iter()
                .find(|u| u.kernel == kernel)
                .cloned()
                .unwrap_or(KernelUnavailable {
                    kernel,
                    missing_feature: "VK_KHR_cooperative_matrix 16x16x16 f16 tile",
                })
        })
    }
}

fn storage_binding(binding: u32) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        binding,
        descriptor_type: DescriptorType::STORAGE_BUFFER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::COMPUTE,
    }
}

fn uniform_binding(binding: u32) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        binding,
        descriptor_type: DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: ShaderStageFlags::COMPUTE,
    }
}
