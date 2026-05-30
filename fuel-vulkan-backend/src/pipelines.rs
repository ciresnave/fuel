//! Compute pipeline management for the precompiled SPIR-V shaders
//! shipped in `fuel-graph-executor`.

use std::sync::Mutex;
use std::sync::OnceLock;
use vulkane::safe::*;

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

/// Construct Fuel's Vulkan shader registry — the embedded byte
/// table from `fuel-graph-executor` plus the dev-time disk override
/// at `FUEL_SHADER_OVERRIDE_DIR`.
fn shader_registry() -> ShaderRegistry {
    ShaderRegistry::new()
        .with_embedded(embedded_shader_sources())
        .with_env_override(fuel_vulkan_kernels::OVERRIDE_ENV)
}

/// All pre-compiled compute pipelines.
pub struct Pipelines {
    /// 2 storage + 1 uniform (unary, affine, softmax, reduce).
    pub layout_2s1u: DescriptorSetLayout,
    /// 3 storage + 1 uniform (binary, matmul).
    pub layout_3s1u: DescriptorSetLayout,
    /// 4 storage + 1 uniform (rope: x, cos, sin, out, params).
    pub layout_4s1u: DescriptorSetLayout,
    /// 5 storage buffers + 1 uniform — used by flash_attention
    /// (q, k, v, alibi, o + params).
    pub layout_5s1u: DescriptorSetLayout,

    pub unary_pipeline: ComputePipeline,
    pub unary_layout: PipelineLayout,
    pub unary_f16_pipeline: ComputePipeline,
    pub unary_f16_layout: PipelineLayout,
    pub unary_f64_pipeline: ComputePipeline,
    pub unary_f64_layout: PipelineLayout,
    pub unary_bf16_pipeline: ComputePipeline,
    pub unary_bf16_layout: PipelineLayout,
    pub binary_pipeline: ComputePipeline,
    pub binary_layout: PipelineLayout,
    pub binary_f16_pipeline: ComputePipeline,
    pub binary_f16_layout: PipelineLayout,
    pub binary_f64_pipeline: ComputePipeline,
    pub binary_f64_layout: PipelineLayout,
    pub binary_bf16_pipeline: ComputePipeline,
    pub binary_bf16_layout: PipelineLayout,
    pub affine_pipeline: ComputePipeline,
    pub affine_layout: PipelineLayout,
    pub affine_f64_pipeline: ComputePipeline,
    pub affine_f64_layout: PipelineLayout,
    pub affine_f16_pipeline: ComputePipeline,
    pub affine_f16_layout: PipelineLayout,
    pub affine_bf16_pipeline: ComputePipeline,
    pub affine_bf16_layout: PipelineLayout,
    pub clamp_pipeline: ComputePipeline,
    pub clamp_layout: PipelineLayout,
    pub powi_pipeline: ComputePipeline,
    pub powi_layout: PipelineLayout,
    pub cast_f32_to_f16_pipeline: ComputePipeline,
    pub cast_f32_to_f16_layout: PipelineLayout,
    pub cast_f16_to_f32_pipeline: ComputePipeline,
    pub cast_f16_to_f32_layout: PipelineLayout,
    pub cast_f32_to_bf16_pipeline: ComputePipeline,
    pub cast_f32_to_bf16_layout: PipelineLayout,
    pub cast_bf16_to_f32_pipeline: ComputePipeline,
    pub cast_bf16_to_f32_layout: PipelineLayout,
    pub cast_f32_to_f8e4m3_pipeline: ComputePipeline,
    pub cast_f32_to_f8e4m3_layout: PipelineLayout,
    pub cast_f8e4m3_to_f32_pipeline: ComputePipeline,
    pub cast_f8e4m3_to_f32_layout: PipelineLayout,
    pub cast_f16_to_f8e4m3_pipeline: ComputePipeline,
    pub cast_f16_to_f8e4m3_layout: PipelineLayout,
    pub cast_f8e4m3_to_f16_pipeline: ComputePipeline,
    pub cast_f8e4m3_to_f16_layout: PipelineLayout,
    pub cast_bf16_to_f8e4m3_pipeline: ComputePipeline,
    pub cast_bf16_to_f8e4m3_layout: PipelineLayout,
    pub cast_f8e4m3_to_bf16_pipeline: ComputePipeline,
    pub cast_f8e4m3_to_bf16_layout: PipelineLayout,
    pub write_slice_b1_pipeline: ComputePipeline,
    pub write_slice_b1_layout: PipelineLayout,
    pub write_slice_b2_pipeline: ComputePipeline,
    pub write_slice_b2_layout: PipelineLayout,
    pub write_slice_b4_pipeline: ComputePipeline,
    pub write_slice_b4_layout: PipelineLayout,

    pub pad_const_b1_pipeline: ComputePipeline,
    pub pad_const_b1_layout: PipelineLayout,
    pub pad_const_b2_pipeline: ComputePipeline,
    pub pad_const_b2_layout: PipelineLayout,
    pub pad_const_b4_pipeline: ComputePipeline,
    pub pad_const_b4_layout: PipelineLayout,
    pub pad_const_b8_pipeline: ComputePipeline,
    pub pad_const_b8_layout: PipelineLayout,

    pub pad_reflect_b1_pipeline: ComputePipeline,
    pub pad_reflect_b1_layout: PipelineLayout,
    pub pad_reflect_b2_pipeline: ComputePipeline,
    pub pad_reflect_b2_layout: PipelineLayout,
    pub pad_reflect_b4_pipeline: ComputePipeline,
    pub pad_reflect_b4_layout: PipelineLayout,
    pub pad_reflect_b8_pipeline: ComputePipeline,
    pub pad_reflect_b8_layout: PipelineLayout,

    pub pad_replicate_b1_pipeline: ComputePipeline,
    pub pad_replicate_b1_layout: PipelineLayout,
    pub pad_replicate_b2_pipeline: ComputePipeline,
    pub pad_replicate_b2_layout: PipelineLayout,
    pub pad_replicate_b4_pipeline: ComputePipeline,
    pub pad_replicate_b4_layout: PipelineLayout,
    pub pad_replicate_b8_pipeline: ComputePipeline,
    pub pad_replicate_b8_layout: PipelineLayout,

    pub pad_backward_const_b1_pipeline: ComputePipeline,
    pub pad_backward_const_b1_layout: PipelineLayout,
    pub pad_backward_const_b2_pipeline: ComputePipeline,
    pub pad_backward_const_b2_layout: PipelineLayout,
    pub pad_backward_const_b4_pipeline: ComputePipeline,
    pub pad_backward_const_b4_layout: PipelineLayout,
    pub pad_backward_const_b8_pipeline: ComputePipeline,
    pub pad_backward_const_b8_layout: PipelineLayout,

    pub pad_backward_reflect_f32_pipeline: ComputePipeline,
    pub pad_backward_reflect_f32_layout: PipelineLayout,
    pub pad_backward_replicate_f32_pipeline: ComputePipeline,
    pub pad_backward_replicate_f32_layout: PipelineLayout,
    pub pad_backward_reflect_f64_pipeline: ComputePipeline,
    pub pad_backward_reflect_f64_layout: PipelineLayout,
    pub pad_backward_replicate_f64_pipeline: ComputePipeline,
    pub pad_backward_replicate_f64_layout: PipelineLayout,
    pub pad_backward_reflect_bf16_pipeline: ComputePipeline,
    pub pad_backward_reflect_bf16_layout: PipelineLayout,
    pub pad_backward_replicate_bf16_pipeline: ComputePipeline,
    pub pad_backward_replicate_bf16_layout: PipelineLayout,
    pub pad_backward_reflect_f16_pipeline: ComputePipeline,
    pub pad_backward_reflect_f16_layout: PipelineLayout,
    pub pad_backward_replicate_f16_pipeline: ComputePipeline,
    pub pad_backward_replicate_f16_layout: PipelineLayout,

    pub masked_fill_b1_pipeline: ComputePipeline,
    pub masked_fill_b1_layout: PipelineLayout,
    pub masked_fill_b2_pipeline: ComputePipeline,
    pub masked_fill_b2_layout: PipelineLayout,
    pub masked_fill_b4_pipeline: ComputePipeline,
    pub masked_fill_b4_layout: PipelineLayout,
    pub masked_fill_b8_pipeline: ComputePipeline,
    pub masked_fill_b8_layout: PipelineLayout,

    pub gather_b1_pipeline: ComputePipeline,
    pub gather_b1_layout: PipelineLayout,
    pub gather_b2_pipeline: ComputePipeline,
    pub gather_b2_layout: PipelineLayout,
    pub gather_b4_pipeline: ComputePipeline,
    pub gather_b4_layout: PipelineLayout,
    pub gather_b8_pipeline: ComputePipeline,
    pub gather_b8_layout: PipelineLayout,
    pub write_slice_b8_pipeline: ComputePipeline,
    pub write_slice_b8_layout: PipelineLayout,
    pub strided_copy_signed_b2_pipeline: ComputePipeline,
    pub strided_copy_signed_b2_layout: PipelineLayout,
    pub strided_copy_signed_b4_pipeline: ComputePipeline,
    pub strided_copy_signed_b4_layout: PipelineLayout,
    pub strided_copy_signed_b8_pipeline: ComputePipeline,
    pub strided_copy_signed_b8_layout: PipelineLayout,
    pub triu_b2_pipeline: ComputePipeline,
    pub triu_b2_layout: PipelineLayout,
    pub triu_b4_pipeline: ComputePipeline,
    pub triu_b4_layout: PipelineLayout,
    pub triu_b8_pipeline: ComputePipeline,
    pub triu_b8_layout: PipelineLayout,
    pub tril_b2_pipeline: ComputePipeline,
    pub tril_b2_layout: PipelineLayout,
    pub tril_b4_pipeline: ComputePipeline,
    pub tril_b4_layout: PipelineLayout,
    pub tril_b8_pipeline: ComputePipeline,
    pub tril_b8_layout: PipelineLayout,
    pub flip_b2_pipeline: ComputePipeline,
    pub flip_b2_layout: PipelineLayout,
    pub flip_b4_pipeline: ComputePipeline,
    pub flip_b4_layout: PipelineLayout,
    pub flip_b8_pipeline: ComputePipeline,
    pub flip_b8_layout: PipelineLayout,
    pub roll_b2_pipeline: ComputePipeline,
    pub roll_b2_layout: PipelineLayout,
    pub roll_b4_pipeline: ComputePipeline,
    pub roll_b4_layout: PipelineLayout,
    pub roll_b8_pipeline: ComputePipeline,
    pub roll_b8_layout: PipelineLayout,
    pub cumsum_f32_pipeline: ComputePipeline,
    pub cumsum_f32_layout: PipelineLayout,
    pub cumsum_f64_pipeline: ComputePipeline,
    pub cumsum_f64_layout: PipelineLayout,
    pub cumsum_f16_pipeline: ComputePipeline,
    pub cumsum_f16_layout: PipelineLayout,
    pub cumsum_bf16_pipeline: ComputePipeline,
    pub cumsum_bf16_layout: PipelineLayout,
    /// WGSL matmul (4x4 register tile, no shared memory). Fast for
    /// short M where the shared-memory tiled version's barriers cost
    /// more than they save.
    pub matmul_pipeline: ComputePipeline,
    pub matmul_layout: PipelineLayout,

    /// GLSL tiled matmul (64x64 output tile, BK=16 shared memory).
    /// Used when M is large enough to amortize the barrier overhead.
    pub matmul_tiled_pipeline: ComputePipeline,
    pub matmul_tiled_layout: PipelineLayout,

    /// GLSL gemv (M==1 specialization). Subgroup-reduced dot product,
    /// one workgroup per output column.
    pub matvec_pipeline: ComputePipeline,
    pub matvec_layout: PipelineLayout,

    /// Mixed-precision gemv: f32 activations × bf16 weights → f32.
    /// Decode-phase path for bf16-on-device weights.
    pub matvec_bf16_b_pipeline: ComputePipeline,
    pub matvec_bf16_b_layout: PipelineLayout,

    /// Mixed-precision tiled matmul: f32 × bf16 → f32 for M > 1.
    /// Prefill / training path for bf16-on-device weights.
    pub matmul_tiled_bf16_b_pipeline: ComputePipeline,
    pub matmul_tiled_bf16_b_layout: PipelineLayout,

    /// Cooperative-matrix (tensor-core) matmul: f32 × bf16 → f32.
    /// `None` when VK_KHR_cooperative_matrix is not available.
    pub matmul_coop_pipeline: Option<ComputePipeline>,
    pub matmul_coop_layout: Option<PipelineLayout>,
    pub matmul_coop_bf16_bf16_pipeline: Option<ComputePipeline>,
    pub matmul_coop_bf16_bf16_layout: Option<PipelineLayout>,
    pub matmul_coop_f16_f16_pipeline: Option<ComputePipeline>,
    pub matmul_coop_f16_f16_layout: Option<PipelineLayout>,
    pub matmul_coop_bf16_bf16_bf16_pipeline: Option<ComputePipeline>,
    pub matmul_coop_bf16_bf16_bf16_layout: Option<PipelineLayout>,
    pub softmax_pipeline: ComputePipeline,
    pub softmax_layout: PipelineLayout,

    pub softmax_f16_pipeline: ComputePipeline,
    pub softmax_f16_layout: PipelineLayout,

    pub softmax_bf16_pipeline: ComputePipeline,
    pub softmax_bf16_layout: PipelineLayout,

    pub softmax_f64_pipeline: ComputePipeline,
    pub softmax_f64_layout: PipelineLayout,
    pub reduce_pipeline: ComputePipeline,
    pub reduce_layout: PipelineLayout,

    pub reduce_f16_pipeline: ComputePipeline,
    pub reduce_f16_layout: PipelineLayout,

    pub reduce_bf16_pipeline: ComputePipeline,
    pub reduce_bf16_layout: PipelineLayout,

    pub reduce_f64_pipeline: ComputePipeline,
    pub reduce_f64_layout: PipelineLayout,

    pub cast_f32_to_f64_pipeline: ComputePipeline,
    pub cast_f32_to_f64_layout: PipelineLayout,

    pub cast_f64_to_f32_pipeline: ComputePipeline,
    pub cast_f64_to_f32_layout: PipelineLayout,

    pub reduce_last_dim_pipeline: ComputePipeline,
    pub reduce_last_dim_layout: PipelineLayout,

    pub arg_reduce_last_dim_f32_pipeline: ComputePipeline,
    pub arg_reduce_last_dim_f32_layout: PipelineLayout,

    pub scatter_add_f32_pipeline: ComputePipeline,
    pub scatter_add_f32_layout: PipelineLayout,
    pub scatter_add_f64_pipeline: ComputePipeline,
    pub scatter_add_f64_layout: PipelineLayout,
    pub scatter_add_bf16_pipeline: ComputePipeline,
    pub scatter_add_bf16_layout: PipelineLayout,
    pub scatter_add_f16_pipeline: ComputePipeline,
    pub scatter_add_f16_layout: PipelineLayout,
    pub arg_reduce_last_dim_f16_pipeline: ComputePipeline,
    pub arg_reduce_last_dim_f16_layout: PipelineLayout,
    pub arg_reduce_last_dim_bf16_pipeline: ComputePipeline,
    pub arg_reduce_last_dim_bf16_layout: PipelineLayout,
    pub arg_reduce_last_dim_f64_pipeline: ComputePipeline,
    pub arg_reduce_last_dim_f64_layout: PipelineLayout,
    pub arg_reduce_any_dim_f32_pipeline: ComputePipeline,
    pub arg_reduce_any_dim_f32_layout: PipelineLayout,
    pub arg_reduce_any_dim_f64_pipeline: ComputePipeline,
    pub arg_reduce_any_dim_f64_layout: PipelineLayout,
    pub arg_reduce_any_dim_bf16_pipeline: ComputePipeline,
    pub arg_reduce_any_dim_bf16_layout: PipelineLayout,
    pub arg_reduce_any_dim_f16_pipeline: ComputePipeline,
    pub arg_reduce_any_dim_f16_layout: PipelineLayout,
    pub index_add_f32_pipeline: ComputePipeline,
    pub index_add_f32_layout: PipelineLayout,
    pub index_add_f64_pipeline: ComputePipeline,
    pub index_add_f64_layout: PipelineLayout,
    pub index_add_bf16_pipeline: ComputePipeline,
    pub index_add_bf16_layout: PipelineLayout,
    pub index_add_f16_pipeline: ComputePipeline,
    pub index_add_f16_layout: PipelineLayout,

    pub reduce_last_dim_f16_pipeline: ComputePipeline,
    pub reduce_last_dim_f16_layout: PipelineLayout,

    pub reduce_last_dim_bf16_pipeline: ComputePipeline,
    pub reduce_last_dim_bf16_layout: PipelineLayout,

    pub reduce_last_dim_f64_pipeline: ComputePipeline,
    pub reduce_last_dim_f64_layout: PipelineLayout,

    pub rms_norm_last_dim_pipeline: ComputePipeline,
    pub rms_norm_last_dim_layout: PipelineLayout,

    pub rms_norm_last_dim_f16_pipeline: ComputePipeline,
    pub rms_norm_last_dim_f16_layout: PipelineLayout,

    pub rms_norm_last_dim_bf16_pipeline: ComputePipeline,
    pub rms_norm_last_dim_bf16_layout: PipelineLayout,

    pub rms_norm_last_dim_f64_pipeline: ComputePipeline,
    pub rms_norm_last_dim_f64_layout: PipelineLayout,

    pub rms_norm_last_dim_backward_pipeline: ComputePipeline,
    pub rms_norm_last_dim_backward_layout: PipelineLayout,

    pub softmax_last_dim_backward_pipeline: ComputePipeline,
    pub softmax_last_dim_backward_layout: PipelineLayout,

    pub softmax_last_dim_backward_f16_pipeline: ComputePipeline,
    pub softmax_last_dim_backward_f16_layout: PipelineLayout,

    pub softmax_last_dim_backward_bf16_pipeline: ComputePipeline,
    pub softmax_last_dim_backward_bf16_layout: PipelineLayout,

    pub softmax_last_dim_backward_f64_pipeline: ComputePipeline,
    pub softmax_last_dim_backward_f64_layout: PipelineLayout,

    pub layer_norm_last_dim_backward_pipeline: ComputePipeline,
    pub layer_norm_last_dim_backward_layout: PipelineLayout,

    pub layer_norm_last_dim_backward_f16_pipeline: ComputePipeline,
    pub layer_norm_last_dim_backward_f16_layout: PipelineLayout,
    pub layer_norm_last_dim_backward_bf16_pipeline: ComputePipeline,
    pub layer_norm_last_dim_backward_bf16_layout: PipelineLayout,
    pub layer_norm_last_dim_backward_f64_pipeline: ComputePipeline,
    pub layer_norm_last_dim_backward_f64_layout: PipelineLayout,

    pub layer_norm_last_dim_pipeline: ComputePipeline,
    pub layer_norm_last_dim_layout: PipelineLayout,
    pub layer_norm_last_dim_f16_pipeline: ComputePipeline,
    pub layer_norm_last_dim_f16_layout: PipelineLayout,
    pub layer_norm_last_dim_bf16_pipeline: ComputePipeline,
    pub layer_norm_last_dim_bf16_layout: PipelineLayout,
    pub layer_norm_last_dim_f64_pipeline: ComputePipeline,
    pub layer_norm_last_dim_f64_layout: PipelineLayout,

    pub strided_copy_pipeline: ComputePipeline,
    pub strided_copy_layout: PipelineLayout,

    pub index_select_pipeline: ComputePipeline,
    pub index_select_layout: PipelineLayout,

    pub index_select_f16_pipeline: ComputePipeline,
    pub index_select_f16_layout: PipelineLayout,

    pub index_select_bf16_pipeline: ComputePipeline,
    pub index_select_bf16_layout: PipelineLayout,

    pub index_select_f64_pipeline: ComputePipeline,
    pub index_select_f64_layout: PipelineLayout,

    pub add_assign_scaled_pipeline: ComputePipeline,
    pub add_assign_scaled_layout: PipelineLayout,

    pub rope_pipeline: ComputePipeline,
    pub rope_layout: PipelineLayout,

    pub rope_f16_pipeline: ComputePipeline,
    pub rope_f16_layout: PipelineLayout,

    pub rope_bf16_pipeline: ComputePipeline,
    pub rope_bf16_layout: PipelineLayout,

    pub rope_f64_pipeline: ComputePipeline,
    pub rope_f64_layout: PipelineLayout,

    pub concat_along_dim_pipeline: ComputePipeline,
    pub concat_along_dim_layout: PipelineLayout,

    pub concat_along_dim_f16_pipeline: ComputePipeline,
    pub concat_along_dim_f16_layout: PipelineLayout,

    pub concat_along_dim_bf16_pipeline: ComputePipeline,
    pub concat_along_dim_bf16_layout: PipelineLayout,

    pub concat_along_dim_f64_pipeline: ComputePipeline,
    pub concat_along_dim_f64_layout: PipelineLayout,

    /// Conv2D im2col patches rearrangement. Pairs with the existing
    /// matmul pipelines: dispatch this first to write the patches
    /// matrix, then dispatch matmul (weight × patches) per group.
    pub conv2d_im2col_pipeline: ComputePipeline,
    pub conv2d_im2col_layout: PipelineLayout,

    /// FlashAttention v2 forward (Phase 8 Tier 2). Single-dispatch
    /// kernel: tiled scaled-dot-product attention with online softmax.
    /// Workgroup grid is (B, Hq, ceil(Sq / BR=16)).
    pub flash_attention_pipeline: ComputePipeline,
    pub flash_attention_layout: PipelineLayout,

    pub dequant_q4_0_pipeline: ComputePipeline,
    pub dequant_q4_0_layout: PipelineLayout,

    pub dequant_q8_0_pipeline: ComputePipeline,
    pub dequant_q8_0_layout: PipelineLayout,

    pub dequant_q4_km_pipeline: ComputePipeline,
    pub dequant_q4_km_layout: PipelineLayout,

    pub qmatvec_q4_0_pipeline: ComputePipeline,
    pub qmatvec_q4_0_layout: PipelineLayout,

    pub matmul_q4_0_tiled_pipeline: ComputePipeline,
    pub matmul_q4_0_tiled_layout: PipelineLayout,

    pub quantize_q8_0_pipeline: ComputePipeline,
    pub quantize_q8_0_layout: PipelineLayout,

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
    pub fn allocate_desc(&self, layout: &DescriptorSetLayout) -> Result<DescriptorSet> {
        let _span = tracing::debug_span!("vk_alloc_desc").entered();
        // Try allocating from the current pool.
        {
            let pool = self.desc_pool.lock().expect("desc_pool poisoned");
            match pool.allocate(layout) {
                Ok(d) => return Ok(d),
                Err(Error::Vk(code)) if is_pool_oom(code) => { /* fall through to retire + recreate */ }
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
        self.retired_desc_pools.lock().expect("retired_desc_pools poisoned").push(old);
        self.desc_pool.lock().expect("desc_pool poisoned").allocate(layout)
    }

    /// Drop every retired descriptor pool. Safe to call only AFTER a
    /// sync point that guarantees the GPU is done with every command
    /// buffer ever recorded against those pools. `VulkanBackend`
    /// calls this from `drain_recorder`, which itself runs after the
    /// D2H copy's fence has signaled.
    pub fn retire_pools_post_drain(&self) {
        let mut retired = self.retired_desc_pools.lock().expect("retired_desc_pools poisoned");
        if !retired.is_empty() {
            let _s = tracing::info_span!("vk_retired_pools_drop", n = retired.len()).entered();
            retired.clear();
        }
    }
}

fn is_pool_oom(code: vulkane::raw::bindings::VkResult) -> bool {
    use vulkane::raw::bindings::VkResult;
    matches!(code,
        VkResult::ERROR_OUT_OF_POOL_MEMORY
        | VkResult::ERROR_FRAGMENTED_POOL)
}

fn make_desc_pool(device: &Device) -> Result<DescriptorPool> {
    DescriptorPool::new(device, 4096, &[
        DescriptorPoolSize {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 16384,
        },
        DescriptorPoolSize {
            descriptor_type: DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 4096,
        },
    ])
}

impl Pipelines {
    pub fn new(device: &Device, has_coop_matrix: bool) -> Result<Self> {
        // Layout: 2 storage buffers (binding 0,1) + 1 uniform (binding 2).
        let layout_2s1u = DescriptorSetLayout::new(device, &[
            storage_binding(0),
            storage_binding(1),
            uniform_binding(2),
        ])?;

        // Layout: 3 storage buffers (binding 0,1,2) + 1 uniform (binding 3).
        let layout_3s1u = DescriptorSetLayout::new(device, &[
            storage_binding(0),
            storage_binding(1),
            storage_binding(2),
            uniform_binding(3),
        ])?;

        // Layout: 4 storage buffers (binding 0..3) + 1 uniform (binding 4).
        // Used by rope: (x, cos, sin, out, params).
        let layout_4s1u = DescriptorSetLayout::new(device, &[
            storage_binding(0),
            storage_binding(1),
            storage_binding(2),
            storage_binding(3),
            uniform_binding(4),
        ])?;

        // Layout: 5 storage buffers (binding 0..4) + 1 uniform (binding 5).
        // Used by flash_attention: (q, k, v, alibi, o, params).
        // alibi is bound to a 1-element dummy buffer when has_alibi=0.
        let layout_5s1u = DescriptorSetLayout::new(device, &[
            storage_binding(0),
            storage_binding(1),
            storage_binding(2),
            storage_binding(3),
            storage_binding(4),
            uniform_binding(5),
        ])?;

        let desc_pool = Mutex::new(make_desc_pool(device)?);

        // Build the registry once and resolve every shader through it
        // — disk-override → embedded fallback, then straight to a
        // ShaderModule. No intermediate SPIR-V word vectors needed.
        use fuel_vulkan_kernels as shaders;
        let registry = shader_registry();
        let unary_mod = registry.load_module(device, shaders::UNARY)?;
        let unary_f16_mod = registry.load_module(device, shaders::UNARY_F16)?;
        let unary_f64_mod = registry.load_module(device, shaders::UNARY_F64)?;
        let unary_bf16_mod = registry.load_module(device, shaders::UNARY_BF16)?;
        let binary_mod = registry.load_module(device, shaders::BINARY)?;
        let binary_f16_mod = registry.load_module(device, shaders::BINARY_F16)?;
        let binary_f64_mod = registry.load_module(device, shaders::BINARY_F64)?;
        let binary_bf16_mod = registry.load_module(device, shaders::BINARY_BF16)?;
        let affine_mod = registry.load_module(device, shaders::AFFINE)?;
        let affine_f64_mod  = registry.load_module(device, shaders::AFFINE_F64)?;
        let affine_f16_mod  = registry.load_module(device, shaders::AFFINE_F16)?;
        let affine_bf16_mod = registry.load_module(device, shaders::AFFINE_BF16)?;
        let clamp_mod = registry.load_module(device, shaders::CLAMP)?;
        let powi_mod = registry.load_module(device, shaders::POWI)?;
        let cast_f32_to_f16_mod  = registry.load_module(device, shaders::CAST_F32_TO_F16)?;
        let cast_f16_to_f32_mod  = registry.load_module(device, shaders::CAST_F16_TO_F32)?;
        let cast_f32_to_bf16_mod = registry.load_module(device, shaders::CAST_F32_TO_BF16)?;
        let cast_bf16_to_f32_mod = registry.load_module(device, shaders::CAST_BF16_TO_F32)?;
        let cast_f32_to_f8e4m3_mod  = registry.load_module(device, shaders::CAST_F32_TO_F8E4M3)?;
        let cast_f8e4m3_to_f32_mod  = registry.load_module(device, shaders::CAST_F8E4M3_TO_F32)?;
        let cast_f16_to_f8e4m3_mod  = registry.load_module(device, shaders::CAST_F16_TO_F8E4M3)?;
        let cast_f8e4m3_to_f16_mod  = registry.load_module(device, shaders::CAST_F8E4M3_TO_F16)?;
        let cast_bf16_to_f8e4m3_mod = registry.load_module(device, shaders::CAST_BF16_TO_F8E4M3)?;
        let cast_f8e4m3_to_bf16_mod = registry.load_module(device, shaders::CAST_F8E4M3_TO_BF16)?;
        let write_slice_b1_mod   = registry.load_module(device, shaders::WRITE_SLICE_B1)?;
        let write_slice_b2_mod   = registry.load_module(device, shaders::WRITE_SLICE_B2)?;
        let write_slice_b4_mod   = registry.load_module(device, shaders::WRITE_SLICE_B4)?;
        let pad_const_b1_mod = registry.load_module(device, shaders::PAD_CONST_B1)?;
        let pad_const_b2_mod = registry.load_module(device, shaders::PAD_CONST_B2)?;
        let pad_const_b4_mod = registry.load_module(device, shaders::PAD_CONST_B4)?;
        let pad_const_b8_mod = registry.load_module(device, shaders::PAD_CONST_B8)?;
        let pad_reflect_b1_mod = registry.load_module(device, shaders::PAD_REFLECT_B1)?;
        let pad_reflect_b2_mod = registry.load_module(device, shaders::PAD_REFLECT_B2)?;
        let pad_reflect_b4_mod = registry.load_module(device, shaders::PAD_REFLECT_B4)?;
        let pad_reflect_b8_mod = registry.load_module(device, shaders::PAD_REFLECT_B8)?;
        let pad_replicate_b1_mod = registry.load_module(device, shaders::PAD_REPLICATE_B1)?;
        let pad_replicate_b2_mod = registry.load_module(device, shaders::PAD_REPLICATE_B2)?;
        let pad_replicate_b4_mod = registry.load_module(device, shaders::PAD_REPLICATE_B4)?;
        let pad_replicate_b8_mod = registry.load_module(device, shaders::PAD_REPLICATE_B8)?;
        let pad_backward_const_b1_mod = registry.load_module(device, shaders::PAD_BACKWARD_CONST_B1)?;
        let pad_backward_const_b2_mod = registry.load_module(device, shaders::PAD_BACKWARD_CONST_B2)?;
        let pad_backward_const_b4_mod = registry.load_module(device, shaders::PAD_BACKWARD_CONST_B4)?;
        let pad_backward_const_b8_mod = registry.load_module(device, shaders::PAD_BACKWARD_CONST_B8)?;
        let pad_backward_reflect_f32_mod   = registry.load_module(device, shaders::PAD_BACKWARD_REFLECT_F32)?;
        let pad_backward_replicate_f32_mod = registry.load_module(device, shaders::PAD_BACKWARD_REPLICATE_F32)?;
        let pad_backward_reflect_f64_mod   = registry.load_module(device, shaders::PAD_BACKWARD_REFLECT_F64)?;
        let pad_backward_replicate_f64_mod = registry.load_module(device, shaders::PAD_BACKWARD_REPLICATE_F64)?;
        let pad_backward_reflect_bf16_mod  = registry.load_module(device, shaders::PAD_BACKWARD_REFLECT_BF16)?;
        let pad_backward_replicate_bf16_mod= registry.load_module(device, shaders::PAD_BACKWARD_REPLICATE_BF16)?;
        let pad_backward_reflect_f16_mod   = registry.load_module(device, shaders::PAD_BACKWARD_REFLECT_F16)?;
        let pad_backward_replicate_f16_mod = registry.load_module(device, shaders::PAD_BACKWARD_REPLICATE_F16)?;
        let masked_fill_b1_mod = registry.load_module(device, shaders::MASKED_FILL_B1)?;
        let masked_fill_b2_mod = registry.load_module(device, shaders::MASKED_FILL_B2)?;
        let masked_fill_b4_mod = registry.load_module(device, shaders::MASKED_FILL_B4)?;
        let masked_fill_b8_mod = registry.load_module(device, shaders::MASKED_FILL_B8)?;
        let gather_b1_mod = registry.load_module(device, shaders::GATHER_B1)?;
        let gather_b2_mod = registry.load_module(device, shaders::GATHER_B2)?;
        let gather_b4_mod = registry.load_module(device, shaders::GATHER_B4)?;
        let gather_b8_mod = registry.load_module(device, shaders::GATHER_B8)?;
        let write_slice_b8_mod   = registry.load_module(device, shaders::WRITE_SLICE_B8)?;
        let strided_copy_signed_b2_mod = registry.load_module(device, shaders::STRIDED_COPY_SIGNED_B2)?;
        let strided_copy_signed_b4_mod = registry.load_module(device, shaders::STRIDED_COPY_SIGNED_B4)?;
        let strided_copy_signed_b8_mod = registry.load_module(device, shaders::STRIDED_COPY_SIGNED_B8)?;
        let triu_b2_mod = registry.load_module(device, shaders::TRIU_B2)?;
        let triu_b4_mod = registry.load_module(device, shaders::TRIU_B4)?;
        let triu_b8_mod = registry.load_module(device, shaders::TRIU_B8)?;
        let tril_b2_mod = registry.load_module(device, shaders::TRIL_B2)?;
        let tril_b4_mod = registry.load_module(device, shaders::TRIL_B4)?;
        let tril_b8_mod = registry.load_module(device, shaders::TRIL_B8)?;
        let flip_b2_mod = registry.load_module(device, shaders::FLIP_B2)?;
        let flip_b4_mod = registry.load_module(device, shaders::FLIP_B4)?;
        let flip_b8_mod = registry.load_module(device, shaders::FLIP_B8)?;
        let roll_b2_mod = registry.load_module(device, shaders::ROLL_B2)?;
        let roll_b4_mod = registry.load_module(device, shaders::ROLL_B4)?;
        let roll_b8_mod = registry.load_module(device, shaders::ROLL_B8)?;
        let cumsum_f32_mod = registry.load_module(device, shaders::CUMSUM_F32)?;
        let cumsum_f64_mod = registry.load_module(device, shaders::CUMSUM_F64)?;
        let cumsum_f16_mod = registry.load_module(device, shaders::CUMSUM_F16)?;
        let cumsum_bf16_mod = registry.load_module(device, shaders::CUMSUM_BF16)?;
        let matmul_mod = registry.load_module(device, shaders::MATMUL)?;
        let matmul_tiled_mod = registry.load_module(device, shaders::MATMUL_TILED_GLSL)?;
        let matvec_mod = registry.load_module(device, shaders::MATVEC_GLSL)?;
        let matvec_bf16_b_mod = registry.load_module(device, shaders::MATVEC_BF16_B_GLSL)?;
        let matmul_tiled_bf16_b_mod = registry.load_module(device, shaders::MATMUL_TILED_BF16_B_GLSL)?;
        let matmul_coop_mod = if has_coop_matrix {
            Some(registry.load_module(device, shaders::MATMUL_COOP)?)
        } else {
            None
        };
        let matmul_coop_bf16_bf16_mod = if has_coop_matrix {
            Some(registry.load_module(device, shaders::MATMUL_COOP_BF16_BF16)?)
        } else {
            None
        };
        let matmul_coop_f16_f16_mod = if has_coop_matrix {
            Some(registry.load_module(device, shaders::MATMUL_COOP_F16_F16)?)
        } else {
            None
        };
        let matmul_coop_bf16_bf16_bf16_mod = if has_coop_matrix {
            Some(registry.load_module(device, shaders::MATMUL_COOP_BF16_BF16_BF16)?)
        } else {
            None
        };
        let softmax_mod = registry.load_module(device, shaders::SOFTMAX)?;
        let softmax_f16_mod = registry.load_module(device, shaders::SOFTMAX_F16)?;
        let softmax_bf16_mod = registry.load_module(device, shaders::SOFTMAX_BF16)?;
        let softmax_f64_mod = registry.load_module(device, shaders::SOFTMAX_F64)?;
        let reduce_mod = registry.load_module(device, shaders::REDUCE)?;
        let reduce_f16_mod = registry.load_module(device, shaders::REDUCE_F16)?;
        let reduce_bf16_mod = registry.load_module(device, shaders::REDUCE_BF16)?;
        let reduce_f64_mod = registry.load_module(device, shaders::REDUCE_F64)?;
        let cast_f32_to_f64_mod = registry.load_module(device, shaders::CAST_F32_TO_F64)?;
        let cast_f64_to_f32_mod = registry.load_module(device, shaders::CAST_F64_TO_F32)?;
        let reduce_last_dim_mod = registry.load_module(device, shaders::REDUCE_LAST_DIM)?;
        let arg_reduce_last_dim_f32_mod  = registry.load_module(device, shaders::ARG_REDUCE_LAST_DIM_F32)?;
        let scatter_add_f32_mod = registry.load_module(device, shaders::SCATTER_ADD_F32)?;
        let scatter_add_f64_mod = registry.load_module(device, shaders::SCATTER_ADD_F64)?;
        let scatter_add_bf16_mod = registry.load_module(device, shaders::SCATTER_ADD_BF16)?;
        let scatter_add_f16_mod = registry.load_module(device, shaders::SCATTER_ADD_F16)?;
        let arg_reduce_last_dim_f16_mod  = registry.load_module(device, shaders::ARG_REDUCE_LAST_DIM_F16)?;
        let arg_reduce_last_dim_bf16_mod = registry.load_module(device, shaders::ARG_REDUCE_LAST_DIM_BF16)?;
        let arg_reduce_last_dim_f64_mod  = registry.load_module(device, shaders::ARG_REDUCE_LAST_DIM_F64)?;
        let arg_reduce_any_dim_f32_mod   = registry.load_module(device, shaders::ARG_REDUCE_ANY_DIM_F32)?;
        let arg_reduce_any_dim_f64_mod   = registry.load_module(device, shaders::ARG_REDUCE_ANY_DIM_F64)?;
        let arg_reduce_any_dim_bf16_mod  = registry.load_module(device, shaders::ARG_REDUCE_ANY_DIM_BF16)?;
        let arg_reduce_any_dim_f16_mod   = registry.load_module(device, shaders::ARG_REDUCE_ANY_DIM_F16)?;
        let index_add_f32_mod  = registry.load_module(device, shaders::INDEX_ADD_F32)?;
        let index_add_f64_mod  = registry.load_module(device, shaders::INDEX_ADD_F64)?;
        let index_add_bf16_mod = registry.load_module(device, shaders::INDEX_ADD_BF16)?;
        let index_add_f16_mod  = registry.load_module(device, shaders::INDEX_ADD_F16)?;
        let reduce_last_dim_f16_mod = registry.load_module(device, shaders::REDUCE_LAST_DIM_F16)?;
        let reduce_last_dim_bf16_mod = registry.load_module(device, shaders::REDUCE_LAST_DIM_BF16)?;
        let reduce_last_dim_f64_mod = registry.load_module(device, shaders::REDUCE_LAST_DIM_F64)?;
        let rms_norm_last_dim_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM)?;
        let rms_norm_last_dim_f16_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM_F16)?;
        let rms_norm_last_dim_bf16_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM_BF16)?;
        let rms_norm_last_dim_f64_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM_F64)?;
        let rms_norm_last_dim_backward_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM_BACKWARD)?;
        let softmax_last_dim_backward_mod = registry.load_module(device, shaders::SOFTMAX_LAST_DIM_BACKWARD)?;
        let softmax_last_dim_backward_f16_mod  = registry.load_module(device, shaders::SOFTMAX_LAST_DIM_BACKWARD_F16)?;
        let softmax_last_dim_backward_bf16_mod = registry.load_module(device, shaders::SOFTMAX_LAST_DIM_BACKWARD_BF16)?;
        let softmax_last_dim_backward_f64_mod  = registry.load_module(device, shaders::SOFTMAX_LAST_DIM_BACKWARD_F64)?;
        let layer_norm_last_dim_backward_mod = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_BACKWARD)?;
        let layer_norm_last_dim_backward_f16_mod  = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_BACKWARD_F16)?;
        let layer_norm_last_dim_backward_bf16_mod = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_BACKWARD_BF16)?;
        let layer_norm_last_dim_backward_f64_mod  = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_BACKWARD_F64)?;
        let layer_norm_last_dim_mod      = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM)?;
        let layer_norm_last_dim_f16_mod  = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_F16)?;
        let layer_norm_last_dim_bf16_mod = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_BF16)?;
        let layer_norm_last_dim_f64_mod  = registry.load_module(device, shaders::LAYER_NORM_LAST_DIM_F64)?;
        let strided_copy_mod = registry.load_module(device, shaders::STRIDED_COPY)?;
        let index_select_mod = registry.load_module(device, shaders::INDEX_SELECT)?;
        let index_select_f16_mod = registry.load_module(device, shaders::INDEX_SELECT_F16)?;
        let index_select_bf16_mod = registry.load_module(device, shaders::INDEX_SELECT_BF16)?;
        let index_select_f64_mod = registry.load_module(device, shaders::INDEX_SELECT_F64)?;
        let add_assign_scaled_mod = registry.load_module(device, shaders::ADD_ASSIGN_SCALED)?;
        let rope_mod = registry.load_module(device, shaders::ROPE)?;
        let rope_f16_mod = registry.load_module(device, shaders::ROPE_F16)?;
        let rope_bf16_mod = registry.load_module(device, shaders::ROPE_BF16)?;
        let rope_f64_mod = registry.load_module(device, shaders::ROPE_F64)?;
        let concat_along_dim_mod = registry.load_module(device, shaders::CONCAT_ALONG_DIM)?;
        let concat_along_dim_f16_mod = registry.load_module(device, shaders::CONCAT_ALONG_DIM_F16)?;
        let concat_along_dim_bf16_mod = registry.load_module(device, shaders::CONCAT_ALONG_DIM_BF16)?;
        let concat_along_dim_f64_mod = registry.load_module(device, shaders::CONCAT_ALONG_DIM_F64)?;
        let conv2d_im2col_mod = registry.load_module(device, shaders::CONV2D_IM2COL)?;
        let flash_attention_mod = registry.load_module(device, shaders::FLASH_ATTENTION)?;
        let dequant_q4_0_mod = registry.load_module(device, shaders::DEQUANT_Q4_0)?;
        let dequant_q4_km_mod = registry.load_module(device, shaders::DEQUANT_Q4_KM)?;
        let dequant_q8_0_mod = registry.load_module(device, shaders::DEQUANT_Q8_0)?;
        let qmatvec_q4_0_mod = registry.load_module(device, shaders::QMATVEC_Q4_0)?;
        let matmul_q4_0_tiled_mod = registry.load_module(device, shaders::MATMUL_Q4_0_TILED)?;
        let quantize_q8_0_mod = registry.load_module(device, shaders::QUANTIZE_Q8_0)?;

        // No push constants — params go through uniform buffers.
        let unary_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let unary_f16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let unary_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let unary_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let binary_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let binary_f16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let binary_f64_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let binary_bf16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let affine_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let affine_f64_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let affine_f16_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let affine_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let clamp_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let powi_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f32_to_f16_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f16_to_f32_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f32_to_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_bf16_to_f32_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f32_to_f8e4m3_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f8e4m3_to_f32_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f16_to_f8e4m3_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f8e4m3_to_f16_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_bf16_to_f8e4m3_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f8e4m3_to_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        // write_slice uses 3 storage (src + dst + shape_buf) + 1 uniform.
        let write_slice_b1_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        let write_slice_b2_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        let write_slice_b4_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_const_b1_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_const_b2_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_const_b4_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_const_b8_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_reflect_b1_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_reflect_b2_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_reflect_b4_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_reflect_b8_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_replicate_b1_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_replicate_b2_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_replicate_b4_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_replicate_b8_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_const_b1_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_const_b2_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_const_b4_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_const_b8_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_reflect_f32_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_replicate_f32_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_reflect_f64_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_replicate_f64_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_reflect_bf16_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_replicate_bf16_layout= PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_reflect_f16_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        let pad_backward_replicate_f16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let masked_fill_b1_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let masked_fill_b2_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let masked_fill_b4_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let masked_fill_b8_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let gather_b1_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let gather_b2_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let gather_b4_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let gather_b8_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let write_slice_b8_layout   = PipelineLayout::new(device, &[&layout_3s1u])?;
        // strided_copy_signed uses 3 storage (input, output, shape_strides) + 1 uniform.
        let strided_copy_signed_b2_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let strided_copy_signed_b4_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let strided_copy_signed_b8_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        // triu / tril / flip / roll all use 2 storage (in, out) + 1 uniform.
        let triu_b2_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let triu_b4_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let triu_b8_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let tril_b2_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let tril_b4_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let tril_b8_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let flip_b2_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let flip_b4_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let flip_b8_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let roll_b2_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let roll_b4_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let roll_b8_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        // cumsum (per-dtype, accumulator-typed) — 2 storage (in, out) + 1 uniform.
        let cumsum_f32_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cumsum_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cumsum_f16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cumsum_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let matmul_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matmul_tiled_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matvec_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matvec_bf16_b_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matmul_tiled_bf16_b_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matmul_coop_layout = if has_coop_matrix {
            Some(PipelineLayout::new(device, &[&layout_3s1u])?)
        } else { None };
        let matmul_coop_bf16_bf16_layout = if has_coop_matrix {
            Some(PipelineLayout::new(device, &[&layout_3s1u])?)
        } else { None };
        let matmul_coop_f16_f16_layout = if has_coop_matrix {
            Some(PipelineLayout::new(device, &[&layout_3s1u])?)
        } else { None };
        let matmul_coop_bf16_bf16_bf16_layout = if has_coop_matrix {
            Some(PipelineLayout::new(device, &[&layout_3s1u])?)
        } else { None };
        let softmax_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let softmax_f16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let softmax_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let softmax_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_f16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f32_to_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let cast_f64_to_f32_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_last_dim_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_last_dim_f32_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let scatter_add_f32_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let scatter_add_f64_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let scatter_add_bf16_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let scatter_add_f16_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let arg_reduce_last_dim_f16_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_last_dim_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_last_dim_f64_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_any_dim_f32_layout   = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_any_dim_f64_layout   = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_any_dim_bf16_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let arg_reduce_any_dim_f16_layout   = PipelineLayout::new(device, &[&layout_2s1u])?;
        let index_add_f32_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_add_f64_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_add_bf16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_add_f16_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let reduce_last_dim_f16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_last_dim_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_last_dim_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rms_norm_last_dim_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rms_norm_last_dim_f16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rms_norm_last_dim_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rms_norm_last_dim_f64_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        // backward takes 3 storage buffers (x, upstream, grad_x) + params
        let rms_norm_last_dim_backward_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_last_dim_backward_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_last_dim_backward_f16_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_last_dim_backward_bf16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_last_dim_backward_f64_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let layer_norm_last_dim_backward_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let layer_norm_last_dim_backward_f16_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let layer_norm_last_dim_backward_bf16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let layer_norm_last_dim_backward_f64_layout  = PipelineLayout::new(device, &[&layout_3s1u])?;
        let layer_norm_last_dim_layout      = PipelineLayout::new(device, &[&layout_2s1u])?;
        let layer_norm_last_dim_f16_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let layer_norm_last_dim_bf16_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let layer_norm_last_dim_f64_layout  = PipelineLayout::new(device, &[&layout_2s1u])?;
        let strided_copy_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_select_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_select_f16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_select_bf16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_select_f64_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let add_assign_scaled_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rope_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let rope_f16_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let rope_bf16_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let rope_f64_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let concat_along_dim_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let concat_along_dim_f16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let concat_along_dim_bf16_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let concat_along_dim_f64_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        // conv2d_im2col uses 1 storage in (x) + 1 storage out (patches) + 1 uniform.
        let conv2d_im2col_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        // flash_attention: q + k + v + alibi (or dummy) + o + params.
        let flash_attention_layout = PipelineLayout::new(device, &[&layout_5s1u])?;
        let dequant_q4_0_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let dequant_q4_km_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let dequant_q8_0_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let qmatvec_q4_0_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matmul_q4_0_tiled_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let quantize_q8_0_layout = PipelineLayout::new(device, &[&layout_2s1u])?;

        let unary_pipeline = ComputePipeline::new(device, &unary_layout, &unary_mod, "main")?;
        let unary_f16_pipeline = ComputePipeline::new(device, &unary_f16_layout, &unary_f16_mod, "main")?;
        let unary_f64_pipeline = ComputePipeline::new(device, &unary_f64_layout, &unary_f64_mod, "main")?;
        let unary_bf16_pipeline = ComputePipeline::new(device, &unary_bf16_layout, &unary_bf16_mod, "main")?;
        let binary_pipeline = ComputePipeline::new(device, &binary_layout, &binary_mod, "main")?;
        let binary_f16_pipeline = ComputePipeline::new(device, &binary_f16_layout, &binary_f16_mod, "main")?;
        let binary_f64_pipeline = ComputePipeline::new(device, &binary_f64_layout, &binary_f64_mod, "main")?;
        let binary_bf16_pipeline = ComputePipeline::new(device, &binary_bf16_layout, &binary_bf16_mod, "main")?;
        let affine_pipeline = ComputePipeline::new(device, &affine_layout, &affine_mod, "main")?;
        let affine_f64_pipeline  = ComputePipeline::new(device, &affine_f64_layout,  &affine_f64_mod,  "main")?;
        let affine_f16_pipeline  = ComputePipeline::new(device, &affine_f16_layout,  &affine_f16_mod,  "main")?;
        let affine_bf16_pipeline = ComputePipeline::new(device, &affine_bf16_layout, &affine_bf16_mod, "main")?;
        let clamp_pipeline = ComputePipeline::new(device, &clamp_layout, &clamp_mod, "main")?;
        let powi_pipeline = ComputePipeline::new(device, &powi_layout, &powi_mod, "main")?;
        let cast_f32_to_f16_pipeline  = ComputePipeline::new(device, &cast_f32_to_f16_layout,  &cast_f32_to_f16_mod,  "main")?;
        let cast_f16_to_f32_pipeline  = ComputePipeline::new(device, &cast_f16_to_f32_layout,  &cast_f16_to_f32_mod,  "main")?;
        let cast_f32_to_bf16_pipeline = ComputePipeline::new(device, &cast_f32_to_bf16_layout, &cast_f32_to_bf16_mod, "main")?;
        let cast_bf16_to_f32_pipeline = ComputePipeline::new(device, &cast_bf16_to_f32_layout, &cast_bf16_to_f32_mod, "main")?;
        let cast_f32_to_f8e4m3_pipeline  = ComputePipeline::new(device, &cast_f32_to_f8e4m3_layout,  &cast_f32_to_f8e4m3_mod,  "main")?;
        let cast_f8e4m3_to_f32_pipeline  = ComputePipeline::new(device, &cast_f8e4m3_to_f32_layout,  &cast_f8e4m3_to_f32_mod,  "main")?;
        let cast_f16_to_f8e4m3_pipeline  = ComputePipeline::new(device, &cast_f16_to_f8e4m3_layout,  &cast_f16_to_f8e4m3_mod,  "main")?;
        let cast_f8e4m3_to_f16_pipeline  = ComputePipeline::new(device, &cast_f8e4m3_to_f16_layout,  &cast_f8e4m3_to_f16_mod,  "main")?;
        let cast_bf16_to_f8e4m3_pipeline = ComputePipeline::new(device, &cast_bf16_to_f8e4m3_layout, &cast_bf16_to_f8e4m3_mod, "main")?;
        let cast_f8e4m3_to_bf16_pipeline = ComputePipeline::new(device, &cast_f8e4m3_to_bf16_layout, &cast_f8e4m3_to_bf16_mod, "main")?;
        let write_slice_b1_pipeline   = ComputePipeline::new(device, &write_slice_b1_layout,   &write_slice_b1_mod,   "main")?;
        let write_slice_b2_pipeline   = ComputePipeline::new(device, &write_slice_b2_layout,   &write_slice_b2_mod,   "main")?;
        let write_slice_b4_pipeline   = ComputePipeline::new(device, &write_slice_b4_layout,   &write_slice_b4_mod,   "main")?;
        let pad_const_b1_pipeline = ComputePipeline::new(device, &pad_const_b1_layout, &pad_const_b1_mod, "main")?;
        let pad_const_b2_pipeline = ComputePipeline::new(device, &pad_const_b2_layout, &pad_const_b2_mod, "main")?;
        let pad_const_b4_pipeline = ComputePipeline::new(device, &pad_const_b4_layout, &pad_const_b4_mod, "main")?;
        let pad_const_b8_pipeline = ComputePipeline::new(device, &pad_const_b8_layout, &pad_const_b8_mod, "main")?;
        let pad_reflect_b1_pipeline = ComputePipeline::new(device, &pad_reflect_b1_layout, &pad_reflect_b1_mod, "main")?;
        let pad_reflect_b2_pipeline = ComputePipeline::new(device, &pad_reflect_b2_layout, &pad_reflect_b2_mod, "main")?;
        let pad_reflect_b4_pipeline = ComputePipeline::new(device, &pad_reflect_b4_layout, &pad_reflect_b4_mod, "main")?;
        let pad_reflect_b8_pipeline = ComputePipeline::new(device, &pad_reflect_b8_layout, &pad_reflect_b8_mod, "main")?;
        let pad_replicate_b1_pipeline = ComputePipeline::new(device, &pad_replicate_b1_layout, &pad_replicate_b1_mod, "main")?;
        let pad_replicate_b2_pipeline = ComputePipeline::new(device, &pad_replicate_b2_layout, &pad_replicate_b2_mod, "main")?;
        let pad_replicate_b4_pipeline = ComputePipeline::new(device, &pad_replicate_b4_layout, &pad_replicate_b4_mod, "main")?;
        let pad_replicate_b8_pipeline = ComputePipeline::new(device, &pad_replicate_b8_layout, &pad_replicate_b8_mod, "main")?;
        let pad_backward_const_b1_pipeline = ComputePipeline::new(device, &pad_backward_const_b1_layout, &pad_backward_const_b1_mod, "main")?;
        let pad_backward_const_b2_pipeline = ComputePipeline::new(device, &pad_backward_const_b2_layout, &pad_backward_const_b2_mod, "main")?;
        let pad_backward_const_b4_pipeline = ComputePipeline::new(device, &pad_backward_const_b4_layout, &pad_backward_const_b4_mod, "main")?;
        let pad_backward_const_b8_pipeline = ComputePipeline::new(device, &pad_backward_const_b8_layout, &pad_backward_const_b8_mod, "main")?;
        let pad_backward_reflect_f32_pipeline   = ComputePipeline::new(device, &pad_backward_reflect_f32_layout,   &pad_backward_reflect_f32_mod,   "main")?;
        let pad_backward_replicate_f32_pipeline = ComputePipeline::new(device, &pad_backward_replicate_f32_layout, &pad_backward_replicate_f32_mod, "main")?;
        let pad_backward_reflect_f64_pipeline   = ComputePipeline::new(device, &pad_backward_reflect_f64_layout,   &pad_backward_reflect_f64_mod,   "main")?;
        let pad_backward_replicate_f64_pipeline = ComputePipeline::new(device, &pad_backward_replicate_f64_layout, &pad_backward_replicate_f64_mod, "main")?;
        let pad_backward_reflect_bf16_pipeline  = ComputePipeline::new(device, &pad_backward_reflect_bf16_layout,  &pad_backward_reflect_bf16_mod,  "main")?;
        let pad_backward_replicate_bf16_pipeline= ComputePipeline::new(device, &pad_backward_replicate_bf16_layout,&pad_backward_replicate_bf16_mod,"main")?;
        let pad_backward_reflect_f16_pipeline   = ComputePipeline::new(device, &pad_backward_reflect_f16_layout,   &pad_backward_reflect_f16_mod,   "main")?;
        let pad_backward_replicate_f16_pipeline = ComputePipeline::new(device, &pad_backward_replicate_f16_layout, &pad_backward_replicate_f16_mod, "main")?;
        let masked_fill_b1_pipeline = ComputePipeline::new(device, &masked_fill_b1_layout, &masked_fill_b1_mod, "main")?;
        let masked_fill_b2_pipeline = ComputePipeline::new(device, &masked_fill_b2_layout, &masked_fill_b2_mod, "main")?;
        let masked_fill_b4_pipeline = ComputePipeline::new(device, &masked_fill_b4_layout, &masked_fill_b4_mod, "main")?;
        let masked_fill_b8_pipeline = ComputePipeline::new(device, &masked_fill_b8_layout, &masked_fill_b8_mod, "main")?;
        let gather_b1_pipeline = ComputePipeline::new(device, &gather_b1_layout, &gather_b1_mod, "main")?;
        let gather_b2_pipeline = ComputePipeline::new(device, &gather_b2_layout, &gather_b2_mod, "main")?;
        let gather_b4_pipeline = ComputePipeline::new(device, &gather_b4_layout, &gather_b4_mod, "main")?;
        let gather_b8_pipeline = ComputePipeline::new(device, &gather_b8_layout, &gather_b8_mod, "main")?;
        let write_slice_b8_pipeline   = ComputePipeline::new(device, &write_slice_b8_layout,   &write_slice_b8_mod,   "main")?;
        let strided_copy_signed_b2_pipeline = ComputePipeline::new(device, &strided_copy_signed_b2_layout, &strided_copy_signed_b2_mod, "main")?;
        let strided_copy_signed_b4_pipeline = ComputePipeline::new(device, &strided_copy_signed_b4_layout, &strided_copy_signed_b4_mod, "main")?;
        let strided_copy_signed_b8_pipeline = ComputePipeline::new(device, &strided_copy_signed_b8_layout, &strided_copy_signed_b8_mod, "main")?;
        let triu_b2_pipeline = ComputePipeline::new(device, &triu_b2_layout, &triu_b2_mod, "main")?;
        let triu_b4_pipeline = ComputePipeline::new(device, &triu_b4_layout, &triu_b4_mod, "main")?;
        let triu_b8_pipeline = ComputePipeline::new(device, &triu_b8_layout, &triu_b8_mod, "main")?;
        let tril_b2_pipeline = ComputePipeline::new(device, &tril_b2_layout, &tril_b2_mod, "main")?;
        let tril_b4_pipeline = ComputePipeline::new(device, &tril_b4_layout, &tril_b4_mod, "main")?;
        let tril_b8_pipeline = ComputePipeline::new(device, &tril_b8_layout, &tril_b8_mod, "main")?;
        let flip_b2_pipeline = ComputePipeline::new(device, &flip_b2_layout, &flip_b2_mod, "main")?;
        let flip_b4_pipeline = ComputePipeline::new(device, &flip_b4_layout, &flip_b4_mod, "main")?;
        let flip_b8_pipeline = ComputePipeline::new(device, &flip_b8_layout, &flip_b8_mod, "main")?;
        let roll_b2_pipeline = ComputePipeline::new(device, &roll_b2_layout, &roll_b2_mod, "main")?;
        let roll_b4_pipeline = ComputePipeline::new(device, &roll_b4_layout, &roll_b4_mod, "main")?;
        let roll_b8_pipeline = ComputePipeline::new(device, &roll_b8_layout, &roll_b8_mod, "main")?;
        let cumsum_f32_pipeline = ComputePipeline::new(device, &cumsum_f32_layout, &cumsum_f32_mod, "main")?;
        let cumsum_f64_pipeline = ComputePipeline::new(device, &cumsum_f64_layout, &cumsum_f64_mod, "main")?;
        let cumsum_f16_pipeline = ComputePipeline::new(device, &cumsum_f16_layout, &cumsum_f16_mod, "main")?;
        let cumsum_bf16_pipeline = ComputePipeline::new(device, &cumsum_bf16_layout, &cumsum_bf16_mod, "main")?;
        let matmul_pipeline = ComputePipeline::new(device, &matmul_layout, &matmul_mod, "main")?;
        let matmul_tiled_pipeline = ComputePipeline::new(device, &matmul_tiled_layout, &matmul_tiled_mod, "main")?;
        let matvec_pipeline = ComputePipeline::new(device, &matvec_layout, &matvec_mod, "main")?;
        let matvec_bf16_b_pipeline = ComputePipeline::new(device, &matvec_bf16_b_layout, &matvec_bf16_b_mod, "main")?;
        let matmul_tiled_bf16_b_pipeline = ComputePipeline::new(device, &matmul_tiled_bf16_b_layout, &matmul_tiled_bf16_b_mod, "main")?;
        let matmul_coop_pipeline = match (&matmul_coop_mod, &matmul_coop_layout) {
            (Some(m), Some(l)) => Some(ComputePipeline::new(device, l, m, "main")?),
            _ => None,
        };
        let matmul_coop_bf16_bf16_pipeline = match (&matmul_coop_bf16_bf16_mod, &matmul_coop_bf16_bf16_layout) {
            (Some(m), Some(l)) => Some(ComputePipeline::new(device, l, m, "main")?),
            _ => None,
        };
        let matmul_coop_f16_f16_pipeline = match (&matmul_coop_f16_f16_mod, &matmul_coop_f16_f16_layout) {
            (Some(m), Some(l)) => Some(ComputePipeline::new(device, l, m, "main")?),
            _ => None,
        };
        let matmul_coop_bf16_bf16_bf16_pipeline = match (&matmul_coop_bf16_bf16_bf16_mod, &matmul_coop_bf16_bf16_bf16_layout) {
            (Some(m), Some(l)) => Some(ComputePipeline::new(device, l, m, "main")?),
            _ => None,
        };
        let softmax_pipeline = ComputePipeline::new(device, &softmax_layout, &softmax_mod, "main")?;
        let softmax_f16_pipeline = ComputePipeline::new(device, &softmax_f16_layout, &softmax_f16_mod, "main")?;
        let softmax_bf16_pipeline = ComputePipeline::new(device, &softmax_bf16_layout, &softmax_bf16_mod, "main")?;
        let softmax_f64_pipeline = ComputePipeline::new(device, &softmax_f64_layout, &softmax_f64_mod, "main")?;
        let reduce_pipeline = ComputePipeline::new(device, &reduce_layout, &reduce_mod, "main")?;
        let reduce_f16_pipeline = ComputePipeline::new(device, &reduce_f16_layout, &reduce_f16_mod, "main")?;
        let reduce_bf16_pipeline = ComputePipeline::new(device, &reduce_bf16_layout, &reduce_bf16_mod, "main")?;
        let reduce_f64_pipeline = ComputePipeline::new(device, &reduce_f64_layout, &reduce_f64_mod, "main")?;
        let cast_f32_to_f64_pipeline = ComputePipeline::new(device, &cast_f32_to_f64_layout, &cast_f32_to_f64_mod, "main")?;
        let cast_f64_to_f32_pipeline = ComputePipeline::new(device, &cast_f64_to_f32_layout, &cast_f64_to_f32_mod, "main")?;
        let reduce_last_dim_pipeline = ComputePipeline::new(device, &reduce_last_dim_layout, &reduce_last_dim_mod, "main")?;
        let arg_reduce_last_dim_f32_pipeline  = ComputePipeline::new(device, &arg_reduce_last_dim_f32_layout,  &arg_reduce_last_dim_f32_mod,  "main")?;
        let scatter_add_f32_pipeline = ComputePipeline::new(device, &scatter_add_f32_layout, &scatter_add_f32_mod, "main")?;
        let scatter_add_f64_pipeline = ComputePipeline::new(device, &scatter_add_f64_layout, &scatter_add_f64_mod, "main")?;
        let scatter_add_bf16_pipeline = ComputePipeline::new(device, &scatter_add_bf16_layout, &scatter_add_bf16_mod, "main")?;
        let scatter_add_f16_pipeline = ComputePipeline::new(device, &scatter_add_f16_layout, &scatter_add_f16_mod, "main")?;
        let arg_reduce_last_dim_f16_pipeline  = ComputePipeline::new(device, &arg_reduce_last_dim_f16_layout,  &arg_reduce_last_dim_f16_mod,  "main")?;
        let arg_reduce_last_dim_bf16_pipeline = ComputePipeline::new(device, &arg_reduce_last_dim_bf16_layout, &arg_reduce_last_dim_bf16_mod, "main")?;
        let arg_reduce_last_dim_f64_pipeline  = ComputePipeline::new(device, &arg_reduce_last_dim_f64_layout,  &arg_reduce_last_dim_f64_mod,  "main")?;
        let arg_reduce_any_dim_f32_pipeline   = ComputePipeline::new(device, &arg_reduce_any_dim_f32_layout,   &arg_reduce_any_dim_f32_mod,   "main")?;
        let arg_reduce_any_dim_f64_pipeline   = ComputePipeline::new(device, &arg_reduce_any_dim_f64_layout,   &arg_reduce_any_dim_f64_mod,   "main")?;
        let arg_reduce_any_dim_bf16_pipeline  = ComputePipeline::new(device, &arg_reduce_any_dim_bf16_layout,  &arg_reduce_any_dim_bf16_mod,  "main")?;
        let arg_reduce_any_dim_f16_pipeline   = ComputePipeline::new(device, &arg_reduce_any_dim_f16_layout,   &arg_reduce_any_dim_f16_mod,   "main")?;
        let index_add_f32_pipeline  = ComputePipeline::new(device, &index_add_f32_layout,  &index_add_f32_mod,  "main")?;
        let index_add_f64_pipeline  = ComputePipeline::new(device, &index_add_f64_layout,  &index_add_f64_mod,  "main")?;
        let index_add_bf16_pipeline = ComputePipeline::new(device, &index_add_bf16_layout, &index_add_bf16_mod, "main")?;
        let index_add_f16_pipeline  = ComputePipeline::new(device, &index_add_f16_layout,  &index_add_f16_mod,  "main")?;
        let reduce_last_dim_f16_pipeline = ComputePipeline::new(device, &reduce_last_dim_f16_layout, &reduce_last_dim_f16_mod, "main")?;
        let reduce_last_dim_bf16_pipeline = ComputePipeline::new(device, &reduce_last_dim_bf16_layout, &reduce_last_dim_bf16_mod, "main")?;
        let reduce_last_dim_f64_pipeline = ComputePipeline::new(device, &reduce_last_dim_f64_layout, &reduce_last_dim_f64_mod, "main")?;
        let rms_norm_last_dim_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_layout, &rms_norm_last_dim_mod, "main")?;
        let rms_norm_last_dim_f16_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_f16_layout, &rms_norm_last_dim_f16_mod, "main")?;
        let rms_norm_last_dim_bf16_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_bf16_layout, &rms_norm_last_dim_bf16_mod, "main")?;
        let rms_norm_last_dim_f64_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_f64_layout, &rms_norm_last_dim_f64_mod, "main")?;
        let rms_norm_last_dim_backward_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_backward_layout, &rms_norm_last_dim_backward_mod, "main")?;
        let softmax_last_dim_backward_pipeline = ComputePipeline::new(device, &softmax_last_dim_backward_layout, &softmax_last_dim_backward_mod, "main")?;
        let softmax_last_dim_backward_f16_pipeline  = ComputePipeline::new(device, &softmax_last_dim_backward_f16_layout, &softmax_last_dim_backward_f16_mod, "main")?;
        let softmax_last_dim_backward_bf16_pipeline = ComputePipeline::new(device, &softmax_last_dim_backward_bf16_layout, &softmax_last_dim_backward_bf16_mod, "main")?;
        let softmax_last_dim_backward_f64_pipeline  = ComputePipeline::new(device, &softmax_last_dim_backward_f64_layout, &softmax_last_dim_backward_f64_mod, "main")?;
        let layer_norm_last_dim_backward_pipeline = ComputePipeline::new(device, &layer_norm_last_dim_backward_layout, &layer_norm_last_dim_backward_mod, "main")?;
        let layer_norm_last_dim_backward_f16_pipeline  = ComputePipeline::new(device, &layer_norm_last_dim_backward_f16_layout,  &layer_norm_last_dim_backward_f16_mod,  "main")?;
        let layer_norm_last_dim_backward_bf16_pipeline = ComputePipeline::new(device, &layer_norm_last_dim_backward_bf16_layout, &layer_norm_last_dim_backward_bf16_mod, "main")?;
        let layer_norm_last_dim_backward_f64_pipeline  = ComputePipeline::new(device, &layer_norm_last_dim_backward_f64_layout,  &layer_norm_last_dim_backward_f64_mod,  "main")?;
        let layer_norm_last_dim_pipeline      = ComputePipeline::new(device, &layer_norm_last_dim_layout,      &layer_norm_last_dim_mod,      "main")?;
        let layer_norm_last_dim_f16_pipeline  = ComputePipeline::new(device, &layer_norm_last_dim_f16_layout,  &layer_norm_last_dim_f16_mod,  "main")?;
        let layer_norm_last_dim_bf16_pipeline = ComputePipeline::new(device, &layer_norm_last_dim_bf16_layout, &layer_norm_last_dim_bf16_mod, "main")?;
        let layer_norm_last_dim_f64_pipeline  = ComputePipeline::new(device, &layer_norm_last_dim_f64_layout,  &layer_norm_last_dim_f64_mod,  "main")?;
        let strided_copy_pipeline = ComputePipeline::new(device, &strided_copy_layout, &strided_copy_mod, "main")?;
        let index_select_pipeline = ComputePipeline::new(device, &index_select_layout, &index_select_mod, "main")?;
        let index_select_f16_pipeline = ComputePipeline::new(device, &index_select_f16_layout, &index_select_f16_mod, "main")?;
        let index_select_bf16_pipeline = ComputePipeline::new(device, &index_select_bf16_layout, &index_select_bf16_mod, "main")?;
        let index_select_f64_pipeline = ComputePipeline::new(device, &index_select_f64_layout, &index_select_f64_mod, "main")?;
        let add_assign_scaled_pipeline = ComputePipeline::new(device, &add_assign_scaled_layout, &add_assign_scaled_mod, "main")?;
        let rope_pipeline = ComputePipeline::new(device, &rope_layout, &rope_mod, "main")?;
        let rope_f16_pipeline = ComputePipeline::new(device, &rope_f16_layout, &rope_f16_mod, "main")?;
        let rope_bf16_pipeline = ComputePipeline::new(device, &rope_bf16_layout, &rope_bf16_mod, "main")?;
        let rope_f64_pipeline = ComputePipeline::new(device, &rope_f64_layout, &rope_f64_mod, "main")?;
        let concat_along_dim_pipeline = ComputePipeline::new(device, &concat_along_dim_layout, &concat_along_dim_mod, "main")?;
        let concat_along_dim_f16_pipeline = ComputePipeline::new(device, &concat_along_dim_f16_layout, &concat_along_dim_f16_mod, "main")?;
        let concat_along_dim_bf16_pipeline = ComputePipeline::new(device, &concat_along_dim_bf16_layout, &concat_along_dim_bf16_mod, "main")?;
        let concat_along_dim_f64_pipeline = ComputePipeline::new(device, &concat_along_dim_f64_layout, &concat_along_dim_f64_mod, "main")?;
        let conv2d_im2col_pipeline = ComputePipeline::new(device, &conv2d_im2col_layout, &conv2d_im2col_mod, "main")?;
        let flash_attention_pipeline = ComputePipeline::new(device, &flash_attention_layout, &flash_attention_mod, "main")?;
        let dequant_q4_0_pipeline = ComputePipeline::new(device, &dequant_q4_0_layout, &dequant_q4_0_mod, "main")?;
        let dequant_q4_km_pipeline = ComputePipeline::new(device, &dequant_q4_km_layout, &dequant_q4_km_mod, "main")?;
        let dequant_q8_0_pipeline = ComputePipeline::new(device, &dequant_q8_0_layout, &dequant_q8_0_mod, "main")?;
        let qmatvec_q4_0_pipeline = ComputePipeline::new(device, &qmatvec_q4_0_layout, &qmatvec_q4_0_mod, "main")?;
        let matmul_q4_0_tiled_pipeline = ComputePipeline::new(device, &matmul_q4_0_tiled_layout, &matmul_q4_0_tiled_mod, "main")?;
        let quantize_q8_0_pipeline = ComputePipeline::new(device, &quantize_q8_0_layout, &quantize_q8_0_mod, "main")?;

        Ok(Self {
            layout_2s1u, layout_3s1u, layout_4s1u, layout_5s1u,
            unary_pipeline, unary_layout,
            unary_f16_pipeline, unary_f16_layout,
            unary_f64_pipeline, unary_f64_layout,
            unary_bf16_pipeline, unary_bf16_layout,
            binary_pipeline, binary_layout,
            binary_f16_pipeline, binary_f16_layout,
            binary_f64_pipeline, binary_f64_layout,
            binary_bf16_pipeline, binary_bf16_layout,
            affine_pipeline, affine_layout,
            affine_f64_pipeline,  affine_f64_layout,
            affine_f16_pipeline,  affine_f16_layout,
            affine_bf16_pipeline, affine_bf16_layout,
            clamp_pipeline, clamp_layout,
            powi_pipeline, powi_layout,
            cast_f32_to_f16_pipeline, cast_f32_to_f16_layout,
            cast_f16_to_f32_pipeline, cast_f16_to_f32_layout,
            cast_f32_to_bf16_pipeline, cast_f32_to_bf16_layout,
            cast_bf16_to_f32_pipeline, cast_bf16_to_f32_layout,
            cast_f32_to_f8e4m3_pipeline, cast_f32_to_f8e4m3_layout,
            cast_f8e4m3_to_f32_pipeline, cast_f8e4m3_to_f32_layout,
            cast_f16_to_f8e4m3_pipeline, cast_f16_to_f8e4m3_layout,
            cast_f8e4m3_to_f16_pipeline, cast_f8e4m3_to_f16_layout,
            cast_bf16_to_f8e4m3_pipeline, cast_bf16_to_f8e4m3_layout,
            cast_f8e4m3_to_bf16_pipeline, cast_f8e4m3_to_bf16_layout,
            write_slice_b1_pipeline, write_slice_b1_layout,
            write_slice_b2_pipeline, write_slice_b2_layout,
            write_slice_b4_pipeline, write_slice_b4_layout,
            pad_const_b1_pipeline, pad_const_b1_layout,
            pad_const_b2_pipeline, pad_const_b2_layout,
            pad_const_b4_pipeline, pad_const_b4_layout,
            pad_const_b8_pipeline, pad_const_b8_layout,
            pad_reflect_b1_pipeline, pad_reflect_b1_layout,
            pad_reflect_b2_pipeline, pad_reflect_b2_layout,
            pad_reflect_b4_pipeline, pad_reflect_b4_layout,
            pad_reflect_b8_pipeline, pad_reflect_b8_layout,
            pad_replicate_b1_pipeline, pad_replicate_b1_layout,
            pad_replicate_b2_pipeline, pad_replicate_b2_layout,
            pad_replicate_b4_pipeline, pad_replicate_b4_layout,
            pad_replicate_b8_pipeline, pad_replicate_b8_layout,
            pad_backward_const_b1_pipeline, pad_backward_const_b1_layout,
            pad_backward_const_b2_pipeline, pad_backward_const_b2_layout,
            pad_backward_const_b4_pipeline, pad_backward_const_b4_layout,
            pad_backward_const_b8_pipeline, pad_backward_const_b8_layout,
            pad_backward_reflect_f32_pipeline,   pad_backward_reflect_f32_layout,
            pad_backward_replicate_f32_pipeline, pad_backward_replicate_f32_layout,
            pad_backward_reflect_f64_pipeline,   pad_backward_reflect_f64_layout,
            pad_backward_replicate_f64_pipeline, pad_backward_replicate_f64_layout,
            pad_backward_reflect_bf16_pipeline,  pad_backward_reflect_bf16_layout,
            pad_backward_replicate_bf16_pipeline,pad_backward_replicate_bf16_layout,
            pad_backward_reflect_f16_pipeline,   pad_backward_reflect_f16_layout,
            pad_backward_replicate_f16_pipeline, pad_backward_replicate_f16_layout,
            masked_fill_b1_pipeline, masked_fill_b1_layout,
            masked_fill_b2_pipeline, masked_fill_b2_layout,
            masked_fill_b4_pipeline, masked_fill_b4_layout,
            masked_fill_b8_pipeline, masked_fill_b8_layout,
            gather_b1_pipeline, gather_b1_layout,
            gather_b2_pipeline, gather_b2_layout,
            gather_b4_pipeline, gather_b4_layout,
            gather_b8_pipeline, gather_b8_layout,
            write_slice_b8_pipeline, write_slice_b8_layout,
            strided_copy_signed_b2_pipeline, strided_copy_signed_b2_layout,
            strided_copy_signed_b4_pipeline, strided_copy_signed_b4_layout,
            strided_copy_signed_b8_pipeline, strided_copy_signed_b8_layout,
            triu_b2_pipeline, triu_b2_layout,
            triu_b4_pipeline, triu_b4_layout,
            triu_b8_pipeline, triu_b8_layout,
            tril_b2_pipeline, tril_b2_layout,
            tril_b4_pipeline, tril_b4_layout,
            tril_b8_pipeline, tril_b8_layout,
            flip_b2_pipeline, flip_b2_layout,
            flip_b4_pipeline, flip_b4_layout,
            flip_b8_pipeline, flip_b8_layout,
            roll_b2_pipeline, roll_b2_layout,
            roll_b4_pipeline, roll_b4_layout,
            roll_b8_pipeline, roll_b8_layout,
            cumsum_f32_pipeline, cumsum_f32_layout,
            cumsum_f64_pipeline, cumsum_f64_layout,
            cumsum_f16_pipeline, cumsum_f16_layout,
            cumsum_bf16_pipeline, cumsum_bf16_layout,
            matmul_pipeline, matmul_layout,
            matmul_tiled_pipeline, matmul_tiled_layout,
            matvec_pipeline, matvec_layout,
            matvec_bf16_b_pipeline, matvec_bf16_b_layout,
            matmul_tiled_bf16_b_pipeline, matmul_tiled_bf16_b_layout,
            matmul_coop_pipeline,
            matmul_coop_layout,
            matmul_coop_bf16_bf16_pipeline,
            matmul_coop_bf16_bf16_layout,
            matmul_coop_f16_f16_pipeline,
            matmul_coop_f16_f16_layout,
            matmul_coop_bf16_bf16_bf16_pipeline,
            matmul_coop_bf16_bf16_bf16_layout,
            softmax_pipeline, softmax_layout,
            softmax_f16_pipeline, softmax_f16_layout,
            softmax_bf16_pipeline, softmax_bf16_layout,
            softmax_f64_pipeline, softmax_f64_layout,
            reduce_pipeline, reduce_layout,
            reduce_f16_pipeline, reduce_f16_layout,
            reduce_bf16_pipeline, reduce_bf16_layout,
            reduce_f64_pipeline, reduce_f64_layout,
            cast_f32_to_f64_pipeline, cast_f32_to_f64_layout,
            cast_f64_to_f32_pipeline, cast_f64_to_f32_layout,
            reduce_last_dim_pipeline, reduce_last_dim_layout,
            arg_reduce_last_dim_f32_pipeline,  arg_reduce_last_dim_f32_layout,
            scatter_add_f32_pipeline, scatter_add_f32_layout,
            scatter_add_f64_pipeline, scatter_add_f64_layout,
            scatter_add_bf16_pipeline, scatter_add_bf16_layout,
            scatter_add_f16_pipeline, scatter_add_f16_layout,
            arg_reduce_last_dim_f16_pipeline,  arg_reduce_last_dim_f16_layout,
            arg_reduce_last_dim_bf16_pipeline, arg_reduce_last_dim_bf16_layout,
            arg_reduce_last_dim_f64_pipeline,  arg_reduce_last_dim_f64_layout,
            arg_reduce_any_dim_f32_pipeline,   arg_reduce_any_dim_f32_layout,
            arg_reduce_any_dim_f64_pipeline,   arg_reduce_any_dim_f64_layout,
            arg_reduce_any_dim_bf16_pipeline,  arg_reduce_any_dim_bf16_layout,
            arg_reduce_any_dim_f16_pipeline,   arg_reduce_any_dim_f16_layout,
            index_add_f32_pipeline,  index_add_f32_layout,
            index_add_f64_pipeline,  index_add_f64_layout,
            index_add_bf16_pipeline, index_add_bf16_layout,
            index_add_f16_pipeline,  index_add_f16_layout,
            reduce_last_dim_f16_pipeline, reduce_last_dim_f16_layout,
            reduce_last_dim_bf16_pipeline, reduce_last_dim_bf16_layout,
            reduce_last_dim_f64_pipeline, reduce_last_dim_f64_layout,
            rms_norm_last_dim_pipeline, rms_norm_last_dim_layout,
            rms_norm_last_dim_f16_pipeline, rms_norm_last_dim_f16_layout,
            rms_norm_last_dim_bf16_pipeline, rms_norm_last_dim_bf16_layout,
            rms_norm_last_dim_f64_pipeline, rms_norm_last_dim_f64_layout,
            rms_norm_last_dim_backward_pipeline, rms_norm_last_dim_backward_layout,
            softmax_last_dim_backward_pipeline, softmax_last_dim_backward_layout,
            softmax_last_dim_backward_f16_pipeline,  softmax_last_dim_backward_f16_layout,
            softmax_last_dim_backward_bf16_pipeline, softmax_last_dim_backward_bf16_layout,
            softmax_last_dim_backward_f64_pipeline,  softmax_last_dim_backward_f64_layout,
            layer_norm_last_dim_backward_pipeline, layer_norm_last_dim_backward_layout,
            layer_norm_last_dim_backward_f16_pipeline,  layer_norm_last_dim_backward_f16_layout,
            layer_norm_last_dim_backward_bf16_pipeline, layer_norm_last_dim_backward_bf16_layout,
            layer_norm_last_dim_backward_f64_pipeline,  layer_norm_last_dim_backward_f64_layout,
            layer_norm_last_dim_pipeline,      layer_norm_last_dim_layout,
            layer_norm_last_dim_f16_pipeline,  layer_norm_last_dim_f16_layout,
            layer_norm_last_dim_bf16_pipeline, layer_norm_last_dim_bf16_layout,
            layer_norm_last_dim_f64_pipeline,  layer_norm_last_dim_f64_layout,
            strided_copy_pipeline, strided_copy_layout,
            index_select_pipeline, index_select_layout,
            index_select_f16_pipeline, index_select_f16_layout,
            index_select_bf16_pipeline, index_select_bf16_layout,
            index_select_f64_pipeline, index_select_f64_layout,
            add_assign_scaled_pipeline, add_assign_scaled_layout,
            rope_pipeline, rope_layout,
            rope_f16_pipeline, rope_f16_layout,
            rope_bf16_pipeline, rope_bf16_layout,
            rope_f64_pipeline, rope_f64_layout,
            concat_along_dim_pipeline, concat_along_dim_layout,
            concat_along_dim_f16_pipeline, concat_along_dim_f16_layout,
            concat_along_dim_bf16_pipeline, concat_along_dim_bf16_layout,
            concat_along_dim_f64_pipeline, concat_along_dim_f64_layout,
            conv2d_im2col_pipeline, conv2d_im2col_layout,
            flash_attention_pipeline, flash_attention_layout,
            dequant_q4_0_pipeline, dequant_q4_0_layout,
            dequant_q4_km_pipeline, dequant_q4_km_layout,
            dequant_q8_0_pipeline, dequant_q8_0_layout,
            qmatvec_q4_0_pipeline, qmatvec_q4_0_layout,
            matmul_q4_0_tiled_pipeline, matmul_q4_0_tiled_layout,
            quantize_q8_0_pipeline, quantize_q8_0_layout,
            desc_pool,
            retired_desc_pools: Mutex::new(Vec::new()),
            device: device.clone(),
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

