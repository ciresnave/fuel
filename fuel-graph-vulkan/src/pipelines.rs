//! Compute pipeline management for the precompiled SPIR-V shaders
//! shipped in `fuel-graph-executor`.

use std::cell::RefCell;
use std::sync::OnceLock;
use vulkane::safe::*;

/// Shader-registry contents, lazily materialized as `&'static
/// [ShaderSource]` so it satisfies `ShaderRegistry::with_embedded`'s
/// `'static` bound. Built once on first access from
/// `fuel_graph_executor::shaders::EMBEDDED`.
fn embedded_shader_sources() -> &'static [ShaderSource] {
    static SOURCES: OnceLock<Vec<ShaderSource>> = OnceLock::new();
    SOURCES
        .get_or_init(|| {
            fuel_graph_executor::shaders::EMBEDDED
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
        .with_env_override(fuel_graph_executor::shaders::OVERRIDE_ENV)
}

/// All pre-compiled compute pipelines.
pub struct Pipelines {
    /// 2 storage + 1 uniform (unary, affine, softmax, reduce).
    pub layout_2s1u: DescriptorSetLayout,
    /// 3 storage + 1 uniform (binary, matmul).
    pub layout_3s1u: DescriptorSetLayout,
    /// 4 storage + 1 uniform (rope: x, cos, sin, out, params).
    pub layout_4s1u: DescriptorSetLayout,

    pub unary_pipeline: ComputePipeline,
    pub unary_layout: PipelineLayout,
    pub binary_pipeline: ComputePipeline,
    pub binary_layout: PipelineLayout,
    pub affine_pipeline: ComputePipeline,
    pub affine_layout: PipelineLayout,
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
    pub softmax_pipeline: ComputePipeline,
    pub softmax_layout: PipelineLayout,
    pub reduce_pipeline: ComputePipeline,
    pub reduce_layout: PipelineLayout,

    pub reduce_last_dim_pipeline: ComputePipeline,
    pub reduce_last_dim_layout: PipelineLayout,

    pub rms_norm_last_dim_pipeline: ComputePipeline,
    pub rms_norm_last_dim_layout: PipelineLayout,

    pub rms_norm_last_dim_backward_pipeline: ComputePipeline,
    pub rms_norm_last_dim_backward_layout: PipelineLayout,

    pub strided_copy_pipeline: ComputePipeline,
    pub strided_copy_layout: PipelineLayout,

    pub index_select_pipeline: ComputePipeline,
    pub index_select_layout: PipelineLayout,

    pub add_assign_scaled_pipeline: ComputePipeline,
    pub add_assign_scaled_layout: PipelineLayout,

    pub rope_pipeline: ComputePipeline,
    pub rope_layout: PipelineLayout,

    pub concat_along_dim_pipeline: ComputePipeline,
    pub concat_along_dim_layout: PipelineLayout,

    /// Active descriptor pool — the one new allocations come from.
    pub desc_pool: RefCell<DescriptorPool>,

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
    pub retired_desc_pools: RefCell<Vec<DescriptorPool>>,

    pub device: Device,
}

impl Pipelines {
    pub fn allocate_desc(&self, layout: &DescriptorSetLayout) -> Result<DescriptorSet> {
        let _span = tracing::debug_span!("vk_alloc_desc").entered();
        // Try allocating from the current pool.
        {
            let pool = self.desc_pool.borrow();
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
        let old = std::mem::replace(&mut *self.desc_pool.borrow_mut(), fresh);
        self.retired_desc_pools.borrow_mut().push(old);
        self.desc_pool.borrow().allocate(layout)
    }

    /// Drop every retired descriptor pool. Safe to call only AFTER a
    /// sync point that guarantees the GPU is done with every command
    /// buffer ever recorded against those pools. `VulkanBackend`
    /// calls this from `drain_recorder`, which itself runs after the
    /// D2H copy's fence has signaled.
    pub fn retire_pools_post_drain(&self) {
        let mut retired = self.retired_desc_pools.borrow_mut();
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
    pub fn new(device: &Device) -> Result<Self> {
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

        let desc_pool = RefCell::new(make_desc_pool(device)?);

        // Build the registry once and resolve every shader through it
        // — disk-override → embedded fallback, then straight to a
        // ShaderModule. No intermediate SPIR-V word vectors needed.
        use fuel_graph_executor::shaders;
        let registry = shader_registry();
        let unary_mod = registry.load_module(device, shaders::UNARY)?;
        let binary_mod = registry.load_module(device, shaders::BINARY)?;
        let affine_mod = registry.load_module(device, shaders::AFFINE)?;
        let matmul_mod = registry.load_module(device, shaders::MATMUL)?;
        let matmul_tiled_mod = registry.load_module(device, shaders::MATMUL_TILED_GLSL)?;
        let matvec_mod = registry.load_module(device, shaders::MATVEC_GLSL)?;
        let matvec_bf16_b_mod = registry.load_module(device, shaders::MATVEC_BF16_B_GLSL)?;
        let softmax_mod = registry.load_module(device, shaders::SOFTMAX)?;
        let reduce_mod = registry.load_module(device, shaders::REDUCE)?;
        let reduce_last_dim_mod = registry.load_module(device, shaders::REDUCE_LAST_DIM)?;
        let rms_norm_last_dim_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM)?;
        let rms_norm_last_dim_backward_mod = registry.load_module(device, shaders::RMS_NORM_LAST_DIM_BACKWARD)?;
        let strided_copy_mod = registry.load_module(device, shaders::STRIDED_COPY)?;
        let index_select_mod = registry.load_module(device, shaders::INDEX_SELECT)?;
        let add_assign_scaled_mod = registry.load_module(device, shaders::ADD_ASSIGN_SCALED)?;
        let rope_mod = registry.load_module(device, shaders::ROPE)?;
        let concat_along_dim_mod = registry.load_module(device, shaders::CONCAT_ALONG_DIM)?;

        // No push constants — params go through uniform buffers.
        let unary_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let binary_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let affine_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let matmul_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matmul_tiled_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matvec_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let matvec_bf16_b_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_last_dim_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rms_norm_last_dim_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        // backward takes 3 storage buffers (x, upstream, grad_x) + params
        let rms_norm_last_dim_backward_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let strided_copy_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let index_select_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let add_assign_scaled_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let rope_layout = PipelineLayout::new(device, &[&layout_4s1u])?;
        let concat_along_dim_layout = PipelineLayout::new(device, &[&layout_3s1u])?;

        let unary_pipeline = ComputePipeline::new(device, &unary_layout, &unary_mod, "main")?;
        let binary_pipeline = ComputePipeline::new(device, &binary_layout, &binary_mod, "main")?;
        let affine_pipeline = ComputePipeline::new(device, &affine_layout, &affine_mod, "main")?;
        let matmul_pipeline = ComputePipeline::new(device, &matmul_layout, &matmul_mod, "main")?;
        let matmul_tiled_pipeline = ComputePipeline::new(device, &matmul_tiled_layout, &matmul_tiled_mod, "main")?;
        let matvec_pipeline = ComputePipeline::new(device, &matvec_layout, &matvec_mod, "main")?;
        let matvec_bf16_b_pipeline = ComputePipeline::new(device, &matvec_bf16_b_layout, &matvec_bf16_b_mod, "main")?;
        let softmax_pipeline = ComputePipeline::new(device, &softmax_layout, &softmax_mod, "main")?;
        let reduce_pipeline = ComputePipeline::new(device, &reduce_layout, &reduce_mod, "main")?;
        let reduce_last_dim_pipeline = ComputePipeline::new(device, &reduce_last_dim_layout, &reduce_last_dim_mod, "main")?;
        let rms_norm_last_dim_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_layout, &rms_norm_last_dim_mod, "main")?;
        let rms_norm_last_dim_backward_pipeline = ComputePipeline::new(device, &rms_norm_last_dim_backward_layout, &rms_norm_last_dim_backward_mod, "main")?;
        let strided_copy_pipeline = ComputePipeline::new(device, &strided_copy_layout, &strided_copy_mod, "main")?;
        let index_select_pipeline = ComputePipeline::new(device, &index_select_layout, &index_select_mod, "main")?;
        let add_assign_scaled_pipeline = ComputePipeline::new(device, &add_assign_scaled_layout, &add_assign_scaled_mod, "main")?;
        let rope_pipeline = ComputePipeline::new(device, &rope_layout, &rope_mod, "main")?;
        let concat_along_dim_pipeline = ComputePipeline::new(device, &concat_along_dim_layout, &concat_along_dim_mod, "main")?;

        Ok(Self {
            layout_2s1u, layout_3s1u, layout_4s1u,
            unary_pipeline, unary_layout,
            binary_pipeline, binary_layout,
            affine_pipeline, affine_layout,
            matmul_pipeline, matmul_layout,
            matmul_tiled_pipeline, matmul_tiled_layout,
            matvec_pipeline, matvec_layout,
            matvec_bf16_b_pipeline, matvec_bf16_b_layout,
            softmax_pipeline, softmax_layout,
            reduce_pipeline, reduce_layout,
            reduce_last_dim_pipeline, reduce_last_dim_layout,
            rms_norm_last_dim_pipeline, rms_norm_last_dim_layout,
            rms_norm_last_dim_backward_pipeline, rms_norm_last_dim_backward_layout,
            strided_copy_pipeline, strided_copy_layout,
            index_select_pipeline, index_select_layout,
            add_assign_scaled_pipeline, add_assign_scaled_layout,
            rope_pipeline, rope_layout,
            concat_along_dim_pipeline, concat_along_dim_layout,
            desc_pool,
            retired_desc_pools: RefCell::new(Vec::new()),
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

