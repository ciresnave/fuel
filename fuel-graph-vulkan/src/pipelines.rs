//! Compute pipeline management for WGSL shaders.
//!
//! Compiles WGSL → SPIR-V via naga at backend init, creates Vulkan
//! compute pipelines, and provides typed dispatch helpers.

use vulkane::safe::*;

/// All pre-compiled compute pipelines for the Vulkan backend.
pub struct Pipelines {
    // Descriptor set layout: varies per shader (1, 2, or 3 storage buffers).
    pub layout_1buf: DescriptorSetLayout,  // unary, affine, reduce
    pub layout_2buf: DescriptorSetLayout,  // softmax (read + write of same shape)
    pub layout_3buf: DescriptorSetLayout,  // binary, matmul

    pub unary_pipeline: ComputePipeline,
    pub unary_layout: PipelineLayout,

    pub binary_pipeline: ComputePipeline,
    pub binary_layout: PipelineLayout,

    pub affine_pipeline: ComputePipeline,
    pub affine_layout: PipelineLayout,

    pub matmul_pipeline: ComputePipeline,
    pub matmul_layout: PipelineLayout,

    pub softmax_pipeline: ComputePipeline,
    pub softmax_layout: PipelineLayout,

    pub reduce_pipeline: ComputePipeline,
    pub reduce_layout: PipelineLayout,

    pub desc_pool: DescriptorPool,
}

impl Pipelines {
    pub fn new(device: &Device) -> Result<Self> {
        // Descriptor set layouts.
        let layout_1buf = DescriptorSetLayout::new(device, &[
            DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
            },
        ])?;

        let layout_3buf = DescriptorSetLayout::new(device, &[
            DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
            },
            DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
            },
        ])?;

        // Descriptor pool (large enough for many dispatches).
        let desc_pool = DescriptorPool::new(device, 1024, &[
            DescriptorPoolSize {
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4096,
            },
        ])?;

        // Compile WGSL → SPIR-V from the shared shader sources.
        use fuel_graph_executor::shaders;
        let unary_spv = compile_wgsl(shaders::UNARY)?;
        let binary_spv = compile_wgsl(shaders::BINARY)?;
        let affine_spv = compile_wgsl(shaders::AFFINE)?;
        let matmul_spv = compile_wgsl(shaders::MATMUL)?;
        let softmax_spv = compile_wgsl(shaders::SOFTMAX)?;
        let reduce_spv = compile_wgsl(shaders::REDUCE)?;

        let unary_mod = ShaderModule::from_spirv(device, &unary_spv)?;
        let binary_mod = ShaderModule::from_spirv(device, &binary_spv)?;
        let affine_mod = ShaderModule::from_spirv(device, &affine_spv)?;
        let matmul_mod = ShaderModule::from_spirv(device, &matmul_spv)?;
        let softmax_mod = ShaderModule::from_spirv(device, &softmax_spv)?;
        let reduce_mod = ShaderModule::from_spirv(device, &reduce_spv)?;

        // Pipeline layouts with push constants.
        let pc_8 = PushConstantRange {
            stage_flags: ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 8,
        };
        let pc_16 = PushConstantRange {
            stage_flags: ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 16,
        };
        let pc_24 = PushConstantRange {
            stage_flags: ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 24,
        };

        let unary_layout = PipelineLayout::with_push_constants(
            device, &[&layout_1buf], &[pc_8],
        )?;
        let binary_layout = PipelineLayout::with_push_constants(
            device, &[&layout_3buf], &[pc_8],
        )?;
        let affine_layout = PipelineLayout::with_push_constants(
            device, &[&layout_1buf], &[pc_16],
        )?;
        let matmul_layout = PipelineLayout::with_push_constants(
            device, &[&layout_3buf], &[pc_24],
        )?;
        let softmax_layout = PipelineLayout::with_push_constants(
            device, &[&layout_1buf], &[pc_8],
        )?;
        let reduce_layout = PipelineLayout::with_push_constants(
            device, &[&layout_1buf], &[pc_8],
        )?;

        // Pipelines.
        let unary_pipeline = ComputePipeline::new(device, &unary_layout, &unary_mod, "main")?;
        let binary_pipeline = ComputePipeline::new(device, &binary_layout, &binary_mod, "main")?;
        let affine_pipeline = ComputePipeline::new(device, &affine_layout, &affine_mod, "main")?;
        let matmul_pipeline = ComputePipeline::new(device, &matmul_layout, &matmul_mod, "main")?;
        let softmax_pipeline = ComputePipeline::new(device, &softmax_layout, &softmax_mod, "main")?;
        let reduce_pipeline = ComputePipeline::new(device, &reduce_layout, &reduce_mod, "main")?;

        Ok(Self {
            layout_1buf,
            layout_2buf: layout_1buf.clone(),  // TODO: proper 2-buf layout
            layout_3buf,
            unary_pipeline, unary_layout,
            binary_pipeline, binary_layout,
            affine_pipeline, affine_layout,
            matmul_pipeline, matmul_layout,
            softmax_pipeline, softmax_layout,
            reduce_pipeline, reduce_layout,
            desc_pool,
        })
    }
}

/// Compile WGSL source to SPIR-V words via naga.
fn compile_wgsl(source: &str) -> Result<Vec<u32>> {
    use vulkane::safe::naga_compile_wgsl;
    naga_compile_wgsl(source)
}
