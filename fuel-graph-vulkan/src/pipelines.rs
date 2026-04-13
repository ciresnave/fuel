//! Compute pipeline management for WGSL shaders.

use vulkane::safe::*;

/// All pre-compiled compute pipelines.
pub struct Pipelines {
    /// 2 storage + 1 uniform (unary, affine, softmax, reduce).
    pub layout_2s1u: DescriptorSetLayout,
    /// 3 storage + 1 uniform (binary, matmul).
    pub layout_3s1u: DescriptorSetLayout,

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

        let desc_pool = DescriptorPool::new(device, 2048, &[
            DescriptorPoolSize {
                descriptor_type: DescriptorType::STORAGE_BUFFER,
                descriptor_count: 8192,
            },
            DescriptorPoolSize {
                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 2048,
            },
        ])?;

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

        // No push constants — params go through uniform buffers.
        let unary_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let binary_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let affine_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let matmul_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_layout = PipelineLayout::new(device, &[&layout_2s1u])?;

        let unary_pipeline = ComputePipeline::new(device, &unary_layout, &unary_mod, "main")?;
        let binary_pipeline = ComputePipeline::new(device, &binary_layout, &binary_mod, "main")?;
        let affine_pipeline = ComputePipeline::new(device, &affine_layout, &affine_mod, "main")?;
        let matmul_pipeline = ComputePipeline::new(device, &matmul_layout, &matmul_mod, "main")?;
        let softmax_pipeline = ComputePipeline::new(device, &softmax_layout, &softmax_mod, "main")?;
        let reduce_pipeline = ComputePipeline::new(device, &reduce_layout, &reduce_mod, "main")?;

        Ok(Self {
            layout_2s1u, layout_3s1u,
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

fn compile_wgsl(source: &str) -> Result<Vec<u32>> {
    vulkane::safe::naga::compile_wgsl(source)
        .map_err(|e| Error::NagaCompile(format!("{e:?}")))
}
