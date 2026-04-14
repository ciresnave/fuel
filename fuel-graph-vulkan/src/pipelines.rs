//! Compute pipeline management for WGSL shaders.

use std::cell::RefCell;
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

    pub strided_copy_pipeline: ComputePipeline,
    pub strided_copy_layout: PipelineLayout,

    /// Descriptor pool wrapped in RefCell so dispatch helpers can
    /// recreate it on VK_ERROR_OUT_OF_POOL_MEMORY. Each pool holds a
    /// bounded number of descriptor sets; when it fills, we drop it
    /// and allocate a fresh one. The GPU work that used the old sets
    /// has already completed (one_shot synchronizes) so this is safe.
    pub desc_pool: RefCell<DescriptorPool>,
    pub device: Device,
}

impl Pipelines {
    pub fn allocate_desc(&self, layout: &DescriptorSetLayout) -> Result<DescriptorSet> {
        // Try allocating from the current pool.
        {
            let pool = self.desc_pool.borrow();
            match pool.allocate(layout) {
                Ok(d) => return Ok(d),
                Err(Error::Vk(code)) if is_pool_oom(code) => { /* fall through to recreate */ }
                Err(e) => return Err(e),
            }
        }
        // Recreate a fresh pool and try again.
        let fresh = make_desc_pool(&self.device)?;
        *self.desc_pool.borrow_mut() = fresh;
        self.desc_pool.borrow().allocate(layout)
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

        let desc_pool = RefCell::new(make_desc_pool(device)?);

        use fuel_graph_executor::shaders;
        let unary_spv = compile_wgsl(shaders::UNARY)?;
        let binary_spv = compile_wgsl(shaders::BINARY)?;
        let affine_spv = compile_wgsl(shaders::AFFINE)?;
        let matmul_spv = compile_wgsl(shaders::MATMUL)?;
        let softmax_spv = compile_wgsl(shaders::SOFTMAX)?;
        let reduce_spv = compile_wgsl(shaders::REDUCE)?;
        let strided_copy_spv = compile_wgsl(shaders::STRIDED_COPY)?;

        let unary_mod = ShaderModule::from_spirv(device, &unary_spv)?;
        let binary_mod = ShaderModule::from_spirv(device, &binary_spv)?;
        let affine_mod = ShaderModule::from_spirv(device, &affine_spv)?;
        let matmul_mod = ShaderModule::from_spirv(device, &matmul_spv)?;
        let softmax_mod = ShaderModule::from_spirv(device, &softmax_spv)?;
        let reduce_mod = ShaderModule::from_spirv(device, &reduce_spv)?;
        let strided_copy_mod = ShaderModule::from_spirv(device, &strided_copy_spv)?;

        // No push constants — params go through uniform buffers.
        let unary_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let binary_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let affine_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let matmul_layout = PipelineLayout::new(device, &[&layout_3s1u])?;
        let softmax_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let reduce_layout = PipelineLayout::new(device, &[&layout_2s1u])?;
        let strided_copy_layout = PipelineLayout::new(device, &[&layout_3s1u])?;

        let unary_pipeline = ComputePipeline::new(device, &unary_layout, &unary_mod, "main")?;
        let binary_pipeline = ComputePipeline::new(device, &binary_layout, &binary_mod, "main")?;
        let affine_pipeline = ComputePipeline::new(device, &affine_layout, &affine_mod, "main")?;
        let matmul_pipeline = ComputePipeline::new(device, &matmul_layout, &matmul_mod, "main")?;
        let softmax_pipeline = ComputePipeline::new(device, &softmax_layout, &softmax_mod, "main")?;
        let reduce_pipeline = ComputePipeline::new(device, &reduce_layout, &reduce_mod, "main")?;
        let strided_copy_pipeline = ComputePipeline::new(device, &strided_copy_layout, &strided_copy_mod, "main")?;

        Ok(Self {
            layout_2s1u, layout_3s1u,
            unary_pipeline, unary_layout,
            binary_pipeline, binary_layout,
            affine_pipeline, affine_layout,
            matmul_pipeline, matmul_layout,
            softmax_pipeline, softmax_layout,
            reduce_pipeline, reduce_layout,
            strided_copy_pipeline, strided_copy_layout,
            desc_pool,
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

fn compile_wgsl(source: &str) -> Result<Vec<u32>> {
    vulkane::safe::naga::compile_wgsl(source)
        .map_err(|e| Error::NagaCompile(format!("{e:?}")))
}
