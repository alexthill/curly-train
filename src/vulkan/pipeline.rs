use super::buffer;
use super::context::VkContext;
use super::structs::{ShaderSpv, Vertex};
use super::swapchain::SwapchainProperties;

use ash::{vk, Device};
use std::{
    error::Error,
    ffi::CString,
    io::Cursor,
    mem::size_of_val,
};

#[derive(Copy, Clone)]
pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub geometry: Option<Geometry>,
}

impl Pipeline {
    pub fn new(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        shader_spv: ShaderSpv,
    ) -> Self {
        let (pipeline, layout) = Self::create_pipeline(
            device,
            swapchain_properties,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
            shader_spv,
        );

        Self {
            layout,
            pipeline,
            geometry: None,
        }
    }

    pub unsafe fn cleanup(&mut self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
        if let Some(g) = self.geometry.take() {
            g.cleanup(device);
        }
    }

    fn create_shader_module(
        device: &Device,
        bytes: &[u8],
    ) -> Result<vk::ShaderModule, Box<dyn Error>> {
        let mut cursor = Cursor::new(bytes);
        let code = ash::util::read_spv(&mut cursor)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        unsafe {
            Ok(device.create_shader_module(&create_info, None)?)
        }
    }

    fn create_pipeline(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
        shader_spv: ShaderSpv,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_shader_module = Self::create_shader_module(device, shader_spv.vert)
            .expect("failed to load vertex shader spv file");
        let fragment_shader_module = Self::create_shader_module(device, shader_spv.frag)
            .expect("failed to load fragment shader spv file");

        let entry_point_name = CString::new("main").unwrap();
        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point_name);
        let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point_name);
        let shader_states_infos = [vertex_shader_state_info, fragment_shader_state_info];

        let vertex_binding_descs = [Vertex::get_binding_description()];
        let vertex_attribute_descs = Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attribute_descs);

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_properties.extent.width as _,
            height: swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_properties.extent,
        };
        let scissors = [scissor];
        let viewport_info = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0);

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default());

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);
        let color_blend_attachments = [color_blend_attachment];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let layout = {
            let layouts = [descriptor_set_layout];
            let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
            unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() }
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_states_infos)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisampling_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipeline_infos = [pipeline_info];

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        };

        (pipeline, layout)
    }
}

#[derive(Copy, Clone)]
pub struct Geometry {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub index_count: usize,
}

impl Geometry {
    pub fn new(
        vk_context: &VkContext,
        transient_command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Self {
        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer_with_data::<u32, _>(
            vk_context,
            transient_command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
        );
        let (index_buffer, index_buffer_memory) = Self::create_buffer_with_data::<u16, _>(
            vk_context,
            transient_command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            indices,
        );

        Self {
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: indices.len(),
        }
    }

    pub unsafe fn cleanup(self, device: &Device) {
        device.free_memory(self.index_buffer_memory, None);
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.vertex_buffer_memory, None);
        device.destroy_buffer(self.vertex_buffer, None);
    }

    /// Create a buffer and its gpu memory and fill it.
    ///
    /// This function internally creates an host visible staging buffer and
    /// a device local buffer. The data is first copied from the cpu to the
    /// staging buffer. Then we copy the data from the staging buffer to the
    /// final buffer using a one-time command buffer.
    fn create_buffer_with_data<A, T: Copy>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let device = vk_context.device();
        let size = size_of_val(data) as vk::DeviceSize;
        let (staging_buffer, staging_memory, staging_mem_size) = buffer::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<A>() as _, staging_mem_size);
            align.copy_from_slice(data);
            device.unmap_memory(staging_memory);
        };

        let (buffer, memory, _) = buffer::create_buffer(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        buffer::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer,
            buffer,
            size,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        };

        (buffer, memory)
    }
}
