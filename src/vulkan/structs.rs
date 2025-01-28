use crate::math::{Matrix4, Vector3};

use ash::vk;
use std::mem::offset_of;

#[derive(Debug, Clone, Copy)]
pub struct ShaderSpv {
    pub vert: &'static [u8],
    pub frag: &'static [u8],
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub coords: [f32; 2],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let position_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as _);
        let color_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as _);
        let coords_desc = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, coords) as _);
        [position_desc, color_desc, coords_desc]
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
#[repr(C)]
pub struct UniformBufferObject {
    pub model: Matrix4,
    pub view: Matrix4,
    pub proj: Matrix4,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_binding<'a>() -> vk::DescriptorSetLayoutBinding<'a> {
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
    }

    pub fn view_matrix() -> Matrix4 {
        Matrix4::look_at_rh(
            Vector3::from([0., 0., 3.]),
            Vector3::from([0., 0., 0.]),
            Vector3::from([0., 1., 0.]),
        )
    }

    pub fn model_matrix(extent_min: Vector3, extent_max: Vector3) -> Matrix4 {
        let model_sizes = extent_max - extent_min;
        let max_size = model_sizes.x().max(model_sizes.y()).max(model_sizes.z());
        let scale = Matrix4::from_scale(1. / max_size);
        let translate = Matrix4::from_translation(-extent_min - model_sizes / 2.);
        scale * translate
    }
}
