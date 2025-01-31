use crate::fs;
use crate::math::{self, Deg, Matrix4, Vector3};
use crate::obj::NormalizedObj;
use super::buffer;
use super::cmd;
use super::context::VkContext;
use super::debug::*;
use super::pipeline::{Geometry, Pipeline};
use super::structs::{ShaderSpv, UniformBufferObject, Vertex};
use super::swapchain::{SwapchainProperties, SwapchainSupportDetails};
use super::texture::Texture;

use anyhow::Context;
use ash::{
    ext::debug_utils,
    khr::{surface, swapchain as khr_swapchain},
    vk, Device, Entry, Instance,
};
use image::ImageReader;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    ffi::CString,
    mem::{align_of, size_of},
    path::Path,
};
use winit::window::Window;

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

pub struct VkApp {
    pub dirty_swapchain: bool,

    pub view_matrix: Matrix4,
    pub model_matrix: Matrix4,
    pub texture_weight: f32,
    pub cull_mode: vk::CullModeFlags,
    pub show_cubemap: bool,
    initial_model_matrix: Matrix4,
    model_extent: (Vector3, Vector3),

    vk_context: VkContext,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: khr_swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline: Pipeline,
    pipeline_cubemap: Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,
    msaa_samples: vk::SampleCountFlags,
    color_texture: Texture,
    depth_format: vk::Format,
    depth_texture: Texture,
    textures: [Texture; 2],
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffer_memories: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,
    in_flight_frames: InFlightFrames,
    shader_spv: ShaderSpv,
    cubemap_spv: ShaderSpv,
}

impl VkApp {
    pub fn new<P: AsRef<Path>>(
        window: &Window,
        width: u32,
        height: u32,
        image_path: P,
        nobj: NormalizedObj,
        shader_spv: ShaderSpv,
        cubemap_spv: ShaderSpv,
    ) -> Result<Self, anyhow::Error> {
        log::debug!("Creating application.");

        let entry = unsafe { Entry::load().expect("Failed to create entry.") };
        let instance = Self::create_instance(&entry, window);

        let surface = surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap()
        };

        let vk_context = VkContext::new(
            entry,
            instance,
            surface,
            surface_khr,
        ).context("Failed to create vulkan context")?;
        let graphics_queue = unsafe {
            vk_context.device().get_device_queue(vk_context.graphics_queue_index(), 0)
        };
        let present_queue = unsafe {
            vk_context.device().get_device_queue(vk_context.present_queue_index(), 0)
        };

        let (swapchain, swapchain_khr, properties, images) =
            Self::create_swapchain_and_images(&vk_context, [width, height]);
        let swapchain_image_views =
            Self::create_swapchain_image_views(vk_context.device(), &images, properties);

        let msaa_samples = vk_context.get_max_usable_sample_count();
        log::debug!("Chosen msaa: {msaa_samples:?}");
        let depth_format = Self::find_depth_format(&vk_context);

        let render_pass =
            Self::create_render_pass(vk_context.device(), properties, msaa_samples, depth_format);
        let descriptor_set_layout = Self::create_descriptor_set_layout(vk_context.device());

        let command_pool =
            vk_context.create_command_pool(vk::CommandPoolCreateFlags::empty());
        let transient_command_pool =
            vk_context.create_command_pool(vk::CommandPoolCreateFlags::TRANSIENT);

        let color_texture = Self::create_color_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            properties,
            msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &vk_context,
            command_pool,
            graphics_queue,
            depth_format,
            properties.extent,
            msaa_samples,
        );

        let swapchain_framebuffers = Self::create_framebuffers(
            vk_context.device(),
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        let texture = Self::create_texture_image(
            &vk_context,
            command_pool,
            graphics_queue,
            image_path,
        ).unwrap();
        let texture_cubemap = Self::create_cubemap(
            &vk_context,
            command_pool,
            graphics_queue,
            [
                "assets/cubemap/right.png",
                "assets/cubemap/left.png",
                "assets/cubemap/top.png",
                "assets/cubemap/bottom.png",
                "assets/cubemap/back.png",
                "assets/cubemap/front.png",
            ],
        ).unwrap();

        let (pipeline, model_extent) = {
            let mut pipeline = Pipeline::new(
                vk_context.device(),
                properties,
                vk::CullModeFlags::NONE,
                msaa_samples,
                render_pass,
                descriptor_set_layout,
                shader_spv,
            );
            let (vertices, indices, model_extent) = Self::load_model(nobj);
            pipeline.geometry = Some(Geometry::new(
                &vk_context,
                transient_command_pool,
                graphics_queue,
                &vertices,
                &indices,
            ));
            (pipeline, model_extent)
        };

        let pipeline_cubemap = {
            let mut pipeline = Pipeline::new(
                vk_context.device(),
                properties,
                vk::CullModeFlags::BACK,
                msaa_samples,
                render_pass,
                descriptor_set_layout,
                cubemap_spv,
            );
            let nobj = NormalizedObj::from_reader(fs::load("assets/cubemap/skybox.obj")?)?;
            let (vertices, indices, _) = Self::load_model(nobj);
            pipeline.geometry = Some(Geometry::new(
                &vk_context,
                transient_command_pool,
                graphics_queue,
                &vertices,
                &indices,
            ));
            pipeline
        };

        let (uniform_buffers, uniform_buffer_memories) =
            Self::create_uniform_buffers(&vk_context, images.len());

        let descriptor_pool = Self::create_descriptor_pool(vk_context.device(), images.len() as _);
        let descriptor_sets = Self::create_descriptor_sets(
            vk_context.device(),
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            &[texture, texture_cubemap],
            //&[texture],
        );

        let command_buffers = Self::create_and_register_command_buffers(
            vk_context.device(),
            command_pool,
            &swapchain_framebuffers,
            render_pass,
            properties,
            &descriptor_sets,
            &[pipeline_cubemap, pipeline],
        );

        let in_flight_frames = Self::create_sync_objects(vk_context.device());

        Ok(Self {
            view_matrix: UniformBufferObject::view_matrix(),
            model_matrix: Matrix4::unit(),
            initial_model_matrix: UniformBufferObject::model_matrix(
                model_extent.0,
                model_extent.1,
            ),
            texture_weight: 0.,
            cull_mode: vk::CullModeFlags::NONE,
            show_cubemap: true,
            model_extent,
            dirty_swapchain: false,
            vk_context,
            graphics_queue,
            present_queue,
            swapchain,
            swapchain_khr,
            swapchain_properties: properties,
            images,
            swapchain_image_views,
            render_pass,
            descriptor_set_layout,
            pipeline,
            pipeline_cubemap,
            swapchain_framebuffers,
            command_pool,
            transient_command_pool,
            msaa_samples,
            color_texture,
            depth_format,
            depth_texture,
            textures: [texture, texture_cubemap],
            uniform_buffers,
            uniform_buffer_memories,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            in_flight_frames,
            shader_spv,
            cubemap_spv,
        })
    }

    fn create_instance(entry: &Entry, window: &Window) -> Instance {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap();
        let mut extension_names = extension_names.to_vec();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(debug_utils::NAME.as_ptr());
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };
        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);
        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    /// Create the swapchain with optimal settings possible with `device`.
    ///
    /// # Returns
    ///
    /// A tuple containing the swapchain loader and the actual swapchain.
    fn create_swapchain_and_images(
        vk_context: &VkContext,
        dimensions: [u32; 2],
    ) -> (
        khr_swapchain::Device,
        vk::SwapchainKHR,
        SwapchainProperties,
        Vec<vk::Image>,
    ) {
        let details = SwapchainSupportDetails::new(
            vk_context.physical_device(),
            vk_context.surface(),
            vk_context.surface_khr(),
        );
        let properties = details.get_ideal_swapchain_properties(dimensions);

        let format = properties.format;
        let present_mode = properties.present_mode;
        let extent = properties.extent;
        let image_count = {
            let max = details.capabilities.max_image_count;
            let mut preferred = details.capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {:?}",
            format.format,
            format.color_space,
            present_mode,
            extent,
            image_count,
        );

        let graphics = vk_context.graphics_queue_index();
        let present = vk_context.present_queue_index();
        let families_indices = [graphics, present];

        let create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::default()
                .surface(vk_context.surface_khr())
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            builder = if graphics != present {
                builder
                    .image_sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families_indices)
            } else {
                builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            };

            builder
                .pre_transform(details.capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
        };

        let swapchain = khr_swapchain::Device::new(vk_context.instance(), vk_context.device());
        let swapchain_khr = unsafe { swapchain.create_swapchain(&create_info, None).unwrap() };
        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        (swapchain, swapchain_khr, properties, images)
    }

    /// Create one image view for each image of the swapchain.
    fn create_swapchain_image_views(
        device: &Device,
        swapchain_images: &[vk::Image],
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::ImageView> {
        swapchain_images.iter()
            .map(|image| {
                Self::create_image_view(
                    device,
                    *image,
                    1,
                    swapchain_properties.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_image_view(
        device: &Device,
        image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
    ) -> vk::ImageView {
        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe { device.create_image_view(&create_info, None).unwrap() }
    }

    fn create_render_pass(
        device: &Device,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_attachement_desc = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let resolve_attachment_desc = vk::AttachmentDescription::default()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let attachment_descs = [
            color_attachment_desc,
            depth_attachement_desc,
            resolve_attachment_desc,
        ];

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_refs = [color_attachment_ref];

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let resolve_attachment_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_attachment_refs = [resolve_attachment_ref];

        let subpass_desc = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .resolve_attachments(&resolve_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref);
        let subpass_descs = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            );
        let subpass_deps = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&subpass_deps);

        unsafe { device.create_render_pass(&render_pass_info, None).unwrap() }
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let ubo_binding = UniformBufferObject::get_descriptor_set_layout_binding();
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let cubemap_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let bindings = [ubo_binding, sampler_binding, cubemap_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            device.create_descriptor_set_layout(&layout_info, None).unwrap()
        }
    }

    /// Create a descriptor pool to allocate the descriptor sets.
    fn create_descriptor_pool(device: &Device, size: u32) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: size,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: size * 2,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(size);

        unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() }
    }

    /// Create one descriptor set for each uniform buffer.
    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        textures: &[Texture],
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| layout)
            .collect::<Vec<_>>();
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        for (set, buffer) in descriptor_sets.iter().zip(uniform_buffers.iter()) {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(*buffer)
                .offset(0)
                .range(size_of::<UniformBufferObject>() as vk::DeviceSize);
            let buffer_infos = [buffer_info];
            let ubo_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos);

            let model_texture = textures[0];
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(model_texture.view)
                .sampler(model_texture.sampler.unwrap());
            let image_infos = [image_info];
            let sampler_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos);

            let cubemap_texture = textures[1];
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(cubemap_texture.view)
                .sampler(cubemap_texture.sampler.unwrap());
            let image_infos2 = [image_info];
            let sampler2_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos2);

            let writes = [ubo_descriptor_write, sampler_descriptor_write, sampler2_descriptor_write];
            unsafe { device.update_descriptor_sets(&writes, &[]) }
        }

        descriptor_sets
    }

    fn create_framebuffers(
        device: &Device,
        image_views: &[vk::ImageView],
        color_texture: Texture,
        depth_texture: Texture,
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
    ) -> Vec<vk::Framebuffer> {
        image_views.iter()
            .map(|view| [color_texture.view, depth_texture.view, *view])
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(swapchain_properties.extent.width)
                    .height(swapchain_properties.extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<_>>()
    }

    fn create_color_texture(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        swapchain_properties: SwapchainProperties,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let format = swapchain_properties.format.format;
        let (image, memory) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            swapchain_properties.extent,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        );

        Self::transition_image_layout(
            vk_context.device(),
            command_pool,
            transition_queue,
            image,
            1,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
        );

        let view = Self::create_image_view(
            vk_context.device(),
            image,
            1,
            format,
            vk::ImageAspectFlags::COLOR,
        );

        Texture::new(image, memory, view, None)
    }

    /// Create the depth buffer texture (image, memory and view).
    ///
    /// This function also transitions the image to be ready to be used
    /// as a depth/stencil attachement.
    fn create_depth_texture(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        format: vk::Format,
        extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let (image, mem) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        let device = vk_context.device();
        Self::transition_image_layout(
            device,
            command_pool,
            transition_queue,
            image,
            1,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            1,
        );

        let view = Self::create_image_view(device, image, 1, format, vk::ImageAspectFlags::DEPTH);

        Texture::new(image, mem, view, None)
    }

    fn find_depth_format(vk_context: &VkContext) -> vk::Format {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        vk_context
            .find_supported_format(
                &candidates,
                vk::ImageTiling::OPTIMAL,
                vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
            )
            .expect("Failed to find a supported depth format")
    }

    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn create_cubemap<P: AsRef<Path>>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        copy_queue: vk::Queue,
        pathes: [P; 6],
    ) -> Result<Texture, anyhow::Error> {
        let mut dims = None;
        let mut images = Vec::new();
        for path in pathes.iter() {
            let image = ImageReader::open(path)
                .with_context(|| format!("Failed to open image at {:?}", path.as_ref()))?
                .decode()
                .with_context(|| format!("Failed to decode image at {:?}", path.as_ref()))?;
            let image_as_rgb = image.to_rgba8();
            let width = image_as_rgb.width();
            let height = image_as_rgb.height();
            if let Some((w, h)) = dims {
                if w != width || h != height {
                    return Err(anyhow::anyhow!("cubemap images must have all the same size"))
                }
            } else {
                dims = Some((width, height));
            }
            let pixels = image_as_rgb.into_raw();
            images.push(pixels);
        }
        let (width, height) = dims.unwrap();
        let max_mip_levels = ((width.min(height) as f32).log2().floor() + 1.0) as u32;
        let extent = vk::Extent2D { width, height };
        let image_size = (images[0].len() * size_of::<u8>()) as vk::DeviceSize;
        let device = vk_context.device();

        let (buffer, memory, mem_size) = buffer::create_buffer(
            vk_context,
            image_size * 6,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            for (i, image) in images.into_iter().enumerate() {
                let offset = image_size * i as vk::DeviceSize;
                let ptr = device
                    .map_memory(memory, offset, image_size, vk::MemoryMapFlags::empty())
                    .context("Failed to map memory for cubemap image")?;
                let mut align = ash::util::Align::new(ptr, align_of::<u8>() as _, mem_size);
                align.copy_from_slice(&image);
                device.unmap_memory(memory);
            }
        }

        let (image, image_memory) = {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(max_mip_levels)
                .array_layers(6)
                .format(vk::Format::R8G8B8A8_UNORM)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(vk::SampleCountFlags::TYPE_1)
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);
            let device = vk_context.device();
            let image = unsafe { device.create_image(&image_info, None).unwrap() };
            let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
            let mem_type_index = vk_context.find_memory_type(
                mem_requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type_index);
            let memory = unsafe {
                let mem = device.allocate_memory(&alloc_info, None).unwrap();
                device.bind_image_memory(image, mem, 0).unwrap();
                mem
            };
            (image, memory)
        };

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            Self::transition_image_layout(
                device,
                command_pool,
                copy_queue,
                image,
                max_mip_levels,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                6,
            );

            Self::copy_buffer_to_image(device, command_pool, copy_queue, buffer, image, extent, 6);

            Self::generate_mipmaps(
                vk_context,
                command_pool,
                copy_queue,
                image,
                extent,
                vk::Format::R8G8B8A8_UNORM,
                max_mip_levels,
                6,
            );
        }

        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }

        let create_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: max_mip_levels,
                base_array_layer: 0,
                layer_count: 6,
            });
        let image_view = unsafe {
            device.create_image_view(&create_info, None).unwrap()
        };

        let max_aniso = vk_context.physical_device_properties().limits.max_sampler_anisotropy;
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(max_aniso.max(16.))
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(max_mip_levels as _);
        let sampler = unsafe {
            device.create_sampler(&sampler_info, None)
                .context("Failed to create sampler for cubemap")?
        };

        Ok(Texture::new(image, image_memory, image_view, Some(sampler)))
    }

    fn create_texture_image<P: AsRef<Path>>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        copy_queue: vk::Queue,
        path: P,
    ) -> Result<Texture, anyhow::Error> {
        let image = ImageReader::open(path)
            .context("Failed to open image")?
            .decode()
            .context("Failed to decode image")?
            .flipv();
        let image_as_rgb = image.to_rgba8();
        let width = image_as_rgb.width();
        let height = image_as_rgb.height();
        let max_mip_levels = ((width.min(height) as f32).log2().floor() + 1.0) as u32;
        let extent = vk::Extent2D { width, height };
        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * size_of::<u8>()) as vk::DeviceSize;
        let device = vk_context.device();

        let (buffer, memory, mem_size) = buffer::create_buffer(
            vk_context,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let ptr = device.map_memory(memory, 0, image_size, vk::MemoryMapFlags::empty())
                .context("Failed to map memory for texture image")?;
            let mut align = ash::util::Align::new(ptr, align_of::<u8>() as _, mem_size);
            align.copy_from_slice(&pixels);
            device.unmap_memory(memory);
        }

        let (image, image_memory) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        );

        // Transition the image layout and copy the buffer into the image
        // and transition the layout again to be readable from fragment shader.
        {
            Self::transition_image_layout(
                device,
                command_pool,
                copy_queue,
                image,
                max_mip_levels,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                1,
            );

            Self::copy_buffer_to_image(device, command_pool, copy_queue, buffer, image, extent, 1);

            Self::generate_mipmaps(
                vk_context,
                command_pool,
                copy_queue,
                image,
                extent,
                vk::Format::R8G8B8A8_UNORM,
                max_mip_levels,
                1,
            );
        }

        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }

        let image_view = Self::create_image_view(
            device,
            image,
            max_mip_levels,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        );

        let max_aniso = vk_context.physical_device_properties().limits.max_sampler_anisotropy;
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(max_aniso.max(16.))
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(max_mip_levels as _);
        let sampler = unsafe {
            device.create_sampler(&sampler_info, None)
                .context("Failed to create sampler for texture")?
        };

        Ok(Texture::new(image, image_memory, image_view, Some(sampler)))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_image(
        vk_context: &VkContext,
        mem_properties: vk::MemoryPropertyFlags,
        extent: vk::Extent2D,
        mip_levels: u32,
        sample_count: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(vk::ImageCreateFlags::empty());

        let device = vk_context.device();
        let image = unsafe { device.create_image(&image_info, None).unwrap() };
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_type_index = vk_context.find_memory_type(mem_requirements, mem_properties);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        let memory = unsafe {
            let mem = device.allocate_memory(&alloc_info, None).unwrap();
            device.bind_image_memory(image, mem, 0).unwrap();
            mem
        };

        (image, memory)
    }

    #[allow(clippy::too_many_arguments)]
    fn transition_image_layout(
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        layer_count: u32,
    ) {
        cmd::execute_one_time_commands(device, command_pool, transition_queue, |buffer| {
            let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
                match (old_layout, new_layout) {
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ) => (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    ),
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    _ => panic!(
                        "Unsupported layout transition({:?} => {:?}).",
                        old_layout, new_layout
                    ),
                };

            let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                let mut mask = vk::ImageAspectFlags::DEPTH;
                if Self::has_stencil_component(format) {
                    mask |= vk::ImageAspectFlags::STENCIL;
                }
                mask
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count,
                })
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask);

            unsafe {
                device.cmd_pipeline_barrier(
                    buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
            };
        });
    }

    fn copy_buffer_to_image(
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        extent: vk::Extent2D,
        layer_count: u32,
    ) {
        cmd::execute_one_time_commands(device, command_pool, transition_queue, |command_buffer| {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            let regions = [region];
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_mipmaps(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        image: vk::Image,
        extent: vk::Extent2D,
        format: vk::Format,
        mip_levels: u32,
        layer_count: u32,
    ) {
        let format_properties = unsafe {
            vk_context.instance()
                .get_physical_device_format_properties(vk_context.physical_device(), format)
        };
        if !format_properties.optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("Linear blitting is not supported for format {:?}.", format)
        }

        cmd::execute_one_time_commands(
            vk_context.device(),
            command_pool,
            transfer_queue,
            |buffer| {
                let mut barrier = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count,
                        level_count: 1,
                        ..Default::default()
                    });

                let mut mip_width = extent.width as i32;
                let mut mip_height = extent.height as i32;
                for level in 1..mip_levels {
                    let next_mip_width = if mip_width > 1 {
                        mip_width / 2
                    } else {
                        mip_width
                    };
                    let next_mip_height = if mip_height > 1 {
                        mip_height / 2
                    } else {
                        mip_height
                    };

                    barrier.subresource_range.base_mip_level = level - 1;
                    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                    barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
                    let barriers = [barrier];

                    unsafe {
                        vk_context.device().cmd_pipeline_barrier(
                            buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &barriers,
                        )
                    };

                    let blit = vk::ImageBlit::default()
                        .src_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: mip_width,
                                y: mip_height,
                                z: 1,
                            },
                        ])
                        .src_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: level - 1,
                            base_array_layer: 0,
                            layer_count,
                        })
                        .dst_offsets([
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: next_mip_width,
                                y: next_mip_height,
                                z: 1,
                            },
                        ])
                        .dst_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: level,
                            base_array_layer: 0,
                            layer_count,
                        });
                    let blits = [blit];

                    unsafe {
                        vk_context.device().cmd_blit_image(
                            buffer,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &blits,
                            vk::Filter::LINEAR,
                        )
                    };

                    barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                    let barriers = [barrier];

                    unsafe {
                        vk_context.device().cmd_pipeline_barrier(
                            buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::FRAGMENT_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &barriers,
                        )
                    };

                    mip_width = next_mip_width;
                    mip_height = next_mip_height;
                }

                barrier.subresource_range.base_mip_level = mip_levels - 1;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                let barriers = [barrier];

                unsafe {
                    vk_context.device().cmd_pipeline_barrier(
                        buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    )
                };
            },
        );
    }

    fn load_model(nobj: NormalizedObj) -> (Vec<Vertex>, Vec<u32>, (Vector3, Vector3)) {
        let mut min = Vector3::new(f32::MAX);
        let mut max = Vector3::new(f32::MIN);
        let vertices = nobj.vertices.iter().map(|vertex| {
            for (i, &coord) in vertex.pos_coords.iter().enumerate() {
                min[i] = min[i].min(coord);
                max[i] = max[i].max(coord);
            }
            let tex_coords = if nobj.has_tex_coords {
                vertex.tex_coords
            } else {
                [vertex.pos_coords[2], vertex.pos_coords[1]]
            };
            Vertex {
                pos: vertex.pos_coords,
                color: [1.0, 1.0, 1.0],
                coords: tex_coords,
            }
        }).collect();

        (vertices, nobj.indices, (min, max))
    }

    fn create_uniform_buffers(
        vk_context: &VkContext,
        count: usize,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
        let size = size_of::<UniformBufferObject>() as vk::DeviceSize;
        let mut buffers = Vec::new();
        let mut memories = Vec::new();

        for _ in 0..count {
            let (buffer, memory, _) = buffer::create_buffer(
                vk_context,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buffer);
            memories.push(memory);
        }

        (buffers, memories)
    }

    fn recreate_command_buffers(&mut self) {
        let device = self.vk_context.device();
        unsafe {
            device.free_command_buffers(self.command_pool, &self.command_buffers);
        }

        let pipelines: &[Pipeline] = if self.show_cubemap {
            // render cubemap after object for performance gain
            // (avoids rendering the parts occluded by the object)
            &[self.pipeline, self.pipeline_cubemap]
        } else {
            &[self.pipeline]
        };
        self.command_buffers = Self::create_and_register_command_buffers(
            device,
            self.command_pool,
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain_properties,
            &self.descriptor_sets,
            pipelines,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn create_and_register_command_buffers(
        device: &Device,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_properties: SwapchainProperties,
        descriptor_sets: &[vk::DescriptorSet],
        pipelines: &[Pipeline],
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _);
        let buffers = unsafe { device.allocate_command_buffers(&allocate_info).unwrap() };

        for (i, &buffer) in buffers.iter().enumerate() {
            // begin command buffer
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
            unsafe {
                device.begin_command_buffer(buffer, &command_buffer_begin_info).unwrap()
            };

            // begin render pass
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            let render_pass_begin_info = vk::RenderPassBeginInfo::default()
                .render_pass(render_pass)
                .framebuffer(framebuffers[i])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain_properties.extent,
                })
                .clear_values(&clear_values);
            unsafe {
                device.cmd_begin_render_pass(
                    buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                )
            };

            for pipeline in pipelines {
                // bind pipeline, vertex and index buffer
                let mut index_count = 0;
                unsafe {
                    device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
                    if let Some(g) = pipeline.geometry {
                        device.cmd_bind_vertex_buffers(buffer, 0, &[g.vertex_buffer], &[0]);
                        device.cmd_bind_index_buffer(buffer, g.index_buffer, 0, vk::IndexType::UINT32);
                        index_count = g.index_count;
                    }
                };

                // bind descriptor set
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &descriptor_sets[i..=i],
                        &[],
                    )
                };

                unsafe { device.cmd_draw_indexed(buffer, index_count as _, 1, 0, 0, 0) };
            }

            // end render pass and command buffer
            unsafe {
                device.cmd_end_render_pass(buffer);
                device.end_command_buffer(buffer).unwrap();
            };
        }

        buffers
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects)
        }

        InFlightFrames::new(sync_objects_vec)
    }

    pub fn wait_gpu_idle(&self) {
        unsafe { self.vk_context.device().device_wait_idle().unwrap() };
    }

    /// Draws a frame.
    ///
    /// #Returns
    ///
    /// True if the swapchain is dirty and needs to be recreated.
    pub fn draw_frame(&mut self) -> bool {
        log::trace!("Drawing frame.");
        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;
        let wait_fences = [in_flight_fence];

        unsafe {
            self.vk_context.device().wait_for_fences(&wait_fences, true, u64::MAX).unwrap()
        };

        let result = unsafe {
            self.swapchain.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };
        let image_index = match result {
            // ignore suboptimal swap chain here because we already aquired an image
            Ok((image_index, _suboptimal)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return true;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        // it is important to only reset the fence when we know that we are going to do work
        unsafe { self.vk_context.device().reset_fences(&wait_fences).unwrap() };

        self.update_uniform_buffers(image_index);

        let device = self.vk_context.device();
        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // Submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            let submit_infos = [submit_info];
            unsafe {
                device.queue_submit(self.graphics_queue, &submit_infos, in_flight_fence).unwrap()
            };
        }

        let swapchains = [self.swapchain_khr];
        let images_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&images_indices);
        // .results() null since we only have one swapchain
        let result = unsafe {
            self.swapchain.queue_present(self.present_queue, &present_info)
        };
        match result {
            Ok(value) => value,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
        }
    }

    pub fn load_new_texture<P: AsRef<Path>>(&mut self, path: P) -> Result<(), anyhow::Error> {
        log::info!("Loading image {:?}", path.as_ref().as_os_str());
        self.wait_gpu_idle();

        let texture = Self::create_texture_image(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            path,
        )?;
        let device = self.vk_context.device();

        for set in self.descriptor_sets.iter() {
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(texture.sampler.unwrap());
            let image_infos = [image_info];
            let sampler_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(*set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos);
            unsafe { device.update_descriptor_sets(&[sampler_descriptor_write], &[]) }
        }

        self.recreate_command_buffers();
        Ok(())
    }

    pub fn load_new_model(&mut self, nobj: NormalizedObj) {
        let device = self.vk_context.device();
        let (vertices, indices, model_extent) = Self::load_model(nobj);
        self.initial_model_matrix = UniformBufferObject::model_matrix(
            model_extent.0,
            model_extent.1,
        );
        self.model_extent = model_extent;

        self.wait_gpu_idle();

        if let Some(g) = self.pipeline.geometry.take() {
            unsafe { g.cleanup(device) };
        }
        self.pipeline.geometry = Some(Geometry::new(
            &self.vk_context,
            self.transient_command_pool,
            self.graphics_queue,
            &vertices,
            &indices,
        ));

        self.recreate_command_buffers();
    }

    /// Recreates the swapchain with new dimensions.
    ///
    /// # Panics
    ///
    /// Panics if either `width` or `height` is zero.
    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        log::debug!("Recreating swapchain");
        if width == 0 || height == 0 {
            panic!("invalid dimensions: ({width}, {height})");
        }

        self.wait_gpu_idle();

        let geometry = self.pipeline.geometry.take();
        let geometry_cubemap = self.pipeline_cubemap.geometry.take();
        self.cleanup_swapchain();

        let device = self.vk_context.device();

        let dimensions = [width, height];
        let (swapchain, swapchain_khr, properties, images) = Self::create_swapchain_and_images(
            &self.vk_context,
            dimensions,
        );
        let swapchain_image_views = Self::create_swapchain_image_views(device, &images, properties);

        let render_pass =
            Self::create_render_pass(device, properties, self.msaa_samples, self.depth_format);
        let mut pipeline = Pipeline::new(
            device,
            properties,
            self.cull_mode,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
            self.shader_spv,
        );
        pipeline.geometry = geometry;

        let mut pipeline_cubemap = Pipeline::new(
            device,
            properties,
            vk::CullModeFlags::BACK,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
            self.cubemap_spv,
        );
        pipeline_cubemap.geometry = geometry_cubemap;

        let color_texture = Self::create_color_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            properties,
            self.msaa_samples,
        );

        let depth_texture = Self::create_depth_texture(
            &self.vk_context,
            self.command_pool,
            self.graphics_queue,
            self.depth_format,
            properties.extent,
            self.msaa_samples,
        );

        let swapchain_framebuffers = Self::create_framebuffers(
            device,
            &swapchain_image_views,
            color_texture,
            depth_texture,
            render_pass,
            properties,
        );

        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;
        self.swapchain_properties = properties;
        self.images = images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.pipeline_cubemap = pipeline_cubemap;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.recreate_command_buffers();
    }

    /// Clean up the swapchain and all resources that depend on it.
    fn cleanup_swapchain(&mut self) {
        let device = self.vk_context.device();
        unsafe {
            self.depth_texture.destroy(device);
            self.color_texture.destroy(device);
            for framebuffer in self.swapchain_framebuffers.iter() {
                device.destroy_framebuffer(*framebuffer, None);
            }
            self.pipeline.cleanup(device);
            self.pipeline_cubemap.cleanup(device);
            device.destroy_render_pass(self.render_pass, None);
            for image_view in self.swapchain_image_views.iter() {
                device.destroy_image_view(*image_view, None);
            }
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }

    fn update_uniform_buffers(&mut self, current_image: u32) {
        let aspect = self.get_extent().width as f32 / self.get_extent().height as f32;
        let ubo = UniformBufferObject {
            model: self.model_matrix * self.initial_model_matrix,
            view: self.view_matrix,
            proj: math::perspective(Deg(75.0), aspect, 0.1, 20.0),
            texture_weight: self.texture_weight,
        };
        let ubos = [ubo];

        let buffer_mem = self.uniform_buffer_memories[current_image as usize];
        let size = size_of::<UniformBufferObject>() as vk::DeviceSize;
        unsafe {
            let device = self.vk_context.device();
            let data_ptr = device
                .map_memory(buffer_mem, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<f32>() as _, size);
            align.copy_from_slice(&ubos);
            device.unmap_memory(buffer_mem);
        }
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.swapchain_properties.extent
    }

    pub fn reset_ubo(&mut self) {
        self.view_matrix = UniformBufferObject::view_matrix();
        self.model_matrix = Matrix4::unit();
        self.initial_model_matrix = UniformBufferObject::model_matrix(
            self.model_extent.0,
            self.model_extent.1,
        );
    }
}

impl Drop for VkApp {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        self.cleanup_swapchain();

        let device = self.vk_context.device();
        self.in_flight_frames.destroy(device);
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for &mem in &self.uniform_buffer_memories {
                device.free_memory(mem, None);
            }
            for &buffer in &self.uniform_buffers {
                device.destroy_buffer(buffer, None);
            }
            for mut texture in self.textures {
                texture.destroy(device);
            }
            device.free_command_buffers(self.command_pool, &self.command_buffers);
            device.destroy_command_pool(self.transient_command_pool, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}
