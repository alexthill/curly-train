use super::cmd;
use super::context::VkContext;

use ash::{vk, Device};

/// Create a buffer and allocate its memory.
///
/// # Returns
///
/// The buffer, its memory and the actual size in bytes of the
/// allocated memory since in may differ from the requested size.
pub fn create_buffer(
    vk_context: &VkContext,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mem_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory, vk::DeviceSize) {
    let device = vk_context.device();
    let buffer = {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        unsafe { device.create_buffer(&buffer_info, None).unwrap() }
    };

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory = {
        let mem_type_index = vk_context.find_memory_type(mem_requirements, mem_properties);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        unsafe { device.allocate_memory(&alloc_info, None).unwrap() }
    };

    unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() };

    (buffer, memory, mem_requirements.size)
}

/// Copy the `size` first bytes of `src` into `dst`.
///
/// It's done using a command buffer allocated from `command_pool`.
/// The command buffer is submitted to `transfer_queue`.
pub fn copy_buffer(
    device: &Device,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: vk::DeviceSize,
) {
    cmd::execute_one_time_commands(device, command_pool, transfer_queue, |buffer| {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size,
        };
        let regions = [region];

        unsafe { device.cmd_copy_buffer(buffer, src, dst, &regions) };
    });
}
