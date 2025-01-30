use super::debug::setup_debug_messenger;
use super::swapchain::SwapchainSupportDetails;

use anyhow::anyhow;
use ash::{
    ext::debug_utils,
    khr::{surface, swapchain as khr_swapchain},
    vk, Device, Entry, Instance
};
use std::ffi::CStr;

#[derive(Debug, Clone, Copy)]
pub struct QueueFamiliesIndices {
    pub graphics_index: u32,
    pub present_index: u32,
}

pub struct VkContext {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface: surface::Instance,
    surface_khr: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue_families_indices: QueueFamiliesIndices,
}

impl VkContext {
    pub fn new(
        entry: Entry,
        instance: Instance,
        surface: surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Result<Self, anyhow::Error> {
        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        let (physical_device, queue_families_indices) =
            Self::pick_physical_device(&instance, &surface, surface_khr)
            .ok_or(anyhow!("No suitable physical device found"))?;

        let device = Self::create_logical_device(
            &instance,
            physical_device,
            queue_families_indices,
        )?;

        Ok(VkContext {
            _entry: entry,
            instance,
            debug_report_callback,
            surface,
            surface_khr,
            physical_device,
            device,
            queue_families_indices,
        })
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn surface(&self) -> &surface::Instance {
        &self.surface
    }

    pub fn surface_khr(&self) -> vk::SurfaceKHR {
        self.surface_khr
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn graphics_queue_index(&self) -> u32 {
        self.queue_families_indices.graphics_index
    }

    pub fn present_queue_index(&self) -> u32 {
        self.queue_families_indices.present_index
    }

    pub fn physical_device_properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.instance.get_physical_device_properties(self.physical_device)
        }
    }

    pub fn get_mem_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        }
    }

    /// Find a memory type in `mem_properties` that is suitable
    /// for `requirements` and supports `required_properties`.
    ///
    /// # Returns
    ///
    /// The index of the memory type from `mem_properties`.
    pub fn find_memory_type(
        &self,
        requirements: vk::MemoryRequirements,
        required_properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let mem_properties = self.get_mem_properties();
        for i in 0..mem_properties.memory_type_count {
            if requirements.memory_type_bits & (1 << i) != 0
                && mem_properties.memory_types[i as usize]
                    .property_flags
                    .contains(required_properties)
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type.")
    }

    pub fn create_command_pool(&self, create_flags: vk::CommandPoolCreateFlags) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.queue_families_indices.graphics_index)
            .flags(create_flags);

        unsafe {
            self.device.create_command_pool(&command_pool_info, None).unwrap()
        }
    }

    /// Find the first compatible format from `candidates`.
    pub fn find_supported_format(
        &self,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Option<vk::Format> {
        candidates.iter().cloned().find(|candidate| {
            let props = unsafe {
                self.instance.get_physical_device_format_properties(self.physical_device, *candidate)
            };
            (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features))
                || (tiling == vk::ImageTiling::OPTIMAL
                    && props.optimal_tiling_features.contains(features))
        })
    }

    /// Return the maximum sample count supported.
    pub fn get_max_usable_sample_count(&self) -> vk::SampleCountFlags {
        let props = self.physical_device_properties();
        let color_sample_counts = props.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = props.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    /// Pick the first suitable physical device.
    ///
    /// # Requirements
    /// - At least one queue family with one queue supportting graphics.
    /// - At least one queue family with one queue supporting presentation to `surface_khr`.
    /// - Swapchain extension support.
    ///
    /// # Returns
    ///
    /// None if no suitable device is found.
    fn pick_physical_device(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Option<(vk::PhysicalDevice, QueueFamiliesIndices)> {
        let devices = unsafe { instance.enumerate_physical_devices().ok()? };
        let (device, queue_families_indices) = devices
            .into_iter()
            .filter_map(|device| {
                if !Self::check_device_extension_support(instance, device) {
                    return None;
                }

                let details = SwapchainSupportDetails::new(device, surface, surface_khr);
                if details.formats.is_empty() || details.present_modes.is_empty() {
                    return None;
                }

                let features = unsafe { instance.get_physical_device_features(device) };
                if features.sampler_anisotropy != vk::TRUE
                    || features.geometry_shader != vk::TRUE
                {
                    return None;
                }

                let queue_families_indices =
                    Self::find_queue_families(instance, surface, surface_khr, device)?;
                Some((device, queue_families_indices))
            })
            .next()?;

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        Some((device, queue_families_indices))
    }

    /// Create the logical device to interact with the physical `device`.
    fn create_logical_device(
        instance: &Instance,
        device: vk::PhysicalDevice,
        queue_families_indices: QueueFamiliesIndices,
    ) -> Result<Device, anyhow::Error> {
        let graphics_family_index = queue_families_indices.graphics_index;
        let present_family_index = queue_families_indices.present_index;
        let queue_priorities = [1.0f32];

        let queue_create_infos = {
            // Vulkan specs does not allow passing an array containing duplicated family indices.
            // And since the family for graphics and presentation could be the same we need to
            // deduplicate it.
            let mut indices = vec![graphics_family_index, present_family_index];
            indices.dedup();

            // Now we build an array of `DeviceQueueCreateInfo`.
            // One for each different family index.
            indices.iter()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(*index)
                        .queue_priorities(&queue_priorities)
                })
                .collect::<Vec<_>>()
        };

        let device_extensions = Self::get_required_device_extensions();
        let device_extensions_ptrs = device_extensions.iter()
            .map(|ext| ext.as_ptr())
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default()
            .geometry_shader(true)
            .sampler_anisotropy(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions_ptrs)
            .enabled_features(&device_features);

        // Build device
        let device = unsafe {
            instance.create_device(device, &device_create_info, None)?
        };
        Ok(device)
    }

    fn check_device_extension_support(instance: &Instance, device: vk::PhysicalDevice) -> bool {
        let extension_props = unsafe {
            instance.enumerate_device_extension_properties(device).unwrap()
        };

        Self::get_required_device_extensions().into_iter().all(|required_ext| {
            extension_props.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                required_ext == name
            })
        })
    }

    fn get_required_device_extensions() -> [&'static CStr; 1] {
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        [khr_swapchain::NAME]
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn get_required_device_extensions() -> [&'static CStr; 2] {
        [khr_swapchain::NAME, ash::khr::portability_subset::NAME]
    }

    /// Find a queue family with at least one graphics queue and one with
    /// at least one presentation queue from `device`.
    fn find_queue_families(
        instance: &Instance,
        surface: &surface::Instance,
        surface_khr: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> Option<QueueFamiliesIndices> {
        let mut graphics = None;
        let mut present = None;

        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        for (index, family) in props.iter().enumerate() {
            if family.queue_count == 0 {
                // this should not be possible according to the vulkan spec:
                // "Each queue family must support at least one queue."
                continue;
            }

            let index = index as u32;
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }
            let present_support = unsafe {
                surface.get_physical_device_surface_support(device, index, surface_khr)
            };
            if present_support.unwrap_or(false) && present.is_none() {
                present = Some(index);
            }

            if let (Some(graphics), Some(present)) = (graphics, present) {
                return Some(QueueFamiliesIndices {
                    graphics_index: graphics,
                    present_index: present,
                });
            }
        }

        None
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((utils, messenger)) = self.debug_report_callback.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
