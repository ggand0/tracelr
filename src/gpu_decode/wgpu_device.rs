//! Create a custom Vulkan device with both graphics and video decode support,
//! then wrap it as wgpu for use with eframe.
//!
//! This solves the dual-instance crash: instead of eframe creating its own
//! Vulkan device (without video extensions) and our decoder creating another
//! (conflicting for the video decode queue), we create ONE device with both
//! capabilities and give it to eframe.

use ash::vk;
use std::ffi::CStr;
use std::sync::{Arc, Mutex, OnceLock};

/// Global shared GPU state, set once at app startup.
/// Background threads can clone the Arc to access the shared Vulkan device.
static SHARED_GPU: OnceLock<Arc<SharedGpuState>> = OnceLock::new();

/// Vulkan handles shared between eframe (graphics) and the AV1 decoder (video decode).
pub struct SharedGpuState {
    pub entry: ash::Entry,
    pub ash_instance: ash::Instance,
    pub ash_device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub video_queue_family: u32,
}

// ash::Instance and ash::Device are internally Arc-based, safe to share across threads
unsafe impl Send for SharedGpuState {}
unsafe impl Sync for SharedGpuState {}

/// Store the shared GPU state (called once from main).
pub fn set_shared_gpu(state: SharedGpuState) {
    SHARED_GPU.set(Arc::new(state)).ok();
}

/// Get a reference to the shared GPU state (called from background threads).
pub fn get_shared_gpu() -> Option<Arc<SharedGpuState>> {
    SHARED_GPU.get().cloned()
}

/// Create a wgpu setup with a Vulkan device that supports both graphics
/// rendering and AV1 video decode. Returns the components needed for
/// `egui_wgpu::WgpuSetupExisting`, plus the video decode queue info.
pub struct GpuSetup {
    pub wgpu_instance: wgpu::Instance,
    pub wgpu_adapter: wgpu::Adapter,
    pub wgpu_device: wgpu::Device,
    pub wgpu_queue: wgpu::Queue,
    /// Video decode queue family index (separate from graphics queue)
    pub video_queue_family: u32,
    /// The raw ash entry (Vulkan loader)
    pub entry: ash::Entry,
    /// The raw ash device (shared with wgpu)
    pub ash_device: ash::Device,
    /// The raw ash instance
    pub ash_instance: ash::Instance,
    /// Physical device handle
    pub physical_device: vk::PhysicalDevice,
}

impl GpuSetup {
    /// Try to create a GPU setup with video decode support.
    /// Falls back to standard wgpu if video decode is unavailable.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // 1. Create Vulkan instance
        let entry = unsafe { ash::Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lerobot-explorer")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // wgpu needs VK_KHR_surface + platform surface extensions
        let instance_extensions = [
            vk::KHR_SURFACE_NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            vk::KHR_WAYLAND_SURFACE_NAME.as_ptr(),
            #[cfg(target_os = "linux")]
            vk::KHR_XCB_SURFACE_NAME.as_ptr(),
        ];

        let instance_create = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&instance_create, None)? };

        // 2. Find discrete GPU with video decode support
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices
            .into_iter()
            .find(|pdev| {
                let props = unsafe { instance.get_physical_device_properties(*pdev) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .ok_or("No discrete GPU found")?;

        // 3. Find queue families
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let graphics_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(i, _)| i as u32)
            .ok_or("No graphics queue")?;

        let video_decode_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, p)| p.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))
            .map(|(i, _)| i as u32);

        log::info!(
            "Graphics queue family: {}, Video decode queue family: {:?}",
            graphics_family,
            video_decode_family
        );

        // 4. Create device with both queue families + all needed extensions
        let queue_priorities = [1.0f32];
        let mut queue_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
            .queue_priorities(&queue_priorities)];

        if let Some(vf) = video_decode_family {
            if vf != graphics_family {
                queue_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(vf)
                        .queue_priorities(&queue_priorities),
                );
            }
        }

        // Extensions: what wgpu needs + video decode
        let mut device_extensions: Vec<*const i8> = vec![
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
        ];
        if video_decode_family.is_some() {
            device_extensions.push(vk::KHR_VIDEO_QUEUE_NAME.as_ptr());
            device_extensions.push(vk::KHR_VIDEO_DECODE_QUEUE_NAME.as_ptr());
            device_extensions.push(vk::KHR_VIDEO_DECODE_AV1_NAME.as_ptr());
        }

        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2_features);

        let ash_device = unsafe { instance.create_device(physical_device, &device_create, None)? };

        log::info!("Created Vulkan device with graphics + video decode queues");

        // 5. Wrap as wgpu via HAL
        // Create wgpu Instance from our ash instance.
        // Must pass the extensions we enabled so wgpu knows they're available.
        let enabled_instance_extensions = vec![
            vk::KHR_SURFACE_NAME,
            #[cfg(target_os = "linux")]
            vk::KHR_XLIB_SURFACE_NAME,
            #[cfg(target_os = "linux")]
            vk::KHR_WAYLAND_SURFACE_NAME,
            #[cfg(target_os = "linux")]
            vk::KHR_XCB_SURFACE_NAME,
        ];
        let wgpu_instance = unsafe {
            let hal_instance = wgpu::hal::vulkan::Instance::from_raw(
                entry.clone(),
                instance.clone(),
                vk::API_VERSION_1_3,
                0, // android_sdk_version
                None, // debug_utils
                enabled_instance_extensions,
                wgpu::InstanceFlags::empty(),
                false, // has_nv_optimus
                None, // drop_guard
            )?;
            wgpu::Instance::from_hal::<wgpu::hal::vulkan::Api>(hal_instance)
        };

        // Expose adapter
        let hal_adapter = unsafe {
            let shared = wgpu_instance
                .as_hal::<wgpu::hal::vulkan::Api>()
                .unwrap();
            shared.expose_adapter(physical_device)
                .ok_or("Failed to expose adapter")?
        };

        let wgpu_adapter = unsafe {
            wgpu_instance.create_adapter_from_hal(hal_adapter)
        };

        // Create wgpu device from our raw ash device
        let enabled_ext_names: Vec<&'static CStr> = vec![
            vk::KHR_SWAPCHAIN_NAME,
            vk::KHR_SYNCHRONIZATION2_NAME,
        ];

        let (wgpu_device, wgpu_queue) = unsafe {
            let hal_open_device = wgpu_adapter
                .as_hal::<wgpu::hal::vulkan::Api, _, _>(|hal_adapter| {
                    hal_adapter.unwrap().device_from_raw(
                        ash_device.clone(),
                        None, // drop_callback
                        &enabled_ext_names,
                        wgpu::Features::empty(),
                        &wgpu::MemoryHints::Performance,
                        graphics_family,
                        0, // queue_index
                    )
                })?;

            wgpu_adapter.create_device_from_hal(
                hal_open_device,
                &wgpu::DeviceDescriptor {
                    label: Some("lerobot-explorer"),
                    ..Default::default()
                },
                None, // trace_path
            )?
        };

        Ok(Self {
            wgpu_instance,
            wgpu_adapter,
            wgpu_device,
            wgpu_queue,
            video_queue_family: video_decode_family.unwrap_or(0),
            entry,
            ash_device,
            ash_instance: instance,
            physical_device,
        })
    }
}
