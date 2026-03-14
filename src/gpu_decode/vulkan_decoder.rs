//! Vulkan Video AV1 decoder using VK_KHR_video_decode_av1.
//!
//! Creates a Vulkan device with video decode queue, sets up an AV1 video
//! session, and decodes frames on the GPU. Decoded frames are NV12 textures
//! that can be imported into wgpu.
//!
//! This is a minimal implementation focused on AV1 decode for the LeRobot
//! annotation tool. It does NOT support encoding, transcoding, or other codecs.

use ash::vk;
use std::ffi::{CStr, CString};
use std::sync::Arc;

use super::av1_obu::{self, FrameHeader, SequenceHeader};

/// Errors from the Vulkan Video decoder.
#[derive(Debug)]
pub enum VulkanDecoderError {
    VulkanError(vk::Result),
    NoVideoDecodeQueue,
    NoAV1Support,
    SessionCreationFailed(String),
    DecodeError(String),
}

impl std::fmt::Display for VulkanDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VulkanError(e) => write!(f, "Vulkan error: {:?}", e),
            Self::NoVideoDecodeQueue => write!(f, "No video decode queue family found"),
            Self::NoAV1Support => write!(f, "GPU does not support AV1 decode"),
            Self::SessionCreationFailed(msg) => write!(f, "Session creation failed: {}", msg),
            Self::DecodeError(msg) => write!(f, "Decode error: {}", msg),
        }
    }
}

impl std::error::Error for VulkanDecoderError {}

impl From<vk::Result> for VulkanDecoderError {
    fn from(e: vk::Result) -> Self {
        Self::VulkanError(e)
    }
}

/// Check if a physical device supports AV1 video decoding.
pub fn check_av1_decode_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(u32, vk::QueueFamilyProperties), VulkanDecoderError> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    for (i, props) in queue_families.iter().enumerate() {
        if props
            .queue_flags
            .contains(vk::QueueFlags::VIDEO_DECODE_KHR)
        {
            log::info!(
                "Found video decode queue family {} with {} queues",
                i,
                props.queue_count
            );
            return Ok((i as u32, *props));
        }
    }

    Err(VulkanDecoderError::NoVideoDecodeQueue)
}

/// Query AV1 decode capabilities for the physical device.
pub fn query_av1_decode_caps(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<AV1DecodeCaps, VulkanDecoderError> {
    // Build AV1 decode profile
    let mut av1_profile_info = vk::VideoDecodeAV1ProfileInfoKHR::default()
        .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
        .film_grain_support(false);

    let mut decode_usage = vk::VideoDecodeUsageInfoKHR::default()
        .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT);

    let video_profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut av1_profile_info)
        .push_next(&mut decode_usage);

    let mut av1_caps = vk::VideoDecodeAV1CapabilitiesKHR::default();
    let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
    let mut video_caps = vk::VideoCapabilitiesKHR::default()
        .push_next(&mut decode_caps)
        .push_next(&mut av1_caps);

    let video_queue_fn = ash::khr::video_queue::Instance::new(entry, instance);

    unsafe {
        let result = (video_queue_fn.fp().get_physical_device_video_capabilities_khr)(
            physical_device,
            &video_profile,
            &mut video_caps,
        );
        if result != vk::Result::SUCCESS {
            return Err(VulkanDecoderError::NoAV1Support);
        }
    }

    log::info!(
        "AV1 decode caps: max_coded_extent={}x{}, max_dpb_slots={}, max_active_refs={}",
        video_caps.max_coded_extent.width,
        video_caps.max_coded_extent.height,
        video_caps.max_dpb_slots,
        video_caps.max_active_reference_pictures,
    );

    Ok(AV1DecodeCaps {
        max_coded_extent: video_caps.max_coded_extent,
        min_coded_extent: video_caps.min_coded_extent,
        max_dpb_slots: video_caps.max_dpb_slots,
        max_active_reference_pictures: video_caps.max_active_reference_pictures,
        max_level_idc: av1_caps.max_level,
    })
}

/// AV1 decode capabilities reported by the GPU.
#[derive(Debug)]
pub struct AV1DecodeCaps {
    pub max_coded_extent: vk::Extent2D,
    pub min_coded_extent: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub max_level_idc: vk::native::StdVideoAV1Level,
}

/// Full AV1 Vulkan Video decoder.
///
/// Owns the Vulkan device, video session, DPB images, and command buffers.
/// Call `decode_frame()` with AV1 packet data to decode a frame on the GPU.
pub struct AV1Decoder {
    // Vulkan handles
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,

    // Queue
    video_decode_queue: vk::Queue,
    video_queue_family: u32,
    transfer_queue: vk::Queue,
    transfer_queue_family: u32,

    // Extension function pointers
    video_queue_fn: ash::khr::video_queue::Device,

    // Video session
    video_session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // DPB (Decoded Picture Buffer)
    dpb_images: Vec<vk::Image>,
    dpb_views: Vec<vk::ImageView>,
    dpb_memory: Vec<vk::DeviceMemory>,
    dpb_slot_active: [bool; 8],

    // Output image (separate from DPB for display)
    output_image: vk::Image,
    output_view: vk::ImageView,
    output_memory: vk::DeviceMemory,

    // Bitstream buffer
    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    bitstream_capacity: usize,

    // Command pool
    command_pool: vk::CommandPool,

    // Stream state
    sequence_header: SequenceHeader,
    width: u32,
    height: u32,
    frame_count: u64,
}

impl AV1Decoder {
    /// Create a new AV1 decoder for the given sequence header parameters.
    pub fn new(seq: &SequenceHeader) -> Result<Self, VulkanDecoderError> {
        let width = seq.max_frame_width_minus_1 as u32 + 1;
        let height = seq.max_frame_height_minus_1 as u32 + 1;

        log::info!("Creating AV1 Vulkan decoder for {}x{}", width, height);

        // 1. Create Vulkan instance
        let entry = unsafe {
            ash::Entry::load().map_err(|e| {
                VulkanDecoderError::SessionCreationFailed(format!("Failed to load Vulkan: {:?}", e))
            })?
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lerobot-explorer")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let instance_create = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_create, None)? };

        // 2. Find physical device with AV1 decode
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices
            .into_iter()
            .find(|pdev| check_av1_decode_support(&instance, *pdev).is_ok())
            .ok_or(VulkanDecoderError::NoAV1Support)?;

        let (video_queue_family, _) = check_av1_decode_support(&instance, physical_device)?;

        // Find a transfer queue (may be same family or different)
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let transfer_queue_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(i, _)| i as u32)
            .unwrap_or(video_queue_family);

        // 3. Create device with video decode queue
        let queue_priorities = [1.0f32];
        let mut queue_create_infos = vec![vk::DeviceQueueCreateInfo::default()
            .queue_family_index(video_queue_family)
            .queue_priorities(&queue_priorities)];

        if transfer_queue_family != video_queue_family {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_queue_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let device_extensions = [
            vk::KHR_VIDEO_QUEUE_NAME.as_ptr(),
            vk::KHR_VIDEO_DECODE_QUEUE_NAME.as_ptr(),
            vk::KHR_VIDEO_DECODE_AV1_NAME.as_ptr(),
            vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
        ];

        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2_features);

        let device = unsafe { instance.create_device(physical_device, &device_create, None)? };
        let video_decode_queue = unsafe { device.get_device_queue(video_queue_family, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family, 0) };

        let video_queue_fn = ash::khr::video_queue::Device::new(&instance, &device);

        // 4. Create AV1 video session
        let mut av1_profile_info = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support(false);

        let mut decode_usage = vk::VideoDecodeUsageInfoKHR::default()
            .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT);

        let video_profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut av1_profile_info)
            .push_next(&mut decode_usage);

        let profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(std::slice::from_ref(&video_profile));

        let picture_format = vk::Format::G8_B8R8_2PLANE_420_UNORM; // NV12

        let std_header_version = vk::ExtensionProperties {
            extension_name: {
                let mut name = [0i8; 256];
                let src = b"VK_STD_vulkan_video_codec_av1_decode\0";
                for (i, &b) in src.iter().enumerate() {
                    name[i] = b as i8;
                }
                name
            },
            spec_version: vk::make_api_version(0, 1, 0, 0),
        };

        let session_create = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(video_queue_family)
            .video_profile(&video_profile)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D { width, height })
            .reference_picture_format(picture_format)
            .max_dpb_slots(8)
            .max_active_reference_pictures(7)
            .std_header_version(&std_header_version);

        let mut video_session_handle = vk::VideoSessionKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_khr)(
                device.handle(),
                &session_create,
                std::ptr::null(),
                &mut video_session_handle,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(VulkanDecoderError::SessionCreationFailed(format!(
                "vkCreateVideoSessionKHR failed: {:?}",
                result
            )));
        }
        log::info!("Created AV1 video session");

        // 5. Bind memory to video session
        let mut mem_req_count = 0u32;
        unsafe {
            (video_queue_fn
                .fp()
                .get_video_session_memory_requirements_khr)(
                device.handle(),
                video_session_handle,
                &mut mem_req_count,
                std::ptr::null_mut(),
            );
        }
        let mut mem_reqs =
            vec![vk::VideoSessionMemoryRequirementsKHR::default(); mem_req_count as usize];
        unsafe {
            (video_queue_fn
                .fp()
                .get_video_session_memory_requirements_khr)(
                device.handle(),
                video_session_handle,
                &mut mem_req_count,
                mem_reqs.as_mut_ptr(),
            );
        }

        let mem_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let mut session_memory = Vec::new();
        let mut bind_infos = Vec::new();
        for req in &mem_reqs {
            let mem_type_index = find_memory_type(
                &mem_props,
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                VulkanDecoderError::SessionCreationFailed("No suitable memory type".into())
            })?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.memory_requirements.size)
                .memory_type_index(mem_type_index);
            let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
            session_memory.push(memory);

            bind_infos.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(req.memory_bind_index)
                    .memory(memory)
                    .memory_offset(0)
                    .memory_size(req.memory_requirements.size),
            );
        }

        unsafe {
            let result = (video_queue_fn.fp().bind_video_session_memory_khr)(
                device.handle(),
                video_session_handle,
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            );
            if result != vk::Result::SUCCESS {
                return Err(VulkanDecoderError::SessionCreationFailed(format!(
                    "vkBindVideoSessionMemoryKHR failed: {:?}",
                    result
                )));
            }
        }
        log::info!(
            "Bound {} memory allocations to video session",
            session_memory.len()
        );

        // 6. Create session parameters from sequence header
        let std_seq_owned = seq_header_to_std_owned(seq);
        let mut av1_params = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default()
            .std_sequence_header(&std_seq_owned.header);
        let params_create = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(video_session_handle)
            .push_next(&mut av1_params);

        let mut session_params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_parameters_khr)(
                device.handle(),
                &params_create,
                std::ptr::null(),
                &mut session_params,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(VulkanDecoderError::SessionCreationFailed(format!(
                "vkCreateVideoSessionParametersKHR failed: {:?}",
                result
            )));
        }
        log::info!("Created AV1 session parameters");

        // 7. Allocate DPB images (8 reference frame slots)
        // Re-create profile_list for image/buffer creation (previous one was consumed by session)
        let mut av1_pi2 = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support(false);
        let mut du2 = vk::VideoDecodeUsageInfoKHR::default()
            .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT);
        let vp2 = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut av1_pi2)
            .push_next(&mut du2);
        let mut pl2 = vk::VideoProfileListInfoKHR::default()
            .profiles(std::slice::from_ref(&vp2));

        let (dpb_images, dpb_views, dpb_memory) =
            allocate_dpb_images(&device, &mem_props, &mut pl2, width, height, 8)?;
        log::info!("Allocated {} DPB images ({}x{})", dpb_images.len(), width, height);

        // 8. Allocate output image
        let (out_img, out_view, out_mem) =
            allocate_single_image(&device, &mem_props, &mut pl2, width, height)?;
        log::info!("Allocated output image");

        // 9. Allocate bitstream buffer
        let bitstream_capacity = 4 * 1024 * 1024; // 4MB should cover any frame
        let (bitstream_buffer, bitstream_memory) =
            allocate_bitstream_buffer(&device, &mem_props, &mut pl2, bitstream_capacity)?;
        log::info!("Allocated {}MB bitstream buffer", bitstream_capacity / (1024 * 1024));

        // 10. Create command pool for video decode queue
        let pool_create = vk::CommandPoolCreateInfo::default()
            .queue_family_index(video_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_create, None)? };

        Ok(Self {
            _entry: entry,
            instance,
            device,
            physical_device,
            video_decode_queue,
            video_queue_family,
            transfer_queue,
            transfer_queue_family,
            video_queue_fn,
            video_session: video_session_handle,
            session_params,
            session_memory,
            dpb_images,
            dpb_views,
            dpb_memory,
            dpb_slot_active: [false; 8],
            output_image: out_img,
            output_view: out_view,
            output_memory: out_mem,
            bitstream_buffer,
            bitstream_memory,
            bitstream_capacity,
            command_pool,
            sequence_header: seq.clone(),
            width,
            height,
            frame_count: 0,
        })
    }

    /// Get the decoded frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    // TODO: decode_frame(), read_back_output(), export_to_wgpu()
}

impl Drop for AV1Decoder {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            self.device
                .destroy_command_pool(self.command_pool, None);
            self.device.destroy_buffer(self.bitstream_buffer, None);
            self.device.free_memory(self.bitstream_memory, None);
            self.device
                .destroy_image_view(self.output_view, None);
            self.device.destroy_image(self.output_image, None);
            self.device.free_memory(self.output_memory, None);

            for view in &self.dpb_views {
                self.device.destroy_image_view(*view, None);
            }
            for image in &self.dpb_images {
                self.device.destroy_image(*image, None);
            }
            for mem in &self.dpb_memory {
                self.device.free_memory(*mem, None);
            }

            (self.video_queue_fn.fp().destroy_video_session_parameters_khr)(
                self.device.handle(),
                self.session_params,
                std::ptr::null(),
            );
            (self.video_queue_fn.fp().destroy_video_session_khr)(
                self.device.handle(),
                self.video_session,
                std::ptr::null(),
            );
            for mem in &self.session_memory {
                self.device.free_memory(*mem, None);
            }

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// -- Helper functions --

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // First try exact match
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(required_flags)
        {
            return Some(i);
        }
    }
    // Fallback: any memory type that matches the type bits
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0 {
            return Some(i);
        }
    }
    None
}

fn allocate_dpb_images(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    profile_list: &mut vk::VideoProfileListInfoKHR,
    width: u32,
    height: u32,
    count: usize,
) -> Result<(Vec<vk::Image>, Vec<vk::ImageView>, Vec<vk::DeviceMemory>), VulkanDecoderError> {
    let mut images = Vec::with_capacity(count);
    let mut views = Vec::with_capacity(count);
    let mut memories = Vec::with_capacity(count);

    for _ in 0..count {
        let image_create = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                    | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(profile_list);

        let image = unsafe { device.create_image(&image_create, None)? };
        let mem_req = unsafe { device.get_image_memory_requirements(image) };

        let mem_type = find_memory_type(mem_props, mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .ok_or_else(|| VulkanDecoderError::SessionCreationFailed("No device-local memory for DPB".into()))?;

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { device.allocate_memory(&alloc, None)? };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        let view_create = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { device.create_image_view(&view_create, None)? };

        images.push(image);
        views.push(view);
        memories.push(memory);
    }

    Ok((images, views, memories))
}

fn allocate_single_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    profile_list: &mut vk::VideoProfileListInfoKHR,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory), VulkanDecoderError> {
    let (mut imgs, mut views, mut mems) =
        allocate_dpb_images(device, mem_props, profile_list, width, height, 1)?;
    Ok((imgs.remove(0), views.remove(0), mems.remove(0)))
}

fn allocate_bitstream_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    profile_list: &mut vk::VideoProfileListInfoKHR,
    capacity: usize,
) -> Result<(vk::Buffer, vk::DeviceMemory), VulkanDecoderError> {
    let buffer_create = vk::BufferCreateInfo::default()
        .size(capacity as u64)
        .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(profile_list);

    let buffer = unsafe { device.create_buffer(&buffer_create, None)? };
    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mem_type = find_memory_type(
        mem_props,
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        VulkanDecoderError::SessionCreationFailed("No host-visible memory for bitstream".into())
    })?;

    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    Ok((buffer, memory))
}

/// Owned AV1 sequence header data for Vulkan, keeping all pointed-to data alive.
pub struct StdSequenceHeaderOwned {
    pub color_config: Box<vk::native::StdVideoAV1ColorConfig>,
    pub timing_info: Option<Box<vk::native::StdVideoAV1TimingInfo>>,
    pub header: vk::native::StdVideoAV1SequenceHeader,
}

/// Convert our parsed SequenceHeader to owned Vulkan StdVideoAV1SequenceHeader.
/// The returned struct keeps all pointer targets alive.
pub fn seq_header_to_std_owned(seq: &SequenceHeader) -> StdSequenceHeaderOwned {
    use vk::native::*;

    let flags = StdVideoAV1SequenceHeaderFlags {
        _bitfield_1: StdVideoAV1SequenceHeaderFlags::new_bitfield_1(
            seq.still_picture as u32,
            seq.reduced_still_picture_header as u32,
            seq.use_128x128_superblock as u32,
            seq.enable_filter_intra as u32,
            seq.enable_intra_edge_filter as u32,
            seq.enable_interintra_compound as u32,
            seq.enable_masked_compound as u32,
            seq.enable_warped_motion as u32,
            seq.enable_dual_filter as u32,
            seq.enable_order_hint as u32,
            seq.enable_jnt_comp as u32,
            seq.enable_ref_frame_mvs as u32,
            seq.frame_id_numbers_present as u32,
            seq.enable_superres as u32,
            seq.enable_cdef as u32,
            seq.enable_restoration as u32,
            seq.film_grain_params_present as u32,
            seq.timing_info_present as u32,
            0, // initial_display_delay_present
            0, // reserved
        ),
        _bitfield_align_1: [],
    };

    let color_config = Box::new(StdVideoAV1ColorConfig {
        flags: StdVideoAV1ColorConfigFlags {
            _bitfield_1: StdVideoAV1ColorConfigFlags::new_bitfield_1(
                seq.color_config.mono_chrome as u32,
                seq.color_config.color_range as u32,
                seq.color_config.separate_uv_delta_q as u32,
                seq.color_config.color_description_present as u32,
                0, // reserved
            ),
            _bitfield_align_1: [],
        },
        BitDepth: seq.color_config.bit_depth,
        subsampling_x: seq.color_config.subsampling_x as u8,
        subsampling_y: seq.color_config.subsampling_y as u8,
        reserved1: 0,
        color_primaries: seq.color_config.color_primaries as StdVideoAV1ColorPrimaries,
        transfer_characteristics: seq.color_config.transfer_characteristics
            as StdVideoAV1TransferCharacteristics,
        matrix_coefficients: seq.color_config.matrix_coefficients
            as StdVideoAV1MatrixCoefficients,
        chroma_sample_position: seq.color_config.chroma_sample_position
            as StdVideoAV1ChromaSamplePosition,
    });

    let timing_info = if seq.timing_info_present {
        Some(Box::new(StdVideoAV1TimingInfo {
            flags: StdVideoAV1TimingInfoFlags {
                _bitfield_1: StdVideoAV1TimingInfoFlags::new_bitfield_1(
                    seq.equal_picture_interval as u32,
                    0,
                ),
                _bitfield_align_1: [],
            },
            num_units_in_display_tick: seq.num_units_in_display_tick,
            time_scale: seq.time_scale,
            num_ticks_per_picture_minus_1: 0,
        }))
    } else {
        None
    };

    let header = StdVideoAV1SequenceHeader {
        flags,
        seq_profile: seq.seq_profile as StdVideoAV1Profile,
        frame_width_bits_minus_1: seq.frame_width_bits_minus_1,
        frame_height_bits_minus_1: seq.frame_height_bits_minus_1,
        max_frame_width_minus_1: seq.max_frame_width_minus_1,
        max_frame_height_minus_1: seq.max_frame_height_minus_1,
        delta_frame_id_length_minus_2: seq.delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1: seq.additional_frame_id_length_minus_1,
        order_hint_bits_minus_1: seq.order_hint_bits_minus_1,
        seq_force_integer_mv: seq.seq_force_integer_mv,
        seq_force_screen_content_tools: seq.seq_force_screen_content_tools,
        reserved1: [0; 5],
        pColorConfig: &*color_config as *const _,
        pTimingInfo: timing_info
            .as_ref()
            .map(|t| &**t as *const _)
            .unwrap_or(std::ptr::null()),
    };

    StdSequenceHeaderOwned {
        color_config,
        timing_info,
        header,
    }
}
