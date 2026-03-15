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

    /// AV1 Virtual Buffer Index — maps VBI slot [0..7] to DPB slot index.
    /// Updated by refresh_frame_flags after each decode.
    /// -1 means the VBI slot is empty.
    vbi_to_dpb: [i32; 8],
    /// Order hint stored for each VBI slot (for reference frame ordering).
    vbi_order_hint: [u8; 8],
    /// Frame type stored for each VBI slot (for reference info).
    vbi_frame_type: [u8; 8],

    // Output image (separate from DPB for display)
    output_image: vk::Image,
    output_view: vk::ImageView,
    output_memory: vk::DeviceMemory,

    // Bitstream buffer
    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    bitstream_capacity: usize,

    // Persistent staging buffer for readback (avoids per-frame alloc)
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_capacity: usize,

    // Command pool
    command_pool: vk::CommandPool,

    // Stream state
    sequence_header: SequenceHeader,
    width: u32,
    height: u32,
    frame_count: u64,

    // Whether we own the Vulkan instance/device (false when using shared from GpuSetup)
    owns_device: bool,
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

        let instance_create = vk::InstanceCreateInfo::default()
            .application_info(&app_info);
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

        // 10. Allocate persistent staging buffer for readback
        let staging_capacity = (width * height * 3 / 2) as usize; // NV12 size
        let staging_create = vk::BufferCreateInfo::default()
            .size(staging_capacity as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe { device.create_buffer(&staging_create, None)? };
        let staging_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_mem_type = find_memory_type(
            &mem_props,
            staging_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| {
            VulkanDecoderError::SessionCreationFailed("No host memory for staging".into())
        })?;
        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_req.size)
            .memory_type_index(staging_mem_type);
        let staging_memory = unsafe { device.allocate_memory(&staging_alloc, None)? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0)? };
        log::info!("Allocated persistent staging buffer for readback");

        // 11. Create command pool for video decode queue
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
            vbi_to_dpb: [-1; 8],
            vbi_order_hint: [0; 8],
            vbi_frame_type: [0; 8],
            output_image: out_img,
            output_view: out_view,
            output_memory: out_mem,
            bitstream_buffer,
            bitstream_memory,
            bitstream_capacity,
            staging_buffer,
            staging_memory,
            staging_capacity,
            command_pool,
            sequence_header: seq.clone(),
            width,
            height,
            frame_count: 0,
            owns_device: true,
        })
    }

    /// Create an AV1 decoder using an existing shared Vulkan device (from GpuSetup).
    /// The device must have been created with video decode extensions and queue.
    pub fn from_shared(
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        video_queue_family: u32,
        seq: &SequenceHeader,
    ) -> Result<Self, VulkanDecoderError> {
        let width = seq.max_frame_width_minus_1 as u32 + 1;
        let height = seq.max_frame_height_minus_1 as u32 + 1;

        log::info!("Creating shared AV1 Vulkan decoder for {}x{}", width, height);

        let video_decode_queue = unsafe { device.get_device_queue(video_queue_family, 0) };

        // Find a transfer queue
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let transfer_queue_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(i, _)| i as u32)
            .unwrap_or(video_queue_family);
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family, 0) };

        let video_queue_fn = ash::khr::video_queue::Device::new(&instance, &device);

        // Build AV1 profile info (reused for session + images + buffers)
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

        let picture_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

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
        log::info!("Created AV1 video session (shared device)");

        // Bind memory
        let mut mem_req_count = 0u32;
        unsafe {
            let _ = (video_queue_fn.fp().get_video_session_memory_requirements_khr)(
                device.handle(), video_session_handle, &mut mem_req_count, std::ptr::null_mut(),
            );
        }
        let mut mem_reqs = vec![vk::VideoSessionMemoryRequirementsKHR::default(); mem_req_count as usize];
        unsafe {
            let _ = (video_queue_fn.fp().get_video_session_memory_requirements_khr)(
                device.handle(), video_session_handle, &mut mem_req_count, mem_reqs.as_mut_ptr(),
            );
        }
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let mut session_memory = Vec::new();
        let mut bind_infos = Vec::new();
        for req in &mem_reqs {
            let mem_type = find_memory_type(&mem_props, req.memory_requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or_else(|| VulkanDecoderError::SessionCreationFailed("No memory type".into()))?;
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.memory_requirements.size)
                .memory_type_index(mem_type);
            let memory = unsafe { device.allocate_memory(&alloc, None)? };
            session_memory.push(memory);
            bind_infos.push(vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(req.memory_bind_index)
                .memory(memory)
                .memory_offset(0)
                .memory_size(req.memory_requirements.size));
        }
        unsafe {
            let result = (video_queue_fn.fp().bind_video_session_memory_khr)(
                device.handle(), video_session_handle, bind_infos.len() as u32, bind_infos.as_ptr(),
            );
            if result != vk::Result::SUCCESS {
                return Err(VulkanDecoderError::SessionCreationFailed(format!("bind memory: {:?}", result)));
            }
        }

        // Session parameters
        let std_seq_owned = seq_header_to_std_owned(seq);
        let mut av1_params = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default()
            .std_sequence_header(&std_seq_owned.header);
        let params_create = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(video_session_handle)
            .push_next(&mut av1_params);
        let mut session_params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_parameters_khr)(
                device.handle(), &params_create, std::ptr::null(), &mut session_params,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(VulkanDecoderError::SessionCreationFailed(format!("session params: {:?}", result)));
        }

        // DPB + output + bitstream + staging + command pool (same as new())
        let mut pl = vk::VideoProfileListInfoKHR::default()
            .profiles(std::slice::from_ref(&video_profile));
        let (dpb_images, dpb_views, dpb_memory) =
            allocate_dpb_images(&device, &mem_props, &mut pl, width, height, 8)?;
        let (out_img, out_view, out_mem) =
            allocate_single_image(&device, &mem_props, &mut pl, width, height)?;
        let bitstream_capacity = 4 * 1024 * 1024;
        let (bitstream_buffer, bitstream_memory) =
            allocate_bitstream_buffer(&device, &mem_props, &mut pl, bitstream_capacity)?;

        let staging_capacity = (width * height * 3 / 2) as usize;
        let staging_create = vk::BufferCreateInfo::default()
            .size(staging_capacity as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe { device.create_buffer(&staging_create, None)? };
        let staging_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let staging_mem_type = find_memory_type(&mem_props, staging_req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            .ok_or_else(|| VulkanDecoderError::SessionCreationFailed("No staging memory".into()))?;
        let staging_memory = unsafe { device.allocate_memory(
            &vk::MemoryAllocateInfo::default().allocation_size(staging_req.size).memory_type_index(staging_mem_type), None)? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0)? };

        let pool_create = vk::CommandPoolCreateInfo::default()
            .queue_family_index(video_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_create, None)? };

        // 12. Pre-transition all DPB images from UNDEFINED to VIDEO_DECODE_DPB_KHR.
        // The spec requires DPB images to be in this layout when referenced.
        {
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd_bufs = unsafe { device.allocate_command_buffers(&alloc)? };
            let cmd = cmd_bufs[0];
            unsafe {
                device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;

                let mut barriers = Vec::with_capacity(dpb_images.len());
                for img in &dpb_images {
                    barriers.push(
                        vk::ImageMemoryBarrier::default()
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .image(*img)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .src_access_mask(vk::AccessFlags::NONE)
                            .dst_access_mask(vk::AccessFlags::NONE),
                    );
                }
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );

                device.end_command_buffer(cmd)?;

                let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
                let cmd_submit = [cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmd_submit);
                device.queue_submit(
                    device.get_device_queue(video_queue_family, 0),
                    &[submit],
                    fence,
                )?;
                device.wait_for_fences(&[fence], true, u64::MAX)?;
                device.destroy_fence(fence, None);
                device.free_command_buffers(command_pool, &[cmd]);
            }
            log::info!("Pre-transitioned {} DPB images to VIDEO_DECODE_DPB_KHR", dpb_images.len());
        }

        log::info!("Shared AV1 decoder fully initialized");

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
            vbi_to_dpb: [-1; 8],
            vbi_order_hint: [0; 8],
            vbi_frame_type: [0; 8],
            output_image: out_img,
            output_view: out_view,
            output_memory: out_mem,
            bitstream_buffer,
            bitstream_memory,
            bitstream_capacity,
            staging_buffer,
            staging_memory,
            staging_capacity,
            command_pool,
            sequence_header: seq.clone(),
            width,
            height,
            frame_count: 0,
            owns_device: false,
        })
    }

    /// Create an AV1 decoder with a video-decode-only VkDevice.
    /// Uses a separate VkDevice from wgpu's graphics device, but only claims
    /// the video decode queue (family 3), avoiding conflicts.
    pub fn new_video_only(seq: &SequenceHeader) -> Result<Self, VulkanDecoderError> {
        let width = seq.max_frame_width_minus_1 as u32 + 1;
        let height = seq.max_frame_height_minus_1 as u32 + 1;

        log::info!("Creating video-only AV1 decoder for {}x{}", width, height);

        let entry = unsafe {
            ash::Entry::load().map_err(|e| {
                VulkanDecoderError::SessionCreationFailed(format!("Load: {:?}", e))
            })?
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"lerobot-video-decode")
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let instance_create = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_create, None)? };

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices
            .into_iter()
            .find(|pdev| check_av1_decode_support(&instance, *pdev).is_ok())
            .ok_or(VulkanDecoderError::NoAV1Support)?;

        let (video_queue_family, _) = check_av1_decode_support(&instance, physical_device)?;

        // Create device with ONLY video decode queue — no graphics queue
        let queue_priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(video_queue_family)
            .queue_priorities(&queue_priorities)];

        let device_extensions = [
            vk::KHR_VIDEO_QUEUE_NAME.as_ptr(),
            vk::KHR_VIDEO_DECODE_QUEUE_NAME.as_ptr(),
            vk::KHR_VIDEO_DECODE_AV1_NAME.as_ptr(),
            vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
        ];

        let mut sync2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let device_create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2);

        let device = unsafe { instance.create_device(physical_device, &device_create, None)? };

        // Now delegate to from_shared with owns_device=true
        let mut decoder = Self::from_shared(
            entry,
            instance.clone(),
            device.clone(),
            physical_device,
            video_queue_family,
            seq,
        )?;
        decoder.owns_device = true;
        Ok(decoder)
    }

    /// Get the decoded frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Decode a single AV1 frame from packet data.
    ///
    /// `packet_data` is the raw AV1 OBU data for one frame (from ffmpeg demuxer).
    /// The frame is decoded on the GPU into the output image.
    /// Returns the frame index and whether it should be displayed (show_frame).
    pub fn decode_frame(
        &mut self,
        packet_data: &[u8],
    ) -> Result<DecodeOutput, VulkanDecoderError> {
        // 1. Parse OBUs to find Frame/FrameHeader and tile data
        let obus = av1_obu::parse_obu_headers(packet_data)
            .map_err(|e| VulkanDecoderError::DecodeError(format!("OBU parse: {}", e)))?;

        let mut frame_header: Option<FrameHeader> = None;
        let mut frame_header_offset_in_packet = 0usize;
        let mut tile_data_offset_in_packet = 0usize;

        for obu in &obus {
            match obu.obu_type {
                av1_obu::ObuType::Frame => {
                    let payload =
                        &packet_data[obu.data_offset..obu.data_offset + obu.data_size];
                    let (fh, header_bytes) =
                        av1_obu::parse_frame_header(payload, &self.sequence_header)
                            .map_err(|e| {
                                VulkanDecoderError::DecodeError(format!(
                                    "Frame header parse: {}",
                                    e
                                ))
                            })?;
                    frame_header = Some(fh);
                    frame_header_offset_in_packet = obu.data_offset; // payload start
                    // Tile data starts after the frame header within the Frame OBU
                    tile_data_offset_in_packet = obu.data_offset + header_bytes;
                }
                av1_obu::ObuType::FrameHeader => {
                    let payload =
                        &packet_data[obu.data_offset..obu.data_offset + obu.data_size];
                    let (fh, header_bytes) =
                        av1_obu::parse_frame_header(payload, &self.sequence_header)
                            .map_err(|e| {
                                VulkanDecoderError::DecodeError(format!(
                                    "Frame header parse: {}",
                                    e
                                ))
                            })?;
                    frame_header = Some(fh);
                    frame_header_offset_in_packet = obu.data_offset; // payload start
                    tile_data_offset_in_packet = obu.data_offset + header_bytes;
                }
                _ => {}
            }
        }

        let fh = frame_header
            .ok_or_else(|| VulkanDecoderError::DecodeError("No frame header found".into()))?;

        // 2. Upload Frame OBU PAYLOAD (stripped of OBU header type+size bytes).
        // FFmpeg uploads raw uncompressed header + tile data, NOT complete OBUs.
        // frameHeaderOffset=0 points to the uncompressed header start.
        let frame_obu_for_upload = obus.iter().find(|o| {
            o.obu_type == av1_obu::ObuType::Frame || o.obu_type == av1_obu::ObuType::TileGroup
                || o.obu_type == av1_obu::ObuType::FrameHeader
        });
        let bitstream_data = match frame_obu_for_upload {
            Some(obu) => &packet_data[obu.data_offset..obu.data_offset + obu.data_size],
            None => packet_data,
        };
        let data_size = bitstream_data.len();
        if data_size > self.bitstream_capacity {
            return Err(VulkanDecoderError::DecodeError(format!(
                "Packet too large: {} > {}",
                data_size, self.bitstream_capacity
            )));
        }

        // Offsets relative to the OBU payload start (= buffer start)
        let payload_start = frame_obu_for_upload.map(|o| o.data_offset).unwrap_or(0);
        let fh_offset_in_buffer = 0usize; // Uncompressed header at buffer byte 0
        let tile_offset_in_buffer = tile_data_offset_in_packet.saturating_sub(payload_start);

        unsafe {
            let ptr = self.device.map_memory(
                self.bitstream_memory,
                0,
                data_size as u64,
                vk::MemoryMapFlags::empty(),
            )? as *mut u8;
            std::ptr::copy_nonoverlapping(bitstream_data.as_ptr(), ptr, data_size);
            self.device.unmap_memory(self.bitstream_memory);
        }

        // 3. Determine DPB slot for this frame's output
        let dst_slot = self.allocate_dpb_slot()?;

        // 4. Build reference slot info using VBI → DPB mapping
        //
        // AV1 reference names: LAST_FRAME(1)..ALTREF_FRAME(7) map to ref_frame_idx[0..6].
        // Each ref_frame_idx[i] is a VBI slot (0..7).
        // vbi_to_dpb[vbi] gives the DPB slot index.
        // referenceNameSlotIndices[i] = DPB slot for reference name i+1.
        let mut ref_slot_indices: [i32; 7] = [-1; 7];

        // Pre-allocate arrays to avoid reallocation (which would invalidate pointers).
        // Max 8 unique DPB slots can be referenced.
        let mut ref_pic_resources: [vk::VideoPictureResourceInfoKHR; 8] =
            std::array::from_fn(|_| vk::VideoPictureResourceInfoKHR::default());
        let mut ref_dpb_slots: [i32; 8] = [-1; 8];
        let mut num_refs = 0usize;

        if !fh.is_intra_frame {
            let mut seen_dpb_slots = [false; 8];
            for i in 0..7 {
                let vbi_slot = fh.ref_frame_idx[i] as usize;
                if vbi_slot < 8 {
                    let dpb_slot = self.vbi_to_dpb[vbi_slot];
                    if dpb_slot >= 0 && (dpb_slot as usize) < 8 {
                        ref_slot_indices[i] = dpb_slot;

                        if !seen_dpb_slots[dpb_slot as usize] {
                            seen_dpb_slots[dpb_slot as usize] = true;
                            ref_pic_resources[num_refs] = vk::VideoPictureResourceInfoKHR::default()
                                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                                .coded_extent(vk::Extent2D {
                                    width: self.width,
                                    height: self.height,
                                })
                                .base_array_layer(0)
                                .image_view_binding(self.dpb_views[dpb_slot as usize]);
                            ref_dpb_slots[num_refs] = dpb_slot;
                            num_refs += 1;
                        }
                    }
                }
            }
        }

        // Build AV1 DPB slot info for each reference (required by Vulkan spec).
        // Each reference slot needs VkVideoDecodeAV1DpbSlotInfoKHR in its pNext chain
        // containing StdVideoDecodeAV1ReferenceInfo with frame type and order hint.
        let mut ref_std_infos: [vk::native::StdVideoDecodeAV1ReferenceInfo; 8] =
            unsafe { std::mem::zeroed() };
        let mut ref_dpb_slot_infos: [vk::VideoDecodeAV1DpbSlotInfoKHR; 8] =
            std::array::from_fn(|_| vk::VideoDecodeAV1DpbSlotInfoKHR::default());

        for i in 0..num_refs {
            let dpb = ref_dpb_slots[i] as usize;
            // Find the VBI slot that maps to this DPB slot
            let vbi_slot = self.vbi_to_dpb.iter().position(|&d| d == dpb as i32).unwrap_or(0);
            ref_std_infos[i] = vk::native::StdVideoDecodeAV1ReferenceInfo {
                flags: unsafe { std::mem::zeroed() },
                frame_type: self.vbi_frame_type[vbi_slot],
                RefFrameSignBias: 0,
                OrderHint: self.vbi_order_hint[vbi_slot],
                SavedOrderHints: self.vbi_order_hint,
            };
            ref_dpb_slot_infos[i] = vk::VideoDecodeAV1DpbSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
                p_next: std::ptr::null(),
                p_std_reference_info: &ref_std_infos[i],
                _marker: std::marker::PhantomData,
            };
        }

        // Build reference_slots with DPB slot info in pNext
        let mut reference_slots: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::with_capacity(num_refs);
        for i in 0..num_refs {
            reference_slots.push(vk::VideoReferenceSlotInfoKHR {
                s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                p_next: &ref_dpb_slot_infos[i] as *const _ as *const std::ffi::c_void,
                slot_index: ref_dpb_slots[i],
                p_picture_resource: &ref_pic_resources[i] as *const _,
                _marker: std::marker::PhantomData,
            });
        }

        // 5. Build AV1 decode picture info
        // Build flags using setters instead of new_bitfield_1 (30 args is error-prone)
        let mut pic_flags = unsafe { std::mem::zeroed::<vk::native::StdVideoDecodeAV1PictureInfoFlags>() };
        pic_flags.set_error_resilient_mode(fh.error_resilient_mode as u32);
        pic_flags.set_disable_cdf_update(fh.disable_cdf_update as u32);
        pic_flags.set_use_superres((fh.use_superres && fh.coded_denom != 8) as u32);
        pic_flags.set_allow_screen_content_tools(fh.allow_screen_content_tools as u32);
        pic_flags.set_force_integer_mv(fh.force_integer_mv as u32);
        pic_flags.set_frame_size_override_flag(fh.frame_size_override_flag as u32);
        pic_flags.set_allow_intrabc(fh.allow_intrabc as u32);

        // Build required sub-structures. These must stay alive for the decode command.
        // For fields we don't fully parse, use safe defaults.
        let sb_size: u32 = if self.sequence_header.use_128x128_superblock { 128 } else { 64 };
        let mi_cols = (self.width + 3) / 4; // 4-pixel MI units
        let mi_rows = (self.height + 3) / 4;
        let sb_cols = (mi_cols + (sb_size / 4) - 1) / (sb_size / 4);
        let sb_rows = (mi_rows + (sb_size / 4) - 1) / (sb_size / 4);

        // Single tile covering entire frame
        let mi_col_starts: [u16; 2] = [0, mi_cols as u16];
        let mi_row_starts: [u16; 2] = [0, mi_rows as u16];
        let width_in_sbs: [u16; 1] = [sb_cols.saturating_sub(1) as u16];
        let height_in_sbs: [u16; 1] = [sb_rows.saturating_sub(1) as u16];

        let mut tile_info_flags: vk::native::StdVideoAV1TileInfoFlags = unsafe { std::mem::zeroed() };
        tile_info_flags.set_uniform_tile_spacing_flag(1);
        let tile_info = vk::native::StdVideoAV1TileInfo {
            flags: tile_info_flags,
            TileCols: 1,
            TileRows: 1,
            context_update_tile_id: 0,
            tile_size_bytes_minus_1: 0,
            reserved1: [0; 7],
            pMiColStarts: mi_col_starts.as_ptr(),
            pMiRowStarts: mi_row_starts.as_ptr(),
            pWidthInSbsMinus1: width_in_sbs.as_ptr(),
            pHeightInSbsMinus1: height_in_sbs.as_ptr(),
        };

        let quantization = vk::native::StdVideoAV1Quantization {
            flags: unsafe { std::mem::zeroed() },
            base_q_idx: {
                eprintln!("  base_q_idx={}, delta_q_y_dc={}, fh_offset={}, tile_offset={}",
                    fh.base_q_idx, fh.delta_q_y_dc, frame_header_offset_in_packet, tile_data_offset_in_packet);
                fh.base_q_idx
            },
            DeltaQYDc: fh.delta_q_y_dc,
            DeltaQUDc: fh.delta_q_u_dc,
            DeltaQUAc: fh.delta_q_u_ac,
            DeltaQVDc: fh.delta_q_v_dc,
            DeltaQVAc: fh.delta_q_v_ac,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        };

        let loop_filter = vk::native::StdVideoAV1LoopFilter {
            flags: unsafe { std::mem::zeroed() },
            loop_filter_level: fh.loop_filter_level,
            loop_filter_sharpness: fh.loop_filter_sharpness,
            update_ref_delta: 0,
            loop_filter_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1], // AV1 defaults
            update_mode_delta: 0,
            loop_filter_mode_deltas: [0, 0],
        };

        let cdef = vk::native::StdVideoAV1CDEF {
            cdef_damping_minus_3: fh.cdef_damping_minus_3,
            cdef_bits: fh.cdef_bits,
            cdef_y_pri_strength: [0; 8], // TODO: pass from parser
            cdef_y_sec_strength: [0; 8],
            cdef_uv_pri_strength: [0; 8],
            cdef_uv_sec_strength: [0; 8],
        };

        let loop_restoration = vk::native::StdVideoAV1LoopRestoration {
            FrameRestorationType: [
                fh.lr_type[0] as u32,
                fh.lr_type[1] as u32,
                fh.lr_type[2] as u32,
            ],
            LoopRestorationSize: [256, 256, 256],
        };

        let global_motion: vk::native::StdVideoAV1GlobalMotion = unsafe { std::mem::zeroed() };

        let segmentation: vk::native::StdVideoAV1Segmentation = unsafe { std::mem::zeroed() };

        let std_pic_info = vk::native::StdVideoDecodeAV1PictureInfo {
            flags: pic_flags,
            frame_type: fh.frame_type as vk::native::StdVideoAV1FrameType,
            current_frame_id: fh.current_frame_id,
            OrderHint: fh.order_hint,
            primary_ref_frame: fh.primary_ref_frame,
            refresh_frame_flags: fh.refresh_frame_flags,
            reserved1: 0,
            interpolation_filter: 0, // EIGHTTAP_REGULAR
            TxMode: 2, // TX_MODE_SELECT
            delta_q_res: 0,
            delta_lf_res: 0,
            SkipModeFrame: [0; 2],
            coded_denom: fh.coded_denom,
            reserved2: [0; 3],
            OrderHints: self.vbi_order_hint,
            expectedFrameId: [0; 8],
            pTileInfo: &tile_info,
            pQuantization: &quantization,
            pSegmentation: &segmentation,
            pLoopFilter: &loop_filter,
            pCDEF: &cdef,
            pLoopRestoration: &loop_restoration,
            pGlobalMotion: &global_motion,
            pFilmGrain: std::ptr::null(),
        };

        // tile_offsets: byte offset from buffer start to the compressed tile payload.
        // tile_sizes: byte size of the compressed tile payload.
        // Per Vulkan spec: these point to the actual tile compressed data,
        // NOT to any OBU headers or tile group framing.
        let tile_offsets = [tile_offset_in_buffer as u32];
        let tile_data_size = (data_size - tile_offset_in_buffer) as u32;
        let tile_sizes = [tile_data_size];

        eprintln!("  tile: fh_offset={}, tile_offset={}, tile_size={}, buf_size={}, first_bytes={:02x?}",
            fh_offset_in_buffer, tile_offset_in_buffer, tile_sizes[0], data_size,
            &bitstream_data[..20.min(data_size)]);

        // NVIDIA driver crashes on inter frames when p_tile_sizes is non-null.
        // Provide tile_sizes only for intra frames.
        let mut av1_pic_info = vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(&std_pic_info)
            .reference_name_slot_indices(ref_slot_indices)
            .frame_header_offset(fh_offset_in_buffer as u32)
            .tile_offsets(&tile_offsets);
        if fh.is_intra_frame {
            av1_pic_info = av1_pic_info.tile_sizes(&tile_sizes);
        }

        // 6. Destination picture resource — same as DPB slot (coincide mode)
        let dst_pic_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.width,
                height: self.height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb_views[dst_slot]);

        let setup_pic_resource = dst_pic_resource;

        // Setup slot — also needs VkVideoDecodeAV1DpbSlotInfoKHR in pNext
        let setup_std_ref_info = vk::native::StdVideoDecodeAV1ReferenceInfo {
            flags: unsafe { std::mem::zeroed() },
            frame_type: fh.frame_type as u8,
            RefFrameSignBias: 0,
            OrderHint: fh.order_hint,
            SavedOrderHints: self.vbi_order_hint,
        };
        let setup_dpb_slot_info = vk::VideoDecodeAV1DpbSlotInfoKHR {
            s_type: vk::StructureType::VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR,
            p_next: std::ptr::null(),
            p_std_reference_info: &setup_std_ref_info,
            _marker: std::marker::PhantomData,
        };
        let dst_slot_info = vk::VideoReferenceSlotInfoKHR {
            s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: &setup_dpb_slot_info as *const _ as *const std::ffi::c_void,
            slot_index: dst_slot as i32,
            p_picture_resource: &setup_pic_resource as *const _,
            _marker: std::marker::PhantomData,
        };

        // 7. Build decode info
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream_buffer)
            .src_buffer_offset(0)
            .src_buffer_range(data_size as u64)
            .dst_picture_resource(dst_pic_resource)
            .setup_reference_slot(&dst_slot_info)
            .reference_slots(&reference_slots)
            .push_next(&mut av1_pic_info);

        // 8. Record and submit command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        let cmd = cmd_bufs[0];

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            log::debug!(
                "decode_frame: frame_count={}, type={:?}, is_intra={}, num_refs={}, dst_slot={}",
                self.frame_count, fh.frame_type, fh.is_intra_frame, num_refs, dst_slot
            );

            // Begin video coding — must include ALL DPB slots (active with index, inactive with -1)
            // per vk-video's pattern and Vulkan spec requirements.
            let mut all_dpb_pic_resources: [vk::VideoPictureResourceInfoKHR; 8] =
                std::array::from_fn(|i| {
                    vk::VideoPictureResourceInfoKHR::default()
                        .coded_offset(vk::Offset2D { x: 0, y: 0 })
                        .coded_extent(vk::Extent2D {
                            width: self.width,
                            height: self.height,
                        })
                        .base_array_layer(0)
                        .image_view_binding(self.dpb_views[i])
                });
            let all_dpb_slots: Vec<vk::VideoReferenceSlotInfoKHR> = (0..8)
                .map(|i| vk::VideoReferenceSlotInfoKHR {
                    s_type: vk::StructureType::VIDEO_REFERENCE_SLOT_INFO_KHR,
                    p_next: std::ptr::null(),
                    slot_index: if self.dpb_slot_active[i] { i as i32 } else { -1 },
                    p_picture_resource: &all_dpb_pic_resources[i] as *const _,
                    _marker: std::marker::PhantomData,
                })
                .collect();

            let begin_info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.video_session)
                .video_session_parameters(self.session_params)
                .reference_slots(&all_dpb_slots);

            log::debug!("  calling cmd_begin_video_coding_khr");
            (self.video_queue_fn.fp().cmd_begin_video_coding_khr)(cmd, &begin_info);
            log::debug!("  cmd_begin_video_coding_khr done");

            // Reset DPB for key frames
            if fh.frame_type == av1_obu::FrameType::KeyFrame {
                let control = vk::VideoCodingControlInfoKHR::default()
                    .flags(vk::VideoCodingControlFlagsKHR::RESET);
                (self.video_queue_fn.fp().cmd_control_video_coding_khr)(cmd, &control);
            }

            // Decode
            log::debug!(
                "  cmd_decode: refs={}, ref_slots={:?}, dst_slot={}, data_size={}",
                reference_slots.len(),
                &ref_slot_indices,
                dst_slot,
                data_size,
            );
            log::debug!("  calling cmd_decode_video_khr");
            let decode_fn = ash::khr::video_decode_queue::Device::new(
                &self.instance,
                &self.device,
            );
            (decode_fn.fp().cmd_decode_video_khr)(cmd, &decode_info);
            log::debug!("  cmd_decode_video_khr done");

            // End video coding
            let end_info = vk::VideoEndCodingInfoKHR::default();
            (self.video_queue_fn.fp().cmd_end_video_coding_khr)(cmd, &end_info);
            log::debug!("  cmd_end + end_command_buffer");

            self.device.end_command_buffer(cmd)?;

            // Submit
            // Submit with fence for synchronization
            let fence_create = vk::FenceCreateInfo::default();
            let fence = self.device.create_fence(&fence_create, None)?;
            let cmd_bufs_decode = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs_decode);
            self.device
                .queue_submit(self.video_decode_queue, &[submit], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);

            // Free command buffer
            self.device
                .free_command_buffers(self.command_pool, &[cmd]);
        }

        // 9. Update VBI → DPB mapping from refresh_frame_flags.
        // For each VBI slot flagged in refresh_frame_flags, map it to the dst_slot
        // where this frame was decoded.
        for i in 0..8 {
            if fh.refresh_frame_flags & (1 << i) != 0 {
                self.vbi_to_dpb[i] = dst_slot as i32;
                self.vbi_order_hint[i] = fh.order_hint;
                self.vbi_frame_type[i] = fh.frame_type as u8;
            }
        }
        self.dpb_slot_active[dst_slot] = true;

        self.frame_count += 1;

        Ok(DecodeOutput {
            frame_index: self.frame_count - 1,
            show_frame: fh.show_frame,
            frame_type: fh.frame_type,
            dpb_slot: dst_slot,
        })
    }

    /// Allocate a DPB slot for the decoded frame.
    /// Finds a slot not referenced by any current VBI entry.
    fn allocate_dpb_slot(&self) -> Result<usize, VulkanDecoderError> {
        // Build set of DPB slots currently in use by VBI
        let mut in_use = [false; 8];
        for &dpb in &self.vbi_to_dpb {
            if dpb >= 0 && (dpb as usize) < 8 {
                in_use[dpb as usize] = true;
            }
        }
        // Find first free slot
        for i in 0..8 {
            if !in_use[i] {
                return Ok(i);
            }
        }
        // All slots in use — this shouldn't happen with correct AV1 streams.
        // Reuse slot 0 as fallback.
        log::warn!("All 8 DPB slots in use, reusing slot 0");
        Ok(0)
    }

    /// Read back the decoded frame from the DPB slot to CPU as NV12 data.
    pub fn read_back_nv12(
        &self,
        dpb_slot: usize,
    ) -> Result<(Vec<u8>, Vec<u8>), VulkanDecoderError> {
        let y_size = (self.width * self.height) as usize;
        let uv_size = (self.width * self.height / 2) as usize;
        let total_size = y_size + uv_size;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        let cmd = cmd_bufs[0];

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // Transition DPB image: VIDEO_DECODE_DPB → TRANSFER_SRC
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.dpb_images[dpb_slot])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::NONE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy Y and UV planes
            let y_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                });

            let uv_region = vk::BufferImageCopy::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.width / 2,
                    height: self.height / 2,
                    depth: 1,
                });

            self.device.cmd_copy_image_to_buffer(
                cmd,
                self.dpb_images[dpb_slot],
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.staging_buffer,
                &[y_region, uv_region],
            );

            // Transition back: TRANSFER_SRC → VIDEO_DECODE_DPB
            let barrier_back = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.dpb_images[dpb_slot])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::NONE);

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_back],
            );

            self.device.end_command_buffer(cmd)?;

            let readback_fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmd_bufs_submit = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs_submit);
            self.device
                .queue_submit(self.video_decode_queue, &[submit], readback_fence)?;
            self.device.wait_for_fences(&[readback_fence], true, u64::MAX)?;
            self.device.destroy_fence(readback_fence, None);

            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }

        // Map staging buffer and read
        let mut y_data = vec![0u8; y_size];
        let mut uv_data = vec![0u8; uv_size];
        unsafe {
            let ptr = self.device.map_memory(
                self.staging_memory,
                0,
                total_size as u64,
                vk::MemoryMapFlags::empty(),
            )? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, y_data.as_mut_ptr(), y_size);
            std::ptr::copy_nonoverlapping(ptr.add(y_size), uv_data.as_mut_ptr(), uv_size);
            self.device.unmap_memory(self.staging_memory);
        }

        Ok((y_data, uv_data))
    }

    /// Convert NV12 to RGBA for display.
    pub fn nv12_to_rgba(
        y_data: &[u8],
        uv_data: &[u8],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut rgba = vec![0u8; w * h * 4];

        for row in 0..h {
            for col in 0..w {
                let y = y_data[row * w + col] as f32;
                let uv_row = row / 2;
                let uv_col = (col / 2) * 2;
                let u = uv_data[uv_row * w + uv_col] as f32 - 128.0;
                let v = uv_data[uv_row * w + uv_col + 1] as f32 - 128.0;

                let r = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
                let g = (y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
                let b = (y + 1.772 * u).clamp(0.0, 255.0) as u8;

                let idx = (row * w + col) * 4;
                rgba[idx] = r;
                rgba[idx + 1] = g;
                rgba[idx + 2] = b;
                rgba[idx + 3] = 255;
            }
        }

        rgba
    }
}

/// Result of decoding one frame.
pub struct DecodeOutput {
    pub frame_index: u64,
    pub show_frame: bool,
    pub frame_type: av1_obu::FrameType,
    pub dpb_slot: usize,
}

impl Drop for AV1Decoder {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            self.device
                .destroy_command_pool(self.command_pool, None);
            self.device.destroy_buffer(self.bitstream_buffer, None);
            self.device.free_memory(self.bitstream_memory, None);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);
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

            if self.owns_device {
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
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
                    | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::TRANSFER_SRC,
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
