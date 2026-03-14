//! Vulkan Video AV1 decoder using VK_KHR_video_decode_av1.
//!
//! Creates a Vulkan device with video decode queue, sets up an AV1 video
//! session, and decodes frames on the GPU. Decoded frames are NV12 textures
//! that can be imported into wgpu.
//!
//! This is a minimal implementation focused on AV1 decode for the LeRobot
//! annotation tool. It does NOT support encoding, transcoding, or other codecs.

use ash::vk;
use std::ffi::CStr;

use super::av1_obu::SequenceHeader;

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

// TODO: Next steps:
// 1. Create VkVideoSessionKHR with AV1 profile
// 2. Create VkVideoSessionParametersKHR from parsed SequenceHeader
// 3. Allocate DPB images (NV12 format)
// 4. Record and submit decode commands (cmd_decode_video_khr)
// 5. Export decoded images to wgpu::Texture via DMA-BUF or external memory
