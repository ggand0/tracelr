//! GPU-accelerated AV1 video decoding via Vulkan Video.
//!
//! Architecture:
//! - `av1_obu`: AV1 OBU (Open Bitstream Unit) parser extracting sequence headers,
//!   frame headers, and tile info needed by the Vulkan AV1 decode API.
//! - `vulkan_decoder`: Vulkan Video session management, DPB, and decode submission
//!   using ash (raw Vulkan bindings) with VK_KHR_video_decode_av1.
//! - Decoded frames are wgpu::Texture (zero-copy, never leave GPU memory).

pub mod av1_obu;
pub mod vulkan_decoder;
#[cfg(test)]
mod test_av1;
