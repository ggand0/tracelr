use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Once};
use std::time::Instant;

use eframe::egui;

static FFMPEG_INIT: Once = Once::new();

fn ensure_ffmpeg_init() {
    FFMPEG_INIT.call_once(|| {
        ffmpeg_next::init().expect("Failed to initialize ffmpeg");
    });
}

/// Result of decoding a single frame for the episode cache.
pub(crate) struct DecodeResult {
    pub episode_index: usize,
    pub image: Option<egui::ColorImage>,
    #[allow(dead_code)]
    pub decode_ms: f64,
}

/// Decode a single frame from approximately the middle of a video file.
///
/// Opens the mp4, seeks to the midpoint, decodes the first available frame
/// after the seek position, and converts to RGBA.
/// `seek_range` is an optional (from_seconds, to_seconds) range for v3.0 datasets
/// where multiple episodes share one video file.
pub(crate) fn decode_middle_frame(
    video_path: &Path,
    seek_range: Option<(f64, f64)>,
) -> Result<egui::ColorImage, Box<dyn std::error::Error + Send + Sync>> {
    ensure_ffmpeg_init();

    let mut ictx = ffmpeg_next::format::input(video_path)?;

    let video_stream_index;
    let total_frames;
    let decoder_params;
    {
        let stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or("No video stream found")?;
        video_stream_index = stream.index();
        total_frames = stream.frames() as usize;
        decoder_params = stream.parameters();
    }

    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(decoder_params)?
        .decoder()
        .video()?;

    // Seek to the midpoint of the episode's time range
    let target_us = if let Some((from_s, to_s)) = seek_range {
        let mid_s = (from_s + to_s) / 2.0;
        (mid_s * 1_000_000.0) as i64
    } else {
        let duration = ictx.duration();
        if duration > 0 && total_frames > 2 {
            duration / 2
        } else {
            0
        }
    };
    if target_us > 0 {
        let _ = ictx.seek(target_us, ..target_us);
    }

    // Decode the first available frame after the seek position.
    let mut decoded = ffmpeg_next::frame::Video::empty();

    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            return frame_to_color_image(&decoded);
        }
    }

    // Flush the decoder in case there are buffered frames.
    decoder.send_eof()?;
    if decoder.receive_frame(&mut decoded).is_ok() {
        return frame_to_color_image(&decoded);
    }

    Err("No frames could be decoded".into())
}

/// Decode the first frame of a video file (fast, no seeking).
#[allow(dead_code)]
pub(crate) fn decode_first_frame(
    video_path: &Path,
) -> Result<egui::ColorImage, Box<dyn std::error::Error + Send + Sync>> {
    ensure_ffmpeg_init();

    let mut ictx = ffmpeg_next::format::input(video_path)?;

    let video_stream_index;
    let decoder_params;
    {
        let stream = ictx
            .streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or("No video stream found")?;
        video_stream_index = stream.index();
        decoder_params = stream.parameters();
    }

    let mut decoder = ffmpeg_next::codec::context::Context::from_parameters(decoder_params)?
        .decoder()
        .video()?;

    let mut decoded = ffmpeg_next::frame::Video::empty();
    for (stream, packet) in ictx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            return frame_to_color_image(&decoded);
        }
    }

    Err("No frames could be decoded".into())
}

/// Convert an ffmpeg Video frame to an egui ColorImage (RGBA).
fn frame_to_color_image(
    frame: &ffmpeg_next::frame::Video,
) -> Result<egui::ColorImage, Box<dyn std::error::Error + Send + Sync>> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;

    // Convert to RGBA using swscale
    let mut scaler = ffmpeg_next::software::scaling::Context::get(
        frame.format(),
        frame.width(),
        frame.height(),
        ffmpeg_next::format::Pixel::RGBA,
        frame.width(),
        frame.height(),
        ffmpeg_next::software::scaling::Flags::BILINEAR,
    )?;

    let mut rgba_frame = ffmpeg_next::frame::Video::empty();
    scaler.run(frame, &mut rgba_frame)?;

    let data = rgba_frame.data(0);
    let stride = rgba_frame.stride(0);

    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let offset = y * stride + x * 4;
            pixels.push(egui::Color32::from_rgba_premultiplied(
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ));
        }
    }

    Ok(egui::ColorImage {
        size: [width, height],
        pixels,
    })
}

/// Result of decoding a single video frame (for FrameCache).
pub(crate) struct FrameDecodeResult {
    pub frame_index: usize,
    pub image: Option<egui::ColorImage>,
}

/// Decode frames sequentially from a video file, sending via bounded channel.
/// Uses `SyncSender` so the thread blocks when the buffer is full, naturally
/// pacing itself to stay ~30 frames ahead of the consumer.
///
/// For 480×640 AV1, sequential decode is ~2-5ms/frame.
pub(crate) fn decode_all_frames_sync(
    video_path: &Path,
    tx: mpsc::SyncSender<FrameDecodeResult>,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
    seek_to_frame: Option<usize>,
) {
    decode_all_frames_inner(video_path, &tx, &cancel, &ctx, seek_to_frame);
}

/// Decode frames sequentially, sending via unbounded channel.
#[allow(dead_code)]
pub(crate) fn decode_all_frames(
    video_path: &Path,
    tx: mpsc::Sender<FrameDecodeResult>,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
    seek_to_frame: Option<usize>,
) {
    decode_all_frames_inner(video_path, &tx, &cancel, &ctx, seek_to_frame);
}

/// Trait to abstract over Sender and SyncSender for decode_all_frames_inner.
trait FrameSender {
    fn send_frame(&self, result: FrameDecodeResult) -> bool;
}
impl FrameSender for mpsc::Sender<FrameDecodeResult> {
    fn send_frame(&self, result: FrameDecodeResult) -> bool {
        self.send(result).is_ok()
    }
}
impl FrameSender for mpsc::SyncSender<FrameDecodeResult> {
    fn send_frame(&self, result: FrameDecodeResult) -> bool {
        self.send(result).is_ok()
    }
}

fn decode_all_frames_inner(
    video_path: &Path,
    tx: &dyn FrameSender,
    cancel: &AtomicBool,
    ctx: &egui::Context,
    seek_to_frame: Option<usize>,
) {
    ensure_ffmpeg_init();

    let mut ictx = match ffmpeg_next::format::input(video_path) {
        Ok(c) => c,
        Err(e) => {
            log::error!("Failed to open video {}: {}", video_path.display(), e);
            return;
        }
    };

    let video_stream_index;
    let decoder_params;
    let fps;
    {
        let stream = match ictx.streams().best(ffmpeg_next::media::Type::Video) {
            Some(s) => s,
            None => return,
        };
        video_stream_index = stream.index();
        decoder_params = stream.parameters();
        // Get fps from stream rate
        let rate = stream.rate();
        fps = if rate.1 > 0 {
            rate.0 as f64 / rate.1 as f64
        } else {
            30.0
        };
    }

    let mut decoder = match ffmpeg_next::codec::context::Context::from_parameters(decoder_params)
        .and_then(|c| c.decoder().video())
    {
        Ok(d) => d,
        Err(e) => {
            log::error!("Failed to create decoder: {}", e);
            return;
        }
    };

    // Optionally seek to approximate position
    if let Some(target_frame) = seek_to_frame {
        if target_frame > 0 {
            let target_us = (target_frame as f64 / fps * 1_000_000.0) as i64;
            let _ = ictx.seek(target_us, ..target_us);
        }
    }

    let mut frame_count: usize = 0;
    let mut decoded = ffmpeg_next::frame::Video::empty();

    // If we seeked, we don't know the exact frame index. Estimate from PTS.
    let mut pts_base: Option<(i64, ffmpeg_next::Rational)> = None;

    for (stream, packet) in ictx.packets() {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        if stream.index() != video_stream_index {
            continue;
        }

        if pts_base.is_none() {
            pts_base = Some((packet.pts().unwrap_or(0), stream.time_base()));
        }

        decoder.send_packet(&packet).ok();
        while decoder.receive_frame(&mut decoded).is_ok() {
            if cancel.load(Ordering::Relaxed) {
                return;
            }

            // Calculate frame index from PTS
            let actual_frame = if let (Some(pts), Some((_, tb))) = (decoded.pts(), &pts_base) {
                let time_s = pts as f64 * tb.0 as f64 / tb.1 as f64;
                (time_s * fps).round() as usize
            } else {
                frame_count
            };

            let image = frame_to_color_image(&decoded).ok();
            if !tx.send_frame(FrameDecodeResult {
                frame_index: actual_frame,
                image,
            }) {
                return; // receiver dropped
            }
            ctx.request_repaint();
            frame_count += 1;
        }
    }

    // Flush
    decoder.send_eof().ok();
    while decoder.receive_frame(&mut decoded).is_ok() {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let actual_frame = if let (Some(pts), Some((_, tb))) = (decoded.pts(), &pts_base) {
            let time_s = pts as f64 * tb.0 as f64 / tb.1 as f64;
            (time_s * fps).round() as usize
        } else {
            frame_count
        };
        let image = frame_to_color_image(&decoded).ok();
        if !tx.send_frame(FrameDecodeResult {
            frame_index: actual_frame,
            image,
        }) {
            return;
        }
        frame_count += 1;
    }
}

/// Decode frames using GPU hardware decoder (Vulkan Video via vk-video).
/// Supports H.264 and AV1 codecs.
/// Uses ffmpeg for demuxing only — the actual decode happens on the GPU.
/// Falls back to software decode if GPU init fails.
pub(crate) fn decode_all_frames_gpu(
    video_path: &Path,
    tx: mpsc::SyncSender<FrameDecodeResult>,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
    seek_to_frame: Option<usize>,
) {
    use std::sync::Arc;
    use vk_video::{
        EncodedInputChunk, OutputFrame, VulkanInstance,
        parameters::{DecoderParameters, VulkanAdapterDescriptor, VulkanDeviceDescriptor},
    };

    ensure_ffmpeg_init();

    // 1. Open video for demuxing
    let mut ictx = match ffmpeg_next::format::input(video_path) {
        Ok(c) => c,
        Err(e) => {
            log::error!("GPU decode: failed to open {}: {}", video_path.display(), e);
            return;
        }
    };

    let video_stream_index;
    let codec_id;
    let fps;
    let width;
    let height;
    let mut extradata_annexb = Vec::new();
    {
        let stream = match ictx.streams().best(ffmpeg_next::media::Type::Video) {
            Some(s) => s,
            None => return,
        };
        video_stream_index = stream.index();
        let params = stream.parameters();
        let rate = stream.rate();
        fps = if rate.1 > 0 { rate.0 as f64 / rate.1 as f64 } else { 30.0 };

        // Get codec and dimensions
        let codec_ctx = match ffmpeg_next::codec::context::Context::from_parameters(params) {
            Ok(c) => c,
            Err(e) => {
                log::error!("GPU decode: failed to create codec context: {}", e);
                return;
            }
        };
        codec_id = unsafe { (*codec_ctx.as_ptr()).codec_id };
        width = unsafe { (*codec_ctx.as_ptr()).width as u32 };
        height = unsafe { (*codec_ctx.as_ptr()).height as u32 };

        // For H.264 in MP4: extract SPS/PPS from extradata and convert to Annex B
        unsafe {
            let ptr = (*codec_ctx.as_ptr()).extradata;
            let size = (*codec_ctx.as_ptr()).extradata_size as usize;
            if !ptr.is_null() && size > 0 {
                let extra = std::slice::from_raw_parts(ptr, size);
                extradata_annexb = avc_to_annexb_extradata(extra);
            }
        }
    }

    let is_h264 = codec_id == ffmpeg_next::ffi::AVCodecID::AV_CODEC_ID_H264;
    let is_av1 = codec_id == ffmpeg_next::ffi::AVCodecID::AV_CODEC_ID_AV1;
    log::info!("GPU decode: codec={:?} h264={} av1={} {}x{}", codec_id, is_h264, is_av1, width, height);

    if !is_h264 {
        // AV1 GPU decode has parser issues with some encodings — use software for now
        if is_av1 {
            log::info!("GPU decode: AV1 detected, using software decode (GPU AV1 parser WIP)");
        } else {
            log::warn!("GPU decode: unsupported codec, falling back to software");
        }
        decode_all_frames_inner(video_path, &tx, &cancel, &ctx, seek_to_frame);
        return;
    }

    // 2. Create Vulkan device (shared between H.264 and AV1)
    let vulkan_instance = match VulkanInstance::new() {
        Ok(i) => i,
        Err(e) => {
            log::warn!("GPU decode: VulkanInstance failed ({}), falling back", e);
            decode_all_frames_inner(video_path, &tx, &cancel, &ctx, seek_to_frame);
            return;
        }
    };
    let vulkan_adapter = match vulkan_instance.create_adapter(&VulkanAdapterDescriptor::default()) {
        Ok(a) => a,
        Err(e) => {
            log::warn!("GPU decode: adapter failed ({}), falling back", e);
            decode_all_frames_inner(video_path, &tx, &cancel, &ctx, seek_to_frame);
            return;
        }
    };
    let vulkan_device = match vulkan_adapter.create_device(&VulkanDeviceDescriptor::default()) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("GPU decode: device failed ({}), falling back", e);
            decode_all_frames_inner(video_path, &tx, &cancel, &ctx, seek_to_frame);
            return;
        }
    };

    // 3. Optionally seek
    if let Some(target_frame) = seek_to_frame {
        if target_frame > 0 {
            let target_us = (target_frame as f64 / fps * 1_000_000.0) as i64;
            let _ = ictx.seek(target_us, ..target_us);
        }
    }

    // 4. Dispatch to codec-specific decode loop
    if is_h264 {
        gpu_decode_h264(&vulkan_device, &mut ictx, video_stream_index, &extradata_annexb,
            &tx, &cancel, &ctx, video_path);
    } else {
        gpu_decode_av1(&vulkan_device, &mut ictx, video_stream_index, &extradata_annexb,
            width, height, &tx, &cancel, &ctx, video_path);
    }
}

fn gpu_decode_h264(
    vulkan_device: &std::sync::Arc<vk_video::VulkanDevice>,
    ictx: &mut ffmpeg_next::format::context::Input,
    video_stream_index: usize,
    extradata_annexb: &[u8],
    tx: &mpsc::SyncSender<FrameDecodeResult>,
    cancel: &Arc<AtomicBool>,
    ctx: &egui::Context,
    video_path: &Path,
) {
    use vk_video::{EncodedInputChunk, OutputFrame, parameters::DecoderParameters};

    let mut decoder = match vulkan_device.create_bytes_decoder(DecoderParameters::default()) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("GPU H.264 decode failed ({}), falling back", e);
            decode_all_frames_inner(video_path, tx, cancel, ctx, None);
            return;
        }
    };
    log::info!("GPU H.264 decoder initialized");

    if !extradata_annexb.is_empty() {
        let chunk = EncodedInputChunk { data: extradata_annexb, pts: None };
        let _ = decoder.decode(chunk);
    }

    let mut frame_count: usize = 0;
    for (stream, packet) in ictx.packets() {
        if cancel.load(Ordering::Relaxed) { return; }
        if stream.index() != video_stream_index { continue; }
        let pkt_data = match packet.data() { Some(d) => d, None => continue };

        let annexb = avc_to_annexb(pkt_data);
        let chunk = EncodedInputChunk {
            data: annexb.as_slice(),
            pts: packet.pts().map(|p| p as u64),
        };
        match decoder.decode(chunk) {
            Ok(frames) => {
                for OutputFrame { data: raw, .. } in frames {
                    let rgba = nv12_to_rgba(&raw.frame, raw.width, raw.height);
                    let image = egui::ColorImage {
                        size: [raw.width as usize, raw.height as usize],
                        pixels: rgba.chunks_exact(4)
                            .map(|c| egui::Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3]))
                            .collect(),
                    };
                    if !tx.send_frame(FrameDecodeResult { frame_index: frame_count, image: Some(image) }) { return; }
                    ctx.request_repaint();
                    frame_count += 1;
                }
            }
            Err(e) => { log::warn!("GPU H.264 decode error at frame {}: {}", frame_count, e); }
        }
    }
    if let Ok(frames) = decoder.flush() {
        for OutputFrame { data: raw, .. } in frames {
            let rgba = nv12_to_rgba(&raw.frame, raw.width, raw.height);
            let image = egui::ColorImage {
                size: [raw.width as usize, raw.height as usize],
                pixels: rgba.chunks_exact(4)
                    .map(|c| egui::Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3]))
                    .collect(),
            };
            if !tx.send_frame(FrameDecodeResult { frame_index: frame_count, image: Some(image) }) { return; }
            ctx.request_repaint();
            frame_count += 1;
        }
    }
    log::info!("GPU H.264 decode complete: {} frames from {}", frame_count, video_path.display());
}

fn gpu_decode_av1(
    vulkan_device: &std::sync::Arc<vk_video::VulkanDevice>,
    ictx: &mut ffmpeg_next::format::context::Input,
    video_stream_index: usize,
    extradata_raw: &[u8],
    width: u32,
    height: u32,
    tx: &mpsc::SyncSender<FrameDecodeResult>,
    cancel: &Arc<AtomicBool>,
    ctx: &egui::Context,
    video_path: &Path,
) {
    use vk_video::{AV1VulkanDecoder, AV1DecoderError};
    use vk_video::parser::av1::{self, obu as av1_obu};
    use vk_video::parser::av1::decoder_instructions::compile_temporal_unit;
    use vk_video::parser::av1::reference_manager::AV1ReferenceContext;

    // Parse sequence header from av1C extradata (skip 4-byte config header)
    log::info!("GPU AV1: extradata len={}", extradata_raw.len());
    let obu_data = if extradata_raw.len() > 4 { &extradata_raw[4..] } else { extradata_raw };
    let seq = match av1_obu::parse_obu_headers(obu_data)
        .ok()
        .and_then(|obus| obus.iter().find(|o| o.obu_type == av1_obu::ObuType::SequenceHeader).cloned())
        .and_then(|seq_obu| av1_obu::parse_sequence_header(&obu_data[seq_obu.data_offset..seq_obu.data_offset + seq_obu.data_size]).ok())
    {
        Some(s) => s,
        None => {
            // Try parsing first packet for sequence header
            log::warn!("GPU AV1: no SH in extradata, will try first packet");
            // Fall back: get SH from first packet
            let mut seq_from_pkt = None;
            for (stream, packet) in ictx.packets() {
                if stream.index() != video_stream_index { continue; }
                if let Some(d) = packet.data() {
                    if let Ok(obus) = av1_obu::parse_obu_headers(d) {
                        for obu in &obus {
                            if obu.obu_type == av1_obu::ObuType::SequenceHeader {
                                if let Ok(s) = av1_obu::parse_sequence_header(&d[obu.data_offset..obu.data_offset + obu.data_size]) {
                                    seq_from_pkt = Some(s);
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }
            }
            match seq_from_pkt {
                Some(s) => {
                    // Re-seek to beginning
                    let _ = ictx.seek(0, ..0);
                    s
                }
                None => {
                    log::error!("GPU AV1: cannot find sequence header, falling back");
                    decode_all_frames_inner(video_path, tx, cancel, ctx, None);
                    return;
                }
            }
        }
    };

    log::info!("GPU AV1: sequence header parsed, {}x{}", seq.max_frame_width_minus_1 + 1, seq.max_frame_height_minus_1 + 1);
    let mut decoder = match AV1VulkanDecoder::new(vulkan_device.clone(), &seq) {
        Ok(d) => d,
        Err(e) => {
            log::warn!("GPU AV1 decoder failed ({}), falling back", e);
            decode_all_frames_inner(video_path, tx, cancel, ctx, None);
            return;
        }
    };
    log::info!("GPU AV1 decoder initialized ({}x{})", width, height);

    let mut ref_ctx = AV1ReferenceContext::new();
    let mut frame_count: usize = 0;

    for (stream, packet) in ictx.packets() {
        if cancel.load(Ordering::Relaxed) { return; }
        if stream.index() != video_stream_index { continue; }
        let pkt_data = match packet.data() { Some(d) => d, None => continue };

        let tus = match av1::parse_packet(pkt_data, &seq, Some(frame_count as u64)) {
            Ok(t) => t,
            Err(_) => continue,
        };

        for tu in &tus {
            let instructions = compile_temporal_unit(tu, &seq, &mut ref_ctx);
            for instr in &instructions {
                use vk_video::parser::av1::decoder_instructions::AV1DecoderInstruction;
                match instr {
                    AV1DecoderInstruction::Decode { info, reference_id, is_key_frame } => {
                        match decoder.decode_frame(info, *reference_id, *is_key_frame) {
                            Ok(Some(raw)) => {
                                let rgba = nv12_to_rgba(&raw.frame, raw.width, raw.height);
                                let image = egui::ColorImage {
                                    size: [raw.width as usize, raw.height as usize],
                                    pixels: rgba.chunks_exact(4)
                                        .map(|c| egui::Color32::from_rgba_premultiplied(c[0], c[1], c[2], c[3]))
                                        .collect(),
                                };
                                if !tx.send_frame(FrameDecodeResult { frame_index: frame_count, image: Some(image) }) { return; }
                                ctx.request_repaint();
                                frame_count += 1;
                            }
                            Ok(None) => {} // show_frame=false, decoded but not displayed
                            Err(e) => { log::warn!("GPU AV1 decode error: {}", e); }
                        }
                    }
                    AV1DecoderInstruction::Drop { reference_ids } => {
                        for rid in reference_ids { decoder.drop_reference(rid); }
                    }
                    _ => {}
                }
            }
        }
    }
    log::info!("GPU AV1 decode complete: {} frames from {}", frame_count, video_path.display());
}

/// Convert AVC extradata (SPS/PPS in avcc format) to Annex B format.
fn avc_to_annexb_extradata(extra: &[u8]) -> Vec<u8> {
    if extra.len() < 7 { return Vec::new(); }
    let mut out = Vec::new();
    let nalu_len_size = ((extra[4] & 0x03) + 1) as usize;
    let _ = nalu_len_size; // used in avc_to_annexb

    // SPS
    let num_sps = (extra[5] & 0x1F) as usize;
    let mut off = 6;
    for _ in 0..num_sps {
        if off + 2 > extra.len() { break; }
        let len = u16::from_be_bytes([extra[off], extra[off+1]]) as usize;
        off += 2;
        if off + len > extra.len() { break; }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&extra[off..off+len]);
        off += len;
    }

    // PPS
    if off >= extra.len() { return out; }
    let num_pps = extra[off] as usize;
    off += 1;
    for _ in 0..num_pps {
        if off + 2 > extra.len() { break; }
        let len = u16::from_be_bytes([extra[off], extra[off+1]]) as usize;
        off += 2;
        if off + len > extra.len() { break; }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&extra[off..off+len]);
        off += len;
    }

    out
}

/// Convert AVC packet (length-prefixed NALUs) to Annex B (start code prefixed).
fn avc_to_annexb(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 32);
    let mut off = 0;
    while off + 4 <= data.len() {
        let nalu_len = u32::from_be_bytes([data[off], data[off+1], data[off+2], data[off+3]]) as usize;
        off += 4;
        if off + nalu_len > data.len() { break; }
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(&data[off..off+nalu_len]);
        off += nalu_len;
    }
    out
}

/// Convert NV12 frame data to RGBA.
fn nv12_to_rgba(nv12: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let y_plane = &nv12[..w * h];
    let uv_plane = &nv12[w * h..];
    let mut rgba = vec![0u8; w * h * 4];
    for row in 0..h {
        for col in 0..w {
            let y = y_plane[row * w + col] as f32;
            let uv_row = row / 2;
            let uv_col = (col / 2) * 2;
            let u = uv_plane[uv_row * w + uv_col] as f32 - 128.0;
            let v = uv_plane[uv_row * w + uv_col + 1] as f32 - 128.0;
            let r = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
            let g = (y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * u).clamp(0.0, 255.0) as u8;
            let idx = (row * w + col) * 4;
            rgba[idx] = r; rgba[idx+1] = g; rgba[idx+2] = b; rgba[idx+3] = 255;
        }
    }
    rgba
}

/// Decode middle frame with timing info, for use in background threads.
pub(crate) fn decode_middle_frame_timed(
    video_path: &Path,
    episode_index: usize,
    seek_range: Option<(f64, f64)>,
) -> DecodeResult {
    let start = Instant::now();
    let image = decode_middle_frame(video_path, seek_range)
        .map_err(|e| {
            log::warn!(
                "Failed to decode episode {} from {}: {}",
                episode_index,
                video_path.display(),
                e
            );
        })
        .ok();
    let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    DecodeResult {
        episode_index,
        image,
        decode_ms,
    }
}
