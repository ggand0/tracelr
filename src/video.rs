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
pub(crate) fn decode_middle_frame(
    video_path: &Path,
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

    // Seek to approximately the middle of the file.
    // duration() returns microseconds (AV_TIME_BASE = 1_000_000).
    let duration = ictx.duration();
    if duration > 0 && total_frames > 2 {
        let target_us = duration / 2;
        // seek() takes a timestamp range in AV_TIME_BASE units.
        // Seek backward to the nearest keyframe before target.
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

/// Decode all frames from a video file sequentially, sending each via channel.
/// Starts from the beginning (or optionally seeks first). Checks `cancel` flag
/// between frames so the thread can be stopped when the user switches episodes.
///
/// For 480×640 AV1, sequential decode is ~2-5ms/frame. A 600-frame episode
/// takes ~1.2-3 seconds to fully decode from the start.
pub(crate) fn decode_all_frames(
    video_path: &Path,
    tx: mpsc::Sender<FrameDecodeResult>,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
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
            if tx
                .send(FrameDecodeResult {
                    frame_index: actual_frame,
                    image,
                })
                .is_err()
            {
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
        let _ = tx.send(FrameDecodeResult {
            frame_index: actual_frame,
            image,
        });
        frame_count += 1;
    }
}

/// Decode middle frame with timing info, for use in background threads.
pub(crate) fn decode_middle_frame_timed(
    video_path: &Path,
    episode_index: usize,
) -> DecodeResult {
    let start = Instant::now();
    let image = decode_middle_frame(video_path)
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
