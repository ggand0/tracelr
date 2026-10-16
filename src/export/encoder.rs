use std::path::Path;

use ffmpeg_next::{
    codec, encoder, format, frame, software::scaling, Dictionary, Packet, Rational,
};

/// Encodes straight RGBA frames into an h264 (or mpeg4 fallback) yuv420p mp4.
pub(crate) struct Mp4Writer {
    octx: format::context::Output,
    encoder: encoder::Video,
    stream_index: usize,
    stream_time_base: Rational,
    frame_time_base: Rational,
    scaler: scaling::Context,
    rgba: frame::Video,
    yuv: frame::Video,
    frame_index: i64,
    width: u32,
    height: u32,
    /// Name of the codec in use ("libx264" or "mpeg4").
    pub codec_name: &'static str,
}

impl Mp4Writer {
    /// `width` and `height` must be even (yuv420p).
    pub fn create(path: &Path, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        crate::video::ensure_ffmpeg_init();
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(format!("Video size must be even, got {}x{}", width, height));
        }
        let fps = fps.max(1) as i32;

        let (codec, codec_name) = match encoder::find_by_name("libx264") {
            Some(c) => (c, "libx264"),
            None => {
                log::warn!("libx264 not available, falling back to mpeg4");
                let c = encoder::find(codec::Id::MPEG4)
                    .ok_or("No h264 or mpeg4 encoder available in this ffmpeg build")?;
                (c, "mpeg4")
            }
        };

        let mut octx = format::output(path)
            .map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
        let global_header = octx.format().flags().contains(format::Flags::GLOBAL_HEADER);
        let stream_index = octx
            .add_stream(codec)
            .map_err(|e| format!("Failed to add video stream: {}", e))?
            .index();

        let mut enc = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|e| format!("Failed to create encoder: {}", e))?;
        let frame_time_base = Rational(1, fps);
        enc.set_width(width);
        enc.set_height(height);
        enc.set_format(format::Pixel::YUV420P);
        enc.set_time_base(frame_time_base);
        enc.set_frame_rate(Some(Rational(fps, 1)));
        // swscale's RGB -> YUV conversion is BT.601 limited range; tag it so players agree.
        enc.set_colorspace(ffmpeg_next::color::Space::SMPTE170M);
        enc.set_color_range(ffmpeg_next::color::Range::MPEG);
        if global_header {
            enc.set_flags(codec::Flags::GLOBAL_HEADER);
        }

        let mut opts = Dictionary::new();
        if codec_name == "libx264" {
            opts.set("preset", "veryfast");
            opts.set("crf", "20");
        } else {
            enc.set_bit_rate(4_000_000);
        }
        let encoder = enc
            .open_with(opts)
            .map_err(|e| format!("Failed to open {} encoder: {}", codec_name, e))?;

        octx.stream_mut(stream_index)
            .ok_or("Output stream disappeared")?
            .set_parameters(&encoder);
        octx.write_header()
            .map_err(|e| format!("Failed to write mp4 header: {}", e))?;
        let stream_time_base = octx
            .stream(stream_index)
            .ok_or("Output stream disappeared")?
            .time_base();

        let scaler = scaling::Context::get(
            format::Pixel::RGBA,
            width,
            height,
            format::Pixel::YUV420P,
            width,
            height,
            scaling::Flags::BILINEAR,
        )
        .map_err(|e| format!("Failed to create scaler: {}", e))?;

        Ok(Self {
            octx,
            encoder,
            stream_index,
            stream_time_base,
            frame_time_base,
            scaler,
            rgba: frame::Video::new(format::Pixel::RGBA, width, height),
            yuv: frame::Video::new(format::Pixel::YUV420P, width, height),
            frame_index: 0,
            width,
            height,
            codec_name,
        })
    }

    /// Encode one frame. `rgba.len()` must be `width * height * 4`.
    pub fn push_rgba(&mut self, rgba: &[u8]) -> Result<(), String> {
        let row = self.width as usize * 4;
        let rows = self.height as usize;
        if rgba.len() != row * rows {
            return Err(format!(
                "Frame is {} bytes, expected {} for {}x{}",
                rgba.len(),
                row * rows,
                self.width,
                self.height
            ));
        }

        let stride = self.rgba.stride(0);
        let data = self.rgba.data_mut(0);
        for y in 0..rows {
            data[y * stride..y * stride + row].copy_from_slice(&rgba[y * row..(y + 1) * row]);
        }
        self.scaler
            .run(&self.rgba, &mut self.yuv)
            .map_err(|e| format!("Color conversion failed: {}", e))?;
        self.yuv.set_pts(Some(self.frame_index));
        self.frame_index += 1;

        self.encoder
            .send_frame(&self.yuv)
            .map_err(|e| format!("Encoding failed: {}", e))?;
        self.write_packets()
    }

    /// Flush the encoder and finalize the file.
    pub fn finish(mut self) -> Result<(), String> {
        self.encoder
            .send_eof()
            .map_err(|e| format!("Encoder flush failed: {}", e))?;
        self.write_packets()?;
        self.octx
            .write_trailer()
            .map_err(|e| format!("Failed to finalize mp4: {}", e))
    }

    fn write_packets(&mut self) -> Result<(), String> {
        let mut packet = Packet::empty();
        while self.encoder.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.stream_index);
            // The encoder leaves durations at 0 because frames carry none. The mp4
            // muxer ends the edit list at the last pts + duration, so a 0 duration
            // would cut the last frame from playback.
            if packet.duration() <= 0 {
                packet.set_duration(1);
            }
            packet.rescale_ts(self.frame_time_base, self.stream_time_base);
            packet
                .write_interleaved(&mut self.octx)
                .map_err(|e| format!("Failed to write packet: {}", e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::source::EpisodeFrames;
    use eframe::egui;

    #[test]
    fn encode_decode_round_trip() {
        let (w, h) = (64u32, 48u32);
        let path = std::env::temp_dir().join(format!("tracelr_encoder_test_{}.mp4", std::process::id()));

        let mut writer = Mp4Writer::create(&path, w, h, 30).unwrap();
        for i in 0..12 {
            let color: [u8; 4] = if i < 6 { [255, 0, 0, 255] } else { [0, 0, 255, 255] };
            let frame: Vec<u8> = color.iter().copied().cycle().take((w * h * 4) as usize).collect();
            writer.push_rgba(&frame).unwrap();
        }
        writer.finish().unwrap();

        let mut frames = EpisodeFrames::open(&path, None, None).unwrap();
        let mut decoded = Vec::new();
        while let Some(img) = frames.next().unwrap() {
            decoded.push(img);
        }
        let _ = std::fs::remove_file(&path);

        assert_eq!(decoded.len(), 12);
        for img in &decoded {
            assert_eq!(img.size, [w as usize, h as usize]);
        }
        let near = |c: egui::Color32, rgb: [u8; 3]| {
            (c.r() as i32 - rgb[0] as i32).abs() <= 20
                && (c.g() as i32 - rgb[1] as i32).abs() <= 20
                && (c.b() as i32 - rgb[2] as i32).abs() <= 20
        };
        let center = |img: &egui::ColorImage| img.pixels[(h as usize / 2) * w as usize + w as usize / 2];
        assert!(near(center(&decoded[0]), [255, 0, 0]), "frame 0 = {:?}", center(&decoded[0]));
        assert!(near(center(&decoded[11]), [0, 0, 255]), "frame 11 = {:?}", center(&decoded[11]));
    }
}
