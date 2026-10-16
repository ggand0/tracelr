use std::path::Path;

use eframe::egui;

/// Sequential reader for the frames of one episode in one camera's video.
///
/// For v3.0 datasets several episodes share one mp4, so `seek_range` gives the
/// episode's (from_seconds, to_seconds) inside the file: the reader seeks to the
/// keyframe before `from`, drops frames decoded before it, and stops at `to`.
pub(crate) struct EpisodeFrames {
    ictx: ffmpeg_next::format::context::Input,
    decoder: ffmpeg_next::decoder::Video,
    stream_index: usize,
    time_base: ffmpeg_next::Rational,
    fps: f64,
    /// First frame index (in the file) that belongs to the episode.
    start_frame: usize,
    /// First frame index past the episode, None when the file ends the episode.
    end_frame: Option<usize>,
    /// Safety bound on the number of frames returned.
    max_frames: Option<usize>,
    returned: usize,
    /// Latest frame inside the episode (valid once `returned > 0`).
    current: ffmpeg_next::frame::Video,
    /// Receive buffer; swapped into `current` when it holds an episode frame.
    scratch: ffmpeg_next::frame::Video,
    eof_sent: bool,
    done: bool,
}

impl EpisodeFrames {
    pub fn open(
        path: &Path,
        seek_range: Option<(f64, f64)>,
        max_frames: Option<usize>,
    ) -> Result<Self, String> {
        crate::video::ensure_ffmpeg_init();

        let mut ictx = ffmpeg_next::format::input(path)
            .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;

        let (stream_index, params, time_base, fps) = {
            let stream = ictx
                .streams()
                .best(ffmpeg_next::media::Type::Video)
                .ok_or_else(|| format!("No video stream in {}", path.display()))?;
            let rate = stream.rate();
            let fps = if rate.1 > 0 {
                rate.0 as f64 / rate.1 as f64
            } else {
                30.0
            };
            (stream.index(), stream.parameters(), stream.time_base(), fps)
        };

        let decoder = ffmpeg_next::codec::context::Context::from_parameters(params)
            .and_then(|c| c.decoder().video())
            .map_err(|e| format!("Failed to create decoder: {}", e))?;

        let (start_frame, end_frame) = match seek_range {
            Some((from_s, to_s)) => {
                // ffmpeg seeks to the nearest keyframe at or before the target.
                let target_us = (from_s * 1_000_000.0) as i64;
                if target_us > 0 {
                    let _ = ictx.seek(target_us, ..target_us);
                }
                let start = (from_s * fps).round() as usize;
                let end = (to_s * fps).round() as usize;
                (start, (end > start).then_some(end))
            }
            None => (0, None),
        };

        Ok(Self {
            ictx,
            decoder,
            stream_index,
            time_base,
            fps,
            start_frame,
            end_frame,
            max_frames,
            returned: 0,
            current: ffmpeg_next::frame::Video::empty(),
            scratch: ffmpeg_next::frame::Video::empty(),
            eof_sent: false,
            done: false,
        })
    }

    /// Frame size (width, height) of the video stream.
    pub fn size(&self) -> (usize, usize) {
        (self.decoder.width() as usize, self.decoder.height() as usize)
    }

    /// Next frame of the episode, or None at the end of the episode or file.
    pub fn next(&mut self) -> Result<Option<egui::ColorImage>, String> {
        if !self.advance() {
            return Ok(None);
        }
        self.convert_current().map(Some)
    }

    /// Like `next`, but at the end of a short stream returns its last frame again.
    pub fn next_or_last(&mut self) -> Result<Option<egui::ColorImage>, String> {
        if self.advance() || self.returned > 0 {
            self.convert_current().map(Some)
        } else {
            Ok(None)
        }
    }

    fn convert_current(&self) -> Result<egui::ColorImage, String> {
        crate::video::frame_to_color_image(&self.current)
            .map_err(|e| format!("Failed to convert frame: {}", e))
    }

    /// Skip up to `n` frames without converting them to RGBA. Returns how many were skipped.
    pub fn skip(&mut self, n: usize) -> usize {
        let mut skipped = 0;
        while skipped < n && self.advance() {
            skipped += 1;
        }
        skipped
    }

    /// Decode the next episode frame into `self.current`. Returns false at the end.
    fn advance(&mut self) -> bool {
        if self.max_frames.is_some_and(|m| self.returned >= m) {
            self.done = true;
        }
        while !self.done {
            if self.receive_raw() {
                self.returned += 1;
                return true;
            }
            if self.done {
                break;
            }
            if self.eof_sent {
                self.done = true;
                break;
            }
            self.feed();
        }
        false
    }

    /// Receive frames from the decoder until one inside the episode range arrives,
    /// and move it into `self.current`. Sets `done` when a frame past the episode
    /// end shows up.
    fn receive_raw(&mut self) -> bool {
        while self.decoder.receive_frame(&mut self.scratch).is_ok() {
            if let Some(idx) = self.frame_index() {
                if idx < self.start_frame {
                    continue;
                }
                if self.end_frame.is_some_and(|end| idx >= end) {
                    self.done = true;
                    return false;
                }
            }
            std::mem::swap(&mut self.current, &mut self.scratch);
            return true;
        }
        false
    }

    /// Send the next video packet to the decoder, or EOF when the file is exhausted.
    fn feed(&mut self) {
        loop {
            match self.ictx.packets().next() {
                Some((stream, packet)) => {
                    if stream.index() != self.stream_index {
                        continue;
                    }
                    if let Err(e) = self.decoder.send_packet(&packet) {
                        log::debug!("send_packet failed: {}", e);
                    }
                    return;
                }
                None => {
                    let _ = self.decoder.send_eof();
                    self.eof_sent = true;
                    return;
                }
            }
        }
    }

    fn frame_index(&self) -> Option<usize> {
        self.scratch.pts().map(|pts| {
            let time_s = pts as f64 * self.time_base.0 as f64 / self.time_base.1 as f64;
            (time_s * self.fps).round().max(0.0) as usize
        })
    }
}
