use std::time::Instant;

use eframe::egui;

use crate::app::App;
use crate::cache::{EpisodeCache, VideoPlayer};
use crate::grid::{GridDataset, GridView};
use crate::video;

impl App {
    /// Initialize the episode cache centered on current_episode.
    pub(crate) fn init_cache(&mut self, ctx: &egui::Context) {
        if self.video_paths.is_empty() {
            return;
        }
        let mut cache = EpisodeCache::new(ctx, self.cache_count);
        cache.initialize(self.current_episode, &self.video_paths, &self.seek_ranges);
        self.current_texture = cache.current_texture_for(self.current_episode);
        if self.current_texture.is_some() {
            self.perf.record_display();
        }
        self.episode_cache = Some(cache);
    }

    /// Synchronous fallback: decode and set texture directly.
    #[allow(dead_code)]
    pub(crate) fn load_current_frame_sync(&mut self, ctx: &egui::Context) {
        if let Some(path) = self.video_paths.get(self.current_episode) {
            let seek_range = self.episode_seek_range();
            let start = Instant::now();
            match video::decode_middle_frame(path, seek_range) {
                Ok(image) => {
                    let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
                    log::debug!(
                        "Sync decoded ep {} in {:.1}ms ({}x{})",
                        self.current_episode,
                        decode_ms,
                        image.size[0],
                        image.size[1],
                    );
                    let name = format!("ep_{:03}_sync", self.current_episode);
                    self.current_texture = Some(ctx.load_texture(
                        name,
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                    self.perf.record_display();
                }
                Err(e) => {
                    log::error!("Failed to decode video: {}", e);
                    self.loading_error = Some(format!("Decode error: {}", e));
                    self.current_texture = None;
                }
            }
        }
    }

    /// Navigate by ±1 using the sliding window cache.
    pub(crate) fn navigate_step(&mut self, delta: isize, ctx: &egui::Context) {
        let total = self.video_paths.len();
        if total == 0 {
            return;
        }
        let new_ep = if delta > 0 {
            (self.current_episode + delta as usize).min(total - 1)
        } else {
            self.current_episode.saturating_sub((-delta) as usize)
        };
        if new_ep == self.current_episode {
            return;
        }

        self.current_episode = new_ep;
        self.scroll_to_selected = true;

        // Show episode thumbnail immediately from cache (while video loads)
        if let Some(cache) = &mut self.episode_cache {
            let tex = if delta > 0 {
                cache.navigate_forward(new_ep, &self.video_paths, &self.seek_ranges)
            } else {
                cache.navigate_backward(new_ep, &self.video_paths, &self.seek_ranges)
            };
            if let Some(tex) = tex {
                self.current_texture = Some(tex);
                self.perf.record_display();
            }
        }

        // Start video playback for the new episode
        self.enter_video_mode(ctx);
    }

    /// Jump to an arbitrary episode (click, Home/End, slider release).
    pub(crate) fn navigate_jump(&mut self, episode: usize, ctx: &egui::Context) {
        let total = self.video_paths.len();
        if total == 0 {
            return;
        }
        let episode = episode.min(total - 1);
        if episode == self.current_episode && self.current_texture.is_some() {
            return;
        }

        self.current_episode = episode;
        self.scroll_to_selected = true;

        if let Some(cache) = &mut self.episode_cache {
            cache.jump_to(episode, &self.video_paths, &self.seek_ranges);
            if let Some(tex) = cache.current_texture_for(episode) {
                self.current_texture = Some(tex);
            }
        }

        self.enter_video_mode(ctx);
    }

    /// Navigate during slider drag — throttled sync decode with LRU cache.
    pub(crate) fn navigate_slider_drag(&mut self, episode: usize, ctx: &egui::Context) {
        let total = self.video_paths.len();
        if total == 0 {
            return;
        }
        let episode = episode.min(total - 1);
        if episode == self.current_episode && self.current_texture.is_some() {
            return;
        }

        self.current_episode = episode;

        // Check episode cache first
        if let Some(cache) = &self.episode_cache {
            if let Some(tex) = cache.current_texture_for(episode) {
                self.current_texture = Some(tex);
                self.perf.record_display();
                return;
            }
        }

        // Throttled sync decode
        if !self.slider_loader.should_load() {
            return;
        }

        // Check LRU cache
        if let Some(image) = self.decode_cache.get(episode) {
            let name = format!("ep_{:03}_lru", episode);
            self.current_texture = Some(ctx.load_texture(
                name,
                image.clone(),
                egui::TextureOptions::LINEAR,
            ));
            self.perf.record_display();
            return;
        }

        // Sync decode and insert into LRU
        if let Some(path) = self.video_paths.get(episode) {
            let seek_range = self.dataset.as_ref().and_then(|ds| {
                let vk = ds.info.video_keys.get(self.current_video_key_index)?;
                let (from, to) = ds.episode_time_range(episode, vk);
                if to > from { Some((from, to)) } else { None }
            });
            match video::decode_middle_frame(path, seek_range) {
                Ok(image) => {
                    let name = format!("ep_{:03}_slider", episode);
                    self.current_texture = Some(ctx.load_texture(
                        name,
                        image.clone(),
                        egui::TextureOptions::LINEAR,
                    ));
                    self.decode_cache.insert(episode, image);
                    self.perf.record_display();
                }
                Err(e) => {
                    log::warn!("Slider decode failed for ep {}: {}", episode, e);
                }
            }
        }
    }

    pub(crate) fn enter_video_mode(&mut self, ctx: &egui::Context) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };
        let ep = match ds.episodes.get(self.current_episode) {
            Some(ep) => ep,
            None => return,
        };
        let video_path = match self.video_paths.get(self.current_episode) {
            Some(p) => p.clone(),
            None => return,
        };

        let total_frames = ep.length;
        let fps = ds.info.fps;

        let video_key = ds.info.video_keys.get(self.current_video_key_index)
            .cloned().unwrap_or_default();
        let (from_ts, _to_ts) = ds.episode_time_range(self.current_episode, &video_key);
        let start_frame = if from_ts > 0.0 {
            (from_ts * fps as f64) as usize
        } else {
            0
        };

        log::info!(
            "Entering video mode: ep {}, {} frames, {}fps, start_frame={}",
            self.current_episode,
            total_frames,
            fps,
            start_frame
        );

        let player_total = start_frame + total_frames;
        let player = VideoPlayer::new(ctx, &video_path, player_total, fps, start_frame);
        self.player = Some(player);
        self.current_frame = start_frame;
        self.episode_start_frame = start_frame;
        self.viewing_video = true;
        self.playing = true;
        self.last_frame_time = None;
    }

    pub(crate) fn exit_video_mode(&mut self) {
        if !self.viewing_video {
            return;
        }
        self.viewing_video = false;
        self.playing = false;
        self.player = None;
        self.last_frame_time = None;
        if let Some(cache) = &self.episode_cache {
            if let Some(tex) = cache.current_texture_for(self.current_episode) {
                self.current_texture = Some(tex);
            }
        }
    }

    pub(crate) fn tick_playback(&mut self, ctx: &egui::Context) {
        if !self.playing || !self.viewing_video {
            return;
        }
        let fps = self.player.as_ref().map(|p| p.fps).unwrap_or(30);
        let frame_duration = std::time::Duration::from_secs_f64(1.0 / fps as f64);

        let now = Instant::now();
        let should_advance = match self.last_frame_time {
            Some(last) => now.duration_since(last) >= frame_duration,
            None => {
                self.last_frame_time = Some(now);
                false
            }
        };

        if !should_advance {
            ctx.request_repaint();
            return;
        }

        let total = self.player.as_ref().map(|p| p.total_frames).unwrap_or(0);
        if self.current_frame + 1 >= total {
            self.playing = false;
            return;
        }

        if let Some(player) = &mut self.player {
            if let Some(tex) = player.poll_next_frame() {
                self.current_frame = player.current_frame;
                self.current_texture = Some(tex);
                self.perf.record_display();
                self.last_frame_time = Some(now);
            }
        }

        ctx.request_repaint();
    }

    /// Toggle between single-video and grid view.
    pub(crate) fn toggle_grid_view(&mut self, ctx: &egui::Context) {
        if self.grid_view.is_some() {
            self.grid_view = None;
            self.enter_video_mode(ctx);
        } else {
            self.exit_video_mode();
            if let Some(ds) = &self.dataset {
                let gds = GridDataset {
                    video_paths: &self.video_paths,
                    seek_ranges: &self.seek_ranges,
                    episodes: &ds.episodes,
                    fps: ds.info.fps,
                };
                let grid = GridView::new(ctx, self.grid_cols, self.grid_rows, self.current_episode, &gds);
                self.grid_view = Some(grid);
            }
        }
    }

    /// Resize grid by delta steps. Grid sizes cycle through: 1x1, 2x2, 3x3, 4x4.
    pub(crate) fn grid_resize(&mut self, delta: isize, ctx: &egui::Context) {
        let current = self.grid_cols as isize;
        let new_size = (current + delta).clamp(1, 10) as usize;
        if new_size == self.grid_cols {
            return;
        }
        self.grid_cols = new_size;
        self.grid_rows = new_size;

        if let Some(ds) = &self.dataset {
            let gds = GridDataset {
                video_paths: &self.video_paths,
                seek_ranges: &self.seek_ranges,
                episodes: &ds.episodes,
                fps: ds.info.fps,
            };
            if let Some(grid) = &mut self.grid_view {
                grid.resize(new_size, new_size, ctx, &gds);
            }
        }
    }

    /// Jump the grid to start at a specific episode.
    pub(crate) fn grid_jump_to(&mut self, episode: usize, ctx: &egui::Context) {
        if let Some(ds) = &self.dataset {
            let gds = GridDataset {
                video_paths: &self.video_paths,
                seek_ranges: &self.seek_ranges,
                episodes: &ds.episodes,
                fps: ds.info.fps,
            };
            if let Some(grid) = &mut self.grid_view {
                grid.jump_to(episode, ctx, &gds);
            }
        }
        self.current_episode = episode;
        self.scroll_to_selected = true;
    }
}
