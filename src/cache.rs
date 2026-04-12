use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Instant;

use eframe::egui;

use crate::video;

const COL_LOADED: egui::Color32 = egui::Color32::from_rgb(76, 175, 80);
const COL_LOADING: egui::Color32 = egui::Color32::from_rgb(255, 183, 77);
const COL_EMPTY: egui::Color32 = egui::Color32::from_rgb(60, 60, 60);

/// Sliding window cache that preloads neighboring episodes' mid-frames
/// in background threads.
///
/// Adapted from viewskater-egui's SlidingWindowCache. The window has
/// `cache_count * 2 + 1` slots. `first_episode` tracks which episode
/// index slot 0 maps to. The current episode sits at the center slot,
/// but off-center near dataset boundaries.
pub(crate) struct EpisodeCache {
    slots: VecDeque<Option<egui::TextureHandle>>,
    first_episode: usize,
    cache_count: usize,

    tx: mpsc::Sender<video::DecodeResult>,
    rx: mpsc::Receiver<video::DecodeResult>,
    in_flight: HashSet<usize>,

    ctx: egui::Context,
}

impl EpisodeCache {
    pub fn new(ctx: &egui::Context, cache_count: usize) -> Self {
        let cache_size = cache_count * 2 + 1;
        let (tx, rx) = mpsc::channel();

        Self {
            slots: VecDeque::from(vec![None; cache_size]),
            first_episode: 0,
            cache_count,
            tx,
            rx,
            in_flight: HashSet::new(),
            ctx: ctx.clone(),
        }
    }

    fn cache_size(&self) -> usize {
        self.cache_count * 2 + 1
    }

    /// Initialize the cache centered on `center_episode`.
    /// `seek_ranges[i]` is an optional (from_s, to_s) for v3.0 concatenated videos.
    pub fn initialize(&mut self, center_episode: usize, video_paths: &[PathBuf], seek_ranges: &[Option<(f64, f64)>]) {
        let num_episodes = video_paths.len();
        if num_episodes == 0 {
            return;
        }

        // Drain any pending results from previous window
        while self.rx.try_recv().is_ok() {}
        self.in_flight.clear();

        let cache_size = self.cache_size();

        let max_first = num_episodes.saturating_sub(cache_size);
        self.first_episode = center_episode.saturating_sub(self.cache_count).min(max_first);

        self.slots.clear();
        self.slots.resize(cache_size, None);

        // Synchronously decode the center episode
        let center_slot = center_episode - self.first_episode;
        let center_seek = seek_ranges.get(center_episode).copied().flatten();
        if let Some(tex) = Self::decode_sync(&video_paths[center_episode], center_episode, center_seek, &self.ctx) {
            self.slots[center_slot] = Some(tex);
        }

        // Spawn background loads for all other valid slots
        for i in 0..cache_size {
            if i == center_slot {
                continue;
            }
            let ep_index = self.first_episode + i;
            if ep_index < num_episodes {
                let seek = seek_ranges.get(ep_index).copied().flatten();
                self.spawn_load(ep_index, &video_paths[ep_index], seek);
            }
        }
    }

    /// Poll for completed background decodes and upload textures.
    /// Call this every frame from `update()`.
    pub fn poll(&mut self) {
        while let Ok(result) = self.rx.try_recv() {
            self.in_flight.remove(&result.episode_index);

            let slot_idx = self.slot_index_for(result.episode_index);
            if let Some(slot_idx) = slot_idx {
                if let Some(color_image) = result.image {
                    let name = format!("ep_{:03}_cached", result.episode_index);
                    let texture = self.ctx.load_texture(
                        name,
                        color_image,
                        egui::TextureOptions::LINEAR,
                    );
                    self.slots[slot_idx] = Some(texture);
                }
            }
        }
    }

    /// Shift the cache window for forward navigation.
    /// Returns the TextureHandle for the new current episode, or None on cache miss.
    pub fn navigate_forward(
        &mut self,
        new_episode: usize,
        video_paths: &[PathBuf],
        seek_ranges: &[Option<(f64, f64)>],
    ) -> Option<egui::TextureHandle> {
        let num_episodes = video_paths.len();
        let current_slot = new_episode - self.first_episode;

        if current_slot > self.cache_count {
            self.slots.pop_front();
            self.slots.push_back(None);
            self.first_episode += 1;

            let new_ep_index = self.first_episode + self.cache_size() - 1;
            if new_ep_index < num_episodes {
                let seek = seek_ranges.get(new_ep_index).copied().flatten();
                self.spawn_load(new_ep_index, &video_paths[new_ep_index], seek);
            }
        }

        self.current_texture_for(new_episode)
    }

    pub fn navigate_backward(
        &mut self,
        new_episode: usize,
        video_paths: &[PathBuf],
        seek_ranges: &[Option<(f64, f64)>],
    ) -> Option<egui::TextureHandle> {
        let current_slot = new_episode - self.first_episode;

        if current_slot < self.cache_count && self.first_episode > 0 {
            self.slots.pop_back();
            self.slots.push_front(None);
            self.first_episode -= 1;

            let seek = seek_ranges.get(self.first_episode).copied().flatten();
            self.spawn_load(self.first_episode, &video_paths[self.first_episode], seek);
        }

        self.current_texture_for(new_episode)
    }

    pub fn jump_to(&mut self, new_episode: usize, video_paths: &[PathBuf], seek_ranges: &[Option<(f64, f64)>]) {
        self.initialize(new_episode, video_paths, seek_ranges);
    }

    /// Get the TextureHandle for a given episode index, if cached.
    pub fn current_texture_for(&self, episode_index: usize) -> Option<egui::TextureHandle> {
        let slot_idx = episode_index.checked_sub(self.first_episode)?;
        self.slots.get(slot_idx).and_then(|opt| opt.clone())
    }

    /// Check if the next episode in a given direction is cached.
    pub fn is_next_cached(&self, current: usize, delta: isize) -> bool {
        let next = if delta > 0 {
            current + delta as usize
        } else {
            current.saturating_sub((-delta) as usize)
        };
        self.current_texture_for(next).is_some()
    }

    fn slot_index_for(&self, episode_index: usize) -> Option<usize> {
        if episode_index < self.first_episode {
            return None;
        }
        let idx = episode_index - self.first_episode;
        if idx < self.slots.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// Spawn a background thread to decode a mid-episode frame.
    fn spawn_load(&mut self, episode_index: usize, video_path: &Path, seek_range: Option<(f64, f64)>) {
        if self.in_flight.contains(&episode_index) {
            return;
        }
        self.in_flight.insert(episode_index);

        let path = video_path.to_path_buf();
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();

        std::thread::spawn(move || {
            let result = video::decode_middle_frame_timed(&path, episode_index, seek_range);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    /// Synchronously decode a mid-episode frame and upload as a texture.
    fn decode_sync(
        video_path: &Path,
        episode_index: usize,
        seek_range: Option<(f64, f64)>,
        ctx: &egui::Context,
    ) -> Option<egui::TextureHandle> {
        match video::decode_middle_frame(video_path, seek_range) {
            Ok(image) => {
                let name = format!("ep_{:03}_sync", episode_index);
                Some(ctx.load_texture(name, image, egui::TextureOptions::LINEAR))
            }
            Err(e) => {
                log::error!("Failed to decode ep {}: {}", episode_index, e);
                None
            }
        }
    }

    /// Draw debug overlay visualizing cache slot states.
    pub fn show_debug_overlay(
        &self,
        ctx: &egui::Context,
        current_episode: usize,
        num_episodes: usize,
    ) {
        let cache_size = self.cache_size();

        egui::Window::new("cache_state")
            .title_bar(false)
            .resizable(false)
            .auto_sized()
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 28.0])
            .interactable(false)
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_black_alpha(200))
                    .corner_radius(6.0)
                    .inner_margin(10.0),
            )
            .show(ctx, |ui| {
                let last_ep = self.first_episode + cache_size - 1;
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!(
                            "Cache [{}\u{2013}{}]",
                            self.first_episode,
                            last_ep.min(num_episodes.saturating_sub(1))
                        ))
                        .monospace()
                        .color(egui::Color32::from_gray(200))
                        .size(12.0),
                    )
                    .wrap_mode(egui::TextWrapMode::Extend),
                );

                ui.add_space(4.0);

                let cell_w: f32 = 28.0;
                let cell_h: f32 = 20.0;
                let gap: f32 = 2.0;
                let label_h: f32 = 12.0;
                let total_w = cache_size as f32 * (cell_w + gap) - gap;
                let total_h = cell_h + gap + label_h;

                let (area, _) =
                    ui.allocate_exact_size(egui::vec2(total_w, total_h), egui::Sense::hover());

                let painter = ui.painter();

                for i in 0..cache_size {
                    let ep_index = self.first_episode + i;
                    let is_current = ep_index == current_episode;
                    let is_loaded = self.slots.get(i).is_some_and(|s| s.is_some());
                    let is_in_flight = self.in_flight.contains(&ep_index);
                    let is_valid = ep_index < num_episodes;

                    let x = area.min.x + i as f32 * (cell_w + gap);
                    let cell_rect = egui::Rect::from_min_size(
                        egui::pos2(x, area.min.y),
                        egui::vec2(cell_w, cell_h),
                    );

                    let fill = if !is_valid {
                        egui::Color32::from_gray(25)
                    } else if is_loaded {
                        COL_LOADED
                    } else if is_in_flight {
                        COL_LOADING
                    } else {
                        COL_EMPTY
                    };

                    painter.rect_filled(cell_rect, 3.0, fill);

                    if is_current {
                        painter.rect_stroke(
                            cell_rect,
                            3.0,
                            egui::Stroke::new(2.0, egui::Color32::WHITE),
                            egui::epaint::StrokeKind::Outside,
                        );
                    }

                    if is_valid {
                        painter.text(
                            egui::pos2(x + cell_w / 2.0, area.min.y + cell_h + gap),
                            egui::Align2::CENTER_TOP,
                            ep_index.to_string(),
                            egui::FontId::monospace(9.0),
                            if is_current {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_gray(120)
                            },
                        );
                    }
                }

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    legend_swatch(ui, COL_LOADED, "Loaded");
                    ui.add_space(4.0);
                    legend_swatch(ui, COL_LOADING, "Loading");
                    ui.add_space(4.0);
                    legend_swatch(ui, COL_EMPTY, "Empty");
                });
            });
    }
}

/// Throttled synchronous loader for slider scrubbing.
/// Limits how often we block the UI thread with sync decodes during fast drags.
pub(crate) struct SliderLoader {
    last_load: Instant,
}

const SLIDER_THROTTLE_MS: u128 = 10;

impl SliderLoader {
    pub fn new() -> Self {
        Self {
            last_load: Instant::now(),
        }
    }

    /// Returns true if enough time has passed since the last decode.
    pub fn should_load(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now
            .checked_duration_since(self.last_load)
            .map(|d| d.as_millis())
            .unwrap_or(SLIDER_THROTTLE_MS);

        if elapsed >= SLIDER_THROTTLE_MS {
            self.last_load = now;
            true
        } else {
            false
        }
    }
}

/// LRU cache of decoded images, keyed by episode index.
///
/// Stores decoded `ColorImage` in CPU memory so that revisiting an episode
/// during slider scrubbing skips the ~6ms decode+seek.
/// Memory budget: 480×640 RGBA = ~1.2MB per frame. Capacity 50 = ~60MB.
pub(crate) struct DecodeLruCache {
    entries: HashMap<usize, egui::ColorImage>,
    order: VecDeque<usize>,
    capacity: usize,
}

impl DecodeLruCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    /// Get a decoded image if cached. Moves entry to most-recently-used.
    pub fn get(&mut self, episode_index: usize) -> Option<&egui::ColorImage> {
        if self.entries.contains_key(&episode_index) {
            self.order.retain(|&i| i != episode_index);
            self.order.push_back(episode_index);
            self.entries.get(&episode_index)
        } else {
            None
        }
    }

    /// Insert a decoded image. Evicts the least recently used if at capacity.
    pub fn insert(&mut self, episode_index: usize, image: egui::ColorImage) {
        if self.entries.contains_key(&episode_index) {
            self.order.retain(|&i| i != episode_index);
        } else if self.entries.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(episode_index, image);
        self.order.push_back(episode_index);
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}

// ============================================================================
// VideoPlayer: persistent decoder thread for frame-level video playback
// ============================================================================

/// Lookahead buffer size for the bounded channel between decoder and UI.
/// 30 frames × ~1.2MB = ~36MB. The decoder thread blocks when full,
/// naturally pacing itself to stay ~1 second ahead of playback.
const PLAYER_BUFFER_SIZE: usize = 30;

/// Video player backed by a background decoder thread.
///
/// The decoder thread opens the video file, optionally seeks, then decodes
/// frames sequentially — sending each via a bounded `sync_channel`. The
/// main thread consumes one frame per 1/fps during playback. On seek,
/// the old thread is cancelled and a new one starts from the seek position.
///
/// Memory: bounded to ~36MB (30 frames in the channel) plus 1 TextureHandle
/// on the GPU for the currently displayed frame.
pub(crate) struct VideoPlayer {
    frame_rx: mpsc::Receiver<video::FrameDecodeResult>,
    cancel: Arc<AtomicBool>,
    _handle: Option<JoinHandle<()>>,

    pub current_frame: usize,
    pub total_frames: usize,
    pub fps: u32,

    video_path: PathBuf,
    ctx: egui::Context,

    // Persistent scrub worker (created lazily on first scrub)
    scrub_tx: Option<mpsc::Sender<usize>>,
    scrub_result_rx: Option<mpsc::Receiver<Option<egui::ColorImage>>>,
}

impl VideoPlayer {
    /// Create a player and start decoding from `start_frame`.
    pub fn new(
        ctx: &egui::Context,
        video_path: &Path,
        total_frames: usize,
        fps: u32,
        start_frame: usize,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel(PLAYER_BUFFER_SIZE);
        let cancel = Arc::new(AtomicBool::new(false));

        let seek_frame = if start_frame > 0 { Some(start_frame) } else { None };
        let path = video_path.to_path_buf();
        let cancel_clone = cancel.clone();
        let ctx_clone = ctx.clone();
        let handle = std::thread::spawn(move || {
            video::decode_all_frames_sync(&path, tx, cancel_clone, ctx_clone, seek_frame);
        });

        Self {
            frame_rx: rx,
            cancel,
            _handle: Some(handle),
            current_frame: start_frame,
            total_frames,
            fps,
            video_path: video_path.to_path_buf(),
            ctx: ctx.clone(),
            scrub_tx: None,
            scrub_result_rx: None,
        }
    }

    /// Try to receive the next decoded frame. Returns a TextureHandle if
    /// a frame is ready. Call this at playback rate (every 1/fps seconds)
    /// or once after seeking to get the frame at the seek position.
    pub fn poll_next_frame(&mut self) -> Option<egui::TextureHandle> {
        match self.frame_rx.try_recv() {
            Ok(result) => {
                self.current_frame = result.frame_index;
                result.image.map(|img| {
                    self.ctx.load_texture(
                        format!("vframe_{}", result.frame_index),
                        img,
                        egui::TextureOptions::LINEAR,
                    )
                })
            }
            Err(_) => None,
        }
    }

    /// Blocking drain: discard frames from the channel until `frame_index >= target`,
    /// then return that frame's texture. Used when we need an exact frame after seek
    /// (e.g., camera switch) without showing intermediate keyframe frames.
    /// Bounded by `timeout_ms` total wait time.
    pub fn drain_to_frame(&mut self, target: usize, timeout_ms: u64) -> Option<egui::TextureHandle> {
        let deadline = Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            match self.frame_rx.recv_timeout(remaining) {
                Ok(result) => {
                    if result.frame_index < target {
                        continue; // discard intermediate frame
                    }
                    self.current_frame = result.frame_index;
                    return result.image.map(|img| {
                        self.ctx.load_texture(
                            format!("vframe_{}", result.frame_index),
                            img,
                            egui::TextureOptions::LINEAR,
                        )
                    });
                }
                Err(_) => return None,
            }
        }
    }

    /// Seek to a new position. Cancels the current decoder thread and
    /// starts a new one from the seek position.
    pub fn seek(&mut self, frame: usize) {
        self.cancel.store(true, Ordering::Relaxed);

        let (tx, rx) = mpsc::sync_channel(PLAYER_BUFFER_SIZE);
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        self.frame_rx = rx;
        self.current_frame = frame;

        let seek_frame = if frame > 0 { Some(frame) } else { None };
        let path = self.video_path.clone();
        let ctx = self.ctx.clone();
        self._handle = Some(std::thread::spawn(move || {
            video::decode_all_frames_sync(&path, tx, cancel, ctx, seek_frame);
        }));
    }

    /// Send a scrub request to the persistent worker. The worker decodes
    /// forward from its current position (fast for forward scrubs) or seeks
    /// if needed. Created lazily on first call.
    pub fn scrub_to(&mut self, frame: usize) {
        // Drain stale results so poll only sees the latest
        if let Some(rx) = &self.scrub_result_rx {
            while rx.try_recv().is_ok() {}
        }

        // Lazily spawn the persistent scrub worker
        if self.scrub_tx.is_none() {
            let (req_tx, req_rx) = mpsc::channel();
            let (res_tx, res_rx) = mpsc::channel();
            self.scrub_tx = Some(req_tx);
            self.scrub_result_rx = Some(res_rx);

            let path = self.video_path.clone();
            let fps = self.fps;
            let ctx = self.ctx.clone();
            std::thread::spawn(move || {
                video::scrub_worker(path, fps, req_rx, res_tx, ctx);
            });
        }

        if let Some(tx) = &self.scrub_tx {
            let _ = tx.send(frame);
        }
    }

    /// Poll for a completed scrub frame from the persistent worker.
    pub fn poll_scrub_frame(&mut self) -> Option<egui::ColorImage> {
        let rx = self.scrub_result_rx.as_ref()?;
        rx.try_recv().ok().flatten()
    }

    /// Drain pending scrub results. The worker stays alive for reuse.
    pub fn cancel_scrub(&mut self) {
        if let Some(rx) = &self.scrub_result_rx {
            while rx.try_recv().is_ok() {}
        }
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        // Dropping scrub_tx closes the channel, signaling the worker to exit
        self.scrub_tx = None;
    }
}

fn legend_swatch(ui: &mut egui::Ui, color: egui::Color32, label: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.label(
        egui::RichText::new(label)
            .color(egui::Color32::from_gray(160))
            .size(10.0),
    );
}
