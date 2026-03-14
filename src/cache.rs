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
    /// Synchronously decodes the center episode, spawns background loads for neighbors.
    pub fn initialize(&mut self, center_episode: usize, video_paths: &[PathBuf]) {
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
        if let Some(tex) = Self::decode_sync(&video_paths[center_episode], center_episode, &self.ctx) {
            self.slots[center_slot] = Some(tex);
        }

        // Spawn background loads for all other valid slots
        for i in 0..cache_size {
            if i == center_slot {
                continue;
            }
            let ep_index = self.first_episode + i;
            if ep_index < num_episodes {
                self.spawn_load(ep_index, &video_paths[ep_index]);
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
    ) -> Option<egui::TextureHandle> {
        let num_episodes = video_paths.len();
        let current_slot = new_episode - self.first_episode;

        if current_slot > self.cache_count {
            self.slots.pop_front();
            self.slots.push_back(None);
            self.first_episode += 1;

            let new_ep_index = self.first_episode + self.cache_size() - 1;
            if new_ep_index < num_episodes {
                self.spawn_load(new_ep_index, &video_paths[new_ep_index]);
            }
        }

        self.current_texture_for(new_episode)
    }

    /// Shift the cache window for backward navigation.
    /// Returns the TextureHandle for the new current episode, or None on cache miss.
    pub fn navigate_backward(
        &mut self,
        new_episode: usize,
        video_paths: &[PathBuf],
    ) -> Option<egui::TextureHandle> {
        let current_slot = new_episode - self.first_episode;

        if current_slot < self.cache_count && self.first_episode > 0 {
            self.slots.pop_back();
            self.slots.push_front(None);
            self.first_episode -= 1;

            self.spawn_load(self.first_episode, &video_paths[self.first_episode]);
        }

        self.current_texture_for(new_episode)
    }

    /// Rebuild cache around a new position (slider release, Home/End, click).
    pub fn jump_to(&mut self, new_episode: usize, video_paths: &[PathBuf]) {
        self.initialize(new_episode, video_paths);
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
            current.checked_sub((-delta) as usize).unwrap_or(0)
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
    fn spawn_load(&mut self, episode_index: usize, video_path: &PathBuf) {
        if self.in_flight.contains(&episode_index) {
            return;
        }
        self.in_flight.insert(episode_index);

        let path = video_path.clone();
        let tx = self.tx.clone();
        let ctx = self.ctx.clone();

        std::thread::spawn(move || {
            let result = video::decode_middle_frame_timed(&path, episode_index);
            let _ = tx.send(result);
            ctx.request_repaint();
        });
    }

    /// Synchronously decode a mid-episode frame and upload as a texture.
    fn decode_sync(
        video_path: &PathBuf,
        episode_index: usize,
        ctx: &egui::Context,
    ) -> Option<egui::TextureHandle> {
        match video::decode_middle_frame(video_path) {
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
}

// ============================================================================
// FrameCache: sliding window over frames within a single video episode
// ============================================================================

const FRAME_CACHE_COUNT: usize = 30; // ±30 frames = 61 slots (~73MB for 480×640 RGBA)

/// Sliding window cache for video frames within a single episode.
///
/// A background decoder thread decodes frames sequentially from the video,
/// sending each via channel. The main thread stores a window of ±30 frames
/// as TextureHandles. Frames outside the window are dropped.
///
/// For playback, the decoder stays ahead of display. For scrubbing/jumps,
/// the old thread is cancelled and a new one starts from the seek position.
pub(crate) struct FrameCache {
    slots: VecDeque<Option<egui::TextureHandle>>,
    first_frame: usize,
    cache_count: usize,

    result_rx: mpsc::Receiver<video::FrameDecodeResult>,
    cancel: Arc<AtomicBool>,
    _handle: Option<JoinHandle<()>>,

    pub total_frames: usize,
    pub fps: u32,
    ctx: egui::Context,
}

impl FrameCache {
    /// Start decoding a video from `center_frame`.
    /// Spawns a background thread that decodes all frames sequentially.
    pub fn new(
        ctx: &egui::Context,
        video_path: &Path,
        total_frames: usize,
        fps: u32,
        center_frame: usize,
    ) -> Self {
        let cache_count = FRAME_CACHE_COUNT;
        let cache_size = cache_count * 2 + 1;

        let max_first = total_frames.saturating_sub(cache_size);
        let first_frame = center_frame.saturating_sub(cache_count).min(max_first);

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));

        // Seek to slightly before our window start for sequential decode
        let seek_frame = if first_frame > 0 {
            Some(first_frame)
        } else {
            None
        };

        let path = video_path.to_path_buf();
        let cancel_clone = cancel.clone();
        let ctx_clone = ctx.clone();
        let handle = std::thread::spawn(move || {
            video::decode_all_frames(&path, tx, cancel_clone, ctx_clone, seek_frame);
        });

        Self {
            slots: VecDeque::from(vec![None; cache_size]),
            first_frame,
            cache_count,
            result_rx: rx,
            cancel,
            _handle: Some(handle),
            total_frames,
            fps,
            ctx: ctx.clone(),
        }
    }

    fn cache_size(&self) -> usize {
        self.cache_count * 2 + 1
    }

    /// Poll for decoded frames from the background thread.
    /// Call every frame from update().
    pub fn poll(&mut self) {
        // Collect up to 10 frames per poll to avoid blocking the UI
        for _ in 0..10 {
            match self.result_rx.try_recv() {
                Ok(result) => {
                    if let Some(slot) = self.slot_index_for(result.frame_index) {
                        if let Some(image) = result.image {
                            let name = format!("frame_{}", result.frame_index);
                            let texture = self.ctx.load_texture(
                                name,
                                image,
                                egui::TextureOptions::LINEAR,
                            );
                            self.slots[slot] = Some(texture);
                        }
                    }
                    // Frames outside window are silently dropped
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }

    /// Get the texture for a frame index, if cached.
    pub fn texture_for(&self, frame_index: usize) -> Option<egui::TextureHandle> {
        let slot = self.slot_index_for(frame_index)?;
        self.slots.get(slot).and_then(|opt| opt.clone())
    }

    /// Check if a frame is cached (for playback gating).
    pub fn is_cached(&self, frame_index: usize) -> bool {
        self.texture_for(frame_index).is_some()
    }

    /// Shift the window forward for sequential navigation.
    pub fn navigate_forward(&mut self, new_frame: usize) -> Option<egui::TextureHandle> {
        let current_slot = new_frame.checked_sub(self.first_frame)?;
        if current_slot > self.cache_count && self.first_frame + self.cache_size() < self.total_frames {
            self.slots.pop_front();
            self.slots.push_back(None);
            self.first_frame += 1;
        }
        self.texture_for(new_frame)
    }

    /// Shift the window backward for sequential navigation.
    pub fn navigate_backward(&mut self, new_frame: usize) -> Option<egui::TextureHandle> {
        let current_slot = new_frame.checked_sub(self.first_frame).unwrap_or(0);
        if current_slot < self.cache_count && self.first_frame > 0 {
            self.slots.pop_back();
            self.slots.push_front(None);
            self.first_frame -= 1;
        }
        self.texture_for(new_frame)
    }

    /// Restart decode from a new position (slider jump).
    pub fn jump_to(&mut self, center_frame: usize, video_path: &Path) {
        // Cancel old decoder
        self.cancel.store(true, Ordering::Relaxed);

        let cache_size = self.cache_size();
        let max_first = self.total_frames.saturating_sub(cache_size);
        self.first_frame = center_frame.saturating_sub(self.cache_count).min(max_first);
        self.slots.clear();
        self.slots.resize(cache_size, None);

        // Start new decoder
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        self.result_rx = rx;

        let seek_frame = if self.first_frame > 0 {
            Some(self.first_frame)
        } else {
            None
        };
        let path = video_path.to_path_buf();
        let ctx = self.ctx.clone();
        self._handle = Some(std::thread::spawn(move || {
            video::decode_all_frames(&path, tx, cancel, ctx, seek_frame);
        }));
    }

    fn slot_index_for(&self, frame_index: usize) -> Option<usize> {
        let idx = frame_index.checked_sub(self.first_frame)?;
        if idx < self.slots.len() {
            Some(idx)
        } else {
            None
        }
    }
}

impl Drop for FrameCache {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
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
