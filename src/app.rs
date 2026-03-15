use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::annotation::AnnotationState;
use crate::cache::{DecodeLruCache, EpisodeCache, SliderLoader, VideoPlayer};
use crate::dataset::{self, LeRobotDataset};
use crate::perf::PerfTracker;
use crate::theme::UiTheme;
use crate::video;

const CACHE_COUNT: usize = 5; // ±5 episodes = 11 total slots
const LRU_CAPACITY: usize = 50;

pub struct App {
    // Data
    dataset: Option<LeRobotDataset>,
    annotations: AnnotationState,

    // Viewer state
    current_episode: usize,
    current_video_key_index: usize,
    current_texture: Option<egui::TextureHandle>,
    video_paths: Vec<PathBuf>,
    loading_error: Option<String>,

    // Episode cache
    episode_cache: Option<EpisodeCache>,
    slider_loader: SliderLoader,
    decode_cache: DecodeLruCache,
    slider_dragging: bool,

    // Video playback
    viewing_video: bool,
    player: Option<VideoPlayer>,
    current_frame: usize,
    playing: bool,
    last_frame_time: Option<Instant>,
    frame_slider_dragging: bool,

    // UI
    theme: UiTheme,
    perf: PerfTracker,
    initial_size_set: bool,
    show_cache_overlay: bool,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext, initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            dataset: None,
            annotations: AnnotationState::load_prompts(initial_path.as_deref()),
            current_episode: 0,
            current_video_key_index: 0,
            current_texture: None,
            video_paths: Vec::new(),
            loading_error: None,
            episode_cache: None,
            slider_loader: SliderLoader::new(),
            decode_cache: DecodeLruCache::new(LRU_CAPACITY),
            slider_dragging: false,
            viewing_video: false,
            player: None,
            current_frame: 0,
            playing: false,
            last_frame_time: None,
            frame_slider_dragging: false,
            theme: UiTheme::teal_dark(),
            perf: PerfTracker::new(),
            initial_size_set: false,
            show_cache_overlay: false,
        };

        if let Some(path) = initial_path {
            app.load_dataset(&path);
            if app.dataset.is_some() {
                app.init_cache(&_cc.egui_ctx);
                app.enter_video_mode(&_cc.egui_ctx);
            }
        }

        app
    }

    fn load_dataset(&mut self, path: &std::path::Path) {
        match LeRobotDataset::load(path) {
            Ok(ds) => {
                log::info!("Dataset loaded: {} episodes", ds.episodes.len());
                let wrist_idx = ds
                    .info
                    .video_keys
                    .iter()
                    .position(|k| k.contains("wrist"))
                    .unwrap_or(0);
                self.current_video_key_index = wrist_idx;

                // Build video paths for all episodes
                let video_key = ds.info.video_keys.get(wrist_idx).cloned().unwrap_or_default();
                self.video_paths = ds
                    .episodes
                    .iter()
                    .map(|ep| ds.video_path(ep.episode_index, &video_key))
                    .collect();

                self.dataset = Some(ds);
                self.current_episode = 0;
                self.current_texture = None;
                self.loading_error = None;

                // Load prompts config for this dataset
                self.annotations = AnnotationState::load_prompts(Some(path));

                // Try to load saved annotations
                let annot_path = path.join("annotations.json");
                if annot_path.exists() {
                    if let Err(e) = self.annotations.load_json(&annot_path) {
                        log::warn!("Failed to load annotations: {}", e);
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to load dataset: {}", e);
                self.loading_error = Some(e);
            }
        }
    }

    /// Initialize the episode cache centered on current_episode.
    fn init_cache(&mut self, ctx: &egui::Context) {
        if self.video_paths.is_empty() {
            return;
        }
        let mut cache = EpisodeCache::new(ctx, CACHE_COUNT);
        cache.initialize(self.current_episode, &self.video_paths);
        // Set the current texture from the cache (center was decoded synchronously)
        self.current_texture = cache.current_texture_for(self.current_episode);
        if self.current_texture.is_some() {
            self.perf.record_display();
        }
        self.episode_cache = Some(cache);
    }

    /// Synchronous fallback: decode and set texture directly.
    #[allow(dead_code)]
    fn load_current_frame_sync(&mut self, ctx: &egui::Context) {
        if let Some(path) = self.video_paths.get(self.current_episode) {
            let seek_range = self.episode_seek_range();
            let start = std::time::Instant::now();
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
    fn navigate_step(&mut self, delta: isize, ctx: &egui::Context) {
        let total = self.video_paths.len();
        if total == 0 {
            return;
        }
        let new_ep = if delta > 0 {
            (self.current_episode + delta as usize).min(total - 1)
        } else {
            self.current_episode
                .checked_sub((-delta) as usize)
                .unwrap_or(0)
        };
        if new_ep == self.current_episode {
            return;
        }

        self.current_episode = new_ep;

        // Show episode thumbnail immediately from cache (while video loads)
        if let Some(cache) = &mut self.episode_cache {
            let tex = if delta > 0 {
                cache.navigate_forward(new_ep, &self.video_paths)
            } else {
                cache.navigate_backward(new_ep, &self.video_paths)
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
    /// Reinitializes the cache window around the new position.
    fn navigate_jump(&mut self, episode: usize, ctx: &egui::Context) {
        let total = self.video_paths.len();
        if total == 0 {
            return;
        }
        let episode = episode.min(total - 1);
        if episode == self.current_episode && self.current_texture.is_some() {
            return;
        }

        self.current_episode = episode;

        // Show episode thumbnail from cache while video loads
        if let Some(cache) = &mut self.episode_cache {
            cache.jump_to(episode, &self.video_paths);
            if let Some(tex) = cache.current_texture_for(episode) {
                self.current_texture = Some(tex);
            }
        }

        // Start video playback
        self.enter_video_mode(ctx);
    }

    /// Navigate during slider drag — throttled sync decode with LRU cache.
    fn navigate_slider_drag(&mut self, episode: usize, ctx: &egui::Context) {
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

    // -- Video playback --

    fn enter_video_mode(&mut self, ctx: &egui::Context) {
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

        log::info!(
            "Entering video mode: ep {}, {} frames, {}fps",
            self.current_episode,
            total_frames,
            fps
        );

        let player = VideoPlayer::new(ctx, &video_path, total_frames, fps, 0);
        self.player = Some(player);
        self.current_frame = 0;
        self.viewing_video = true;
        self.playing = true;
        self.last_frame_time = None;
    }

    fn exit_video_mode(&mut self) {
        if !self.viewing_video {
            return;
        }
        self.viewing_video = false;
        self.playing = false;
        self.player = None;
        self.last_frame_time = None;
        // Restore episode thumbnail
        if let Some(cache) = &self.episode_cache {
            if let Some(tex) = cache.current_texture_for(self.current_episode) {
                self.current_texture = Some(tex);
            }
        }
    }

    fn tick_playback(&mut self, ctx: &egui::Context) {
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

        // Pull next decoded frame from the player channel
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

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        if self.dataset.is_none() {
            return;
        }

        // Global keys (both modes)
        let mut enter_pressed = false;
        let mut escape_pressed = false;
        let mut space_pressed = false;

        ctx.input(|i| {
            enter_pressed = i.key_pressed(egui::Key::Enter);
            escape_pressed = i.key_pressed(egui::Key::Escape);
            space_pressed = i.key_pressed(egui::Key::Space);

            // Annotation shortcuts (1-9) — work in both modes
            for (key, idx) in [
                (egui::Key::Num1, 0),
                (egui::Key::Num2, 1),
                (egui::Key::Num3, 2),
                (egui::Key::Num4, 3),
                (egui::Key::Num5, 4),
                (egui::Key::Num6, 5),
                (egui::Key::Num7, 6),
                (egui::Key::Num8, 7),
                (egui::Key::Num9, 8),
            ] {
                if i.key_pressed(key) && idx < self.annotations.prompts.len() {
                    self.annotations.set(self.current_episode, idx);
                }
            }

            if i.modifiers.command && i.key_pressed(egui::Key::S) {
                self.save_annotations();
            }
        });

        // Escape: pause and exit to thumbnail view
        if escape_pressed && self.viewing_video {
            self.exit_video_mode();
            return;
        }
        // Enter: re-enter video mode if exited
        if enter_pressed && !self.viewing_video {
            self.enter_video_mode(ctx);
            return;
        }

        if self.viewing_video {
            self.handle_keyboard_video(ctx, space_pressed);
        } else {
            self.handle_keyboard_episode(ctx);
        }
    }

    fn handle_keyboard_video(&mut self, ctx: &egui::Context, space_pressed: bool) {
        if space_pressed {
            self.playing = !self.playing;
            if self.playing {
                self.last_frame_time = Some(Instant::now());
            } else {
                self.last_frame_time = None;
            }
        }

        let total_frames = self.player.as_ref().map(|p| p.total_frames).unwrap_or(0);
        let mut frame_step: Option<isize> = None;
        let mut frame_jump: Option<usize> = None;

        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                frame_step = Some(1);
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                frame_step = Some(-1);
            }
            if i.key_pressed(egui::Key::Home) {
                frame_jump = Some(0);
            }
            if i.key_pressed(egui::Key::End) {
                frame_jump = Some(total_frames.saturating_sub(1));
            }
        });

        if let Some(delta) = frame_step {
            if delta > 0 {
                // Forward: pull next frame from decoder
                if let Some(player) = &mut self.player {
                    if let Some(tex) = player.poll_next_frame() {
                        self.current_frame = player.current_frame;
                        self.current_texture = Some(tex);
                        self.perf.record_display();
                    }
                }
            } else {
                // Backward: seek to previous frame
                let new_frame = self.current_frame.saturating_sub(1);
                if new_frame != self.current_frame {
                    self.current_frame = new_frame;
                    if let Some(player) = &mut self.player {
                        player.seek(new_frame);
                    }
                }
            }
        } else if let Some(frame) = frame_jump {
            self.current_frame = frame;
            if let Some(player) = &mut self.player {
                player.seek(frame);
            }
        }
    }

    fn handle_keyboard_episode(&mut self, ctx: &egui::Context) {
        let total = self.video_paths.len();

        let mut step: Option<isize> = None;
        let mut jump: Option<usize> = None;

        let next_cached = self
            .episode_cache
            .as_ref()
            .map(|c| c.is_next_cached(self.current_episode, 1))
            .unwrap_or(false);
        let prev_cached = self
            .episode_cache
            .as_ref()
            .map(|c| c.is_next_cached(self.current_episode, -1))
            .unwrap_or(false);

        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                step = Some(1);
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                step = Some(-1);
            }
            if i.modifiers.shift {
                if (i.key_down(egui::Key::ArrowRight) || i.key_down(egui::Key::D)) && next_cached {
                    step = Some(1);
                }
                if (i.key_down(egui::Key::ArrowLeft) || i.key_down(egui::Key::A)) && prev_cached {
                    step = Some(-1);
                }
            }
            if i.key_pressed(egui::Key::Home) {
                jump = Some(0);
            }
            if i.key_pressed(egui::Key::End) {
                jump = Some(total.saturating_sub(1));
            }
        });

        if let Some(delta) = step {
            self.navigate_step(delta, ctx);
        } else if let Some(ep) = jump {
            self.navigate_jump(ep, ctx);
        }
    }

    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });

        if let Some(path) = dropped.first() {
            if dataset::is_lerobot_dataset(path) {
                self.load_dataset(path);
                if self.dataset.is_some() {
                    self.init_cache(ctx);
                }
            } else {
                self.loading_error = Some(format!(
                    "Not a LeRobot dataset: {}\nExpected meta/info.json",
                    path.display()
                ));
            }
        }
    }

    fn save_annotations(&mut self) {
        if let Some(ds) = &self.dataset {
            let path = ds.root.join("annotations.json");
            match self
                .annotations
                .save_json(&path, &ds.root.to_string_lossy())
            {
                Ok(()) => {
                    self.annotations.dirty = false;
                    log::info!("Annotations saved to {}", path.display());
                }
                Err(e) => {
                    log::error!("Failed to save annotations: {}", e);
                }
            }
        }
    }

    fn update_title(&self, ctx: &egui::Context) {
        let title = if let Some(ds) = &self.dataset {
            let dir_name = ds
                .root
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let (done, total) = self.annotations.progress(ds.episodes.len());
            format!("lerobot-explorer - {} [{}/{}]", dir_name, done, total)
        } else {
            "lerobot-explorer".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
    }

    fn episode_seek_range(&self) -> Option<(f64, f64)> {
        let ds = self.dataset.as_ref()?;
        let vk = ds.info.video_keys.get(self.current_video_key_index)?;
        let (from, to) = ds.episode_time_range(self.current_episode, vk);
        if to > from { Some((from, to)) } else { None }
    }

    fn annotation_json_path(&self) -> Option<String> {
        self.dataset
            .as_ref()
            .map(|ds| ds.root.join("annotations.json").to_string_lossy().to_string())
    }

    // -- UI panels --

    fn show_menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Dataset...").clicked() {
                        ui.close_menu();
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.load_dataset(&path);
                            if self.dataset.is_some() {
                                self.init_cache(ctx);
                            }
                        }
                    }
                    if ui.button("Save Annotations  Ctrl+S").clicked() {
                        ui.close_menu();
                        self.save_annotations();
                    }
                    if ui
                        .add_enabled(
                            self.dataset.is_some(),
                            egui::Button::new("Export to LeRobot..."),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        if let Some(ds) = &self.dataset {
                            match self.annotations.export_lerobot(ds) {
                                Ok(()) => log::info!("Exported to LeRobot format"),
                                Err(e) => log::error!("Export failed: {}", e),
                            }
                        }
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    if ui
                        .checkbox(&mut self.show_cache_overlay, "Cache Overlay")
                        .changed()
                    {
                        ui.close_menu();
                    }
                });

                // FPS display (right-aligned)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(self.perf.fps_text())
                            .monospace()
                            .color(self.theme.muted)
                            .size(11.0),
                    );
                });
            });
        });
    }

    fn show_episode_list(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => {
                ui.label("No dataset loaded");
                return;
            }
        };

        let (done, total) = self.annotations.progress(ds.episodes.len());
        ui.label(
            egui::RichText::new(format!("Episodes ({}/{})", done, total))
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        let mut navigate_to = None;

        let episode_annotations: Vec<(usize, Option<(String, egui::Color32)>)> = ds
            .episodes
            .iter()
            .map(|ep| {
                let annot = self.annotations.get(ep.episode_index).and_then(|idx| {
                    self.annotations
                        .prompts
                        .get(idx)
                        .map(|p| (p.label.clone(), p.color))
                });
                (ep.episode_index, annot)
            })
            .collect();

        egui::ScrollArea::vertical().show(ui, |ui| {
            for (episode_index, annot_info) in &episode_annotations {
                let is_selected = *episode_index == self.current_episode;

                let response = ui.horizontal(|ui| {
                    if let Some((_, color)) = annot_info {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 4.0, *color);
                    } else {
                        ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    }

                    let mut label_text = format!("ep {:03}", episode_index);
                    if let Some((label, _)) = annot_info {
                        label_text = format!("{} - {}", label_text, label);
                    }

                    ui.selectable_label(is_selected, &label_text)
                });

                if response.inner.clicked() {
                    navigate_to = Some(*episode_index);
                }
            }
        });

        if let Some(ep) = navigate_to {
            if self.viewing_video {
                self.exit_video_mode();
            }
            self.navigate_jump(ep, ctx);
        }
    }

    fn show_info_panel(&mut self, ui: &mut egui::Ui) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };

        ui.label(
            egui::RichText::new("Episode Info")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        if let Some(ep) = ds.episodes.get(self.current_episode) {
            let info_rows: Vec<(&str, String)> = vec![
                (
                    "Episode:",
                    format!("{} / {}", ep.episode_index, ds.episodes.len()),
                ),
                ("Frames:", format!("{}", ep.length)),
                (
                    "Duration:",
                    format!("{:.1}s", ep.length as f64 / ds.info.fps as f64),
                ),
                ("FPS:", format!("{}", ds.info.fps)),
            ];

            for (label, value) in &info_rows {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(*label).color(self.theme.muted));
                    ui.label(value);
                });
            }

            if ds.info.video_keys.len() > 1 {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Camera:").color(self.theme.muted));
                    let current_key = ds
                        .info
                        .video_keys
                        .get(self.current_video_key_index)
                        .cloned()
                        .unwrap_or_default();
                    let display_name = current_key
                        .strip_prefix("observation.images.")
                        .unwrap_or(&current_key);
                    ui.label(display_name);
                });
            }

            if !ep.tasks.is_empty() {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Task:").color(self.theme.muted));
                    ui.label(&ep.tasks[0]);
                });
            }
        }

        ui.add_space(16.0);

        ui.label(
            egui::RichText::new("Annotation")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        let current_annotation = self.annotations.get(self.current_episode);

        let prompt_data: Vec<(usize, String, egui::Color32)> = self
            .annotations
            .prompts
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.label.clone(), p.color))
            .collect();

        for (i, label, color) in &prompt_data {
            let is_selected = current_annotation == Some(*i);
            let shortcut = format!("[{}] ", i + 1);

            let response = ui.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                ui.painter().circle_filled(rect.center(), 6.0, *color);

                let text = format!("{}{}", shortcut, label);
                let rich = if is_selected {
                    egui::RichText::new(text).strong().color(*color)
                } else {
                    egui::RichText::new(text)
                };
                ui.selectable_label(is_selected, rich)
            });

            if response.inner.clicked() {
                if is_selected {
                    self.annotations.clear(self.current_episode);
                } else {
                    self.annotations.set(self.current_episode, *i);
                }
            }
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("Save (Ctrl+S)").clicked() {
                self.save_annotations();
            }
            if self.annotations.dirty {
                ui.label(
                    egui::RichText::new("unsaved")
                        .color(egui::Color32::from_rgb(255, 200, 60))
                        .small(),
                );
            }
        });

        if let Some(path) = self.annotation_json_path() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(path).color(self.theme.muted).small());
        }
    }

    fn show_frame_display(&self, ui: &mut egui::Ui) {
        if let Some(tex) = &self.current_texture {
            let available = ui.available_size();
            let tex_size = tex.size_vec2();
            let scale = (available.x / tex_size.x)
                .min(available.y / tex_size.y)
                .min(1.0);
            let display_size = tex_size * scale;
            ui.centered_and_justified(|ui| {
                ui.image(egui::load::SizedTexture::new(tex.id(), display_size));
            });
        } else if self.dataset.is_some() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Loading...")
                        .color(self.theme.muted)
                        .size(18.0),
                );
            });
        } else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Drag and drop a LeRobot dataset folder here")
                        .color(self.theme.muted)
                        .size(18.0),
                );
            });
        }
    }

    /// Custom painted navigation slider with drag/release tracking.
    fn show_nav_slider(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let total = self.video_paths.len();
        if total <= 1 {
            return;
        }

        let max = total - 1;
        let accent = self.theme.accent;

        let slider_width = ui.available_width();
        let thickness = ui
            .text_style_height(&egui::TextStyle::Body)
            .max(ui.spacing().interact_size.y);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(slider_width, thickness), egui::Sense::drag());

        let handle_radius = rect.height() / 2.5;
        let rail_radius = 4.0_f32;
        let cy = rect.center().y;
        let handle_range = (rect.left() + handle_radius)..=(rect.right() - handle_radius);

        let mut idx = self.current_episode;

        if let Some(pos) = response.interact_pointer_pos() {
            let usable = rect.x_range().shrink(handle_radius);
            let drag_t = ((pos.x - usable.min) / (usable.max - usable.min)).clamp(0.0, 1.0);
            let new_idx = (max as f32 * drag_t).round() as usize;

            if !self.slider_dragging {
                self.slider_dragging = true;
            }

            if new_idx != self.current_episode {
                self.navigate_slider_drag(new_idx, ctx);
            }
            idx = self.current_episode;
        }

        if response.drag_stopped() && self.slider_dragging {
            self.slider_dragging = false;
            // Recenter cache around final position
            self.navigate_jump(self.current_episode, ctx);
        }

        // Draw rail
        let rail = egui::Rect::from_min_max(
            egui::pos2(rect.left(), cy - rail_radius),
            egui::pos2(rect.right(), cy + rail_radius),
        );
        let t = idx as f32 / max as f32;
        let handle_x = egui::lerp(handle_range, t);

        ui.painter()
            .rect_filled(rail, rail_radius, egui::Color32::from_gray(60));
        let filled = egui::Rect::from_min_max(rail.min, egui::pos2(handle_x, rail.max.y));
        ui.painter().rect_filled(filled, rail_radius, accent);
        ui.painter().circle(
            egui::pos2(handle_x, cy),
            handle_radius,
            accent,
            egui::Stroke::NONE,
        );
    }

    fn show_footer(&self, ui: &mut egui::Ui) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };

        let total = ds.episodes.len();
        let ep = ds.episodes.get(self.current_episode);
        let font = egui::FontId::monospace(13.0);
        let bright = egui::Color32::from_gray(200);
        let dim = egui::Color32::from_gray(160);

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("ep {:03}", self.current_episode))
                    .font(font.clone())
                    .color(bright),
            );

            if let Some(tex) = &self.current_texture {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{}x{}", tex.size()[0], tex.size()[1]))
                        .font(font.clone())
                        .color(dim),
                );
            }

            if let Some(ep) = ep {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{} frames", ep.length))
                        .font(font.clone())
                        .color(dim),
                );
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!("{} / {}", self.current_episode + 1, total))
                        .font(font.clone())
                        .color(bright),
                );
            });
        });
    }

    // -- Video mode UI --

    fn show_frame_slider(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        let total_frames = self.player.as_ref().map(|p| p.total_frames).unwrap_or(0);
        if total_frames <= 1 {
            return;
        }

        let max = total_frames - 1;
        let accent = self.theme.accent;

        let slider_width = ui.available_width();
        let thickness = ui
            .text_style_height(&egui::TextStyle::Body)
            .max(ui.spacing().interact_size.y);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(slider_width, thickness), egui::Sense::drag());

        let handle_radius = rect.height() / 2.5;
        let rail_radius = 4.0_f32;
        let cy = rect.center().y;
        let handle_range = (rect.left() + handle_radius)..=(rect.right() - handle_radius);

        let mut idx = self.current_frame;

        if let Some(pos) = response.interact_pointer_pos() {
            let usable = rect.x_range().shrink(handle_radius);
            let drag_t = ((pos.x - usable.min) / (usable.max - usable.min)).clamp(0.0, 1.0);
            let new_idx = (max as f32 * drag_t).round() as usize;

            if !self.frame_slider_dragging {
                self.frame_slider_dragging = true;
                self.playing = false; // Pause on slider grab
            }

            if new_idx != self.current_frame {
                self.current_frame = new_idx;
            }
            idx = self.current_frame;
        }

        if response.drag_stopped() && self.frame_slider_dragging {
            self.frame_slider_dragging = false;
            // Seek decoder to new position
            if let Some(player) = &mut self.player {
                player.seek(self.current_frame);
            }
        }

        // Draw rail
        let rail = egui::Rect::from_min_max(
            egui::pos2(rect.left(), cy - rail_radius),
            egui::pos2(rect.right(), cy + rail_radius),
        );
        let t = idx as f32 / max as f32;
        let handle_x = egui::lerp(handle_range, t);

        ui.painter()
            .rect_filled(rail, rail_radius, egui::Color32::from_gray(60));
        let filled = egui::Rect::from_min_max(rail.min, egui::pos2(handle_x, rail.max.y));
        ui.painter().rect_filled(filled, rail_radius, accent);
        ui.painter().circle(
            egui::pos2(handle_x, cy),
            handle_radius,
            accent,
            egui::Stroke::NONE,
        );
    }

    fn show_frame_footer(&self, ui: &mut egui::Ui) {
        let font = egui::FontId::monospace(13.0);
        let bright = egui::Color32::from_gray(200);
        let dim = egui::Color32::from_gray(160);

        let total_frames = self.player.as_ref().map(|p| p.total_frames).unwrap_or(0);
        let fps = self.player.as_ref().map(|p| p.fps).unwrap_or(30);

        ui.horizontal(|ui| {
            // Play state
            let play_icon = if self.playing { "\u{23f8}" } else { "\u{25b6}" };
            ui.label(
                egui::RichText::new(format!("{} ep {:03}", play_icon, self.current_episode))
                    .font(font.clone())
                    .color(bright),
            );

            // Timecode
            let time_s = self.current_frame as f64 / fps as f64;
            let total_s = total_frames as f64 / fps as f64;
            ui.separator();
            ui.label(
                egui::RichText::new(format!("{:.1}s / {:.1}s", time_s, total_s))
                    .font(font.clone())
                    .color(dim),
            );

            // Right-aligned: frame index
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "frame {} / {}",
                        self.current_frame + 1,
                        total_frames
                    ))
                    .font(font.clone())
                    .color(bright),
                );
            });
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.theme.apply_to_visuals(ctx);

        // DPI scaling on first frame
        if !self.initial_size_set {
            self.initial_size_set = true;
            let ppp = ctx.pixels_per_point();
            if (ppp - 1.0).abs() > 0.01 {
                let target_w = 1280.0 / ppp;
                let target_h = 720.0 / ppp;
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                    target_w, target_h,
                )));
            }
        }

        // Poll caches for completed background decodes
        if let Some(cache) = &mut self.episode_cache {
            cache.poll();
        }
        // In video mode, if we're waiting for the first frame after seek/init,
        // poll one frame from the player to display.
        if self.viewing_video && !self.playing {
            if let Some(player) = &mut self.player {
                if self.current_texture.is_none() {
                    if let Some(tex) = player.poll_next_frame() {
                        self.current_frame = player.current_frame;
                        self.current_texture = Some(tex);
                        self.perf.record_display();
                    }
                }
            }
        }

        // Advance playback
        self.tick_playback(ctx);

        self.handle_dropped_files(ctx);
        self.handle_keyboard(ctx);
        self.update_title(ctx);

        // Menu bar
        self.show_menu_bar(ctx);

        // Left panel: episode list
        egui::SidePanel::left("episode_list")
            .default_width(160.0)
            .min_width(120.0)
            .show(ctx, |ui| {
                self.show_episode_list(ctx, ui);
            });

        // Right panel: info + annotation
        egui::SidePanel::right("info_panel")
            .default_width(200.0)
            .min_width(160.0)
            .show(ctx, |ui| {
                self.show_info_panel(ui);
            });

        // Bottom: slider row + footer row
        egui::TopBottomPanel::bottom("footer")
            .exact_height(22.0)
            .show(ctx, |ui| {
                if self.viewing_video {
                    self.show_frame_footer(ui);
                } else {
                    self.show_footer(ui);
                }
            });

        egui::TopBottomPanel::bottom("nav_slider")
            .exact_height(28.0)
            .show(ctx, |ui| {
                if self.viewing_video {
                    self.show_frame_slider(ctx, ui);
                } else {
                    self.show_nav_slider(ctx, ui);
                }
            });

        // Central panel: frame display
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.loading_error {
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), err);
                ui.separator();
            }
            self.show_frame_display(ui);
        });

        // Cache debug overlay
        if self.show_cache_overlay {
            if let Some(cache) = &self.episode_cache {
                cache.show_debug_overlay(ctx, self.current_episode, self.video_paths.len());
            }
        }
    }
}
