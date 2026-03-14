use std::path::PathBuf;

use eframe::egui;

use crate::annotation::AnnotationState;
use crate::dataset::{self, LeRobotDataset};
use crate::perf::PerfTracker;
use crate::theme::UiTheme;
use crate::video;

pub struct App {
    // Data
    dataset: Option<LeRobotDataset>,
    annotations: AnnotationState,

    // Viewer state
    current_episode: usize,
    current_video_key_index: usize,
    current_texture: Option<egui::TextureHandle>,
    loading_error: Option<String>,

    // UI
    theme: UiTheme,
    perf: PerfTracker,
    initial_size_set: bool,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext, initial_path: Option<PathBuf>) -> Self {
        let mut app = Self {
            dataset: None,
            annotations: AnnotationState::new_cube_task(),
            current_episode: 0,
            current_video_key_index: 0,
            current_texture: None,
            loading_error: None,
            theme: UiTheme::teal_dark(),
            perf: PerfTracker::new(),
            initial_size_set: false,
        };

        if let Some(path) = initial_path {
            app.load_dataset(&path);
        }

        app
    }

    fn load_dataset(&mut self, path: &std::path::Path) {
        match LeRobotDataset::load(path) {
            Ok(ds) => {
                log::info!("Dataset loaded: {} episodes", ds.episodes.len());
                // Default to wrist camera if available, otherwise first
                let wrist_idx = ds
                    .info
                    .video_keys
                    .iter()
                    .position(|k| k.contains("wrist"))
                    .unwrap_or(0);
                self.current_video_key_index = wrist_idx;
                self.dataset = Some(ds);
                self.current_episode = 0;
                self.current_texture = None;
                self.loading_error = None;

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

    fn load_current_frame(&mut self, ctx: &egui::Context) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };

        let video_key = match ds.info.video_keys.get(self.current_video_key_index) {
            Some(k) => k.clone(),
            None => return,
        };

        let video_path = ds.video_path(self.current_episode, &video_key);
        log::debug!("Loading frame from: {}", video_path.display());

        let start = std::time::Instant::now();
        match video::decode_middle_frame(&video_path) {
            Ok(image) => {
                let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
                log::debug!(
                    "Decoded ep {} in {:.1}ms ({}x{})",
                    self.current_episode,
                    decode_ms,
                    image.size[0],
                    image.size[1],
                );
                let name = format!("ep{}_{}", self.current_episode, video_key);
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

    fn navigate_to_episode(&mut self, episode: usize, ctx: &egui::Context) {
        let total = self
            .dataset
            .as_ref()
            .map(|ds| ds.episodes.len())
            .unwrap_or(0);
        if total == 0 {
            return;
        }
        let episode = episode.min(total - 1);
        if episode != self.current_episode || self.current_texture.is_none() {
            self.current_episode = episode;
            self.current_texture = None;
            self.load_current_frame(ctx);
        }
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        if self.dataset.is_none() {
            return;
        }
        let total = self
            .dataset
            .as_ref()
            .map(|ds| ds.episodes.len())
            .unwrap_or(0);

        ctx.input(|i| {
            // Episode navigation
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                let next = (self.current_episode + 1).min(total.saturating_sub(1));
                self.current_episode = next;
                self.current_texture = None;
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                self.current_episode = self.current_episode.saturating_sub(1);
                self.current_texture = None;
            }
            if i.key_pressed(egui::Key::Home) {
                self.current_episode = 0;
                self.current_texture = None;
            }
            if i.key_pressed(egui::Key::End) {
                self.current_episode = total.saturating_sub(1);
                self.current_texture = None;
            }

            // Annotation shortcuts (1-4)
            for (key, idx) in [
                (egui::Key::Num1, 0),
                (egui::Key::Num2, 1),
                (egui::Key::Num3, 2),
                (egui::Key::Num4, 3),
            ] {
                if i.key_pressed(key) {
                    self.annotations.set(self.current_episode, idx);
                }
            }

            // Save annotations (Ctrl+S)
            if i.modifiers.command && i.key_pressed(egui::Key::S) {
                self.save_annotations();
            }
        });

        // Load frame if texture was cleared by navigation
        if self.current_texture.is_none() && self.dataset.is_some() {
            self.load_current_frame(ctx);
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
                    self.load_current_frame(ctx);
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
            format!(
                "lerobot-explorer - {} [{}/{}]",
                dir_name, done, total,
            )
        } else {
            "lerobot-explorer".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
    }

    // -- UI panels --

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

        egui::ScrollArea::vertical().show(ui, |ui| {
            for ep in &ds.episodes {
                let is_selected = ep.episode_index == self.current_episode;
                let annotation = self.annotations.get(ep.episode_index);

                let mut label_text = format!("ep {:03}", ep.episode_index);
                if let Some(prompt_idx) = annotation {
                    if let Some(prompt) = self.annotations.prompts.get(prompt_idx) {
                        label_text = format!("{} - {}", label_text, prompt.label);
                    }
                }

                let response = ui.selectable_label(is_selected, &label_text);

                // Draw colored dot for annotated episodes
                if let Some(prompt_idx) = annotation {
                    if let Some(prompt) = self.annotations.prompts.get(prompt_idx) {
                        let rect = response.rect;
                        let dot_pos = egui::pos2(
                            rect.right() - 8.0,
                            rect.center().y,
                        );
                        ui.painter().circle_filled(dot_pos, 4.0, prompt.color);
                    }
                }

                if response.clicked() {
                    navigate_to = Some(ep.episode_index);
                }
            }
        });

        if let Some(ep) = navigate_to {
            self.navigate_to_episode(ep, ctx);
        }
    }

    fn show_info_panel(&mut self, ui: &mut egui::Ui) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };

        // Episode info
        ui.label(
            egui::RichText::new("Episode Info")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        if let Some(ep) = ds.episodes.get(self.current_episode) {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Episode:").color(self.theme.muted));
                ui.label(format!("{}", ep.episode_index));
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Frames:").color(self.theme.muted));
                ui.label(format!("{}", ep.length));
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Duration:").color(self.theme.muted));
                ui.label(format!("{:.1}s", ds.episode_duration(self.current_episode)));
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("FPS:").color(self.theme.muted));
                ui.label(format!("{}", ds.info.fps));
            });

            // Camera selector
            if ds.info.video_keys.len() > 1 {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Camera:").color(self.theme.muted));
                    let current_key = ds
                        .info
                        .video_keys
                        .get(self.current_video_key_index)
                        .cloned()
                        .unwrap_or_default();
                    // Strip the "observation.images." prefix for display
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

        // Annotation prompt cards
        ui.label(
            egui::RichText::new("Annotation")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        let current_annotation = self.annotations.get(self.current_episode);

        // Collect prompt display data to avoid borrowing self.annotations during mutation
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
                let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
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
        if self.annotations.dirty {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Unsaved changes")
                        .color(egui::Color32::from_rgb(255, 200, 60))
                        .small(),
                );
                if ui.small_button("Save (Ctrl+S)").clicked() {
                    self.save_annotations();
                }
            });
        }
    }

    fn show_frame_display(&self, ui: &mut egui::Ui) {
        if let Some(tex) = &self.current_texture {
            let available = ui.available_size();
            let tex_size = tex.size_vec2();
            let scale = (available.x / tex_size.x).min(available.y / tex_size.y).min(1.0);
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

    fn show_nav_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let total = self
            .dataset
            .as_ref()
            .map(|ds| ds.episodes.len())
            .unwrap_or(0);
        if total == 0 {
            return;
        }

        ui.horizontal(|ui| {
            // Episode slider
            let mut ep = self.current_episode as f64;
            let slider = egui::Slider::new(&mut ep, 0.0..=((total - 1) as f64))
                .show_value(false)
                .step_by(1.0);
            let response = ui.add(slider);
            if response.changed() {
                let new_ep = ep as usize;
                self.navigate_to_episode(new_ep, ctx);
            }

            // Episode counter
            ui.label(
                egui::RichText::new(format!("ep {} / {}", self.current_episode, total))
                    .monospace()
                    .color(self.theme.muted),
            );

            // FPS
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(self.perf.fps_text())
                        .monospace()
                        .color(self.theme.muted)
                        .size(11.0),
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

        self.handle_dropped_files(ctx);
        self.handle_keyboard(ctx);
        self.update_title(ctx);

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

        // Bottom panel: navigation bar
        egui::TopBottomPanel::bottom("nav_bar")
            .exact_height(32.0)
            .show(ctx, |ui| {
                self.show_nav_bar(ctx, ui);
            });

        // Central panel: frame display
        egui::CentralPanel::default().show(ctx, |ui| {
            // Show error if any
            if let Some(err) = &self.loading_error {
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), err);
                ui.separator();
            }
            self.show_frame_display(ui);
        });
    }
}
