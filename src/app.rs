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
            format!("lerobot-explorer - {} [{}/{}]", dir_name, done, total)
        } else {
            "lerobot-explorer".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
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
                                self.load_current_frame(ctx);
                            }
                        }
                    }
                    if ui.button("Save Annotations  Ctrl+S").clicked() {
                        ui.close_menu();
                        self.save_annotations();
                    }
                    if ui
                        .add_enabled(self.dataset.is_some(), egui::Button::new("Export to LeRobot..."))
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

        // Collect annotation data upfront to avoid borrow issues
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
                    // Colored dot (left side, before text)
                    if let Some((_, color)) = annot_info {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 4.0, *color);
                    } else {
                        // Reserve space for alignment
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
            let info_rows: Vec<(&str, String)> = vec![
                ("Episode:", format!("{} / {}", ep.episode_index, ds.episodes.len())),
                ("Frames:", format!("{}", ep.length)),
                ("Duration:", format!("{:.1}s", ep.length as f64 / ds.info.fps as f64)),
                ("FPS:", format!("{}", ds.info.fps)),
            ];

            for (label, value) in &info_rows {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(*label).color(self.theme.muted));
                    ui.label(value);
                });
            }

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

        // Save button (always visible)
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

        // Annotation file path
        if let Some(path) = self.annotation_json_path() {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new(path)
                    .color(self.theme.muted)
                    .small(),
            );
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

    fn show_nav_slider(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let total = self
            .dataset
            .as_ref()
            .map(|ds| ds.episodes.len())
            .unwrap_or(0);
        if total == 0 {
            return;
        }

        let mut ep = self.current_episode as f64;
        let slider = egui::Slider::new(&mut ep, 0.0..=((total - 1) as f64))
            .show_value(false)
            .step_by(1.0);
        let response = ui.add_sized([ui.available_width(), 20.0], slider);
        if response.changed() {
            let new_ep = ep as usize;
            self.navigate_to_episode(new_ep, ctx);
        }
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
            // Episode index
            ui.label(
                egui::RichText::new(format!("ep {:03}", self.current_episode))
                    .font(font.clone())
                    .color(bright),
            );

            // Resolution
            if let Some(tex) = &self.current_texture {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{}x{}", tex.size()[0], tex.size()[1]))
                        .font(font.clone())
                        .color(dim),
                );
            }

            // Frame count
            if let Some(ep) = ep {
                ui.separator();
                ui.label(
                    egui::RichText::new(format!("{} frames", ep.length))
                        .font(font.clone())
                        .color(dim),
                );
            }

            // Right-aligned: index / total
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!("{} / {}", self.current_episode + 1, total))
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
                self.show_footer(ui);
            });

        egui::TopBottomPanel::bottom("nav_slider")
            .exact_height(28.0)
            .show(ctx, |ui| {
                self.show_nav_slider(ctx, ui);
            });

        // Central panel: frame display
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.loading_error {
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), err);
                ui.separator();
            }
            self.show_frame_display(ui);
        });
    }
}
