use eframe::egui;

use crate::app::App;
use crate::trajectory;
use crate::trajectory_view::TrajectoryEntry;

impl App {
    pub(crate) fn show_menu_bar(&mut self, ctx: &egui::Context) {
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
                    if self.annotate_mode {
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
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                ui.menu_button("View", |ui| {
                    let in_grid = self.grid_view.is_some();
                    let has_multi_cam = self.dataset.as_ref()
                        .map(|ds| ds.info.video_keys.len() > 1)
                        .unwrap_or(false);

                    if ui.button(if in_grid { "Single View  [G]" } else { "Grid View  [G]" }).clicked() {
                        ui.close_menu();
                        self.toggle_grid_view(ctx);
                    }
                    if has_multi_cam {
                        let is_camera_grid = self.is_camera_grid();
                        let mc_label = if is_camera_grid { "Single Camera  [M]" } else { "Multi-Camera  [M]" };
                        if ui.button(mc_label).clicked() {
                            ui.close_menu();
                            self.toggle_multi_camera(ctx);
                        }

                        let is_tiled_now = self.camera_display == crate::app::CameraDisplay::Tiled;
                        let tiled_label = if is_tiled_now { "Exit Tiled  [N]" } else { "Tiled View  [N]" };
                        if ui.button(tiled_label).clicked() {
                            ui.close_menu();
                            self.toggle_tiled(ctx);
                        }
                    }
                    ui.separator();

                    // Grid size picker: controls the episode grid (SingleCamera / Subgrid)
                    ui.label(
                        egui::RichText::new("Grid Size")
                            .color(self.theme.muted)
                            .small(),
                    );
                    if let Some((cols, rows)) = grid_size_picker(ui, self.grid_cols, self.grid_rows, self.theme.accent) {
                        self.grid_cols = cols;
                        self.grid_rows = rows;
                        // Grid picker enters episode-grid mode (SingleCamera or Subgrid, not Tiled)
                        if self.camera_display == crate::app::CameraDisplay::Tiled {
                            self.camera_display = crate::app::CameraDisplay::SingleCamera;
                        }
                        if !in_grid {
                            self.toggle_grid_view(ctx);
                        } else {
                            let start = self.grid_view.as_ref().map(|g| g.start_episode).unwrap_or(0);
                            self.enter_grid_with_camera_display(ctx, start);
                        }
                    }

                    // Matrix rows slider: always visible when multi-cam is available.
                    // Interacting enters Tiled mode directly.
                    if has_multi_cam {
                        ui.separator();
                        ui.label(
                            egui::RichText::new("Tiled View Rows  [N]")
                                .color(self.theme.muted)
                                .small(),
                        );
                        let mut tiled_rows = self.grid_rows;
                        if ui.add(egui::Slider::new(&mut tiled_rows, 1..=10).text("rows")).changed() {
                            self.grid_rows = tiled_rows;
                            self.camera_display = crate::app::CameraDisplay::Tiled;
                            let start = self.grid_view.as_ref()
                                .map(|g| g.start_episode)
                                .unwrap_or(self.current_episode);
                            if !in_grid {
                                self.exit_video_mode();
                            }
                            self.enter_grid_with_camera_display(ctx, start);
                        }
                    }

                    ui.separator();
                    if ui
                        .checkbox(&mut self.show_trajectory, "EE Trajectory")
                        .changed()
                    {
                        ui.close_menu();
                    }
                    if ui
                        .checkbox(&mut self.show_cache_overlay, "Cache Overlay")
                        .changed()
                    {
                        ui.close_menu();
                    }
                });

                ui.menu_button("Help", |ui| {
                    let shortcut_label = if self.show_shortcut_bar {
                        "Hide Shortcuts  [?]"
                    } else {
                        "Keyboard Shortcuts  [?]"
                    };
                    if ui.button(shortcut_label).clicked() {
                        self.show_shortcut_bar = !self.show_shortcut_bar;
                        ui.close_menu();
                    }
                    if ui.button("About...").clicked() {
                        self.show_about = true;
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

    pub(crate) fn show_shortcut_bar(&self, ctx: &egui::Context) {
        if !self.show_shortcut_bar {
            return;
        }
        egui::TopBottomPanel::top("shortcut_bar").show(ctx, |ui| {
            let font = egui::FontId::monospace(12.0);
            let dim = egui::Color32::from_gray(140);
            ui.horizontal(|ui| {
                let in_grid = self.grid_view.is_some();
                let has_multi_cam = self.dataset.as_ref()
                    .map(|ds| ds.info.video_keys.len() > 1)
                    .unwrap_or(false);
                let has_traj = !self.arms.is_empty();
                let is_tiled = in_grid && self.grid_view.as_ref()
                    .map(|g| g.cam_count > 1 && !g.subgrid).unwrap_or(false);
                let is_subgrid = in_grid && self.grid_view.as_ref()
                    .map(|g| g.subgrid).unwrap_or(false);

                let hints: Vec<&str> = if !in_grid {
                    // Single video mode
                    let mut h = vec!["[G] grid", "[Space] play/pause", "[A/D] episode"];
                    if has_multi_cam {
                        h.push("[M] multi-cam");
                        h.push("[C] camera");
                    }
                    if has_traj { h.push("[T] trajectory"); }
                    h.push("[Enter] video mode");
                    h
                } else {
                    match self.grid_view.as_ref().map(|g| g.mode) {
                        Some(crate::grid::GridMode::MultiCamera) => {
                            vec!["[G] exit", "[M] exit", "[Space] play/pause", "[Esc] reset"]
                        }
                        Some(crate::grid::GridMode::MultiEpisode) if is_tiled => {
                            let mut h = vec!["[G] exit"];
                            if has_multi_cam { h.push("[M] subgrid"); }
                            h.extend_from_slice(&["[N] untile", "[+/-] resize", "[A/D] page", "[L] labels", "[Esc] reset"]);
                            h
                        }
                        Some(crate::grid::GridMode::MultiEpisode) if is_subgrid => {
                            let mut h = vec!["[G] exit"];
                            if has_multi_cam {
                                h.push("[M] single-cam");
                                h.push("[N] tiled");
                            }
                            h.extend_from_slice(&["[+/-] resize", "[A/D] page", "[L] labels", "[Esc] reset"]);
                            h
                        }
                        _ => {
                            // Flat single-cam grid
                            let mut h = vec!["[G] exit"];
                            if has_multi_cam {
                                h.push("[M] subgrid");
                                h.push("[N] tiled");
                                h.push("[C] camera");
                            }
                            h.push("[+/-] resize");
                            h.push("[A/D] page");
                            h.push("[L] labels");
                            if has_traj { h.push("[T] trajectory"); }
                            h.push("[Esc] reset");
                            h
                        }
                    }
                };

                for (i, hint) in hints.iter().enumerate() {
                    if i > 0 {
                        ui.label(egui::RichText::new("|").font(font.clone()).color(egui::Color32::from_gray(60)));
                    }
                    ui.label(egui::RichText::new(*hint).font(font.clone()).color(dim));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new("[?] hide").font(font.clone()).color(egui::Color32::from_gray(100)));
                });
            });
        });
    }

    pub(crate) fn show_episode_list(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => {
                ui.label("No dataset loaded");
                return;
            }
        };

        let heading = if self.annotate_mode {
            let (done, total) = self.annotations.progress(ds.episodes.len());
            format!("Episodes ({}/{})", done, total)
        } else {
            format!("Episodes ({})", ds.episodes.len())
        };
        ui.label(
            egui::RichText::new(heading)
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        let mut navigate_to = None;

        // Determine which episodes are highlighted (single or grid range)
        let grid_range = self.grid_view.as_ref().map(|g| match g.mode {
            crate::grid::GridMode::MultiCamera => g.fixed_episode..g.fixed_episode + 1,
            crate::grid::GridMode::MultiEpisode => {
                let end = g.start_episode + g.episodes_shown();
                g.start_episode..end
            }
        });

        let should_scroll = self.scroll_to_selected;
        let mut scroll_rect: Option<egui::Rect> = None;

        egui::ScrollArea::vertical().show(ui, |ui| {
            for ep in &ds.episodes {
                let episode_index = ep.episode_index;
                let is_selected = if let Some(range) = &grid_range {
                    range.contains(&episode_index)
                } else {
                    episode_index == self.current_episode
                };

                let annot_info = if self.annotate_mode {
                    self.annotations.get(episode_index).and_then(|idx| {
                        self.annotations
                            .prompts
                            .get(idx)
                            .map(|p| (p.label.clone(), p.color))
                    })
                } else {
                    None
                };

                let response = ui.horizontal(|ui| {
                    if self.annotate_mode {
                        if let Some((_, color)) = &annot_info {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 4.0, *color);
                        } else {
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        }
                    }

                    let mut label_text = format!("ep {:03}", episode_index);
                    if let Some((label, _)) = &annot_info {
                        label_text = format!("{} - {}", label_text, label);
                    }

                    ui.selectable_label(is_selected, &label_text)
                });

                // Accumulate rect of all selected items for scroll_to_rect
                if should_scroll && is_selected {
                    scroll_rect = Some(match scroll_rect {
                        Some(r) => r.union(response.response.rect),
                        None => response.response.rect,
                    });
                }

                if response.inner.clicked() {
                    navigate_to = Some(episode_index);
                }
            }

            // Scroll to show all selected episodes
            if let Some(rect) = scroll_rect {
                ui.scroll_to_rect(rect, None);
            }
        });

        if should_scroll {
            self.scroll_to_selected = false;
        }

        if let Some(ep) = navigate_to {
            let grid_mode = self.grid_view.as_ref().map(|g| g.mode);
            match grid_mode {
                Some(crate::grid::GridMode::MultiCamera) => {
                    self.current_episode = ep;
                    if let Some(ds) = &self.dataset {
                        let grid = crate::grid::GridView::new_multi_camera(
                            ctx, ep, ds, &self.selected_cameras,
                        );
                        self.grid_view = Some(grid);
                    }
                    self.scroll_to_selected = true;
                }
                Some(crate::grid::GridMode::MultiEpisode) => {
                    let is_tiled = self.grid_view.as_ref()
                        .map(|g| g.cam_count > 1).unwrap_or(false);
                    if is_tiled {
                        self.current_episode = ep;
                        if let Some(ds) = &self.dataset {
                            let grid = crate::grid::GridView::new_tiled(
                                ctx, self.grid_cols, self.grid_rows,
                                ep, ds, &self.selected_cameras,
                            );
                            self.grid_view = Some(grid);
                        }
                        self.scroll_to_selected = true;
                    } else {
                        self.grid_jump_to(ep, ctx);
                    }
                }
                None => {
                    if self.viewing_video {
                        self.exit_video_mode();
                    }
                    self.navigate_jump(ep, ctx);
                }
            }
        }
    }

    pub(crate) fn show_info_panel(&mut self, ui: &mut egui::Ui) {
        // Capture owned snapshots so later `&mut self` calls (e.g. show_camera_combobox)
        // don't conflict with the dataset borrow.
        let snapshot = self.dataset.as_ref().and_then(|ds| {
            let ep = ds.episodes.get(self.current_episode)?;
            let total_eps = ds.episodes.len();
            let fps = ds.info.fps;
            let cam_count = ds.info.video_keys.len();
            let single_cam_name = if cam_count == 1 {
                Some(crate::dataset::camera_display_name(&ds.info.video_keys[0]).to_string())
            } else {
                None
            };
            Some((
                ep.episode_index,
                total_eps,
                ep.length,
                fps,
                cam_count,
                single_cam_name,
                ep.tasks.first().cloned(),
            ))
        });

        ui.label(
            egui::RichText::new("Episode Info")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        if let Some((ep_idx, total_eps, ep_length, fps, cam_count, single_cam_name, task)) = snapshot {
            let info_rows: [(&str, String); 4] = [
                ("Episode:", format!("{} / {}", ep_idx, total_eps)),
                ("Frames:", format!("{}", ep_length)),
                ("Duration:", format!("{:.1}s", ep_length as f64 / fps as f64)),
                ("FPS:", format!("{}", fps)),
            ];
            for (label, value) in &info_rows {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(*label).color(self.theme.muted));
                    ui.label(value);
                });
            }

            if cam_count > 0 {
                let muted = self.theme.muted;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Camera:").color(muted));
                    if cam_count > 1 {
                        self.show_camera_combobox(ui, "camera_select", 100.0);
                    } else if let Some(name) = single_cam_name {
                        ui.label(name);
                    }
                });
            }

            if let Some(task) = task {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Task:").color(self.theme.muted));
                    ui.label(task);
                });
            }
        }

        if self.annotate_mode {
            ui.add_space(16.0);
            self.show_annotation_panel(ui);
        }

        // Trajectory view in single-view mode
        if self.show_trajectory && self.dataset.is_some() {
            ui.add_space(16.0);
            self.show_trajectory_panel(ui);
        }
    }

    /// Render a camera ComboBox that sets `pending_camera_switch` on change.
    /// Used in both the single-video Episode Info panel and the single-cam grid
    /// Cameras panel.
    fn show_camera_combobox(&mut self, ui: &mut egui::Ui, id_salt: &str, width: f32) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };
        if ds.info.video_keys.len() <= 1 {
            return;
        }
        let camera_names: Vec<(usize, String)> = ds.info.video_keys.iter().enumerate()
            .map(|(i, k)| (i, crate::dataset::camera_display_name(k).to_string()))
            .collect();
        let current_display = camera_names.iter()
            .find(|(i, _)| *i == self.current_video_key_index)
            .map(|(_, name)| name.as_str())
            .unwrap_or("unknown");
        let mut selected = self.current_video_key_index;
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(current_display)
            .width(width)
            .show_ui(ui, |ui| {
                for (idx, name) in &camera_names {
                    ui.selectable_value(&mut selected, *idx, name);
                }
            });
        if selected != self.current_video_key_index {
            self.pending_camera_switch = Some(selected);
        }
    }

    /// Grid-mode right panel: shows the camera selector (ComboBox in single-cam
    /// grid, checkboxes in multi-cam modes) and the trajectory view below it.
    /// No Episode Info section because it would be ambiguous when the grid shows
    /// multiple episodes.
    pub(crate) fn show_cameras_panel(&mut self, ui: &mut egui::Ui) {
        // Capture owned snapshot so later &mut self calls don't conflict
        // with the dataset borrow.
        let snapshot: Option<(Vec<String>, Option<String>)> = self.dataset.as_ref().map(|ds| {
            let names: Vec<String> = ds.info.video_keys.iter()
                .map(|k| crate::dataset::camera_display_name(k).to_string())
                .collect();
            let single = if names.len() == 1 { Some(names[0].clone()) } else { None };
            (names, single)
        });
        let Some((cam_names, single_cam_name)) = snapshot else { return };

        ui.label(
            egui::RichText::new("Cameras")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        let in_multi_cam = self.is_camera_grid();
        if in_multi_cam && cam_names.len() > 1 {
            // Multi-cam mode: checkboxes
            let mut changed = false;
            for (i, display) in cam_names.iter().enumerate() {
                let mut checked = self.selected_cameras.get(i).copied().unwrap_or(true);
                if ui.checkbox(&mut checked, display).changed() {
                    let other_selected = self.selected_cameras.iter().enumerate()
                        .any(|(j, &v)| j != i && v);
                    if checked || other_selected {
                        if let Some(slot) = self.selected_cameras.get_mut(i) {
                            *slot = checked;
                        }
                        changed = true;
                    }
                }
            }
            if changed {
                self.pending_multi_camera_rebuild = true;
            }
        } else if cam_names.len() > 1 {
            self.show_camera_combobox(ui, "camera_select_grid", 140.0);
        } else if let Some(name) = single_cam_name {
            ui.label(name);
        }

        // Trajectory view below the cameras section
        if self.show_trajectory && self.dataset.is_some() {
            ui.add_space(16.0);
            self.show_trajectory_panel(ui);
        }
    }

    fn show_annotation_panel(&mut self, ui: &mut egui::Ui) {
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

    pub(crate) fn show_frame_display(&self, ui: &mut egui::Ui) {
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

    pub(crate) fn show_nav_slider(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
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

    pub(crate) fn show_footer(&self, ui: &mut egui::Ui) {
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

    pub(crate) fn show_frame_slider(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        let ep_length = self.dataset.as_ref()
            .and_then(|ds| ds.episodes.get(self.current_episode))
            .map(|ep| ep.length)
            .unwrap_or(0);
        if ep_length <= 1 {
            return;
        }

        let max = ep_length - 1;
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

        // Episode-relative frame position
        let ep_frame = self.current_frame.saturating_sub(self.episode_start_frame);
        let mut idx = ep_frame;

        if let Some(pos) = response.interact_pointer_pos() {
            let usable = rect.x_range().shrink(handle_radius);
            let drag_t = ((pos.x - usable.min) / (usable.max - usable.min)).clamp(0.0, 1.0);
            let new_ep_frame = (max as f32 * drag_t).round() as usize;

            if !self.frame_slider_dragging {
                self.frame_slider_dragging = true;
                self.playing = false;
            }

            let new_abs_frame = self.episode_start_frame + new_ep_frame;
            if new_abs_frame != self.current_frame {
                self.current_frame = new_abs_frame;

                // Throttled scrub: decode a single frame at the new position (~15 Hz)
                let now = std::time::Instant::now();
                let scrub_interval = std::time::Duration::from_millis(67);
                let should_scrub = self.last_scrub_seek
                    .is_none_or(|t| now.duration_since(t) >= scrub_interval);
                if should_scrub {
                    if let Some(player) = &mut self.player {
                        player.scrub_to(self.current_frame);
                    }
                    self.last_scrub_seek = Some(now);
                }
            }
            idx = new_ep_frame;
        }

        if response.drag_stopped() && self.frame_slider_dragging {
            self.frame_slider_dragging = false;
            self.last_scrub_seek = None;
            if let Some(player) = &mut self.player {
                player.cancel_scrub();
                player.seek(self.current_frame);
            }
            // Resume playback immediately from the new position
            self.playing = true;
            self.last_frame_time = None;
        }

        // Draw rail
        let rail = egui::Rect::from_min_max(
            egui::pos2(rect.left(), cy - rail_radius),
            egui::pos2(rect.right(), cy + rail_radius),
        );
        let t = if max > 0 { idx as f32 / max as f32 } else { 0.0 };
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

    pub(crate) fn show_frame_footer(&mut self, ui: &mut egui::Ui) {
        let font = egui::FontId::monospace(13.0);
        let bright = egui::Color32::from_gray(200);
        let dim = egui::Color32::from_gray(160);

        let ep_length = self.dataset.as_ref()
            .and_then(|ds| ds.episodes.get(self.current_episode))
            .map(|ep| ep.length)
            .unwrap_or(0);
        let fps = self.player.as_ref().map(|p| p.fps).unwrap_or(30);
        let ep_frame = self.current_frame.saturating_sub(self.episode_start_frame);

        ui.horizontal(|ui| {
            // Clickable play/pause button
            let play_icon = if self.playing { "\u{23f8}" } else { "\u{25b6}" };
            let btn = ui.button(
                egui::RichText::new(play_icon)
                    .font(font.clone())
                    .color(bright),
            );
            if btn.clicked() {
                self.playing = !self.playing;
                if self.playing {
                    self.last_frame_time = Some(std::time::Instant::now());
                } else {
                    self.last_frame_time = None;
                }
            }

            ui.label(
                egui::RichText::new(format!("ep {:03}", self.current_episode))
                    .font(font.clone())
                    .color(bright),
            );

            // Timecode (episode-relative)
            let time_s = ep_frame as f64 / fps as f64;
            let total_s = ep_length as f64 / fps as f64;
            ui.separator();
            ui.label(
                egui::RichText::new(format!("{:.1}s / {:.1}s", time_s, total_s))
                    .font(font.clone())
                    .color(dim),
            );

            // Right-aligned: frame index (episode-relative)
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "frame {} / {}",
                        ep_frame + 1,
                        ep_length
                    ))
                    .font(font.clone())
                    .color(bright),
                );
            });
        });
    }

    pub(crate) fn show_grid_display(&mut self, ui: &mut egui::Ui) {
        let accent = self.theme.accent;
        if let Some(grid) = &mut self.grid_view {
            grid.show(ui, accent, self.label_mode);
        }
    }

    pub(crate) fn show_grid_frame_slider(&mut self, _ctx: &egui::Context, ui: &mut egui::Ui) {
        let (max_len, current_rel) = match &self.grid_view {
            Some(grid) => (grid.max_episode_length(), grid.current_relative_frame()),
            None => return,
        };
        if max_len <= 1 {
            return;
        }

        let max = max_len - 1;
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

        let mut idx = current_rel;

        if let Some(pos) = response.interact_pointer_pos() {
            let usable = rect.x_range().shrink(handle_radius);
            let drag_t = ((pos.x - usable.min) / (usable.max - usable.min)).clamp(0.0, 1.0);
            let new_rel_frame = (max as f32 * drag_t).round() as usize;

            if let Some(grid) = &mut self.grid_view {
                if !grid.frame_slider_dragging {
                    grid.frame_slider_dragging = true;
                    grid.playing = false;
                }

                // Throttled scrub across all panes (~15 Hz)
                let now = std::time::Instant::now();
                let scrub_interval = std::time::Duration::from_millis(67);
                let should_scrub = self.last_scrub_seek
                    .is_none_or(|t| now.duration_since(t) >= scrub_interval);
                if should_scrub {
                    grid.scrub_all_to(new_rel_frame);
                    self.last_scrub_seek = Some(now);
                }
            }
            idx = new_rel_frame;
        }

        if response.drag_stopped() {
            let dragging = self.grid_view.as_ref().is_some_and(|g| g.frame_slider_dragging);
            if dragging {
                self.last_scrub_seek = None;
                if let Some(grid) = &mut self.grid_view {
                    grid.finish_scrub(idx);
                }
            }
        }

        // Draw rail
        let rail = egui::Rect::from_min_max(
            egui::pos2(rect.left(), cy - rail_radius),
            egui::pos2(rect.right(), cy + rail_radius),
        );
        let t = if max > 0 { idx as f32 / max as f32 } else { 0.0 };
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

    pub(crate) fn show_grid_footer(&self, ui: &mut egui::Ui) {
        let font = egui::FontId::monospace(13.0);
        let bright = egui::Color32::from_gray(200);

        ui.horizontal(|ui| {
            if let Some(grid) = &self.grid_view {
                let playing_icon = if grid.playing { "\u{23f8}" } else { "\u{25b6}" };
                ui.label(
                    egui::RichText::new(playing_icon)
                        .font(font.clone())
                        .color(bright),
                );

                match grid.mode {
                    crate::grid::GridMode::MultiCamera => {
                        ui.label(
                            egui::RichText::new(format!(
                                "ep {:03}  {} cameras",
                                grid.fixed_episode, grid.pane_count(),
                            ))
                            .font(font.clone())
                            .color(bright),
                        );
                    }
                    crate::grid::GridMode::MultiEpisode => {
                        let total = self.video_paths.len();
                        let eps = grid.episodes_shown();
                        let end = (grid.start_episode + eps).min(total);
                        let mut info = format!(
                            "ep {}-{} / {}",
                            grid.start_episode, end.saturating_sub(1), total,
                        );
                        if grid.cam_count > 1 {
                            info.push_str(&format!("  {} cams", grid.cam_count));
                        } else if let Some(ds) = &self.dataset {
                            if ds.info.video_keys.len() > 1 {
                                let cam = ds.info.video_keys
                                    .get(self.current_video_key_index)
                                    .map(|k| crate::dataset::camera_display_name(k))
                                    .unwrap_or("?");
                                info.push_str(&format!("  {}", cam));
                            }
                        }
                        ui.label(
                            egui::RichText::new(info)
                                .font(font.clone())
                                .color(bright),
                        );
                    }
                }
            }
        });
    }

    pub(crate) fn show_trajectory_panel(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new("EE Trajectory")
                .strong()
                .color(self.theme.heading),
        );
        ui.separator();

        if self.arms.is_empty() {
            let robot_type = self.dataset.as_ref()
                .and_then(|ds| ds.info.robot_type.clone());
            self.show_urdf_missing_panel(ui, robot_type.as_deref());
            return;
        }

        // Arm selector (only shown when multiple arms are loaded)
        if self.arms.len() > 1 {
            let prev_arm = self.active_arm_index;
            let current_name = self.arms.get(self.active_arm_index)
                .map(|a| a.name.as_str())
                .unwrap_or("default");
            egui::ComboBox::from_id_salt("arm_select")
                .selected_text(current_name)
                .show_ui(ui, |ui| {
                    for (i, arm) in self.arms.iter().enumerate() {
                        ui.selectable_value(
                            &mut self.active_arm_index,
                            i,
                            &arm.name,
                        );
                    }
                });
            if self.active_arm_index != prev_arm {
                self.trajectory_cache = crate::trajectory::TrajectoryCache::new(100);
            }
            ui.add_space(4.0);
        }

        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };

        let arm_idx = self.active_arm_index.min(self.arms.len().saturating_sub(1));
        let kin = &self.arms[arm_idx].kinematics;
        let pos_indices = &self.arms[arm_idx].pos_indices;

        // Collect episode indices + frames to display
        let (pane_episodes, selected_episodes) = if let Some(grid) = &self.grid_view {
            let all = grid.all_pane_episodes();
            let sel = grid.selected_episodes();
            (all, sel)
        } else {
            let frame = if self.viewing_video {
                self.current_frame.saturating_sub(self.episode_start_frame)
            } else {
                0
            };
            let mut sel = std::collections::HashSet::new();
            sel.insert(self.current_episode);
            (vec![(self.current_episode, frame)], sel)
        };

        if pane_episodes.is_empty() {
            return;
        }

        // Ensure all trajectories are cached
        for &(ep_idx, _) in &pane_episodes {
            if self.trajectory_cache.get(ep_idx).is_some() {
                continue;
            }
            let is_v3 = ds.info.codebase_version.starts_with("v3");
            let ep_meta = ds.episodes.get(ep_idx);
            let (data_chunk, data_file) = ep_meta
                .map(|m| (m.data_chunk_index, m.data_file_index))
                .unwrap_or((0, 0));
            let parquet_path = trajectory::episode_data_path(
                &ds.root,
                ep_idx,
                ds.info.chunks_size,
                &ds.info.codebase_version,
                data_chunk,
                data_file,
            );
            let filter_ep = if is_v3 { Some(ep_idx) } else { None };
            match trajectory::load_episode_states(&parquet_path, filter_ep) {
                Ok(states) => {
                    let traj = kin.compute_trajectory(&states, pos_indices);
                    log::debug!(
                        "Computed trajectory for ep {}: {} points",
                        ep_idx,
                        traj.positions.len(),
                    );
                    self.trajectory_cache.insert(ep_idx, traj);
                }
                Err(e) => {
                    log::warn!("Failed to load trajectory for ep {}: {}", ep_idx, e);
                }
            }
        }

        // Build overlay entries
        let mut entries: Vec<TrajectoryEntry> = Vec::new();
        for &(ep_idx, frame) in &pane_episodes {
            if let Some(traj) = self.trajectory_cache.get(ep_idx) {
                entries.push(TrajectoryEntry {
                    trajectory: traj.clone(),
                    episode_index: ep_idx,
                    current_frame: frame,
                    selected: selected_episodes.contains(&ep_idx),
                });
            }
        }

        if entries.is_empty() {
            return;
        }

        // Label
        if self.grid_view.is_some() {
            let sel_text = if selected_episodes.is_empty() {
                "No selection".to_string()
            } else {
                let mut eps: Vec<usize> = selected_episodes.iter().copied().collect();
                eps.sort();
                let ep_strs: Vec<String> = eps.iter().map(|e| format!("{}", e)).collect();
                format!("Selected: ep {}", ep_strs.join(", "))
            };
            ui.label(
                egui::RichText::new(sel_text)
                    .monospace()
                    .color(self.theme.muted),
            );
        } else {
            ui.label(
                egui::RichText::new(format!("Episode {}", self.current_episode))
                    .monospace()
                    .color(self.theme.muted),
            );
        }
        ui.add_space(4.0);

        // Draw overlay
        let accent = self.theme.accent;
        crate::trajectory_view::show_trajectory_overlay(
            ui,
            &entries,
            &mut self.orbit_camera,
            accent,
        );
    }

    fn show_urdf_missing_panel(&mut self, ui: &mut egui::Ui, robot_type_opt: Option<&str>) {
        let robot_type = robot_type_opt.unwrap_or("unknown");
        ui.label(
            egui::RichText::new(format!("No URDF found for \"{}\"", robot_type))
                .color(self.theme.muted),
        );
        ui.add_space(4.0);

        ui.label(
            egui::RichText::new(format!("Expected: {}.urdf", robot_type))
                .monospace()
                .small()
                .color(self.theme.muted),
        );
        ui.label(
            egui::RichText::new("Drop a .urdf file here to install it")
                .small()
                .color(self.theme.muted),
        );
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button("Open robots folder").clicked() {
                if let Some(robots_dir) = crate::trajectory::robots_config_dir() {
                    if let Err(e) = std::fs::create_dir_all(&robots_dir) {
                        log::warn!("Failed to create robots dir: {}", e);
                    }
                    #[cfg(target_os = "linux")]
                    let _ = std::process::Command::new("xdg-open").arg(&robots_dir).spawn();
                    #[cfg(target_os = "macos")]
                    let _ = std::process::Command::new("open").arg(&robots_dir).spawn();
                    #[cfg(target_os = "windows")]
                    let _ = std::process::Command::new("explorer").arg(&robots_dir).spawn();
                }
            }

            if ui.button("Browse...").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("URDF", &["urdf"])
                    .pick_file()
                {
                    match crate::trajectory::install_urdf(&path, robot_type_opt) {
                        Ok(dest) => {
                            log::info!("URDF installed via browse: {}", dest.display());
                            self.reload_kinematics_from(&dest);
                        }
                        Err(e) => {
                            log::warn!("URDF install failed: {}", e);
                            self.loading_error = Some(format!("Failed to install URDF: {}", e));
                        }
                    }
                }
            }
        });
    }
}

/// Interactive grid size picker (like Google Docs table insertion).
/// Displays a 4x4 mini grid. Hovering highlights from (1,1) to (col, row).
/// Clicking returns the selected (cols, rows). Returns None if no click.
fn grid_size_picker(
    ui: &mut egui::Ui,
    current_cols: usize,
    current_rows: usize,
    accent: egui::Color32,
) -> Option<(usize, usize)> {
    let max_cols = 10;
    let max_rows = 10;
    let cell_size = 10.0;
    let cell_gap = 3.0;
    let total_w = max_cols as f32 * (cell_size + cell_gap) - cell_gap;
    let total_h = max_rows as f32 * (cell_size + cell_gap) - cell_gap;

    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(total_w, total_h),
        egui::Sense::click_and_drag(),
    );

    // Determine hover position → (hover_col, hover_row) 1-based
    let (hover_col, hover_row) = if let Some(pos) = response.hover_pos() {
        let rel = pos - rect.min;
        let c = ((rel.x / (cell_size + cell_gap)).floor() as usize + 1).min(max_cols);
        let r = ((rel.y / (cell_size + cell_gap)).floor() as usize + 1).min(max_rows);
        (c, r)
    } else {
        (current_cols, current_rows)
    };

    // Draw cells
    for row in 0..max_rows {
        for col in 0..max_cols {
            let x = rect.min.x + col as f32 * (cell_size + cell_gap);
            let y = rect.min.y + row as f32 * (cell_size + cell_gap);
            let cell_rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(cell_size, cell_size));

            let in_selection = (col + 1) <= hover_col && (row + 1) <= hover_row;
            let color = if in_selection {
                accent
            } else {
                egui::Color32::from_gray(60)
            };

            ui.painter().rect_filled(cell_rect, 2.0, color);
        }
    }

    // Label below
    let label_pos = egui::pos2(rect.min.x, rect.max.y + 2.0);
    ui.painter().text(
        label_pos,
        egui::Align2::LEFT_TOP,
        format!("{}x{}", hover_col, hover_row),
        egui::FontId::monospace(11.0),
        egui::Color32::from_gray(180),
    );
    // Reserve space for the label
    ui.allocate_exact_size(egui::vec2(total_w, 14.0), egui::Sense::hover());

    if response.clicked() {
        Some((hover_col, hover_row))
    } else {
        None
    }
}
