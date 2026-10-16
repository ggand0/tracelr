use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::app::App;
use crate::export::{
    self, AnnotationMeta, CameraSource, ExportEvent, ExportOutput, ExportRequest, FrameRange,
    Layout, ReportMeta, Sink, TrajectoryInput,
};

const TOAST_DURATION: Duration = Duration::from_secs(4);

/// Export window options. Kept for the session, not persisted.
pub(crate) struct ExportDialogState {
    pub open: bool,
    /// One bool per `info.video_keys`.
    pub cameras: Vec<bool>,
    /// Tile order: a permutation of `info.video_keys` indices.
    pub order: Vec<usize>,
    /// Composite row count; None = Auto.
    pub rows: Option<usize>,
    pub include_trajectory: bool,
    pub half_scale: bool,
}

impl ExportDialogState {
    pub fn new() -> Self {
        Self {
            open: false,
            cameras: Vec::new(),
            order: Vec::new(),
            rows: None,
            include_trajectory: true,
            half_scale: false,
        }
    }

    /// Reset camera state when the dataset's camera count changes.
    fn fit_to(&mut self, n_cameras: usize) {
        if self.cameras.len() != n_cameras {
            self.cameras = vec![true; n_cameras];
        }
        if self.order.len() != n_cameras {
            self.order = (0..n_cameras).collect();
        }
    }

    /// `cameras` sorted by the tile order.
    fn in_order(&self, mut cameras: Vec<usize>) -> Vec<usize> {
        cameras.sort_by_key(|c| self.order.iter().position(|o| o == c).unwrap_or(usize::MAX));
        cameras
    }

    /// Checked cameras in tile order.
    fn checked_in_order(&self) -> Vec<usize> {
        self.order
            .iter()
            .copied()
            .filter(|&i| self.cameras.get(i).copied().unwrap_or(false))
            .collect()
    }
}

/// What the viewer currently shows: the episode, the episode-relative frame, and
/// the cameras on screen (indices into `info.video_keys`).
struct ViewTarget {
    episode: usize,
    frame: usize,
    cameras: Vec<usize>,
}

/// Options for one export, taken either from the view or from the dialog.
struct ExportOptions {
    episode: usize,
    /// In tile order.
    cameras: Vec<usize>,
    trajectory: bool,
    scale: f32,
    rows: Option<usize>,
}

impl App {
    fn view_target(&self) -> Option<ViewTarget> {
        let ds = self.dataset.as_ref()?;
        let selected: Vec<usize> = self
            .selected_cameras
            .iter()
            .enumerate()
            .filter_map(|(i, &on)| on.then_some(i))
            .collect();
        let current = vec![self.current_video_key_index.min(ds.info.video_keys.len().saturating_sub(1))];

        let Some(grid) = &self.grid_view else {
            let frame = if self.viewing_video {
                self.current_frame.saturating_sub(self.episode_start_frame)
            } else {
                0
            };
            return Some(ViewTarget { episode: self.current_episode, frame, cameras: current });
        };

        let panes = grid.all_pane_episodes();
        let episode = match grid.mode {
            crate::grid::GridMode::MultiCamera => grid.fixed_episode,
            crate::grid::GridMode::MultiEpisode => grid
                .selected_episodes()
                .into_iter()
                .min()
                .or_else(|| panes.first().map(|&(ep, _)| ep))
                .unwrap_or(self.current_episode),
        };
        let frame = panes
            .iter()
            .find(|&&(ep, _)| ep == episode)
            .map(|&(_, f)| f)
            .unwrap_or(0);
        let cameras = if self.is_camera_grid() && !selected.is_empty() { selected } else { current };
        Some(ViewTarget { episode, frame, cameras })
    }

    /// Whether the trajectory panel can be included in an export at all.
    pub(crate) fn export_trajectory_available(&self) -> bool {
        !self.arms.is_empty() && self.gpu.is_some()
    }

    pub(crate) fn can_start_export(&self) -> bool {
        self.dataset.is_some() && self.export_job.is_none()
    }

    /// Copy the frame on screen (Cmd/Ctrl+C, File > Copy Frame).
    pub(crate) fn copy_current_frame(&mut self, ctx: &egui::Context) {
        self.export_current_frame(ctx, Sink::Clipboard);
    }

    fn save_current_frame_png(&mut self, ctx: &egui::Context) {
        let Some(target) = self.view_target() else { return };
        let name = format!("{}_ep{:04}_f{:04}.png", self.dataset_dir_name(), target.episode, target.frame);
        if let Some(path) = rfd::FileDialog::new().set_file_name(name).save_file() {
            self.export_current_frame(ctx, Sink::Png(with_extension(path, "png")));
        }
    }

    fn export_current_frame(&mut self, ctx: &egui::Context, sink: Sink) {
        if !self.can_start_export() {
            return;
        }
        let Some(target) = self.view_target() else { return };
        // Cameras come from the view; their order and the rows from the Export window.
        let n = self.dataset.as_ref().map(|ds| ds.info.video_keys.len()).unwrap_or(0);
        self.export_dialog.fit_to(n);
        let opts = ExportOptions {
            episode: target.episode,
            cameras: self.export_dialog.in_order(target.cameras),
            trajectory: self.show_trajectory && self.export_trajectory_available(),
            scale: 1.0,
            rows: self.export_dialog.rows,
        };
        self.start_export(ctx, &opts, FrameRange::Single(target.frame), sink);
    }

    /// Open the export window with the cameras on screen preselected.
    pub(crate) fn open_export_dialog(&mut self) {
        let Some(target) = self.view_target() else { return };
        let n = self.dataset.as_ref().map(|ds| ds.info.video_keys.len()).unwrap_or(0);
        self.export_dialog.fit_to(n);
        let mut cameras = vec![false; n];
        for i in target.cameras {
            if let Some(c) = cameras.get_mut(i) {
                *c = true;
            }
        }
        self.export_dialog.cameras = cameras;
        self.export_dialog.open = true;
    }

    fn dialog_options(&self, episode: usize) -> ExportOptions {
        ExportOptions {
            episode,
            cameras: self.export_dialog.checked_in_order(),
            trajectory: self.export_dialog.include_trajectory && self.export_trajectory_available(),
            scale: if self.export_dialog.half_scale { 0.5 } else { 1.0 },
            rows: self.export_dialog.rows,
        }
    }

    fn dataset_dir_name(&self) -> String {
        self.dataset
            .as_ref()
            .and_then(|ds| ds.root.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "dataset".to_string())
    }

    fn start_export(&mut self, ctx: &egui::Context, opts: &ExportOptions, range: FrameRange, sink: Sink) {
        match self.build_export_request(opts, range, sink) {
            Ok(req) => self.export_job = Some(export::spawn(req, ctx.clone())),
            Err(e) => self.show_toast(format!("Export failed: {}", e)),
        }
    }

    fn build_export_request(
        &mut self,
        opts: &ExportOptions,
        range: FrameRange,
        sink: Sink,
    ) -> Result<ExportRequest, String> {
        let ep = opts.episode;
        if opts.cameras.is_empty() {
            return Err("select at least one camera".into());
        }

        // The report writes trajectory.png whenever a trajectory exists; the
        // composite only includes the panel when asked for.
        let is_report = matches!(sink, Sink::Report(_));
        let want_traj = opts.trajectory || (is_report && self.export_trajectory_available());
        let trajectory = if want_traj && self.ensure_trajectory_cached(ep) {
            let gpu = self.gpu.clone();
            let traj = self.trajectory_cache.get(ep).cloned();
            match (traj, gpu) {
                (Some(trajectory), Some(gpu)) => Some(TrajectoryInput {
                    trajectory,
                    episode: ep,
                    camera: self.orbit_camera.clone(),
                    accent: self.theme.accent,
                    gpu,
                }),
                _ => None,
            }
        } else {
            None
        };
        let layout = if opts.trajectory && trajectory.is_some() {
            Layout::CamerasWithTrajectory
        } else {
            Layout::Cameras
        };

        let dataset_name = self.dataset_dir_name();
        let ds = self.dataset.as_ref().ok_or("no dataset loaded")?;
        let ep_meta = ds.episodes.get(ep).ok_or_else(|| format!("episode {} not found", ep))?;
        let is_v3 = ds.info.codebase_version.starts_with("v3");

        let cameras: Vec<CameraSource> = opts
            .cameras
            .iter()
            .filter_map(|&i| ds.info.video_keys.get(i))
            .map(|key| {
                let (from, to) = ds.episode_time_range(ep, key);
                CameraSource {
                    display_name: crate::dataset::camera_display_name(key).to_string(),
                    path: ds.video_path(ep, key),
                    seek_range: (is_v3 && to > from).then_some((from, to)),
                }
            })
            .collect();

        let meta = is_report.then(|| ReportMeta {
            dataset_name,
            dataset_root: ds.root.clone(),
            codebase_version: ds.info.codebase_version.clone(),
            robot_type: ds.info.robot_type.clone(),
            episode: ep,
            task: ep_meta.tasks.join("; "),
            fps: ds.info.fps,
            length: ep_meta.length,
            duration_s: ep_meta.length as f64 / ds.info.fps.max(1) as f64,
            cameras: cameras.iter().map(|c| c.display_name.clone()).collect(),
            curation_labels: self
                .curation
                .episodes
                .get(&ep)
                .map(|labels| labels.iter().cloned().collect())
                .unwrap_or_default(),
            annotation: self
                .annotations
                .get(ep)
                .and_then(|i| self.annotations.prompts.get(i))
                .map(|p| AnnotationMeta { label: p.label.clone(), prompt: p.prompt.clone() }),
            app_version: crate::build_info::BuildInfo::version().to_string(),
            exported_at: export::report::now_rfc3339(),
        });

        Ok(ExportRequest {
            episode_length: ep_meta.length,
            fps: ds.info.fps,
            cameras,
            layout,
            trajectory,
            range,
            sink,
            scale: opts.scale,
            rows: opts.rows,
            meta,
        })
    }

    fn show_toast(&mut self, text: String) {
        self.export_toast = Some((text, Instant::now()));
    }

    /// Drain events from the running export job.
    pub(crate) fn poll_export_job(&mut self) {
        let Some(job) = &mut self.export_job else { return };
        let mut finished = None;
        while let Ok(event) = job.rx.try_recv() {
            match event {
                ExportEvent::Progress { done, total } => job.progress = (done, total),
                ExportEvent::Finished(out) => finished = Some(Ok(out)),
                ExportEvent::Failed(e) => finished = Some(Err(e)),
            }
        }
        let Some(result) = finished else { return };
        self.export_job = None;

        let text = match result {
            Ok(ExportOutput::Image(img)) => match self.set_clipboard_image(&img) {
                Ok(()) => "Frame copied to clipboard".to_string(),
                Err(e) => format!("Copy failed: {}", e),
            },
            Ok(ExportOutput::Saved { path, note }) => {
                log::info!("Exported {}", path.display());
                match note {
                    Some(note) => format!("Saved {} ({})", path.display(), note),
                    None => format!("Saved {}", path.display()),
                }
            }
            Err(e) if export::is_cancelled(&e) => "Export cancelled".to_string(),
            Err(e) => {
                log::error!("Export failed: {}", e);
                format!("Export failed: {}", e)
            }
        };
        self.show_toast(text);
    }

    fn set_clipboard_image(&mut self, img: &egui::ColorImage) -> Result<(), String> {
        if self.clipboard.is_none() {
            self.clipboard = Some(arboard::Clipboard::new().map_err(|e| e.to_string())?);
        }
        let clipboard = self.clipboard.as_mut().expect("clipboard initialised above");
        // Frames are opaque, so premultiplied bytes are the straight RGBA bytes.
        clipboard
            .set_image(arboard::ImageData {
                width: img.size[0],
                height: img.size[1],
                bytes: std::borrow::Cow::Borrowed(img.as_raw()),
            })
            .map_err(|e| e.to_string())
    }

    /// Progress bar + Cancel at the very bottom while an export runs.
    pub(crate) fn show_export_status(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.export_job else { return };
        let (done, total) = job.progress;
        let fraction = if total > 0 { done as f32 / total as f32 } else { 0.0 };
        let label = job.label.clone();
        let cancel = job.cancel.clone();
        egui::TopBottomPanel::bottom("export_status")
            .exact_height(24.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .desired_height(14.0)
                            .text(format!("{} {}/{}", label, done, total)),
                    );
                });
            });
    }

    /// Bottom-right message after an export finishes.
    pub(crate) fn show_export_toast(&mut self, ctx: &egui::Context) {
        let Some((text, shown_at)) = &self.export_toast else { return };
        let elapsed = shown_at.elapsed();
        if elapsed >= TOAST_DURATION {
            self.export_toast = None;
            return;
        }
        ctx.request_repaint_after(TOAST_DURATION - elapsed);
        egui::Area::new(egui::Id::new("export_toast"))
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-12.0, -64.0))
            .order(egui::Order::Foreground)
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(text.as_str());
                });
            });
    }

    /// The Export window opened by File > Export Clip / Export Report.
    pub(crate) fn show_export_dialog(&mut self, ctx: &egui::Context) {
        if !self.export_dialog.open {
            return;
        }
        let Some(ds) = &self.dataset else {
            self.export_dialog.open = false;
            return;
        };
        let names: Vec<String> = ds
            .info
            .video_keys
            .iter()
            .map(|k| crate::dataset::camera_display_name(k).to_string())
            .collect();
        let sizes = ds.info.video_sizes.clone();
        self.export_dialog.fit_to(names.len());
        let Some(target) = self.view_target() else { return };
        let traj_available = self.export_trajectory_available();
        let can_start = self.export_job.is_none();
        let muted = self.theme.muted;
        let accent = self.theme.accent;
        let traj_tip = if self.arms.is_empty() {
            "No URDF loaded for this robot"
        } else {
            "GPU renderer not available"
        };

        let mut open = self.export_dialog.open;
        let mut action: Option<DialogAction> = None;
        egui::Window::new("Export")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                let state = &mut self.export_dialog;
                ui.label(egui::RichText::new(format!("Episode {}", target.episode)).strong());
                ui.add_space(4.0);

                ui.label(egui::RichText::new("Cameras").color(muted).small());
                let checked = state.cameras.iter().filter(|&&c| c).count();
                let mut swap: Option<(usize, usize)> = None;
                let last = state.order.len().saturating_sub(1);
                egui::Grid::new("export_cameras").num_columns(2).show(ui, |ui| {
                    for (pos, &cam) in state.order.iter().enumerate() {
                        let on = &mut state.cameras[cam];
                        // Keep at least one camera checked.
                        let locked = *on && checked == 1;
                        ui.add_enabled(!locked, egui::Checkbox::new(on, names[cam].as_str()));
                        ui.horizontal(|ui| {
                            if arrow_button(ui, true, pos > 0) {
                                swap = Some((pos, pos - 1));
                            }
                            if arrow_button(ui, false, pos < last) {
                                swap = Some((pos, pos + 1));
                            }
                        });
                        ui.end_row();
                    }
                });
                if let Some((a, b)) = swap {
                    state.order.swap(a, b);
                }
                ui.add_enabled(
                    traj_available,
                    egui::Checkbox::new(&mut state.include_trajectory, "Include trajectory panel"),
                )
                .on_disabled_hover_text(traj_tip);
                ui.add_space(4.0);

                let cameras = state.checked_in_order();
                let with_traj = traj_available && state.include_trajectory;
                let cells = cameras.len() + usize::from(with_traj);
                egui::Grid::new("export_options").num_columns(2).show(ui, |ui| {
                    ui.label(egui::RichText::new("Rows").color(muted));
                    let rows = [
                        (None, "Auto", true),
                        (Some(1), "1", cells >= 1),
                        (Some(2), "2", cells >= 2),
                        (Some(3), "3", cells >= 3),
                    ];
                    segmented(ui, "export_rows", &mut state.rows, &rows, accent);
                    ui.end_row();
                    ui.label(egui::RichText::new("Scale").color(muted));
                    let scales = [(false, "Full", true), (true, "Half", true)];
                    segmented(ui, "export_scale", &mut state.half_scale, &scales, accent);
                    ui.end_row();
                });
                ui.add_space(6.0);

                // Camera sizes from info.json; a camera without one borrows the
                // first known size.
                let known = sizes.iter().flatten().copied().next().unwrap_or((640, 480));
                let cam_sizes: Vec<_> = cameras
                    .iter()
                    .map(|&c| sizes.get(c).copied().flatten().unwrap_or(known))
                    .collect();
                let scale = if state.half_scale { 0.5 } else { 1.0 };
                let plan = crate::export::compose::plan(&cam_sizes, with_traj, scale, state.rows);
                let labels: Vec<&str> = cameras.iter().map(|&c| names[c].as_str()).collect();
                layout_preview(ui, &plan, &labels, accent);

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.add_enabled(can_start, egui::Button::new("Copy Frame")).clicked() {
                        action = Some(DialogAction::CopyFrame);
                    }
                    if ui.add_enabled(can_start, egui::Button::new("Save Clip...")).clicked() {
                        action = Some(DialogAction::SaveClip);
                    }
                    if ui.add_enabled(can_start, egui::Button::new("Export Report...")).clicked() {
                        action = Some(DialogAction::ExportReport);
                    }
                });
            });
        self.export_dialog.open = open;

        let opts = self.dialog_options(target.episode);
        match action {
            Some(DialogAction::CopyFrame) => {
                self.start_export(ctx, &opts, FrameRange::Single(target.frame), Sink::Clipboard);
            }
            Some(DialogAction::SaveClip) => {
                let name = format!("{}_ep{:04}.mp4", self.dataset_dir_name(), target.episode);
                if let Some(path) = rfd::FileDialog::new().set_file_name(name).save_file() {
                    self.start_export(ctx, &opts, FrameRange::Whole, Sink::Mp4(with_extension(path, "mp4")));
                }
            }
            Some(DialogAction::ExportReport) => {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.start_export(ctx, &opts, FrameRange::Whole, Sink::Report(dir));
                }
            }
            None => {}
        }
    }

    /// Export entries for the File menu.
    pub(crate) fn show_export_menu_items(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let enabled = self.can_start_export();
        if ui.add_enabled(enabled, egui::Button::new("Copy Frame  Ctrl+C")).clicked() {
            ui.close_menu();
            self.copy_current_frame(ctx);
        }
        if ui.add_enabled(enabled, egui::Button::new("Save Frame as PNG...")).clicked() {
            ui.close_menu();
            self.save_current_frame_png(ctx);
        }
        // Clip and report share the options in the Export window.
        let clip = ui.add_enabled(enabled, egui::Button::new("Export Clip...")).clicked();
        let report = ui.add_enabled(enabled, egui::Button::new("Export Report...")).clicked();
        if clip || report {
            ui.close_menu();
            self.open_export_dialog();
        }
    }
}

/// Small button with a painted up/down triangle (the default fonts have no
/// glyphs for U+25B2 / U+25BC, which render as empty boxes).
fn arrow_button(ui: &mut egui::Ui, up: bool, enabled: bool) -> bool {
    let resp = ui.add_enabled(enabled, egui::Button::new("").min_size(egui::vec2(20.0, 18.0)));
    let color = if enabled {
        ui.style().interact(&resp).fg_stroke.color
    } else {
        ui.visuals().widgets.noninteractive.fg_stroke.color.gamma_multiply(0.35)
    };
    let c = resp.rect.center();
    let (w, h) = (4.5, 3.0);
    let points = if up {
        vec![c + egui::vec2(0.0, -h), c + egui::vec2(w, h), c + egui::vec2(-w, h)]
    } else {
        vec![c + egui::vec2(-w, -h), c + egui::vec2(w, -h), c + egui::vec2(0.0, h)]
    };
    ui.painter()
        .add(egui::Shape::convex_polygon(points, color, egui::Stroke::NONE));
    resp.clicked()
}

/// Pill-shaped segmented control: a rounded track with the selected option
/// filled in the accent colour. Each option is (value, label, enabled).
fn segmented<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    id_salt: &str,
    value: &mut T,
    options: &[(T, &str, bool)],
    accent: egui::Color32,
) {
    let font = egui::TextStyle::Button.resolve(ui.style());
    let pad = egui::vec2(12.0, 3.0);
    let galleys: Vec<_> = options
        .iter()
        .map(|(_, label, _)| {
            ui.painter()
                .layout_no_wrap(label.to_string(), font.clone(), egui::Color32::PLACEHOLDER)
        })
        .collect();
    let height = galleys.iter().map(|g| g.size().y).fold(0.0, f32::max) + 2.0 * pad.y;
    let widths: Vec<f32> = galleys
        .iter()
        .map(|g| (g.size().x + 2.0 * pad.x).max(height * 1.4))
        .collect();
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(widths.iter().sum(), height),
        egui::Sense::hover(),
    );

    let visuals = ui.visuals().clone();
    let painter = ui.painter();
    let radius = (height / 2.0).round().min(255.0) as u8;
    painter.rect_filled(rect, egui::CornerRadius::same(radius), visuals.widgets.inactive.bg_fill);

    let base_id = ui.id().with(id_salt);
    let mut x = rect.min.x;
    for (i, ((option, _, enabled), galley)) in options.iter().zip(galleys).enumerate() {
        let seg = egui::Rect::from_min_size(egui::pos2(x, rect.min.y), egui::vec2(widths[i], height));
        x += widths[i];
        let sense = if *enabled { egui::Sense::click() } else { egui::Sense::hover() };
        let resp = ui.interact(seg, base_id.with(i), sense);
        if *enabled && resp.clicked() {
            *value = *option;
        }

        let selected = *value == *option;
        let pill = seg.shrink(2.0);
        let pill_radius = egui::CornerRadius::same(radius.saturating_sub(2));
        let text_color = if selected {
            painter.rect_filled(pill, pill_radius, accent);
            egui::Color32::from_gray(15)
        } else if !*enabled {
            visuals.widgets.noninteractive.fg_stroke.color.gamma_multiply(0.35)
        } else {
            if resp.hovered() {
                painter.rect_filled(pill, pill_radius, visuals.widgets.hovered.bg_fill);
            }
            visuals.widgets.inactive.fg_stroke.color
        };
        painter.galley(seg.center() - galley.size() / 2.0, galley, text_color);
        if *enabled && !selected {
            resp.on_hover_cursor(egui::CursorIcon::PointingHand);
        }
    }
}

/// Schematic of the composite: labelled boxes for the camera tiles, the
/// trajectory cell in the accent colour, and the output resolution.
fn layout_preview(
    ui: &mut egui::Ui,
    plan: &crate::export::compose::Plan,
    labels: &[&str],
    accent: egui::Color32,
) {
    const MAX: egui::Vec2 = egui::vec2(260.0, 150.0);
    let (cw, ch) = (plan.canvas.0 as f32, plan.canvas.1 as f32);
    let k = (MAX.x / cw).min(MAX.y / ch);
    let (resp, painter) = ui.allocate_painter(egui::vec2(cw * k, ch * k), egui::Sense::hover());
    let origin = resp.rect.min;
    let to_screen = |r: &crate::export::compose::Rect| {
        egui::Rect::from_min_size(
            origin + egui::vec2(r.x as f32, r.y as f32) * k,
            egui::vec2(r.w as f32, r.h as f32) * k,
        )
        .shrink(1.0)
    };
    let font = egui::FontId::proportional(11.0);

    painter.rect_filled(resp.rect, 2.0, egui::Color32::from_gray(12));
    for (tile, label) in plan.tiles.iter().zip(labels) {
        let r = to_screen(tile);
        painter.rect_filled(r, 2.0, egui::Color32::from_gray(48));
        painter.with_clip_rect(r).text(
            r.center(),
            egui::Align2::CENTER_CENTER,
            *label,
            font.clone(),
            egui::Color32::from_gray(210),
        );
    }
    if let Some(traj) = &plan.traj {
        let r = to_screen(traj);
        painter.rect_filled(r, 2.0, accent.gamma_multiply(0.35));
        painter.with_clip_rect(r).text(
            r.center(),
            egui::Align2::CENTER_CENTER,
            "trajectory",
            font.clone(),
            egui::Color32::from_gray(230),
        );
    }

    ui.label(
        egui::RichText::new(format!("{} \u{00d7} {}", plan.canvas.0, plan.canvas.1))
            .small()
            .monospace(),
    );
}

enum DialogAction {
    CopyFrame,
    SaveClip,
    ExportReport,
}

/// Add `ext` when the user typed a name without an extension.
fn with_extension(path: PathBuf, ext: &str) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        path.with_extension(ext)
    }
}
