use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::annotation::AnnotationState;
use crate::cache::{DecodeLruCache, EpisodeCache, SliderLoader, VideoPlayer};
use crate::dataset::LeRobotDataset;
use crate::grid::GridView;
use crate::perf::PerfTracker;
use crate::theme::UiTheme;
use crate::trajectory::{ArmKinematics, RobotKinematics, TrajectoryCache};
use crate::trajectory_view::OrbitCamera;

const CACHE_COUNT: usize = 5; // ±5 episodes = 11 total slots
const LRU_CAPACITY: usize = 50;

/// How cameras are shown in the multi-episode grid.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CameraDisplay {
    /// One camera per pane (default).
    SingleCamera,
    /// Each episode's cameras get their own flat panes, tiled into the grid.
    Tiled,
    /// Each episode pane renders all cameras as a subgrid inside it.
    Subgrid,
}

/// What label to show on each grid pane.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LabelMode {
    /// Compact badge at top-left (ep number only, dark background).
    Compact,
    /// Full label at bottom-left (ep + camera + frame counter, dark background).
    Verbose,
    /// No label.
    Hidden,
}

impl LabelMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::Compact => Self::Verbose,
            Self::Verbose => Self::Hidden,
            Self::Hidden => Self::Compact,
        }
    }
}


pub struct App {
    // Data
    pub(crate) dataset: Option<LeRobotDataset>,
    pub(crate) annotations: AnnotationState,

    // Viewer state
    pub(crate) current_episode: usize,
    pub(crate) current_video_key_index: usize,
    pub(crate) current_texture: Option<egui::TextureHandle>,
    pub(crate) video_paths: Vec<PathBuf>,
    pub(crate) seek_ranges: Vec<Option<(f64, f64)>>,
    pub(crate) loading_error: Option<String>,

    // Episode cache
    pub(crate) episode_cache: Option<EpisodeCache>,
    pub(crate) slider_loader: SliderLoader,
    pub(crate) decode_cache: DecodeLruCache,
    pub(crate) slider_dragging: bool,
    pub(crate) cache_count: usize,

    // Video playback
    pub(crate) viewing_video: bool,
    pub(crate) player: Option<VideoPlayer>,
    pub(crate) current_frame: usize,
    pub(crate) episode_start_frame: usize,
    pub(crate) playing: bool,
    pub(crate) last_frame_time: Option<Instant>,
    pub(crate) frame_slider_dragging: bool,
    pub(crate) last_scrub_seek: Option<Instant>,

    // Grid view
    pub(crate) grid_view: Option<GridView>,
    pub(crate) grid_cols: usize,
    pub(crate) grid_rows: usize,
    /// Which cameras are selected for multi-camera mode (one bool per video_key).
    pub(crate) selected_cameras: Vec<bool>,
    /// How cameras are displayed in multi-episode grid mode.
    pub(crate) camera_display: CameraDisplay,
    /// Label display mode for grid panes.
    pub(crate) label_mode: LabelMode,

    /// Set to true when navigation changes the selected episode(s),
    /// consumed after one frame to auto-scroll the episode list.
    pub(crate) scroll_to_selected: bool,

    // Trajectory visualization
    pub(crate) arms: Vec<ArmKinematics>,
    pub(crate) active_arm_index: usize,
    pub(crate) trajectory_cache: TrajectoryCache,
    pub(crate) orbit_camera: OrbitCamera,
    pub(crate) show_trajectory: bool,
    /// CLI override for URDF path.
    pub(crate) urdf_override: Option<PathBuf>,

    // Mode
    pub(crate) annotate_mode: bool,

    // Pending actions from UI panels (applied in update loop where ctx is available)
    pub(crate) pending_camera_switch: Option<usize>,
    pub(crate) pending_multi_camera_rebuild: bool,

    // UI
    pub(crate) theme: UiTheme,
    pub(crate) perf: PerfTracker,
    pub(crate) initial_size_set: bool,
    pub(crate) show_cache_overlay: bool,
    /// Whether the keyboard shortcut bar is visible below the menu bar.
    pub(crate) show_shortcut_bar: bool,
    /// Whether the About modal is visible.
    pub(crate) show_about: bool,
}

impl App {
    pub fn new(_cc: &eframe::CreationContext, initial_path: Option<PathBuf>, annotate: bool, urdf_override: Option<PathBuf>) -> Self {
        let annotations = if annotate {
            AnnotationState::load_prompts(initial_path.as_deref())
        } else {
            AnnotationState::default_empty()
        };
        let mut app = Self {
            dataset: None,
            annotations,
            current_episode: 0,
            current_video_key_index: 0,
            current_texture: None,
            video_paths: Vec::new(),
            seek_ranges: Vec::new(),
            loading_error: None,
            episode_cache: None,
            slider_loader: SliderLoader::new(),
            decode_cache: DecodeLruCache::new(LRU_CAPACITY),
            slider_dragging: false,
            cache_count: CACHE_COUNT,
            viewing_video: false,
            player: None,
            current_frame: 0,
            episode_start_frame: 0,
            playing: false,
            last_frame_time: None,
            frame_slider_dragging: false,
            last_scrub_seek: None,
            grid_view: None,
            grid_cols: 2,
            grid_rows: 2,
            selected_cameras: Vec::new(),
            camera_display: CameraDisplay::SingleCamera,
            label_mode: LabelMode::Compact,
            scroll_to_selected: false,
            arms: Vec::new(),
            active_arm_index: 0,
            trajectory_cache: TrajectoryCache::new(100),
            orbit_camera: OrbitCamera::default(),
            show_trajectory: true,
            urdf_override,
            pending_camera_switch: None,
            pending_multi_camera_rebuild: false,
            annotate_mode: annotate,
            theme: UiTheme::teal_dark(),
            perf: PerfTracker::new(),
            initial_size_set: false,
            show_cache_overlay: false,
            show_shortcut_bar: false,
            show_about: false,
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

    pub(crate) fn load_dataset(&mut self, path: &std::path::Path) {
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

                // Try to load robot kinematics for EE trajectory visualization
                self.trajectory_cache = TrajectoryCache::new(100);
                self.active_arm_index = 0;

                if let Some(urdf_path) = self.urdf_override.clone().filter(|p| p.is_file()) {
                    match RobotKinematics::from_urdf(&urdf_path, None) {
                        Ok(kin) => {
                            let pos_indices = crate::trajectory::pos_indices_from_state_names(&ds.info.state_names);
                            self.arms = vec![ArmKinematics {
                                name: "default".to_string(),
                                kinematics: kin,
                                pos_indices,
                            }];
                        }
                        Err(e) => {
                            log::warn!("Failed to load kinematics: {}", e);
                            self.arms = Vec::new();
                        }
                    }
                } else {
                    self.arms = crate::trajectory::discover_arms(
                        path,
                        ds.info.robot_type.as_deref(),
                        &ds.info.state_names,
                    );
                }

                if self.arms.is_empty() {
                    log::info!("No URDF found for trajectory visualization");
                } else {
                    for arm in &self.arms {
                        log::info!(
                            "Arm '{}': DOF={}, pos_indices={:?}",
                            arm.name, arm.kinematics.dof(), arm.pos_indices,
                        );
                    }
                }

                self.selected_cameras = vec![true; ds.info.video_keys.len()];
                self.camera_display = CameraDisplay::SingleCamera;
                self.grid_view = None;
                self.dataset = Some(ds);
                self.rebuild_video_paths();
                self.current_episode = 0;
                self.current_texture = None;
                self.loading_error = None;

                if self.annotate_mode {
                    self.annotations = AnnotationState::load_prompts(Some(path));
                    let annot_path = path.join("annotations.json");
                    if annot_path.exists() {
                        if let Err(e) = self.annotations.load_json(&annot_path) {
                            log::warn!("Failed to load annotations: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to load dataset: {}", e);
                self.loading_error = Some(e);
            }
        }
    }

    pub(crate) fn save_annotations(&mut self) {
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

    pub(crate) fn update_title(&self, ctx: &egui::Context) {
        let title = if let Some(ds) = &self.dataset {
            let dir_name = ds
                .root
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if self.annotate_mode {
                let (done, total) = self.annotations.progress(ds.episodes.len());
                format!("tracelr - {} [{}/{}]", dir_name, done, total)
            } else {
                format!("tracelr - {} ({} episodes)", dir_name, ds.episodes.len())
            }
        } else {
            "tracelr".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
    }

    /// Rebuild `video_paths` and `seek_ranges` for the current `current_video_key_index`.
    /// Call this after changing the camera selection.
    pub(crate) fn rebuild_video_paths(&mut self) {
        let ds = match &self.dataset {
            Some(ds) => ds,
            None => return,
        };
        let video_key = ds
            .info
            .video_keys
            .get(self.current_video_key_index)
            .cloned()
            .unwrap_or_default();
        self.video_paths = ds
            .episodes
            .iter()
            .map(|ep| ds.video_path(ep.episode_index, &video_key))
            .collect();
        self.seek_ranges = ds
            .episodes
            .iter()
            .map(|ep| {
                let (from, to) = ds.episode_time_range(ep.episode_index, &video_key);
                if to > from { Some((from, to)) } else { None }
            })
            .collect();
    }

    /// Whether the grid is in a multi-camera mode (MultiCamera grid, or MultiEpisode
    /// with Tiled/Subgrid camera_display). Remains true even when only 1 camera is
    /// selected, so the info panel keeps showing the camera checkboxes.
    pub(crate) fn is_camera_grid(&self) -> bool {
        self.grid_view.as_ref()
            .map(|g| {
                g.mode == crate::grid::GridMode::MultiCamera
                    || self.camera_display != CameraDisplay::SingleCamera
            })
            .unwrap_or(false)
    }

    pub(crate) fn episode_seek_range(&self) -> Option<(f64, f64)> {
        let ds = self.dataset.as_ref()?;
        let vk = ds.info.video_keys.get(self.current_video_key_index)?;
        let (from, to) = ds.episode_time_range(self.current_episode, vk);
        if to > from { Some((from, to)) } else { None }
    }

    pub(crate) fn annotation_json_path(&self) -> Option<String> {
        self.dataset
            .as_ref()
            .map(|ds| ds.root.join("annotations.json").to_string_lossy().to_string())
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

        // Grid mode tick
        if let Some(grid) = &mut self.grid_view {
            grid.tick(ctx);
            self.perf.record_display();
        } else {
            // Single-video mode: poll first frame after seek/init or during scrub
            if self.viewing_video && !self.playing {
                if let Some(player) = &mut self.player {
                    if self.frame_slider_dragging {
                        // During scrub: poll the dedicated scrub decoder
                        if let Some(image) = player.poll_scrub_frame() {
                            let name = format!("scrub_{}", self.current_frame);
                            self.current_texture = Some(ctx.load_texture(
                                name,
                                image,
                                egui::TextureOptions::LINEAR,
                            ));
                            self.perf.record_display();
                        }
                        ctx.request_repaint();
                    } else if self.current_texture.is_none() {
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
        }

        self.handle_dropped_files(ctx);
        self.handle_keyboard(ctx);

        // Apply deferred actions (requested by UI panels that lack ctx)
        if let Some(new_idx) = self.pending_camera_switch.take() {
            self.switch_camera(new_idx, ctx);
        }
        if self.pending_multi_camera_rebuild {
            self.pending_multi_camera_rebuild = false;
            // Capture state before rebuild so position/playing/selection are preserved
            let state = self.grid_view.as_ref().map(|g| g.preserved_state());
            let rebuild_info = self.grid_view.as_ref().map(|g| (g.mode, g.fixed_episode, g.start_episode));
            if let Some((mode, fixed_ep, start_ep)) = rebuild_info {
                match mode {
                    crate::grid::GridMode::MultiCamera => {
                        if let Some(ds) = &self.dataset {
                            let grid = GridView::new_multi_camera(ctx, fixed_ep, ds, &self.selected_cameras);
                            self.grid_view = Some(grid);
                        }
                    }
                    crate::grid::GridMode::MultiEpisode
                        if self.camera_display != CameraDisplay::SingleCamera =>
                    {
                        self.enter_grid_with_camera_display(ctx, start_ep);
                    }
                    _ => {}
                }

                if let (Some(grid), Some(state)) = (&mut self.grid_view, &state) {
                    grid.restore_state(state);
                }
            }
        }

        self.update_title(ctx);

        // Menu bar and shortcut bar
        self.show_menu_bar(ctx);
        self.show_shortcut_bar(ctx);

        let in_grid = self.grid_view.is_some();

        // Left panel: episode list (always visible)
        egui::SidePanel::left("episode_list")
            .default_width(160.0)
            .min_width(120.0)
            .show(ctx, |ui| {
                self.show_episode_list(ctx, ui);
            });

        // Right side panel:
        // - Single-video mode: Episode Info panel (with camera ComboBox, trajectory, etc.)
        // - Any grid mode: Cameras panel (camera widget + trajectory), no Episode Info
        if !in_grid {
            egui::SidePanel::right("info_panel")
                .default_width(200.0)
                .min_width(160.0)
                .show(ctx, |ui| {
                    self.show_info_panel(ui);
                });
        } else {
            egui::SidePanel::right("cameras_panel")
                .default_width(280.0)
                .min_width(200.0)
                .show(ctx, |ui| {
                    self.show_cameras_panel(ui);
                });
        }

        if !in_grid {
            // Single-video mode: footer + slider
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
        } else {
            // Grid mode: footer + slider (trajectory is inside the cameras panel)
            egui::TopBottomPanel::bottom("grid_footer")
                .exact_height(22.0)
                .show(ctx, |ui| {
                    self.show_grid_footer(ui);
                });

            egui::TopBottomPanel::bottom("grid_slider")
                .exact_height(28.0)
                .show(ctx, |ui| {
                    self.show_grid_frame_slider(ctx, ui);
                });
        }

        // Central panel
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.loading_error {
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), err);
                ui.separator();
            }
            if self.grid_view.is_some() {
                self.show_grid_display(ui);
            } else {
                self.show_frame_display(ui);
            }
        });

        // Cache debug overlay
        if self.show_cache_overlay {
            if let Some(cache) = &self.episode_cache {
                cache.show_debug_overlay(ctx, self.current_episode, self.video_paths.len());
            }
        }

        // About modal (rendered last so it sits on top)
        crate::about::show_about_modal(ctx, &mut self.show_about, &self.theme);
    }
}
