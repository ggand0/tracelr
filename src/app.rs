use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::annotation::AnnotationState;
use crate::cache::{DecodeLruCache, EpisodeCache, SliderLoader, VideoPlayer};
use crate::dataset::LeRobotDataset;
use crate::grid::GridView;
use crate::perf::PerfTracker;
use crate::theme::UiTheme;
use crate::trajectory::{RobotKinematics, TrajectoryCache};
use crate::trajectory_view::OrbitCamera;

const CACHE_COUNT: usize = 5; // ±5 episodes = 11 total slots
const LRU_CAPACITY: usize = 50;

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

    /// Set to true when navigation changes the selected episode(s),
    /// consumed after one frame to auto-scroll the episode list.
    pub(crate) scroll_to_selected: bool,

    // Trajectory visualization
    pub(crate) robot_kinematics: Option<RobotKinematics>,
    pub(crate) trajectory_cache: TrajectoryCache,
    pub(crate) orbit_camera: OrbitCamera,
    pub(crate) show_trajectory: bool,
    /// Indices into observation.state that are joint positions (for FK).
    pub(crate) state_pos_indices: Vec<usize>,
    /// CLI override for URDF path.
    pub(crate) urdf_override: Option<PathBuf>,

    // Mode
    pub(crate) annotate_mode: bool,

    // UI
    pub(crate) theme: UiTheme,
    pub(crate) perf: PerfTracker,
    pub(crate) initial_size_set: bool,
    pub(crate) show_cache_overlay: bool,
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
            scroll_to_selected: false,
            robot_kinematics: None,
            trajectory_cache: TrajectoryCache::new(100),
            orbit_camera: OrbitCamera::default(),
            show_trajectory: true,
            state_pos_indices: Vec::new(),
            urdf_override,
            annotate_mode: annotate,
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

                let video_key = ds.info.video_keys.get(wrist_idx).cloned().unwrap_or_default();
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

                // Try to load robot kinematics for EE trajectory visualization
                self.robot_kinematics = None;
                self.trajectory_cache = TrajectoryCache::new(100);
                self.state_pos_indices = crate::trajectory::pos_indices_from_state_names(&ds.info.state_names);
                log::info!("State pos indices: {:?} (from {} state names)", self.state_pos_indices, ds.info.state_names.len());

                let urdf_path = self.urdf_override.clone()
                    .filter(|p| p.is_file())
                    .or_else(|| crate::trajectory::discover_urdf(
                        path,
                        ds.info.robot_type.as_deref(),
                    ));
                if let Some(urdf_path) = urdf_path {
                    match RobotKinematics::from_urdf(&urdf_path, None) {
                        Ok(kin) => {
                            log::info!("Robot kinematics loaded: {} (DOF={})", urdf_path.display(), kin.dof());
                            self.robot_kinematics = Some(kin);
                        }
                        Err(e) => {
                            log::warn!("Failed to load kinematics: {}", e);
                        }
                    }
                } else {
                    log::info!("No URDF found for trajectory visualization");
                }

                self.dataset = Some(ds);
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
        self.update_title(ctx);

        // Menu bar
        self.show_menu_bar(ctx);

        let in_grid = self.grid_view.is_some();

        // Left panel: episode list (always visible)
        egui::SidePanel::left("episode_list")
            .default_width(160.0)
            .min_width(120.0)
            .show(ctx, |ui| {
                self.show_episode_list(ctx, ui);
            });

        if !in_grid {
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
        } else {
            // Grid mode: trajectory panel + minimal footer
            if self.show_trajectory && self.robot_kinematics.is_some() {
                egui::SidePanel::right("trajectory_panel")
                    .default_width(280.0)
                    .min_width(200.0)
                    .show(ctx, |ui| {
                        self.show_trajectory_panel(ui);
                    });
            }

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
    }
}
