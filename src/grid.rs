use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::cache::VideoPlayer;

const PANE_SPACING: f32 = 4.0;
const PANE_LABEL_HEIGHT: f32 = 18.0;
const PANE_LABEL_FONT_SIZE: f32 = 11.0;
const PANE_BORDER_RADIUS: f32 = 2.0;
const PANE_BORDER_WIDTH: f32 = 2.0;

/// Preserved state captured before a grid rebuild and restored after.
/// Keeps per-pane frame positions, playing state, and pane selection
/// stable across rebuilds (camera switch, checkbox toggle, etc.).
pub(crate) struct PreservedGridState {
    pub relative_frames: Vec<usize>,
    pub playing: bool,
    pub selected_panes: HashSet<usize>,
}

/// What the grid is displaying.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GridMode {
    /// One camera, multiple episodes (default grid view).
    /// When `cam_count > 1`, panes are tiled with camera groups per episode.
    MultiEpisode,
    /// One episode, multiple cameras.
    MultiCamera,
}

/// Dataset context needed to create grid panes.
pub(crate) struct GridDataset<'a> {
    pub video_paths: &'a [PathBuf],
    pub seek_ranges: &'a [Option<(f64, f64)>],
    pub episodes: &'a [crate::dataset::EpisodeMeta],
    pub fps: u32,
}

/// A single pane in the grid, displaying one episode's video.
struct GridPane {
    episode_index: usize,
    /// Which camera this pane shows (e.g. "observation.images.wrist").
    #[allow(dead_code)]
    video_key: String,
    /// Display label for the pane (e.g. "ep 001" or "wrist").
    label: String,
    player: VideoPlayer,
    current_texture: Option<egui::TextureHandle>,
    current_frame: usize,
    episode_start_frame: usize,
    total_frames: usize,
}

/// Grid view: displays multiple episodes simultaneously in a cols x rows layout.
/// Each pane has its own VideoPlayer with an independent decode thread.
pub(crate) struct GridView {
    panes: Vec<GridPane>,
    pub cols: usize,
    pub rows: usize,
    pub mode: GridMode,
    /// First episode index shown in the grid (top-left pane).
    pub start_episode: usize,
    /// Fixed episode for MultiCamera mode.
    pub fixed_episode: usize,
    /// Number of cameras per episode when tiled/subgrid (1 = single camera).
    pub cam_count: usize,
    /// If true, panes are rendered grouped inside episode cells (subgrid layout).
    /// If false, each pane is a flat cell in the grid (tiled layout).
    pub subgrid: bool,
    pub playing: bool,
    last_frame_time: Option<Instant>,
    fps: u32,
    /// Selected panes (for highlighting). Empty = no selection.
    pub selected_panes: HashSet<usize>,
    /// Frame slider drag state for grid scrubbing.
    pub frame_slider_dragging: bool,
}

impl GridView {
    /// Create a new multi-episode grid starting at `start_episode` with `cols x rows` panes.
    pub fn new(
        ctx: &egui::Context,
        cols: usize,
        rows: usize,
        start_episode: usize,
        ds: &GridDataset,
    ) -> Self {
        let total_panes = cols * rows;
        let mut panes = Vec::with_capacity(total_panes);

        for i in 0..total_panes {
            let ep_idx = start_episode + i;
            if ep_idx >= ds.video_paths.len() {
                break;
            }
            let video_path = match ds.video_paths.get(ep_idx) {
                Some(p) => p,
                None => continue,
            };
            let ep = match ds.episodes.get(ep_idx) {
                Some(ep) => ep,
                None => continue,
            };
            let seek_range = ds.seek_ranges.get(ep_idx).and_then(|r| *r);
            let label = format!("ep {:03}", ep_idx);
            if let Some(pane) = Self::create_pane(ctx, ep_idx, "", &label, video_path, ep.length, seek_range, ds.fps) {
                panes.push(pane);
            }
        }

        Self {
            panes,
            cols,
            rows,
            mode: GridMode::MultiEpisode,
            start_episode,
            fixed_episode: start_episode,
            cam_count: 1,
            subgrid: false,
            playing: true,
            last_frame_time: None,
            fps: ds.fps,
            selected_panes: HashSet::new(),
            frame_slider_dragging: false,
        }
    }

    /// Create a new multi-camera grid showing all selected cameras for a single episode.
    pub fn new_multi_camera(
        ctx: &egui::Context,
        episode_index: usize,
        ds: &crate::dataset::LeRobotDataset,
        selected_cameras: &[bool],
    ) -> Self {
        let ep = match ds.episodes.get(episode_index) {
            Some(ep) => ep,
            None => {
                return Self {
                    panes: Vec::new(),
                    cols: 1,
                    rows: 1,
                    mode: GridMode::MultiCamera,
                    start_episode: episode_index,
                    fixed_episode: episode_index,
                    cam_count: 1,
                    subgrid: false,
                    playing: true,
                    last_frame_time: None,
                    fps: ds.info.fps,
                    selected_panes: HashSet::new(),
                    frame_slider_dragging: false,
                };
            }
        };

        let selected_keys = selected_video_keys(&ds.info.video_keys, selected_cameras);

        let cam_count = selected_keys.len();
        let (cols, rows) = camera_grid_size(cam_count);

        let mut panes = Vec::with_capacity(cam_count);
        for &video_key in &selected_keys {
            let video_path = ds.video_path(episode_index, video_key);
            let (from, to) = ds.episode_time_range(episode_index, video_key);
            let seek_range = if to > from { Some((from, to)) } else { None };
            let display_name = crate::dataset::camera_display_name(video_key);
            if let Some(pane) = Self::create_pane(
                ctx, episode_index, video_key, display_name,
                &video_path, ep.length, seek_range, ds.info.fps,
            ) {
                panes.push(pane);
            }
        }

        Self {
            panes,
            cols,
            rows,
            mode: GridMode::MultiCamera,
            start_episode: episode_index,
            fixed_episode: episode_index,
            cam_count,
            subgrid: false,
            playing: true,
            last_frame_time: None,
            fps: ds.info.fps,
            selected_panes: HashSet::new(),
            frame_slider_dragging: false,
        }
    }

    /// Create a tiled multi-episode grid where each episode gets `cam_count` panes.
    /// Cols snap to `cam_count * groups_per_row` where groups_per_row = max(1, grid_cols / cam_count).
    pub fn new_tiled(
        ctx: &egui::Context,
        grid_cols: usize,
        grid_rows: usize,
        start_episode: usize,
        ds: &crate::dataset::LeRobotDataset,
        selected_cameras: &[bool],
    ) -> Self {
        let selected_keys = selected_video_keys(&ds.info.video_keys, selected_cameras);
        let cam_count = selected_keys.len().max(1);
        let groups_per_row = (grid_cols / cam_count).max(1);
        let actual_cols = cam_count * groups_per_row;
        let rows = grid_rows;
        let mut panes = Vec::with_capacity(rows * actual_cols);
        for row in 0..rows {
            for group in 0..groups_per_row {
                let ep_idx = start_episode + row * groups_per_row + group;
                let ep = match ds.episodes.get(ep_idx) {
                    Some(ep) => ep,
                    None => break,
                };
                for &video_key in &selected_keys {
                    let video_path = ds.video_path(ep_idx, video_key);
                    let (from, to) = ds.episode_time_range(ep_idx, video_key);
                    let seek_range = if to > from { Some((from, to)) } else { None };
                    let cam_name = crate::dataset::camera_display_name(video_key);
                    let label = format!("ep {:03} {}", ep_idx, cam_name);
                    if let Some(pane) = Self::create_pane(
                        ctx, ep_idx, video_key, &label,
                        &video_path, ep.length, seek_range, ds.info.fps,
                    ) {
                        panes.push(pane);
                    }
                }
            }
        }

        Self {
            panes,
            cols: actual_cols,
            rows,
            mode: GridMode::MultiEpisode,
            start_episode,
            fixed_episode: start_episode,
            cam_count,
            subgrid: false,
            playing: true,
            last_frame_time: None,
            fps: ds.info.fps,
            selected_panes: HashSet::new(),
            frame_slider_dragging: false,
        }
    }

    /// Create a subgrid multi-episode grid where each episode cell contains cam_count sub-panes.
    /// Outer grid is `grid_cols × grid_rows` (user's layout). Total panes = episodes × cam_count.
    pub fn new_subgrid(
        ctx: &egui::Context,
        grid_cols: usize,
        grid_rows: usize,
        start_episode: usize,
        ds: &crate::dataset::LeRobotDataset,
        selected_cameras: &[bool],
    ) -> Self {
        let selected_keys = selected_video_keys(&ds.info.video_keys, selected_cameras);
        let cam_count = selected_keys.len().max(1);
        let total_cells = grid_cols * grid_rows;
        let mut panes = Vec::with_capacity(total_cells * cam_count);

        for cell in 0..total_cells {
            let ep_idx = start_episode + cell;
            let ep = match ds.episodes.get(ep_idx) {
                Some(ep) => ep,
                None => break,
            };
            for &video_key in &selected_keys {
                let video_path = ds.video_path(ep_idx, video_key);
                let (from, to) = ds.episode_time_range(ep_idx, video_key);
                let seek_range = if to > from { Some((from, to)) } else { None };
                let cam_name = crate::dataset::camera_display_name(video_key);
                let label = format!("ep {:03} {}", ep_idx, cam_name);
                if let Some(pane) = Self::create_pane(
                    ctx, ep_idx, video_key, &label,
                    &video_path, ep.length, seek_range, ds.info.fps,
                ) {
                    panes.push(pane);
                }
            }
        }

        Self {
            panes,
            cols: grid_cols,
            rows: grid_rows,
            mode: GridMode::MultiEpisode,
            start_episode,
            fixed_episode: start_episode,
            cam_count,
            subgrid: true,
            playing: true,
            last_frame_time: None,
            fps: ds.info.fps,
            selected_panes: HashSet::new(),
            frame_slider_dragging: false,
        }
    }

    /// Number of episodes visible when tiled.
    pub fn episodes_shown(&self) -> usize {
        if self.cam_count > 1 {
            self.panes.len() / self.cam_count
        } else {
            self.panes.len()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_pane(
        ctx: &egui::Context,
        episode_index: usize,
        video_key: &str,
        label: &str,
        video_path: &std::path::Path,
        total_frames: usize,
        seek_range: Option<(f64, f64)>,
        fps: u32,
    ) -> Option<GridPane> {
        let from_ts = seek_range.map(|(from, _)| from).unwrap_or(0.0);
        let start_frame = if from_ts > 0.0 {
            (from_ts * fps as f64) as usize
        } else {
            0
        };

        let player_total = start_frame + total_frames;
        let player = VideoPlayer::new(ctx, video_path, player_total, fps, start_frame);

        Some(GridPane {
            episode_index,
            video_key: video_key.to_string(),
            label: label.to_string(),
            player,
            current_texture: None,
            current_frame: start_frame,
            episode_start_frame: start_frame,
            total_frames,
        })
    }

    /// Number of panes actually active (may be less than cols*rows near end of dataset).
    pub fn pane_count(&self) -> usize {
        self.panes.len()
    }

    /// Advance all panes by one frame if enough time has elapsed.
    pub fn tick(&mut self, ctx: &egui::Context) {
        if !self.playing {
            if self.frame_slider_dragging {
                // Poll scrub frames from all panes
                for pane in &mut self.panes {
                    if let Some(image) = pane.player.poll_scrub_frame() {
                        let name = format!("grid_scrub_{}_{}", pane.episode_index, pane.current_frame);
                        pane.current_texture = Some(ctx.load_texture(
                            name,
                            image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
                ctx.request_repaint();
            } else {
                // Still poll first frames for panes that haven't received one yet
                for pane in &mut self.panes {
                    if pane.current_texture.is_none() {
                        if let Some(tex) = pane.player.poll_next_frame() {
                            pane.current_frame = pane.player.current_frame;
                            pane.current_texture = Some(tex);
                        }
                    }
                }
            }
            return;
        }

        let frame_duration = std::time::Duration::from_secs_f64(1.0 / self.fps as f64);
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

        let mut any_active = false;
        for pane in &mut self.panes {
            if pane.current_frame + 1 >= pane.episode_start_frame + pane.total_frames {
                continue; // this pane's episode is done
            }
            any_active = true;
            if let Some(tex) = pane.player.poll_next_frame() {
                pane.current_frame = pane.player.current_frame;
                pane.current_texture = Some(tex);
            }
        }

        if !any_active {
            self.playing = false;
            return;
        }

        self.last_frame_time = Some(now);
        ctx.request_repaint();
    }

    /// Toggle play/pause for all panes.
    /// If all panes have finished, restart from the beginning.
    pub fn toggle_playing(&mut self) {
        if !self.playing {
            // Check if all panes are at the end — if so, seek to start (replay)
            let all_done = self.panes.iter().all(|p| {
                p.current_frame + 1 >= p.episode_start_frame + p.total_frames
            });
            if all_done && !self.panes.is_empty() {
                for pane in &mut self.panes {
                    pane.player.seek(pane.episode_start_frame);
                    pane.current_frame = pane.episode_start_frame;
                    pane.current_texture = None;
                }
            }
        }

        self.playing = !self.playing;
        if self.playing {
            self.last_frame_time = Some(Instant::now());
        } else {
            self.last_frame_time = None;
        }
    }

    /// Rebuild all panes from `self.start_episode` using current cols/rows (multi-episode only).
    /// Preserves `playing` state; callers that want to force playback should set it explicitly.
    fn rebuild(&mut self, ctx: &egui::Context, ds: &GridDataset) {
        self.panes.clear();
        self.selected_panes.clear();

        let total = ds.video_paths.len();
        for i in 0..(self.cols * self.rows) {
            let ep_idx = self.start_episode + i;
            if ep_idx >= total {
                break;
            }
            let video_path = match ds.video_paths.get(ep_idx) {
                Some(p) => p,
                None => continue,
            };
            let ep = match ds.episodes.get(ep_idx) {
                Some(ep) => ep,
                None => continue,
            };
            let seek_range = ds.seek_ranges.get(ep_idx).and_then(|r| *r);
            let label = format!("ep {:03}", ep_idx);
            if let Some(pane) = Self::create_pane(ctx, ep_idx, "", &label, video_path, ep.length, seek_range, ds.fps) {
                self.panes.push(pane);
            }
        }

        self.last_frame_time = None;
    }

    /// Shift the grid forward or backward by `cols * rows` episodes.
    pub fn navigate_page(&mut self, delta: isize, ctx: &egui::Context, ds: &GridDataset) {
        let page_size = self.cols * self.rows;
        let total = ds.video_paths.len();
        if total == 0 {
            return;
        }

        let new_start = if delta > 0 {
            let s = self.start_episode + page_size;
            if s >= total { return; }
            s
        } else {
            if self.start_episode == 0 { return; }
            self.start_episode.saturating_sub(page_size)
        };

        self.start_episode = new_start;
        self.rebuild(ctx, ds);
    }

    /// Jump to a specific start episode, rebuilding all panes.
    pub fn jump_to(&mut self, start: usize, ctx: &egui::Context, ds: &GridDataset) {
        let total = ds.video_paths.len();
        self.start_episode = start.min(total.saturating_sub(1));
        self.rebuild(ctx, ds);
    }

    /// Resize the grid (change cols/rows) and rebuild panes from current start_episode.
    pub fn resize(&mut self, cols: usize, rows: usize, ctx: &egui::Context, ds: &GridDataset) {
        self.cols = cols;
        self.rows = rows;
        self.rebuild(ctx, ds);
    }

    /// Page episodes in tiled/subgrid mode (shift by episodes_shown).
    pub fn navigate_page_tiled(
        &mut self,
        delta: isize,
        ctx: &egui::Context,
        grid_cols: usize,
        grid_rows: usize,
        ds: &crate::dataset::LeRobotDataset,
        selected_cameras: &[bool],
    ) {
        let total = ds.episodes.len();
        if total == 0 {
            return;
        }
        let eps_shown = self.episodes_shown();
        let new_start = if delta > 0 {
            let s = self.start_episode + eps_shown;
            if s >= total { return; }
            s
        } else {
            if self.start_episode == 0 { return; }
            self.start_episode.saturating_sub(eps_shown)
        };
        if self.subgrid {
            *self = Self::new_subgrid(ctx, grid_cols, grid_rows, new_start, ds, selected_cameras);
        } else {
            *self = Self::new_tiled(ctx, grid_cols, grid_rows, new_start, ds, selected_cameras);
        }
    }

    /// Render the grid into the given UI area.
    pub fn show(&mut self, ui: &mut egui::Ui, theme_accent: egui::Color32, label_mode: crate::app::LabelMode) {
        if self.subgrid {
            self.show_subgrid(ui, theme_accent, label_mode);
        } else {
            self.show_flat(ui, theme_accent, label_mode);
        }
    }

    /// Flat rendering: each pane is its own cell in the grid.
    fn show_flat(&mut self, ui: &mut egui::Ui, theme_accent: egui::Color32, label_mode: crate::app::LabelMode) {
        let available = ui.available_size();
        let cols = self.cols;
        let rows = self.rows;
        let cam_count = self.cam_count;

        let pane_w = (available.x - PANE_SPACING * (cols as f32 - 1.0)) / cols as f32;
        let pane_h = (available.y - PANE_SPACING * (rows as f32 - 1.0)) / rows as f32;

        let origin = ui.cursor().min;
        let mut clicked_episode: Option<usize> = None;
        // Collect rects for grouped selection border (tiled multi-cam)
        let mut pane_rects: Vec<egui::Rect> = Vec::with_capacity(self.panes.len());

        for (idx, pane) in self.panes.iter().enumerate() {
            let col = idx % cols;
            let row = idx / cols;

            let x = origin.x + col as f32 * (pane_w + PANE_SPACING);
            let y = origin.y + row as f32 * (pane_h + PANE_SPACING);
            let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(pane_w, pane_h));
            pane_rects.push(rect);

            let response = ui.allocate_rect(rect, egui::Sense::click());

            if self.mode == GridMode::MultiEpisode && response.clicked() {
                clicked_episode = Some(pane.episode_index);
            }

            let is_selected = self.selected_panes.contains(&idx);
            let bg = if is_selected { egui::Color32::from_gray(50) } else { egui::Color32::from_gray(30) };
            ui.painter().rect_filled(rect, PANE_BORDER_RADIUS, bg);

            Self::draw_pane_frame(ui, pane, rect, pane_w, pane_h);

            Self::draw_pane_label(ui, pane, rect, is_selected, label_mode, theme_accent);

            // Per-pane border only in single-camera mode
            if is_selected && cam_count <= 1 {
                ui.painter().rect_stroke(rect, PANE_BORDER_RADIUS, egui::Stroke::new(PANE_BORDER_WIDTH, theme_accent), egui::StrokeKind::Outside);
            }
        }

        // Grouped selection border: one border per episode spanning all its camera panes
        if cam_count > 1 {
            let mut drawn_episodes: HashSet<usize> = HashSet::new();
            for &idx in &self.selected_panes {
                let ep = match self.panes.get(idx) {
                    Some(p) => p.episode_index,
                    None => continue,
                };
                if !drawn_episodes.insert(ep) {
                    continue;
                }
                // Find bounding rect of all panes for this episode
                let mut group_rect: Option<egui::Rect> = None;
                for (i, pane) in self.panes.iter().enumerate() {
                    if pane.episode_index == ep {
                        let r = pane_rects[i];
                        group_rect = Some(match group_rect {
                            Some(acc) => acc.union(r),
                            None => r,
                        });
                    }
                }
                if let Some(r) = group_rect {
                    ui.painter().rect_stroke(r, PANE_BORDER_RADIUS, egui::Stroke::new(PANE_BORDER_WIDTH, theme_accent), egui::StrokeKind::Outside);
                }
            }
        }

        if let Some(ep) = clicked_episode {
            self.toggle_episode_selection(ep);
        }
    }

    /// Subgrid rendering: each episode cell contains cam_count sub-panes.
    fn show_subgrid(&mut self, ui: &mut egui::Ui, theme_accent: egui::Color32, label_mode: crate::app::LabelMode) {
        let available = ui.available_size();
        let cols = self.cols;
        let rows = self.rows;
        let cam_count = self.cam_count;

        let cell_w = (available.x - PANE_SPACING * (cols as f32 - 1.0)) / cols as f32;
        let cell_h = (available.y - PANE_SPACING * (rows as f32 - 1.0)) / rows as f32;

        let (sub_cols, sub_rows) = camera_grid_size(cam_count);
        let sub_spacing = 2.0_f32;

        let origin = ui.cursor().min;
        let ep_count = self.panes.len() / cam_count.max(1);

        // Subtle border color for cell separation (slightly brighter than bg)
        let cell_border = egui::Color32::from_gray(70);

        let mut clicked_episode: Option<usize> = None;

        for cell_idx in 0..ep_count {
            let cell_col = cell_idx % cols;
            let cell_row = cell_idx / cols;
            if cell_row >= rows { break; }

            let cx = origin.x + cell_col as f32 * (cell_w + PANE_SPACING);
            let cy = origin.y + cell_row as f32 * (cell_h + PANE_SPACING);
            let cell_rect = egui::Rect::from_min_size(egui::pos2(cx, cy), egui::vec2(cell_w, cell_h));

            let response = ui.allocate_rect(cell_rect, egui::Sense::click());
            let first_pane_idx = cell_idx * cam_count;
            let episode_index = self.panes.get(first_pane_idx).map(|p| p.episode_index);

            if response.clicked() {
                if let Some(ep) = episode_index {
                    clicked_episode = Some(ep);
                }
            }

            let is_selected = self.selected_panes.contains(&first_pane_idx);
            let bg = if is_selected { egui::Color32::from_gray(50) } else { egui::Color32::from_gray(30) };
            ui.painter().rect_filled(cell_rect, PANE_BORDER_RADIUS, bg);

            // Sub-pane area (reserve bottom space only for verbose labels)
            let label_reserve = if label_mode == crate::app::LabelMode::Verbose { PANE_LABEL_HEIGHT } else { 0.0 };
            let content_h = cell_h - label_reserve;
            let sub_w = (cell_w - sub_spacing * (sub_cols as f32 - 1.0)) / sub_cols as f32;
            let sub_h = (content_h - sub_spacing * (sub_rows as f32 - 1.0)) / sub_rows as f32;

            for cam_idx in 0..cam_count {
                let pane_idx = cell_idx * cam_count + cam_idx;
                let pane = match self.panes.get(pane_idx) {
                    Some(p) => p,
                    None => break,
                };

                let sc = cam_idx % sub_cols;
                let sr = cam_idx / sub_cols;
                let sx = cx + sc as f32 * (sub_w + sub_spacing);
                let sy = cy + sr as f32 * (sub_h + sub_spacing);
                let sub_rect = egui::Rect::from_min_size(egui::pos2(sx, sy), egui::vec2(sub_w, sub_h));

                Self::draw_pane_frame(ui, pane, sub_rect, sub_w, sub_h);
            }

            // Episode label for the cell
            let first_pane = &self.panes[first_pane_idx];
            Self::draw_pane_label(ui, first_pane, cell_rect, is_selected, label_mode, theme_accent);

            // Cell boundary: subtle border always visible to separate episodes
            let stroke = if is_selected {
                egui::Stroke::new(PANE_BORDER_WIDTH, theme_accent)
            } else {
                egui::Stroke::new(1.0, cell_border)
            };
            ui.painter().rect_stroke(cell_rect, PANE_BORDER_RADIUS, stroke, egui::StrokeKind::Outside);
        }

        if let Some(ep) = clicked_episode {
            self.toggle_episode_selection(ep);
        }
    }

    /// Toggle selection of all panes for an episode. Used by multi-cam click handlers.
    fn toggle_episode_selection(&mut self, episode_index: usize) {
        // Find all pane indices with this episode_index
        let matching: Vec<usize> = self.panes.iter().enumerate()
            .filter(|(_, p)| p.episode_index == episode_index)
            .map(|(i, _)| i)
            .collect();
        if matching.is_empty() {
            return;
        }
        // If any of the matching panes are selected, deselect all; otherwise select all
        let any_selected = matching.iter().any(|i| self.selected_panes.contains(i));
        if any_selected {
            for i in matching {
                self.selected_panes.remove(&i);
            }
        } else {
            for i in matching {
                self.selected_panes.insert(i);
            }
        }
    }

    /// Draw a video frame texture into a rect, centered and scaled to fit.
    fn draw_pane_frame(ui: &egui::Ui, pane: &GridPane, rect: egui::Rect, w: f32, h: f32) {
        if let Some(tex) = &pane.current_texture {
            let tex_size = tex.size_vec2();
            let img_rect = egui::Rect::from_min_size(rect.min, egui::vec2(w, h));
            let scale = (img_rect.width() / tex_size.x)
                .min(img_rect.height() / tex_size.y)
                .min(1.0);
            let display_size = tex_size * scale;
            let img_pos = egui::pos2(
                img_rect.center().x - display_size.x / 2.0,
                img_rect.center().y - display_size.y / 2.0,
            );
            let img_draw_rect = egui::Rect::from_min_size(img_pos, display_size);
            ui.painter().image(
                tex.id(), img_draw_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        }
    }

    /// Draw a label on a pane or cell according to the current label mode.
    fn draw_pane_label(
        ui: &egui::Ui,
        pane: &GridPane,
        rect: egui::Rect,
        is_selected: bool,
        label_mode: crate::app::LabelMode,
        theme_accent: egui::Color32,
    ) {
        let badge_bg = egui::Color32::from_black_alpha(160);
        let badge_fg = egui::Color32::from_gray(230);
        let font = egui::FontId::monospace(PANE_LABEL_FONT_SIZE);
        let pad = 3.0_f32;

        match label_mode {
            crate::app::LabelMode::Compact => {
                let text = format!("ep {:03}", pane.episode_index);
                let galley = ui.painter().layout_no_wrap(text, font, badge_fg);
                let text_size = galley.size();
                let bg_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.min.x, rect.min.y),
                    egui::vec2(text_size.x + pad * 2.0, text_size.y + pad * 2.0),
                );
                ui.painter().rect_filled(bg_rect, PANE_BORDER_RADIUS, badge_bg);
                ui.painter().galley(
                    egui::pos2(rect.min.x + pad, rect.min.y + pad),
                    galley,
                    badge_fg,
                );
            }
            crate::app::LabelMode::Verbose => {
                let ep_frame = pane.current_frame.saturating_sub(pane.episode_start_frame);
                let text = format!("{}  {}/{}", pane.label, ep_frame + 1, pane.total_frames);
                let text_color = if is_selected { theme_accent } else { badge_fg };
                let galley = ui.painter().layout_no_wrap(text, font, text_color);
                let text_size = galley.size();
                let label_x = rect.min.x + PANE_SPACING;
                let label_y = rect.max.y - PANE_LABEL_HEIGHT + 2.0;
                let bg_rect = egui::Rect::from_min_size(
                    egui::pos2(label_x - pad, label_y - pad),
                    egui::vec2(text_size.x + pad * 2.0, text_size.y + pad * 2.0),
                );
                ui.painter().rect_filled(bg_rect, PANE_BORDER_RADIUS, badge_bg);
                ui.painter().galley(
                    egui::pos2(label_x, label_y),
                    galley,
                    text_color,
                );
            }
            crate::app::LabelMode::Hidden => {}
        }
    }

    /// Get the episode indices of all selected panes.
    pub fn selected_episodes(&self) -> HashSet<usize> {
        self.selected_panes
            .iter()
            .filter_map(|&idx| self.panes.get(idx).map(|p| p.episode_index))
            .collect()
    }

    /// Get all pane episode indices and their current frame (relative to episode start).
    pub fn all_pane_episodes(&self) -> Vec<(usize, usize)> {
        self.panes
            .iter()
            .map(|p| (p.episode_index, p.current_frame.saturating_sub(p.episode_start_frame)))
            .collect()
    }

    /// Seek each pane to a preserved relative frame position. Used by camera
    /// switching to preserve per-pane playback position across grid rebuild.
    /// `relative_frames` should be in the same order as panes (pane-index-aligned).
    /// Drains intermediate keyframe frames so the displayed frame matches the target.
    pub fn seek_panes_to_relative(&mut self, relative_frames: &[usize]) {
        for (pane, &rel) in self.panes.iter_mut().zip(relative_frames.iter()) {
            let clamped = rel.min(pane.total_frames.saturating_sub(1));
            let abs_frame = pane.episode_start_frame + clamped;
            pane.player.seek(abs_frame);
            if let Some(tex) = pane.player.drain_to_frame(abs_frame, 200) {
                pane.current_texture = Some(tex);
            }
            pane.current_frame = abs_frame;
        }
    }

    /// Capture preserved state (per-pane frames, playing, selection) before rebuild.
    pub fn preserved_state(&self) -> PreservedGridState {
        PreservedGridState {
            relative_frames: self.panes.iter()
                .map(|p| p.current_frame.saturating_sub(p.episode_start_frame))
                .collect(),
            playing: self.playing,
            selected_panes: self.selected_panes.clone(),
        }
    }

    /// Restore preserved state after rebuild. If the new pane count differs from
    /// the captured state, all panes are seeked to the first captured frame
    /// (the synchronized playback position).
    pub fn restore_state(&mut self, state: &PreservedGridState) {
        let frames = if state.relative_frames.len() == self.panes.len() {
            state.relative_frames.clone()
        } else {
            let sync = state.relative_frames.first().copied().unwrap_or(0);
            vec![sync; self.panes.len()]
        };
        self.seek_panes_to_relative(&frames);
        self.playing = state.playing;
        self.selected_panes = state.selected_panes.clone();
    }

    /// Maximum episode length across all panes (for slider range).
    pub fn max_episode_length(&self) -> usize {
        self.panes.iter().map(|p| p.total_frames).max().unwrap_or(0)
    }

    /// Current relative frame of the first pane (for slider position during playback).
    pub fn current_relative_frame(&self) -> usize {
        self.panes.first()
            .map(|p| p.current_frame.saturating_sub(p.episode_start_frame))
            .unwrap_or(0)
    }

    /// Scrub all panes to a relative frame position.
    pub fn scrub_all_to(&mut self, relative_frame: usize) {
        for pane in &mut self.panes {
            let clamped = relative_frame.min(pane.total_frames.saturating_sub(1));
            let abs_frame = pane.episode_start_frame + clamped;
            pane.current_frame = abs_frame;
            pane.player.scrub_to(abs_frame);
        }
    }

    /// Finish scrubbing: cancel scrub decoders, seek all panes for playback,
    /// and resume playing.
    pub fn finish_scrub(&mut self, relative_frame: usize) {
        self.frame_slider_dragging = false;
        for pane in &mut self.panes {
            let clamped = relative_frame.min(pane.total_frames.saturating_sub(1));
            let abs_frame = pane.episode_start_frame + clamped;
            pane.current_frame = abs_frame;
            pane.player.cancel_scrub();
            pane.player.seek(abs_frame);
        }
        self.playing = true;
        self.last_frame_time = None;
    }
}

/// Filter video keys by the selected_cameras bitmask.
fn selected_video_keys<'a>(video_keys: &'a [String], selected: &[bool]) -> Vec<&'a str> {
    video_keys.iter().enumerate()
        .filter(|(i, _)| selected.get(*i).copied().unwrap_or(false))
        .map(|(_, k)| k.as_str())
        .collect()
}

/// Compute (cols, rows) for a given camera count.
pub(crate) fn camera_grid_size(count: usize) -> (usize, usize) {
    match count {
        0 | 1 => (1, 1),
        2 => (2, 1),
        3 | 4 => (2, 2),
        5 | 6 => (3, 2),
        _ => {
            let cols = (count as f64).sqrt().ceil() as usize;
            let rows = count.div_ceil(cols);
            (cols, rows)
        }
    }
}
