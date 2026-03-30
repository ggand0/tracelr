use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::cache::VideoPlayer;

/// A single pane in the grid, displaying one episode's video.
struct GridPane {
    episode_index: usize,
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
    /// First episode index shown in the grid (top-left pane).
    pub start_episode: usize,
    pub playing: bool,
    last_frame_time: Option<Instant>,
    fps: u32,
    /// Which pane is selected (for highlighting). None = no selection.
    pub selected_pane: Option<usize>,
}

impl GridView {
    /// Create a new grid view starting at `start_episode` with `cols x rows` panes.
    pub fn new(
        ctx: &egui::Context,
        cols: usize,
        rows: usize,
        start_episode: usize,
        video_paths: &[PathBuf],
        seek_ranges: &[Option<(f64, f64)>],
        episodes: &[crate::dataset::EpisodeMeta],
        fps: u32,
    ) -> Self {
        let total_panes = cols * rows;
        let mut panes = Vec::with_capacity(total_panes);

        for i in 0..total_panes {
            let ep_idx = start_episode + i;
            if ep_idx >= video_paths.len() {
                break;
            }
            if let Some(pane) = Self::create_pane(ctx, ep_idx, video_paths, seek_ranges, episodes, fps) {
                panes.push(pane);
            }
        }

        Self {
            panes,
            cols,
            rows,
            start_episode,
            playing: true,
            last_frame_time: None,
            fps,
            selected_pane: None,
        }
    }

    fn create_pane(
        ctx: &egui::Context,
        ep_idx: usize,
        video_paths: &[PathBuf],
        seek_ranges: &[Option<(f64, f64)>],
        episodes: &[crate::dataset::EpisodeMeta],
        fps: u32,
    ) -> Option<GridPane> {
        let video_path = video_paths.get(ep_idx)?;
        let ep = episodes.get(ep_idx)?;
        let total_frames = ep.length;

        let from_ts = seek_ranges.get(ep_idx)
            .and_then(|r| r.as_ref())
            .map(|(from, _)| *from)
            .unwrap_or(0.0);
        let start_frame = if from_ts > 0.0 {
            (from_ts * fps as f64) as usize
        } else {
            0
        };

        let player_total = start_frame + total_frames;
        let player = VideoPlayer::new(ctx, video_path, player_total, fps, start_frame);

        Some(GridPane {
            episode_index: ep_idx,
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
            // Still poll first frames for panes that haven't received one yet
            for pane in &mut self.panes {
                if pane.current_texture.is_none() {
                    if let Some(tex) = pane.player.poll_next_frame() {
                        pane.current_frame = pane.player.current_frame;
                        pane.current_texture = Some(tex);
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
    pub fn toggle_playing(&mut self) {
        self.playing = !self.playing;
        if self.playing {
            self.last_frame_time = Some(Instant::now());
        } else {
            self.last_frame_time = None;
        }
    }

    /// Shift the grid forward or backward by `cols * rows` episodes.
    pub fn navigate_page(
        &mut self,
        delta: isize,
        ctx: &egui::Context,
        video_paths: &[PathBuf],
        seek_ranges: &[Option<(f64, f64)>],
        episodes: &[crate::dataset::EpisodeMeta],
    ) {
        let page_size = self.cols * self.rows;
        let total = video_paths.len();
        if total == 0 {
            return;
        }

        let new_start = if delta > 0 {
            let s = self.start_episode + page_size;
            if s >= total { return; } // already at end
            s
        } else {
            if self.start_episode == 0 { return; }
            self.start_episode.saturating_sub(page_size)
        };

        // Drop old panes (cancels decode threads via Drop)
        self.panes.clear();
        self.start_episode = new_start;
        self.selected_pane = None;

        for i in 0..(self.cols * self.rows) {
            let ep_idx = new_start + i;
            if ep_idx >= total {
                break;
            }
            if let Some(pane) = Self::create_pane(ctx, ep_idx, video_paths, seek_ranges, episodes, self.fps) {
                self.panes.push(pane);
            }
        }

        self.playing = true;
        self.last_frame_time = None;
    }

    /// Resize the grid (change cols/rows) and rebuild panes from current start_episode.
    pub fn resize(
        &mut self,
        cols: usize,
        rows: usize,
        ctx: &egui::Context,
        video_paths: &[PathBuf],
        seek_ranges: &[Option<(f64, f64)>],
        episodes: &[crate::dataset::EpisodeMeta],
    ) {
        self.panes.clear();
        self.cols = cols;
        self.rows = rows;
        self.selected_pane = None;

        let total = video_paths.len();
        for i in 0..(cols * rows) {
            let ep_idx = self.start_episode + i;
            if ep_idx >= total {
                break;
            }
            if let Some(pane) = Self::create_pane(ctx, ep_idx, video_paths, seek_ranges, episodes, self.fps) {
                self.panes.push(pane);
            }
        }

        self.playing = true;
        self.last_frame_time = None;
    }

    /// Render the grid into the given UI area.
    pub fn show(&mut self, ui: &mut egui::Ui, theme_muted: egui::Color32, theme_accent: egui::Color32) {
        let available = ui.available_size();
        let cols = self.cols;
        let rows = self.rows;

        let spacing = 4.0;
        let pane_w = (available.x - spacing * (cols as f32 - 1.0)) / cols as f32;
        let pane_h = (available.y - spacing * (rows as f32 - 1.0)) / rows as f32;

        let origin = ui.cursor().min;

        for (idx, pane) in self.panes.iter().enumerate() {
            let col = idx % cols;
            let row = idx / cols;

            let x = origin.x + col as f32 * (pane_w + spacing);
            let y = origin.y + row as f32 * (pane_h + spacing);
            let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(pane_w, pane_h));

            // Allocate interactive area
            let response = ui.allocate_rect(rect, egui::Sense::click());

            if response.clicked() {
                self.selected_pane = Some(idx);
            }

            // Background
            let is_selected = self.selected_pane == Some(idx);
            let bg = if is_selected {
                egui::Color32::from_gray(50)
            } else {
                egui::Color32::from_gray(30)
            };
            ui.painter().rect_filled(rect, 2.0, bg);

            // Video frame
            if let Some(tex) = &pane.current_texture {
                let tex_size = tex.size_vec2();
                // Fit within pane, leaving room for label
                let label_h = 18.0;
                let img_rect = egui::Rect::from_min_size(
                    rect.min,
                    egui::vec2(pane_w, pane_h - label_h),
                );
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
                    tex.id(),
                    img_draw_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            // Episode label at bottom of pane
            let ep_frame = pane.current_frame.saturating_sub(pane.episode_start_frame);
            let label = format!("ep {:03}  {}/{}", pane.episode_index, ep_frame + 1, pane.total_frames);
            let label_pos = egui::pos2(rect.min.x + 4.0, rect.max.y - 16.0);
            ui.painter().text(
                label_pos,
                egui::Align2::LEFT_TOP,
                &label,
                egui::FontId::monospace(11.0),
                if is_selected { theme_accent } else { theme_muted },
            );

            // Selection border
            if is_selected {
                ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(2.0, theme_accent), egui::StrokeKind::Outside);
            }
        }
    }

    /// Get the episode index of the selected pane.
    pub fn selected_episode(&self) -> Option<usize> {
        self.selected_pane.and_then(|idx| self.panes.get(idx).map(|p| p.episode_index))
    }
}
