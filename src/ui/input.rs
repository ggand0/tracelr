use std::path::PathBuf;
use std::time::Instant;

use eframe::egui;

use crate::app::App;
use crate::dataset;

impl App {
    pub(crate) fn handle_keyboard(&mut self, ctx: &egui::Context) {
        if self.dataset.is_none() {
            return;
        }

        let mut g_pressed = false;
        let mut enter_pressed = false;
        let mut escape_pressed = false;
        let mut space_pressed = false;
        let mut plus_pressed = false;
        let mut minus_pressed = false;

        ctx.input(|i| {
            g_pressed = i.key_pressed(egui::Key::G);
            enter_pressed = i.key_pressed(egui::Key::Enter);
            escape_pressed = i.key_pressed(egui::Key::Escape);
            space_pressed = i.key_pressed(egui::Key::Space);
            plus_pressed = i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals);
            minus_pressed = i.key_pressed(egui::Key::Minus);

            if self.annotate_mode {
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
            }
        });

        // G toggles grid view
        if g_pressed {
            self.toggle_grid_view(ctx);
            return;
        }

        // Grid mode keyboard
        if self.grid_view.is_some() {
            if space_pressed {
                if let Some(grid) = &mut self.grid_view {
                    grid.toggle_playing();
                }
            }
            if escape_pressed {
                self.grid_view = None;
                return;
            }
            // +/- resize grid
            if plus_pressed {
                self.grid_resize(1, ctx);
            }
            if minus_pressed {
                self.grid_resize(-1, ctx);
            }
            self.handle_keyboard_grid(ctx);
            return;
        }

        // Single-video mode
        if escape_pressed && self.viewing_video {
            self.exit_video_mode();
            return;
        }
        if enter_pressed && !self.viewing_video {
            self.enter_video_mode(ctx);
            return;
        }

        if space_pressed && self.viewing_video {
            self.playing = !self.playing;
            if self.playing {
                self.last_frame_time = Some(Instant::now());
            } else {
                self.last_frame_time = None;
            }
        }

        self.handle_keyboard_episode(ctx);
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

    pub(crate) fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });

        if let Some(path) = dropped.first() {
            if dataset::is_lerobot_dataset(path) {
                let was_grid = self.grid_view.is_some();
                self.grid_view = None;
                self.exit_video_mode();
                self.load_dataset(path);
                if self.dataset.is_some() {
                    self.init_cache(ctx);
                    if was_grid {
                        self.toggle_grid_view(ctx);
                    } else {
                        self.enter_video_mode(ctx);
                    }
                }
            } else {
                self.loading_error = Some(format!(
                    "Not a LeRobot dataset: {}\nExpected meta/info.json",
                    path.display()
                ));
            }
        }
    }

    fn handle_keyboard_grid(&mut self, ctx: &egui::Context) {
        let mut page_delta: Option<isize> = None;

        ctx.input(|i| {
            if i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::D) {
                page_delta = Some(1);
            }
            if i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::A) {
                page_delta = Some(-1);
            }
        });

        if let Some(delta) = page_delta {
            if let Some(ds) = &self.dataset {
                let gds = crate::grid::GridDataset {
                    video_paths: &self.video_paths,
                    seek_ranges: &self.seek_ranges,
                    episodes: &ds.episodes,
                    fps: ds.info.fps,
                };
                if let Some(grid) = &mut self.grid_view {
                    grid.navigate_page(delta, ctx, &gds);
                    self.scroll_to_selected = true;
                }
            }
        }
    }
}
