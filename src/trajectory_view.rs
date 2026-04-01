use eframe::egui;

use crate::trajectory::EeTrajectory;

/// Orbit camera state for 3D trajectory visualization.
pub(crate) struct OrbitCamera {
    /// Azimuth angle in radians (horizontal rotation around Y axis).
    pub azimuth: f32,
    /// Elevation angle in radians (vertical tilt).
    pub elevation: f32,
    /// Zoom level (distance scaling factor).
    pub zoom: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            azimuth: std::f32::consts::FRAC_PI_4,      // 45 degrees
            elevation: std::f32::consts::FRAC_PI_6,     // 30 degrees
            zoom: 1.0,
        }
    }
}

impl OrbitCamera {
    /// Project a 3D point [x, y, z] to 2D screen coordinates within the given rect.
    fn project(&self, point: [f64; 3], center: [f64; 3], rect: egui::Rect) -> egui::Pos2 {
        let px = point[0] - center[0];
        let py = point[1] - center[1];
        let pz = point[2] - center[2];

        let cos_a = self.azimuth.cos() as f64;
        let sin_a = self.azimuth.sin() as f64;
        let rx = px * cos_a - py * sin_a;
        let ry = px * sin_a + py * cos_a;
        let rz = pz;

        let cos_e = self.elevation.cos() as f64;
        let sin_e = self.elevation.sin() as f64;
        let fz = ry * sin_e + rz * cos_e;

        let scale = self.zoom * rect.width().min(rect.height()) * 0.4;
        let screen_x = rect.center().x + (rx as f32) * scale;
        let screen_y = rect.center().y - (fz as f32) * scale;

        egui::pos2(screen_x, screen_y)
    }
}

/// A trajectory entry for the overlay view.
pub(crate) struct TrajectoryEntry {
    pub trajectory: EeTrajectory,
    pub episode_index: usize,
    pub current_frame: usize,
    pub selected: bool,
}

/// Render multiple EE trajectories overlaid in one 3D view.
/// Selected entries are drawn bright with playhead; others are dimmed.
pub(crate) fn show_trajectory_overlay(
    ui: &mut egui::Ui,
    entries: &[TrajectoryEntry],
    camera: &mut OrbitCamera,
    accent_color: egui::Color32,
) -> bool {
    let available = ui.available_size();
    let (response, painter) =
        ui.allocate_painter(available, egui::Sense::click_and_drag());
    let rect = response.rect;

    let mut interacted = false;

    // Mouse drag: orbit camera
    if response.dragged() {
        let delta = response.drag_delta();
        camera.azimuth += delta.x * 0.01;
        camera.elevation = (camera.elevation + delta.y * 0.01)
            .clamp(-std::f32::consts::FRAC_PI_2 + 0.05, std::f32::consts::FRAC_PI_2 - 0.05);
        interacted = true;
    }

    // Scroll: zoom
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.0 {
            camera.zoom = (camera.zoom * (1.0 + scroll * 0.005)).clamp(0.1, 20.0);
            interacted = true;
        }
    }

    if entries.is_empty() {
        return interacted;
    }

    // Compute bounding box across ALL trajectories
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for entry in entries {
        for p in &entry.trajectory.positions {
            for i in 0..3 {
                min[i] = min[i].min(p[i]);
                max[i] = max[i].max(p[i]);
            }
        }
    }
    // Guard against empty/degenerate bounds
    if min[0] > max[0] {
        return interacted;
    }

    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];

    // Auto-zoom to fit
    let extent = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    if extent > 1e-6 && (camera.zoom - 1.0).abs() < 0.001 {
        camera.zoom = 1.0 / extent as f32;
    }

    // Background
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(20));

    let ground_z = min[2];

    // --- Ground plane grid ---
    let grid_color = egui::Color32::from_rgb(20, 75, 82);
    let pad = 0.03;
    let grid_step = nice_grid_step((max[0] - min[0] + 2.0 * pad).max(max[1] - min[1] + 2.0 * pad));

    let gx0 = ((min[0] - pad) / grid_step).floor() * grid_step;
    let gx1 = ((max[0] + pad) / grid_step).ceil() * grid_step;
    let gy0 = ((min[1] - pad) / grid_step).floor() * grid_step;
    let gy1 = ((max[1] + pad) / grid_step).ceil() * grid_step;

    {
        let mut y = gy0;
        while y <= gy1 + grid_step * 0.01 {
            let p0 = camera.project([gx0, y, ground_z], center, rect);
            let p1 = camera.project([gx1, y, ground_z], center, rect);
            painter.line_segment([p0, p1], egui::Stroke::new(1.0, grid_color));
            y += grid_step;
        }
    }
    {
        let mut x = gx0;
        while x <= gx1 + grid_step * 0.01 {
            let p0 = camera.project([x, gy0, ground_z], center, rect);
            let p1 = camera.project([x, gy1, ground_z], center, rect);
            painter.line_segment([p0, p1], egui::Stroke::new(1.0, grid_color));
            x += grid_step;
        }
    }

    // --- Axis indicator (bottom-left corner) ---
    let axis_origin = egui::pos2(rect.min.x + 30.0, rect.max.y - 30.0);
    let axis_len = 20.0;
    {
        let unit_pts = [[0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]];
        let colors = [
            egui::Color32::from_rgb(220, 60, 60),
            egui::Color32::from_rgb(60, 180, 60),
            egui::Color32::from_rgb(60, 100, 220),
        ];
        let labels = ["X", "Y", "Z"];
        let axis_rect = egui::Rect::from_center_size(axis_origin, egui::vec2(100.0, 100.0));
        let zero = camera.project([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], axis_rect);
        for (i, upt) in unit_pts.iter().enumerate() {
            let proj = camera.project(*upt, [0.0, 0.0, 0.0], axis_rect);
            let dir = (proj - zero).normalized() * axis_len;
            let end = axis_origin + dir;
            painter.line_segment([axis_origin, end], egui::Stroke::new(1.5, colors[i]));
            painter.text(
                end + dir.normalized() * 8.0,
                egui::Align2::CENTER_CENTER,
                labels[i],
                egui::FontId::monospace(9.0),
                colors[i],
            );
        }
    }

    // Dim color for unselected trajectories
    let dim_color = egui::Color32::from_rgb(
        accent_color.r() / 5,
        accent_color.g() / 5,
        accent_color.b() / 5,
    );

    // --- Draw unselected trajectories first (behind) ---
    for entry in entries {
        if entry.selected {
            continue;
        }
        let positions = &entry.trajectory.positions;
        let n = positions.len();
        if n < 2 {
            continue;
        }
        for i in 0..n - 1 {
            let p0 = camera.project(positions[i], center, rect);
            let p1 = camera.project(positions[i + 1], center, rect);
            painter.line_segment([p0, p1], egui::Stroke::new(1.0, dim_color));
        }
    }

    // --- Draw selected trajectories on top ---
    let dim_accent = egui::Color32::from_rgb(
        accent_color.r() / 4,
        accent_color.g() / 4,
        accent_color.b() / 4,
    );

    for entry in entries {
        if !entry.selected {
            continue;
        }
        let positions = &entry.trajectory.positions;
        let n = positions.len();
        if n < 2 {
            continue;
        }

        let playhead = entry.current_frame.min(n - 1);

        // Drop lines (only for selected)
        let drop_color = egui::Color32::from_gray(40);
        let drop_interval = (n / 20).max(1);
        for i in (0..n).step_by(drop_interval) {
            let top = camera.project(positions[i], center, rect);
            let bot = camera.project([positions[i][0], positions[i][1], ground_z], center, rect);
            let dist = ((top.x - bot.x).powi(2) + (top.y - bot.y).powi(2)).sqrt();
            if dist > 3.0 {
                painter.line_segment([top, bot], egui::Stroke::new(0.5, drop_color));
            }
        }
        // Playhead drop line
        {
            let top = camera.project(positions[playhead], center, rect);
            let bot = camera.project([positions[playhead][0], positions[playhead][1], ground_z], center, rect);
            painter.line_segment([top, bot], egui::Stroke::new(1.0, egui::Color32::from_gray(70)));
            painter.circle_filled(bot, 3.0, egui::Color32::from_gray(60));
        }

        // Trajectory line
        for i in 0..n - 1 {
            let p0 = camera.project(positions[i], center, rect);
            let p1 = camera.project(positions[i + 1], center, rect);
            let (color, width) = if i < playhead {
                (accent_color, 2.0)
            } else {
                (dim_accent, 1.0)
            };
            painter.line_segment([p0, p1], egui::Stroke::new(width, color));
        }

        // Markers
        let start_pt = camera.project(positions[0], center, rect);
        painter.circle_filled(start_pt, 3.0, egui::Color32::from_gray(100));

        let current_pt = camera.project(positions[playhead], center, rect);
        painter.circle_filled(current_pt, 5.0, egui::Color32::WHITE);
        painter.circle_stroke(current_pt, 7.0, egui::Stroke::new(1.5, accent_color));

        if playhead == n - 1 {
            let end_pt = camera.project(positions[n - 1], center, rect);
            painter.circle_filled(end_pt, 4.0, accent_color);
        }

        // Frame counter label
        let pos = positions[playhead];
        let frame_text = format!(
            "ep{} f{}/{} ({:.0},{:.0},{:.0})mm",
            entry.episode_index,
            playhead + 1, n,
            pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0,
        );
        painter.text(
            egui::pos2(rect.min.x + 6.0, rect.min.y + 16.0),
            egui::Align2::LEFT_TOP,
            &frame_text,
            egui::FontId::monospace(10.0),
            egui::Color32::from_gray(180),
        );
    }

    // --- Global labels ---
    let span_text = format!(
        "dx={:.0}mm dy={:.0}mm dz={:.0}mm",
        (max[0] - min[0]) * 1000.0,
        (max[1] - min[1]) * 1000.0,
        (max[2] - min[2]) * 1000.0,
    );
    painter.text(
        egui::pos2(rect.min.x + 6.0, rect.min.y + 4.0),
        egui::Align2::LEFT_TOP,
        &span_text,
        egui::FontId::monospace(10.0),
        egui::Color32::from_gray(140),
    );

    // Episode count
    let count_text = format!("{} episodes", entries.len());
    painter.text(
        egui::pos2(rect.max.x - 6.0, rect.min.y + 4.0),
        egui::Align2::RIGHT_TOP,
        &count_text,
        egui::FontId::monospace(10.0),
        egui::Color32::from_gray(100),
    );

    interacted
}

/// Pick a "nice" grid step so there are ~5-8 grid lines across the given span.
fn nice_grid_step(span: f64) -> f64 {
    if span <= 0.0 {
        return 0.01;
    }
    let raw = span / 6.0;
    let mag = 10.0_f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    };
    step * mag
}
