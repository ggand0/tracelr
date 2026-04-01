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
    /// Uses orthographic projection with orbit rotation.
    fn project(&self, point: [f64; 3], center: [f64; 3], rect: egui::Rect) -> egui::Pos2 {
        // Translate to center
        let px = point[0] - center[0];
        let py = point[1] - center[1];
        let pz = point[2] - center[2];

        // Rotate by azimuth (around Z axis — up in robot frame)
        let cos_a = self.azimuth.cos() as f64;
        let sin_a = self.azimuth.sin() as f64;
        let rx = px * cos_a - py * sin_a;
        let ry = px * sin_a + py * cos_a;
        let rz = pz;

        // Rotate by elevation (tilt around the resulting X axis)
        let cos_e = self.elevation.cos() as f64;
        let sin_e = self.elevation.sin() as f64;
        let _fy = ry * cos_e - rz * sin_e;
        let fz = ry * sin_e + rz * cos_e;

        // Orthographic projection: use rx as screen X, fz as screen Y (up)
        let scale = self.zoom * rect.width().min(rect.height()) * 0.4;
        let screen_x = rect.center().x + (rx as f32) * scale;
        let screen_y = rect.center().y - (fz as f32) * scale; // flip Y (screen Y is down)

        egui::pos2(screen_x, screen_y)
    }
}

/// Render the EE trajectory as a 3D projected line in the given UI area.
/// Returns true if the camera was interacted with (drag/scroll).
pub(crate) fn show_trajectory_3d(
    ui: &mut egui::Ui,
    trajectory: &EeTrajectory,
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

    let positions = &trajectory.positions;
    if positions.is_empty() {
        return interacted;
    }

    // Compute bounding box center for centering the view
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];

    // Auto-zoom to fit the trajectory extent
    let extent = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2) + (max[2] - min[2]).powi(2)).sqrt();
    if extent > 1e-6 {
        let base_zoom = 1.0 / extent as f32;
        // Only auto-set zoom on first render (when zoom is default 1.0)
        // After that, user controls zoom via scroll
        if (camera.zoom - 1.0).abs() < 0.001 {
            camera.zoom = base_zoom;
        }
    }

    // Background
    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(20));

    // Draw axis indicator in bottom-left corner
    let axis_origin = egui::pos2(rect.min.x + 30.0, rect.max.y - 30.0);
    let axis_len = 20.0;
    {
        // Project unit vectors
        let unit_pts = [[0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]];
        let colors = [
            egui::Color32::from_rgb(220, 60, 60),   // X = red
            egui::Color32::from_rgb(60, 180, 60),    // Y = green
            egui::Color32::from_rgb(60, 100, 220),   // Z = blue
        ];
        let labels = ["X", "Y", "Z"];
        let zero = camera.project([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], egui::Rect::from_center_size(axis_origin, egui::vec2(100.0, 100.0)));
        for (i, upt) in unit_pts.iter().enumerate() {
            let proj = camera.project(*upt, [0.0, 0.0, 0.0], egui::Rect::from_center_size(axis_origin, egui::vec2(100.0, 100.0)));
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

    // Draw trajectory line segments with time-based color gradient
    let n = positions.len();
    if n < 2 {
        return interacted;
    }

    for i in 0..n - 1 {
        let t = i as f32 / (n - 1) as f32;
        let p0 = camera.project(positions[i], center, rect);
        let p1 = camera.project(positions[i + 1], center, rect);

        // Color gradient: accent color with varying alpha/brightness
        let r = lerp_u8(80, accent_color.r(), t);
        let g = lerp_u8(80, accent_color.g(), t);
        let b = lerp_u8(80, accent_color.b(), t);
        let color = egui::Color32::from_rgb(r, g, b);

        painter.line_segment([p0, p1], egui::Stroke::new(1.5, color));
    }

    // Draw start point (bright) and end point (accent)
    let start_pt = camera.project(positions[0], center, rect);
    let end_pt = camera.project(positions[n - 1], center, rect);
    painter.circle_filled(start_pt, 4.0, egui::Color32::from_rgb(100, 100, 100));
    painter.circle_filled(end_pt, 4.0, accent_color);

    // Label: EE span
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

    interacted
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}
