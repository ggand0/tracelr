use std::path::Path;

use eframe::egui;

/// Pixel rectangle inside a canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// Straight RGBA8 frame buffer that exports are composited into.
#[derive(Clone)]
pub(crate) struct Canvas {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Canvas {
    /// Opaque black canvas.
    pub fn new(width: usize, height: usize) -> Self {
        let mut rgba = vec![0u8; width * height * 4];
        for px in rgba.chunks_exact_mut(4) {
            px[3] = 255;
        }
        Self { width, height, rgba }
    }

    /// Reset every pixel to opaque black.
    pub fn clear(&mut self) {
        for px in self.rgba.chunks_exact_mut(4) {
            px.copy_from_slice(&[0, 0, 0, 255]);
        }
    }

    pub fn to_color_image(&self) -> egui::ColorImage {
        egui::ColorImage::from_rgba_unmultiplied([self.width, self.height], &self.rgba)
    }

    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        image::save_buffer(
            path,
            &self.rgba,
            self.width as u32,
            self.height as u32,
            image::ColorType::Rgba8,
        )
        .map_err(|e| format!("Failed to write {}: {}", path.display(), e))
    }
}

/// Where each camera tile and the trajectory panel go on the canvas.
#[derive(Debug)]
pub(crate) struct Plan {
    pub canvas: (usize, usize),
    pub tiles: Vec<Rect>,
    pub traj: Option<Rect>,
}

/// Round down to an even number, never below 2 (x264 needs even dimensions).
fn even(v: usize) -> usize {
    (v & !1).max(2)
}

/// Row count used when the user leaves rows on Auto: up to 3 cameras sit in a
/// single row; with the trajectory, or with more cameras, two rows up to 8 cells;
/// beyond that a near-square grid.
pub(crate) fn auto_rows(n_cameras: usize, with_traj: bool) -> usize {
    let n = n_cameras.max(1);
    let cells = n + usize::from(with_traj);
    if !with_traj && n <= 3 {
        1
    } else if cells <= 8 {
        2
    } else {
        let cols = (cells as f64).sqrt().ceil() as usize;
        cells.div_ceil(cols)
    }
}

/// Lay out `n_cameras` tiles of size `tile` (scaled by `scale`) in `rows` rows
/// (`None` = `auto_rows`). With the trajectory, the panel takes a full tile in
/// the bottom-right cell and the cameras fill the other cells in order, so empty
/// cells end up just before the panel.
pub(crate) fn plan(
    n_cameras: usize,
    tile: (usize, usize),
    with_traj: bool,
    scale: f32,
    rows: Option<usize>,
) -> Plan {
    let n = n_cameras.max(1);
    let tw = even((tile.0 as f32 * scale) as usize);
    let th = even((tile.1 as f32 * scale) as usize);

    let cells = n + usize::from(with_traj);
    let requested = rows.unwrap_or_else(|| auto_rows(n, with_traj)).clamp(1, cells);
    let cols = cells.div_ceil(requested);
    // A row count that cannot be filled (e.g. 3 rows for 4 cells) collapses to
    // the rows actually needed, so the canvas never has empty rows.
    let rows = cells.div_ceil(cols);
    let cell = |i: usize| Rect {
        x: (i % cols) * tw,
        y: (i / cols) * th,
        w: tw,
        h: th,
    };

    // The trajectory owns the bottom-right cell; cameras skip it.
    let traj_cell = with_traj.then_some(rows * cols - 1);
    let tiles = (0..rows * cols)
        .filter(|&i| Some(i) != traj_cell)
        .take(n)
        .map(cell)
        .collect();

    Plan {
        canvas: (cols * tw, rows * th),
        tiles,
        traj: traj_cell.map(cell),
    }
}

/// Copy `img` into `dst`, resizing to fit inside it with the aspect ratio kept
/// (centred, the rest of `dst` is left untouched).
pub(crate) fn blit(canvas: &mut Canvas, img: &egui::ColorImage, dst: Rect) {
    let [iw, ih] = img.size;
    if iw == 0 || ih == 0 || dst.w == 0 || dst.h == 0 {
        return;
    }

    // Fit inside dst, keeping aspect.
    let scale = (dst.w as f64 / iw as f64).min(dst.h as f64 / ih as f64);
    let fw = ((iw as f64 * scale).round() as usize).clamp(1, dst.w);
    let fh = ((ih as f64 * scale).round() as usize).clamp(1, dst.h);
    let ox = dst.x + (dst.w - fw) / 2;
    let oy = dst.y + (dst.h - fh) / 2;

    // Video frames are opaque, so premultiplied bytes equal straight bytes.
    let resized;
    let src: &[u8] = if (fw, fh) == (iw, ih) {
        img.as_raw()
    } else {
        let buf = image::RgbaImage::from_raw(iw as u32, ih as u32, img.as_raw().to_vec())
            .expect("ColorImage byte length matches its size");
        resized = image::imageops::resize(
            &buf,
            fw as u32,
            fh as u32,
            image::imageops::FilterType::Triangle,
        );
        resized.as_raw()
    };

    let row_bytes = fw * 4;
    for y in 0..fh {
        let s = y * row_bytes;
        let d = ((oy + y) * canvas.width + ox) * 4;
        canvas.rgba[d..d + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_camera_fills_canvas() {
        let p = plan(1, (640, 480), false, 1.0, None);
        assert_eq!(p.canvas, (640, 480));
        assert_eq!(p.tiles, vec![Rect { x: 0, y: 0, w: 640, h: 480 }]);
        assert!(p.traj.is_none());
    }

    #[test]
    fn up_to_three_cameras_in_one_row() {
        for n in 1..=3 {
            let p = plan(n, (640, 480), false, 1.0, None);
            assert_eq!(p.canvas, (640 * n, 480));
            assert!(p.tiles.iter().all(|t| t.y == 0));
        }
        // 4 cameras: 2x2
        assert_eq!(plan(4, (640, 480), false, 1.0, None).canvas, (1280, 960));
    }

    #[test]
    fn trajectory_takes_bottom_right_cell() {
        // 1 camera: camera on top, trajectory below
        let p = plan(1, (640, 480), true, 1.0, None);
        assert_eq!(p.canvas, (640, 960));
        assert_eq!(p.tiles, vec![Rect { x: 0, y: 0, w: 640, h: 480 }]);
        assert_eq!(p.traj, Some(Rect { x: 0, y: 480, w: 640, h: 480 }));

        // 2 cameras: top row, trajectory under the second
        let p = plan(2, (640, 480), true, 1.0, None);
        assert_eq!(p.canvas, (1280, 960));
        assert_eq!(p.tiles[1], Rect { x: 640, y: 0, w: 640, h: 480 });
        assert_eq!(p.traj, Some(Rect { x: 640, y: 480, w: 640, h: 480 }));

        // 3 cameras at half scale: 2x2 of 320x240
        let p = plan(3, (640, 480), true, 0.5, None);
        assert_eq!(p.canvas, (640, 480));
        assert_eq!(p.tiles[2], Rect { x: 0, y: 240, w: 320, h: 240 });
        assert_eq!(p.traj, Some(Rect { x: 320, y: 240, w: 320, h: 240 }));

        // 4 cameras: 3x2, trajectory still bottom right
        let p = plan(4, (640, 480), true, 1.0, None);
        assert_eq!(p.canvas, (1920, 960));
        assert_eq!(p.tiles[3], Rect { x: 0, y: 480, w: 640, h: 480 });
        assert_eq!(p.traj, Some(Rect { x: 1280, y: 480, w: 640, h: 480 }));
    }

    #[test]
    fn explicit_rows() {
        // 3 cameras in 2 rows: 2x2 with the last cell empty
        let p = plan(3, (640, 480), false, 1.0, Some(2));
        assert_eq!(p.canvas, (1280, 960));
        assert_eq!(p.tiles[2], Rect { x: 0, y: 480, w: 640, h: 480 });

        // 3 cameras + trajectory in 1 row
        let p = plan(3, (640, 480), true, 1.0, Some(1));
        assert_eq!(p.canvas, (2560, 480));
        assert_eq!(p.traj, Some(Rect { x: 1920, y: 0, w: 640, h: 480 }));

        // 3 rows for 4 cells collapses to 2 rows of 2
        assert_eq!(plan(3, (640, 480), true, 1.0, Some(3)).canvas, (1280, 960));

        // more rows than cells clamps to one cell per row
        assert_eq!(plan(2, (640, 480), false, 1.0, Some(3)).canvas, (640, 960));
    }

    #[test]
    fn auto_rows_for_common_rigs() {
        let rows = |n, t| auto_rows(n, t);
        assert_eq!([2, 3, 4, 5, 6].map(|n| rows(n, false)), [1, 1, 2, 2, 2]);
        assert_eq!([2, 3, 4, 5, 6].map(|n| rows(n, true)), [2, 2, 2, 2, 2]);
        assert_eq!(rows(9, false), 3);
    }

    #[test]
    fn odd_sizes_round_down_to_even() {
        for with_traj in [false, true] {
            let p = plan(3, (641, 481), with_traj, 1.0, None);
            assert_eq!(p.canvas.0 % 2, 0);
            assert_eq!(p.canvas.1 % 2, 0);
            for t in p.tiles.iter().chain(p.traj.iter()) {
                assert_eq!(t.w % 2, 0);
                assert_eq!(t.h % 2, 0);
            }
        }
    }

    #[test]
    fn blit_letterboxes_mismatched_aspect() {
        let mut canvas = Canvas::new(8, 4);
        let img = egui::ColorImage::new([2, 2], egui::Color32::WHITE);
        blit(&mut canvas, &img, Rect { x: 0, y: 0, w: 8, h: 4 });
        let px = |x: usize, y: usize| &canvas.rgba[(y * 8 + x) * 4..(y * 8 + x) * 4 + 4];
        // 2x2 scaled to 4x4, centred at x = 2..6
        assert_eq!(px(0, 0), &[0, 0, 0, 255]);
        assert_eq!(px(3, 2), &[255, 255, 255, 255]);
        assert_eq!(px(7, 3), &[0, 0, 0, 255]);
    }
}
