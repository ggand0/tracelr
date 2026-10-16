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
    /// Grid cells with nothing in them (filled black).
    pub empty: Vec<Rect>,
}

/// Round down to an even number, never below 2 (x264 needs even dimensions).
fn even(v: usize) -> usize {
    (v & !1).max(2)
}

fn even_f(v: f64) -> usize {
    even(v.max(0.0) as usize)
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

#[derive(Clone, Copy, PartialEq)]
enum Cell {
    Camera(usize),
    Traj,
    Empty,
}

/// Lay out cameras of the given native `sizes` (in tile order, scaled by
/// `scale`) in `rows` rows (`None` = `auto_rows`).
///
/// Every camera keeps its own aspect ratio: the tiles of a row share one height,
/// and each row is scaled to the width of the first row, so frames sit edge to
/// edge with no letterbox bars and nothing is cropped. Cameras of the same size
/// give a plain grid. With the trajectory, the panel takes the bottom-right cell
/// and the cameras fill the other cells in order, so empty cells end up just
/// before it. The panel and empty cells use the first camera's aspect ratio.
pub(crate) fn plan(
    sizes: &[(usize, usize)],
    with_traj: bool,
    scale: f32,
    rows: Option<usize>,
) -> Plan {
    let default = [(640, 480)];
    let sizes = if sizes.is_empty() { &default[..] } else { sizes };
    let n = sizes.len();
    let aspect = |(w, h): (usize, usize)| w.max(1) as f64 / h.max(1) as f64;
    let base_aspect = aspect(sizes[0]);

    let cells = n + usize::from(with_traj);
    let requested = rows.unwrap_or_else(|| auto_rows(n, with_traj)).clamp(1, cells);
    let cols = cells.div_ceil(requested);
    // A row count that cannot be filled (e.g. 3 rows for 4 cells) collapses to
    // the rows actually needed, so the canvas never has empty rows.
    let rows = cells.div_ceil(cols);

    // Assign cells row-major; the trajectory owns the last one.
    let traj_cell = with_traj.then_some(rows * cols - 1);
    let mut next_cam = 0;
    let grid: Vec<Cell> = (0..rows * cols)
        .map(|i| {
            if Some(i) == traj_cell {
                Cell::Traj
            } else if next_cam < n {
                next_cam += 1;
                Cell::Camera(next_cam - 1)
            } else {
                Cell::Empty
            }
        })
        .collect();
    let cell_aspect = |c: Cell| match c {
        Cell::Camera(j) => aspect(sizes[j]),
        Cell::Traj | Cell::Empty => base_aspect,
    };

    // The first row sets the canvas width at the first camera's scaled height.
    let first_height = (sizes[0].1 as f64 * scale as f64).max(2.0);
    let row_aspect = |r: usize| -> f64 { grid[r * cols..(r + 1) * cols].iter().map(|&c| cell_aspect(c)).sum() };
    let width = even_f(first_height * row_aspect(0));

    let mut tiles = vec![Rect { x: 0, y: 0, w: 0, h: 0 }; n];
    let mut traj = None;
    let mut empty = Vec::new();
    let mut y = 0;
    for r in 0..rows {
        let height = even_f(width as f64 / row_aspect(r));
        let row = &grid[r * cols..(r + 1) * cols];
        let mut x = 0;
        for (k, &c) in row.iter().enumerate() {
            // The last cell absorbs rounding so the row is exactly `width`.
            let w = if k + 1 == row.len() {
                width.saturating_sub(x).max(2)
            } else {
                even_f(height as f64 * cell_aspect(c))
            };
            let rect = Rect { x, y, w, h: height };
            match c {
                Cell::Camera(j) => tiles[j] = rect,
                Cell::Traj => traj = Some(rect),
                Cell::Empty => empty.push(rect),
            }
            x += w;
        }
        y += height;
    }

    Plan {
        canvas: (width, y),
        tiles,
        traj,
        empty,
    }
}

/// Copy `img` into `dst`, resized to exactly fill it. `plan` gives every tile its
/// camera's aspect ratio, so this only absorbs a pixel or two of rounding.
pub(crate) fn blit(canvas: &mut Canvas, img: &egui::ColorImage, dst: Rect) {
    let [iw, ih] = img.size;
    if iw == 0 || ih == 0 || dst.w == 0 || dst.h == 0 {
        return;
    }

    // Video frames are opaque, so premultiplied bytes equal straight bytes.
    let resized;
    let src: &[u8] = if (dst.w, dst.h) == (iw, ih) {
        img.as_raw()
    } else {
        let buf = image::RgbaImage::from_raw(iw as u32, ih as u32, img.as_raw().to_vec())
            .expect("ColorImage byte length matches its size");
        resized = image::imageops::resize(
            &buf,
            dst.w as u32,
            dst.h as u32,
            image::imageops::FilterType::Triangle,
        );
        resized.as_raw()
    };

    let row_bytes = dst.w * 4;
    for y in 0..dst.h {
        let s = y * row_bytes;
        let d = ((dst.y + y) * canvas.width + dst.x) * 4;
        canvas.rgba[d..d + row_bytes].copy_from_slice(&src[s..s + row_bytes]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_n(n: usize, tile: (usize, usize), with_traj: bool, scale: f32, rows: Option<usize>) -> Plan {
        plan(&vec![tile; n], with_traj, scale, rows)
    }

    #[test]
    fn single_camera_fills_canvas() {
        let p = plan_n(1, (640, 480), false, 1.0, None);
        assert_eq!(p.canvas, (640, 480));
        assert_eq!(p.tiles, vec![Rect { x: 0, y: 0, w: 640, h: 480 }]);
        assert!(p.traj.is_none());
    }

    #[test]
    fn up_to_three_cameras_in_one_row() {
        for n in 1..=3 {
            let p = plan_n(n, (640, 480), false, 1.0, None);
            assert_eq!(p.canvas, (640 * n, 480));
            assert!(p.tiles.iter().all(|t| t.y == 0));
        }
        // 4 cameras: 2x2
        assert_eq!(plan_n(4, (640, 480), false, 1.0, None).canvas, (1280, 960));
    }

    #[test]
    fn trajectory_takes_bottom_right_cell() {
        // 1 camera: camera on top, trajectory below
        let p = plan_n(1, (640, 480), true, 1.0, None);
        assert_eq!(p.canvas, (640, 960));
        assert_eq!(p.tiles, vec![Rect { x: 0, y: 0, w: 640, h: 480 }]);
        assert_eq!(p.traj, Some(Rect { x: 0, y: 480, w: 640, h: 480 }));

        // 2 cameras: top row, trajectory under the second
        let p = plan_n(2, (640, 480), true, 1.0, None);
        assert_eq!(p.canvas, (1280, 960));
        assert_eq!(p.tiles[1], Rect { x: 640, y: 0, w: 640, h: 480 });
        assert_eq!(p.traj, Some(Rect { x: 640, y: 480, w: 640, h: 480 }));

        // 3 cameras at half scale: 2x2 of 320x240
        let p = plan_n(3, (640, 480), true, 0.5, None);
        assert_eq!(p.canvas, (640, 480));
        assert_eq!(p.tiles[2], Rect { x: 0, y: 240, w: 320, h: 240 });
        assert_eq!(p.traj, Some(Rect { x: 320, y: 240, w: 320, h: 240 }));

        // 4 cameras: 3x2, trajectory still bottom right
        let p = plan_n(4, (640, 480), true, 1.0, None);
        assert_eq!(p.canvas, (1920, 960));
        assert_eq!(p.tiles[3], Rect { x: 0, y: 480, w: 640, h: 480 });
        assert_eq!(p.traj, Some(Rect { x: 1280, y: 480, w: 640, h: 480 }));
    }

    #[test]
    fn explicit_rows() {
        // 3 cameras in 2 rows: 2x2 with the last cell empty
        let p = plan_n(3, (640, 480), false, 1.0, Some(2));
        assert_eq!(p.canvas, (1280, 960));
        assert_eq!(p.tiles[2], Rect { x: 0, y: 480, w: 640, h: 480 });

        // 3 cameras + trajectory in 1 row
        let p = plan_n(3, (640, 480), true, 1.0, Some(1));
        assert_eq!(p.canvas, (2560, 480));
        assert_eq!(p.traj, Some(Rect { x: 1920, y: 0, w: 640, h: 480 }));

        // 3 rows for 4 cells collapses to 2 rows of 2
        assert_eq!(plan_n(3, (640, 480), true, 1.0, Some(3)).canvas, (1280, 960));

        // more rows than cells clamps to one cell per row
        assert_eq!(plan_n(2, (640, 480), false, 1.0, Some(3)).canvas, (640, 960));
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
            let p = plan_n(3, (641, 481), with_traj, 1.0, None);
            assert_eq!(p.canvas.0 % 2, 0);
            assert_eq!(p.canvas.1 % 2, 0);
            for t in p.tiles.iter().chain(p.traj.iter()) {
                assert_eq!(t.w % 2, 0);
                assert_eq!(t.h % 2, 0);
            }
        }
    }

    #[test]
    fn mixed_aspect_cameras_have_no_bars() {
        // nova rig: 16:9 head + two 16:10 hand cameras, half scale
        let sizes = [(640, 360), (960, 600), (960, 600)];
        let p = plan(&sizes, false, 0.5, None);
        assert_eq!(p.canvas, (896, 180));
        assert_eq!(p.tiles[0], Rect { x: 0, y: 0, w: 320, h: 180 });
        assert_eq!(p.tiles[1], Rect { x: 320, y: 0, w: 288, h: 180 });
        assert_eq!(p.tiles[2], Rect { x: 608, y: 0, w: 288, h: 180 });
        assert!(p.empty.is_empty());
    }

    #[test]
    fn rows_are_justified_to_one_width() {
        // Two rows with different aspect sums still end exactly at the canvas edge.
        let sizes = [(640, 360), (640, 360), (960, 600)];
        for with_traj in [false, true] {
            let p = plan(&sizes, with_traj, 1.0, Some(2));
            let mut covered = 0;
            for r in p.tiles.iter().chain(p.traj.iter()).chain(p.empty.iter()) {
                assert!(r.x + r.w <= p.canvas.0 && r.y + r.h <= p.canvas.1);
                assert_eq!(r.w % 2, 0);
                assert_eq!(r.h % 2, 0);
                covered += r.w * r.h;
            }
            assert_eq!(covered, p.canvas.0 * p.canvas.1, "cells tile the canvas exactly");
        }
    }

    #[test]
    fn empty_cells_are_reported() {
        // 2 cameras + trajectory: bottom-left cell is empty
        let p = plan_n(2, (640, 480), true, 1.0, None);
        assert_eq!(p.empty, vec![Rect { x: 0, y: 480, w: 640, h: 480 }]);
    }

    #[test]
    fn blit_fills_destination() {
        let mut canvas = Canvas::new(8, 4);
        let img = egui::ColorImage::new([2, 2], egui::Color32::WHITE);
        blit(&mut canvas, &img, Rect { x: 0, y: 0, w: 8, h: 4 });
        assert!(canvas.rgba.chunks_exact(4).all(|px| px == [255, 255, 255, 255]));
    }
}
