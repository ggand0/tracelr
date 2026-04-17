use std::path::PathBuf;

use clap::Parser;
use eframe::egui;

/// Build a GridDataset from App fields. Uses direct field access so
/// the borrow checker can track individual borrows (unlike a method,
/// which borrows all of `self`).
macro_rules! grid_dataset {
    ($self:expr) => {
        $self.dataset.as_ref().map(|ds| crate::grid::GridDataset {
            video_paths: &$self.video_paths,
            seek_ranges: &$self.seek_ranges,
            episodes: &ds.episodes,
            fps: ds.info.fps,
        })
    };
}

mod about;
mod annotation;
mod app;
mod build_info;
mod cache;
mod dataset;
mod grid;
mod perf;
mod playback;
mod theme;
mod trajectory;
mod trajectory_view;
mod ui;
mod video;

#[derive(Parser)]
#[command(name = "tracelr", about = "A fast desktop tool for exploring and tracing LeRobot datasets")]
struct Args {
    /// Path to a LeRobot dataset directory
    path: Option<PathBuf>,

    /// Enable annotation mode (prompt assignment, save/export)
    #[arg(long)]
    annotate: bool,

    /// Path to robot URDF file for trajectory visualization
    #[arg(long)]
    urdf: Option<PathBuf>,
}

fn load_icon() -> Option<egui::IconData> {
    static ICON: &[u8] = include_bytes!("../assets/icon_256.png");
    let img = image::load_from_memory(ICON).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    })
}

fn main() -> eframe::Result {
    env_logger::init();
    let args = Args::parse();

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 720.0])
        .with_drag_and_drop(true)
        .with_app_id("tracelr");

    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    let options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "tracelr",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, args.path, args.annotate, args.urdf)))),
    )
}
