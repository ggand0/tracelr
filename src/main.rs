use std::path::PathBuf;

use clap::Parser;
use eframe::egui;

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
#[command(name = "lerobot-explorer", about = "LeRobot dataset explorer and annotation tool")]
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

fn main() -> eframe::Result {
    env_logger::init();
    let args = Args::parse();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "lerobot-explorer",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, args.path, args.annotate, args.urdf)))),
    )
}
