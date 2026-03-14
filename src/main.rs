use std::path::PathBuf;

use clap::Parser;
use eframe::egui;

mod annotation;
mod app;
mod build_info;
mod cache;
mod dataset;
mod gpu_decode;
mod perf;
mod theme;
mod video;

#[derive(Parser)]
#[command(name = "lerobot-explorer", about = "LeRobot dataset annotation tool")]
struct Args {
    /// Path to a LeRobot dataset directory
    path: Option<PathBuf>,
}

fn main() -> eframe::Result {
    env_logger::init();
    let args = Args::parse();

    // Try to create a custom Vulkan device with both graphics + video decode
    let wgpu_setup = match gpu_decode::wgpu_device::GpuSetup::new() {
        Ok(setup) => {
            log::info!("Custom GPU setup with video decode support created");
            Some(setup)
        }
        Err(e) => {
            log::warn!("Failed to create custom GPU setup: {}. Using default.", e);
            None
        }
    };

    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    if let Some(setup) = wgpu_setup {
        options.wgpu_options.wgpu_setup = egui_wgpu::WgpuSetup::Existing(
            egui_wgpu::WgpuSetupExisting {
                instance: setup.wgpu_instance,
                adapter: setup.wgpu_adapter,
                device: setup.wgpu_device,
                queue: setup.wgpu_queue,
            },
        );
        // TODO: store setup.ash_device, setup.video_queue_family etc.
        // for the AV1 decoder to use
    }

    eframe::run_native(
        "lerobot-explorer",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, args.path)))),
    )
}
