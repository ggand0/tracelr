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

    // Create a single VkInstance + VkDevice with both graphics and video decode.
    // NVIDIA's driver corrupts video decode if multiple VkInstances exist,
    // so we create ONE and share it between wgpu (rendering) and the AV1 decoder.
    let gpu_setup = gpu_decode::wgpu_device::GpuSetup::new().ok();

    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    if let Some(setup) = gpu_setup {
        // Store shared handles for the video decoder
        gpu_decode::wgpu_device::set_shared_gpu(
            gpu_decode::wgpu_device::SharedGpuState {
                entry: setup.entry,
                ash_instance: setup.ash_instance,
                ash_device: setup.ash_device,
                physical_device: setup.physical_device,
                video_queue_family: setup.video_queue_family,
            },
        );

        // Give the wgpu-wrapped device to eframe
        options.wgpu_options.wgpu_setup = egui_wgpu::WgpuSetup::Existing(
            egui_wgpu::WgpuSetupExisting {
                instance: setup.wgpu_instance,
                adapter: setup.wgpu_adapter,
                device: setup.wgpu_device,
                queue: setup.wgpu_queue,
            },
        );
    }

    eframe::run_native(
        "lerobot-explorer",
        options,
        Box::new(move |cc| Ok(Box::new(app::App::new(cc, args.path)))),
    )
}
