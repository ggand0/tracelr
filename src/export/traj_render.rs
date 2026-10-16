use eframe::egui;

use super::{GpuHandles, TrajectoryInput};
use crate::trajectory_view::{show_trajectory_overlay, OrbitCamera, TrajectoryEntry};

/// Short side of the panel in points. The pixels-per-point is derived from it
/// so text and line widths look like the on-screen panel at any export resolution.
const PANEL_POINTS: f32 = 320.0;

/// Renders the trajectory panel into an offscreen texture with its own egui
/// context and egui-wgpu renderer, and reads the pixels back.
pub(crate) struct TrajectoryRenderer {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    gpu: GpuHandles,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
    pixels_per_point: f32,
    entry: TrajectoryEntry,
    camera: OrbitCamera,
    accent: egui::Color32,
}

impl TrajectoryRenderer {
    pub fn new(input: TrajectoryInput, width: usize, height: usize) -> Result<Self, String> {
        let (width, height) = (width.max(2) as u32, height.max(2) as u32);
        let device = &input.gpu.device;
        let max_side = device.limits().max_texture_dimension_2d;
        if width.max(height) > max_side {
            return Err(format!(
                "Trajectory panel {}x{}px exceeds GPU texture limit {}px",
                width, height, max_side
            ));
        }

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let renderer = egui_wgpu::Renderer::new(device, format, None, 1, false);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export_trajectory"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (width * 4).div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export_trajectory_readback"),
            size: padded_bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            ctx: egui::Context::default(),
            renderer,
            gpu: input.gpu,
            texture,
            view,
            readback,
            padded_bytes_per_row,
            width,
            height,
            pixels_per_point: (width.min(height) as f32 / PANEL_POINTS).max(1.0),
            entry: TrajectoryEntry {
                trajectory: input.trajectory,
                episode_index: input.episode,
                current_frame: 0,
                selected: true,
            },
            camera: input.camera,
            accent: input.accent,
        })
    }

    /// Render the panel with the playhead at episode-relative `frame`.
    pub fn render(&mut self, frame: usize) -> Result<egui::ColorImage, String> {
        self.entry.current_frame = frame;
        let ppp = self.pixels_per_point;
        let size_points = egui::vec2(self.width as f32, self.height as f32) / ppp;

        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size_points)),
            max_texture_side: Some(self.gpu.device.limits().max_texture_dimension_2d as usize),
            ..Default::default()
        };
        raw.viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .native_pixels_per_point = Some(ppp);

        let (entry, camera, accent) = (&self.entry, &mut self.camera, self.accent);
        let out = self.ctx.run(raw, |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    // The overlay persists its auto-zoom into the camera on the first call.
                    show_trajectory_overlay(ui, std::slice::from_ref(entry), camera, accent);
                });
        });

        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        for (id, delta) in &out.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        let jobs = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        let desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.width, self.height],
            pixels_per_point: out.pixels_per_point,
        };

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export_trajectory"),
        });
        let cmds = self.renderer.update_buffers(device, queue, &mut enc, &jobs, &desc);
        {
            let pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export_trajectory"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.renderer.render(&mut pass.forget_lifetime(), &jobs, &desc);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(cmds.into_iter().chain([enc.finish()]));

        for id in &out.textures_delta.free {
            self.renderer.free_texture(id);
        }

        self.read_pixels()
    }

    /// Map the readback buffer and strip the row padding. The texture is sRGB, so
    /// the bytes are already gamma encoded.
    fn read_pixels(&self) -> Result<egui::ColorImage, String> {
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "GPU readback was dropped".to_string())?
            .map_err(|e| format!("GPU readback failed: {}", e))?;

        let (w, h) = (self.width as usize, self.height as usize);
        let row = w * 4;
        let mut rgba = Vec::with_capacity(row * h);
        {
            let data = slice.get_mapped_range();
            for y in 0..h {
                let start = y * self.padded_bytes_per_row as usize;
                rgba.extend_from_slice(&data[start..start + row]);
            }
        }
        self.readback.unmap();
        Ok(egui::ColorImage::from_rgba_premultiplied([w, h], &rgba))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs a GPU: `cargo test -- --ignored traj_render`
    #[test]
    #[ignore]
    fn traj_render_draws_something() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("no GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .expect("no GPU device");

        let input = TrajectoryInput {
            trajectory: crate::trajectory::EeTrajectory {
                positions: vec![[0.0, 0.0, 0.0], [0.1, 0.05, 0.02], [0.2, 0.0, 0.1]],
            },
            episode: 0,
            camera: OrbitCamera::default(),
            accent: egui::Color32::from_rgb(0, 200, 200),
            gpu: GpuHandles { device, queue },
        };
        let mut r = TrajectoryRenderer::new(input, 160, 128).unwrap();
        let img = r.render(1).unwrap();
        assert_eq!(img.size, [160, 128]);
        assert!(img.pixels.iter().any(|p| *p != egui::Color32::BLACK));
    }
}
