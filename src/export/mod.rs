//! Shareable export of the current episode.
//!
//! Every export runs one pipeline on a background thread: decode the episode's
//! frames for each camera -> composite (camera tiles + optional trajectory panel)
//! -> sink (clipboard image, png, mp4, or a report folder).

pub(crate) mod compose;
pub(crate) mod encoder;
pub(crate) mod report;
pub(crate) mod source;
pub(crate) mod traj_render;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use eframe::egui;

use compose::{Canvas, Plan};
use encoder::Mp4Writer;
pub(crate) use report::{AnnotationMeta, ReportMeta};
use source::EpisodeFrames;
use traj_render::TrajectoryRenderer;

/// Side of the standalone `trajectory.png` in a report.
const REPORT_TRAJECTORY_SIDE: usize = 720;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    Cameras,
    CamerasWithTrajectory,
}

#[derive(Clone, Copy)]
pub(crate) enum FrameRange {
    Whole,
    /// Episode-relative frame index.
    Single(usize),
}

pub(crate) enum Sink {
    /// Only valid with `FrameRange::Single`.
    Clipboard,
    /// Only valid with `FrameRange::Single`.
    Png(PathBuf),
    Mp4(PathBuf),
    /// Parent folder; the report goes into a new `<dataset>_ep<NNNN>` folder inside it.
    Report(PathBuf),
}

pub(crate) struct CameraSource {
    pub display_name: String,
    pub path: PathBuf,
    /// None for v2.1 (the whole file is the episode).
    pub seek_range: Option<(f64, f64)>,
}

#[derive(Clone)]
pub(crate) struct GpuHandles {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[derive(Clone)]
pub(crate) struct TrajectoryInput {
    pub trajectory: crate::trajectory::EeTrajectory,
    pub episode: usize,
    pub camera: crate::trajectory_view::OrbitCamera,
    pub accent: egui::Color32,
    pub gpu: GpuHandles,
}

pub(crate) struct ExportRequest {
    pub episode_length: usize,
    pub fps: u32,
    /// At least one.
    pub cameras: Vec<CameraSource>,
    pub layout: Layout,
    /// Required for `Layout::CamerasWithTrajectory`. With `Sink::Report` it is also
    /// used for `trajectory.png` regardless of the layout.
    pub trajectory: Option<TrajectoryInput>,
    pub range: FrameRange,
    pub sink: Sink,
    /// Applied to the composite (1.0 or 0.5). Per-camera report clips stay native.
    pub scale: f32,
    /// Composite row count; None = `compose::auto_rows`.
    pub rows: Option<usize>,
    /// Used by `Sink::Report` only.
    pub meta: Option<ReportMeta>,
}

pub(crate) enum ExportEvent {
    Progress { done: usize, total: usize },
    Finished(ExportOutput),
    Failed(String),
}

pub(crate) enum ExportOutput {
    /// For `Sink::Clipboard`: the UI thread puts it on the clipboard.
    Image(egui::ColorImage),
    Saved { path: PathBuf, note: Option<String> },
}

pub(crate) struct ExportJob {
    pub rx: mpsc::Receiver<ExportEvent>,
    pub cancel: Arc<AtomicBool>,
    pub label: String,
    pub progress: (usize, usize),
}

pub(crate) fn spawn(req: ExportRequest, ctx: egui::Context) -> ExportJob {
    let label = match req.sink {
        Sink::Clipboard => "Copying frame",
        Sink::Png(_) => "Saving frame",
        Sink::Mp4(_) => "Exporting clip",
        Sink::Report(_) => "Exporting report",
    }
    .to_string();
    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_thread = cancel.clone();

    std::thread::Builder::new()
        .name("export".into())
        .spawn(move || {
            let progress = Progress { tx: &tx, ctx: &ctx };
            // A panic must still end the job, or the UI would wait on it forever.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(req, &progress, &cancel_thread)
            }));
            let event = match result {
                Ok(Ok(out)) => ExportEvent::Finished(out),
                Ok(Err(e)) => ExportEvent::Failed(e),
                Err(panic) => {
                    let msg = panic
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| panic.downcast_ref::<&str>().copied())
                        .unwrap_or("unknown error");
                    ExportEvent::Failed(format!("export crashed: {}", msg))
                }
            };
            let _ = tx.send(event);
            ctx.request_repaint();
        })
        .expect("failed to spawn export thread");

    ExportJob {
        rx,
        cancel,
        label,
        progress: (0, 0),
    }
}

struct Progress<'a> {
    tx: &'a mpsc::Sender<ExportEvent>,
    ctx: &'a egui::Context,
}

impl Progress<'_> {
    fn report(&self, done: usize, total: usize) {
        let _ = self.tx.send(ExportEvent::Progress { done, total });
        self.ctx.request_repaint();
    }
}

const CANCELLED: &str = "cancelled";

fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Relaxed) {
        Err(CANCELLED.to_string())
    } else {
        Ok(())
    }
}

/// Whether an export error is a user cancel rather than a failure.
pub(crate) fn is_cancelled(err: &str) -> bool {
    err == CANCELLED
}

fn run(req: ExportRequest, progress: &Progress, cancel: &AtomicBool) -> Result<ExportOutput, String> {
    if req.cameras.is_empty() {
        return Err("No cameras selected".into());
    }
    if req.layout == Layout::CamerasWithTrajectory && req.trajectory.is_none() {
        return Err("Trajectory is not available for this episode".into());
    }

    match (req.range, &req.sink) {
        (FrameRange::Single(frame), Sink::Clipboard) => {
            let canvas = single_frame(&req, frame, progress, cancel)?;
            Ok(ExportOutput::Image(canvas.to_color_image()))
        }
        (FrameRange::Single(frame), Sink::Png(path)) => {
            let canvas = single_frame(&req, frame, progress, cancel)?;
            write_replacing(path, |tmp| canvas.save_png(tmp))?;
            Ok(ExportOutput::Saved { path: path.clone(), note: None })
        }
        (FrameRange::Whole, Sink::Mp4(path)) => {
            write_replacing(path, |tmp| export_clip(&req, tmp, progress, cancel)).map(|out| match out {
                ExportOutput::Saved { note, .. } => ExportOutput::Saved { path: path.clone(), note },
                other => other,
            })
        }
        (FrameRange::Whole, Sink::Report(parent)) => export_report(&req, parent, progress, cancel),
        _ => Err("Unsupported export combination".into()),
    }
}

/// Run `write` against a hidden sibling of `path` and move the result onto `path`
/// only when it succeeds, so a failed or cancelled export never destroys a file
/// that already exists there. The extension is kept because ffmpeg and the image
/// crate pick the format from it.
fn write_replacing<T>(path: &Path, write: impl FnOnce(&Path) -> Result<T, String>) -> Result<T, String> {
    let name = path
        .file_name()
        .ok_or_else(|| format!("Invalid output path {}", path.display()))?
        .to_string_lossy();
    let ext = path.extension().map(|e| e.to_string_lossy()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{}.partial.{}", name, ext));

    let result = write(&tmp).and_then(|value| {
        std::fs::rename(&tmp, path)
            .map(|()| value)
            .map_err(|e| format!("Failed to write {}: {}", path.display(), e))
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Decodes all cameras in lockstep and composites them onto one canvas.
struct Compositor {
    sources: Vec<EpisodeFrames>,
    /// Latest frame per camera; a camera that ends early keeps its last frame.
    frames: Vec<Option<egui::ColorImage>>,
    /// Whether each camera produced a new frame on the last `advance`.
    fresh: Vec<bool>,
    plan: Plan,
    canvas: Canvas,
    traj: Option<TrajectoryRenderer>,
}

impl Compositor {
    fn new(req: &ExportRequest) -> Result<Self, String> {
        let sources = req
            .cameras
            .iter()
            .map(|c| EpisodeFrames::open(&c.path, c.seek_range, Some(req.episode_length)))
            .collect::<Result<Vec<_>, _>>()?;
        let with_traj = req.layout == Layout::CamerasWithTrajectory;
        let plan = {
            let sizes: Vec<_> = sources.iter().map(EpisodeFrames::size).collect();
            compose::plan(&sizes, with_traj, req.scale, req.rows)
        };
        let traj = match (&plan.traj, &req.trajectory) {
            (Some(rect), Some(input)) => Some(TrajectoryRenderer::new(input.clone(), rect.w, rect.h)?),
            _ => None,
        };
        let canvas = Canvas::new(plan.canvas.0, plan.canvas.1);
        let n = sources.len();
        Ok(Self {
            sources,
            frames: vec![None; n],
            fresh: vec![false; n],
            plan,
            canvas,
            traj,
        })
    }

    /// Decode the next frame of every camera. Returns false once all cameras ended.
    fn advance(&mut self) -> Result<bool, String> {
        let mut any = false;
        for (i, src) in self.sources.iter_mut().enumerate() {
            let next = src.next()?;
            self.fresh[i] = next.is_some();
            if let Some(img) = next {
                self.frames[i] = Some(img);
                any = true;
            }
        }
        Ok(any)
    }

    /// Composite the current camera frames plus the trajectory at `frame`.
    fn compose(&mut self, frame: usize) -> Result<&Canvas, String> {
        self.canvas.clear();
        for (img, rect) in self.frames.iter().zip(&self.plan.tiles) {
            if let Some(img) = img {
                compose::blit(&mut self.canvas, img, *rect);
            }
        }
        if let (Some(traj), Some(rect)) = (&mut self.traj, self.plan.traj) {
            let img = traj.render(frame)?;
            compose::blit(&mut self.canvas, &img, rect);
        }
        Ok(&self.canvas)
    }
}

/// Composite of one episode-relative frame. Decodes sequentially up to it, which
/// is exact (a keyframe seek is not).
fn single_frame(
    req: &ExportRequest,
    frame: usize,
    progress: &Progress,
    cancel: &AtomicBool,
) -> Result<Canvas, String> {
    let mut comp = Compositor::new(req)?;
    let frame = frame.min(req.episode_length.saturating_sub(1));
    let total = frame + 1;
    for done in 0..frame {
        check_cancel(cancel)?;
        for src in &mut comp.sources {
            src.skip(1);
        }
        if done % 30 == 0 {
            progress.report(done, total);
        }
    }
    for (i, src) in comp.sources.iter_mut().enumerate() {
        comp.frames[i] = src.next_or_last()?;
    }
    if comp.frames.iter().all(Option::is_none) {
        return Err("No frames could be decoded".into());
    }
    progress.report(total, total);
    Ok(comp.compose(frame)?.clone())
}

fn export_clip(
    req: &ExportRequest,
    path: &Path,
    progress: &Progress,
    cancel: &AtomicBool,
) -> Result<ExportOutput, String> {
    let mut comp = Compositor::new(req)?;
    let (w, h) = comp.plan.canvas;
    let mut writer = Mp4Writer::create(path, w as u32, h as u32, req.fps)?;
    let codec = writer.codec_name;
    let total = req.episode_length;

    for f in 0..total {
        check_cancel(cancel)?;
        if !comp.advance()? {
            if f == 0 {
                return Err("No frames could be decoded".into());
            }
            break;
        }
        writer.push_rgba(&comp.compose(f)?.rgba)?;
        progress.report(f + 1, total);
    }
    writer.finish()?;
    Ok(ExportOutput::Saved {
        path: path.to_path_buf(),
        note: codec_note(codec),
    })
}

fn codec_note(codec: &str) -> Option<String> {
    (codec != "libx264").then(|| format!("encoded with {} (libx264 not available)", codec))
}

fn export_report(
    req: &ExportRequest,
    parent: &Path,
    progress: &Progress,
    cancel: &AtomicBool,
) -> Result<ExportOutput, String> {
    let meta = req.meta.as_ref().ok_or("Report metadata missing")?;
    let dir = parent.join(report::folder_name(&meta.dataset_name, meta.episode));
    let existed = dir.exists();
    if existed {
        let non_empty = std::fs::read_dir(&dir)
            .map_err(|e| format!("Cannot read {}: {}", dir.display(), e))?
            .next()
            .is_some();
        if non_empty {
            return Err(format!("{} already exists and is not empty", dir.display()));
        }
    }

    let result = write_report(req, meta, &dir, progress, cancel);
    if result.is_err() {
        // Leave the destination as it was: remove what this export created.
        if existed {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
                }
            }
        } else {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    result
}

fn write_report(
    req: &ExportRequest,
    meta: &ReportMeta,
    dir: &Path,
    progress: &Progress,
    cancel: &AtomicBool,
) -> Result<ExportOutput, String> {
    let clips_dir = dir.join("clips");
    std::fs::create_dir_all(&clips_dir)
        .map_err(|e| format!("Failed to create {}: {}", clips_dir.display(), e))?;

    let mut comp = Compositor::new(req)?;

    // Composite clip with the dialog's options.
    let (w, h) = comp.plan.canvas;
    let mut composite = Mp4Writer::create(&dir.join("composite.mp4"), w as u32, h as u32, req.fps)?;
    let codec = composite.codec_name;

    // Raw per-camera clips at native size (rounded down to even for yuv420p).
    let mut clips = Vec::with_capacity(req.cameras.len());
    let mut files = vec!["composite.mp4".to_string()];
    for (cam, src) in req.cameras.iter().zip(&comp.sources) {
        let plan = compose::plan(&[src.size()], false, 1.0, None);
        let canvas = Canvas::new(plan.canvas.0, plan.canvas.1);
        let name = format!("{}.mp4", sanitize_file_name(&cam.display_name));
        let writer = Mp4Writer::create(&clips_dir.join(&name), canvas.width as u32, canvas.height as u32, req.fps)?;
        files.push(format!("clips/{}", name));
        clips.push((writer, canvas, plan.tiles[0]));
    }

    let has_traj_png = req.trajectory.is_some();
    let total = req.episode_length + usize::from(has_traj_png);
    let mut thumbnail: Option<Canvas> = None;

    for f in 0..req.episode_length {
        check_cancel(cancel)?;
        if !comp.advance()? {
            if f == 0 {
                return Err("No frames could be decoded".into());
            }
            break;
        }
        for (i, (writer, canvas, rect)) in clips.iter_mut().enumerate() {
            if let (true, Some(img)) = (comp.fresh[i], &comp.frames[i]) {
                compose::blit(canvas, img, *rect);
                writer.push_rgba(&canvas.rgba)?;
            }
        }
        let canvas = comp.compose(f)?;
        composite.push_rgba(&canvas.rgba)?;
        if f == 0 {
            thumbnail = Some(canvas.clone());
        }
        progress.report(f + 1, total);
    }

    composite.finish()?;
    for (writer, _, _) in clips {
        writer.finish()?;
    }

    if let Some(thumb) = &thumbnail {
        thumb.save_png(&dir.join("thumbnail.png"))?;
        files.push("thumbnail.png".into());
    }

    if let Some(input) = &req.trajectory {
        check_cancel(cancel)?;
        let mut traj = TrajectoryRenderer::new(input.clone(), REPORT_TRAJECTORY_SIDE, REPORT_TRAJECTORY_SIDE)?;
        let img = traj.render(req.episode_length.saturating_sub(1))?;
        let mut canvas = Canvas::new(img.size[0], img.size[1]);
        canvas.rgba.copy_from_slice(img.as_raw());
        canvas.save_png(&dir.join("trajectory.png"))?;
        files.push("trajectory.png".into());
        progress.report(total, total);
    }

    files.extend(["summary.md".to_string(), "metadata.json".to_string()]);
    report::write(dir, meta, &files)?;

    Ok(ExportOutput::Saved {
        path: dir.to_path_buf(),
        note: codec_note(codec),
    })
}

/// Replace characters that are not safe in file names on any platform.
fn sanitize_file_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_replacing_keeps_existing_file_on_failure() {
        let dir = std::env::temp_dir().join(format!("tracelr_replace_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clip.mp4");
        std::fs::write(&path, b"old").unwrap();

        let failed: Result<(), String> = write_replacing(&path, |tmp| {
            assert_eq!(tmp.extension().unwrap(), "mp4");
            std::fs::write(tmp, b"partial").unwrap();
            Err(CANCELLED.into())
        });
        assert!(failed.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old");

        write_replacing(&path, |tmp| std::fs::write(tmp, b"new").map_err(|e| e.to_string())).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1, "no partial file left behind");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
