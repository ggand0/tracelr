use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Episode and dataset facts written into `summary.md` and `metadata.json`.
#[derive(Clone, serde::Serialize)]
pub(crate) struct ReportMeta {
    pub dataset_name: String,
    pub dataset_root: PathBuf,
    pub codebase_version: String,
    pub robot_type: Option<String>,
    pub episode: usize,
    pub task: String,
    pub fps: u32,
    pub length: usize,
    pub duration_s: f64,
    pub cameras: Vec<String>,
    pub curation_labels: Vec<String>,
    pub annotation: Option<AnnotationMeta>,
    pub app_version: String,
    pub exported_at: String,
}

#[derive(Clone, serde::Serialize)]
pub(crate) struct AnnotationMeta {
    pub label: String,
    pub prompt: String,
}

/// Name of the report folder for an episode, e.g. `my_dataset_ep0005`.
pub(crate) fn folder_name(dataset_name: &str, episode: usize) -> String {
    format!("{}_ep{:04}", dataset_name, episode)
}

pub(crate) fn summary_markdown(meta: &ReportMeta, files: &[String]) -> String {
    let mut s = format!("# {} / episode {}\n\n", meta.dataset_name, meta.episode);
    s.push_str(&format!("- Task: {}\n", meta.task));
    s.push_str(&format!(
        "- Frames: {} ({:.1} s @ {} fps)\n",
        meta.length, meta.duration_s, meta.fps
    ));
    s.push_str(&format!("- Cameras: {}\n", meta.cameras.join(", ")));
    if !meta.curation_labels.is_empty() {
        s.push_str(&format!("- Labels: {}\n", meta.curation_labels.join(", ")));
    }
    if let Some(a) = &meta.annotation {
        s.push_str(&format!("- Annotation: {}: {}\n", a.label, a.prompt));
    }
    s.push_str(&format!(
        "- Robot: {}, dataset {}\n",
        meta.robot_type.as_deref().unwrap_or("unknown"),
        meta.codebase_version
    ));
    s.push_str(&format!(
        "- Exported with tracelr {} on {}\n",
        meta.app_version,
        meta.exported_at.get(..10).unwrap_or(&meta.exported_at)
    ));
    s.push_str(&format!("\nFiles: {}\n", files.join(", ")));
    s
}

/// Write `summary.md` and `metadata.json` into `dir`.
pub(crate) fn write(dir: &Path, meta: &ReportMeta, files: &[String]) -> Result<(), String> {
    let summary = dir.join("summary.md");
    std::fs::write(&summary, summary_markdown(meta, files))
        .map_err(|e| format!("Failed to write {}: {}", summary.display(), e))?;

    let json = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    let metadata = dir.join("metadata.json");
    std::fs::write(&metadata, json)
        .map_err(|e| format!("Failed to write {}: {}", metadata.display(), e))
}

/// Current UTC time as RFC 3339 (`2026-10-16T09:30:00Z`).
pub(crate) fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    rfc3339_utc(secs)
}

fn rfc3339_utc(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let rem = unix_secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Days since 1970-01-01 to (year, month, day), Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ReportMeta {
        ReportMeta {
            dataset_name: "pick_cube".into(),
            dataset_root: PathBuf::from("/data/pick_cube"),
            codebase_version: "v3.0".into(),
            robot_type: Some("so101_follower".into()),
            episode: 5,
            task: "Pick the cube".into(),
            fps: 30,
            length: 300,
            duration_s: 10.0,
            cameras: vec!["front".into(), "wrist".into()],
            curation_labels: vec![],
            annotation: None,
            app_version: "0.1.2".into(),
            exported_at: "2026-10-16T09:30:00Z".into(),
        }
    }

    #[test]
    fn rfc3339_known_dates() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_utc(1_791_970_245), "2026-10-14T09:30:45Z");
    }

    #[test]
    fn summary_omits_empty_labels_and_annotation() {
        let s = summary_markdown(&meta(), &["composite.mp4".into()]);
        assert!(s.starts_with("# pick_cube / episode 5\n"));
        assert!(s.contains("- Frames: 300 (10.0 s @ 30 fps)\n"));
        assert!(s.contains("- Cameras: front, wrist\n"));
        assert!(!s.contains("Labels"));
        assert!(!s.contains("Annotation"));
        assert!(s.contains("on 2026-10-16\n"));
    }

    #[test]
    fn summary_includes_labels_and_annotation() {
        let mut m = meta();
        m.curation_labels = vec!["good".into(), "slow".into()];
        m.annotation = Some(AnnotationMeta {
            label: "A".into(),
            prompt: "grasp from left".into(),
        });
        let s = summary_markdown(&m, &[]);
        assert!(s.contains("- Labels: good, slow\n"));
        assert!(s.contains("- Annotation: A: grasp from left\n"));
    }
}
