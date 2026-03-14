use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Parsed LeRobot v2.1 dataset metadata.
#[allow(dead_code)] // tasks used by annotation export
pub(crate) struct LeRobotDataset {
    pub root: PathBuf,
    pub info: DatasetInfo,
    pub episodes: Vec<EpisodeMeta>,
    pub tasks: Vec<TaskMeta>,
}

#[allow(dead_code)] // metadata fields used in info panel expansions
pub(crate) struct DatasetInfo {
    pub fps: u32,
    pub total_episodes: usize,
    pub total_frames: usize,
    pub video_keys: Vec<String>,
    pub video_path_template: String,
    pub chunks_size: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EpisodeMeta {
    pub episode_index: usize,
    pub tasks: Vec<String>,
    pub length: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // fields used by annotation export
pub(crate) struct TaskMeta {
    pub task_index: usize,
    pub task: String,
}

// -- Raw serde types for JSON parsing --

#[derive(Deserialize)]
struct RawInfo {
    fps: u32,
    total_episodes: usize,
    total_frames: usize,
    chunks_size: Option<usize>,
    video_path: Option<String>,
    features: Option<HashMap<String, RawFeature>>,
}

#[derive(Deserialize)]
struct RawFeature {
    dtype: String,
}

#[derive(Deserialize)]
struct RawEpisode {
    episode_index: usize,
    tasks: Vec<String>,
    length: usize,
}

#[derive(Deserialize)]
struct RawTask {
    task_index: usize,
    task: String,
}

impl LeRobotDataset {
    /// Load a LeRobot v2.1 dataset from its root directory.
    /// Expects `meta/info.json` and `meta/episodes.jsonl` to exist.
    pub fn load(root: &Path) -> Result<Self, String> {
        let meta_dir = root.join("meta");
        if !meta_dir.is_dir() {
            return Err(format!("No meta/ directory found in {}", root.display()));
        }

        // Parse info.json
        let info_path = meta_dir.join("info.json");
        let info_text = fs::read_to_string(&info_path)
            .map_err(|e| format!("Failed to read {}: {}", info_path.display(), e))?;
        let raw_info: RawInfo = serde_json::from_str(&info_text)
            .map_err(|e| format!("Failed to parse info.json: {}", e))?;

        // Extract video keys from features (dtype == "video")
        let video_keys: Vec<String> = raw_info
            .features
            .as_ref()
            .map(|feats| {
                let mut keys: Vec<String> = feats
                    .iter()
                    .filter(|(_, f)| f.dtype == "video")
                    .map(|(k, _)| k.clone())
                    .collect();
                keys.sort();
                keys
            })
            .unwrap_or_default();

        let info = DatasetInfo {
            fps: raw_info.fps,
            total_episodes: raw_info.total_episodes,
            total_frames: raw_info.total_frames,
            video_keys,
            video_path_template: raw_info.video_path.unwrap_or_else(|| {
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                    .to_string()
            }),
            chunks_size: raw_info.chunks_size.unwrap_or(1000),
        };

        // Parse episodes.jsonl
        let episodes_path = meta_dir.join("episodes.jsonl");
        let episodes = if episodes_path.exists() {
            let text = fs::read_to_string(&episodes_path)
                .map_err(|e| format!("Failed to read episodes.jsonl: {}", e))?;
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let raw: RawEpisode = serde_json::from_str(line)
                        .map_err(|e| format!("Failed to parse episode line: {}", e))?;
                    Ok(EpisodeMeta {
                        episode_index: raw.episode_index,
                        tasks: raw.tasks,
                        length: raw.length,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            // Fallback: generate from total_episodes
            (0..raw_info.total_episodes)
                .map(|i| EpisodeMeta {
                    episode_index: i,
                    tasks: vec![],
                    length: 0,
                })
                .collect()
        };

        // Parse tasks.jsonl (optional)
        let tasks_path = meta_dir.join("tasks.jsonl");
        let tasks = if tasks_path.exists() {
            let text = fs::read_to_string(&tasks_path)
                .map_err(|e| format!("Failed to read tasks.jsonl: {}", e))?;
            text.lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let raw: RawTask = serde_json::from_str(line)
                        .map_err(|e| format!("Failed to parse task line: {}", e))?;
                    Ok(TaskMeta {
                        task_index: raw.task_index,
                        task: raw.task,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        } else {
            vec![]
        };

        log::info!(
            "Loaded dataset: {} episodes, {} tasks, {} video keys, {}fps",
            episodes.len(),
            tasks.len(),
            info.video_keys.len(),
            info.fps,
        );

        Ok(Self {
            root: root.to_path_buf(),
            info,
            episodes,
            tasks,
        })
    }

    /// Build the video file path for a given episode and camera key.
    pub fn video_path(&self, episode_index: usize, video_key: &str) -> PathBuf {
        let chunk = episode_index / self.info.chunks_size;
        let path_str = self
            .info
            .video_path_template
            .replace("{episode_chunk:03d}", &format!("{:03}", chunk))
            .replace("{video_key}", video_key)
            .replace("{episode_index:06d}", &format!("{:06}", episode_index));
        self.root.join(path_str)
    }

    /// Duration of an episode in seconds.
    #[allow(dead_code)]
    pub fn episode_duration(&self, episode_index: usize) -> f64 {
        self.episodes
            .get(episode_index)
            .map(|ep| ep.length as f64 / self.info.fps as f64)
            .unwrap_or(0.0)
    }
}

/// Check if a directory looks like a LeRobot dataset (has meta/info.json).
pub(crate) fn is_lerobot_dataset(path: &Path) -> bool {
    path.is_dir() && path.join("meta").join("info.json").is_file()
}
