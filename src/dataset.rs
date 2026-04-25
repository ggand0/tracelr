use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Parsed LeRobot dataset metadata (supports v2.1 and v3.0).
#[allow(dead_code)]
pub(crate) struct LeRobotDataset {
    pub root: PathBuf,
    pub info: DatasetInfo,
    pub episodes: Vec<EpisodeMeta>,
    pub tasks: Vec<TaskMeta>,
}

#[allow(dead_code)]
pub(crate) struct DatasetInfo {
    pub fps: u32,
    pub total_episodes: usize,
    pub total_frames: usize,
    pub video_keys: Vec<String>,
    pub video_path_template: String,
    pub chunks_size: usize,
    pub codebase_version: String,
    pub robot_type: Option<String>,
    /// Joint/state feature names from observation.state (e.g. ["shoulder_pan.pos", ...])
    pub state_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct EpisodeMeta {
    pub episode_index: usize,
    pub tasks: Vec<String>,
    pub length: usize,
    /// v3.0: per-video-key mapping to chunk/file index and timestamp range.
    /// Key = video_key, Value = (chunk_index, file_index, from_timestamp, to_timestamp)
    pub video_mapping: HashMap<String, VideoMapping>,
    /// v3.0: which data parquet file holds this episode's observation/action data.
    pub data_chunk_index: usize,
    pub data_file_index: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct VideoMapping {
    pub chunk_index: usize,
    pub file_index: usize,
    pub from_timestamp: f64,
    pub to_timestamp: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TaskMeta {
    pub task_index: usize,
    pub task: String,
}

// -- Raw serde types for JSON parsing --

#[derive(Deserialize)]
struct RawInfo {
    #[serde(default)]
    codebase_version: Option<String>,
    #[serde(default)]
    robot_type: Option<String>,
    fps: u32,
    total_episodes: usize,
    total_frames: usize,
    chunks_size: Option<usize>,
    video_path: Option<String>,
    features: Option<HashMap<String, RawFeature>>,
}

#[derive(Deserialize, Clone)]
struct RawFeature {
    dtype: String,
    #[serde(default)]
    names: Option<Vec<String>>,
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
    /// Load a LeRobot dataset from its root directory.
    /// Supports both v2.1 (episodes.jsonl) and v3.0 (episodes parquet) formats.
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

        let codebase_version = raw_info
            .codebase_version
            .clone()
            .unwrap_or_else(|| "v2.1".to_string());

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

        // Extract state feature names (e.g. ["shoulder_pan.pos", ...])
        let state_names = raw_info
            .features
            .as_ref()
            .and_then(|feats| feats.get("observation.state"))
            .and_then(|f| f.names.clone())
            .unwrap_or_default();

        let info = DatasetInfo {
            fps: raw_info.fps,
            total_episodes: raw_info.total_episodes,
            total_frames: raw_info.total_frames,
            video_keys: video_keys.clone(),
            video_path_template: raw_info.video_path.unwrap_or_else(|| {
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
                    .to_string()
            }),
            chunks_size: raw_info.chunks_size.unwrap_or(1000),
            codebase_version: codebase_version.clone(),
            robot_type: raw_info.robot_type,
            state_names,
        };

        // Load episodes — v3.0 uses parquet, v2.1 uses jsonl
        let episodes = if codebase_version.starts_with("v3") {
            load_episodes_v3(root, &meta_dir, &video_keys, raw_info.total_episodes)?
        } else {
            load_episodes_v2(&meta_dir, raw_info.total_episodes)?
        };

        // Parse tasks — v3.0 uses tasks.parquet, v2.1 uses tasks.jsonl
        let tasks_jsonl = meta_dir.join("tasks.jsonl");
        let tasks = if tasks_jsonl.exists() {
            let text = fs::read_to_string(&tasks_jsonl)
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
            "Loaded {} dataset: {} episodes, {} tasks, {} video keys, {}fps",
            codebase_version,
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
        // v3.0: use per-episode video mapping if available
        if let Some(ep) = self.episodes.get(episode_index) {
            if let Some(mapping) = ep.video_mapping.get(video_key) {
                let path_str = self
                    .info
                    .video_path_template
                    .replace("{video_key}", video_key)
                    .replace("{chunk_index:03d}", &format!("{:03}", mapping.chunk_index))
                    .replace("{file_index:03d}", &format!("{:03}", mapping.file_index));
                return self.root.join(path_str);
            }
        }

        // v2.1 fallback: derive path from episode index
        let chunk = episode_index / self.info.chunks_size;
        let path_str = self
            .info
            .video_path_template
            .replace("{episode_chunk:03d}", &format!("{:03}", chunk))
            .replace("{video_key}", video_key)
            .replace("{episode_index:06d}", &format!("{:06}", episode_index));
        self.root.join(path_str)
    }

    /// Get the timestamp range for an episode within a concatenated video (v3.0).
    /// Returns (from_timestamp, to_timestamp) in seconds, or (0, duration) for v2.1.
    pub fn episode_time_range(&self, episode_index: usize, video_key: &str) -> (f64, f64) {
        if let Some(ep) = self.episodes.get(episode_index) {
            if let Some(mapping) = ep.video_mapping.get(video_key) {
                return (mapping.from_timestamp, mapping.to_timestamp);
            }
            // v2.1: episode is the entire video
            let duration = ep.length as f64 / self.info.fps as f64;
            return (0.0, duration);
        }
        (0.0, 0.0)
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

/// Strip the "observation.images." prefix from a video key for display.
pub(crate) fn camera_display_name(video_key: &str) -> &str {
    video_key
        .strip_prefix("observation.images.")
        .unwrap_or(video_key)
}

/// Load episodes from v2.1 format (episodes.jsonl).
fn load_episodes_v2(meta_dir: &Path, total_episodes: usize) -> Result<Vec<EpisodeMeta>, String> {
    let episodes_path = meta_dir.join("episodes.jsonl");
    if episodes_path.exists() {
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
                    video_mapping: HashMap::new(),
                    data_chunk_index: 0,
                    data_file_index: 0,
                })
            })
            .collect()
    } else {
        Ok((0..total_episodes)
            .map(|i| EpisodeMeta {
                episode_index: i,
                tasks: vec![],
                length: 0,
                video_mapping: HashMap::new(),
                data_chunk_index: 0,
                data_file_index: 0,
            })
            .collect())
    }
}

/// Load episodes from v3.0 format (parquet files under meta/episodes/).
fn load_episodes_v3(
    _root: &Path,
    meta_dir: &Path,
    video_keys: &[String],
    total_episodes: usize,
) -> Result<Vec<EpisodeMeta>, String> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let episodes_dir = meta_dir.join("episodes");
    if !episodes_dir.is_dir() {
        // Fallback to generating from total_episodes
        log::warn!("No meta/episodes/ directory, generating {} episodes", total_episodes);
        return Ok((0..total_episodes)
            .map(|i| EpisodeMeta {
                episode_index: i,
                tasks: vec![],
                length: 0,
                video_mapping: HashMap::new(),
                data_chunk_index: 0,
                data_file_index: 0,
            })
            .collect());
    }

    // Find all parquet files under meta/episodes/
    let mut parquet_files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&episodes_dir).map_err(|e| format!("Read episodes dir: {}", e))? {
        let entry = entry.map_err(|e| format!("Read entry: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            // chunk directory
            for sub in fs::read_dir(&path).map_err(|e| format!("Read chunk dir: {}", e))? {
                let sub = sub.map_err(|e| format!("Read sub: {}", e))?;
                if sub.path().extension().map(|e| e == "parquet").unwrap_or(false) {
                    parquet_files.push(sub.path());
                }
            }
        } else if path.extension().map(|e| e == "parquet").unwrap_or(false) {
            parquet_files.push(path);
        }
    }
    parquet_files.sort();

    let mut episodes: Vec<EpisodeMeta> = Vec::new();

    for pq_path in &parquet_files {
        let file = fs::File::open(pq_path)
            .map_err(|e| format!("Open {}: {}", pq_path.display(), e))?;
        let reader = SerializedFileReader::new(file)
            .map_err(|e| format!("Read parquet {}: {}", pq_path.display(), e))?;

        let schema = reader.metadata().file_metadata().schema();
        let _has_field = |name: &str| schema.get_fields().iter().any(|f| f.name() == name);

        for row in reader.get_row_iter(None)
            .map_err(|e| format!("Row iter: {}", e))?
        {
            let row = row.map_err(|e| format!("Read row: {}", e))?;
            let mut ep_index = 0usize;
            let mut length = 0usize;
            let mut data_chunk_index = 0usize;
            let mut data_file_index = 0usize;
            let mut video_mapping = HashMap::new();

            for (name, field) in row.get_column_iter() {
                match name.as_str() {
                    "episode_index" => {
                        if let parquet::record::Field::Long(v) = field {
                            ep_index = *v as usize;
                        }
                    }
                    "length" => {
                        if let parquet::record::Field::Long(v) = field {
                            length = *v as usize;
                        }
                    }
                    "data/chunk_index" => {
                        if let parquet::record::Field::Long(v) = field {
                            data_chunk_index = *v as usize;
                        }
                    }
                    "data/file_index" => {
                        if let parquet::record::Field::Long(v) = field {
                            data_file_index = *v as usize;
                        }
                    }
                    _ => {}
                }
            }

            // Extract video mapping for each video key
            for vk in video_keys {
                let chunk_col = format!("videos/{}/chunk_index", vk);
                let file_col = format!("videos/{}/file_index", vk);
                let from_col = format!("videos/{}/from_timestamp", vk);
                let to_col = format!("videos/{}/to_timestamp", vk);

                let mut chunk_idx = 0usize;
                let mut file_idx = 0usize;
                let mut from_ts = 0.0f64;
                let mut to_ts = 0.0f64;

                for (name, field) in row.get_column_iter() {
                    if name == &chunk_col {
                        if let parquet::record::Field::Long(v) = field {
                            chunk_idx = *v as usize;
                        }
                    } else if name == &file_col {
                        if let parquet::record::Field::Long(v) = field {
                            file_idx = *v as usize;
                        }
                    } else if name == &from_col {
                        if let parquet::record::Field::Double(v) = field {
                            from_ts = *v;
                        }
                    } else if name == &to_col {
                        if let parquet::record::Field::Double(v) = field {
                            to_ts = *v;
                        }
                    }
                }

                if to_ts > from_ts {
                    video_mapping.insert(
                        vk.clone(),
                        VideoMapping {
                            chunk_index: chunk_idx,
                            file_index: file_idx,
                            from_timestamp: from_ts,
                            to_timestamp: to_ts,
                        },
                    );
                }
            }

            episodes.push(EpisodeMeta {
                episode_index: ep_index,
                tasks: vec![],
                length,
                video_mapping,
                data_chunk_index,
                data_file_index,
            });
        }
    }

    episodes.sort_by_key(|e| e.episode_index);
    Ok(episodes)
}

/// Check if a directory looks like a LeRobot dataset (has meta/info.json).
pub(crate) fn is_lerobot_dataset(path: &Path) -> bool {
    path.is_dir() && path.join("meta").join("info.json").is_file()
}
