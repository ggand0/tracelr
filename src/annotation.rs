use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use eframe::egui;
use serde::{Deserialize, Serialize};

use crate::dataset::LeRobotDataset;

/// A single prompt option for annotation.
#[derive(Debug, Clone)]
pub(crate) struct PromptCard {
    pub label: String,
    pub prompt: String,
    pub color: egui::Color32,
}

/// Annotation state: maps episode indices to prompt indices.
pub(crate) struct AnnotationState {
    pub annotations: BTreeMap<usize, usize>,
    pub prompts: Vec<PromptCard>,
    pub dirty: bool,
}

// -- YAML prompt config --

#[derive(Serialize, Deserialize)]
struct PromptsConfig {
    prompts: Vec<PromptEntry>,
}

#[derive(Serialize, Deserialize)]
struct PromptEntry {
    label: String,
    prompt: String,
    #[serde(default = "default_color")]
    color: [u8; 3],
}

fn default_color() -> [u8; 3] {
    [140, 140, 140]
}

// -- Annotation JSON file --

/// On-disk annotation format. `annotations` maps integer episode index
/// to prompt index. serde_json renders BTreeMap<usize, usize> with
/// integer-sortable string keys ("0", "1", "2", ..., "10", "11").
/// We use a custom serializer to keep keys as integers in the JSON.
#[derive(Serialize, Deserialize)]
struct AnnotationFile {
    dataset_root: String,
    prompts: Vec<String>,
    /// episode_index → prompt_index, sorted by episode.
    #[serde(serialize_with = "serialize_int_keys", deserialize_with = "deserialize_int_keys")]
    annotations: BTreeMap<usize, usize>,
}

fn serialize_int_keys<S: serde::Serializer>(
    map: &BTreeMap<usize, usize>,
    s: S,
) -> Result<S::Ok, S::Error> {
    // Serialize as a JSON object with string keys (JSON requirement)
    // but BTreeMap<usize> ensures numeric order
    use serde::ser::SerializeMap;
    let mut m = s.serialize_map(Some(map.len()))?;
    for (k, v) in map {
        m.serialize_entry(&k.to_string(), v)?;
    }
    m.end()
}

fn deserialize_int_keys<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<usize, usize>, D::Error> {
    let string_map: BTreeMap<String, usize> = BTreeMap::deserialize(d)?;
    Ok(string_map
        .into_iter()
        .filter_map(|(k, v)| k.parse::<usize>().ok().map(|k| (k, v)))
        .collect())
}

impl AnnotationState {
    /// Load prompts from YAML config. Search order:
    /// 1. `<dataset_dir>/prompts.yaml`
    /// 2. `~/.config/lerobot-explorer/prompts.yaml`
    /// 3. Hardcoded cube task defaults
    pub fn load_prompts(dataset_root: Option<&Path>) -> Self {
        // 1. Dataset-specific prompts
        if let Some(root) = dataset_root {
            let dataset_prompts = root.join("prompts.yaml");
            if dataset_prompts.exists() {
                if let Ok(state) = Self::from_yaml(&dataset_prompts) {
                    log::info!("Loaded prompts from {}", dataset_prompts.display());
                    return state;
                }
            }
        }

        // 2. User default prompts
        if let Some(config_dir) = dirs::config_dir() {
            let user_prompts = config_dir.join("lerobot-explorer").join("prompts.yaml");
            if user_prompts.exists() {
                if let Ok(state) = Self::from_yaml(&user_prompts) {
                    log::info!("Loaded prompts from {}", user_prompts.display());
                    return state;
                }
            }
        }

        // 3. Hardcoded defaults
        log::info!("Using default cube task prompts");
        Self::default_cube_task()
    }

    /// Parse prompts from a YAML file.
    fn from_yaml(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        let config: PromptsConfig = serde_yaml::from_str(&text)?;

        let prompts = config
            .prompts
            .into_iter()
            .map(|e| PromptCard {
                label: e.label,
                prompt: e.prompt,
                color: egui::Color32::from_rgb(e.color[0], e.color[1], e.color[2]),
            })
            .collect();

        Ok(Self {
            annotations: BTreeMap::new(),
            prompts,
            dirty: false,
        })
    }

    /// Empty state for viewer-only mode (no annotation).
    pub fn default_empty() -> Self {
        Self {
            annotations: BTreeMap::new(),
            prompts: Vec::new(),
            dirty: false,
        }
    }

    /// Hardcoded cube organization task prompts.
    fn default_cube_task() -> Self {
        Self {
            annotations: BTreeMap::new(),
            prompts: vec![
                PromptCard {
                    label: "Red cube".into(),
                    prompt: "Pick up the red cube and place it in the bowl".into(),
                    color: egui::Color32::from_rgb(220, 60, 60),
                },
                PromptCard {
                    label: "Orange cube".into(),
                    prompt: "Pick up the orange cube and place it in the bowl".into(),
                    color: egui::Color32::from_rgb(240, 160, 40),
                },
                PromptCard {
                    label: "Yellow cube".into(),
                    prompt: "Pick up the yellow cube and place it in the bowl".into(),
                    color: egui::Color32::from_rgb(240, 220, 40),
                },
                PromptCard {
                    label: "Green cube".into(),
                    prompt: "Pick up the green cube and place it in the bowl".into(),
                    color: egui::Color32::from_rgb(60, 180, 75),
                },
            ],
            dirty: false,
        }
    }

    /// Assign a prompt to an episode.
    pub fn set(&mut self, episode_index: usize, prompt_index: usize) {
        if prompt_index < self.prompts.len() {
            self.annotations.insert(episode_index, prompt_index);
            self.dirty = true;
        }
    }

    /// Remove annotation for an episode.
    pub fn clear(&mut self, episode_index: usize) {
        if self.annotations.remove(&episode_index).is_some() {
            self.dirty = true;
        }
    }

    /// Get the assigned prompt index for an episode, if any.
    pub fn get(&self, episode_index: usize) -> Option<usize> {
        self.annotations.get(&episode_index).copied()
    }

    /// (annotated_count, total_episodes)
    pub fn progress(&self, total_episodes: usize) -> (usize, usize) {
        (self.annotations.len(), total_episodes)
    }

    /// Save annotations to a JSON file.
    pub fn save_json(
        &self,
        path: &Path,
        dataset_root: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = AnnotationFile {
            dataset_root: dataset_root.to_string(),
            prompts: self.prompts.iter().map(|p| p.prompt.clone()).collect(),
            annotations: self.annotations.clone(),
        };
        let json = serde_json::to_string_pretty(&file)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load annotations from a JSON file.
    pub fn load_json(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        let file: AnnotationFile = serde_json::from_str(&text)?;
        self.annotations = file.annotations;
        self.dirty = false;
        log::info!(
            "Loaded {} annotations from {}",
            self.annotations.len(),
            path.display()
        );
        Ok(())
    }

    /// Export annotations in LeRobot tasks.jsonl + episodes.jsonl format.
    pub fn export_lerobot(
        &self,
        dataset: &LeRobotDataset,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let meta_dir = dataset.root.join("meta");

        // Write tasks.jsonl
        let tasks_path = meta_dir.join("tasks.jsonl");
        let mut tasks_lines = Vec::new();
        for (i, prompt) in self.prompts.iter().enumerate() {
            let line = serde_json::json!({
                "task_index": i,
                "task": prompt.prompt,
            });
            tasks_lines.push(serde_json::to_string(&line)?);
        }
        fs::write(&tasks_path, tasks_lines.join("\n") + "\n")?;
        log::info!(
            "Exported {} tasks to {}",
            self.prompts.len(),
            tasks_path.display()
        );

        // Write episodes.jsonl with updated task assignments
        let episodes_path = meta_dir.join("episodes.jsonl");
        let mut episode_lines = Vec::new();
        for ep in &dataset.episodes {
            let task_name = self
                .get(ep.episode_index)
                .and_then(|idx| self.prompts.get(idx))
                .map(|p| p.prompt.clone())
                .unwrap_or_default();

            let line = serde_json::json!({
                "episode_index": ep.episode_index,
                "tasks": [task_name],
                "length": ep.length,
            });
            episode_lines.push(serde_json::to_string(&line)?);
        }
        fs::write(&episodes_path, episode_lines.join("\n") + "\n")?;
        log::info!(
            "Exported {} episodes to {}",
            dataset.episodes.len(),
            episodes_path.display()
        );

        Ok(())
    }
}
