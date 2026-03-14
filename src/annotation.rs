use std::collections::HashMap;
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
    pub annotations: HashMap<usize, usize>,
    pub prompts: Vec<PromptCard>,
    pub dirty: bool,
}

#[derive(Serialize, Deserialize)]
struct AnnotationFile {
    dataset_root: String,
    prompts: Vec<String>,
    annotations: HashMap<String, usize>,
}

impl AnnotationState {
    /// Create annotation state for the cube organization task (4 color prompts).
    pub fn new_cube_task() -> Self {
        Self {
            annotations: HashMap::new(),
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
            annotations: self
                .annotations
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        };
        let json = serde_json::to_string_pretty(&file)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load annotations from a JSON file.
    pub fn load_json(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        let file: AnnotationFile = serde_json::from_str(&text)?;
        self.annotations = file
            .annotations
            .into_iter()
            .map(|(k, v)| (k.parse::<usize>().unwrap_or(0), v))
            .collect();
        self.dirty = false;
        log::info!("Loaded {} annotations from {}", self.annotations.len(), path.display());
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
        log::info!("Exported {} tasks to {}", self.prompts.len(), tasks_path.display());

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
