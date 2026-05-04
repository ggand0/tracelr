use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use eframe::egui;
use serde::{Deserialize, Serialize};

const PALETTE: &[[u8; 3]] = &[
    [220, 60, 60],
    [60, 100, 220],
    [60, 180, 75],
    [240, 160, 40],
    [160, 60, 200],
    [40, 200, 200],
    [220, 120, 180],
    [128, 128, 0],
    [0, 128, 128],
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CurationLabel {
    pub name: String,
    #[serde(with = "color_serde")]
    pub color: egui::Color32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CurationFile {
    labels: Vec<CurationLabel>,
    episodes: BTreeMap<String, Vec<String>>,
}

pub(crate) struct CurationState {
    pub labels: Vec<CurationLabel>,
    pub episodes: BTreeMap<usize, BTreeSet<String>>,
    pub active_label: Option<usize>,
    pub dirty: bool,
    pub new_label_buf: String,
}

impl CurationState {
    pub fn new() -> Self {
        Self {
            labels: Vec::new(),
            episodes: BTreeMap::new(),
            active_label: None,
            dirty: false,
            new_label_buf: String::new(),
        }
    }

    pub fn load(path: &Path) -> Self {
        let file_path = path.join("curation.json");
        if !file_path.exists() {
            return Self::new();
        }
        let Ok(text) = fs::read_to_string(&file_path) else {
            log::warn!("Failed to read {}", file_path.display());
            return Self::new();
        };
        let Ok(file) = serde_json::from_str::<CurationFile>(&text) else {
            log::warn!("Failed to parse {}", file_path.display());
            return Self::new();
        };
        let episodes = file
            .episodes
            .into_iter()
            .filter_map(|(k, v)| {
                k.parse::<usize>()
                    .ok()
                    .map(|idx| (idx, v.into_iter().collect()))
            })
            .collect();
        Self {
            labels: file.labels,
            episodes,
            active_label: None,
            dirty: false,
            new_label_buf: String::new(),
        }
    }

    pub fn save(&mut self, dataset_root: &Path) -> Result<(), std::io::Error> {
        let file = CurationFile {
            labels: self.labels.clone(),
            episodes: self
                .episodes
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .map(|(k, v)| (k.to_string(), v.iter().cloned().collect()))
                .collect(),
        };
        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let path = dataset_root.join("curation.json");
        fs::write(&path, json)?;
        self.dirty = false;
        log::info!("Curation saved to {}", path.display());
        Ok(())
    }

    pub fn add_label(&mut self, name: String) {
        if self.labels.iter().any(|l| l.name == name) {
            return;
        }
        let color_idx = self.labels.len() % PALETTE.len();
        let [r, g, b] = PALETTE[color_idx];
        self.labels.push(CurationLabel {
            name,
            color: egui::Color32::from_rgb(r, g, b),
        });
        self.dirty = true;
    }

    pub fn toggle_label(&mut self, episode: usize, label_index: usize) {
        let Some(label) = self.labels.get(label_index) else {
            return;
        };
        let name = label.name.clone();
        let entry = self.episodes.entry(episode).or_default();
        if !entry.remove(&name) {
            entry.insert(name);
        }
        self.dirty = true;
    }

    pub fn episode_labels(&self, episode: usize) -> Option<&BTreeSet<String>> {
        self.episodes.get(&episode)
    }

    pub fn label_count(&self, label_index: usize) -> usize {
        let Some(label) = self.labels.get(label_index) else {
            return 0;
        };
        self.episodes
            .values()
            .filter(|s| s.contains(&label.name))
            .count()
    }

    pub fn color_for_label(&self, name: &str) -> egui::Color32 {
        self.labels
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.color)
            .unwrap_or(egui::Color32::GRAY)
    }
}

mod color_serde {
    use eframe::egui;
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(color: &egui::Color32, s: S) -> Result<S::Ok, S::Error> {
        let arr = [color.r(), color.g(), color.b()];
        s.serialize_some(&arr)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<egui::Color32, D::Error> {
        let arr: [u8; 3] = Deserialize::deserialize(d)?;
        Ok(egui::Color32::from_rgb(arr[0], arr[1], arr[2]))
    }
}
