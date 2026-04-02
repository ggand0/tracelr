use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use arrow::array::Array;

/// Computed EE trajectory for a single episode.
#[derive(Clone)]
pub(crate) struct EeTrajectory {
    /// Per-frame end-effector position [x, y, z] in meters.
    pub positions: Vec<[f64; 3]>,
}

/// Robot kinematics context loaded from a URDF file.
/// Wraps the `k` crate's serial chain for FK computation.
pub(crate) struct RobotKinematics {
    serial: k::SerialChain<f64>,
    /// Joint names in the serial chain (movable joints only), in order.
    joint_names: Vec<String>,
}

impl RobotKinematics {
    /// Load a URDF and build the kinematic chain.
    /// If `ee_frame` is Some, use that link; otherwise auto-detect the deepest leaf link.
    pub fn from_urdf(urdf_path: &Path, ee_frame: Option<&str>) -> Result<Self, String> {
        let chain = k::Chain::<f64>::from_urdf_file(urdf_path)
            .map_err(|e| format!("Failed to load URDF {}: {}", urdf_path.display(), e))?;

        let auto_leaf;
        let ee_link = if let Some(name) = ee_frame {
            chain
                .find_link(name)
                .ok_or_else(|| format!("EE frame '{}' not found in URDF", name))?
        } else {
            // Auto-detect: find the leaf node with the longest chain (most ancestors)
            auto_leaf = find_deepest_leaf(&chain)?;
            &auto_leaf
        };

        let serial = k::SerialChain::from_end(ee_link);

        let joint_names: Vec<String> = serial
            .iter()
            .filter(|n| {
                matches!(
                    n.joint().joint_type,
                    k::joint::JointType::Rotational { .. }
                )
            })
            .map(|n| n.joint().name.clone())
            .collect();

        let ee_name = ee_link.joint().name.clone();
        log::info!(
            "Loaded URDF: {} -> {} (DOF={}, joints={:?})",
            urdf_path.display(),
            ee_name,
            serial.dof(),
            joint_names,
        );

        Ok(Self {
            serial,
            joint_names,
        })
    }

    /// Compute FK for one set of joint angles (in degrees).
    /// Returns EE [x, y, z] position in meters.
    pub fn forward_kinematics_deg(&self, joint_angles_deg: &[f64]) -> [f64; 3] {
        let dof = self.serial.dof();
        let mut positions = vec![0.0f64; dof];
        for (i, &deg) in joint_angles_deg.iter().enumerate() {
            if i >= dof {
                break;
            }
            positions[i] = deg.to_radians();
        }
        // Use unchecked — real robot data can slightly exceed URDF limits
        self.serial.set_joint_positions_unchecked(&positions);
        self.serial.update_transforms();

        let t = self.serial.end_transform().translation;
        [t.x, t.y, t.z]
    }

    /// Compute EE trajectory for a full episode of joint states.
    /// `pos_indices` specifies which indices in each state row are joint positions (degrees).
    /// If empty, uses the first `dof` values (backwards compat for SO101-style data).
    pub fn compute_trajectory(&self, states: &[Vec<f32>], pos_indices: &[usize]) -> EeTrajectory {
        let positions: Vec<[f64; 3]> = states
            .iter()
            .map(|state| {
                let angles: Vec<f64> = if pos_indices.is_empty() {
                    state.iter().map(|v| *v as f64).collect()
                } else {
                    pos_indices.iter().map(|&i| state.get(i).copied().unwrap_or(0.0) as f64).collect()
                };
                self.forward_kinematics_deg(&angles)
            })
            .collect();
        EeTrajectory { positions }
    }

    pub fn joint_names(&self) -> &[String] {
        &self.joint_names
    }

    pub fn dof(&self) -> usize {
        self.serial.dof()
    }
}

/// Load `observation.state` from a parquet data file.
/// For v3.0 shared files, filters rows by `episode_index`.
/// `filter_episode` should be Some(idx) for v3.0, None for v2.1 (entire file is one episode).
pub(crate) fn load_episode_states(parquet_path: &Path, filter_episode: Option<usize>) -> Result<Vec<Vec<f32>>, String> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(parquet_path)
        .map_err(|e| format!("Open {}: {}", parquet_path.display(), e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Parquet reader {}: {}", parquet_path.display(), e))?;

    let mut reader = builder.build()
        .map_err(|e| format!("Build reader {}: {}", parquet_path.display(), e))?;

    let mut all_states: Vec<Vec<f32>> = Vec::new();

    while let Some(batch) = reader.next() {
        let batch = batch.map_err(|e| format!("Read batch: {}", e))?;

        // For v3.0: filter rows by episode_index column
        let row_mask: Option<Vec<bool>> = if let Some(target_ep) = filter_episode {
            if let Some(ep_col) = batch.column_by_name("episode_index") {
                let ep_arr = ep_col
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .ok_or_else(|| "episode_index is not Int64Array".to_string())?;
                Some((0..ep_arr.len()).map(|i| ep_arr.value(i) as usize == target_ep).collect())
            } else {
                None
            }
        } else {
            None
        };

        let state_col = batch
            .column_by_name("observation.state")
            .ok_or_else(|| "No 'observation.state' column in parquet".to_string())?;

        let list_arr = state_col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .ok_or_else(|| "observation.state is not FixedSizeListArray".to_string())?;

        let values = list_arr
            .values()
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .ok_or_else(|| "observation.state values are not Float32".to_string())?;

        let list_size = list_arr.value_length() as usize;
        for i in 0..list_arr.len() {
            if let Some(ref mask) = row_mask {
                if !mask[i] {
                    continue;
                }
            }
            let offset = i * list_size;
            let row: Vec<f32> = (0..list_size).map(|j| values.value(offset + j)).collect();
            all_states.push(row);
        }
    }

    Ok(all_states)
}

/// Build the parquet data path for an episode.
/// v2.1: `data/chunk-NNN/episode_NNNNNN.parquet` (one file per episode)
/// v3.0: `data/chunk-NNN/file-NNN.parquet` (shared files, need to filter by episode_index)
pub(crate) fn episode_data_path(dataset_root: &Path, episode_index: usize, chunks_size: usize, codebase_version: &str) -> PathBuf {
    let chunk = episode_index / chunks_size;
    if codebase_version.starts_with("v3") {
        // v3.0: episodes are packed into shared files. Find the right file.
        // Each file holds `chunks_size` episodes, file index = episode_index / chunks_size within chunk.
        // For simplicity, scan for files in the chunk dir.
        let chunk_dir = dataset_root
            .join("data")
            .join(format!("chunk-{:03}", chunk));
        // v3.0 uses file-NNN.parquet; episode_index within chunk determines file
        let file_idx = episode_index % chunks_size;
        // Actually in v3.0, all episodes in a chunk may be in a single file or split.
        // Try file-000 first (most common for small datasets).
        let path = chunk_dir.join(format!("file-{:03}.parquet", 0));
        if path.is_file() {
            return path;
        }
        // Fallback: try matching file index
        chunk_dir.join(format!("file-{:03}.parquet", file_idx))
    } else {
        dataset_root
            .join("data")
            .join(format!("chunk-{:03}", chunk))
            .join(format!("episode_{:06}.parquet", episode_index))
    }
}

/// LRU cache of computed EE trajectories, keyed by episode index.
pub(crate) struct TrajectoryCache {
    entries: HashMap<usize, EeTrajectory>,
    order: VecDeque<usize>,
    capacity: usize,
}

impl TrajectoryCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn get(&mut self, episode_index: usize) -> Option<&EeTrajectory> {
        if self.entries.contains_key(&episode_index) {
            self.order.retain(|&i| i != episode_index);
            self.order.push_back(episode_index);
            self.entries.get(&episode_index)
        } else {
            None
        }
    }

    pub fn insert(&mut self, episode_index: usize, trajectory: EeTrajectory) {
        if self.entries.contains_key(&episode_index) {
            self.order.retain(|&i| i != episode_index);
        } else if self.entries.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(episode_index, trajectory);
        self.order.push_back(episode_index);
    }
}

/// Find the deepest leaf link in a kinematic chain (most ancestors).
/// Used to auto-detect the end-effector frame.
fn find_deepest_leaf(chain: &k::Chain<f64>) -> Result<k::node::Node<f64>, String> {
    let mut best: Option<(k::node::Node<f64>, usize)> = None;
    for node in chain.iter() {
        if node.children().is_empty() {
            // Leaf node — count depth by walking parents
            let mut depth = 0;
            let mut cur = node.clone();
            while let Some(parent) = cur.parent() {
                depth += 1;
                cur = parent;
            }
            if best.as_ref().map_or(true, |&(_, d)| depth > d) {
                best = Some((node.clone(), depth));
            }
        }
    }
    best.map(|(n, _)| n)
        .ok_or_else(|| "No leaf links found in URDF".to_string())
}

/// Extract the indices of `.pos` values from state feature names.
/// e.g. ["joint_1.pos", "joint_1.vel", "joint_1.torque", "joint_2.pos", ...]
/// Returns indices of names ending in ".pos", excluding "gripper".
pub(crate) fn pos_indices_from_state_names(state_names: &[String]) -> Vec<usize> {
    state_names
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            name.ends_with(".pos") && !name.starts_with("gripper")
        })
        .map(|(i, _)| i)
        .collect()
}

/// Discover the URDF file for a dataset.
/// Search order:
/// 1. `<dataset_root>/robot.urdf`
/// 2. `~/.config/lerobot-explorer/robots/<robot_type>.urdf`
/// 3. Bundled paths for known robots
pub(crate) fn discover_urdf(dataset_root: &Path, robot_type: Option<&str>) -> Option<PathBuf> {
    // 1. Dataset-local
    let local = dataset_root.join("robot.urdf");
    if local.is_file() {
        return Some(local);
    }

    // 2. User config directory
    if let Some(rt) = robot_type {
        if let Some(config_dir) = dirs::config_dir() {
            let user_urdf = config_dir
                .join("lerobot-explorer")
                .join("robots")
                .join(format!("{}.urdf", rt));
            if user_urdf.is_file() {
                return Some(user_urdf);
            }
        }
    }

    None
}
