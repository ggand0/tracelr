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
    /// Load a URDF and build the kinematic chain from base to `ee_frame`.
    pub fn from_urdf(urdf_path: &Path, ee_frame: &str) -> Result<Self, String> {
        let chain = k::Chain::<f64>::from_urdf_file(urdf_path)
            .map_err(|e| format!("Failed to load URDF {}: {}", urdf_path.display(), e))?;

        let ee_link = chain
            .find_link(ee_frame)
            .ok_or_else(|| format!("EE frame '{}' not found in URDF", ee_frame))?;

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

        log::info!(
            "Loaded URDF: {} -> {} (DOF={}, joints={:?})",
            urdf_path.display(),
            ee_frame,
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
    /// `states` is per-frame, each row is [joint0_deg, joint1_deg, ..., gripper_deg].
    /// Only the first `dof` values are used for FK (gripper is ignored).
    pub fn compute_trajectory(&self, states: &[Vec<f32>]) -> EeTrajectory {
        let positions: Vec<[f64; 3]> = states
            .iter()
            .map(|state| {
                let angles: Vec<f64> = state.iter().map(|v| *v as f64).collect();
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

/// Load `observation.state` from a v2.1 data parquet file.
/// Returns per-frame joint states as Vec<Vec<f32>>.
pub(crate) fn load_episode_states(parquet_path: &Path) -> Result<Vec<Vec<f32>>, String> {
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
            let offset = i * list_size;
            let row: Vec<f32> = (0..list_size).map(|j| values.value(offset + j)).collect();
            all_states.push(row);
        }
    }

    Ok(all_states)
}

/// Build the parquet data path for a v2.1 episode.
pub(crate) fn episode_data_path(dataset_root: &Path, episode_index: usize, chunks_size: usize) -> PathBuf {
    let chunk = episode_index / chunks_size;
    dataset_root
        .join("data")
        .join(format!("chunk-{:03}", chunk))
        .join(format!("episode_{:06}.parquet", episode_index))
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
