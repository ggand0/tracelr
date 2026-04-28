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

        log::info!(
            "Loaded URDF: {} -> {} (DOF={}, joints={:?})",
            urdf_path.display(),
            ee_link.joint().name,
            serial.dof(),
            joint_names,
        );

        Ok(Self { serial })
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

    let reader = builder.build()
        .map_err(|e| format!("Build reader {}: {}", parquet_path.display(), e))?;

    let mut all_states: Vec<Vec<f32>> = Vec::new();

    for batch in reader {
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

        if let Some(fixed_arr) = state_col.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
            let values = fixed_arr
                .values()
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or_else(|| "observation.state values are not Float32".to_string())?;
            let list_size = fixed_arr.value_length() as usize;
            for i in 0..fixed_arr.len() {
                if let Some(ref mask) = row_mask {
                    if !mask[i] { continue; }
                }
                let offset = i * list_size;
                let row: Vec<f32> = (0..list_size).map(|j| values.value(offset + j)).collect();
                all_states.push(row);
            }
        } else if let Some(var_arr) = state_col.as_any().downcast_ref::<arrow::array::ListArray>() {
            let values = var_arr
                .values()
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or_else(|| "observation.state values are not Float32".to_string())?;
            for i in 0..var_arr.len() {
                if let Some(ref mask) = row_mask {
                    if !mask[i] { continue; }
                }
                let start = var_arr.value_offsets()[i] as usize;
                let end = var_arr.value_offsets()[i + 1] as usize;
                let row: Vec<f32> = (start..end).map(|j| values.value(j)).collect();
                all_states.push(row);
            }
        } else {
            return Err(format!(
                "observation.state is neither FixedSizeListArray nor ListArray (type: {:?})",
                state_col.data_type(),
            ));
        }
    }

    Ok(all_states)
}

/// Build the parquet data path for an episode.
/// v2.1: `data/chunk-NNN/episode_NNNNNN.parquet` (one file per episode)
/// v3.0: `data/chunk-NNN/file-NNN.parquet` using chunk/file indices from episode metadata.
pub(crate) fn episode_data_path(
    dataset_root: &Path,
    episode_index: usize,
    chunks_size: usize,
    codebase_version: &str,
    data_chunk_index: usize,
    data_file_index: usize,
) -> PathBuf {
    if codebase_version.starts_with("v3") {
        dataset_root
            .join("data")
            .join(format!("chunk-{:03}", data_chunk_index))
            .join(format!("file-{:03}.parquet", data_file_index))
    } else {
        let chunk = episode_index / chunks_size;
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
            if best.as_ref().is_none_or(|&(_, d)| depth > d) {
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
                .join("tracelr")
                .join("robots")
                .join(format!("{}.urdf", rt));
            if user_urdf.is_file() {
                return Some(user_urdf);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
    }

    fn load_so101_kin() -> RobotKinematics {
        let urdf = fixtures_dir().join("so101_follower.urdf");
        RobotKinematics::from_urdf(&urdf, None).expect("failed to load SO101 URDF")
    }

    #[derive(serde::Deserialize)]
    struct PoseFile {
        poses: Vec<Pose>,
    }

    #[derive(serde::Deserialize)]
    struct Pose {
        name: String,
        #[allow(dead_code)]
        joints_deg: Vec<f64>,
        actual_joints_deg: Option<Vec<f64>>,
        fk_actual_ee_m: Option<Vec<f64>>,
        measured_ee_m: Option<Vec<f64>>,
        #[allow(dead_code)]
        measurement_accuracy_cm: Option<f64>,
    }

    fn load_ground_truth_poses() -> Vec<Pose> {
        let path = fixtures_dir().join("so101_ground_truth_poses.json");
        let data = std::fs::read_to_string(&path).expect("failed to read poses JSON");
        let file: PoseFile = serde_json::from_str(&data).expect("failed to parse poses JSON");
        file.poses
    }

    // --- Layer 1: k crate vs placo (tight tolerance) ---

    #[test]
    fn fk_matches_placo_all_poses() {
        let kin = load_so101_kin();
        let poses = load_ground_truth_poses();
        let tolerance_m = 0.0005; // 0.5mm

        for pose in &poses {
            let actual_joints = pose.actual_joints_deg.as_ref()
                .unwrap_or_else(|| panic!("pose '{}' missing actual_joints_deg", pose.name));
            let placo_ee = pose.fk_actual_ee_m.as_ref()
                .unwrap_or_else(|| panic!("pose '{}' missing fk_actual_ee_m", pose.name));

            let k_ee = kin.forward_kinematics_deg(actual_joints);

            for (axis, (&k_val, &placo_val)) in ["x", "y", "z"].iter()
                .zip(k_ee.iter().zip(placo_ee.iter()))
            {
                let diff = (k_val - placo_val).abs();
                assert!(
                    diff < tolerance_m,
                    "pose '{}' axis {}: k={:.4} placo={:.4} diff={:.4}m (tolerance={:.4}m)",
                    pose.name, axis, k_val, placo_val, diff, tolerance_m,
                );
            }
        }
    }

    #[test]
    fn fk_matches_placo_reset() {
        let kin = load_so101_kin();
        let poses = load_ground_truth_poses();
        let reset = poses.iter().find(|p| p.name == "reset").expect("no reset pose");

        let actual = reset.actual_joints_deg.as_ref().unwrap();
        let expected = reset.fk_actual_ee_m.as_ref().unwrap();
        let ee = kin.forward_kinematics_deg(actual);

        for i in 0..3 {
            assert!((ee[i] - expected[i]).abs() < 0.0005,
                "reset axis {}: k={:.4} placo={:.4}", i, ee[i], expected[i]);
        }
    }

    #[test]
    fn fk_matches_placo_straight_forward() {
        let kin = load_so101_kin();
        let poses = load_ground_truth_poses();
        let pose = poses.iter().find(|p| p.name == "straight-forward").expect("no straight-forward pose");

        let actual = pose.actual_joints_deg.as_ref().unwrap();
        let expected = pose.fk_actual_ee_m.as_ref().unwrap();
        let ee = kin.forward_kinematics_deg(actual);

        for i in 0..3 {
            assert!((ee[i] - expected[i]).abs() < 0.0005,
                "straight-forward axis {}: k={:.4} placo={:.4}", i, ee[i], expected[i]);
        }
    }

    // --- Layer 2: FK vs physical measurement (loose tolerance) ---

    #[test]
    fn fk_within_physical_measurement_tolerance() {
        let kin = load_so101_kin();
        let poses = load_ground_truth_poses();
        let tolerance_m = 0.03; // 3cm -- tape measure accuracy + base origin offset

        for pose in &poses {
            let actual_joints = match &pose.actual_joints_deg {
                Some(j) => j,
                None => continue,
            };
            let measured = match &pose.measured_ee_m {
                Some(m) => m,
                None => continue,
            };

            let ee = kin.forward_kinematics_deg(actual_joints);

            // X (forward) -- consistently good across all poses
            let x_diff = (ee[0] - measured[0]).abs();
            assert!(
                x_diff < tolerance_m,
                "pose '{}' X: fk={:.3} measured={:.3} diff={:.3}m",
                pose.name, ee[0], measured[0], x_diff,
            );

            // Y (lateral) -- check magnitude, sign can differ by measurement convention
            let y_mag_diff = (ee[1].abs() - measured[1].abs()).abs();
            assert!(
                y_mag_diff < tolerance_m,
                "pose '{}' |Y|: fk={:.3} measured={:.3} diff={:.3}m",
                pose.name, ee[1].abs(), measured[1].abs(), y_mag_diff,
            );
        }
    }

    // --- Tier 2: Data pipeline unit tests ---

    #[test]
    fn pos_indices_so101_contiguous() {
        let names: Vec<String> = vec![
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        ].into_iter().map(String::from).collect();
        assert_eq!(pos_indices_from_state_names(&names), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pos_indices_openarm_interleaved() {
        let names: Vec<String> = vec![
            "joint_1.pos", "joint_1.vel", "joint_1.torque",
            "joint_2.pos", "joint_2.vel", "joint_2.torque",
            "joint_3.pos", "joint_3.vel", "joint_3.torque",
            "joint_4.pos", "joint_4.vel", "joint_4.torque",
            "joint_5.pos", "joint_5.vel", "joint_5.torque",
            "joint_6.pos", "joint_6.vel", "joint_6.torque",
            "joint_7.pos", "joint_7.vel", "joint_7.torque",
            "gripper.pos",
        ].into_iter().map(String::from).collect();
        assert_eq!(pos_indices_from_state_names(&names), vec![0, 3, 6, 9, 12, 15, 18]);
    }

    #[test]
    fn pos_indices_empty_input() {
        let names: Vec<String> = vec![];
        assert_eq!(pos_indices_from_state_names(&names), Vec::<usize>::new());
    }

    #[test]
    fn pos_indices_no_pos_suffix() {
        let names: Vec<String> = vec!["joint_1.vel", "joint_2.vel"]
            .into_iter().map(String::from).collect();
        assert_eq!(pos_indices_from_state_names(&names), Vec::<usize>::new());
    }

    #[test]
    fn compute_trajectory_uses_pos_indices() {
        let kin = load_so101_kin();
        let states = vec![
            vec![10.0f32, 999.0, 20.0, 999.0, 30.0, 999.0, 0.0, 999.0, 0.0, 999.0],
            vec![0.0f32, 999.0, 0.0, 999.0, 0.0, 999.0, 0.0, 999.0, 0.0, 999.0],
        ];
        let pos_indices = vec![0, 2, 4, 6, 8];

        let traj = kin.compute_trajectory(&states, &pos_indices);
        assert_eq!(traj.positions.len(), 2);

        let ee_direct = kin.forward_kinematics_deg(&[10.0, 20.0, 30.0, 0.0, 0.0]);
        for i in 0..3 {
            assert!((traj.positions[0][i] - ee_direct[i]).abs() < 1e-10,
                "pos_indices extraction mismatch at axis {}", i);
        }

        let ee_zero = kin.forward_kinematics_deg(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        for i in 0..3 {
            assert!((traj.positions[1][i] - ee_zero[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn so101_dof_is_five() {
        let kin = load_so101_kin();
        assert_eq!(kin.dof(), 5);
    }

    #[test]
    fn so101_ee_autodetect() {
        let urdf = fixtures_dir().join("so101_follower.urdf");
        let chain = k::Chain::<f64>::from_urdf_file(&urdf).unwrap();
        let leaf = find_deepest_leaf(&chain).unwrap();
        let name = leaf.joint().name.clone();
        // gripper_frame_joint is a fixed joint at the tip of the wrist chain -- correct EE
        assert_eq!(name, "gripper_frame_joint",
            "SO101 auto-detected EE should be gripper_frame_joint (fixed), got '{}'", name);
    }

    #[test]
    fn episode_data_path_v21() {
        let root = Path::new("/fake/dataset");
        let path = episode_data_path(root, 0, 1000, "v2.1");
        assert_eq!(path, root.join("data/chunk-000/episode_000000.parquet"));

        let path = episode_data_path(root, 74, 1000, "v2.1");
        assert_eq!(path, root.join("data/chunk-000/episode_000074.parquet"));

        let path = episode_data_path(root, 1500, 1000, "v2.1");
        assert_eq!(path, root.join("data/chunk-001/episode_001500.parquet"));
    }

    #[test]
    fn home_position_sanity() {
        let kin = load_so101_kin();
        let ee = kin.forward_kinematics_deg(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        // At home (all zeros), EE should be somewhere in front of the base
        assert!(ee[0] > 0.1, "home X should be positive (forward), got {:.3}", ee[0]);
        assert!(ee[1].abs() < 0.05, "home Y should be near zero (centered), got {:.3}", ee[1]);
    }
}
