# Usage Guide

## Multi-camera views

Datasets with multiple cameras (wrist, top, side, etc.) support several layouts:

- **Single camera, single episode** (default): one video player
- **Single camera, multi-episode grid** (`G`): episode grid, one camera across all panes. Use `C` to cycle cameras or the dropdown in the Cameras panel
- **Multi-camera single-episode** (`M` from single video): all cameras for one episode in an auto-sized grid
- **Subgrid** (`M` from grid, or `G` from multi-camera): each episode cell shows all cameras as mini-frames
- **Tiled** (`N`): each camera gets its own flat pane. Cols snap to a multiple of camera count. Adjust rows via the **Tiled View Rows** slider in the View menu

The Cameras panel on the right side has checkboxes to enable or disable individual cameras.

## EE trajectory visualization

3D end-effector trajectory plots computed via forward kinematics from URDF files, rendered alongside video playback with orbit camera, ground grid, and live playhead tracking.

### Setting up a URDF

The easiest way is to drag and drop a `.urdf` file onto the app window. The file is copied to the robots config directory and kinematics load immediately. You can also use the "Browse..." button in the trajectory panel, or place files manually:

```bash
# Linux
mkdir -p ~/.config/tracelr/robots/
cp /path/to/so101.urdf ~/.config/tracelr/robots/so101_follower.urdf

# macOS
mkdir -p ~/Library/Application\ Support/tracelr/robots/
cp /path/to/so101.urdf ~/Library/Application\ Support/tracelr/robots/so101_follower.urdf
```

When no URDF is found, the trajectory panel shows the expected filename and an "Open robots folder" button.

### URDF discovery order

1. `--urdf /path/to/robot.urdf` (CLI flag, highest priority)
2. `<dataset_dir>/robot.urdf` (dataset-local)
3. `<robot_type>.toml` in the config dir (explicit multi-arm config)
4. `<robot_type>*.urdf` glob in the config dir (auto-detection)

The `<robot_type>` comes from `robot_type` in the dataset's `meta/info.json`.

### Multi-arm robots

Place separate URDFs with the robot type as prefix:

```
~/.config/tracelr/robots/
  openarm_follower_left.urdf
  openarm_follower_right.urdf
```

The app auto-detects multiple URDFs and derives arm names from the suffix ("left", "right"). A dropdown appears in the trajectory panel to switch arms. Your selection is saved per dataset to `~/.config/tracelr/arm_preferences.json`.

For explicit control (custom arm names, joint prefix filtering, EE frame overrides), create a `<robot_type>.toml`:

```toml
[[arm]]
name = "Left Arm"
urdf = "openarm_follower_left.urdf"
# joint_prefix = "openarm_left_"   # for bimanual datasets with prefixed state columns
# ee_frame = "custom_frame"        # override auto-detected EE frame

[[arm]]
name = "Right Arm"
urdf = "openarm_follower_right.urdf"
```

### Joint name matching

Joint names in the URDF must match the `.pos` column base names in the dataset's `observation.state` features. For example, if the dataset has `shoulder_pan.pos`, the URDF joint should be named `shoulder_pan`. The app auto-detects the end-effector frame (deepest leaf link in the kinematic chain) and extracts only `.pos` indices, so interleaved pos/vel/torque formats (like OpenArm) work automatically.

### Tested robots

- SO101 follower (5 DOF)
- OpenArm v10 bimanual left/right (7 DOF)

Any robot with a URDF and `observation.state` containing `.pos` columns should work. Trajectory visualization currently expects joint values in **degrees** (datasets recorded with `use_degrees=True`). Datasets using `RANGE_M100_100` normalization are not yet supported.

## Annotation mode

Enable with `--annotate` to assign text prompts to episodes for VLA model training.

- Assign prompts via keyboard shortcuts (`1`-`9`) or clickable prompt cards
- Color-coded status in the episode list
- Annotations save to `annotations.json` in the dataset directory, auto-loaded on reopen
- Export to LeRobot format via File > Export to LeRobot (`meta/tasks.jsonl` + `meta/episodes.jsonl`)

### Configurable prompts

Create a `prompts.yaml` in the dataset directory or `~/.config/tracelr/prompts.yaml`:

```yaml
prompts:
  - label: "Red cube"
    prompt: "Pick up the red cube and place it in the bowl"
    color: [220, 60, 60]

  - label: "Blue cube"
    prompt: "Pick up the blue cube and place it in the bowl"
    color: [60, 100, 220]
```

See [`configs/prompts.example.yaml`](../configs/prompts.example.yaml) for a full example. Search order: dataset directory > user config > built-in defaults.

### Annotation output

Annotations save to `<dataset_dir>/annotations.json`:

```json
{
  "dataset_root": "/path/to/dataset",
  "prompts": [
    "Pick up the red cube and place it in the bowl",
    "Pick up the blue cube and place it in the bowl"
  ],
  "annotations": {
    "0": 0,
    "1": 1,
    "2": 0
  }
}
```

## Linux desktop integration

On GNOME 46+ (Ubuntu 24.04+), the taskbar icon requires installing a `.desktop` file and icon to XDG locations:

```bash
for size in 16 32 48 64 128 256 512; do
    mkdir -p ~/.local/share/icons/hicolor/${size}x${size}/apps
    cp assets/icon_${size}.png \
        ~/.local/share/icons/hicolor/${size}x${size}/apps/tracelr.png
done
gtk-update-icon-cache -f ~/.local/share/icons/hicolor/

TRACELR_BIN="$PWD/target/opt-dev/tracelr"
sed "s|Exec=.*|Exec=${TRACELR_BIN} %f|" resources/tracelr.desktop \
    > ~/.local/share/applications/tracelr.desktop
```

Make sure `Exec=` points to an absolute path to the binary or a name that resolves via `$PATH`. GNOME silently ignores `.desktop` files with an invalid `Exec=` path.

## Windows FFmpeg setup

1. **LLVM/Clang** (required by `bindgen`):
   ```powershell
   winget install LLVM.LLVM
   [System.Environment]::SetEnvironmentVariable("LIBCLANG_PATH", "C:\Program Files\LLVM\bin", "User")
   ```

2. **FFmpeg** shared build from [ffmpeg.org/download](https://ffmpeg.org/download.html#build-windows) (links to gyan.dev):
   ```powershell
   [System.Environment]::SetEnvironmentVariable("FFMPEG_DIR", "C:\ffmpeg", "User")
   $p = [System.Environment]::GetEnvironmentVariable("PATH", "User")
   [System.Environment]::SetEnvironmentVariable("PATH", "$p;C:\ffmpeg\bin", "User")
   ```

Restart your terminal after setting environment variables.
