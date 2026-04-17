# tracelr

<img src="https://github.com/user-attachments/assets/1abfe612-9c27-4916-9974-df93640d5f7c" alt="Alt text" width="600"/>

A fast desktop tool for exploring and tracing [LeRobot](https://github.com/huggingface/lerobot) datasets (pronounced "tracer"). Built with Rust, egui, and ffmpeg for real-time video playback of robot demonstration episodes.

Browse episodes, play back videos, and inspect metadata. Optionally enable annotation mode (`--annotate`) to assign text prompts for VLA model training.

## Features

- **Video playback** - auto-plays episodes at native framerate with Play/Pause (Space) and scrubbing (slider drag)
- **Episode navigation** - arrow keys, skate mode (Shift+Arrow for continuous advance), click the episode list, or drag the slider
- **Episode cache** - sliding window cache preloads neighboring episodes for instant navigation
- **Drag and drop** - drop a dataset folder onto the window to open it
- **EE trajectory visualization** - 3D end-effector trajectory plots computed via forward kinematics from URDF files, with orbit camera, ground grid, and live playhead tracking
- **Grid view** - play multiple episodes simultaneously in a tiled grid (press G), with multi-trajectory overlay to compare episodes at a glance

### Annotation mode (`--annotate`)

- **Annotation** - assign text prompts to episodes via keyboard shortcuts (1-9) or clickable prompt cards, with color-coded status in the episode list
- **Configurable prompts** - define prompts per dataset via `prompts.yaml` (see below)
- **Persistence** - annotations save to `annotations.json` in the dataset directory, auto-loaded on reopen
- **Export** - export annotations to LeRobot's `tasks.jsonl` + `episodes.jsonl` format

## Supported LeRobot formats

| Format | Version | Video layout | Episode metadata | Status |
|--------|---------|-------------|-----------------|--------|
| LeRobot v2.1 | `codebase_version: "v2.1"` | One mp4 per episode (`episode_000000.mp4`) | `meta/episodes.jsonl` | Supported |
| LeRobot v3.0 | `codebase_version: "v3.0"` | Concatenated mp4 with timestamp ranges (`file-000.mp4`) | `meta/episodes/chunk-NNN/file-NNN.parquet` | Supported |

Both formats are auto-detected from `meta/info.json`.

## Install

### Prerequisites

- Rust toolchain ([rustup](https://rustup.rs/))
- FFmpeg development libraries and pkg-config:

  **macOS (Homebrew)**
  ```
  brew install pkgconf ffmpeg
  ```

  **Ubuntu/Debian**
  ```
  sudo apt install pkg-config libavcodec-dev libavformat-dev libswscale-dev libavutil-dev
  ```

  **Fedora/RHEL**
  ```
  sudo dnf install pkgconf-pkg-config ffmpeg-free-devel
  ```

  **Windows**

  1. **LLVM/Clang** - required by `bindgen` to generate FFmpeg bindings. Install and set the environment variable in PowerShell:
     ```powershell
     winget install LLVM.LLVM
     [System.Environment]::SetEnvironmentVariable("LIBCLANG_PATH", "C:\Program Files\LLVM\bin", "User")
     ```

  2. **FFmpeg** - download the **shared** build from [ffmpeg.org/download](https://ffmpeg.org/download.html#build-windows) (links to gyan.dev), extract it (e.g. to `C:\ffmpeg`), then set environment variables in PowerShell:
     ```powershell
     [System.Environment]::SetEnvironmentVariable("FFMPEG_DIR", "C:\ffmpeg", "User")
     # Add DLLs to PATH for runtime
     $p = [System.Environment]::GetEnvironmentVariable("PATH", "User")
     [System.Environment]::SetEnvironmentVariable("PATH", "$p;C:\ffmpeg\bin", "User")
     ```

  Restart your terminal after setting environment variables.

### Build

```bash
cargo build --profile opt-dev
```

The `opt-dev` profile gives release-level optimization with faster incremental builds (no LTO).

### Linux icon setup

On GNOME 46+ (Ubuntu 24.04+), the taskbar icon requires installing a `.desktop` file and icon to XDG locations. Install the PNG to the hicolor icon theme and the desktop file to `~/.local/share/applications/`, pointing `Exec=` at the absolute path of your binary:

```bash
# Install icons at every size GNOME might request
for size in 16 32 48 64 128 256 512; do
    mkdir -p ~/.local/share/icons/hicolor/${size}x${size}/apps
    cp assets/icon_${size}.png \
        ~/.local/share/icons/hicolor/${size}x${size}/apps/tracelr.png
done
gtk-update-icon-cache -f ~/.local/share/icons/hicolor/

# Install the desktop file, substituting Exec= with your binary path
# (e.g. $PWD/target/opt-dev/tracelr for a local cargo build, or
#  /path/to/tracelr.AppImage for an AppImage build)
TRACELR_BIN="$PWD/target/opt-dev/tracelr"
sed "s|Exec=.*|Exec=${TRACELR_BIN} %f|" resources/tracelr.desktop \
    > ~/.local/share/applications/tracelr.desktop
```

If `Exec=` points to a command that cannot be found (a bare name that is not in `$PATH`, or an absolute path that does not exist), GNOME silently ignores the entire `.desktop` file and no icon will appear — even though the file looks valid. Make sure the value you substitute above is either an absolute path to the binary or a name that resolves via `$PATH`.

## Usage

```bash
# Open a dataset (viewer mode - browse and play episodes)
cargo run --profile opt-dev -- /path/to/lerobot/dataset/

# Enable annotation mode (prompt assignment, save/export)
cargo run --profile opt-dev -- --annotate /path/to/lerobot/dataset/

# Specify a custom URDF for trajectory visualization
cargo run --profile opt-dev -- --urdf /path/to/robot.urdf /path/to/dataset/

# Or launch and drag-drop a dataset folder onto the window
cargo run --profile opt-dev

# With debug logging
RUST_LOG=tracelr=debug cargo run --profile opt-dev -- /path/to/dataset/
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Left` / `Right` | Previous / next episode (or page grid) |
| `Shift+Left/Right` | Skate (continuous advance while held) |
| `Home` / `End` | First / last episode |
| `Space` | Play / pause video |
| `Enter` | Re-enter video mode |
| `Escape` | Reset to the initial single view (default camera, all checkboxes re-checked, current episode preserved) |
| `G` | Toggle the episode-grid dimension (single video ↔ multi-episode grid) |
| `M` | Toggle the camera dimension. From single video: enter multi-camera grid (one episode, all cameras). From multi-episode grid: enter Subgrid (cameras nested inside each episode cell). From multi-camera grid: exit to single video. |
| `N` | Toggle the Tiled camera layout (each camera is a flat pane, cols snap to a multiple of camera count). Works from single video, episode grid, and multi-camera grid. |
| `C` / `Shift+C` | Cycle the active single camera (single video mode and single-camera grid) |
| `T` | Toggle trajectory panel |
| `+` / `-` | Resize grid (in grid mode) |
| `1`-`9` | Assign prompt to current episode (annotation mode) |
| `Ctrl+S` | Save annotations (annotation mode) |

`G`, `M`, and `N` are composable: `M` adds or removes the camera dimension,
`G` adds or removes the episode dimension, and `N` toggles the Tiled layout
on the current view. For example, pressing `M` then `G` from single video
lands you in Subgrid (multi-episode, multi-camera). `Escape` always returns
to the initial single-video view if you are lost in a state combination.

### Multi-camera view

Datasets with multiple cameras (wrist, top, side, etc.) can be viewed in
several layouts:

- **Single camera, single episode** (default): one video player
- **Single camera, multi-episode grid** (`G`): the episode grid, one camera
  across all panes. Use `C` to cycle cameras, or the camera dropdown in the
  right-side Cameras panel
- **Multi-camera single-episode** (`M` from single video): all cameras for
  one episode in an auto-sized grid
- **Subgrid** (`M` from grid, or `G` from multi-camera): each episode cell
  shows all cameras as mini-frames inside it, preserving your grid layout
- **Tiled** (`N` from grid, or `N` from single video): each camera gets its
  own flat pane. Cols snap to a multiple of the camera count. Use the
  **Tiled View Rows** slider in the View menu to adjust rows.

In multi-camera views, the right-side Cameras panel shows checkboxes to
enable or disable individual cameras.

### EE trajectory visualization

The app computes end-effector positions via forward kinematics from URDF files and renders 3D trajectory plots alongside video playback.

**URDF discovery order:**

1. `--urdf /path/to/robot.urdf` (CLI flag, highest priority)
2. `<dataset_dir>/robot.urdf` (dataset-local)
3. `~/.config/tracelr/robots/<robot_type>.urdf` (user config, Linux)
4. `~/Library/Application Support/tracelr/robots/<robot_type>.urdf` (macOS)

The `<robot_type>` comes from the `robot_type` field in the dataset's `meta/info.json` (e.g. `"so101_follower"`, `"openarm_follower"`).

**Setting up a URDF:**

Place the robot's URDF file in the config directory with a filename matching the `robot_type`:

```bash
# Linux
mkdir -p ~/.config/tracelr/robots/
cp /path/to/so101.urdf ~/.config/tracelr/robots/so101_follower.urdf

# macOS
mkdir -p ~/Library/Application\ Support/tracelr/robots/
cp /path/to/so101.urdf ~/Library/Application\ Support/tracelr/robots/so101_follower.urdf
```

Joint names in the URDF must match the `.pos` column base names in the dataset's `observation.state` features. For example, if the dataset has `shoulder_pan.pos`, the URDF joint should be named `shoulder_pan`. The app auto-detects the end-effector frame (deepest leaf link in the kinematic chain) and extracts only `.pos` indices from `observation.state`, so interleaved pos/vel/torque formats (like OpenArm) work automatically.

**Tested robots:**

- SO101 follower (5 DOF)
- OpenArm v10 bimanual left/right (7 DOF)

Any robot with a URDF and `observation.state` containing `.pos` columns should work. Note that trajectory visualization currently expects joint values in **degrees** (i.e. datasets recorded with `use_degrees=True` in lerobot). Datasets using the default `RANGE_M100_100` normalization (values in [-100, 100]) are not yet supported and will produce incorrect trajectories.

### Configurable prompts (annotation mode)

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

See [`configs/prompts.example.yaml`](configs/prompts.example.yaml) for a full example.

Search order: dataset directory > user config > built-in defaults.

### Annotation output (annotation mode)


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

Use File > Export to LeRobot to write `meta/tasks.jsonl` and update `meta/episodes.jsonl` with task assignments.

## License

tracelr is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
