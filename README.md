# lerobot-explorer

A fast desktop tool for exploring [LeRobot](https://github.com/huggingface/lerobot) datasets. Built with Rust, egui, and ffmpeg for real-time video playback of robot demonstration episodes.

Browse episodes, play back videos, and inspect metadata. Optionally enable annotation mode (`--annotate`) to assign text prompts for VLA model training.

## Features

- **Video playback** — auto-plays episodes at native framerate with Play/Pause (Space) and scrubbing (slider drag)
- **Episode navigation** — arrow keys, skate mode (Shift+Arrow for continuous advance), click the episode list, or drag the slider
- **Episode cache** — sliding window cache preloads neighboring episodes for instant navigation
- **Drag and drop** — drop a dataset folder onto the window to open it

### Annotation mode (`--annotate`)

- **Annotation** — assign text prompts to episodes via keyboard shortcuts (1-9) or clickable prompt cards, with color-coded status in the episode list
- **Configurable prompts** — define prompts per dataset via `prompts.yaml` (see below)
- **Persistence** — annotations save to `annotations.json` in the dataset directory, auto-loaded on reopen
- **Export** — export annotations to LeRobot's `tasks.jsonl` + `episodes.jsonl` format

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

  Download an FFmpeg "shared" build from [ffmpeg.org/download](https://ffmpeg.org/download.html#build-windows), extract it, and set the `FFMPEG_DIR` environment variable to the extracted folder:
  ```
  set FFMPEG_DIR=C:\path\to\ffmpeg
  cargo build --profile opt-dev
  ```
  Ensure the FFmpeg `bin` directory is on your `PATH` at runtime so the DLLs are found.

### Build

```bash
cargo build --profile opt-dev
```

The `opt-dev` profile gives release-level optimization with faster incremental builds (no LTO).

## Usage

```bash
# Open a dataset (viewer mode — browse and play episodes)
cargo run --profile opt-dev -- /path/to/lerobot/dataset/

# Enable annotation mode (prompt assignment, save/export)
cargo run --profile opt-dev -- --annotate /path/to/lerobot/dataset/

# Or launch and drag-drop a dataset folder onto the window
cargo run --profile opt-dev

# With debug logging
RUST_LOG=lerobot_explorer=debug cargo run --profile opt-dev -- /path/to/dataset/
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Left` / `Right` | Previous / next episode |
| `Shift+Left/Right` | Skate (continuous advance while held) |
| `Home` / `End` | First / last episode |
| `Space` | Play / pause video |
| `Escape` | Exit video mode (show thumbnail) |
| `Enter` | Re-enter video mode |
| `1`-`9` | Assign prompt to current episode (annotation mode) |
| `Ctrl+S` | Save annotations (annotation mode) |

### Configurable prompts (annotation mode)

Create a `prompts.yaml` in the dataset directory or `~/.config/lerobot-explorer/prompts.yaml`:

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
