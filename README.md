# tracelr

<img src="https://github.com/user-attachments/assets/1abfe612-9c27-4916-9974-df93640d5f7c" alt="Alt text" width="600"/>

A fast desktop tool for exploring and tracing [LeRobot](https://github.com/huggingface/lerobot) datasets (pronounced "tracer"). Built with Rust, egui, and ffmpeg for real-time video playback of robot demonstration episodes.

Browse episodes, play back videos, and inspect metadata. Optionally enable annotation mode (`--annotate`) to assign text prompts for VLA model training.

## Features

- **Video playback** with Play/Pause, scrubbing, and native framerate
- **Episode navigation** via arrow keys, skate mode (Shift+Arrow), episode list, or slider
- **Episode cache** preloads neighboring episodes for instant navigation
- **Drag and drop** dataset folders or `.urdf` files onto the window
- **Multi-camera views** with composable grid (`G`), multi-camera (`M`), and tiled (`N`) layouts
- **EE trajectory visualization** via forward kinematics from URDF files, with 3D orbit camera and live playhead
- **Multi-arm support** for bimanual robots (auto-detected from URDF filenames)
- **Grid view** to play multiple episodes simultaneously with multi-trajectory overlay
- **Annotation mode** (`--annotate`) to assign text prompts to episodes, with configurable prompts, persistence, and LeRobot export

## Supported formats

| Format | Version | Video layout | Episode metadata |
|--------|---------|-------------|-----------------|
| LeRobot v2.1 | `"v2.1"` | One mp4 per episode | `meta/episodes.jsonl` |
| LeRobot v3.0 | `"v3.0"` | Concatenated mp4 with timestamp ranges | `meta/episodes/chunk-NNN/file-NNN.parquet` |

Auto-detected from `meta/info.json`.

## Install

**Prerequisites:** [Rust toolchain](https://rustup.rs/) and FFmpeg dev libraries.

<details>
<summary>Platform-specific FFmpeg install</summary>

**macOS:** `brew install pkgconf ffmpeg`

**Ubuntu/Debian:** `sudo apt install pkg-config libavcodec-dev libavformat-dev libswscale-dev libavutil-dev`

**Fedora/RHEL:** `sudo dnf install pkgconf-pkg-config ffmpeg-free-devel`

**Windows:** Install [LLVM](https://llvm.org/) (set `LIBCLANG_PATH`) and FFmpeg shared build (set `FFMPEG_DIR`, add `bin/` to `PATH`). See [docs/usage.md](docs/usage.md#windows-ffmpeg-setup) for full steps.

</details>

```bash
cargo build --profile opt-dev
```

## Usage

```bash
cargo run --profile opt-dev -- /path/to/dataset/          # browse episodes
cargo run --profile opt-dev -- --annotate /path/to/dataset/ # annotation mode
cargo run --profile opt-dev -- --urdf robot.urdf /path/to/dataset/
cargo run --profile opt-dev                                # launch, then drag-drop
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Left` / `Right` | Previous / next episode (or page grid) |
| `Shift+Left/Right` | Skate (continuous advance while held) |
| `Home` / `End` | First / last episode |
| `Space` | Play / pause |
| `Escape` | Reset to single view |
| `G` | Toggle episode grid |
| `M` | Toggle multi-camera |
| `N` | Toggle tiled layout |
| `C` / `Shift+C` | Cycle camera |
| `T` | Toggle trajectory panel |
| `+` / `-` | Resize grid |
| `1`-`9` | Assign prompt (annotation mode) |
| `Ctrl+S` | Save annotations |

`G`, `M`, and `N` are composable. `Escape` returns to single view.

## Docs

- **[Usage guide](docs/usage.md)**: multi-camera views, trajectory/URDF setup, annotation mode, Linux desktop integration

## License

Apache-2.0 OR MIT, at your option. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
