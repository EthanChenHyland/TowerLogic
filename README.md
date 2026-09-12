# TowerLogic

TowerLogic is an experimental Clash Royale automation and game-state analysis
project. It combines screenshot-based computer vision, small learned models,
template matching, policy scoring, and Android emulator control behind a
desktop GUI.

It is a research/portfolio project, not a production bot and not an official
Supercell project.

## What it does

The runtime follows this path:

```text
BlueStacks/ADB screenshot
        -> navigation and battle-state checks
        -> hand-card recognition and field-object detection
        -> normalized game-state vector
        -> candidate placements and policy/heuristic scoring
        -> intended card + coordinate
        -> emulator input (or a logged dry-run action)
```

The GUI configures the worker process, displays runtime statistics, and shows
in-memory learning progress. The worker owns the emulator and communicates
status through a multiprocessing queue.

## Technologies

- Python 3.12–3.14
- OpenCV and NumPy for screenshots, color/region checks, templates, and image preprocessing
- PyTorch and torchvision for the hand-card classifier and policy network
- ONNX Runtime for the field/object detector
- ttkbootstrap/Tkinter for the desktop GUI
- ADB and emulator-specific adapters for device control
- `ultralytics` only for optional YOLO training/export experiments

## ML and computer vision

### Hand-card classifier

`towerlogic/models/hand_card_classifier.pt` is a compact PyTorch image
classifier. Its class map is stored in
`towerlogic/models/hand_card_classifier.class_map.json`. Runtime crops are
converted from OpenCV BGR to RGB, resized to the checkpoint's input size, and
normalized with the same `.5/.5` transform used by training.

The repository also contains card templates under
`towerlogic/bot/detection/card_templates/`. Template matching is a deterministic
fallback/diagnostic path; classifier use is controlled by
`PYCLASHBOT_HAND_USE_CLASSIFIER=1`.

### Field detector

`towerlogic/models/field_detector.onnx` is the runtime detector. The matching
PyTorch checkpoint is retained as `towerlogic/models/field_detector.pt` for
experiments. The detector uses a 640×640 RGB float32 input, decodes YOLO-style
`xywh` predictions, applies confidence filtering and NMS, and maps boxes back
to the source screenshot. The current runtime selects ONNX Runtime's CPU
provider for predictable macOS behavior.

The YOLO-format dataset in `Clash royale.v6i.yolov8/` is included for
experimentation and is attributed to its Roboflow export. It is not required
to run the packaged inference path.

### Policy

`models/policy.pt` contains the trained `SimplePolicyNet` state dictionary. The
runtime builds a 26-value normalized game-state vector and appends a 4-value
candidate-action vector (card index, card group, normalized x/y), for a
30-value model input. The network scores candidate actions; rule-based masks
still enforce elixir, arena, lane, and defensive constraints.

The checkpoint is tracked because it is part of the demo artifact. Training is
optional and writes updates to the configured checkpoint path, so keep training
disabled when you want a reproducible clean checkout.

## Safety: dry-run mode

Set `TOWERLOGIC_DRY_RUN=1` to run the complete screenshot, recognition, policy,
and navigation pipeline while suppressing emulator gameplay input:

```bash
TOWERLOGIC_DRY_RUN=1 python -m towerlogic --start
```

In dry-run mode, the shared ADB controller and the MEmu controller log every
tap/click/swipe instead of issuing `shell input tap` or `shell input swipe`.
Screenshots, ADB queries, app launch commands, and display configuration are
not gameplay input and are not suppressed. Do not treat dry-run as an
anti-cheat or game-policy guarantee; it is a local safety switch for this
project's input methods.

## Setup

### Desktop downloads

Download the Windows x64 or macOS Apple Silicon app from
[Releases](https://github.com/EthanChenHyland/TowerLogic/releases).
Extract the Windows ZIP and launch `TowerLogic.exe` inside its folder. On macOS
15 or newer, extract the ZIP and move `TowerLogic.app` to Applications. Intel Macs
are not supported by the packaged build. Python and the ML runtimes are included;
an emulator/game installation is still required. These builds are not
developer-signed or Apple-notarized; see the release notes for first-launch steps.

To reproduce a native build, use Python 3.12 on the target OS:

```bash
python -m pip install -r packaging/requirements.txt .
python -m unittest discover -s tests -v
python -m PyInstaller --clean --noconfirm packaging/TowerLogic.spec
python packaging/verify_build.py
```

Tagging `v<version>` runs both native builds and publishes only after their tests
pass. Versions in `pyproject.toml`, `towerlogic/__version__`, and the tag must agree.

### Run from source

The project uses `pyproject.toml` and does not ship a lockfile. Recreate the
environment with a supported Python version:

```bash
brew install python@3.12 python-tk@3.12 android-platform-tools
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

Install the optional YOLO tooling only when needed:

```bash
python -m pip install -e ".[yolo]"
```

The core GUI imports without the ML extra, but hand classification, policy
inference/training, and ONNX detection require the `[ml]` extra.

## BlueStacks Air on macOS

The validated macOS path uses BlueStacks Air with the `Tiramisu64` instance,
the bundled `hd-adb`, and a private ADB server (port `5041`). The controller
reads the instance's ADB port from `bluestacks.conf`; it does not require the
legacy `MimMetaData.json` file. The usual device serial is
`127.0.0.1:<instance-port>`.

TowerLogic's visual coordinate space is **419×633**. The BlueStacks controller
configures that framebuffer and uses its own default density of **320**; the
old project documentation's blanket “density 160” statement was not correct
for the validated BlueStacks Air path. Keep the emulator/controller settings
consistent and verify the startup log before a run.

Renderer names are BlueStacks-version and machine dependent. On macOS the GUI
offers OpenGL and Vulkan; the controller's fallback is Vulkan. Use the setting
that produces a rendered frame on the installed BlueStacks version and keep it
unchanged during a dry run. A black surface is a renderer/startup problem, not
an ML result.

The controller resolves the Clash Royale launcher activity using Android's
package manager and starts it with a canonical `MAIN`/`LAUNCHER` intent. It
waits for a foreground, rendered state and recognizes both current and legacy
main-menu signatures. Modern reward/trophy-box screens are detected and use
the existing bounded recovery path.

No emulator, APK, Android game installation, ADB device, or BlueStacks license
is bundled with this repository. Clash Royale must already be installed in the
emulator, and its use must comply with applicable third-party terms.

## Running

Launch the GUI without automatically starting a job:

```bash
source .venv/bin/activate
python -m towerlogic
```

Launch the GUI and start the selected jobs in safe mode:

```bash
source .venv/bin/activate
TOWERLOGIC_DRY_RUN=1 python -m towerlogic --start
```

The convenience script uses the repository's `.venv`:

```bash
bash run_bot.sh
```

For a first real-input run, remove `TOWERLOGIC_DRY_RUN` only after manually
checking the emulator, screen dimensions, renderer, selected jobs, and target
coordinates. Real gameplay is intentionally not the default.

## Training the hand classifier

The training utility expects an ImageFolder-style directory with `train/` and
`valid/` class folders:

```bash
python scripts/train_hand_classifier.py \
  --data-dir /path/to/card_dataset \
  --epochs 12 \
  --batch-size 64
```

The script selects Apple MPS when available, then CUDA or CPU, validates after
each epoch, and writes the best checkpoint and class map to the requested
output locations. Training data is not required for runtime inference because
the current classifier checkpoint and class map are included.

## Useful runtime overrides

The code intentionally keeps local machine paths out of the package. Relevant
environment variables include:

| Variable | Purpose |
| --- | --- |
| `TOWERLOGIC_DRY_RUN` | Suppress tap/swipe input when set to `1`, `true`, `yes`, or `on` |
| `PYCLASHBOT_HAND_USE_CLASSIFIER` | Enable classifier-backed hand recognition |
| `PYCLASHBOT_HAND_MODEL` | Override the hand-classifier checkpoint |
| `PYCLASHBOT_FIELD_DETECTOR` | Override the ONNX detector path |
| `PYCLASHBOT_FIELD_CLASSES` | Override detector class YAML/config |
| `PYCLASHBOT_RECORDINGS_DIR` | Store recordings outside the default app-data directory |
| `PYCLASHBOT_BLUESTACKS_APP` | Override the BlueStacks app bundle path |
| `PYCLASHBOT_BLUESTACKS_DATA` | Override the BlueStacks data/config directory |
| `PYCLASHBOT_BLUESTACKS_MIM_APP` | Override the optional Multi-Instance Manager path |
| `PYCLASHBOT_FIELD_MIN_CONF` | Set the runtime field-detection confidence floor |
| `PYCLASHBOT_CONTINUOUS_COORDS` | Enable/disable sampled placement candidates |

Logs and recordings default to OS-appropriate application-data directories,
not the repository. Policy sampling and hand-recognition thresholds have
additional `PYCLASHBOT_*` overrides in their respective modules.

## Platform boundaries and limitations

- BlueStacks is supported on macOS and Windows when its installed layout and
  renderer are compatible with the controller.
- Generic ADB is cross-platform.
- MEmu and Google Play Games adapters are Windows-only in this checkout.
- Visual checks and templates are tied to the 419×633 coordinate space and to
  recognizable Clash Royale UI elements. A game update can invalidate them.
- The field model and card classifier are experimental models trained on
  project-specific imagery; successful loading is not a guarantee of current
  live-game accuracy.
- The policy is a small learned scorer combined with hand-written candidate
  masks and safety rules, not a general game-playing system.
- Learning history is in-memory for the current process and is not a durable
  experiment database. Enabling online training changes `models/policy.pt`.
- The GUI's learning graph shows rolling episode/per-play reward telemetry; it
  is a monitoring aid, not a training dashboard with persisted history.

### Analytics graph semantics

The cyan series is the shaped reward recorded after an online policy update at
the end of a game. The orange series is the immediate per-play reward recorded
while cards are being played. `Last loss` is a scalar label for the most recent
online update; it is not a third plotted series. `Games` counts completed online
updates, not all battles or screenshots. The graph is expected to be empty when
the policy or online-training toggle is off, and history is not retained after
the process exits.

## Project history

TowerLogic began as an emulator-control and image-recognition experiment. The
recent compatibility work focused on making that existing architecture usable
again on modern macOS/BlueStacks Air: portable paths and packaging, bundled
ADB and instance discovery, Android foreground parsing, current UI detection,
bounded reward-screen recovery, and an explicit dry-run safety mode. It did not
replace the original state machine or retrain the included models.

## Repository hygiene

Generated virtual environments, Python caches, macOS metadata, recordings,
logs, and exported ONNX files are ignored by `.gitignore`. Runtime models,
templates, class maps, and the included example datasets remain tracked. Keep
personal emulator settings and credentials in the OS application-data folder,
never in this repository.

## Dataset attribution

The included `Clash royale.v6i.yolov8/` export is attributed to Roboflow
Universe, dataset “Clash royale v6”, under the license and attribution terms
included with the export:

<https://universe.roboflow.com/angelfire/clash-royale-cylln>

## Disclaimer

Clash Royale, its characters, artwork, cards, and related intellectual property
belong to Supercell and their respective rights holders. TowerLogic is an
independent educational/technical project and is not affiliated with,
endorsed by, sponsored by, or associated with Supercell. Users are responsible
for complying with applicable software, platform, game, dataset, and
third-party terms.

## Suggested GitHub metadata

Description:

> Experimental Python computer-vision and ML pipeline for Clash Royale game-state analysis and safe emulator dry runs.

Topics:

`python`, `pytorch`, `computer-vision`, `machine-learning`, `opencv`,
`onnx-runtime`, `object-detection`, `image-classification`, `game-ai`, `adb`,
`automation`
