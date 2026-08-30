# TowerLogic

TowerLogic is a Python-based Clash Royale automation and computer vision project that combines traditional image processing, PyTorch-based machine learning, YOLOv8-formatted detection data, emulator control, and automated gameplay decision logic.

The project was developed iteratively over several months as an experimentation platform for interpreting live game-state information and using those observations to support automated actions during Clash Royale matches.

## Overview

TowerLogic attempts to convert live Clash Royale gameplay into structured information that can be used by an automated policy.

The project includes several complementary approaches:

- computer vision and image recognition for game-state detection
- a trained PyTorch classifier for identifying cards in the player hand
- YOLOv8-formatted object-detection data for arena/game elements
- state-vector construction from gameplay information
- policy-based card and placement selection
- emulator and ADB integration for interacting with the game
- a desktop graphical interface for configuring and monitoring runs

Rather than relying on a single model, TowerLogic combines learned models, image-processing techniques, and rule/policy logic into one gameplay pipeline.

## Key Features

### Machine Learning

TowerLogic includes a PyTorch training pipeline for classifying cards from gameplay images.

The classifier workflow includes:

- training and validation datasets loaded with `torchvision.datasets.ImageFolder`
- image resizing and normalization
- light color augmentation during training
- cross-entropy loss
- Adam optimization
- automatic CPU, CUDA, or Apple Silicon MPS device selection
- validation accuracy tracking
- checkpointing of the best-performing model
- exported class-name mappings for inference

A trained hand-card classifier and corresponding class map are included under:

```text
towerlogic/models/
````

The training utility is located at:

```text
scripts/train_hand_classifier.py
```

### Computer Vision

The project uses OpenCV and NumPy throughout its visual-processing pipeline.

TowerLogic includes:

* gameplay screenshot analysis
* card-template matching
* card and interface recognition
* image preprocessing
* game-state visual detection
* debugging and detection utilities

The repository also contains a YOLOv8-format Clash Royale dataset used for computer-vision experimentation.

### YOLOv8 Dataset

The included detection dataset contains:

* 1,792 annotated images
* YOLOv8-format labels
* annotations covering cards, HP bars, towers, and related game elements

The dataset was exported from Roboflow and is provided under **CC BY 4.0**.

Original dataset:

[https://universe.roboflow.com/angelfire/clash-royale-cylln](https://universe.roboflow.com/angelfire/clash-royale-cylln)

See the dataset documentation under:

```text
Clash royale.v6i.yolov8/
```

for the original attribution and licensing information.

### Gameplay Policy

TowerLogic includes an experimental gameplay policy that transforms detected game-state information into candidate actions.

The policy state can incorporate information such as:

* elapsed match time
* current elixir
* preferred lane
* available cards
* detected enemy activity
* win-condition availability
* tower health
* recent tower-health changes
* lane bias
* defensive state
* crown differential

The policy can then evaluate:

* which card to play
* which side of the arena to use
* candidate placement coordinates
* safe placement bounds
* spell and unit-specific placement behavior

Card placements can use either predefined candidate positions or dynamically sampled coordinates within safe arena regions.

## Automated Decision Logic

TowerLogic combines visual detections and structured game-state information with gameplay decision logic.

The system supports:

* card-selection decisions
* placement-coordinate selection
* lane preferences
* defensive behavior
* elixir-based restrictions
* card-group-specific placement behavior
* optional learned-policy behavior
* configurable exploration settings

The project also includes saved policy model checkpoints under:

```text
models/
```

These were used during experimentation with learned gameplay behavior.

## Card Recognition

In addition to machine-learning-based classification, TowerLogic contains a collection of card-reference templates used by the image-recognition pipeline.

These templates allow the project to compare live screen regions against known visual references when appropriate.

This hybrid approach makes it possible to experiment with both:

* deterministic/template-based recognition
* learned image classification

rather than relying exclusively on one detection method.

## Emulator Support

TowerLogic includes support for multiple Android-emulation and device-control environments.

The codebase contains integration for:

* MEmu
* BlueStacks
* Google Play Games
* generic ADB-connected devices

The emulator layer is responsible for launching or connecting to a game environment and providing the interaction surface needed by the automation system.

## User Interface

TowerLogic includes a desktop GUI built with `ttkbootstrap`.

The interface provides configuration for:

* emulator selection
* gameplay job selection
* deck cycling/randomization
* battle modes
* fight recording
* policy-model settings
* exploration parameters
* training behavior
* placement behavior
* detection debugging
* rendering options
* runtime status and analytics

The interface passes the selected configuration to a separate worker process that runs the automation pipeline.

## Architecture

A simplified project structure looks like this:

```text
TowerLogic/
├── towerlogic/
│   ├── bot/
│   │   ├── detection/
│   │   ├── card_detection.py
│   │   ├── fight.py
│   │   ├── policy.py
│   │   ├── worker.py
│   │   └── ...
│   ├── emulators/
│   │   ├── adb.py
│   │   ├── bluestacks.py
│   │   ├── google_play.py
│   │   └── memu.py
│   ├── interface/
│   │   ├── ui.py
│   │   ├── config.py
│   │   └── ...
│   ├── models/
│   │   ├── hand_card_classifier.pt
│   │   └── hand_card_classifier.class_map.json
│   ├── utils/
│   ├── __init__.py
│   └── __main__.py
├── scripts/
│   └── train_hand_classifier.py
├── models/
│   ├── policy.pt
│   └── policy_backup_2026-03-24.pt
├── Clash royale.v6i.yolov8/
├── pyproject.toml
└── run_bot.sh
```

## Tech Stack

### Core

* Python 3.12+
* OpenCV
* NumPy
* Pillow
* ttkbootstrap

### Machine Learning

* PyTorch
* torchvision
* YOLOv8-format datasets
* image classification
* computer vision
* object-detection experimentation

### Automation / Integration

* ADB
* MEmu integration
* BlueStacks integration
* Google Play Games integration
* multiprocessing

## Installation

Create a Python environment:

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the core project:

```bash
pip install -e .
```

For machine-learning functionality, also install the optional ML dependencies:

```bash
pip install -e ".[ml]"
```

The training script additionally uses `torchvision`, so install it if needed:

```bash
pip install torchvision
```

## Running TowerLogic

TowerLogic exposes its main interface through the Python module:

```bash
python -m towerlogic
```

The repository also contains:

```text
run_bot.sh
```

for launching the project in supported local environments.

Actual emulator configuration will depend on the platform and Android environment being used.

## Training the Hand-Card Classifier

The included classifier-training utility expects an ImageFolder-style dataset with train and validation directories.

Example:

```bash
python scripts/train_hand_classifier.py \
  --data-dir /path/to/card_dataset \
  --epochs 12 \
  --batch-size 64
```

The script:

1. loads training and validation images
2. normalizes class names
3. selects CPU, CUDA, or MPS automatically
4. trains the card classifier
5. evaluates validation accuracy after each epoch
6. saves the best-performing checkpoint
7. writes a corresponding class-name mapping file

The default trained-model output is:

```text
towerlogic/models/hand_card_classifier.pt
```

## Apple Silicon Support

The training code can automatically select Apple's Metal Performance Shaders backend when available:

```text
mps
```

Otherwise it falls back to CUDA or CPU depending on the machine.

## Development Notes

TowerLogic was developed as an experimental project rather than a production gameplay product.

The codebase contains multiple generations of:

* image-recognition logic
* card-detection techniques
* gameplay strategies
* model checkpoints
* policy experimentation
* emulator integrations
* debugging tools

As a result, some components are research-oriented or exploratory rather than part of one finalized architecture.

## Repository Cleanup

Local development environments and generated files should not be committed.

A recommended `.gitignore` includes:

```gitignore
.venv/
venv/

__pycache__/
*.py[cod]

.DS_Store
._*
__MACOSX/

runs/

*.log
```

The virtual environment should be named `.venv/` locally and recreated from project dependencies rather than uploaded to GitHub.

## Dataset Attribution

The included Clash Royale YOLOv8 dataset was obtained through Roboflow.

**Dataset:** Clash royale v6
**Source:** Roboflow Universe
**Images:** 1,792
**Format:** YOLOv8
**License:** CC BY 4.0

Source:

[https://universe.roboflow.com/angelfire/clash-royale-cylln](https://universe.roboflow.com/angelfire/clash-royale-cylln)

The dataset attribution files included with the export should remain in the repository if the dataset itself is redistributed.

## Disclaimer

TowerLogic is an independent technical and educational project focused on machine learning, computer vision, automation, and game-state analysis.

Clash Royale, its characters, artwork, cards, and related intellectual property belong to Supercell and their respective rights holders.

This project is not affiliated with, endorsed by, sponsored by, or associated with Supercell.

Users are responsible for complying with applicable software, platform, game, dataset, and third-party terms when running or modifying the project.

````

**GitHub description:**

```text
Python computer vision and ML project for Clash Royale combining PyTorch card classification, YOLOv8 data, image recognition, emulator control, and automated gameplay policy logic.
````

**Topics:**

```text
python
pytorch
computer-vision
machine-learning
yolov8
opencv
image-classification
object-detection
game-ai
automation
adb
clash-royale
```
