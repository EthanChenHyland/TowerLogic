"""Configuration for TowerLogic interface elements."""

from __future__ import annotations

from dataclasses import dataclass

from towerlogic.interface.enums import (
    BATTLE_STAT_FIELDS,
    BATTLE_STAT_LABELS,
    BOT_STAT_LABELS,
    COLLECTION_STAT_FIELDS,
    COLLECTION_STAT_LABELS,
    BotStatField,
    StatField,
    UIField,
)
from towerlogic.utils.platform import is_macos


@dataclass
class StatConfig:
    """Configuration for a stat display element."""

    key: StatField | BotStatField
    title: str
    size: tuple[int, int] = (6, 1)


@dataclass
class JobConfig:
    """Configuration for a job checkbox element."""

    key: UIField
    title: str
    default: bool = False
    extras: dict[UIField, ComboConfig] | None = None


@dataclass
class RadioConfig:
    """Configuration for a radio button element."""

    key: UIField
    title: str
    group_id: str
    default: bool = False


@dataclass
class ComboConfig:
    """Configuration for a combo box element."""

    key: UIField
    label: str
    values: list[str | int]
    default: str | int = ""
    size: tuple[int, int] = (5, 1)
    label_size: tuple[int, int] = (6, 1)


# Statistics Configuration
BATTLE_STATS = [StatConfig(field, BATTLE_STAT_LABELS[field]) for field in BATTLE_STAT_FIELDS]

COLLECTION_STATS = [StatConfig(field, COLLECTION_STAT_LABELS[field]) for field in COLLECTION_STAT_FIELDS]

BOT_STATS = [
    StatConfig(BotStatField.RESTARTS_AFTER_FAILURE, BOT_STAT_LABELS[BotStatField.RESTARTS_AFTER_FAILURE]),
    StatConfig(BotStatField.TIME_SINCE_START, BOT_STAT_LABELS[BotStatField.TIME_SINCE_START], size=(8, 1)),
]

# Job Configuration
JOBS = [
    JobConfig(UIField.CLASSIC_1V1_USER_TOGGLE, "Classic 1v1 battles", default=False),
    JobConfig(UIField.CLASSIC_2V2_USER_TOGGLE, "Classic 2v2 battles", default=False),
    JobConfig(UIField.TROPHY_ROAD_USER_TOGGLE, "Trophy Road battles", default=True),
]

# Emulator Settings Configuration
MEMU_SETTINGS = [
    RadioConfig(UIField.OPENGL_TOGGLE, "OpenGL", "render_mode_radio"),
    RadioConfig(UIField.DIRECTX_TOGGLE, "DirectX", "render_mode_radio", default=True),
]

# BlueStacks specific renderer settings
# DirectX is default on Windows, Vulkan on macOS (DirectX not available on macOS)
BLUESTACKS_SETTINGS = [
    RadioConfig(UIField.BS_RENDERER_GL, "OpenGL", "bs_render_mode_radio"),
    RadioConfig(UIField.BS_RENDERER_DX, "DirectX", "bs_render_mode_radio", default=not is_macos()),
    RadioConfig(UIField.BS_RENDERER_VK, "Vulkan", "bs_render_mode_radio", default=is_macos()),
]

EMULATOR_CHOICE = [
    RadioConfig(UIField.MEMU_EMULATOR_TOGGLE, "Memu", "emulator_type_radio", default=True),
    RadioConfig(UIField.GOOGLE_PLAY_EMULATOR_TOGGLE, "Google Play", "emulator_type_radio"),
    RadioConfig(UIField.BLUESTACKS_EMULATOR_TOGGLE, "BlueStacks 5", "emulator_type_radio"),
]

# Google Play Settings Configuration
GOOGLE_PLAY_SETTINGS = [
    ComboConfig(UIField.GP_ANGLE, "angle", ["true", "false"]),
    ComboConfig(UIField.GP_VULKAN, "vulkan", ["true", "false"]),
    ComboConfig(UIField.GP_GLES, "gles", ["true", "false"]),
    ComboConfig(UIField.GP_SURFACELESS, "surfaceless", ["true", "false"]),
    ComboConfig(UIField.GP_EGL, "egl", ["true", "false"]),
    ComboConfig(UIField.GP_BACKEND, "backend", ["gfxstream", "angle", "swiftshader"]),
    ComboConfig(UIField.GP_WSI, "wsi", ["vk", "glx"]),
]

# All user configuration keys (auto-generated from configs)
USER_CONFIG_KEYS = (
    [job.key.value for job in JOBS]
    + [radio.key.value for radio in MEMU_SETTINGS + BLUESTACKS_SETTINGS + EMULATOR_CHOICE]
    + [combo.key.value for combo in GOOGLE_PLAY_SETTINGS]
    + [
        UIField.THEME_NAME.value,
        UIField.RECORD_FIGHTS_TOGGLE.value,
        UIField.BS_INSTANCE_NAME.value,
        UIField.POLICY_TOGGLE.value,
        UIField.POLICY_EPSILON.value,
        UIField.POLICY_MODEL_PATH.value,
        UIField.POLICY_TRAIN_TOGGLE.value,
        UIField.POLICY_LR.value,
        UIField.POLICY_FREE_PLACEMENT.value,
        UIField.POLICY_MIN_ELIXIR.value,
        UIField.POLICY_ALLOW_HOLD.value,
        UIField.POLICY_DEBUG_DETECTION.value,
        UIField.POLICY_DEBUG_OVERLAY.value,
        UIField.POLICY_DEBUG_HP_LOG.value,
        UIField.POLICY_DEBUG_HAND_CONF.value,
        UIField.POLICY_STRICT_ELIXIR.value,
        UIField.POLICY_SPELL_GATE.value,
    ]  # Data settings
)

# Keys to disable when bot is running
DISABLE_KEYS = [*USER_CONFIG_KEYS, "Start"]
