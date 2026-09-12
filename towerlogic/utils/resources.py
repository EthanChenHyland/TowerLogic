"""Paths for bundled resources and writable user models."""

import shutil
import sys
from pathlib import Path

from towerlogic.utils.platform import get_app_data_dir


def default_policy_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not getattr(sys, "frozen", False):
        return root / "models" / "policy.pt"
    target = Path(get_app_data_dir("TowerLogic")) / "models" / "policy.pt"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "models" / "policy.pt", target)
    return target
