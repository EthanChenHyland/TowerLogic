import os
import subprocess
import unittest
from unittest.mock import patch

from towerlogic.emulators.adb_base import AdbBasedController
from towerlogic.emulators.bluestacks import BlueStacksEmulatorController


class _Logger:
    def __init__(self):
        self.messages = []

    def log(self, message):
        self.messages.append(message)

    def change_status(self, message):
        self.messages.append(message)


class _FakeAdbController(AdbBasedController):
    def __init__(self):
        self.logger = _Logger()
        self.commands = []

    def adb(self, command, binary_output=False):
        self.commands.append(command)
        return subprocess.CompletedProcess([], 0, b"" if binary_output else "", "")

    def _check_app_installed(self, package_name):
        return True


class DryRunInputTests(unittest.TestCase):
    def test_dry_run_suppresses_click_and_swipe(self):
        controller = _FakeAdbController()
        with patch.dict(os.environ, {"TOWERLOGIC_DRY_RUN": "1"}):
            controller.click(10, 20, clicks=2, interval=0.1)
            controller.swipe(1, 2, 3, 4)

        self.assertEqual(controller.commands, [])
        self.assertTrue(any("tap x=10 y=20" in message for message in controller.logger.messages))
        self.assertTrue(any("swipe (1,2)->(3,4)" in message for message in controller.logger.messages))

    def test_normal_mode_preserves_adb_input(self):
        controller = _FakeAdbController()
        with patch.dict(os.environ, {}, clear=True):
            controller.click(10, 20)
            controller.swipe(1, 2, 3, 4)

        self.assertEqual(
            controller.commands,
            ["shell input tap 10 20", "shell input swipe 1 2 3 4"],
        )

    def test_bluestacks_launch_is_single_canonical_intent(self):
        controller = BlueStacksEmulatorController.__new__(BlueStacksEmulatorController)
        controller.logger = _Logger()
        controller.commands = []
        controller._resolve_launch_activity = lambda package: f"{package}/.GameApp"
        controller._get_foreground_package = lambda: "com.supercell.clashroyale"

        def fake_adb(command, binary_output=False):
            controller.commands.append(command)
            return subprocess.CompletedProcess([], 0, "", "")

        controller.adb = fake_adb
        with patch.dict(os.environ, {"TOWERLOGIC_DRY_RUN": "1"}):
            self.assertTrue(controller._force_launch_clash("com.supercell.clashroyale", retries=1, delay=0))

        self.assertEqual(
            controller.commands,
            [
                "shell am start -W -a android.intent.action.MAIN "
                "-c android.intent.category.LAUNCHER -n "
                "com.supercell.clashroyale/.GameApp"
            ],
        )
        self.assertFalse(any("monkey" in command for command in controller.commands))


if __name__ == "__main__":
    unittest.main()
