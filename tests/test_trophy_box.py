import unittest

import cv2
import numpy as np

from towerlogic.bot.nav import check_for_trophy_box_reward, clear_trophy_box_reward
from towerlogic.emulators.adb_base import AdbBasedController


class _Logger:
    def __init__(self):
        self.messages = []

    def change_status(self, message):
        self.messages.append(message)

    def log(self, message):
        self.messages.append(message)


def _reward_image():
    hsv = np.zeros((633, 419, 3), dtype=np.uint8)
    hsv[:] = (110, 200, 150)
    hsv[220:410, 110:310] = (145, 180, 180)
    hsv[535:625, 80:330] = (15, 200, 220)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _battle_image():
    image = np.zeros((633, 419, 3), dtype=np.uint8)
    for y, x in [(528, 49), (532, 77), (546, 52), (546, 77)]:
        image[y, x] = (255, 255, 255)
    image[618, 115] = (255, 0, 255)
    return image


def _main_menu_image():
    image = np.zeros((633, 419, 3), dtype=np.uint8)
    coords = [(14, 209), (14, 325), (19, 298), (17, 399),
              (581, 261), (584, 166), (621, 166)]
    colors = ([255, 255, 255], [255, 255, 255], [53, 199, 233],
              [25, 198, 65], [138, 105, 71], [139, 105, 72], [155, 120, 82])
    for (y, x), color in zip(coords, colors):
        image[y, x] = color
    return image


class _Emulator:
    def __init__(self, terminal_image=None):
        self.screenshots = 0
        self.clicks = []
        self.terminal_image = terminal_image if terminal_image is not None else _battle_image()

    def screenshot(self):
        self.screenshots += 1
        return _reward_image() if self.screenshots <= 3 else self.terminal_image

    def click(self, x, y):
        self.clicks.append((x, y))


class _DryRunEmulator:
    click = AdbBasedController.click

    def __init__(self):
        self.commands = []
        self.suppressed = []

    def screenshot(self):
        return _reward_image()

    def dry_run_enabled(self):
        return True

    def log_suppressed_input(self, message):
        self.suppressed.append(message)

    def adb(self, command):
        self.commands.append(command)


class TrophyBoxTests(unittest.TestCase):
    def test_detects_reward_screen_and_rejects_blank(self):
        reward = _Emulator()
        self.assertTrue(check_for_trophy_box_reward(reward))

        blank = _Emulator()
        blank.screenshot = lambda: np.zeros((633, 419, 3), dtype=np.uint8)
        self.assertFalse(check_for_trophy_box_reward(blank))

    def test_clicks_until_battle_returns(self):
        emulator = _Emulator()
        logger = _Logger()
        self.assertEqual(clear_trophy_box_reward(emulator, logger, timeout=2.0), "battle")
        self.assertEqual(emulator.clicks, [(209, 316), (209, 316)])
        self.assertTrue(any("Battle restored" in message for message in logger.messages))

    def test_clears_to_main_menu(self):
        emulator = _Emulator(terminal_image=_main_menu_image())
        logger = _Logger()
        self.assertEqual(clear_trophy_box_reward(emulator, logger, timeout=2.0), "main_menu")
        self.assertTrue(any("Main menu reached" in message for message in logger.messages))

    def test_persistent_box_times_out(self):
        emulator = _Emulator(terminal_image=_reward_image())
        emulator.screenshot = lambda: _reward_image()
        result = clear_trophy_box_reward(emulator, _Logger(), timeout=0.01)
        self.assertEqual(result, "timeout")

    def test_dry_run_suppresses_recovery_click(self):
        emulator = _DryRunEmulator()
        result = clear_trophy_box_reward(emulator, _Logger(), timeout=0.01)
        self.assertEqual(result, "timeout")
        self.assertEqual(emulator.commands, [])
        self.assertTrue(any("tap x=209 y=316" in message for message in emulator.suppressed))

    def test_main_menu_does_not_trigger_trophy_recovery(self):
        emulator = _Emulator(terminal_image=_main_menu_image())
        emulator.screenshot = lambda: _main_menu_image()
        self.assertEqual(clear_trophy_box_reward(emulator, _Logger()), "not_detected")
        self.assertEqual(emulator.clicks, [])


if __name__ == "__main__":
    unittest.main()
