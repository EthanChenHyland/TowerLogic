import subprocess
import unittest

import numpy as np

from towerlogic.bot.nav import detect_current_clash_main_menu, inspect_clash_main_menu
from towerlogic.emulators.bluestacks import BlueStacksEmulatorController


class _Logger:
    def __init__(self):
        self.messages = []

    def log(self, message):
        self.messages.append(message)


class ForegroundDetectionTests(unittest.TestCase):
    def _controller(self, outputs):
        controller = BlueStacksEmulatorController.__new__(BlueStacksEmulatorController)
        controller.logger = _Logger()
        controller.commands = []

        def adb(command, binary_output=False):
            controller.commands.append(command)
            return subprocess.CompletedProcess([], 0, outputs.get(command, ""), "")

        controller.adb = adb
        return controller

    def test_android13_top_resumed_activity(self):
        controller = self._controller(
            {
                "shell dumpsys activity activities": (
                    "topResumedActivity=ActivityRecord{abc u0 "
                    "com.supercell.clashroyale/com.supercell.titan.GameApp} t26}\n"
                )
            }
        )
        self.assertEqual(
            controller._get_foreground_component(),
            ("com.supercell.clashroyale", "com.supercell.titan.GameApp"),
        )
        self.assertEqual(controller._get_foreground_package(), "com.supercell.clashroyale")

    def test_legacy_window_focus_fallback(self):
        controller = self._controller(
            {
                "shell dumpsys window": (
                    "mCurrentFocus=Window{abc u0 com.supercell.clashroyale/"
                    "com.supercell.titan.GameApp}\n"
                )
            }
        )
        self.assertEqual(
            controller._get_foreground_component(),
            ("com.supercell.clashroyale", "com.supercell.titan.GameApp"),
        )

    def test_unresolved_state_is_logged(self):
        controller = self._controller({})
        self.assertIsNone(controller._get_foreground_package())
        self.assertTrue(any("Unable to resolve foreground" in m for m in controller.logger.messages))

    def test_main_menu_inspection_reports_pixel_mismatches(self):
        image = np.zeros((633, 419, 3), dtype=np.uint8)
        # BGR values are intentionally set to the legacy expected signature.
        expected = ([255, 255, 255], [255, 255, 255], [53, 199, 233],
                     [25, 198, 65], [138, 105, 71], [139, 105, 72], [155, 120, 82])
        coords = [(14, 209), (14, 325), (19, 298), (17, 399),
                  (581, 261), (584, 166), (621, 166)]
        for (y, x), pixel in zip(coords, expected):
            image[y, x] = pixel
        recognized, pixels, matches = inspect_clash_main_menu(image)
        self.assertTrue(recognized)
        self.assertEqual(len(pixels), 7)
        self.assertTrue(all(matches[0]))

    def test_current_ui_detector_matches_each_mode_template_and_rejects_blank(self):
        cv2 = __import__("cv2")
        modes = {
            "classic_1v1_template_match": "selected_1v1_on_main/1.png",
            "classic_2v2_template_match": "selected_2v2_on_main/1.png",
            "trophy_template_match": "selected_trophy_road_on_main/4.png",
        }
        for detail_key, relative_path in modes.items():
            with self.subTest(mode=detail_key):
                menu = np.zeros((633, 419, 3), dtype=np.uint8)
                menu[455:535, 135:285] = (0, 200, 255)
                template = cv2.imread(f"towerlogic/detection/reference_images/{relative_path}")
                h, w = template.shape[:2]
                menu[470:470 + h, 295:295 + w] = template
                recognized, details = detect_current_clash_main_menu(menu)
                self.assertTrue(recognized)
                self.assertTrue(details[detail_key])
                self.assertGreaterEqual(details["battle_button_score"], 0.20)

        blank = np.zeros((633, 419, 3), dtype=np.uint8)
        recognized, details = detect_current_clash_main_menu(blank)
        self.assertFalse(recognized)
        self.assertFalse(details["selected_mode_template_match"])

    def test_current_ui_detector_accepts_current_mode_art_with_selected_battle_tab(self):
        # Current Classic modes can use art that no longer matches the older
        # selected-mode templates.  The shared Battle button and selected
        # bottom Battle tab are the mode-independent home-screen cues.
        menu = np.zeros((633, 419, 3), dtype=np.uint8)
        menu[455:535, 135:285] = (0, 200, 255)
        menu[572:633, 145:279] = (150, 110, 70)
        recognized, details = detect_current_clash_main_menu(menu)
        self.assertTrue(recognized)
        self.assertFalse(details["selected_mode_template_match"])
        self.assertGreaterEqual(details["battle_tab_score"], 0.45)


if __name__ == "__main__":
    unittest.main()
