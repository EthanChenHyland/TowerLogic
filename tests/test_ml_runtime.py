import unittest
import math
from multiprocessing import Queue

import cv2
import numpy as np

from towerlogic.detection.hand_classifier import _opencv_image_to_pil
from towerlogic.detection.onnx_detector import Detection, OnnxTroopDetector, summarize_detections
from towerlogic.__main__ import handle_process_finished
from towerlogic.interface.ui import TowerLogicUI
from towerlogic.utils.logger import ProcessLogger


class _FakeSession:
    def __init__(self, output):
        self.output = output

    def run(self, _outputs, _inputs):
        return [self.output]


class MlRuntimeRegressionTests(unittest.TestCase):
    def test_opencv_bgr_crop_is_converted_to_rgb(self):
        bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
        self.assertEqual(_opencv_image_to_pil(bgr).getpixel((0, 0)), (30, 20, 10))

    def test_detector_clips_boxes_to_source_image(self):
        # YOLO-style output: [batch, 4 + classes, candidates].
        output = np.zeros((1, 6, 10), dtype=np.float32)
        output[0, :4, 0] = (5, 5, 20, 20)  # extends beyond a 100x100 source
        output[0, 4, 0] = 0.95
        detector = OnnxTroopDetector.__new__(OnnxTroopDetector)
        detector.input_size = 10
        detector.conf_threshold = 0.25
        detector.iou_threshold = 0.45
        detector.class_names = ["ally_unit", "enemy_unit"]
        detector.input_name = "images"
        detector.session = _FakeSession(output)

        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        box = detections[0]
        self.assertEqual((box.x1, box.y1), (0, 0))
        self.assertEqual((box.x2, box.y2), (100, 100))

    def test_detection_summary_counts_each_detection_once(self):
        detections = [
            Detection(10, 10, 20, 20, 0.8, 0),
            Detection(30, 220, 40, 230, 0.7, 1),
            Detection(50, 500, 60, 510, 0.6, 2),
        ]
        summary = summarize_detections(detections, (633, 419, 3))
        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.top + summary.mid + summary.bottom, summary.total)
        self.assertEqual(summary.left + summary.right, summary.total)

    def test_policy_progress_is_published_to_gui_queue(self):
        queue = Queue()
        logger = ProcessLogger(queue, timed=False)
        logger.add_policy_progress(1.25, 0.375)
        stats = queue.get(timeout=2)
        self.assertEqual(stats["policy_progress"], [1.25])
        self.assertEqual(stats["policy_last_reward"], 1.25)
        self.assertEqual(stats["policy_last_loss"], 0.375)
        self.assertEqual(stats["policy_games"], 1)
        queue.close()

    def test_finished_worker_keeps_learning_metrics_visible(self):
        class _UI:
            def __init__(self):
                self.states = []

            def set_button_state(self, state):
                self.states.append(state)

        class _FinishedProcess:
            def is_alive(self):
                return False

        logger = ProcessLogger(None, timed=False)
        logger.policy_progress = [2.0]
        logger.policy_step_progress = [0.25]
        logger._update_stats()
        ui = _UI()
        process, returned_logger = handle_process_finished(ui, _FinishedProcess(), logger)
        self.assertIsNone(process)
        self.assertIs(returned_logger, logger)
        self.assertEqual(returned_logger.policy_progress, [2.0])
        self.assertEqual(returned_logger.policy_step_progress, [0.25])
        self.assertEqual(ui.states, ["idle"])

    def test_learning_graph_ignores_nonfinite_values(self):
        class _Canvas:
            def __init__(self):
                self.calls = []

            def winfo_width(self):
                return 100

            def winfo_height(self):
                return 40

            def delete(self, *args):
                self.calls.append(("delete", args))

            def create_line(self, *args, **kwargs):
                self.calls.append(("line", args, kwargs))

            def create_oval(self, *args, **kwargs):
                self.calls.append(("oval", args, kwargs))

        ui = TowerLogicUI.__new__(TowerLogicUI)
        ui.policy_canvas = _Canvas()
        ui._update_policy_graph([float("nan"), 1.0], [float("inf"), -1.0])
        coordinates = [arg for call in ui.policy_canvas.calls if call[0] in {"line", "oval"} for arg in call[1]]
        self.assertTrue(coordinates)
        self.assertTrue(all(not isinstance(value, float) or math.isfinite(value) for value in coordinates))


if __name__ == "__main__":
    unittest.main()
