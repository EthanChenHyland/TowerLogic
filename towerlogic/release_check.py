"""Exercise the actual frozen GUI, inference runtimes, assets, and process spawn."""

import json
import multiprocessing
import sys
import traceback
from pathlib import Path


def _child(queue):
    queue.put("spawn-ok")


def main(report_path: str) -> None:
    report = Path(report_path)
    result = {"platform": sys.platform, "ok": False}
    ui = None
    try:
        import numpy as np
        import torch
        from PIL import Image
        from towerlogic.bot.policy import SimplePolicyNet
        from towerlogic.detection.hand_classifier import load_hand_classifier
        from towerlogic.detection.onnx_detector import OnnxTroopDetector
        from towerlogic.interface.ui import TowerLogicUI
        from towerlogic.utils.resources import default_policy_path

        root = Path(__file__).resolve().parent
        for name in ("towerlogic.png", "towerlogic.ico", "towerlogic.icns"):
            with Image.open(root / "interface" / "assets" / name) as icon:
                icon.load()
                assert icon.width >= 128
        assert list((root / "detection" / "reference_images").rglob("*.png"))
        assert list((root / "bot" / "detection" / "card_templates").glob("*.png"))
        detector = OnnxTroopDetector()
        assert detector.class_names, "Missing detector class names"
        detector.detect(np.zeros((633, 419, 3), dtype=np.uint8))
        classifier = load_hand_classifier(device="cpu")
        classifier.predict(np.zeros((100, 100, 3), dtype=np.uint8))
        policy = SimplePolicyNet()
        policy.load_state_dict(torch.load(default_policy_path(), map_location="cpu", weights_only=True))
        assert torch.isfinite(policy(torch.zeros(1, 30))).all()
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_child, args=(queue,))
        process.start()
        try:
            assert queue.get(timeout=60) == "spawn-ok"
            process.join(timeout=15)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            queue.close()
        ui = TowerLogicUI()
        ui.update()
        assert ui.winfo_viewable()
        assert ui._icon_image.width() == 128
        assert ui.title_icon.cget("image")
        ui.after(500, ui.quit)
        ui.mainloop()
        result["ok"] = True
        result["checks"] = ["native-gui", "header-logo", "icon-assets", "onnx-inference", "hand-inference", "policy-inference", "multiprocessing-spawn", "reference-images"]
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        if ui is not None:
            ui.destroy()
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
