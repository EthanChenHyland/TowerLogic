"""Frozen entry point; dispatch multiprocessing before importing the GUI."""

import multiprocessing
import sys


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TowerLogic.Desktop")
    if "--smoke-test" in sys.argv:
        from towerlogic.release_check import main

        main(sys.argv[sys.argv.index("--smoke-test") + 1])
    else:
        from towerlogic.__main__ import main_gui
        from towerlogic.utils.cli_config import arg_parser

        main_gui(start_on_run=arg_parser().start)
