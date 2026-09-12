import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).parent
assets = root / "towerlogic" / "interface" / "assets"
version = (root / "towerlogic" / "__version__").read_text().strip()
icon = assets / ("towerlogic.icns" if sys.platform == "darwin" else "towerlogic.ico")
datas = collect_data_files("towerlogic") + collect_data_files("ttkbootstrap")
datas += [(str(root / "models" / "policy.pt"), "models"),
          (str(root / "Clash royale.v6i.yolov8" / "data.yaml"), "Clash royale.v6i.yolov8")]
a = Analysis([str(root / "packaging" / "launcher.py")], pathex=[str(root)],
             binaries=[], datas=datas, hiddenimports=["towerlogic.release_check"],
             excludes=["ultralytics", "matplotlib", "IPython", "pytest", "tensorboard"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TowerLogic",
          console=False, icon=str(icon), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="TowerLogic", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="TowerLogic.app", icon=str(icon),
                 bundle_identifier="com.ethanchenhyland.towerlogic",
                 info_plist={"CFBundleShortVersionString": version,
                             "CFBundleVersion": version,
                             "NSHighResolutionCapable": True})
