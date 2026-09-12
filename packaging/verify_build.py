"""Run the native artifact from an unrelated working directory before shipping."""

import json
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if sys.platform == "darwin":
    bundle = root / "dist" / "TowerLogic.app" / "Contents"
    executable = bundle / "MacOS" / "TowerLogic"
    info = plistlib.loads((bundle / "Info.plist").read_bytes())
    icon = bundle / "Resources" / info["CFBundleIconFile"]
    assert icon.read_bytes() == (root / "towerlogic/interface/assets/towerlogic.icns").read_bytes()
    subprocess.run(["codesign", "--verify", "--deep", str(bundle.parent)], check=True)
else:
    import pefile

    executable = root / "dist" / "TowerLogic" / "TowerLogic.exe"
    pe = pefile.PE(str(executable))
    types = {entry.id for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries}
    assert 3 in types and 14 in types, "Executable has no embedded icon"
    pe.close()
with tempfile.TemporaryDirectory() as directory:
    report = Path(directory) / "smoke.json"
    process = subprocess.run([str(executable), "--smoke-test", str(report)],
                             cwd=directory, timeout=240)
    if not report.exists():
        raise RuntimeError(f"Application did not write its smoke report (exit {process.returncode})")
    result = json.loads(report.read_text())
    print(json.dumps(result, indent=2))
    (root / "dist" / "smoke-test.json").write_text(json.dumps(result, indent=2))
    assert result["ok"], result.get("error")
    assert process.returncode == 0
