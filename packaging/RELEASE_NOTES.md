TowerLogic's first packaged desktop release, with Python, Tk, computer vision runtimes and inference models included.

## Downloads

- **Windows 10/11 (64-bit):** extract `TowerLogic-windows-x64.zip`, keep the whole folder together, and open `TowerLogic.exe`.
- **macOS 15+ (Apple Silicon, M1 or newer):** extract `TowerLogic-macos-arm64.zip`, move `TowerLogic.app` to Applications, and open it. This build does not support Intel Macs.

The TowerLogic logo is embedded in the Windows executable and macOS application bundle and displayed in the app header. The release builds run native GUI, model inference, resource, and multiprocessing smoke checks on both operating systems before publishing.

The apps are not developer-signed or Apple-notarized. Windows may show SmartScreen; macOS may require allowing the downloaded app in System Settings > Privacy & Security after the first launch attempt.

Install and configure your emulator separately (BlueStacks Air on Apple Silicon, or BlueStacks/MEmu on Windows). Generic Android connections require ADB. No emulator or game is bundled. Emulator gameplay is not exercised by the build tests.

Policy training uses a writable copy in your user application-data folder. Existing source checkouts keep their configured model location.

See the README for emulator configuration and dry-run instructions. `SHA256SUMS.txt` contains download checksums.
