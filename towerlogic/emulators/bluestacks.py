import csv
import io
import json
import os
import re
import subprocess
import time
from contextlib import suppress
from os.path import normpath

from towerlogic.bot.nav import (
    check_for_trophy_box_reward,
    check_if_in_battle,
    clear_trophy_box_reward,
    detect_current_clash_main_menu,
    inspect_clash_main_menu,
)
from towerlogic.emulators.adb_base import AdbBasedController
from towerlogic.utils.cancellation import interruptible_sleep
from towerlogic.utils.platform import Platform, is_macos

DEBUG = False


class BlueStacksEmulatorController(AdbBasedController):
    """BlueStacks 5 controller using HD-Adb.

    - Read instance port from bluestacks.conf.
    - All ADB device commands are scoped with -s 127.0.0.1:<instance_port>.
    - Start/stop only this instance. Allows for future multi instance implementation for farming multiple accounts.
    """

    supported_platforms = [Platform.WINDOWS, Platform.MACOS]
    DEFAULT_DPI = 320
    DEFAULT_MAX_FPS = 60

    def __init__(self, logger, render_settings: dict | None = None, instance_name: str | None = None):
        self.logger = logger
        self.expected_dims = (419, 633)  # Bypassing bs5's stupid dim limits

        desired_name = (instance_name or "").strip()
        self.instance_name = desired_name or "towerlogic-136"
        self._explicit_instance_name = bool(desired_name)
        self.internal_name: str | None = None
        self.instance_port: int | None = None
        self.device_serial: str | None = None  # "127.0.0.1:<port>"

        self.adb_server_port: int = 5041
        self.adb_env = os.environ.copy()
        self.adb_env["ADB_SERVER_PORT"] = str(self.adb_server_port)

        # Do not auto-close BlueStacks 5 when controller is GC'd
        self._auto_stop_on_del = False  # yes it leaks, no I don't want to talk about it

        self.render_settings = render_settings or {}

        # Clean up our target instance only
        self.stop()
        while self._is_this_instance_running():  # Because Windows: I'm closing, also Windows: isn't closing
            self.stop()
            interruptible_sleep(1)

        # Discover install
        install_base = self._find_install_location()
        self.base_folder = install_base
        if DEBUG:
            print(f"[Bluestacks 5] InstallDir: {install_base}")

        # Platform-aware executable names
        if is_macos():
            self.emulator_executable_path = os.path.join(install_base, "BlueStacks")
            self.adb_path = os.path.join(install_base, "hd-adb")
        else:
            self.emulator_executable_path = os.path.join(install_base, "HD-Player.exe")
            self.adb_path = os.path.join(install_base, "HD-Adb.exe")
        if DEBUG:
            print(f"[Bluestacks 5] HD-Player: {self.emulator_executable_path}")
            print(f"[Bluestacks 5] HD-Adb: {self.adb_path}")
        for path in [install_base, self.emulator_executable_path, self.adb_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file or directory not found: {path}")

        # Platform-aware config paths
        self.bs_conf_path, self.mim_meta_path = self._find_config_paths()
        # BlueStacks Air no longer always ships the legacy Multi-Instance
        # Manager metadata file.  The authoritative instance/port settings
        # are present in bluestacks.conf, so metadata is optional; retain it
        # only as an additional display-name mapping when available.
        if not os.path.isfile(self.mim_meta_path):
            print(
                f"[Bluestacks 5] Optional MimMetaData.json not found at {self.mim_meta_path}; "
                "using bluestacks.conf for instance discovery."
            )
        if DEBUG:
            print(f"[Bluestacks 5] bs_conf_path: {self.bs_conf_path}")
            print(f"[Bluestacks 5] mim_meta_path: {self.mim_meta_path}")

        # Resolve instance and its ADB port
        self._ensure_managed_instance()
        if not self.instance_port:
            raise RuntimeError("No ADB port found for the target instance in bluestacks.conf.")
        self.device_serial = f"127.0.0.1:{self.instance_port}"

        # Reset our private adb server
        self._reset_adb_server()

        # Boot flow
        while self.restart() is False:
            print("[BlueStacks 5] Restart failed, retrying...")
            interruptible_sleep(2)

    def _find_install_location(self) -> str:
        """Locate BlueStacks 5 installation folder."""
        if is_macos():
            # macOS: Check standard app location
            app_path = os.path.expanduser(os.getenv("PYCLASHBOT_BLUESTACKS_APP", "/Applications/BlueStacks.app"))
            macos_dir = os.path.join(app_path, "Contents", "MacOS")
            # macOS uses "BlueStacks" as the main executable, not "HD-Player"
            if os.path.isdir(macos_dir) and os.path.isfile(os.path.join(macos_dir, "BlueStacks")):
                return macos_dir
            raise FileNotFoundError("BlueStacks.app not found in /Applications")

        # Windows: Registry lookup
        install_dir = self._get_bluestacks_registry_value("InstallDir")
        if not install_dir:
            raise FileNotFoundError("BlueStacks 5 installation not found (InstallDir missing).")
        base = normpath(str(install_dir))
        if os.path.isdir(base) and os.path.isfile(os.path.join(base, "HD-Player.exe")):
            return base
        raise FileNotFoundError(
            "BlueStacks 5 installation not found (HD-Player.exe missing)."
        )  # Seriously wtf happened better reinstall to fix

    def _find_config_paths(self) -> tuple[str, str]:
        """Return (bluestacks_conf_path, mim_meta_path) for the current platform."""
        if is_macos():
            base = os.path.expanduser(
                os.getenv("PYCLASHBOT_BLUESTACKS_DATA", "/Users/Shared/Library/Application Support/BlueStacks")
            )
            bs_conf = os.path.join(base, "bluestacks.conf")
            # macOS stores MimMetaData.json under Engine/UserData
            mim_meta = os.path.join(base, "Engine", "UserData", "MimMetaData.json")
            return bs_conf, mim_meta

        # Windows: Use registry DataDir
        data_dir = self._get_bluestacks_registry_value("DataDir")
        dd = normpath(str(data_dir)) if data_dir else None
        if not dd:
            raise FileNotFoundError("BlueStacks DataDir not found in registry.")
        bs_conf = os.path.join(os.path.dirname(dd), "bluestacks.conf")
        mim_meta = os.path.join(dd, "UserData", "MimMetaData.json")
        return bs_conf, mim_meta

    def _get_bluestacks_registry_value(self, value_name: str) -> str | None:
        """Read a BlueStacks_nxt registry value from HKLM."""
        import winreg

        reg_paths = [r"SOFTWARE\BlueStacks_nxt"]
        with suppress(FileNotFoundError, OSError):
            reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
            for subkey in reg_paths:
                with suppress(FileNotFoundError, OSError):
                    with winreg.OpenKey(reg, subkey) as k:
                        val = winreg.QueryValueEx(k, value_name)[0]
                        if isinstance(val, str) and val.strip():
                            return val
        return None

    def _read_text(self, path: str) -> str | None:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None

    def _read_json(self, path: str) -> dict:
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _get_conf_value(self, conf_text: str, key: str) -> str | None:
        m = re.search(rf'^{re.escape(key)}="([^"]*)"', conf_text, flags=re.M)
        return m.group(1) if m else None

    def _set_conf_value(self, conf_text: str, key: str, val: str) -> str:
        pat = rf'^(?P<k>{re.escape(key)})="[^"]*"'
        repl = rf'\g<k>="{val}"'
        new, n = re.subn(pat, repl, conf_text, flags=re.M)
        if n == 0:
            if not new.endswith("\n"):
                new += "\n"
            new += f'{key}="{val}"\n'
            # bad solution but fuck it, it works
        return new

    def _list_instance_internals(self, conf_text: str) -> list[str]:
        """List all instance internal names (Pie64, Tiramisu64, etc)."""
        names = set()
        # Match any instance pattern: Pie64, Pie64_1, Tiramisu64, Tiramisu64_2, etc.
        for m in re.finditer(r"^bst\.instance\.([A-Za-z0-9]+(?:_\d+)?)\.", conf_text, flags=re.M):
            names.add(m.group(1))
        # Sort: numbered instances first, then base instances
        return sorted(
            names, key=lambda s: (0 if "_" in s else 1, int(s.split("_")[1]) if "_" in s else -1), reverse=True
        )

    def _find_internal_in_conf_by_display(self, conf_path: str, display_name: str) -> str | None:
        text = self._read_text(conf_path) or ""
        m = re.search(rf'^bst\.instance\.([^.=]+)\.display_name="{re.escape(display_name)}"\s*$', text, flags=re.M)
        return m.group(1) if m else None

    def _find_internal_by_display_name(self, mim_path: str, display_name: str) -> str | None:
        data = self._read_json(mim_path)
        org = data.get("Organization") if isinstance(data, dict) else None
        if not isinstance(org, list):
            return None
        for it in org:
            if isinstance(it, dict) and it.get("Name") == display_name:
                return it.get("InstanceName")
        return None

    def _read_instance_adb_port(self, conf_path: str, name_or_internal: str) -> int | None:
        # Just dont touch
        text = self._read_text(conf_path)
        if not text:
            return None
        internal = name_or_internal
        if not re.search(rf"^bst\.instance\.{re.escape(internal)}\.", text, flags=re.M):
            m = re.search(
                rf'^bst\.instance\.([^.=]+)\.display_name="{re.escape(name_or_internal)}"\s*$', text, flags=re.M
            )
            if m:
                internal = m.group(1)
        for key in (f"bst.instance.{internal}.status.adb_port", f"bst.instance.{internal}.adb_port"):
            m = re.search(rf'^{re.escape(key)}="(\d{{3,5}})"\s*$', text, flags=re.M)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    def _normalize_renderer(self, desired: str | None) -> str:
        s = str(desired or "").strip().lower()
        alias = {
            "gl": "gl",
            "dx": "dx",
            "vlcn": "vlcn",
        }
        # Default to Vulkan on macOS (DirectX not available), DirectX on Windows
        default = "vlcn" if is_macos() else "dx"
        code = alias.get(s, default)
        if DEBUG:
            print(f"[Bluestacks 5] Renderer requested='{s}' normalized='{code}'")
        return code

    def _ensure_custom_resolution(self, conf_text: str, token: str = "418 x 633") -> str:
        existing = self._get_conf_value(conf_text, "bst.custom_resolutions")

        def norm(s: str) -> str:
            return s.lower().replace("Ã—", "x").replace(" ", "")  # Because of Unicode stuff

        want = norm(token)
        if not existing:
            return self._set_conf_value(conf_text, "bst.custom_resolutions", token)
        toks = [t.strip() for t in re.split(r"[;,]\s*|\n", existing) if t.strip()]
        if any(norm(t) == want for t in toks):
            return conf_text
        toks.append(token)
        return self._set_conf_value(conf_text, "bst.custom_resolutions", "; ".join(toks))

    def _display_name_exists(self, conf_path: str, display_name: str) -> bool:
        text = self._read_text(conf_path) or ""
        return (
            re.search(rf'^bst\.instance\.[^.=]+\.display_name="{re.escape(display_name)}"\s*$', text, flags=re.M)
            is not None
        )

    def _get_display_name(self, conf_text: str, internal: str) -> str:
        return self._get_conf_value(conf_text, f"bst.instance.{internal}.display_name") or internal

    def _pick_any_instance(self) -> str | None:
        conf = self._read_text(self.bs_conf_path)
        if not conf:
            return None
        instance_list = self._list_instance_internals(conf)
        if not instance_list:
            return None
        return instance_list[0]

    def _use_existing_internal(self, internal: str, rename: bool) -> bool:
        conf = self._read_text(self.bs_conf_path)
        if not conf:
            return False

        current_display = self._get_display_name(conf, internal)
        self.stop(display_name=current_display)

        conf, changed = self._compose_instance_conf(conf, internal, ensure_display=rename)
        if changed:
            with open(self.bs_conf_path, "w", encoding="utf-8") as f:
                f.write(conf)
        if rename:
            self._update_mim_name(self.mim_meta_path, internal, self.instance_name)
        else:
            self.instance_name = current_display

        self.internal_name = internal
        self.instance_port = self._read_instance_adb_port(self.bs_conf_path, internal)
        self.device_serial = f"127.0.0.1:{self.instance_port}" if self.instance_port else None

        self.logger.log(
            f"[BlueStacks 5] Using instance '{self.instance_name}' (internal '{internal}'). Port={self.instance_port}"
        )
        with suppress(Exception):
            self._close_multi_instance_manager()
        return True

    def _update_mim_name(self, mim_path: str, internal_name: str, ui_name: str) -> None:
        if not os.path.isfile(mim_path):
            return
        data = self._read_json(mim_path)
        data.setdefault("Organization", [])
        org = data["Organization"]
        for conf in org:
            if conf.get("InstanceName") == internal_name:
                conf["Name"] = ui_name  # rename here so the Instance Manager stops gaslighting us
                break
        with open(mim_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _close_multi_instance_manager(self) -> None:
        """Close BlueStacks Multi-Instance Manager if it's open after renaming to update instance name in it."""
        if is_macos():
            subprocess.run(["pkill", "-f", "BlueStacksMIM"], check=False)
        else:
            subprocess.run(
                'taskkill /IM "HD-MultiInstanceManager.exe" /F', shell=True, capture_output=True, text=True, check=False
            )

    def _open_multi_instance_manager(self) -> None:
        """Open BlueStacks Multi-Instance Manager."""
        if is_macos():
            mim_app = os.path.expanduser(
                os.getenv("PYCLASHBOT_BLUESTACKS_MIM_APP", "/Applications/BlueStacksMIM.app")
            )
            subprocess.Popen(["open", mim_app])
        else:
            with suppress(Exception):
                os.startfile(os.path.join(self.base_folder, "HD-MultiInstanceManager.exe"))

    def _reuse_and_rename_internal(self, internal: str) -> bool:
        conf = self._read_text(self.bs_conf_path)
        if not conf:
            return False

        # Stop just this instance
        current_display = self._get_conf_value(conf, f"bst.instance.{internal}.display_name") or internal
        self.stop(display_name=current_display)

        conf = self._read_text(self.bs_conf_path) or ""
        conf, changed = self._compose_instance_conf(conf, internal, ensure_display=True)
        if changed:
            with open(self.bs_conf_path, "w", encoding="utf-8") as f:
                f.write(conf)
        self._update_mim_name(self.mim_meta_path, internal, self.instance_name)

        # Cache internals
        self.internal_name = internal
        self.instance_port = self._read_instance_adb_port(self.bs_conf_path, internal)
        self.device_serial = f"127.0.0.1:{self.instance_port}" if self.instance_port else None

        self.logger.log(
            f"[BlueStacks 5] Reused instance '{internal}' as '{self.instance_name}'. Port={self.instance_port}"
        )
        with suppress(Exception):
            self._close_multi_instance_manager()
        return True

    def _pick_unlinked_instance(self) -> str | None:
        """Find an instance without Google account logins (clean instance)."""
        conf = self._read_text(self.bs_conf_path)
        if not conf:
            return None
        instance_list = self._list_instance_internals(conf)
        if not instance_list:
            return None
        ga_re = re.compile(r'^bst\.instance\.([^.=]+)\.google_account_logins="(.*)"\s*$', re.M)
        accounts = {m.group(1): m.group(2) for m in ga_re.finditer(conf)}
        for internal in instance_list:
            if accounts.get(internal, "").strip() == "":
                return internal
        return None

    def _ensure_managed_instance(self):
        # If display entry exists
        if self._display_name_exists(self.bs_conf_path, self.instance_name):
            self.internal_name = self._find_internal_by_display_name(
                self.mim_meta_path, self.instance_name
            ) or self._find_internal_in_conf_by_display(self.bs_conf_path, self.instance_name)
            if self.internal_name:
                self.instance_port = self._read_instance_adb_port(self.bs_conf_path, self.internal_name)
                if self.instance_port:
                    self.device_serial = f"127.0.0.1:{self.instance_port}"
            return

        if self._explicit_instance_name:
            self.logger.log(
                f"[BlueStacks 5] Instance '{self.instance_name}' not found. Falling back to any available instance."
            )

        # Try to reuse an existing clean instance (without Google login)
        internal = self._pick_unlinked_instance()
        if internal and self._use_existing_internal(internal, rename=False):
            return

        # Fall back to any existing instance (even if logged in).
        internal = self._pick_any_instance()
        if internal and self._use_existing_internal(internal, rename=False):
            return

        # Open Multi-Instance Manager and prompt user to create a new instance.
        while True:
            self._request_instance_retry = False
            self.logger.show_temporary_action(
                message="No BlueStacks instances found. Open BlueStacks Multi-Instance Manager and create an instance, then click Retry.",
                action_text="Retry",
                callback=self._request_instance_creation_retry,
            )
            self.logger.log("[BlueStacks 5] No instances found. Opening Multi-Instance Manager...")
            self._open_multi_instance_manager()

            # Wait for user action via Retry callback
            self.instance_creation_waiting = True
            while getattr(self, "instance_creation_waiting", False):
                interruptible_sleep(0.5)

            # User clicked Retry check again for a clean instance outside the callback
            if getattr(self, "_request_instance_retry", False):
                # If our display name exists, resolve internal and port
                if self._display_name_exists(self.bs_conf_path, self.instance_name):
                    internal = self._find_internal_by_display_name(
                        self.mim_meta_path, self.instance_name
                    ) or self._find_internal_in_conf_by_display(self.bs_conf_path, self.instance_name)
                    if internal:
                        self.internal_name = internal
                        self.instance_port = self._read_instance_adb_port(self.bs_conf_path, internal)
                        self.device_serial = f"127.0.0.1:{self.instance_port}" if self.instance_port else None
                        with suppress(Exception):
                            self._close_multi_instance_manager()
                        self.logger.change_status("Clean instance detected - continuing...")
                        return

                # Otherwise try any instance
                internal = self._pick_any_instance()
                if internal and self._use_existing_internal(internal, rename=False):
                    self.logger.change_status("Prepared existing instance - continuing...")
                    return

                self.logger.log("[BlueStacks 5] Still no instances found. Please try again.")

    def _request_instance_creation_retry(self):
        self.logger.log("[BlueStacks 5] Retry clicked - rechecking for clean instance")
        self._request_instance_retry = True
        self.instance_creation_waiting = False

    def _apply_renderer_setting(self) -> None:
        desired = self.render_settings["graphics_renderer"]
        desired_code = self._normalize_renderer(desired)
        if not self.bs_conf_path:
            return
        internal = self.internal_name or self._find_internal_in_conf_by_display(self.bs_conf_path, self.instance_name)
        if not internal:
            return
        conf = self._read_text(self.bs_conf_path) or ""
        current_display = self._get_conf_value(conf, f"bst.instance.{internal}.display_name") or internal
        self.stop(display_name=current_display)
        new_conf = self._set_conf_value(conf, f"bst.instance.{internal}.graphics_renderer", desired_code)
        if new_conf != conf:
            with suppress(Exception):
                with open(self.bs_conf_path, "w", encoding="utf-8") as f:
                    f.write(new_conf)
                self.logger.log(f"[BlueStacks 5] graphics_renderer set to {desired_code} for {internal}")

    def _valid_screen_size(self, expected_dims: tuple[int, int]) -> bool:
        expected_hw = (expected_dims[1], expected_dims[0])
        image = self.screenshot()
        dims = image.shape[:2]
        return dims == expected_hw

    def _set_screen_size(self, width: int, height: int) -> None:
        self.adb(f"shell wm size {width}x{height}")

    def _set_screen_density(self, dpi: int) -> None:
        self.adb(f"shell wm density {dpi}")

    def _compose_instance_conf(self, conf: str, internal: str, ensure_display: bool = True) -> tuple[str, bool]:
        """Return (new_conf, changed) with required BlueStacks settings applied, including renderer."""
        changed = False

        # Ensure global Bs5 ADB access is enabled
        if (self._get_conf_value(conf, "bst.enable_adb_access") or "0") != "1":
            conf = self._set_conf_value(conf, "bst.enable_adb_access", "1")
            changed = True

        # Ensure custom resolution entry exists
        new_conf = self._ensure_custom_resolution(conf, f"{self.expected_dims[0]} x {self.expected_dims[1]}")
        if new_conf != conf:
            conf = new_conf
            changed = True

        desired = {
            f"bst.instance.{internal}.cpus": "2",
            f"bst.instance.{internal}.ram": "2048",
            f"bst.instance.{internal}.fb_width": str(self.expected_dims[0]),
            f"bst.instance.{internal}.fb_height": str(self.expected_dims[1]),
            f"bst.instance.{internal}.dpi": str(self.DEFAULT_DPI),
            f"bst.instance.{internal}.max_fps": str(self.DEFAULT_MAX_FPS),
            f"bst.instance.{internal}.custom_resolution_selected": "1",
        }
        if ensure_display:
            desired[f"bst.instance.{internal}.display_name"] = self.instance_name

        for key, val in desired.items():
            cur = self._get_conf_value(conf, key)
            if cur != val:
                conf = self._set_conf_value(conf, key, val)
                changed = True

        # Apply renderer from render_settings
        desired = self.render_settings["graphics_renderer"]
        desired_code = self._normalize_renderer(desired)
        cur = self._get_conf_value(conf, f"bst.instance.{internal}.graphics_renderer")
        if cur != desired_code:
            conf = self._set_conf_value(conf, f"bst.instance.{internal}.graphics_renderer", desired_code)
            changed = True

        return conf, changed

    def _enforce_instance_config(self) -> None:
        """Re-check and fix drifted VM settings before each start."""
        if not self.bs_conf_path:
            return
        internal = self.internal_name or self._find_internal_in_conf_by_display(self.bs_conf_path, self.instance_name)
        if not internal:
            return

        conf = self._read_text(self.bs_conf_path) or ""
        conf, changed = self._compose_instance_conf(conf, internal, ensure_display=True)

        if changed:
            with suppress(Exception):
                with open(self.bs_conf_path, "w", encoding="utf-8") as f:
                    f.write(conf)
                self.logger.log(f"[BlueStacks 5] Enforced VM config for '{internal}'")

    def _cmd_is_server_scoped(self, command: str) -> bool:
        c = command.strip()
        if c.startswith("-s "):
            return True
        first = c.split()[0] if c else ""
        return first in {"connect", "disconnect", "devices", "start-server", "kill-server", "version", "help", "keys"}

    def adb(self, command: str, binary_output: bool = False) -> subprocess.CompletedProcess:
        """
        Run an adb command via our private server. Device-scoped by default unless server-scoped.
        This is the abstract method implementation for AdbBasedController.
        """
        base = f'"{self.adb_path}" -P {self.adb_server_port} '
        if not self._cmd_is_server_scoped(command) and self.device_serial:
            base += f"-s {self.device_serial} "
        full = base + command
        if DEBUG:
            print(f"[Bluestacks 5/ADB] {full}")
        return subprocess.run(
            full, shell=True, capture_output=True, text=not binary_output, check=False, env=self.adb_env
        )

    def _check_app_installed(self, package_name: str) -> bool:
        """
        Check if app is installed via ADB.
        This is the abstract method implementation for AdbBasedController.
        """
        # BlueStacks can report empty package lists right after boot.
        # Retry a few times and check boot completion before declaring missing.
        for attempt in range(6):
            res = self.adb("shell pm list packages")
            out = (res.stdout or "").strip()
            if out and package_name in out:
                return True
            # Fallback: direct package path check
            res2 = self.adb(f"shell pm path {package_name}")
            out2 = (res2.stdout or "").strip()
            if out2 and package_name in out2:
                return True
            # Check if Android has fully booted
            boot = self.adb("shell getprop sys.boot_completed")
            booted = (boot.stdout or "").strip() == "1"
            if self.logger:
                status = "booted" if booted else "booting"
                self.logger.change_status(f"Checking app install ({status})... attempt {attempt + 1}/6")
            if attempt < 5:
                interruptible_sleep(1.0)
        return False

    def adb_server(self, command: str) -> subprocess.CompletedProcess:
        full = f'"{self.adb_path}" -P {self.adb_server_port} {command}'
        if DEBUG:
            print(f"[Bluestacks 5/ADB-SRV] {full}")
        return subprocess.run(full, shell=True, capture_output=True, text=True, check=False, env=self.adb_env)

    def _reset_adb_server(self) -> None:
        with suppress(Exception):
            self.adb_server("kill-server")  # Cause ADB loves to randomly fuck around

    def _refresh_instance_port(self):
        """Re-read the instance port and update device_serial."""
        if not self.internal_name:
            return
        new_port = self._read_instance_adb_port(self.bs_conf_path, self.internal_name)
        if new_port and new_port != self.instance_port:
            self.instance_port = new_port
            self.device_serial = f"127.0.0.1:{self.instance_port}"
            if DEBUG:
                print(f"[Bluestacks 5] Refreshed instance port -> {self.device_serial}")

    def _connect(self) -> bool:
        """Connect only to this instance's port."""
        if not self.device_serial:
            return False
        self._reset_adb_server()
        self.adb_server(f"disconnect {self.device_serial}")
        interruptible_sleep(0.2)
        self.adb_server(f"connect {self.device_serial}")
        state = self.adb("get-state")
        ok = (state.returncode == 0) and (state.stdout and "device" in state.stdout)
        if not ok:
            self._refresh_instance_port()
            if self.device_serial:
                self.adb_server(f"disconnect {self.device_serial}")
                interruptible_sleep(0.2)
                self.adb_server(f"connect {self.device_serial}")
                state = self.adb("get-state")
                ok = (state.returncode == 0) and (state.stdout and "device" in state.stdout)
        return ok  # False means ADB is in a mood again

    def _is_this_instance_running(self) -> bool:
        if is_macos():
            # Check for BlueStacks process with instance argument
            try:
                res = subprocess.run(
                    ["pgrep", "-fl", "BlueStacks"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                # Parse output for our instance
                return self.internal_name is not None and self.internal_name in (res.stdout or "")
            except Exception:
                return False

        # Windows: tasklist CSV parsing
        title = self.instance_name
        try:
            res = subprocess.run(
                'tasklist /v /fi "IMAGENAME eq HD-Player.exe" /fo csv',
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0 or not res.stdout:
                return False
            reader = csv.reader(io.StringIO(res.stdout))
            next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                image = (row[0] if len(row) > 0 else "").strip().lower()
                window_title = (row[-1] if len(row) > 0 else "").strip()
                if image == "hd-player.exe" and window_title == title:
                    return True  # yes, I parsed CSV from tasklist. no, I'm not proud of it
        except Exception:
            return False
        return False

    def start(self):
        """Start only this instance. Enforce config just before launching."""
        with suppress(Exception):
            self._enforce_instance_config()

        if is_macos():
            # Launch via direct execution
            args = [self.emulator_executable_path]
            if self.internal_name:
                args.extend(["--instance", self.internal_name])
            subprocess.Popen(args)
        else:
            # Windows
            args = []
            if self.internal_name:
                args = ["--instance", self.internal_name]
            cmd = '"' + self.emulator_executable_path + '"' + (" " + " ".join(args) if args else "")
            subprocess.Popen(cmd, shell=True)
        interruptible_sleep(5)

    def stop(self, display_name: str | None = None):
        """Stop only this instance."""
        if is_macos():
            # Kill BlueStacks processes matching our instance
            if self.internal_name:
                subprocess.run(["pkill", "-f", f"BlueStacks.*{self.internal_name}"], check=False)
            else:
                subprocess.run(["pkill", "-f", "BlueStacks"], check=False)
        else:
            # Windows: Stop using window title match
            title = display_name or self.instance_name
            subprocess.run(
                f'taskkill /fi "WINDOWTITLE eq {title}" /IM "HD-Player.exe" /F',
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )

    def restart(self) -> bool:
        start_ts = time.time()
        self.logger.change_status("Starting BlueStacks 5 emulator restart process...")

        self.logger.change_status("Stopping towerlogic BlueStacks 5 instance...")
        self.stop()

        self.logger.change_status(f"Launching BlueStacks 5 ({self.instance_name})...")
        self.start()

        # Wait for only our instance
        boot_timeout = 180
        t0 = time.time()
        while not self._is_this_instance_running():
            if time.time() - t0 > boot_timeout:
                self.logger.change_status("Timeout waiting for towerlogic instance to start - retrying...")
                return False
            interruptible_sleep(0.5)

        # Refresh port after boot
        self._refresh_instance_port()
        if not self.device_serial:
            self.logger.change_status("No ADB port resolved for this instance.")
            return False

        # Connect ADB scoped to our device
        self.logger.change_status(f"Connecting ADB to {self.device_serial} ...")
        t1 = time.time()
        while not self._connect():
            if time.time() - t1 > 60:
                self.logger.change_status("Failed to connect ADB to BlueStacks 5 - retrying...")
                return False  # if this makes issues big problems
            interruptible_sleep(1)

        # Enforce screen size/density to match detection expectations.
        self.logger.change_status(f"Setting emulator screen size to {self.expected_dims}...")
        for _ in range(3):
            self._set_screen_size(*self.expected_dims)
            self._set_screen_density(self.DEFAULT_DPI)
            interruptible_sleep(0.5)
        if not self._valid_screen_size(self.expected_dims):
            self.logger.change_status(
                f"Warning: screen size mismatch (expected {self.expected_dims}). "
                "Card detection may be unreliable."
            )

        # Launch Clash Royale
        clash_pkg = "com.supercell.clashroyale"
        self.logger.change_status("Launching Clash Royale...")

        # Check installation without launching through monkey.  BlueStacks Air
        # can leave a black surface when the app is launched twice in quick
        # succession, so use one canonical activity launch below.
        if not self._check_app_installed(clash_pkg):
            # This means app is not installed and user is being prompted.
            # We must wait for the installation loop (handled by base class) to finish
            # The base class's _wait_for_clash_installation will block until
            # the user clicks 'Retry' and the app is found.
            self.logger.log("Waiting for app installation...")

            # After _wait_for_clash_installation returns True, we need to manually
            # re-trigger the app start, because the original call failed.
            if not self._wait_for_clash_installation(clash_pkg):
                return False

        self._force_launch_clash(clash_pkg, retries=1)
        launch_started = time.time()

        # Wait for main menu
        self.logger.change_status("Waiting for Clash Royale main menu...")
        deadline = time.time() + 240
        while time.time() < deadline:
            foreground = self._get_foreground_package()
            process_running = self._is_clash_process_running(clash_pkg)
            screenshot_available = False
            rendered = False
            main_menu = False
            detector_used = "none"
            current_details = {}
            menu_pixels, menu_matches = [], []
            readiness_image = None
            try:
                readiness_image = self.screenshot()
                screenshot_available = readiness_image is not None
                rendered = bool(screenshot_available and readiness_image.size and int(readiness_image.max()) > 8)
                if screenshot_available:
                    legacy_menu, menu_pixels, menu_matches = inspect_clash_main_menu(readiness_image)
                    current_menu, current_details = detect_current_clash_main_menu(readiness_image)
                    main_menu = current_menu or legacy_menu
                    detector_used = "current_ui" if current_menu else ("legacy_pixels" if legacy_menu else "none")
            except Exception as exc:
                self.logger.log(f"Startup readiness screenshot failed: {exc}")

            battle_detected = check_if_in_battle(self, image=readiness_image) if screenshot_available else False
            trophy_box_detected = (
                check_for_trophy_box_reward(self, image=readiness_image)
                if rendered else False
            )
            self.logger.log(f"process_exists={process_running}")
            self.logger.log(f"foreground_detected={foreground == clash_pkg}")
            self.logger.log(f"foreground_package={foreground}")
            self.logger.log(f"rendered_frame_valid={rendered}")
            self.logger.log(f"screenshot_available={screenshot_available}")
            self.logger.log(f"main_menu_recognized={main_menu}")
            self.logger.log(f"main_menu_detector={detector_used}")
            self.logger.log(f"main_menu_match_scores={current_details}")
            self.logger.log(f"battle_detected={battle_detected}")
            self.logger.log(f"trophy_box_detected={trophy_box_detected}")
            self.logger.log(f"launch_age={time.time() - launch_started:.1f}s")
            self.logger.log(f"startup_timeout_remaining={max(0.0, deadline - time.time()):.1f}s")
            if screenshot_available and not main_menu:
                self.logger.log(f"main_menu_pixels={menu_pixels}")
                self.logger.log(f"main_menu_pixel_matches={menu_matches}")

            if main_menu:
                self.logger.change_status("Clash Royale main menu detected")
                dur = f"{time.time() - start_ts:.1f}s"
                self.logger.log(f"BlueStacks 5 restart completed in {dur}")
                return True
            if trophy_box_detected:
                self.logger.log("Trophy box detected during startup; attempting recovery...")
                recovery_state = clear_trophy_box_reward(self, self.logger)
                self.logger.log(f"Startup trophy-box recovery result: {recovery_state}")

                # Re-evaluate a fresh frame immediately; the reward flow may
                # end at the main menu, a battle, or another reward screen.
                try:
                    fresh_image = self.screenshot()
                    fresh_legacy = inspect_clash_main_menu(fresh_image)[0]
                    fresh_current = detect_current_clash_main_menu(fresh_image)[0]
                    fresh_main_menu = fresh_legacy or fresh_current
                    fresh_battle = check_if_in_battle(self, image=fresh_image)
                    fresh_trophy_box = check_for_trophy_box_reward(self, image=fresh_image)
                    self.logger.log(f"post_recovery_main_menu={fresh_main_menu}")
                    self.logger.log(f"post_recovery_battle={fresh_battle}")
                    self.logger.log(f"post_recovery_trophy_box={fresh_trophy_box}")
                    if fresh_main_menu:
                        self.logger.change_status("Clash Royale main menu detected")
                        dur = f"{time.time() - start_ts:.1f}s"
                        self.logger.log(f"BlueStacks 5 restart completed in {dur}")
                        return True
                    if fresh_battle:
                        self.logger.change_status("Battle detected after trophy-box recovery")
                except Exception as exc:
                    self.logger.log(f"Post-recovery state check failed: {exc}")
                continue
            if battle_detected:
                self.logger.change_status("Battle detected; waiting for it to finish before relaunching...")
                interruptible_sleep(3)
                continue
            if foreground == clash_pkg or process_running:
                # A foreground or still-starting process must be allowed to
                # finish initialization; recognition failure is not a crash.
                self.logger.change_status("Waiting for game initialization...")
                interruptible_sleep(1.5)
                continue
            if time.time() - launch_started > 10.0:
                self.logger.log("Relaunch required because: Clash Royale is neither foreground nor running")
                self._force_launch_clash(clash_pkg, retries=1)
                launch_started = time.time()
            interruptible_sleep(1.0)

        self.logger.change_status("Timeout waiting for Clash main menu - retrying...")
        return False

    def _force_launch_clash(self, package_name: str, retries: int = 1, delay: float = 2.0) -> bool:
        """Start Clash Royale once using its canonical launcher activity."""
        activity = self._resolve_launch_activity(package_name) or f"{package_name}/.GameApp"
        for attempt in range(max(1, retries)):
            self.logger.change_status(f"Force launching Clash Royale (attempt {attempt + 1}/{retries})...")
            result = self.adb(
                f"shell am start -W -a android.intent.action.MAIN "
                f"-c android.intent.category.LAUNCHER -n {activity}"
            )
            interruptible_sleep(delay)
            fg = self._get_foreground_package()
            ok = result.returncode == 0 and fg == package_name
            self.logger.log(f"Clash Royale foreground: {ok} ({fg}) after launcher intent")
            if ok:
                return True
        return False

    def _is_clash_process_running(self, package_name: str) -> bool:
        result = self.adb(f"shell pidof {package_name}")
        return bool((result.stdout or "").strip())

    def _has_rendered_frame(self) -> bool:
        try:
            image = self.screenshot()
            return bool(image.size and int(image.max()) > 8)
        except Exception:
            return False

    def _force_stop_store_shell(self) -> None:
        """Best-effort stop of BlueStacks store shell apps that can steal focus."""
        candidates = [
            "com.bluestacks.store",
            "com.bluestacks.appstore",
            "com.bluestacks.bststore",
            "com.bluestacks.bstlauncher",
            "com.bluestacks.systemapp",
            "com.bluestacks.appplayer",
            "com.bluestacks.x",
            "com.bluestacks.cloudgame",
            "com.bluestacks.helper",
        ]
        for pkg in candidates:
            self.adb(f"shell am force-stop {pkg}")

    def _resolve_launch_activity(self, package_name: str) -> str | None:
        """Resolve the preferred launch activity for a package."""
        result = self.adb(f"shell cmd package resolve-activity --brief {package_name}")
        text = (result.stdout or "").strip() if hasattr(result, "stdout") else ""
        if not text:
            return None
        # The last non-empty line often contains the fully-qualified activity.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        candidate = lines[-1]
        if "/" in candidate:
            return candidate
        # Some Android versions format as "name=package/.Activity"
        if "name=" in candidate:
            parts = candidate.split("name=")
            if len(parts) > 1 and "/" in parts[-1]:
                return parts[-1].strip()
        return None

    def _get_foreground_package(self) -> str | None:
        """Return the foreground package using modern and legacy Android sources."""
        component = self._get_foreground_component()
        return component[0] if component else None

    def _get_foreground_component(self) -> tuple[str, str] | None:
        """Return ``(package, activity)`` from Android foreground-state output."""
        sources = (
            "shell dumpsys activity activities",
            "shell dumpsys activity top",
            "shell dumpsys window windows",
            "shell dumpsys window",
            "shell dumpsys input",
        )
        markers = (
            "topResumedActivity",
            "mResumedActivity",
            "ResumedActivity",
            "topActivity",
            "mCurrentFocus",
            "mFocusedApp",
        )
        for command in sources:
            result = self.adb(command)
            text = (result.stdout or "") if hasattr(result, "stdout") else ""
            for line in text.splitlines():
                if not any(marker in line for marker in markers):
                    continue
                # Handles ActivityRecord/Window forms and optional name= prefixes.
                match = re.search(r"([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_$]+)+)/([A-Za-z0-9_.$]+)", line)
                if match:
                    return match.group(1), match.group(2)
        logger = getattr(self, "logger", None)
        if logger is not None and hasattr(logger, "log"):
            logger.log("Unable to resolve foreground activity from Android activity/window/input dumpsys output")
        return None

    # click() is now inherited from AdbBasedController
    # swipe() is now inherited from AdbBasedController
    # screenshot() is now inherited from AdbBasedController
    # start_app() is now inherited from AdbBasedController
    # _wait_for_clash_installation() is now inherited from AdbBasedController
    # _retry_installation_check() is now inherited from AdbBasedController


if __name__ == "__main__":
    pass
