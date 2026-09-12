from __future__ import annotations

import os
import sys
from pathlib import Path
import math
import tkinter as tk
from contextlib import suppress
from collections.abc import Callable
from tkinter import messagebox
from typing import TYPE_CHECKING

import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, LEFT, READONLY, YES, X
from ttkbootstrap.tooltip import ToolTip

from towerlogic.emulators import EmulatorType, get_available_emulators
from towerlogic.interface.config import (
    BLUESTACKS_SETTINGS,
    GOOGLE_PLAY_SETTINGS,
    JOBS,
    MEMU_SETTINGS,
    ComboConfig,
)
from towerlogic.interface.enums import (
    BATTLE_STAT_FIELDS,
    BATTLE_STAT_LABELS,
    BOT_STAT_FIELDS,
    BOT_STAT_LABELS,
    COLLECTION_STAT_FIELDS,
    COLLECTION_STAT_LABELS,
    BotStatField,
    DerivedStatField,
    StatField,
    UIField,
)

from towerlogic.utils.logger import log_dir

if TYPE_CHECKING:
    from collections.abc import Callable


def no_jobs_popup() -> None:
    messagebox.showerror("Critical Error!", "You must select at least one job!")


class TowerLogicUI(ttk.Window):
    DEFAULT_THEME = "litera"

    def __init__(self) -> None:
        super().__init__(themename=self.DEFAULT_THEME)
        self.title("TowerLogic")
        self.geometry("700x760")
        self.resizable(False, False)
        self.minsize(520, 600)

        self._style = ttk.Style()
        current_theme = self._style.theme_use()
        if not current_theme:
            current_theme = self.DEFAULT_THEME
        self.theme_var = ttk.StringVar(value=current_theme)
        self._theme_retry_pending = False
        self.advanced_settings_var = ttk.BooleanVar(value=False)
        self.bs_instance_var = ttk.StringVar(value="")
        from towerlogic.utils.resources import default_policy_path

        self._default_model_path = str(default_policy_path())
        self.policy_toggle_var = ttk.BooleanVar(value=False)
        self.policy_epsilon_var = ttk.StringVar(value="0.10")
        self.policy_model_var = ttk.StringVar(value=self._default_model_path)
        self.policy_train_var = ttk.BooleanVar(value=False)
        self.policy_lr_var = ttk.StringVar(value="0.0005")
        self.policy_free_place_var = ttk.BooleanVar(value=False)
        self.policy_min_elixir_var = ttk.StringVar(value="0")
        self.policy_allow_hold_var = ttk.BooleanVar(value=True)
        self.policy_debug_detect_var = ttk.BooleanVar(value=False)
        self.policy_debug_overlay_var = ttk.BooleanVar(value=False)
        self.policy_debug_hp_log_var = ttk.BooleanVar(value=False)
        self.policy_debug_hand_conf_var = ttk.BooleanVar(value=False)
        self.policy_strict_elixir_var = ttk.BooleanVar(value=False)
        self.policy_spell_gate_var = ttk.BooleanVar(value=False)
        self._config_callback: Callable[[dict[str, object]], None] | None = None
        self._open_logs_callback: Callable[[], None] | None = None
        self._config_widgets: dict[str, tk.Widget] = {}
        self._theme_labels: list[tk.Widget] = []
        self._traces: list[tuple[tk.Variable, str]] = []
        self._suspend_traces = 0
        self._log_widgets: list[tk.Text] = []
        self._log_lock_scroll = False
        self.log_dump_path_var = ttk.StringVar(value=os.path.join(log_dir, "log_dump.txt"))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Tabs get more space
        self.rowconfigure(1, weight=0, minsize=60)  # Bottom row: button only

        self._set_app_icon()
        self._build_tabs()
        self._build_bottom_row()
        self._refresh_theme_colours()
        self.update_idletasks()
        self.deiconify()
        try:
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass
        self.after(100, self._force_show_window)

    def _force_show_window(self) -> None:
        try:
            self.attributes("-topmost", True)
            self.attributes("-topmost", False)
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass

    def _set_app_icon(self) -> None:
        assets = Path(__file__).resolve().parent / "assets"
        self._icon_image = tk.PhotoImage(file=str(assets / "towerlogic.png"))
        self._icon_header_image = self._icon_image.subsample(4, 4)
        self.iconphoto(True, self._icon_image)
        if sys.platform == "win32":
            self.iconbitmap(str(assets / "towerlogic.ico"))

    def register_config_callback(self, callback: Callable[[dict[str, object]], None]) -> None:
        self._config_callback = callback

    def register_open_logs_callback(self, callback: Callable[[], None]) -> None:
        self._open_logs_callback = callback

    def get_all_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for field, var in self.jobs_vars.items():
            values[field.value] = bool(var.get())

        emulator_choice = self.emulator_var.get()
        values[UIField.MEMU_EMULATOR_TOGGLE.value] = emulator_choice == EmulatorType.MEMU
        values[UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value] = emulator_choice == EmulatorType.GOOGLE_PLAY
        values[UIField.BLUESTACKS_EMULATOR_TOGGLE.value] = emulator_choice == EmulatorType.BLUESTACKS
        values[UIField.ADB_TOGGLE.value] = emulator_choice == EmulatorType.ADB

        memu_render = self.memu_render_var.get()
        values[UIField.DIRECTX_TOGGLE.value] = memu_render == "DirectX"
        values[UIField.OPENGL_TOGGLE.value] = memu_render == "OpenGL"

        bs_render = self.bs_render_var.get()
        values[UIField.BS_RENDERER_DX.value] = bs_render == "DirectX"
        values[UIField.BS_RENDERER_GL.value] = bs_render == "OpenGL"
        values[UIField.BS_RENDERER_VK.value] = bs_render == "Vulkan"

        for field, var in self.gp_vars.items():
            values[field.value] = var.get()

        values[UIField.ADB_SERIAL.value] = self.adb_serial_var.get()
        values[UIField.BS_INSTANCE_NAME.value] = self.bs_instance_var.get().strip()
        values[UIField.POLICY_TOGGLE.value] = bool(self.policy_toggle_var.get())
        values[UIField.POLICY_EPSILON.value] = self._safe_float(self.policy_epsilon_var.get(), fallback=0.10)
        model_path = self.policy_model_var.get().strip() or self._default_model_path
        values[UIField.POLICY_MODEL_PATH.value] = model_path
        values[UIField.POLICY_TRAIN_TOGGLE.value] = bool(self.policy_train_var.get())
        values[UIField.POLICY_LR.value] = self._safe_float(self.policy_lr_var.get(), fallback=0.0005)
        values[UIField.POLICY_FREE_PLACEMENT.value] = bool(self.policy_free_place_var.get())
        values[UIField.POLICY_MIN_ELIXIR.value] = self._safe_int(self.policy_min_elixir_var.get(), fallback=0)
        values[UIField.POLICY_ALLOW_HOLD.value] = bool(self.policy_allow_hold_var.get())
        values[UIField.POLICY_DEBUG_DETECTION.value] = bool(self.policy_debug_detect_var.get())
        values[UIField.POLICY_DEBUG_OVERLAY.value] = bool(self.policy_debug_overlay_var.get())
        values[UIField.POLICY_DEBUG_HP_LOG.value] = bool(self.policy_debug_hp_log_var.get())
        values[UIField.POLICY_DEBUG_HAND_CONF.value] = bool(self.policy_debug_hand_conf_var.get())
        values[UIField.POLICY_STRICT_ELIXIR.value] = bool(self.policy_strict_elixir_var.get())
        values[UIField.POLICY_SPELL_GATE.value] = bool(self.policy_spell_gate_var.get())

        values[UIField.THEME_NAME.value] = self.theme_var.get() or self.DEFAULT_THEME
        return values

    def set_all_values(self, values: dict[str, object]) -> None:
        theme_value: str | None = None
        self._suspend_traces += 1
        try:
            for field, var in self.jobs_vars.items():
                if field.value in values:
                    var.set(bool(values[field.value]))

            if UIField.THEME_NAME.value in values:
                theme_value = str(values[UIField.THEME_NAME.value])

            # Determine saved emulator choice; default to current selection if no data provided
            saved_emulator = self.emulator_var.get()
            emulator_keys = {
                UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value,
                UIField.BLUESTACKS_EMULATOR_TOGGLE.value,
                UIField.ADB_TOGGLE.value,
                UIField.MEMU_EMULATOR_TOGGLE.value,
            }
            if emulator_keys & values.keys():
                if values.get(UIField.GOOGLE_PLAY_EMULATOR_TOGGLE.value):
                    saved_emulator = EmulatorType.GOOGLE_PLAY
                elif values.get(UIField.BLUESTACKS_EMULATOR_TOGGLE.value):
                    saved_emulator = EmulatorType.BLUESTACKS
                elif values.get(UIField.ADB_TOGGLE.value):
                    saved_emulator = EmulatorType.ADB
                elif values.get(UIField.MEMU_EMULATOR_TOGGLE.value):
                    saved_emulator = EmulatorType.MEMU
            # Use saved choice if available on this platform, otherwise fallback
            available = get_available_emulators()
            if saved_emulator in available:
                self.emulator_var.set(saved_emulator)
            elif available:
                self.emulator_var.set(available[0])

            if values.get(UIField.DIRECTX_TOGGLE.value):
                self.memu_render_var.set("DirectX")
            elif values.get(UIField.OPENGL_TOGGLE.value):
                self.memu_render_var.set("OpenGL")

            if values.get(UIField.BS_RENDERER_VK.value):
                self.bs_render_var.set("Vulkan")
            elif values.get(UIField.BS_RENDERER_DX.value):
                self.bs_render_var.set("DirectX")
            elif values.get(UIField.BS_RENDERER_GL.value):
                self.bs_render_var.set("OpenGL")

            if UIField.BS_INSTANCE_NAME.value in values:
                self.bs_instance_var.set(str(values[UIField.BS_INSTANCE_NAME.value]))

            if UIField.POLICY_TOGGLE.value in values:
                self.policy_toggle_var.set(bool(values[UIField.POLICY_TOGGLE.value]))
            if UIField.POLICY_EPSILON.value in values:
                self.policy_epsilon_var.set(str(values[UIField.POLICY_EPSILON.value]))
            if UIField.POLICY_MODEL_PATH.value in values:
                raw_path = str(values[UIField.POLICY_MODEL_PATH.value] or "")
                self.policy_model_var.set(raw_path or self._default_model_path)
            if UIField.POLICY_TRAIN_TOGGLE.value in values:
                self.policy_train_var.set(bool(values[UIField.POLICY_TRAIN_TOGGLE.value]))
            if UIField.POLICY_LR.value in values:
                self.policy_lr_var.set(str(values[UIField.POLICY_LR.value]))
            if UIField.POLICY_FREE_PLACEMENT.value in values:
                self.policy_free_place_var.set(bool(values[UIField.POLICY_FREE_PLACEMENT.value]))
            if UIField.POLICY_MIN_ELIXIR.value in values:
                self.policy_min_elixir_var.set(str(values[UIField.POLICY_MIN_ELIXIR.value]))
            if UIField.POLICY_ALLOW_HOLD.value in values:
                self.policy_allow_hold_var.set(bool(values[UIField.POLICY_ALLOW_HOLD.value]))
            if UIField.POLICY_DEBUG_DETECTION.value in values:
                self.policy_debug_detect_var.set(bool(values[UIField.POLICY_DEBUG_DETECTION.value]))
            if UIField.POLICY_DEBUG_OVERLAY.value in values:
                self.policy_debug_overlay_var.set(bool(values[UIField.POLICY_DEBUG_OVERLAY.value]))
            if UIField.POLICY_DEBUG_HP_LOG.value in values:
                self.policy_debug_hp_log_var.set(bool(values[UIField.POLICY_DEBUG_HP_LOG.value]))
            if UIField.POLICY_DEBUG_HAND_CONF.value in values:
                self.policy_debug_hand_conf_var.set(bool(values[UIField.POLICY_DEBUG_HAND_CONF.value]))
            if UIField.POLICY_STRICT_ELIXIR.value in values:
                self.policy_strict_elixir_var.set(bool(values[UIField.POLICY_STRICT_ELIXIR.value]))
            if UIField.POLICY_SPELL_GATE.value in values:
                self.policy_spell_gate_var.set(bool(values[UIField.POLICY_SPELL_GATE.value]))

            for field, var in self.gp_vars.items():
                config = next((c for c in GOOGLE_PLAY_SETTINGS if c.key == field), None)
                if field.value in values and values[field.value] is not None:
                    var.set(str(values[field.value]))
                elif config:
                    var.set(str(config.default))

            if UIField.ADB_SERIAL.value in values:
                self.adb_serial_var.set(str(values[UIField.ADB_SERIAL.value]))

            self._update_google_play_comboboxes()

        finally:
            self._suspend_traces -= 1

        if theme_value is not None:
            self._apply_theme(theme_value)


        self._show_current_emulator_settings()

    def set_button_state(self, state: str) -> None:
        """Set the main button state: 'idle', 'running', or 'stopping'."""
        self._button_state = state
        if state == "idle":
            self.main_btn.configure(text="Start", bootstyle="success", state=tk.NORMAL)
        elif state == "running":
            self.main_btn.configure(text="Stop", bootstyle="danger", state=tk.NORMAL)
        elif state == "stopping":
            self.main_btn.configure(text="Stop", bootstyle="danger", state=tk.NORMAL)

        # Disable/enable config widgets based on running state
        running = state in ("running", "stopping")
        for key, widget in self._config_widgets.items():
            if key == "main_btn":
                continue
            try:
                if isinstance(widget, ttk.Combobox):
                    if key == "emulator_combobox":
                        widget.configure(state=tk.DISABLED if running else READONLY)
                    elif widget is self.adb_serial_combo:
                        widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                    else:
                        widget.configure(state=tk.DISABLED if running else READONLY)
                elif isinstance(widget, ttk.Spinbox):
                    widget.configure(state=tk.DISABLED if running else READONLY)
                elif isinstance(widget, ttk.Radiobutton) and key in [
                    UIField.DIRECTX_TOGGLE.value,
                    UIField.OPENGL_TOGGLE.value,
                    UIField.BS_RENDERER_DX.value,
                    UIField.BS_RENDERER_GL.value,
                    UIField.BS_RENDERER_VK.value,
                ]:
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif widget in [
                    self.adb_connect_btn,
                    self.adb_refresh_btn,
                    self.adb_restart_btn,
                    self.adb_set_size_btn,
                    self.adb_reset_size_btn,
                ]:
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif isinstance(widget, ttk.Checkbutton):
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
                elif isinstance(widget, ttk.Button):
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)

            except tk.TclError:
                continue
        if running:
            self._hide_action_button()

    def get_button_state(self) -> str:
        """Get the current button state: 'idle', 'running', or 'stopping'."""
        return self._button_state

    def show_action_button(self, text: str, callback: Callable[[], None]) -> None:
        self._action_callback = callback
        self.action_btn.configure(text=text)
        self.main_btn.grid_remove()
        self.action_btn.grid()

    def hide_action_button(self) -> None:
        self._hide_action_button()

    def append_log(self, message: str) -> None:
        if not self._log_widgets:
            return
        for widget in self._log_widgets:
            at_end = widget.yview()[1] >= 0.98
            anchor = widget.yview()[0]
            widget.delete("1.0", "end")
            widget.insert("end", message)
            if at_end and not self._log_lock_scroll:
                widget.see("end")
            else:
                widget.yview_moveto(anchor)

    def _dump_log_to_file(self) -> None:
        path = self.log_dump_path_var.get().strip()
        if not path:
            messagebox.showwarning("Dump Log", "Please provide a dump file path.")
            return
        log_text = ""
        if self._log_widgets:
            log_text = self._log_widgets[0].get("1.0", "end-1c")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(log_text)
            messagebox.showinfo("Dump Log", f"Saved log to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Dump Log", f"Failed to write log:\n{exc}")

    def _open_log_dump_folder(self) -> None:
        path = self.log_dump_path_var.get().strip()
        if not path:
            messagebox.showwarning("Open Folder", "Please provide a dump file path.")
            return
        folder = os.path.dirname(path)
        if not folder:
            messagebox.showwarning("Open Folder", "Unable to determine folder from path.")
            return
        with suppress(Exception):
            os.makedirs(folder, exist_ok=True)
        try:
            import subprocess
            subprocess.Popen(["open", folder])
        except Exception as exc:
            messagebox.showerror("Open Folder", f"Failed to open folder:\n{exc}")

    def set_status(self, text: str) -> None:
        self._status_text = text

    def update_stats(self, stats: dict[str, object] | None) -> None:
        if not stats:
            return

        def as_string(field: StatField, default: str = "0") -> str:
            value = stats.get(field.value, default)
            return str(value)

        def as_int(field: StatField) -> int:
            value = stats.get(field.value)
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        for field, var in self.stat_labels.items():
            var.set(as_string(field))

        runtime = stats.get(BotStatField.TIME_SINCE_START.value)
        if runtime is not None:
            self.bot_labels[BotStatField.TIME_SINCE_START].set(str(runtime))
        failures = stats.get(BotStatField.RESTARTS_AFTER_FAILURE.value)
        if failures is not None:
            self.bot_labels[BotStatField.RESTARTS_AFTER_FAILURE].set(str(failures))

        winrate_raw = stats.get(DerivedStatField.WINRATE.value)
        wins = as_int(StatField.WINS)
        losses = as_int(StatField.LOSSES)
        parsed_winrate = self._parse_winrate_value(winrate_raw)
        winrate = parsed_winrate if parsed_winrate is not None else self._calculate_winrate_percentage(wins, losses)
        self.win_bar_var.set(winrate)
        self.win_bar_label.configure(text=f"{winrate:.1f}%")

        # Update win streak stats
        current_streak = stats.get(DerivedStatField.CURRENT_WIN_STREAK.value, 0)
        best_streak = stats.get(DerivedStatField.BEST_WIN_STREAK.value, 0)
        if hasattr(self, "current_streak_var"):
            self.current_streak_var.set(str(current_streak))
        if hasattr(self, "best_streak_var"):
            self.best_streak_var.set(str(best_streak))

        # Policy progress graph
        if hasattr(self, "policy_canvas"):
            progress = stats.get("policy_progress")
            step_progress = stats.get("policy_step_progress")
            if isinstance(progress, list):
                if isinstance(step_progress, list):
                    self._update_policy_graph(progress, step_progress)
                else:
                    self._update_policy_graph(progress, None)
            last_reward = stats.get("policy_last_reward")
            last_loss = stats.get("policy_last_loss")
            games = stats.get("policy_games")
            if hasattr(self, "policy_last_reward_var") and last_reward is not None:
                try:
                    self.policy_last_reward_var.set(f"{float(last_reward):.3f}")
                except Exception:
                    self.policy_last_reward_var.set(str(last_reward))
            if hasattr(self, "policy_last_loss_var") and last_loss is not None:
                try:
                    self.policy_last_loss_var.set(f"{float(last_loss):.3f}")
                except Exception:
                    self.policy_last_loss_var.set(str(last_loss))
            if hasattr(self, "policy_games_var") and games is not None:
                self.policy_games_var.set(str(games))

    def _update_policy_graph(self, values: list[float], step_values: list[float] | None) -> None:
        def finite_series(series: list[float] | None) -> list[float]:
            result: list[float] = []
            for value in series or []:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    result.append(number)
            return result

        values = finite_series(values)
        step_values = finite_series(step_values)
        if not values and not step_values:
            self.policy_canvas.delete("all")
            return
        width = max(int(self.policy_canvas.winfo_width()), 1)
        height = max(int(self.policy_canvas.winfo_height()), 1)
        if width <= 1 or height <= 1:
            return
        self.policy_canvas.delete("all")
        # Draw baseline
        self.policy_canvas.create_line(0, height // 2, width, height // 2, fill="#3a3a3a")
        series = []
        if values:
            series.extend(values)
        if step_values:
            series.extend(step_values)
        vmin = min(series)
        vmax = max(series)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0
        pad = 6
        span = vmax - vmin
        def scale(v: float) -> float:
            return pad + (1.0 - (v - vmin) / span) * (height - 2 * pad)
        if values:
            n = len(values)
            if n == 1:
                y = scale(values[0])
                self.policy_canvas.create_oval(width // 2 - 2, y - 2, width // 2 + 2, y + 2, fill="#00aaff")
            else:
                step = width / max(1, n - 1)
                points = []
                for i, val in enumerate(values):
                    x = i * step
                    y = scale(val)
                    points.extend([x, y])
                self.policy_canvas.create_line(*points, fill="#00aaff", width=2, smooth=True)

        if step_values:
            n = len(step_values)
            if n == 1:
                y = scale(step_values[0])
                self.policy_canvas.create_oval(width // 2 - 2, y - 2, width // 2 + 2, y + 2, fill="#ffb020")
            else:
                step = width / max(1, n - 1)
                points = []
                for i, val in enumerate(step_values):
                    x = i * step
                    y = scale(val)
                    points.extend([x, y])
                self.policy_canvas.create_line(*points, fill="#ffb020", width=1, smooth=True)

    def _build_tabs(self) -> None:
        self._menu_visible = False
        self._active_page = "Settings"
        self._menu_width = 110

        shell = ttk.Frame(self)
        shell.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = tk.Frame(shell, bg="#F6F7F9", height=44)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        self.menu_btn = tk.Label(
            header,
            text="☰",
            fg="#000000",
            bg="#F6F7F9",
            font=("Helvetica", 16, "bold"),
            cursor="hand2",
        )
        self.menu_btn.bind("<Button-1>", lambda _event: self._toggle_menu())
        self.menu_btn.place(x=10, y=22, anchor="w")
        if hasattr(self, "_icon_header_image"):
            self.title_icon = tk.Label(
                header,
                image=self._icon_header_image,
                bg="#F6F7F9",
            )
            self.title_icon.place(x=40, y=22, anchor="w")
        self.title_label = tk.Label(
            header,
            text="TowerLogic",
            fg="#000000",
            bg="#F6F7F9",
            font=("Helvetica", 14, "bold"),
        )
        self.title_label.place(x=76, y=22, anchor="w")

        body = ttk.Frame(shell)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.menu_frame = ttk.Frame(body)
        self.menu_frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.menu_frame.configure(width=self._menu_width)
        self.menu_frame.grid_propagate(False)

        self.content_frame = ttk.Frame(body)
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.rowconfigure(0, weight=1)

        self.settings_tab = ttk.Frame(self.content_frame)
        self.stats_tab = ttk.Frame(self.content_frame)
        self.logs_tab = ttk.Frame(self.content_frame)
        self.misc_tab = ttk.Frame(self.content_frame)

        self._pages = {
            "Settings": self.settings_tab,
            "Logs": self.logs_tab,
            "Analytics": self.stats_tab,
            "ML": self.misc_tab,
        }
        for page in self._pages.values():
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_remove()

        self._build_menu_buttons()
        self._show_page("Settings")
        self.menu_frame.grid_remove()

        self._create_jobs_tab()
        self._create_emulator_tab()
        self._create_logs_tab()
        self._create_stats_tab()
        self._create_misc_tab()

    def _build_menu_buttons(self) -> None:
        for child in self.menu_frame.winfo_children():
            child.destroy()
        for idx, name in enumerate(["Settings", "Logs", "Analytics", "ML"]):
            lbl = tk.Label(
                self.menu_frame,
                text=name,
                fg="#000000",
                bg="#F6F7F9",
                font=("Helvetica", 12),
                cursor="hand2",
                anchor="center",
            )
            lbl.bind("<Button-1>", lambda _event, n=name: self._show_page(n))
            lbl.grid(row=idx, column=0, sticky="ew", pady=6, padx=10)
        self.menu_frame.columnconfigure(0, weight=1)

    def _toggle_menu(self) -> None:
        self._menu_visible = not self._menu_visible
        if self._menu_visible:
            self.menu_frame.grid()
        else:
            self.menu_frame.grid_remove()

    def _show_page(self, name: str) -> None:
        if name not in self._pages:
            return
        for key, page in self._pages.items():
            if key == name:
                page.grid()
            else:
                page.grid_remove()
        self._active_page = name
        if self._menu_visible:
            self.menu_frame.grid_remove()
            self._menu_visible = False

    def show_page(self, name: str) -> None:
        self._show_page(name)

    def _build_bottom_row(self) -> None:
        bottom = ttk.Frame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom.columnconfigure(0, weight=1)
        self._status_text = "Idle"

        # Single unified button
        self.main_btn = ttk.Button(bottom, text="Start", bootstyle="success")
        self.main_btn.grid(row=0, column=0, sticky="ew", pady=(0, 0), ipady=8)
        self._register_config_widget("main_btn", self.main_btn)
        self._button_state = "idle"  # Track: idle, running, stopping

        # Action button (for retry etc.) - hidden by default
        self.action_btn = ttk.Button(bottom, text="Retry")
        self.action_btn.grid(row=0, column=0, sticky="ew", pady=(0, 0))
        self.action_btn.grid_remove()
        self._action_callback: Callable[[], None] | None = None
        self.action_btn.configure(command=self._on_action_pressed)

    def _create_jobs_tab(self) -> None:
        frame = ttk.Labelframe(self.settings_tab, text="Mode", padding=10)
        frame.pack(padx=10, pady=10, anchor="n", fill="x")

        frame.columnconfigure(0, weight=1)

        job_defaults = {job.key: job.default for job in JOBS}
        self.jobs_vars: dict[UIField, ttk.BooleanVar] = {}

        inner = ttk.Frame(frame)
        inner.grid(row=0, column=0, sticky="n")

        def add_job_checkbox(
            field: UIField,
            text: str,
            row_index: int,
            bootstyle: str,
        ) -> None:
            var = ttk.BooleanVar(value=job_defaults.get(field, False))
            checkbox = ttk.Checkbutton(
                inner,
                text=text,
                variable=var,
                bootstyle=bootstyle,
                command=lambda f=field: self._on_mode_toggle(f),
            )
            checkbox.pack(anchor="center", pady=2)
            self.jobs_vars[field] = var
            self._trace_variable(var)
            self._register_config_widget(field.value, checkbox)

        primary_bootstyle = "info-outline-toolbutton"

        add_job_checkbox(
            UIField.CLASSIC_1V1_USER_TOGGLE,
            "Classic 1v1",
            0,
            primary_bootstyle,
        )
        add_job_checkbox(
            UIField.CLASSIC_2V2_USER_TOGGLE,
            "Classic 2v2",
            1,
            primary_bootstyle,
        )
        add_job_checkbox(
            UIField.TROPHY_ROAD_USER_TOGGLE,
            "Trophy Road",
            2,
            primary_bootstyle,
        )

        # Ensure exactly one mode is selected by default.
        if not any(var.get() for var in self.jobs_vars.values()):
            self.jobs_vars[UIField.CLASSIC_1V1_USER_TOGGLE].set(True)
            self._notify_config_change()

        # Settings tab continues with device/emulator configuration below.

    def _create_emulator_tab(self) -> None:
        # Main container frame for the tab
        container = ttk.Frame(self.settings_tab, padding=10)
        container.pack(fill=BOTH, expand=YES)

        # Device Selection Dropdown
        selection_frame = ttk.Frame(container)
        selection_frame.pack(fill=X, pady=(0, 10))
        ttk.Label(selection_frame, text="Select Device:").pack(side=LEFT, padx=(0, 5))

        available_emulators = get_available_emulators()
        default_emulator = available_emulators[0] if available_emulators else EmulatorType.ADB
        self.emulator_var = ttk.StringVar(value=default_emulator)
        self.emulator_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.emulator_var,
            values=available_emulators,
            state=READONLY,
            width=20,
        )
        self.emulator_combo.pack(side=LEFT, fill=X, expand=True)
        self.emulator_combo.bind("<<ComboboxSelected>>", self._on_emulator_changed)
        # Register the combobox itself for state management
        self._register_config_widget("emulator_combobox", self.emulator_combo)

        ttk.Checkbutton(
            selection_frame,
            text="Show advanced settings",
            variable=self.advanced_settings_var,
            command=self._on_advanced_settings_toggled,
        ).pack(side=LEFT, padx=(8, 0))

        # Frame to hold the currently selected emulator's settings
        self.settings_container = ttk.Frame(container)
        self.settings_container.pack(fill=BOTH, expand=YES)

        # Create the individual settings frames but don't pack them yet
        self.google_play_frame = ttk.Frame(self.settings_container)
        self.memu_frame = ttk.Frame(self.settings_container)
        self.bluestacks_frame = ttk.Frame(self.settings_container)
        self.adb_frame = ttk.Frame(self.settings_container)

        # Store frames in a dictionary for easy access
        self.emulator_settings_frames = {
            EmulatorType.MEMU: self.memu_frame,
            EmulatorType.GOOGLE_PLAY: self.google_play_frame,
            EmulatorType.BLUESTACKS: self.bluestacks_frame,
            EmulatorType.ADB: self.adb_frame,
        }

        # Populate the settings frames
        self.gp_vars: dict[UIField, ttk.StringVar] = {}
        self._create_google_play_settings(self.google_play_frame)
        self._create_memu_settings(self.memu_frame)
        self._create_bluestacks_settings(self.bluestacks_frame)
        self._create_adb_tab(self.adb_frame)

        # Show the initial settings based on the default value
        self._show_current_emulator_settings()
        self._update_advanced_settings_visibility(self.emulator_var.get())

    def _create_logs_tab(self) -> None:
        container = ttk.Frame(self.logs_tab, padding=10)
        container.pack(fill=BOTH, expand=YES)

        header = ttk.Frame(container)
        header.pack(fill=X, pady=(0, 6))
        ttk.Label(header, text="Dump path:").pack(side=LEFT)
        dump_entry = ttk.Entry(header, textvariable=self.log_dump_path_var)
        dump_entry.pack(side=LEFT, fill=X, expand=YES, padx=(6, 6))
        dump_btn = ttk.Button(header, text="Dump Log", command=self._dump_log_to_file)
        dump_btn.pack(side=LEFT, padx=(0, 6))
        open_btn = ttk.Button(header, text="Open Folder", command=self._open_log_dump_folder)
        open_btn.pack(side=LEFT)

        log_frame = ttk.Labelframe(container, text="Logs", padding=6)
        log_frame.pack(fill=BOTH, expand=YES)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        text = tk.Text(log_frame, wrap="word")
        text.bind("<Key>", lambda _event: "break")
        def _update_lock() -> None:
            self._log_lock_scroll = text.yview()[1] < 0.98

        def _on_scrollbar(*args):
            text.yview(*args)
            _update_lock()

        def _on_mousewheel(event):
            if event.delta:
                text.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                text.yview_scroll(-1, "units")
            elif event.num == 5:
                text.yview_scroll(1, "units")
            _update_lock()
            return "break"

        text.bind("<MouseWheel>", _on_mousewheel)
        text.bind("<Button-4>", _on_mousewheel)
        text.bind("<Button-5>", _on_mousewheel)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=_on_scrollbar)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log_widgets.append(text)

    def _create_google_play_settings(self, parent_frame: ttk.Frame) -> None:
        frame = ttk.Labelframe(parent_frame, text="Google Play Options", padding=10)
        frame.pack(fill="x", padx=5, pady=5)

        left_keys = GOOGLE_PLAY_SETTINGS[:4]
        right_keys = GOOGLE_PLAY_SETTINGS[4:]

        for row, config in enumerate(left_keys):
            self._add_google_play_row(frame, row, 0, config)

        for row, config in enumerate(right_keys):
            self._add_google_play_row(frame, row, 3, config)

    def _create_memu_settings(self, parent_frame: ttk.Frame) -> None:
        self.memu_advanced_frame = ttk.Labelframe(parent_frame, text="Render Mode", padding=10)
        self.memu_advanced_frame.pack_forget()

        self.memu_render_var = ttk.StringVar(value="DirectX")
        for config in MEMU_SETTINGS:
            text = "DirectX" if config.key == UIField.DIRECTX_TOGGLE else "OpenGL"
            rb = ttk.Radiobutton(
                self.memu_advanced_frame,
                text=text,
                variable=self.memu_render_var,
                value=text,
                command=self._notify_config_change,
            )
            rb.pack(anchor="w")
            self._register_config_widget(config.key.value, rb)

    def _create_bluestacks_settings(self, parent_frame: ttk.Frame) -> None:
        instance_frame = ttk.Labelframe(parent_frame, text="Instance", padding=10)
        instance_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(instance_frame, text="Instance name (display):").grid(row=0, column=0, sticky="w", padx=(0, 6))
        instance_entry = ttk.Entry(instance_frame, textvariable=self.bs_instance_var)
        instance_entry.grid(row=0, column=1, sticky="ew")
        instance_frame.columnconfigure(1, weight=1)
        ToolTip(
            instance_entry,
            "Optional. If blank, the bot will use the first available instance. "
            "If set, it will use the instance with that display name.",
        )
        self._trace_variable(self.bs_instance_var)
        self._register_config_widget(UIField.BS_INSTANCE_NAME.value, instance_entry)

        self.bluestacks_advanced_frame = ttk.Labelframe(parent_frame, text="Render Mode", padding=10)
        self.bluestacks_advanced_frame.pack_forget()

        self.bs_render_var = ttk.StringVar(value="DirectX")
        for config in BLUESTACKS_SETTINGS:
            if config.key == UIField.BS_RENDERER_DX:
                value = "DirectX"
            elif config.key == UIField.BS_RENDERER_VK:
                value = "Vulkan"
            else:
                value = "OpenGL"
            rb = ttk.Radiobutton(
                self.bluestacks_advanced_frame,
                text=value,
                variable=self.bs_render_var,
                value=value,
                command=self._notify_config_change,
            )
            rb.pack(anchor="w")
            self._register_config_widget(config.key.value, rb)

    def _create_adb_tab(self, parent_frame: ttk.Frame) -> None:
        """Create the widgets for the ADB Device settings tab."""
        frame = ttk.Labelframe(parent_frame, text="Device Settings", padding=10)
        frame.pack(fill="x", padx=5, pady=5)

        # --- Row 1: Serial Input ---
        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(0, 5))
        row1.columnconfigure(1, weight=1)

        ttk.Label(row1, text="Device Serial:").grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.adb_serial_var = ttk.StringVar(value="")
        self.adb_serial_combo = ttk.Combobox(
            row1,
            textvariable=self.adb_serial_var,
            state=tk.NORMAL,
        )
        self.adb_serial_combo.grid(row=0, column=1, padx=5, sticky="ew")
        self._register_config_widget(UIField.ADB_SERIAL.value, self.adb_serial_combo)
        self._trace_variable(self.adb_serial_var)

        # --- Row 2: Connect/Refresh Buttons ---
        row_buttons_connect = ttk.Frame(frame)
        row_buttons_connect.pack(fill="x", pady=(0, 8))
        row_buttons_connect.columnconfigure(0, weight=1)
        row_buttons_connect.columnconfigure(1, weight=1)

        self.adb_connect_btn = ttk.Button(row_buttons_connect, text="Connect", style="success.TButton")
        self.adb_connect_btn.grid(row=0, column=0, padx=(0, 3), sticky="ew")
        self._register_config_widget("adb_connect_btn", self.adb_connect_btn)

        self.adb_refresh_btn = ttk.Button(row_buttons_connect, text="Refresh")
        self.adb_refresh_btn.grid(row=0, column=1, padx=(3, 0), sticky="ew")
        self._register_config_widget("adb_refresh_btn", self.adb_refresh_btn)

        # --- Row 3: Action Buttons (Stacked Vertically) ---
        row_buttons_action = ttk.Frame(frame)
        row_buttons_action.pack(fill="x")

        self.adb_restart_btn = ttk.Button(row_buttons_action, text="Restart ADB")
        self.adb_restart_btn.pack(fill=X, pady=(0, 3))
        self._register_config_widget("adb_restart_btn", self.adb_restart_btn)

        self.adb_set_size_btn = ttk.Button(row_buttons_action, text="Set Size & Density")
        self.adb_set_size_btn.pack(fill=X, pady=3)
        self._register_config_widget("adb_set_size_btn", self.adb_set_size_btn)

        self.adb_reset_size_btn = ttk.Button(row_buttons_action, text="Reset Size & Density")
        self.adb_reset_size_btn.pack(fill=X, pady=(3, 0))
        self._register_config_widget("adb_reset_size_btn", self.adb_reset_size_btn)

        ToolTip(self.adb_set_size_btn, "Sets screen to 419x633 and density to 160")
        ToolTip(self.adb_reset_size_btn, "Resets screen size and density to device defaults")

    def _create_stats_tab(self) -> None:
        container = ttk.Frame(self.stats_tab, padding=10)
        container.pack(fill=BOTH, expand=YES)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)
        left = ttk.Frame(container)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        gauge_frame = ttk.Labelframe(left, text="Win Rate", padding=10, style="Card.TLabelframe")
        gauge_frame.pack(fill=X)
        self.win_bar_var = tk.DoubleVar(value=0.0)
        self.win_bar = ttk.Progressbar(
            gauge_frame,
            length=200,
            mode="determinate",
            maximum=100,
            variable=self.win_bar_var,
            style="Winrate.Horizontal.TProgressbar",
        )
        self.win_bar.pack(fill=X, padx=6, pady=(6, 2))
        self.win_bar_label = ttk.Label(gauge_frame, text="0%")
        self.win_bar_label.pack(anchor="center", pady=(0, 4))

        battle_frame = ttk.Labelframe(left, text="Battle Stats", padding=10, style="Card.TLabelframe")
        battle_frame.pack(fill=BOTH, expand=YES, pady=(8, 0))
        self.stat_labels: dict[StatField, ttk.StringVar] = {}
        for row, field in enumerate(BATTLE_STAT_FIELDS):
            title = BATTLE_STAT_LABELS[field]
            label = ttk.Label(battle_frame, text=title, style="Card.TLabel")
            label.grid(row=row, column=0, sticky="w")
            self._theme_labels.append(label)
            var = ttk.StringVar(value="0")
            ttk.Label(battle_frame, textvariable=var, style="CardValue.TLabel").grid(row=row, column=1, sticky="e")
            self.stat_labels[field] = var

        # Add win streak stats
        ttk.Separator(battle_frame, orient="horizontal").grid(
            row=len(BATTLE_STAT_FIELDS), column=0, columnspan=2, sticky="ew", pady=(8, 4)
        )
        streak_row = len(BATTLE_STAT_FIELDS) + 1
        ttk.Label(battle_frame, text="Current Streak:", style="Card.TLabel").grid(row=streak_row, column=0, sticky="w")
        self.current_streak_var = ttk.StringVar(value="0")
        ttk.Label(battle_frame, textvariable=self.current_streak_var, style="CardValue.TLabel").grid(
            row=streak_row, column=1, sticky="e"
        )
        ttk.Label(battle_frame, text="Best Streak:", style="Card.TLabel").grid(row=streak_row + 1, column=0, sticky="w")
        self.best_streak_var = ttk.StringVar(value="0")
        ttk.Label(battle_frame, textvariable=self.best_streak_var, style="CardValue.TLabel").grid(
            row=streak_row + 1, column=1, sticky="e"
        )

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        bot_frame = ttk.Labelframe(right, text="Bot Stats", padding=10, style="Card.TLabelframe")
        bot_frame.pack(fill=X)
        self.bot_labels = {
            BotStatField.RESTARTS_AFTER_FAILURE: ttk.StringVar(value="0"),
            BotStatField.TIME_SINCE_START: ttk.StringVar(value="00:00:00"),
        }
        for row, field in enumerate(BOT_STAT_FIELDS):
            title = BOT_STAT_LABELS[field]
            label = ttk.Label(bot_frame, text=title, style="Card.TLabel")
            label.grid(row=row, column=0, sticky="w")
            self._theme_labels.append(label)
            ttk.Label(
                bot_frame,
                textvariable=self.bot_labels[field],
                style="CardValue.TLabel",
            ).grid(row=row, column=1, sticky="e")

        progress_frame = ttk.Labelframe(right, text="Learning Progress", padding=10, style="Card.TLabelframe")
        progress_frame.pack(fill=BOTH, expand=YES, pady=(8, 0))
        progress_frame.columnconfigure(0, weight=1)

        self.policy_canvas = tk.Canvas(progress_frame, height=120, highlightthickness=0)
        self.policy_canvas.grid(row=0, column=0, sticky="nsew")

        stats_row = ttk.Frame(progress_frame)
        stats_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        stats_row.columnconfigure(1, weight=1)
        self.policy_last_reward_var = ttk.StringVar(value="0.000")
        self.policy_last_loss_var = ttk.StringVar(value="0.000")
        self.policy_games_var = ttk.StringVar(value="0")
        ttk.Label(stats_row, text="Last reward:", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(stats_row, textvariable=self.policy_last_reward_var, style="CardValue.TLabel").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Label(stats_row, text="Last loss:", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Label(stats_row, textvariable=self.policy_last_loss_var, style="CardValue.TLabel").grid(
            row=0, column=3, sticky="w"
        )
        ttk.Label(stats_row, text="Games:", style="Card.TLabel").grid(row=0, column=4, sticky="w", padx=(10, 0))
        ttk.Label(stats_row, textvariable=self.policy_games_var, style="CardValue.TLabel").grid(
            row=0, column=5, sticky="w"
        )

    def _create_misc_tab(self) -> None:
        policy_frame = ttk.Labelframe(self.misc_tab, text="Policy", padding=10)
        policy_frame.pack(fill="x", padx=10, pady=10)
        policy_frame.columnconfigure(1, weight=1)

        policy_toggle = ttk.Checkbutton(
            policy_frame,
            text="Use PyTorch policy",
            variable=self.policy_toggle_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_toggle.grid(row=0, column=0, columnspan=2, sticky="w")
        self._trace_variable(self.policy_toggle_var)
        self._register_config_widget(UIField.POLICY_TOGGLE.value, policy_toggle)

        ttk.Label(policy_frame, text="Epsilon:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        policy_eps = ttk.Spinbox(
            policy_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.policy_epsilon_var,
            width=8,
            command=self._notify_config_change,
            state=READONLY,
        )
        policy_eps.grid(row=1, column=1, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_epsilon_var)
        self._register_config_widget(UIField.POLICY_EPSILON.value, policy_eps)

        ttk.Label(policy_frame, text="Model path:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        policy_model = ttk.Entry(policy_frame, textvariable=self.policy_model_var)
        policy_model.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        ToolTip(policy_model, "Optional. If blank, policy falls back to heuristics.")
        self._trace_variable(self.policy_model_var)
        self._register_config_widget(UIField.POLICY_MODEL_PATH.value, policy_model)

        reset_btn = ttk.Button(
            policy_frame,
            text="Reset model",
            command=self._on_policy_reset_model,
        )
        reset_btn.grid(row=2, column=2, sticky="w", padx=(6, 0))

        policy_train = ttk.Checkbutton(
            policy_frame,
            text="Online fine-tune (win/loss reward)",
            variable=self.policy_train_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_train.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_train_var)
        self._register_config_widget(UIField.POLICY_TRAIN_TOGGLE.value, policy_train)

        ttk.Label(policy_frame, text="Learning rate:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        policy_lr = ttk.Spinbox(
            policy_frame,
            from_=0.0001,
            to=0.01,
            increment=0.0001,
            textvariable=self.policy_lr_var,
            width=10,
            command=self._notify_config_change,
            state=READONLY,
        )
        policy_lr.grid(row=4, column=1, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_lr_var)
        self._register_config_widget(UIField.POLICY_LR.value, policy_lr)

        policy_free = ttk.Checkbutton(
            policy_frame,
            text="Free placement (ignore templates)",
            variable=self.policy_free_place_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_free.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_free_place_var)
        self._register_config_widget(UIField.POLICY_FREE_PLACEMENT.value, policy_free)

        policy_hold = ttk.Checkbutton(
            policy_frame,
            text="Policy decides when to play (allow hold)",
            variable=self.policy_allow_hold_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_hold.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_allow_hold_var)
        self._register_config_widget(UIField.POLICY_ALLOW_HOLD.value, policy_hold)

        ttk.Label(policy_frame, text="Min elixir before play:").grid(row=7, column=0, sticky="w", pady=(6, 0))
        min_elixir = ttk.Spinbox(
            policy_frame,
            from_=0,
            to=10,
            increment=1,
            textvariable=self.policy_min_elixir_var,
            width=6,
            command=self._notify_config_change,
            state=READONLY,
        )
        min_elixir.grid(row=7, column=1, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_min_elixir_var)
        self._register_config_widget(UIField.POLICY_MIN_ELIXIR.value, min_elixir)

        policy_debug = ttk.Checkbutton(
            policy_frame,
            text="Debug detection overlay",
            variable=self.policy_debug_detect_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_debug.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_debug_detect_var)
        self._register_config_widget(UIField.POLICY_DEBUG_DETECTION.value, policy_debug)

        policy_overlay = ttk.Checkbutton(
            policy_frame,
            text="Live debug overlay window",
            variable=self.policy_debug_overlay_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_overlay.grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_debug_overlay_var)
        self._register_config_widget(UIField.POLICY_DEBUG_OVERLAY.value, policy_overlay)

        policy_hp_log = ttk.Checkbutton(
            policy_frame,
            text="Log tower HP deltas",
            variable=self.policy_debug_hp_log_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_hp_log.grid(row=10, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_debug_hp_log_var)
        self._register_config_widget(UIField.POLICY_DEBUG_HP_LOG.value, policy_hp_log)

        policy_hand_conf = ttk.Checkbutton(
            policy_frame,
            text="Log hand classifier top-3",
            variable=self.policy_debug_hand_conf_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        policy_hand_conf.grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_debug_hand_conf_var)
        self._register_config_widget(UIField.POLICY_DEBUG_HAND_CONF.value, policy_hand_conf)

        hp_probe_btn = ttk.Button(
            policy_frame,
            text="HP Probe Snapshot",
            command=self._on_hp_probe,
        )
        hp_probe_btn.grid(row=12, column=0, columnspan=2, sticky="w", pady=(6, 0))

        hand_probe_btn = ttk.Button(
            policy_frame,
            text="Hand Probe Snapshot",
            command=self._on_hand_probe,
        )
        hand_probe_btn.grid(row=13, column=0, columnspan=2, sticky="w", pady=(6, 0))

        strict_elixir = ttk.Checkbutton(
            policy_frame,
            text="Strict min elixir (no plays under threshold)",
            variable=self.policy_strict_elixir_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        strict_elixir.grid(row=14, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_strict_elixir_var)
        self._register_config_widget(UIField.POLICY_STRICT_ELIXIR.value, strict_elixir)

        spell_gate = ttk.Checkbutton(
            policy_frame,
            text="Spell gate (no spells when calm)",
            variable=self.policy_spell_gate_var,
            bootstyle="round-toggle",
            command=self._notify_config_change,
        )
        spell_gate.grid(row=15, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._trace_variable(self.policy_spell_gate_var)
        self._register_config_widget(UIField.POLICY_SPELL_GATE.value, spell_gate)


    def _register_config_widget(self, key: str, widget: tk.Widget) -> None:
        self._config_widgets[key] = widget

    def _notify_config_change(self, *_: object) -> None:
        if self._suspend_traces > 0 or self._config_callback is None:
            return
        self.after_idle(lambda: self._config_callback(self.get_all_values()))

    def _trace_variable(self, var: tk.Variable) -> None:
        trace_id = var.trace_add("write", self._notify_config_change)
        self._traces.append((var, trace_id))

    def _add_google_play_row(
        self,
        frame: ttk.Labelframe,
        row: int,
        column_offset: int,
        config: ComboConfig,
    ) -> None:
        ttk.Label(frame, text=config.label).grid(row=row, column=column_offset, sticky="w", padx=5, pady=2)
        var = ttk.StringVar(value=str(config.default))
        combo = ttk.Combobox(
            frame,
            values=[str(option) for option in config.values],
            width=12,
            state=READONLY,
            textvariable=var,
        )
        combo.grid(row=row, column=column_offset + 1, sticky="w")
        combo.bind("<<ComboboxSelected>>", self._notify_config_change_event)
        field = config.key
        self.gp_vars[field] = var
        self._trace_variable(var)
        self._register_config_widget(field.value, combo)

    def _notify_config_change_event(self, _event: object) -> None:
        self._notify_config_change()

    def _on_mode_toggle(self, selected: UIField) -> None:
        if selected not in self.jobs_vars:
            self._notify_config_change()
            return
        selected_var = self.jobs_vars[selected]
        if not selected_var.get():
            # Prevent all-off: re-enable the selected toggle.
            selected_var.set(True)
            return
        for field, var in self.jobs_vars.items():
            if field != selected and var.get():
                var.set(False)
        self._notify_config_change()

    def _update_google_play_comboboxes(self) -> None:
        for field, var in self.gp_vars.items():
            widget = self._config_widgets.get(field.value)
            if not widget:
                continue
            values = [str(option) for option in widget.cget("values")]
            if var.get() not in values and values:
                var.set(values[0])

    def _apply_theme(self, theme_name: str, skip_variable_update: bool = False) -> None:
        available = tuple(self._style.theme_names())
        selected = theme_name if theme_name in available else self.DEFAULT_THEME
        if selected not in available and available:
            selected = available[0]
        if not skip_variable_update or self.theme_var.get() != selected:
            self._suspend_traces += 1
            try:
                self.theme_var.set(selected)
            finally:
                self._suspend_traces -= 1
        try:
            self._style.theme_use(selected)
        except tk.TclError:
            # Some Tk builds can throw when styling combobox popdowns too early.
            # Retry once later, then give up to avoid freezing the UI.
            if not self._theme_retry_pending:
                self._theme_retry_pending = True
                self.after(250, lambda: self._apply_theme(selected, skip_variable_update=True))
            return
        self._theme_retry_pending = False
        self._refresh_theme_colours()

    def _label_foreground(self) -> str:
        try:
            colour = self._style.lookup("TLabel", "foreground")
            return colour or "#202020"
        except tk.TclError:
            return "#202020"

    def _refresh_theme_colours(self) -> None:
        bg = "#F6F7F9"
        surface = "#FFFFFF"
        text = "#111827"
        muted = "#6B7280"
        border = "#E5E7EB"
        primary = "#000000"
        success = "#22C55E"
        danger = "#EF4444"

        try:
            self.configure(background=bg)
        except tk.TclError:
            pass

        foreground = text
        for label in self._theme_labels:
            try:
                label.configure(foreground=foreground)
            except tk.TclError:
                continue
        try:
            self._style.configure("TFrame", background=bg)
            self._style.configure("TLabel", background=bg, foreground=foreground)
            self._style.configure("TLabelframe", background=bg, bordercolor=border)
            self._style.configure("TLabelframe.Label", background=bg, foreground=foreground)
            self._style.configure("Card.TLabelframe", background=surface, bordercolor=border)
            self._style.configure("Card.TLabelframe.Label", background=surface, foreground=foreground)
            self._style.configure("Card.TLabel", background=surface, foreground=foreground, relief="flat")
            self._style.configure("CardValue.TLabel", background=surface, foreground=primary)
            self._style.configure("TEntry", fieldbackground=surface, foreground=foreground)
            self._style.configure("TCombobox", fieldbackground=surface, foreground=foreground)
            self._style.configure("secondary-link.TButton", background=bg, foreground=primary)
            self._style.map(
                "secondary-link.TButton",
                foreground=[("active", primary), ("disabled", primary)],
                background=[("active", "#EEF2FF")],
            )
            self._style.configure(
                "secondary-outline-toolbutton.TButton",
                background=bg,
                foreground=primary,
                bordercolor=border,
            )
            self._style.configure(
                "TButton",
                background=surface,
                foreground=foreground,
                bordercolor=border,
                lightcolor=border,
                darkcolor=border,
            )
            self._style.map(
                "TButton",
                background=[("active", "#EEF2FF")],
                foreground=[("active", foreground)],
            )
            self._style.configure(
                "Winrate.Horizontal.TProgressbar",
                troughcolor=danger,  # loss portion
                background=success,   # win portion
                foreground=success,
            )
        except tk.TclError:
            pass
        if hasattr(self, "win_bar_label"):
            try:
                self.win_bar_label.configure(foreground=foreground, background=surface)
            except tk.TclError:
                pass

    def _on_theme_change(self, _event: object | None = None) -> None:
        self._apply_theme(self.theme_var.get(), skip_variable_update=True)
        self._notify_config_change()

    def _on_emulator_changed(self, _event: object = None) -> None:
        if self.emulator_var.get() == EmulatorType.ADB:
            messagebox.showwarning(
                "ADB Mode",
                "ADB mode is intended for advanced users only. Support will not be provided for ADB mode.",
            )
        self._show_current_emulator_settings()
        self._notify_config_change()

    def _show_current_emulator_settings(self) -> None:
        """Hides all emulator settings frames and shows the one selected in the combobox."""
        selected_emulator = self.emulator_var.get()
        show_advanced = bool(self.advanced_settings_var.get())

        # Hide all frames first
        for frame in self.emulator_settings_frames.values():
            frame.pack_forget()

        # Show the selected frame
        frame_to_show = self.emulator_settings_frames.get(selected_emulator)
        should_show = True
        if selected_emulator == EmulatorType.GOOGLE_PLAY:
            should_show = show_advanced
        if frame_to_show and should_show:
            frame_to_show.pack(fill=BOTH, expand=YES)
        self._update_advanced_settings_visibility(selected_emulator)

    def _hide_action_button(self) -> None:
        self.action_btn.grid_remove()
        self.main_btn.grid()

    def _on_action_pressed(self) -> None:
        if self._action_callback:
            self._action_callback()
        self._hide_action_button()

    def _on_open_logs_clicked(self) -> None:
        if self._open_logs_callback:
            self._open_logs_callback()

    def _on_policy_reset_model(self) -> None:
        path = self.policy_model_var.get().strip()
        if not path:
            messagebox.showinfo("Reset Model", "No model path set.")
            return
        if not os.path.isfile(path):
            messagebox.showinfo("Reset Model", "Model file not found.")
            return
        if not messagebox.askyesno("Reset Model", "Delete the policy model file?"):
            return
        try:
            os.remove(path)
            messagebox.showinfo("Reset Model", "Model file deleted.")
        except Exception as exc:
            messagebox.showerror("Reset Model", f"Failed to delete model: {exc}")

    def _on_hp_probe(self) -> None:
        flag_path = os.path.join(log_dir, "hp_probe.flag")
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(flag_path, "w", encoding="utf-8") as f:
                f.write("1")
            messagebox.showinfo("HP Probe", "HP probe requested. Check logs for hp_probe_*.png")
        except Exception as exc:
            messagebox.showerror("HP Probe", f"Failed to request HP probe: {exc}")

    def _on_hand_probe(self) -> None:
        flag_path = os.path.join(log_dir, "hand_probe.flag")
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(flag_path, "w", encoding="utf-8") as f:
                f.write("1")
            messagebox.showinfo(
                "Hand Probe",
                "Hand probe requested. Check logs for [HandProbe] and hand_probe_*.png",
            )
        except Exception as exc:
            messagebox.showerror("Hand Probe", f"Failed to request hand probe: {exc}")


    def _update_advanced_settings_visibility(self, emulator_choice: str) -> None:
        show_advanced = bool(self.advanced_settings_var.get())

        def _toggle_frame(frame: ttk.Frame | None, should_show: bool) -> None:
            if not frame:
                return
            try:
                frame.pack_forget()
                if should_show:
                    frame.pack(fill="x", padx=5, pady=5)
            except tk.TclError:
                return

        is_memu = emulator_choice == EmulatorType.MEMU
        is_bluestacks = emulator_choice == EmulatorType.BLUESTACKS

        _toggle_frame(getattr(self, "memu_advanced_frame", None), show_advanced and is_memu)
        _toggle_frame(getattr(self, "bluestacks_advanced_frame", None), show_advanced and is_bluestacks)

    def _on_advanced_settings_toggled(self) -> None:
        # Re-evaluate which emulator settings should be visible when the toggle changes
        self._show_current_emulator_settings()

    @staticmethod
    def _safe_int(value: object, fallback: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _safe_float(value: object, fallback: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _parse_winrate_value(raw: object) -> float | None:
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.endswith("%"):
                stripped = stripped[:-1]
            try:
                return float(stripped)
            except ValueError:
                return None
        if isinstance(raw, int | float):
            return float(raw)
        return None

    @staticmethod
    def _calculate_winrate_percentage(wins: int, losses: int) -> float:
        total = wins + losses
        if total <= 0:
            return 0.0
        return wins / total * 100
