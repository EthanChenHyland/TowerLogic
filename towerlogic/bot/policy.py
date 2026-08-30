from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable

from towerlogic.bot.card_detection import PLAY_COORDS

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


SCREEN_WIDTH = 419
SCREEN_HEIGHT = 633
USE_CONTINUOUS_COORDS = os.getenv("PYCLASHBOT_CONTINUOUS_COORDS", "1") not in {"0", "false", "False"}
CONTINUOUS_SAMPLES = int(os.getenv("PYCLASHBOT_CONTINUOUS_SAMPLES", "18"))
REWARD_SCALE = 1.5  # Global scaling for reward shaping magnitude
STATE_EXTRA_DIM = 19  # enemy_activity, wincon_available,
# enemy_red_top, enemy_red_mid, enemy_template_top, enemy_template_mid,
# lane_bias, defend_mode, player_hp, enemy_hp, player_hp_delta, enemy_hp_delta,
# player_left/right/king_delta, enemy_left/right/king_delta, crown_diff
STATE_BASE_DIM = 3 + 4 + STATE_EXTRA_DIM  # elapsed, elixir, side + 4 card mask + extras
ACTION_DIM = 4
SAFE_BOUNDS_DEFAULT = (100, 320, 170, 480)  # x_min, x_max, y_min, y_max for 419x633 (tighter safe zone)
SPELL_GROUPS = {
    "spell",
    "earthquake",
    "fireball",
    "freeze",
    "poison",
    "arrows",
    "snowball",
    "zap",
    "rocket",
    "lightning",
    "log",
    "tornado",
    "goblin_drill",
    "graveyard",
}
FORWARD_ALLOWED_GROUPS = SPELL_GROUPS


@dataclass(frozen=True)
class PolicyAction:
    card_index: int
    card_group: str
    coord: tuple[int, int]
    card_id: str | None = None


def _group_index_map() -> dict[str, int]:
    groups = sorted(PLAY_COORDS.keys())
    groups.append("No group")
    return {name: idx for idx, name in enumerate(groups)}


GROUP_INDEX = _group_index_map()


def candidate_coords_for_group(card_group: str, side_preference: str, elapsed_time: float) -> list[tuple[int, int]]:
    def _sample_uniform(x_min: int, x_max: int, y_min: int, y_max: int, count: int) -> list[tuple[int, int]]:
        if x_max <= x_min or y_max <= y_min:
            return []
        coords: list[tuple[int, int]] = []
        for _ in range(max(1, count)):
            x = int(random.uniform(x_min, x_max))
            y = int(random.uniform(y_min, y_max))
            coords.append((x, y))
        return coords

    def _grid_coords(x_min: int, x_max: int, y_min: int, y_max: int) -> list[tuple[int, int]]:
        x_steps = [
            x_min,
            int(x_min + (x_max - x_min) * 0.33),
            int(x_min + (x_max - x_min) * 0.66),
            x_max,
        ]
        y_steps = [
            y_min,
            int(y_min + (y_max - y_min) * 0.25),
            int(y_min + (y_max - y_min) * 0.5),
            int(y_min + (y_max - y_min) * 0.75),
            y_max,
        ]
        return [(x, y) for x in x_steps for y in y_steps]

    def _fallback_grid() -> list[tuple[int, int]]:
        if elapsed_time < 12:
            y_min, y_max = 441, 456
        elif elapsed_time < 80:
            y_min, y_max = 360, 456
        else:
            y_min, y_max = 281, 456

        if side_preference == "left":
            x_min, x_max = 60, 206
        else:
            x_min, x_max = 210, 351
        if USE_CONTINUOUS_COORDS:
            return _sample_uniform(x_min, x_max, y_min, y_max, CONTINUOUS_SAMPLES)
        return _grid_coords(x_min, x_max, y_min, y_max)

    if card_group != "No group":
        group_datum = PLAY_COORDS.get(card_group, {})
        if side_preference in group_datum:
            base = list(group_datum[side_preference])
            if card_group not in SPELL_GROUPS:
                if USE_CONTINUOUS_COORDS:
                    if card_group == "turret":
                        x_min, x_max = 180, 240
                        y_min, y_max = 300, 420
                    elif card_group == "big_win_con":
                        x_min, x_max = (60, 206) if side_preference == "left" else (210, 351)
                        y_min, y_max = (360, 456) if elapsed_time < 80 else (320, 456)
                    else:
                        x_min, x_max = (60, 206) if side_preference == "left" else (210, 351)
                        y_min, y_max = (360, 456) if elapsed_time < 80 else (300, 456)
                    base.extend(_sample_uniform(x_min, x_max, y_min, y_max, CONTINUOUS_SAMPLES))
                else:
                    base.extend(_fallback_grid())
            return list(dict.fromkeys(base))
        if "coords" in group_datum:
            base = list(group_datum["coords"])
            if card_group not in SPELL_GROUPS:
                if USE_CONTINUOUS_COORDS:
                    x_min, x_max = (60, 206) if side_preference == "left" else (210, 351)
                    y_min, y_max = (360, 456) if elapsed_time < 80 else (300, 456)
                    base.extend(_sample_uniform(x_min, x_max, y_min, y_max, CONTINUOUS_SAMPLES))
                else:
                    base.extend(_fallback_grid())
            return list(dict.fromkeys(base))

    # Fallback: sample a small, safe grid based on elapsed time/side
    return _fallback_grid()


def candidate_coords_free(bounds: tuple[int, int, int, int] | None = None) -> list[tuple[int, int]]:
    # Full arena grid (avoid UI: top bar and bottom card bar), with margin to avoid corners.
    x_min, x_max, y_min, y_max = bounds or SAFE_BOUNDS_DEFAULT
    margin = 45
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin
    if x_max <= x_min or y_max <= y_min:
        x_min, x_max, y_min, y_max = SAFE_BOUNDS_DEFAULT
    if USE_CONTINUOUS_COORDS:
        coords: list[tuple[int, int]] = []
        for _ in range(max(1, CONTINUOUS_SAMPLES)):
            coords.append((int(random.uniform(x_min, x_max)), int(random.uniform(y_min, y_max))))
        return coords
    x_steps = [
        x_min,
        int(x_min + (x_max - x_min) * 0.25),
        int(x_min + (x_max - x_min) * 0.5),
        int(x_min + (x_max - x_min) * 0.75),
        x_max,
    ]
    y_steps = [
        y_min,
        int(y_min + (y_max - y_min) * 0.2),
        int(y_min + (y_max - y_min) * 0.4),
        int(y_min + (y_max - y_min) * 0.6),
        int(y_min + (y_max - y_min) * 0.8),
        y_max,
    ]
    return [(x, y) for x in x_steps for y in y_steps]


def safe_bounds_for_image(image) -> tuple[int, int, int, int]:
    if image is None or getattr(image, "shape", None) is None:
        return SAFE_BOUNDS_DEFAULT
    h, w = image.shape[:2]
    x_min = int(w * (60 / 419))
    x_max = int(w * (360 / 419))
    y_min = int(h * (120 / 633))
    y_max = int(h * (520 / 633))
    # Ensure sane bounds
    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w - 1))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h - 1))
    if x_max <= x_min or y_max <= y_min:
        return SAFE_BOUNDS_DEFAULT
    return (x_min, x_max, y_min, y_max)


def clamp_coord(coord: tuple[int, int], bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = bounds
    x = max(x_min, min(x_max, coord[0]))
    y = max(y_min, min(y_max, coord[1]))
    return (x, y)


def is_near_edge(coord: tuple[int, int], bounds: tuple[int, int, int, int], margin: int = 15) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return (
        coord[0] <= x_min + margin
        or coord[0] >= x_max - margin
        or coord[1] <= y_min + margin
        or coord[1] >= y_max - margin
    )


def build_state_vector(
    elapsed_time: float,
    elixir: int,
    side_preference: str,
    available_mask: Iterable[int],
    enemy_activity: float = 0.0,
    wincon_available: float = 0.0,
    enemy_red_top: float = 0.0,
    enemy_red_mid: float = 0.0,
    enemy_template_top: float = 0.0,
    enemy_template_mid: float = 0.0,
    lane_bias: float = 0.5,
    defend_mode: float = 0.0,
    player_hp: float = 1.0,
    enemy_hp: float = 1.0,
    player_hp_delta: float = 0.0,
    enemy_hp_delta: float = 0.0,
    player_left_delta: float = 0.0,
    player_right_delta: float = 0.0,
    player_king_delta: float = 0.0,
    enemy_left_delta: float = 0.0,
    enemy_right_delta: float = 0.0,
    enemy_king_delta: float = 0.0,
    crown_diff: float = 0.0,
) -> list[float]:
    side_flag = 1.0 if side_preference == "right" else 0.0
    base = [elapsed_time / 300.0, elixir / 10.0, side_flag]
    mask = [float(v) for v in available_mask]
    extras = [
        float(max(0.0, min(1.0, enemy_activity))),
        float(max(0.0, min(1.0, wincon_available))),
        float(max(0.0, min(1.0, enemy_red_top))),
        float(max(0.0, min(1.0, enemy_red_mid))),
        float(max(0.0, min(1.0, enemy_template_top))),
        float(max(0.0, min(1.0, enemy_template_mid))),
        float(max(0.0, min(1.0, lane_bias))),
        float(max(0.0, min(1.0, defend_mode))),
        float(max(0.0, min(1.0, player_hp))),
        float(max(0.0, min(1.0, enemy_hp))),
        float(max(-1.0, min(1.0, player_hp_delta))),
        float(max(-1.0, min(1.0, enemy_hp_delta))),
        float(max(-1.0, min(1.0, player_left_delta))),
        float(max(-1.0, min(1.0, player_right_delta))),
        float(max(-1.0, min(1.0, player_king_delta))),
        float(max(-1.0, min(1.0, enemy_left_delta))),
        float(max(-1.0, min(1.0, enemy_right_delta))),
        float(max(-1.0, min(1.0, enemy_king_delta))),
        float(max(-1.0, min(1.0, crown_diff))),
    ]
    return base + mask + extras


def build_action_vector(action: PolicyAction) -> list[float]:
    group_id = GROUP_INDEX.get(action.card_group, GROUP_INDEX["No group"])
    x, y = action.coord
    return [float(action.card_index), float(group_id), x / SCREEN_WIDTH, y / SCREEN_HEIGHT]


def hold_action() -> PolicyAction:
    return PolicyAction(card_index=-1, card_group="No group", coord=(0, 0))


def build_input_batch(state_vec: list[float], actions: list[PolicyAction]) -> list[list[float]]:
    return [state_vec + build_action_vector(action) for action in actions]


class EpsilonGreedyPolicy:
    def __init__(self, model, epsilon: float = 0.1):
        self.model = model
        self.epsilon = float(max(0.0, min(1.0, epsilon)))

    def select_action(self, state_vec: list[float], actions: list[PolicyAction]) -> PolicyAction | None:
        if not actions:
            return None
        if self.model is None or random.random() < self.epsilon:
            return random.choice(actions)

        if torch is None:
            return random.choice(actions)

        batch = build_input_batch(state_vec, actions)
        inputs = torch.tensor(batch, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(inputs)
        if outputs is None:
            return random.choice(actions)

        # Support models returning shape (N,), (N,1), or (N,K)
        if outputs.dim() == 2 and outputs.shape[1] > 1:
            scores = outputs.max(dim=1).values
        else:
            scores = outputs.view(-1)
        best_idx = int(torch.argmax(scores).item())
        return actions[best_idx]


if nn is not None:
    class SimplePolicyNet(nn.Module):
        def __init__(self, input_dim: int = STATE_BASE_DIM + ACTION_DIM):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):  # type: ignore[override]
            return self.net(x)
else:
    class SimplePolicyNet:  # type: ignore[override]
        def __init__(self, input_dim: int = 11):
            raise RuntimeError("PyTorch is not installed. Install torch to use policy training.")


def load_torch_model(
    path: str | None,
    logger=None,
    trainable: bool = False,
    input_dim: int = STATE_BASE_DIM + ACTION_DIM,
):
    if torch is None:
        if logger:
            logger.log("[Policy] PyTorch is not installed; falling back to heuristics.")
        return None
    if not path:
        return SimplePolicyNet(input_dim) if trainable else None
    if not os.path.isfile(path):
        if logger:
            logger.log(f"[Policy] Model path not found: {path}")
        return SimplePolicyNet(input_dim) if trainable else None

    # Try TorchScript first
    try:
        model = torch.jit.load(path)
        model.eval()
        if trainable:
            if logger:
                logger.log("[Policy] TorchScript model loaded (not trainable). Starting fresh trainable model.")
            return SimplePolicyNet(input_dim)
        return model
    except Exception:
        pass

    # Try state_dict
    try:
        state = torch.load(path, map_location="cpu")
        model = SimplePolicyNet(input_dim)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - runtime safety
        if logger:
            logger.log(f"[Policy] Failed to load model: {exc}")
        return SimplePolicyNet(input_dim) if trainable else None


def save_torch_model(model, path: str | None, logger=None):
    if not path or torch is None or model is None:
        return
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), path)
    except Exception as exc:  # pragma: no cover - runtime safety
        if logger:
            logger.log(f"[Policy] Failed to save model: {exc}")


class OnlineTrainer:
    def __init__(self, model, lr: float = 5e-4, model_path: str | None = None, logger=None):
        self.model = model
        self.model_path = model_path
        self.logger = logger
        self.replay_buffer: list[tuple[list[float], PolicyAction, float]] = []
        self.replay_max = 128
        self.buffer: list[list[float]] = []
        self.action_count = 0
        self.spell_count = 0
        self.elixir_sum = 0.0
        self.elixir_samples = 0
        self.used_indices: set[int] = set()
        self.card_index_counts = [0, 0, 0, 0]
        self.last_card_index: int | None = None
        self.repeat_streak = 0
        self.hold_count = 0
        self.low_elixir_plays = 0
        self.invalid_placements = 0
        self.clamped_placements = 0
        self.edge_placements = 0
        self.early_forward = 0
        self.early_back = 0
        self.last_crowns: tuple[int, int] | None = None
        self.prev_enemy_hp: float | None = None
        self.prev_player_hp: float | None = None
        self.enemy_damage = 0.0
        self.player_damage = 0.0
        self.enemy_activity_samples = 0
        self.enemy_activity_present = 0
        self.defensive_plays = 0
        self.offensive_when_clear = 0
        self.hold_when_clear = 0
        self.hold_when_clear_high = 0
        self.no_damage_after_play = 0
        self.player_hp_drop_after_play = 0
        self.ineffective_plays = 0
        self.effective_spells = 0
        self.overcap_holds = 0
        self.wincon_when_clear = 0
        self.wincon_back = 0
        self.wincon_early = 0
        self.wincon_effective = 0
        self.hold_for_wincon = 0
        self.forward_clear = 0
        self.back_clear = 0
        self.step_updates = 0
        self._last_autosave = 0
        self.enemy_tower_damage = 0.0
        self.player_tower_damage = 0.0
        self.attack_on_bias = 0
        self.attack_off_bias = 0
        self.attack_on_target = 0
        self.attack_off_target = 0
        self.spell_calm_spam = 0
        self.ineffective_spells = 0
        self.card_repeat_fast = 0
        self.card_cycle_reward = 0
        self._last_play_time_by_card: dict[str, float] = {}
        self.counter_push_chain = 0
        self.calm_elixir_spend = 0.0
        self.stacked_defense = 0
        self._last_defensive_elapsed: float | None = None
        self._last_defensive_side: str | None = None
        if torch is not None and model is not None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.optimizer = None
            self.loss_fn = None

    def update_step(self, state_vec: list[float], action: PolicyAction, reward: float) -> None:
        """Immediate per-play update with a single (state, action) sample."""
        if self.model is None or torch is None or self.optimizer is None or self.loss_fn is None:
            return
        if not state_vec:
            return
        # Replay buffer
        self.replay_buffer.append((state_vec, action, float(reward)))
        if len(self.replay_buffer) > self.replay_max:
            self.replay_buffer = self.replay_buffer[-self.replay_max :]

        # Sample a small batch from the buffer for stability.
        batch = [self.replay_buffer[-1]]
        if len(self.replay_buffer) >= 8:
            batch = random.sample(self.replay_buffer, k=8)

        self.model.train()
        inputs = torch.tensor(
            [s + build_action_vector(a) for (s, a, _r) in batch],
            dtype=torch.float32,
        )
        targets = torch.tensor([[float(r)] for (_s, _a, r) in batch], dtype=torch.float32)
        out = self.model(inputs)
        if out.dim() == 1:
            out = out.view(-1, 1)
        loss = self.loss_fn(out, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        self.step_updates += 1
        if self.logger:
            try:
                self.logger.add_policy_step(reward)
            except Exception:
                pass
        # Periodic autosave to persist learning mid-game.
        if self.step_updates - self._last_autosave >= 25:
            save_torch_model(self.model, self.model_path, logger=self.logger)
            self._last_autosave = self.step_updates

    def record(
        self,
        state_vec: list[float],
        action: PolicyAction,
        elapsed_time: float | None = None,
        invalid: bool = False,
        clamped: bool = False,
        edge: bool = False,
        crowns: tuple[int, int] | None = None,
        enemy_hp: float | None = None,
        player_hp: float | None = None,
        enemy_activity: float | None = None,
        enemy_activity_after: float | None = None,
        enemy_hp_after: float | None = None,
        player_hp_after: float | None = None,
        target_side: str | None = None,
        action_side: str | None = None,
        wincon_available: bool | None = None,
        player_left_delta: float | None = None,
        player_right_delta: float | None = None,
        player_king_delta: float | None = None,
        enemy_left_delta: float | None = None,
        enemy_right_delta: float | None = None,
        enemy_king_delta: float | None = None,
        lane_bias: float | None = None,
        defend_mode: float | None = None,
        elixir_after: float | None = None,
        card_id: str | None = None,
    ) -> None:
        if self.model is None:
            return
        self.buffer.append(state_vec + build_action_vector(action))
        elixir_now = None
        if len(state_vec) > 1:
            elixir_now = float(state_vec[1]) * 10.0
        if action.card_index < 0:
            self.hold_count += 1
            if elixir_now is not None:
                if elixir_now >= 9.5:
                    self.overcap_holds += 1
                if (enemy_activity or 0.0) < 0.05 and elixir_now >= 7.5:
                    self.hold_when_clear_high += 1
                if (
                    wincon_available
                    and elixir_now >= 5.5
                    and elixir_now < 8.5
                    and (enemy_activity or 0.0) < 0.05
                ):
                    self.hold_for_wincon += 1
        else:
            self.action_count += 1
            if 0 <= action.card_index < len(self.card_index_counts):
                self.card_index_counts[action.card_index] += 1
            self.used_indices.add(action.card_index)
            if self.last_card_index == action.card_index:
                self.repeat_streak += 1
            else:
                self.repeat_streak = 0
            self.last_card_index = action.card_index
        if action.card_group in SPELL_GROUPS:
            self.spell_count += 1
        if len(state_vec) > 1:
            self.elixir_sum += float(state_vec[1]) * 10.0
            self.elixir_samples += 1
            if action.card_index >= 0 and float(state_vec[1]) * 10.0 < 3.0:
                self.low_elixir_plays += 1
            if (
                action.card_index >= 0
                and elixir_after is not None
                and (enemy_activity or 0.0) < 0.05
            ):
                delta = max(0.0, float(state_vec[1]) * 10.0 - float(elixir_after))
                if delta > 0:
                    self.calm_elixir_spend += delta
        if invalid:
            self.invalid_placements += 1
        if clamped:
            self.clamped_placements += 1
        if edge:
            self.edge_placements += 1
        if elapsed_time is not None and action.card_index >= 0:
            y = action.coord[1]
            if elapsed_time < 40 and y < 260:
                self.early_forward += 1
            if elapsed_time < 40 and y > 360:
                self.early_back += 1
        if crowns is not None:
            self.last_crowns = crowns
        if enemy_hp is not None:
            if self.prev_enemy_hp is not None and enemy_hp < self.prev_enemy_hp:
                self.enemy_damage += (self.prev_enemy_hp - enemy_hp)
            self.prev_enemy_hp = enemy_hp
        if player_hp is not None:
            if self.prev_player_hp is not None and player_hp < self.prev_player_hp:
                self.player_damage += (self.prev_player_hp - player_hp)
            self.prev_player_hp = player_hp
        if enemy_activity is not None:
            self.enemy_activity_samples += 1
            if enemy_activity > 0.01:
                self.enemy_activity_present += 1
                if action.card_index >= 0 and action.coord[1] > 360:
                    self.defensive_plays += 1
            else:
                if action.card_index < 0:
                    self.hold_when_clear += 1
                else:
                    if action.coord[1] < 300:
                        self.offensive_when_clear += 1

        # Counter-push chain: after a defensive play, reward a same-lane back-lane followup.
        if action.card_index >= 0 and elapsed_time is not None:
            side = action_side
            if side is None:
                side = "left" if action.coord[0] <= 210 else "right"
            if (
                self._last_defensive_elapsed is not None
                and self._last_defensive_side is not None
                and (elapsed_time - self._last_defensive_elapsed) <= 8.0
                and action.coord[1] >= 340
                and side == self._last_defensive_side
                and action.card_group not in SPELL_GROUPS
            ):
                self.counter_push_chain += 1

            if action.coord[1] >= 360 and action.card_group not in SPELL_GROUPS:
                self._last_defensive_elapsed = elapsed_time
                self._last_defensive_side = side

        # Post-play feedback (quick-response penalties)
        if action.card_index >= 0:
            if enemy_hp is not None and enemy_hp_after is not None:
                if enemy_hp_after >= enemy_hp - 0.005:
                    self.no_damage_after_play += 1
            if player_hp is not None and player_hp_after is not None:
                if player_hp_after < player_hp - 0.005:
                    self.player_hp_drop_after_play += 1
            if (
                enemy_hp is not None
                and enemy_hp_after is not None
            ):
                if enemy_hp_after >= enemy_hp - 0.005:
                    self.ineffective_plays += 1
        if (
            action.card_group in SPELL_GROUPS
            and enemy_hp is not None
            and enemy_hp_after is not None
        ):
            if enemy_hp_after < enemy_hp - 0.005:
                self.effective_spells += 1
            else:
                if enemy_activity is not None and enemy_activity_after is not None:
                    if enemy_activity_after <= enemy_activity + 0.005:
                        self.ineffective_spells += 1
                else:
                    self.ineffective_spells += 1
            if action.card_group == "big_win_con":
                if (enemy_activity or 0.0) < 0.05:
                    self.wincon_when_clear += 1
                if action.coord[1] >= 340:
                    self.wincon_back += 1
                if elixir_now is not None and elixir_now < 4.5:
                    self.wincon_early += 1
                if (
                    enemy_hp is not None
                    and enemy_hp_after is not None
                ):
                    if enemy_hp_after < enemy_hp - 0.005:
                        self.wincon_effective += 1
            # Discourage bridge spam when the board is calm.
            if action.card_group not in FORWARD_ALLOWED_GROUPS and (enemy_activity or 0.0) < 0.05:
                if action.coord[1] < 320:
                    self.forward_clear += 1
                else:
                    self.back_clear += 1
            if (
                action.card_group in SPELL_GROUPS
                and (enemy_activity or 0.0) < 0.05
            ):
                self.spell_calm_spam += 1
            # Accumulate per-tower damage (negative deltas mean damage taken).
            for delta in (enemy_left_delta, enemy_right_delta, enemy_king_delta):
                if delta is not None and delta < 0:
                    self.enemy_tower_damage += (-delta)
            for delta in (player_left_delta, player_right_delta, player_king_delta):
                if delta is not None and delta < 0:
                    self.player_tower_damage += (-delta)
            # Track lane-bias adherence for attack cards.
            if lane_bias is not None and action_side is not None and (enemy_activity or 0.0) < 0.05:
                if action.card_group == "big_win_con":
                    if (lane_bias < 0.5 and action_side == "left") or (lane_bias >= 0.5 and action_side == "right"):
                        self.attack_on_bias += 1
                    else:
                        self.attack_off_bias += 1
                if action.coord[1] < 320 and action.card_group not in SPELL_GROUPS:
                    if (lane_bias < 0.5 and action_side == "left") or (lane_bias >= 0.5 and action_side == "right"):
                        self.attack_on_target += 1
                    else:
                        self.attack_off_target += 1
            # Reward "stacked" defense placements (drop-on-unit) for certain defenders.
            if action.card_id in {"knight", "mini_pekka", "goblins"}:
                if (enemy_activity or 0.0) > 0.05 and action.coord[1] >= 360:
                    self.stacked_defense += 1
            # Card cycle memory (avoid repeating same card too quickly).
            if card_id and elapsed_time is not None:
                last_time = self._last_play_time_by_card.get(card_id)
                if last_time is not None:
                    delta = elapsed_time - last_time
                    if delta < 6.0:
                        self.card_repeat_fast += 1
                    elif delta >= 8.0:
                        self.card_cycle_reward += 1
                self._last_play_time_by_card[card_id] = elapsed_time

    def _compute_shaped_reward(self, base: float, crowns: tuple[int, int] | None) -> float:
        reward = base
        if crowns:
            player, opp = crowns
            reward += 0.35 * (player - opp)
            if player >= 3:
                reward += 0.6
            if opp >= 3:
                reward -= 0.6

        # Encourage fewer cards (proxy for elixir efficiency).
        reward -= 0.01 * self.action_count

        # Penalize spell usage (proxy for \"spell at nothing\" without enemy detection).
        reward -= 0.04 * self.spell_count

        # Penalize high average elixir at play time (avoid elixir leak),
        # but allow a small bias toward saving a bit of elixir.
        if self.elixir_samples > 0:
            avg_elixir = self.elixir_sum / self.elixir_samples
            reward -= 0.015 * avg_elixir
            # Mild reward for holding a modest buffer (roughly 4-7 elixir).
            reward += min(0.06, max(0.0, avg_elixir - 4.0) * 0.015)
            if avg_elixir >= 8:
                reward -= 0.05 * max(0, self.hold_count - 1)

        # Encourage card diversity across the hand slots.
        unique_cards = len(self.used_indices)
        reward += 0.03 * unique_cards
        reward -= 0.02 * max(0, self.action_count - unique_cards)

        # Penalize repeated same-slot spam.
        reward -= 0.02 * max(0, self.repeat_streak)

        # Penalize playing at very low elixir.
        reward -= 0.12 * self.low_elixir_plays

        # Penalize invalid placements (tap didn't deploy a card).
        reward -= 0.15 * self.invalid_placements
        reward -= 0.05 * self.clamped_placements
        reward -= 0.1 * self.edge_placements

        # Encourage back placement early, discourage forward placement early.
        reward += 0.02 * self.early_back
        reward -= 0.05 * self.early_forward

        # Reward estimated enemy HP reduction (best-effort).
        reward += 1.0 * self.enemy_damage
        reward -= 1.2 * self.player_damage

        # Encourage defense when enemy activity is detected.
        reward += 0.18 * self.defensive_plays
        reward += 0.01 * self.offensive_when_clear
        reward += 0.04 * self.hold_when_clear
        reward -= 0.08 * self.hold_when_clear_high
        reward -= 0.06 * self.calm_elixir_spend

        # Penalize "bad move" proxies shortly after a play.
        reward -= 0.14 * self.no_damage_after_play
        reward -= 0.22 * self.player_hp_drop_after_play
        reward -= 0.12 * self.ineffective_plays

        # Reward effective spell usage (HP reduction).
        reward += 0.18 * self.effective_spells
        reward -= 0.28 * self.ineffective_spells

        # Strongly discourage holding at or near elixir cap.
        reward -= 0.12 * self.overcap_holds

        # Win condition shaping: reward safe, back-lane win-cons.
        reward += 0.18 * self.wincon_effective
        reward += 0.08 * self.wincon_when_clear
        reward += 0.06 * self.wincon_back
        reward -= 0.10 * self.wincon_early
        reward += 0.08 * self.hold_for_wincon

        # Reduce bridge spam when the board is calm.
        reward -= 0.20 * self.forward_clear
        reward += 0.03 * self.back_clear
        reward += 0.08 * self.counter_push_chain

        # Tower damage shaping (use per-tower deltas when available).
        reward += 0.6 * self.enemy_tower_damage
        reward -= 0.7 * self.player_tower_damage

        # Lane-bias adherence for attack cards.
        reward += 0.05 * self.attack_on_bias
        reward -= 0.06 * self.attack_off_bias
        reward += 0.08 * self.attack_on_target
        reward -= 0.10 * self.attack_off_target

        # Defend vs attack mode alignment.
        # Spell spam penalty when board is calm.
        reward -= 0.16 * self.spell_calm_spam

        # Card cycle shaping (discourage rapid repeats).
        reward -= 0.08 * self.card_repeat_fast
        reward += 0.04 * self.card_cycle_reward
        reward += 0.12 * self.stacked_defense
        return reward

    def _reset_stats(self) -> None:
        self.action_count = 0
        self.spell_count = 0
        self.elixir_sum = 0.0
        self.elixir_samples = 0
        self.used_indices.clear()
        self.card_index_counts = [0, 0, 0, 0]
        self.last_card_index = None
        self.repeat_streak = 0
        self.hold_count = 0
        self.low_elixir_plays = 0
        self.invalid_placements = 0
        self.clamped_placements = 0
        self.edge_placements = 0
        self.early_forward = 0
        self.early_back = 0
        self.last_crowns = None
        self.prev_enemy_hp = None
        self.prev_player_hp = None
        self.enemy_damage = 0.0
        self.player_damage = 0.0
        self.enemy_activity_samples = 0
        self.enemy_activity_present = 0
        self.defensive_plays = 0
        self.offensive_when_clear = 0
        self.hold_when_clear = 0
        self.no_damage_after_play = 0
        self.player_hp_drop_after_play = 0
        self.ineffective_plays = 0
        self.effective_spells = 0
        self.overcap_holds = 0
        self.wincon_when_clear = 0
        self.wincon_back = 0
        self.wincon_early = 0
        self.wincon_effective = 0
        self.hold_for_wincon = 0
        self.forward_clear = 0
        self.back_clear = 0
        self.enemy_tower_damage = 0.0
        self.player_tower_damage = 0.0
        self.attack_on_bias = 0
        self.attack_off_bias = 0
        self.attack_on_target = 0
        self.attack_off_target = 0
        self.spell_calm_spam = 0
        self.ineffective_spells = 0
        self.card_repeat_fast = 0
        self.card_cycle_reward = 0
        self.counter_push_chain = 0
        self.calm_elixir_spend = 0.0
        self.stacked_defense = 0
        self._last_play_time_by_card = {}
        self._last_defensive_elapsed = None
        self._last_defensive_side = None

    def update(self, reward: float, crowns: tuple[int, int] | None = None) -> None:
        if not self.buffer or self.model is None or torch is None or self.optimizer is None or self.loss_fn is None:
            self.buffer.clear()
            self._reset_stats()
            return
        shaped = self._compute_shaped_reward(reward, crowns)
        shaped *= REWARD_SCALE
        self.model.train()
        inputs = torch.tensor(self.buffer, dtype=torch.float32)
        targets = torch.full((inputs.shape[0], 1), float(shaped), dtype=torch.float32)
        outputs = self.model(inputs)
        if outputs.dim() == 1:
            outputs = outputs.view(-1, 1)
        loss = self.loss_fn(outputs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        if self.logger:
            self.logger.log(
                f"[Policy] Online update complete. Reward={shaped:.3f} (base={reward:.3f}). Loss={loss.item():.4f}"
            )
            try:
                self.logger.add_policy_progress(shaped, loss.item())
            except Exception:
                pass
        save_torch_model(self.model, self.model_path, logger=self.logger)
        self.buffer.clear()
        self._reset_stats()
