"""random module for randomizing fight plays"""

import collections
import os
import random
import time
from contextlib import suppress
from typing import Literal

import cv2
import numpy as np

from towerlogic.bot.card_detection import (
    check_which_cards_are_available,
    create_default_bridge_iar,
    get_hand_card_crop,
    get_hand_card_predictions,
    HAND_USE_CLASSIFIER,
    get_hand_template_scores,
    get_card_group,
    get_all_pixel_data,
    toplefts,
    TOTAL_HEIGHT,
    TOTAL_WIDTH,
    get_play_coords_for_card,
    identify_hand_cards,
    set_active_deck_filter,
    switch_side,
)
from towerlogic.bot.policy import (
    EpsilonGreedyPolicy,
    OnlineTrainer,
    PolicyAction,
    SPELL_GROUPS,
    FORWARD_ALLOWED_GROUPS,
    REWARD_SCALE,
    build_state_vector,
    candidate_coords_for_group,
    candidate_coords_free,
    hold_action,
    clamp_coord,
    is_near_edge,
    safe_bounds_for_image,
    load_torch_model,
)
from towerlogic.bot.constants import CLASH_MAIN_DEADSPACE_COORD
from towerlogic.bot.nav import (
    check_for_in_battle_with_delay,
    check_for_trophy_reward_menu,
    check_if_battle_has_ended,
    check_if_in_battle,
    check_if_on_clash_main_menu,
    get_to_activity_log,
    handle_trophy_reward_menu,
    wait_for_battle_start,
    wait_for_clash_main_menu,
)
from towerlogic.bot.recorder import save_play, save_win_loss
from towerlogic.detection.image_rec import (
    check_line_for_color,
    find_image,
    pixel_is_equal,
)

FIELD_DET_MIN_CONF = float(os.getenv("PYCLASHBOT_FIELD_MIN_CONF", "0.30"))


def _detect_field_with_roi(detector, snapshot, bounds):
    if detector is None or snapshot is None:
        return []
    x_min, x_max, y_min, y_max = bounds
    if x_max <= x_min or y_max <= y_min:
        return detector.detect(snapshot)
    # Slight margin to avoid clipping troops near the edges.
    pad_x = max(2, int((x_max - x_min) * 0.02))
    pad_y = max(2, int((y_max - y_min) * 0.02))
    x1 = max(0, x_min - pad_x)
    x2 = min(snapshot.shape[1], x_max + pad_x)
    y1 = max(0, y_min - pad_y)
    y2 = min(snapshot.shape[0], y_max + pad_y)
    roi = snapshot[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return detector.detect(snapshot)
    detections = detector.detect(roi)
    filtered = []
    for det in detections:
        # Filter tiny boxes (noise) and low confidence.
        if det.conf < FIELD_DET_MIN_CONF:
            continue
        if (det.x2 - det.x1) < 4 or (det.y2 - det.y1) < 4:
            continue
        det.x1 += x1
        det.x2 += x1
        det.y1 += y1
        det.y2 += y1
        filtered.append(det)
    return filtered



from towerlogic.detection.onnx_detector import (
    load_detector_from_env,
    split_detections_by_side,
    summarize_detections,
)
from towerlogic.utils.cancellation import interruptible_sleep
from towerlogic.utils.logger import Logger, log_dir

CLOSE_BATTLE_LOG_BUTTON: tuple[Literal[365], Literal[72]] = (365, 72)
# coords of the cards in the hand
HAND_CARDS_COORDS = [
    (142, 561),
    (210, 563),
    (272, 561),
    (341, 563),
]
CLOSE_THIS_CHALLENGE_PAGE_BUTTON = (27, 22)

QUICKMATCH_BUTTON_COORD = (
    274,
    353,
)  # coord of the quickmatch button after you click the battle button
ELIXER_WAIT_TIMEOUT = 40  # way to high but someone got errors with that so idk

ELIXIR_COORDS = [
    [613, 149],
    [613, 165],
    [613, 188],
    [613, 212],
    [613, 240],
    [613, 262],
    [613, 287],
    [613, 314],
    [613, 339],
    [613, 364],
]
ELIXIR_COLOR = [240, 137, 244]
HP_PROBE_FLAG = os.path.join(log_dir, "hp_probe.flag")
HAND_PROBE_FLAG = os.path.join(log_dir, "hand_probe.flag")
# Fixed player HP box shift (multiples of box height). Keeps the box stable.
FIXED_PLAYER_HP_SHIFT = -0.1
CARD_COSTS = {
    "archers": 3,
    "giant": 5,
    "goblin_cage": 4,
    "knight": 3,
    "mini_pekka": 4,
    "minions": 3,
    "wizard": 5,
    "skeletons": 1,
    "valkyrie": 4,
    # Keep extras for safety if detected.
    "goblins": 2,
    "arrows": 3,
}
AIR_UNIT_IDS = {
    "baby_dragon",
    "balloon",
    "bats",
    "electro_dragon",
    "flying_machine",
    "inferno_dragon",
    "lava_hound",
    "mega_minion",
    "minions",
    "minion_horde",
    "phoenix",
    "skeleton_dragons",
}
GROUND_ONLY_DEFENDERS = {"mini_pekka", "knight", "goblins", "valkyrie", "skeletons"}


def _get_player_hp_shift_mult() -> float:
    return FIXED_PLAYER_HP_SHIFT


def do_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode,
    fight_mode_choosed,
    called_from_launching=False,
    recording_flag: bool = False,
    policy_cfg: dict | None = None,
) -> bool:
    """Handle the entirety of a battle state (start fight, do fight, end fight)."""

    logger.change_status("do_fight_state state")
    logger.change_status("Waiting for battle to start")

    # Wait for battle start
    if wait_for_battle_start(emulator, logger) is False:
        logger.change_status(
            "Error waiting for battle to start in do_fight_state()",
        )
        return False

    logger.change_status("Starting fight loop")
    logger.log(f'This is the fight mode: "{fight_mode_choosed}"')

    # Run regular fight loop if random mode not toggled
    if not random_fight_mode and _fight_loop(emulator, logger, recording_flag, policy_cfg) is False:
        logger.change_status("Failure in fight loop")
        return False

    # Run random fight loop if random mode toggled
    if random_fight_mode and _random_fight_loop(emulator, logger) is False:
        logger.change_status("Failure in fight loop")
        return False

    # Only log the fight if not called from the start
    if not called_from_launching:
        if fight_mode_choosed in ["Classic 1v1", "Trophy Road"]:
            logger.add_1v1_fight()
        elif fight_mode_choosed == "Classic 2v2":
            logger.increment_2v2_fights()

        if fight_mode_choosed == "Trophy Road":
            logger.increment_trophy_road_fights()
        elif fight_mode_choosed == "Classic 1v1":
            logger.increment_classic_1v1_fights()
        elif fight_mode_choosed == "Classic 2v2":
            logger.increment_classic_2v2_fights()

    interruptible_sleep(10)
    return True


def do_2v2_fight_state(
    emulator,
    logger: Logger,
    random_fight_mode,
    recording_flag: bool = False,
    policy_cfg: dict | None = None,
) -> bool:
    """Handle the entirety of the 2v2 battle state (start fight, do fight, end fight)."""
    # Use the same fight logic as 1v1, just with 2v2 mode
    return do_fight_state(
        emulator,
        logger,
        random_fight_mode,
        "Classic 2v2",
        called_from_launching=False,
        recording_flag=recording_flag,
        policy_cfg=policy_cfg,
    )


def start_fight(emulator, logger, mode) -> bool:
    """Start a fight with the specified mode.

    Args:
        emulator: The emulator controller
        logger: Logger instance
        mode: Fight mode - must be one of "Classic 1v1", "Classic 2v2", or "Trophy Road"

    Returns:
        bool: True if fight started successfully, False otherwise
    """
    # Validate mode parameter
    logger.log(f'Input mode type: "{type(mode)}"')
    logger.log(f"Input mode value: {mode}")
    valid_modes = ["Classic 1v1", "Classic 2v2", "Trophy Road"]
    logger.log(f"Valid modes: {valid_modes}")
    if mode not in valid_modes:
        logger.log(f"The valid modes for start_fight() are: {valid_modes}")
        logger.log(f"But start_fight() got an invalid mode: '{mode}'")
        return False

    logger.change_status(f"Starting a {mode} fight")

    # Check if on clash main menu
    logger.log("Checking if on clash main before starting fight...")
    if not check_if_on_clash_main_menu(emulator):
        logger.change_status("Not on clash main menu, cannot start fight")
        return False

    # For all modes (1v1 and 2v2), use the same start button
    # Mode is already set by select_mode() in states.py, just click start button
    emulator.click(203, 487)
    logger.log("Clicked Start button at (203, 487)")

    # if its 2v2 mode, we gotta click that second popup
    if mode == "Classic 2v2":
        logger.change_status("Its 2v2 mode so we gotta click the quickmatch popup option!")
        interruptible_sleep(3)
        quick_match_button_coord = [280, 350]
        emulator.click(quick_match_button_coord[0], quick_match_button_coord[1])
        logger.log(f"Clicked Quickmatch button at {quick_match_button_coord}")

    return True


def mag_dump(emulator, logger):
    card_coords = [
        (137, 559),
        (206, 559),
        (274, 599),
        (336, 555),
    ]

    logger.log("Mag dumping...")
    for index in range(3):
        logger.change_status(f"mag dump play {index}")
        card_index = random.randint(0, 3)
        card_coord = card_coords[card_index]
        play_coord = (random.randint(101, 440), random.randint(50, 526))

        # record play here

        emulator.click(card_coord[0], card_coord[1])
        interruptible_sleep(0.1)

        emulator.click(play_coord[0], play_coord[1])
        interruptible_sleep(0.1)


def wait_for_elixer(
    emulator,
    logger,
    random_elixer_wait,
    WAIT_THRESHOLD=5000,  # noqa: N803
    PLAY_THRESHOLD=10000,  # noqa: N803
    recording_flag: bool = False,
) -> Literal["restart", "no battle"] | bool:
    """Method to wait for 4 elixir during a battle"""
    start_time = time.time()
    battle_detection_lost_count = 0

    while not count_elixer(emulator, random_elixer_wait):
        # debug screenshot saving removed from production
        wait_time = time.time() - start_time
        logger.change_status(
            f"Waiting for {random_elixer_wait} elixir for {str(wait_time)[:4]}s...",
        )

        card_inhand = len(check_which_cards_are_available(emulator, True, False))
        action_offset, _ = switch_side()
        if action_offset > PLAY_THRESHOLD and card_inhand > 0:
            logger.change_status("Too much going on, playing now")
            return True

        if action_offset > WAIT_THRESHOLD and card_inhand == 4:
            logger.change_status("All cards are available!")
            return True

        if wait_time > ELIXER_WAIT_TIMEOUT:
            logger.change_status(status="Waited too long for elixir")
            return "restart"

        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status(status="Not in battle anymore (confirmed), stopping waiting for elixir.")
                return "no battle"

            battle_detection_lost_count += 1
            logger.change_status(
                status="Lost battle detection while waiting for elixir; assuming still in battle.",
            )
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    status="Lost battle detection repeatedly while waiting for elixir; assuming battle ended.",
                )
                return "no battle"

            interruptible_sleep(0.5)
            continue

        battle_detection_lost_count = 0

    logger.change_status(
        f"Took {str(time.time() - start_time)[:4]}s for {random_elixer_wait} elixir.",
    )

    return True


def count_elixer(emulator, elixer_count) -> bool:
    """Method to check for 4 elixir during a battle"""
    iar = emulator.screenshot()

    if pixel_is_equal(
        iar[ELIXIR_COORDS[elixer_count - 1][0], ELIXIR_COORDS[elixer_count - 1][1]],
        ELIXIR_COLOR,
        tol=65,
    ):
        return True
    return False


def estimate_elixir(emulator) -> int:
    for amount in range(10, 0, -1):
        if count_elixer(emulator, amount):
            return amount
    return 0


def end_fight_state(
    emulator,
    logger: Logger,
    recording_flag,
    disable_win_tracker_toggle=True,
):
    """Method to handle the time after a fight and before the next state"""
    # count the crown score on this end-battle screen (best-effort)

    # get to clash main after this fight
    logger.log("Getting to clash main after doing a fight")
    if get_to_main_after_fight(emulator, logger) is False:
        logger.log("Error 69a3d69 Failed to get to clash main after a fight")
        return False

    logger.log("Made it to clash main after doing a fight")
    interruptible_sleep(3)

    # check if the prev game was a win
    if not disable_win_tracker_toggle:
        win_check_return = check_if_previous_game_was_win(emulator, logger)

        if win_check_return == "restart":
            logger.log("Error 885869 Failed while checking if previous game was a win")
            return False

        if win_check_return:
            logger.add_win()

            if recording_flag:
                save_win_loss("win")
            trainer = getattr(logger, "policy_trainer", None)
            if trainer is not None:
                crowns = getattr(logger, "last_crown_result", None)
                trainer.update(1.0, crowns=crowns)
            return True

        logger.add_loss()
        if recording_flag:
            save_win_loss("loss")
        trainer = getattr(logger, "policy_trainer", None)
        if trainer is not None:
            crowns = getattr(logger, "last_crown_result", None)
            trainer.update(-1.0, crowns=crowns)
    else:
        logger.log("Not checking win/loss because check is disabled")

    return True


def check_if_previous_game_was_win(
    emulator,
    logger: Logger,
) -> bool | Literal["restart"]:
    """Method to handle the checking if the previous game was a win or loss"""
    logger.change_status(status="Checking if last game was a win/loss")

    # Use wait_for_clash_main_menu to ensure we are on the main menu.
    if not wait_for_clash_main_menu(emulator, logger, deadspace_click=True):
        logger.change_status(status='Error Not on main menu, returning "restart"')
        return "restart"

    # get to clash main options menu
    if get_to_activity_log(emulator, logger, printmode=False) == "restart":
        logger.change_status(
            status="Error 8967203948 get_to_activity_log() in check_if_previous_game_was_win()",
        )

        return "restart"

    logger.change_status(status="Checking if last game was a win...")
    is_a_win = check_pixels_for_win_in_battle_log(emulator)
    logger.change_status(status=f"Last game is win: {is_a_win}")

    # close battle log
    logger.change_status(status="Returning to clash main")
    emulator.click(CLOSE_BATTLE_LOG_BUTTON[0], CLOSE_BATTLE_LOG_BUTTON[1])
    if wait_for_clash_main_menu(emulator, logger) is False:
        logger.change_status(
            status="Error 95867235 wait_for_clash_main_menu() in check_if_previous_game_was_win()",
        )
        return "restart"
    interruptible_sleep(2)

    return is_a_win


def check_pixels_for_win_in_battle_log(emulator) -> bool:
    """Method to check pixels that appear in the battle
    log to determing if the previous game was a win
    """
    line1 = check_line_for_color(
        emulator,
        x_1=47,
        y_1=135,
        x_2=109,
        y_2=154,
        color=(255, 51, 102),
    )
    line2 = check_line_for_color(
        emulator,
        x_1=46,
        y_1=152,
        x_2=115,
        y_2=137,
        color=(255, 51, 102),
    )
    line3 = check_line_for_color(
        emulator,
        x_1=47,
        y_1=144,
        x_2=110,
        y_2=147,
        color=(255, 51, 102),
    )

    if line1 and line2 and line3:
        return False
    return True


def find_post_battle_button(emulator):
    """Find and return coordinates for post-battle exit/OK button.

    Tries multiple detection methods in order:
    1. Pixel-based detection (fastest)
    2. Image recognition for OK button
    3. Image recognition for exit button

    Returns:
        tuple[int, int] | None: Button coordinates (x, y) or None if not found
    """
    iar = emulator.screenshot()

    # Method 1: Fast pixel-based detection
    pixels = [
        iar[545][178],
        iar[547][239],
        iar[553][214],
        iar[554][201],
    ]
    colors = [
        [255, 187, 104],
        [255, 187, 104],
        [255, 255, 255],
        [255, 255, 255],
    ]

    pixel_match = True
    for i, p in enumerate(pixels):
        if not pixel_is_equal(p, colors[i], tol=20):
            pixel_match = False
            break

    if pixel_match:
        return (200, 550)

    # Method 2: Image recognition for OK button
    coord = find_image(iar, "ok_post_battle_button", tolerance=0.85)
    if coord is not None:
        return coord

    # Method 3: Image recognition for exit button
    coord = find_image(iar, "exit_battle_button", tolerance=0.9)
    if coord is not None:
        return coord

    return None


def _count_distinct_by_x(centers: list[tuple[int, int]], min_dx: int = 12) -> int:
    if not centers:
        return 0
    xs = sorted(c[0] for c in centers)
    count = 1
    last = xs[0]
    for x in xs[1:]:
        if abs(x - last) >= min_dx:
            count += 1
            last = x
    return count


def detect_crown_counts_from_end_screen(image: np.ndarray) -> tuple[int, int] | None:
    """Best-effort crown detection from the end-of-battle screen.

    Returns:
        (player_crowns, opponent_crowns) or None if detection fails.
    """
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.2), int(w * 0.8)
    y_min, y_max = int(h * 0.08), int(h * 0.6)
    roi = image[y_min:y_max, x_min:x_max]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 80, 150])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: list[tuple[int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if 50 <= area <= 2000 and 6 <= cw <= 60 and 6 <= ch <= 60:
            centers.append((x + cw // 2, y + ch // 2))

    if len(centers) < 2:
        return None

    ys = sorted(c[1] for c in centers)
    median_y = ys[len(ys) // 2]
    top = [c for c in centers if c[1] < median_y]
    bottom = [c for c in centers if c[1] >= median_y]
    if not top or not bottom:
        return None

    opp = _count_distinct_by_x(top)
    player = _count_distinct_by_x(bottom)
    return player, opp


def detect_crown_counts_in_battle(image: np.ndarray) -> tuple[int, int] | None:
    """Best-effort crown detection during battle from scoreboard area."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.25), int(w * 0.75)
    y_min, y_max = int(h * 0.02), int(h * 0.12)
    roi = image[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 80, 150])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: list[tuple[int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if 20 <= area <= 1200 and 4 <= cw <= 45 and 4 <= ch <= 45:
            centers.append((x + cw // 2, y + ch // 2))
    if len(centers) < 2:
        return None
    ys = sorted(c[1] for c in centers)
    median_y = ys[len(ys) // 2]
    top = [c for c in centers if c[1] < median_y]
    bottom = [c for c in centers if c[1] >= median_y]
    if not top or not bottom:
        return None
    opp = _count_distinct_by_x(top, min_dx=10)
    player = _count_distinct_by_x(bottom, min_dx=10)
    return player, opp


def _hp_mask(hsv: np.ndarray, mode: str) -> np.ndarray:
    if mode == "enemy":
        lower1 = np.array([0, 70, 60])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 60])
        upper2 = np.array([179, 255, 255])
        return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    if mode == "player":
        lower = np.array([90, 60, 60])
        upper = np.array([140, 255, 255])
        return cv2.inRange(hsv, lower, upper)
    lower = np.array([35, 60, 80])
    upper = np.array([85, 255, 255])
    return cv2.inRange(hsv, lower, upper)


def _estimate_color_bar_ratio(roi: np.ndarray, mode: str) -> float | None:
    if roi is None or roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _hp_mask(hsv, mode)
    col_sum = mask.sum(axis=0)
    active = col_sum > 0
    if not np.any(active):
        return None
    max_run = 0
    cur = 0
    for v in active:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run / max(1, active.size)


def _find_hp_band(
    image: np.ndarray,
    mode: str,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
) -> tuple[int, int] | None:
    if image is None or image.size == 0:
        return None
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _hp_mask(hsv, mode)
    row_ratio = (mask > 0).mean(axis=1)

    # Prefer thin horizontal bands (tower HP bars), avoid wide blue/red areas.
    min_ratio = 0.01
    max_ratio = 0.35 if mode == "player" else 0.50
    candidates = (row_ratio >= min_ratio) & (row_ratio <= max_ratio)
    if not np.any(candidates):
        candidates = row_ratio >= min_ratio
    if not np.any(candidates):
        return None

    # Find best contiguous run: score by mean ratio * length, with slight bias
    # toward the top of the search window (HP bars sit above most UI chrome).
    best_start = None
    best_end = None
    best_score = 0.0
    start = None
    total_rows = max(len(row_ratio) - 1, 1)
    for i, ok in enumerate(candidates):
        if ok and start is None:
            start = i
        if (not ok or i == len(candidates) - 1) and start is not None:
            end = i if ok else i - 1
            run = row_ratio[start : end + 1]
            mid = (start + end) / 2.0
            pos_weight = 1.0 - (mid / total_rows)
            score = float(run.mean()) * (end - start + 1) * (0.5 + pos_weight)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
            start = None
    if best_start is None or best_end is None:
        return None

    band_min = max(y_min + int(best_start) - 1, 0)
    band_max = min(y_min + int(best_end) + 2, image.shape[0])
    return band_min, band_max


def estimate_enemy_hp_ratio(image: np.ndarray) -> float | None:
    """Best-effort enemy tower HP estimate from top UI bar."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.12), int(w * 0.88)
    y_min, y_max = int(h * 0.05), int(h * 0.22)
    band = _find_hp_band(image, "enemy", y_min, y_max, x_min, x_max)
    if band:
        y_min, y_max = band
    roi = image[y_min:y_max, x_min:x_max]
    return _estimate_color_bar_ratio(roi, "enemy")


def estimate_player_hp_ratio(image: np.ndarray) -> float | None:
    """Best-effort player tower HP estimate from bottom UI bar."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.12), int(w * 0.88)
    y_min, y_max = int(h * 0.56), int(h * 0.72)
    band_h = max(y_max - y_min, 1)
    shift = band_h * _get_player_hp_shift_mult()
    y_min = max(0, int(y_min + shift))
    y_max = max(1, int(y_max + shift))
    band = _find_hp_band(image, "player", y_min, y_max, x_min, x_max)
    if band:
        y_min, y_max = band
    roi = image[y_min:y_max, x_min:x_max]
    return _estimate_color_bar_ratio(roi, "player")


def estimate_enemy_tower_hps(
    image: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Estimate enemy (left, right, king) tower HP ratios (best-effort)."""
    if image is None or image.size == 0:
        return None, None, None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.12), int(w * 0.88)
    y_min, y_max = int(h * 0.05), int(h * 0.22)
    band = _find_hp_band(image, "enemy", y_min, y_max, x_min, x_max)
    if band:
        y_min, y_max = band
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None, None
    mid = roi.shape[1] // 2
    left = _estimate_color_bar_ratio(roi[:, :mid], "enemy")
    right = _estimate_color_bar_ratio(roi[:, mid:], "enemy")
    # King tower bar is usually centered and slightly lower.
    kx_min, kx_max = int(w * 0.40), int(w * 0.60)
    ky_min, ky_max = int(h * 0.18), int(h * 0.24)
    king_roi = image[ky_min:ky_max, kx_min:kx_max]
    king = _estimate_color_bar_ratio(king_roi, "enemy")
    return left, right, king


def estimate_player_tower_hps(
    image: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """Estimate player (left, right, king) tower HP ratios (best-effort)."""
    if image is None or image.size == 0:
        return None, None, None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.12), int(w * 0.88)
    y_min, y_max = int(h * 0.56), int(h * 0.72)
    band_h = max(y_max - y_min, 1)
    shift = band_h * _get_player_hp_shift_mult()
    y_min = max(0, int(y_min + shift))
    y_max = max(1, int(y_max + shift))
    band = _find_hp_band(image, "player", y_min, y_max, x_min, x_max)
    if band:
        y_min, y_max = band
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None, None
    mid = roi.shape[1] // 2
    left = _estimate_color_bar_ratio(roi[:, :mid], "player")
    right = _estimate_color_bar_ratio(roi[:, mid:], "player")
    # King tower bar is usually centered and slightly higher.
    kx_min, kx_max = int(w * 0.40), int(w * 0.60)
    ky_min, ky_max = int(h * 0.68), int(h * 0.74)
    king_roi = image[ky_min:ky_max, kx_min:kx_max]
    king = _estimate_color_bar_ratio(king_roi, "player")
    return left, right, king


def detect_enemy_activity(image: np.ndarray, prev_image: np.ndarray | None) -> float | None:
    """Detect enemy-side activity using frame differencing on the top half of the arena."""
    if image is None or image.size == 0 or prev_image is None or prev_image.size == 0:
        return None
    h, w = image.shape[:2]
    # Arena ROI (avoid UI)
    x_min, x_max = int(w * 0.10), int(w * 0.90)
    y_min, y_max = int(h * 0.18), int(h * 0.78)
    roi = image[y_min:y_max, x_min:x_max]
    prev_roi = prev_image[y_min:y_max, x_min:x_max]
    if roi.size == 0 or prev_roi.size == 0:
        return None
    # Enemy side = top half of arena ROI
    mid = roi.shape[0] // 2
    cur = cv2.cvtColor(roi[:mid], cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev_roi[:mid], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(cur, prev)
    motion = (diff > 22).astype(np.uint8)
    motion_ratio = float(motion.mean())
    return motion_ratio


def detect_enemy_activity_zone(image: np.ndarray, prev_image: np.ndarray | None) -> tuple[float | None, float | None]:
    """Detect enemy activity split into top (enemy side) and mid (near bridge)."""
    if image is None or image.size == 0 or prev_image is None or prev_image.size == 0:
        return None, None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.10), int(w * 0.90)
    y_min, y_max = int(h * 0.18), int(h * 0.78)
    roi = image[y_min:y_max, x_min:x_max]
    prev_roi = prev_image[y_min:y_max, x_min:x_max]
    if roi.size == 0 or prev_roi.size == 0:
        return None, None
    mid = roi.shape[0] // 2
    top_slice = roi[: mid // 2]
    mid_slice = roi[mid // 2 : mid]
    prev_top = prev_roi[: mid // 2]
    prev_mid = prev_roi[mid // 2 : mid]
    cur_top = cv2.cvtColor(top_slice, cv2.COLOR_BGR2GRAY)
    cur_mid = cv2.cvtColor(mid_slice, cv2.COLOR_BGR2GRAY)
    prev_top_g = cv2.cvtColor(prev_top, cv2.COLOR_BGR2GRAY)
    prev_mid_g = cv2.cvtColor(prev_mid, cv2.COLOR_BGR2GRAY)
    diff_top = cv2.absdiff(cur_top, prev_top_g)
    diff_mid = cv2.absdiff(cur_mid, prev_mid_g)
    motion_top = (diff_top > 22).astype(np.uint8).mean()
    motion_mid = (diff_mid > 22).astype(np.uint8).mean()
    return float(motion_top), float(motion_mid)


def detect_enemy_red_bar_activity(image: np.ndarray) -> float | None:
    """Detect enemy troop presence via red HP bar pixels on the top half."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.10), int(w * 0.90)
    y_min, y_max = int(h * 0.18), int(h * 0.55)
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Red range (two ranges in HSV)
    lower1 = np.array([0, 120, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 80])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(mask.mean())


def detect_enemy_red_bar_zone(image: np.ndarray) -> tuple[float | None, float | None]:
    """Detect enemy red bar presence split into top (enemy side) and mid (near bridge)."""
    if image is None or image.size == 0:
        return None, None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.10), int(w * 0.90)
    y_min, y_max = int(h * 0.18), int(h * 0.78)
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None
    mid = roi.shape[0] // 2
    top_slice = roi[: mid // 2]
    mid_slice = roi[mid // 2 : mid]
    hsv_top = cv2.cvtColor(top_slice, cv2.COLOR_BGR2HSV)
    hsv_mid = cv2.cvtColor(mid_slice, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 80])
    upper2 = np.array([180, 255, 255])
    mask_top = cv2.bitwise_or(cv2.inRange(hsv_top, lower1, upper1), cv2.inRange(hsv_top, lower2, upper2))
    mask_mid = cv2.bitwise_or(cv2.inRange(hsv_mid, lower1, upper1), cv2.inRange(hsv_mid, lower2, upper2))
    mask_top = cv2.morphologyEx(mask_top, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_mid = cv2.morphologyEx(mask_mid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(mask_top.mean()), float(mask_mid.mean())



def choose_target_side(
    enemy_left_hp: float | None,
    enemy_right_hp: float | None,
    threshold: float = 0.05,
) -> str | None:
    """Pick the weaker enemy tower side if there's a clear difference."""
    if enemy_left_hp is None and enemy_right_hp is None:
        return None
    if enemy_left_hp is None and enemy_right_hp is not None:
        return "right"
    if enemy_right_hp is None and enemy_left_hp is not None:
        return "left"
    if enemy_left_hp is None or enemy_right_hp is None:
        return None
    # If one tower is effectively gone, target the other.
    if enemy_left_hp <= 0.2 and enemy_right_hp > 0.2:
        return "right"
    if enemy_right_hp <= 0.2 and enemy_left_hp > 0.2:
        return "left"
    diff = enemy_left_hp - enemy_right_hp
    if abs(diff) < threshold:
        return None
    return "left" if enemy_left_hp < enemy_right_hp else "right"


def filter_coords_by_side(
    coords: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    if not coords:
        return coords
    x_min, x_max, _y_min, _y_max = bounds
    mid_x = (x_min + x_max) // 2
    if side == "left":
        filtered = [c for c in coords if c[0] <= mid_x]
    else:
        filtered = [c for c in coords if c[0] >= mid_x]
    return filtered or coords


def prefer_far_side_coords(coords: list[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    """Bias coords farther to the chosen side (reduce king pulls)."""
    if not coords:
        return coords
    sorted_coords = sorted(coords, key=lambda c: c[0])
    if side == "left":
        far = sorted_coords[: max(1, len(sorted_coords) // 2)]
    else:
        far = sorted_coords[-max(1, len(sorted_coords) // 2) :]
    return far or coords


def filter_coords_back(
    coords: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    frac: float = 0.6,
) -> list[tuple[int, int]]:
    """Prefer back-half coords (avoid bridge spam when the board is calm)."""
    if not coords:
        return coords
    _x_min, _x_max, y_min, y_max = bounds
    cut = int(y_min + (y_max - y_min) * frac)
    filtered = [c for c in coords if c[1] >= cut]
    return filtered or coords


def mask_actions_for_context(
    actions: list[PolicyAction],
    bounds: tuple[int, int, int, int],
    elixir: int,
    elapsed_time: float,
    prefer_attack_side: str | None,
    defend_mode: float,
    target_side: str | None,
    wincon_in_hand: bool,
    spell_gate: bool = False,
    enemy_activity: float = 0.0,
    enemy_present: bool = False,
    lane_enemy_left: bool = False,
    lane_enemy_right: bool = False,
    recent_enemy_spent: bool = False,
    lane_lock_active: bool = False,
    lane_lock_side: str | None = None,
    enemy_air_present: bool = False,
    enemy_ground_present: bool = True,
    enemy_has_hog: bool = False,
    enemy_has_giant: bool = False,
    enemy_has_battle_ram: bool = False,
    enemy_has_mini_pekka: bool = False,
    enemy_has_knight: bool = False,
    enemy_has_barbarians: bool = False,
    ally_giant_present: bool = False,
    lane_enemy_count_left: int = 0,
    lane_enemy_count_right: int = 0,
    last_wincon_ts: float | None = None,
    last_wincon_side: str | None = None,
) -> list[PolicyAction]:
    if not actions:
        return actions
    # Overcap guardrail: never allow hold at 8.0+ elixir.
    if elixir >= 8.0:
        non_hold = [a for a in actions if a.card_index >= 0]
        actions = non_hold or actions
    x_min, x_max, y_min, y_max = bounds
    mid_x = (x_min + x_max) // 2
    mid_low = int(y_min + (y_max - y_min) * 0.45)
    mid_high = int(y_min + (y_max - y_min) * 0.72)
    bridge_cut = mid_low
    skeleton_center_y = int(y_min + (y_max - y_min) * 0.55)
    filtered: list[PolicyAction] = []
    support_ids = {
        "wizard",
        "archers",
        "minions",
        "skeletons",
        "valkyrie",
    }
    wincon_actions = [a for a in actions if a.card_group == "big_win_con"]
    has_wincon = bool(wincon_actions)
    # Overtime logic disabled for now to avoid instability.
    overtime = False
    wincon_ready = has_wincon and elixir >= 7.0
    enemy_present = enemy_present or lane_enemy_left or lane_enemy_right
    calm_board = not enemy_present
    enemy_on_our_side = (lane_enemy_count_left + lane_enemy_count_right) > 0
    enemy_heavy_side = None
    if enemy_on_our_side:
        if lane_enemy_count_left > lane_enemy_count_right:
            enemy_heavy_side = "left"
        elif lane_enemy_count_right > lane_enemy_count_left:
            enemy_heavy_side = "right"
    # Smarter hold/leak policy.
    hold_threshold = 7.0
    leak_threshold = 9.5
    save_mode = calm_board and elixir < hold_threshold
    spell_ok_base = enemy_present or enemy_on_our_side
    if spell_gate:
        spell_ok = enemy_on_our_side
    else:
        spell_ok = spell_ok_base
    urgent_defense = enemy_on_our_side
    bank_for_wincon = calm_board and elixir < 8.5 and wincon_in_hand

    # Low-elixir guard: hold unless an urgent defense is required.
    if elixir < 5 and not urgent_defense and not enemy_on_our_side:
        hold_actions = [a for a in actions if a.card_index < 0]
        return hold_actions or actions

    # Reserve Mini P.E.K.K.A / Goblin Cage for defense: if enemy present and we have them,
    # hold until we can afford the play.
    if enemy_present and elixir < 4:
        if any(a.card_index >= 0 and a.card_id in {"mini_pekka", "goblin_cage"} for a in actions):
            hold_actions = [a for a in actions if a.card_index < 0]
            return hold_actions or actions

    # Save specific defenders until their trigger enemies are present.
    mini_pekka_trigger = enemy_has_hog or enemy_has_giant or enemy_has_battle_ram or enemy_has_knight
    cage_trigger = (
        enemy_has_hog
        or enemy_has_giant
        or enemy_has_battle_ram
        or enemy_has_barbarians
        or enemy_on_our_side
        or enemy_present
    )
    skeleton_trigger = enemy_has_mini_pekka or enemy_has_knight
    if not mini_pekka_trigger:
        actions = [a for a in actions if a.card_id != "mini_pekka"]
    if not cage_trigger:
        actions = [a for a in actions if a.card_id != "goblin_cage"]
    if not skeleton_trigger and not (enemy_on_our_side or enemy_present):
        actions = [a for a in actions if a.card_id != "skeletons"]

    # Hard hold rule: if calm and no enemies, save until the hold threshold.
    if calm_board and not enemy_present and elixir < hold_threshold:
        hold_actions = [a for a in actions if a.card_index < 0]
        return hold_actions or actions

    # Threat-first defense: if enemies are on your side, only allow defensive actions.
    if enemy_on_our_side:
        defensive_ids = {
            "mini_pekka",
            "goblin_cage",
            "skeletons",
            "knight",
            "valkyrie",
            "archers",
            "minions",
            "wizard",
            "arrows",
            "fireball",
        }
        def_actions = [a for a in actions if a.card_index < 0 or a.card_id in defensive_ids]
        if def_actions:
            filtered: list[PolicyAction] = []
            for a in def_actions:
                if a.card_index < 0:
                    continue
                if a.card_id in SPELL_GROUPS:
                    if spell_ok:
                        filtered.append(a)
                    continue
                if a.coord[1] < bridge_cut:
                    continue
                filtered.append(a)
            if enemy_heavy_side:
                side_filtered = [
                    a
                    for a in filtered
                    if (a.coord[0] <= mid_x) == (enemy_heavy_side == "left")
                ]
                filtered = side_filtered or filtered
            return filtered or def_actions

    # Fast push: if elixir is moderate and no enemy on board, play Giant closer to bridge.
    if elixir >= 7.0 and calm_board and not enemy_present and wincon_actions:
        fast_push = [a for a in wincon_actions if bridge_cut <= a.coord[1] < mid_high]
        if target_side:
            fast_push = [
                a for a in fast_push if (a.coord[0] <= mid_x) == (target_side == "left")
            ] or fast_push
        if fast_push:
            return fast_push

    # If no enemies on your side, only allow wincon (or leak-safe backline support).
    if not enemy_on_our_side and not enemy_present:
        safe_backline = [
            a
            for a in actions
            if a.card_index >= 0
            and a.coord[1] >= mid_high
            and a.card_id in support_ids
        ]
        gated = wincon_actions + (safe_backline if elixir >= leak_threshold else [])
        if gated:
            return gated

    # Overtime aggression: lower threshold to push.
    if overtime and wincon_actions and elixir >= 7 and not enemy_on_our_side:
        wincon_actions = [a for a in wincon_actions if a.coord[1] >= 320] or wincon_actions
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ] or wincon_actions
        return wincon_actions

    # Calm hold: if board is quiet, bank elixir under the hold threshold.
    if calm_board and not overtime and elixir < hold_threshold and not recent_enemy_spent:
        hold_actions = [a for a in actions if a.card_index < 0]
        return hold_actions or actions


    # Punish after enemy spent: if they just invested, push win-con.
    if recent_enemy_spent and wincon_actions and elixir >= 6 and not enemy_on_our_side:
        wincon_actions = [a for a in wincon_actions if a.coord[1] >= mid_high] or wincon_actions
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ] or wincon_actions
        return wincon_actions

    # Win-con bias when there is no immediate enemy presence.
    if elixir >= 8.5 and wincon_actions and not enemy_present:
        wincon_actions = [a for a in wincon_actions if a.coord[1] >= mid_high] or wincon_actions
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ] or wincon_actions
        return wincon_actions

    # Deploy order priority: don't lead with support/swarms before the tank.
    if calm_board and wincon_in_hand and elixir >= 8.5:
        actions = [
            a
            for a in actions
            if a.card_index < 0
            or a.card_group == "big_win_con"
        ]

    # Defend-before-push: if enemies are on your side, block forward pushes.
    if enemy_on_our_side:
        defensive_only = [
            a
            for a in actions
            if a.card_index < 0
            or a.card_group in SPELL_GROUPS
            or a.coord[1] >= mid_high
        ]
        if defensive_only:
            actions = defensive_only

    # Under heavy threat, only allow defensive back-half placements or spells.
    if urgent_defense:
        defensive = [
            a
            for a in actions
            if a.card_index < 0
            or a.card_group in SPELL_GROUPS
            or a.coord[1] >= 330
        ]
        return defensive or actions

    # At/near cap, force a win con if available to avoid leaking elixir.
    if elixir >= 8.5 and wincon_actions:
        wincon_actions = [a for a in wincon_actions if a.coord[1] >= mid_high]
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ]
        if wincon_actions:
            return wincon_actions
        non_hold = [a for a in actions if a.card_index >= 0]
        if non_hold:
            return non_hold

    # When calm and win condition is ready, prefer a back-lane win con (>= 8.5 elixir).
    if calm_board and wincon_ready and elixir >= 8.5:
        wincon_actions = [a for a in wincon_actions if a.coord[1] >= mid_high]
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ]
        if wincon_actions:
            return wincon_actions

    # Hard giant bias: if calm and we have the win-con ready, only allow Giant.
    if calm_board and wincon_in_hand and elixir >= 8.5 and wincon_actions:
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ] or wincon_actions
        return wincon_actions

    # Giant escort: after a back-lane win-con, prefer support troops behind it.
    if ally_giant_present and last_wincon_ts is not None and (elapsed_time - last_wincon_ts) <= 8.0:
        support_actions = [
            a
            for a in actions
            if a.card_id in support_ids and a.coord[1] >= mid_high
        ]
        if last_wincon_side is not None:
            support_actions = [
                a for a in support_actions if (a.coord[0] <= mid_x) == (last_wincon_side == "left")
            ]
        if support_actions:
            return support_actions

    # If we're sitting on a big elixir buffer, bias toward a win con.
    if elixir >= 8.5 and wincon_actions:
        wincon_actions = [a for a in wincon_actions if a.coord[1] >= mid_high] or wincon_actions
        if target_side:
            wincon_actions = [
                a for a in wincon_actions if (a.coord[0] <= mid_x) == (target_side == "left")
            ] or wincon_actions
        return wincon_actions

    # Bank elixir if win con is in hand but not ready.
    if bank_for_wincon:
        hold_actions = [a for a in actions if a.card_index < 0]
        return hold_actions or actions

    # If the board is mostly calm and elixir is low, prefer holding to save for the win con.
    if save_mode:
        hold_actions = [a for a in actions if a.card_index < 0]
        return hold_actions or actions
    for action in actions:
        if action.card_index < 0:
            # Avoid holding at cap when board is calm.
            if elixir >= 8.5 and calm_board:
                continue
            filtered.append(action)
            continue
        y = action.coord[1]
        action_side = "left" if action.coord[0] <= mid_x else "right"
        # Lane lock protection: avoid switching lanes shortly after committing.
        if lane_lock_active and lane_lock_side and action_side != lane_lock_side:
            if action.card_group not in SPELL_GROUPS and calm_board:
                continue
        if action.card_group in SPELL_GROUPS and not spell_ok:
            continue
        if action.card_id == "arrows":
            lane_has_enemy = lane_enemy_left if action_side == "left" else lane_enemy_right
            if not lane_has_enemy and not enemy_present:
                continue
        if action.card_group in SPELL_GROUPS:
            lane_has_enemy = lane_enemy_left if action_side == "left" else lane_enemy_right
            if not lane_has_enemy and not enemy_present:
                continue
            lane_count = lane_enemy_count_left if action_side == "left" else lane_enemy_count_right
            if lane_count < 2 and not enemy_on_our_side:
                continue
        if action.card_group == "big_win_con":
            if elixir < 7.0:
                continue
            if enemy_on_our_side:
                continue
            if y < mid_high and not (calm_board and elixir >= 7.0):
                continue
            if target_side and action_side != target_side:
                continue
            # Remember last win-con for escorting support.
            try:
                setattr(mask_actions_for_context, "_last_wincon_ts", elapsed_time)
                setattr(mask_actions_for_context, "_last_wincon_side", action_side)
            except Exception:
                pass
        # Calm board: discourage forward non-spell plays.
        if action.card_group not in FORWARD_ALLOWED_GROUPS and calm_board and y < 320:
            continue
        if action.card_id in {"wizard"}:
            if enemy_air_present and (abs(action.coord[0] - mid_x) > 40 or y < mid_high):
                continue
            if calm_board and y < 330:
                continue
        if action.card_id in {"archers"}:
            if y < bridge_cut and enemy_present:
                continue
            if enemy_present and (abs(action.coord[0] - mid_x) > 45 or y < mid_high):
                continue
        if action.card_id in {"goblin_cage"}:
            if y < mid_high or abs(action.coord[0] - mid_x) > 30:
                continue
        if action.card_id in {"minions"}:
            if y < bridge_cut and not enemy_air_present and not urgent_defense:
                continue
        if action.card_id in {"skeletons"}:
            if y < bridge_cut:
                continue
            # Never cycle skeletons unless enemies are present.
            if not enemy_present:
                continue
            if enemy_present and (abs(action.coord[0] - mid_x) > 45 or y < skeleton_center_y):
                continue
        if action.card_id in {"valkyrie"}:
            if enemy_present and (abs(action.coord[0] - mid_x) > 45 or y < mid_high):
                continue
        # Bridge block: no non-wincon plays at the bridge unless defending.
        if y < bridge_cut and action.card_group not in {"big_win_con"}:
            lane_has_enemy = lane_enemy_left if action_side == "left" else lane_enemy_right
            if not lane_has_enemy and not urgent_defense:
                continue
        # Split-lane defense: prefer center when enemies are on both sides.
        if lane_enemy_left and lane_enemy_right and action.card_group not in SPELL_GROUPS:
            if abs(action.coord[0] - mid_x) > 50 and y >= mid_high:
                continue
        # Lane commitment for win-con pushes when calm.
        if prefer_attack_side and calm_board and action.card_group == "big_win_con":
            if action_side != prefer_attack_side:
                continue
        filtered.append(action)

    # Hard overcap rule: if we're at/near 10 elixir and have any playable cards, don't allow hold.
    if elixir >= 9.0:
        non_hold = [a for a in filtered if a.card_index >= 0]
        if non_hold:
            # Anti-leak: prefer the cheapest non-spell cycle card when calm.
            if calm_board:
                non_spell = [a for a in non_hold if a.card_group not in SPELL_GROUPS]
                pool = [a for a in (non_spell or non_hold) if a.coord[1] >= mid_high]
                pool = pool or (non_spell or non_hold)
                if pool:
                    def _cost(a: PolicyAction) -> int:
                        return CARD_COSTS.get(a.card_id, 9)
                    min_cost = min(_cost(a) for a in pool)
                    return [a for a in pool if _cost(a) == min_cost]
            return non_hold
    return filtered or actions


def detect_enemy_templates(image: np.ndarray, region: str = "top") -> float | None:
    """Optional template-based enemy troop detection if templates exist."""
    if image is None or image.size == 0:
        return None
    base = os.path.join(os.path.dirname(__file__), "..", "detection", "reference_images", "enemy_troops")
    base = os.path.abspath(base)
    if not os.path.isdir(base):
        return None
    has_images = any(
        name.lower().endswith((".png", ".jpg", ".jpeg")) for name in os.listdir(base)
    )
    if not has_images:
        return None
    h, w = image.shape[:2]
    x_min, x_max = int(w * 0.10), int(w * 0.90)
    if region == "mid":
        y_min, y_max = int(h * 0.35), int(h * 0.55)
    else:
        y_min, y_max = int(h * 0.18), int(h * 0.45)
    coord = find_image(image, "enemy_troops", tolerance=0.85, subcrop=(x_min, y_min, x_max, y_max))
    return 1.0 if coord is not None else 0.0


def _pick_defense_override(
    enemy_dets: list,
    bounds: tuple[int, int, int, int],
    hand_cards: dict[int, str],
    available_indices: list[int],
    elixir: int,
    logger: Logger | None = None,
) -> PolicyAction | None:
    if not enemy_dets:
        return None
    x_min, x_max, y_min, y_max = bounds
    mid_y = (y_min + y_max) / 2.0
    mid_x = (x_min + x_max) / 2.0
    center_defense_y = int(y_min + (y_max - y_min) * 0.55)
    enemy_on_our_side = False
    for det in enemy_dets:
        cy = (det.y1 + det.y2) / 2.0
        if cy >= mid_y - 5:
            enemy_on_our_side = True
            break

    def _center_coord(y_val: float | None = None) -> tuple[int, int]:
        y_coord = int(y_val) if y_val is not None else center_defense_y
        return clamp_coord((int(mid_x), y_coord), bounds)

    # Map card ids to available indices
    card_to_indices: dict[str, list[int]] = {}
    for idx in available_indices:
        card_id = hand_cards.get(idx)
        if not card_id:
            continue
        card_to_indices.setdefault(card_id, []).append(idx)

    def _has_card(card_id: str, cost: int) -> int | None:
        if elixir < cost:
            return None
        indices = card_to_indices.get(card_id)
        if not indices:
            return None
        return indices[0]

    def _enemy_crossing(target_ids: set[str], min_conf: float = 0.35) -> tuple[int, int] | None:
        for det in enemy_dets:
            bot_id = getattr(det, "bot_id", None)
            if bot_id not in target_ids:
                continue
            if getattr(det, "conf", 0.0) < min_conf:
                continue
            cy = (det.y1 + det.y2) / 2.0
            if cy < mid_y - 5:
                continue
            cx = (det.x1 + det.x2) / 2.0
            return int(cx), int(cy)
        return None

    def _enemy_count(target_ids: set[str], min_conf: float = 0.35) -> int:
        count = 0
        for det in enemy_dets:
            bot_id = getattr(det, "bot_id", None)
            if bot_id not in target_ids:
                continue
            if getattr(det, "conf", 0.0) < min_conf:
                continue
            cy = (det.y1 + det.y2) / 2.0
            if cy < mid_y - 5:
                continue
            count += 1
        return count

    # Threat-weighted defense queue (uses detected enemy units on our side).
    threat_defs = [
        ("bridge_rush", {"hog_rider", "battle_ram", "ram_rider"}, 3.0),
        ("tank", {"giant", "golem", "electro_giant", "mega_knight", "pekka", "giant_skeleton"}, 2.6),
        ("air", AIR_UNIT_IDS, 2.2),
        ("swarm", {"skeleton_army", "bats", "goblins", "spear_goblins", "goblin_gang", "skeletons"}, 1.6),
    ]
    threat_hits: dict[str, list] = {name: [] for name, _ids, _w in threat_defs}
    for det in enemy_dets:
        if getattr(det, "conf", 0.0) < 0.35:
            continue
        cy = (det.y1 + det.y2) / 2.0
        if cy < mid_y - 5:
            continue
        bot_id = getattr(det, "bot_id", None)
        for name, ids, _w in threat_defs:
            if bot_id in ids:
                threat_hits[name].append(det)
                break
    best_threat = None
    best_score = 0.0
    best_det = None
    for name, _ids, weight in threat_defs:
        dets = threat_hits.get(name) or []
        if not dets:
            continue
        score = sum(getattr(d, "conf", 0.0) for d in dets) * weight
        if score > best_score:
            best_score = score
            best_threat = name
            best_det = max(dets, key=lambda d: getattr(d, "conf", 0.0))
    if best_threat and best_det is not None:
        loc = (int((best_det.x1 + best_det.x2) / 2.0), int((best_det.y1 + best_det.y2) / 2.0))
        def _choose_from(order: list[tuple[str, int, str]]) -> PolicyAction | None:
            for card_id, cost, placement in order:
                idx = _has_card(card_id, cost)
                if idx is None:
                    continue
                if placement == "center":
                    coord = _center_coord(max(loc[1], center_defense_y))
                elif placement == "on_top":
                    coord = clamp_coord(loc, bounds)
                else:
                    coord = _center_coord()
                if logger:
                    logger.log(f"[Override] Threat queue {best_threat}: dropping {card_id}")
                return PolicyAction(
                    card_index=idx,
                    card_group=get_card_group(hand_cards[idx]),
                    coord=coord,
                    card_id=hand_cards[idx],
                )
            return None
        if best_threat == "bridge_rush":
            action = _choose_from(
                [
                    ("goblin_cage", 4, "center"),
                    ("mini_pekka", 4, "center"),
                    ("knight", 3, "center"),
                    ("valkyrie", 4, "center"),
                ]
            )
            if action:
                return action
        if best_threat == "tank":
            action = _choose_from(
                [
                    ("mini_pekka", 4, "center"),
                    ("goblin_cage", 4, "center"),
                    ("valkyrie", 4, "center"),
                    ("knight", 3, "center"),
                ]
            )
            if action:
                return action
        if best_threat == "air":
            action = _choose_from(
                [
                    ("minions", 3, "on_top"),
                    ("wizard", 5, "center"),
                    ("archers", 3, "center"),
                    ("arrows", 3, "on_top"),
                ]
            )
            if action:
                return action
        if best_threat == "swarm":
            action = _choose_from(
                [
                    ("valkyrie", 4, "center"),
                    ("archers", 3, "on_top"),
                    ("wizard", 5, "center"),
                    ("arrows", 3, "on_top"),
                ]
            )
            if action:
                return action

    # 0) Tower-targeters: prioritize immediate defense (exclude reserved defenders).
    tower_targeters_ground = {
        "golem",
        "electro_giant",
        "royale_giant",
        "giant_skeleton",
        "ram_rider",
        "mega_knight",
        "pekka",
    }
    tower_targeters_air = {"balloon", "lava_hound"}
    # 0a) If enemies are on our side, force a defensive response immediately.
    if enemy_on_our_side:
        idx = _has_card("goblin_cage", 4)
        if idx is None:
            idx = _has_card("mini_pekka", 4)
        if idx is None:
            idx = _has_card("knight", 3)
        if idx is None:
            idx = _has_card("valkyrie", 4)
        if idx is None:
            idx = _has_card("archers", 3)
        if idx is None:
            idx = _has_card("minions", 3)
        if idx is None:
            idx = _has_card("skeletons", 1)
        if idx is not None:
            coord = _center_coord()
            if logger:
                logger.log("[Override] Enemy on our side: forcing immediate defense")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])
    loc = _enemy_crossing(tower_targeters_air)
    if loc:
        idx = _has_card("minions", 3)
        if idx is None:
            idx = _has_card("wizard", 5)
        if idx is None:
            idx = _has_card("archers", 3)
        if idx is None:
            idx = _has_card("arrows", 3)
        if idx is not None:
            card_id = hand_cards[idx]
            if card_id in {"wizard", "archers"}:
                coord = _center_coord()
            else:
                coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Air tower-targeter detected: dropping defender")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 0b) Hog/Giant/Battle Ram -> Goblin Cage center, else Mini P.E.K.K.A on top.
    loc = _enemy_crossing({"hog_rider", "giant", "battle_ram"}, min_conf=0.4)
    if loc:
        idx = _has_card("goblin_cage", 4)
        if idx is not None:
            coord = _center_coord(max(loc[1], center_defense_y))
            if logger:
                logger.log("[Override] Hog/Giant/Battle Ram detected: dropping Goblin Cage")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])
        idx = _has_card("mini_pekka", 4)
        if idx is not None:
            coord = _center_coord(max(loc[1], center_defense_y))
            if logger:
                logger.log("[Override] Hog/Giant/Battle Ram detected: dropping Mini P.E.K.K.A")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])
    # 0b2) Barbarians crossing -> Goblin Cage center.
    loc = _enemy_crossing({"barbarians"}, min_conf=0.4)
    if loc:
        idx = _has_card("goblin_cage", 4)
        if idx is not None:
            coord = _center_coord(max(loc[1], center_defense_y))
            if logger:
                logger.log("[Override] Barbarians detected: dropping Goblin Cage")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])
    loc = _enemy_crossing(tower_targeters_ground, min_conf=0.4)
    if loc:
        idx = _has_card("wizard", 5)
        if idx is None:
            idx = _has_card("knight", 3)
        if idx is None:
            idx = _has_card("valkyrie", 4)
        if idx is not None:
            card_id = hand_cards[idx]
            if card_id in {"valkyrie"}:
                coord = _center_coord(max(loc[1], center_defense_y))
            else:
                coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Ground tower-targeter detected: dropping defender")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 1) Hog Rider at the bridge -> Mini P.E.K.K.A then Knight.
    loc = _enemy_crossing({"hog_rider"}, min_conf=0.4)
    if loc:
        idx = _has_card("mini_pekka", 4)
        if idx is None:
            idx = _has_card("knight", 3)
        if idx is not None:
            coord = _center_coord(max(loc[1], center_defense_y))
            if logger:
                logger.log("[Override] Hog rider detected: dropping defender")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 1b) Heavy tanks crossing -> Wizard/Knight.
    loc = _enemy_crossing({"golem", "electro_giant", "mega_knight", "pekka", "giant_skeleton"}, min_conf=0.4)
    if loc:
        idx = _has_card("wizard", 5)
        if idx is None:
            idx = _has_card("knight", 3)
        if idx is not None:
            card_id = hand_cards[idx]
            if card_id == "mini_pekka":
                coord = _center_coord(max(loc[1], center_defense_y))
            else:
                coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Tank detected: dropping defender")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 1c) Fast bridge threats -> Knight.
    loc = _enemy_crossing({"bandit", "prince", "dark_prince", "ram_rider", "battle_ram"}, min_conf=0.4)
    if loc:
        idx = _has_card("knight", 3)
        if idx is not None:
            coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Fast threat detected: dropping defender")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 1c2) Enemy Mini P.E.K.K.A -> Skeletons center to distract.
    loc = _enemy_crossing({"mini_pekka"}, min_conf=0.4)
    if loc:
        idx = _has_card("skeletons", 1)
        if idx is not None:
            coord = _center_coord(max(loc[1], center_defense_y))
            if logger:
                logger.log("[Override] Mini P.E.K.K.A detected: dropping Skeletons center")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 1c3) Single-target ground threats -> Skeletons in center to distract.
    loc = _enemy_crossing({"knight"}, min_conf=0.4)
    if loc:
        idx = _has_card("skeletons", 1)
        if idx is not None:
            coord = _center_coord(max(loc[1], center_defense_y))
            if logger:
                logger.log("[Override] Single-target detected: dropping Skeletons center")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 1d) Air win-cons -> Minions/Musketeer/Archers, else Arrows.
    loc = _enemy_crossing({"balloon", "lava_hound", "flying_machine", "inferno_dragon"})
    if loc:
        idx = _has_card("minions", 3)
        if idx is None:
            idx = _has_card("wizard", 5)
        if idx is None:
            idx = _has_card("archers", 3)
        if idx is None:
            idx = _has_card("arrows", 3)
        if idx is not None:
            card_id = hand_cards[idx]
            if card_id in {"wizard", "archers"}:
                coord = _center_coord()
            else:
                coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Air threat detected: dropping defender")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 2) Goblins / Spear Goblins crossing -> Archers/Valkyrie.
    loc = _enemy_crossing({"goblins", "spear_goblins", "goblin_gang"}, min_conf=0.4)
    if loc:
        idx = _has_card("archers", 3)
        if idx is None:
            idx = _has_card("valkyrie", 4)
        if idx is not None:
            card_id = hand_cards[idx]
            if card_id in {"valkyrie"}:
                coord = _center_coord(max(loc[1], center_defense_y))
            else:
                coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Goblins detected: dropping counter")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 2b) Swarm units -> Archers/Minions, else Valkyrie for ground swarms.
    swarm_ids = {"skeleton_army", "bats", "skeletons"}
    loc = _enemy_crossing(swarm_ids)
    if loc:
        swarm_count = _enemy_count(swarm_ids)
        has_skel_army = False
        for det in enemy_dets:
            bot_id = getattr(det, "bot_id", None)
            if bot_id != "skeleton_army":
                continue
            if getattr(det, "conf", 0.0) < 0.35:
                continue
            cy = (det.y1 + det.y2) / 2.0
            if cy < mid_y - 5:
                continue
            has_skel_army = True
            break
        idx = _has_card("archers", 3) if (swarm_count >= 2 or has_skel_army) else None
        if idx is None:
            idx = _has_card("minions", 3)
        if idx is None:
            idx = _has_card("valkyrie", 4)
        if idx is not None:
            card_id = hand_cards[idx]
            if card_id == "valkyrie":
                coord = _center_coord(max(loc[1], center_defense_y))
            else:
                coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Swarm detected: dropping counter")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 3) Minions / Horde targeting -> Minions, else Arrows.
    loc = _enemy_crossing({"minions", "minion_horde"})
    if loc:
        idx = _has_card("minions", 3)
        if idx is None:
            idx = _has_card("arrows", 3)
        if idx is not None:
            coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Minions detected: dropping counter")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # 4) Baby Dragon / Wizard crossing -> Minions/Musketeer (defensive).
    loc = _enemy_crossing({"baby_dragon", "wizard"})
    if loc:
        idx = _has_card("minions", 3)
        if idx is None:
            idx = _has_card("wizard", 5)
        if idx is not None:
            coord = clamp_coord(loc, bounds)
            if logger:
                logger.log("[Override] Splash unit detected: dropping counter")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    return None


def _pick_escort_override(
    bounds: tuple[int, int, int, int],
    hand_cards: dict[int, str],
    available_indices: list[int],
    elixir: int,
    last_wincon_ts: float | None,
    last_wincon_side: str | None,
    last_escort_ts: float | None,
    escort_stage: int,
    elapsed_time: float,
    enemy_air_present: bool = False,
    ally_giant_present: bool = False,
    logger: Logger | None = None,
) -> PolicyAction | None:
    # Escort overrides only when a Giant is detected on board.
    if not ally_giant_present:
        return None
    if last_wincon_ts is None or last_wincon_side is None:
        return None
    if elapsed_time - last_wincon_ts > 8.0:
        return None
    # Only escort if we have enough elixir for support.
    if elixir < 3:
        return None
    x_min, x_max, _y_min, _y_max = bounds
    mid_x = (x_min + x_max) // 2
    lane_x = mid_x - 40 if last_wincon_side == "left" else mid_x + 40
    back_y = int(_y_min + (_y_max - _y_min) * 0.70)

    # Stage 2: after Giant + Wizard, add Archers (>=3 elixir) and Minions (>=4 elixir) if possible.
    if escort_stage >= 1 and last_escort_ts is not None and (elapsed_time - last_escort_ts) <= 8.0:
        support_order = ("archers", "minions") if enemy_air_present else ("archers", "minions")
        for support_id in support_order:
            min_cost = 4 if support_id == "minions" else 3
            if elixir < min_cost:
                continue
            idxs = [i for i in available_indices if hand_cards.get(i) == support_id]
            if not idxs:
                continue
            idx = idxs[0]
            coord = (lane_x, back_y)
            if logger:
                logger.log("[Override] Giant escort chain: dropping support")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])

    # Stage 1: prefer Wizard directly behind the Giant.
    if escort_stage == 0:
        idxs = [i for i in available_indices if hand_cards.get(i) == "wizard"]
        if idxs and elixir >= 5:
            idx = idxs[0]
            coord = (lane_x, back_y)
            if logger:
                logger.log("[Override] Giant escort: dropping Wizard")
            return PolicyAction(card_index=idx, card_group=get_card_group(hand_cards[idx]), coord=coord, card_id=hand_cards[idx])
    return None


def _save_detection_debug(
    image: np.ndarray,
    enemy_hp: float | None,
    player_hp: float | None,
    enemy_activity: float | None,
    player_hp_delta: float | None,
    enemy_hp_delta: float | None,
    filename: str,
    enemy_dets: list | None = None,
    ally_dets: list | None = None,
    unknown_dets: list | None = None,
    show_window: bool = False,
) -> None:
    if image is None or image.size == 0:
        return
    def _safe(val: float | None, default: float = 0.0) -> float:
        return float(val) if val is not None else default

    enemy_hp = _safe(enemy_hp, 0.0)
    player_hp = _safe(player_hp, 0.0)
    enemy_activity = _safe(enemy_activity, 0.0)
    player_hp_delta = _safe(player_hp_delta, 0.0)
    enemy_hp_delta = _safe(enemy_hp_delta, 0.0)

    img = image.copy()
    h, w = img.shape[:2]

    # Arena ROI (extend upward to match top UI band)
    ax1, ax2 = int(w * 0.10), int(w * 0.90)
    ay1, ay2 = int(h * 0.055), int(h * 0.78)
    cv2.rectangle(img, (ax1, ay1), (ax2, ay2), (255, 255, 0), 2)

    # Enemy HP (top) - dynamic band if possible
    ex1, ex2 = int(w * 0.12), int(w * 0.88)
    ey1, ey2 = int(h * 0.05), int(h * 0.22)
    enemy_band = _find_hp_band(img, "enemy", ey1, ey2, ex1, ex2)
    if enemy_band:
        ey1, ey2 = enemy_band
    cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (0, 255, 0), 2)

    # Player HP (bottom) - dynamic band if possible
    px1, px2 = int(w * 0.12), int(w * 0.88)
    py1, py2 = int(h * 0.56), int(h * 0.72)
    band_h = max(py2 - py1, 1)
    shift = band_h * _get_player_hp_shift_mult()
    py1 = max(0, int(py1 + shift))
    py2 = max(1, int(py2 + shift))
    player_band = _find_hp_band(img, "player", py1, py2, px1, px2)
    if player_band:
        py1, py2 = player_band
    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 200, 0), 2)

    # Text overlay
    text = (
        f"enemy_hp={enemy_hp:.3f} player_hp={player_hp:.3f} "
        f"activity={enemy_activity:.4f} dP={player_hp_delta:.4f} dE={enemy_hp_delta:.4f}"
    )
    cv2.putText(img, text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_dets(dets: list | None, color: tuple[int, int, int]) -> None:
        if not dets:
            return
        for det in dets:
            try:
                cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            except Exception:
                continue

    _draw_dets(enemy_dets, (0, 0, 255))  # red
    _draw_dets(ally_dets, (255, 0, 0))  # blue
    _draw_dets(unknown_dets, (180, 180, 180))  # gray

    if show_window:
        try:
            cv2.imshow("TowerLogic debug", img)
            cv2.waitKey(1)
        except Exception:
            pass

    out_path = os.path.join(log_dir, filename)
    try:
        cv2.imwrite(out_path, img)
    except Exception:
        pass


def _consume_hp_probe_flag() -> bool:
    if os.path.exists(HP_PROBE_FLAG):
        with suppress(Exception):
            os.remove(HP_PROBE_FLAG)
        return True
    return False


def _consume_hand_probe_flag() -> bool:
    if os.path.exists(HAND_PROBE_FLAG):
        with suppress(Exception):
            os.remove(HAND_PROBE_FLAG)
        return True
    return False


def get_to_main_after_fight(emulator, logger):
    timeout = 120  # s
    start_time = time.time()
    clicked_ok_or_exit = False

    logger.change_status("Returning to clash main after the fight...")

    while time.time() - start_time < timeout:
        # if on clash main
        if check_if_on_clash_main_menu(emulator) is True:
            # wait 3 seconds for the trophy road page to maybe appear bc of UI lag
            interruptible_sleep(3)

            # if that trophy road page appears, handle it, then return True
            if check_for_trophy_reward_menu(emulator):
                print("Found trophy reward menu")
                handle_trophy_reward_menu(emulator, logger, printmode=False)
                interruptible_sleep(2)

            print("Made it to clash main after a fight")
            return True

        # check for trophy reward screen
        if check_for_trophy_reward_menu(emulator):
            print("Found trophy reward menu!\nHandling Trophy Reward Menu")
            handle_trophy_reward_menu(emulator, logger, printmode=False)
            interruptible_sleep(3)
            continue

        # check for post-battle button (OK/exit)
        if not clicked_ok_or_exit:
            button_coord = find_post_battle_button(emulator)
            if button_coord is not None:
                crowns = detect_crown_counts_from_end_screen(emulator.screenshot())
                if crowns is not None:
                    logger.last_crown_result = crowns
                    logger.log(f"[Reward] Crowns detected: player={crowns[0]} opp={crowns[1]}")
                print("Found post-battle button, clicking it.")
                emulator.click(button_coord[0], button_coord[1])
                clicked_ok_or_exit = True
                continue

        interruptible_sleep(1)
        print("Clicking on deadspace to close potential pop-up windows.")
        emulator.click(CLASH_MAIN_DEADSPACE_COORD[0], CLASH_MAIN_DEADSPACE_COORD[1])

    return False


# main fight loops

# Initialize a deque with a maximum length of 3 to store the last three chosen cards
last_three_cards = collections.deque(maxlen=3)


def select_card_index(card_indices, last_three_cards):
    if not card_indices:
        raise ValueError("card_indices cannot be empty")

    # First preference: Cards not in the last_three_cards queue
    preferred_cards = [index for index in card_indices if index not in last_three_cards]

    # Second preference: Cards not among the last two added to the queue
    if not preferred_cards and len(last_three_cards) == 3:
        preferred_cards = [index for index in card_indices if index not in list(last_three_cards)[-2:]]

    # Third preference: Any card except the most recently added one
    if not preferred_cards and last_three_cards:
        preferred_cards = [index for index in card_indices if index != last_three_cards[-1]]

    # Fallback: If all else fails, consider all cards
    if not preferred_cards:
        preferred_cards = card_indices

    return random.choice(preferred_cards)


def play_a_card(emulator, logger, recording_flag: bool, battle_strategy: "BattleStrategy") -> bool:
    print("\n")

    # check which cards are available
    logger.change_status("Looking at which cards are available")
    available_card_check_start_time = time.time()
    card_indicies = check_which_cards_are_available(emulator, False, True)

    if not card_indicies:
        logger.change_status("No cards ready yet...")
        return False

    available_card_check_time_taken = str(
        time.time() - available_card_check_start_time,
    )[:3]

    logger.change_status(
        f"These cards are available: {card_indicies} ({available_card_check_time_taken}s)",
    )

    card_index = select_card_index(card_indicies, last_three_cards)
    if card_index not in last_three_cards:
        last_three_cards.append(card_index)
    logger.change_status(f"Choosing this card index: {card_index}")

    # get a coord based on the selected side
    play_coord_calculation_start_time = time.time()
    card_id, play_coord = get_play_coords_for_card(emulator, logger, card_index, battle_strategy.get_elapsed_time())
    play_coord_calculation_time_taken = str(
        time.time() - play_coord_calculation_start_time,
    )[:3]

    logger.change_status(
        f"Calculated play for: {card_id} at {play_coord} ({play_coord_calculation_time_taken}s)",
    )

    # click the card index
    click_and_play_card_start_time = time.time()
    if None in [HAND_CARDS_COORDS, card_index]:
        logger.change_status("[!] Non fatal error: card_index is None")
        return False

    emulator.click(HAND_CARDS_COORDS[card_index][0], HAND_CARDS_COORDS[card_index][1])

    # click the play coord
    if play_coord is None:
        logger.change_status("[!] Non fatal error: play_coord is None")
        return False

    emulator.click(play_coord[0], play_coord[1])
    click_and_play_card_time_taken = str(time.time() - click_and_play_card_start_time)[:3]
    if recording_flag:
        save_play(play_coord, card_index)

    logger.change_status(f"Made the play {click_and_play_card_time_taken}s")
    logger.add_card_played()

    # Emotes disabled
    return True


def play_a_card_with_policy(
    emulator,
    logger: Logger,
    recording_flag: bool,
    battle_strategy: "BattleStrategy",
    policy: EpsilonGreedyPolicy,
    trainer: OnlineTrainer | None = None,
    free_placement: bool = False,
    min_elixir: int = 0,
    allow_hold: bool = True,
    strict_elixir: bool = False,
    spell_gate: bool = False,
    debug_detection: bool = False,
    debug_overlay: bool = False,
    debug_hp_log: bool = False,
    debug_hand_conf: bool = False,
) -> bool:
    if debug_overlay and not debug_detection:
        debug_detection = True
    if debug_detection and not debug_hand_conf:
        debug_hand_conf = True
    logger.change_status("Policy: evaluating action candidates")
    # Improve card recognition for the selected deck.
    set_active_deck_filter(
        [
            "giant",
            "wizard",
            "mini_pekka",
            "archers",
            "minions",
            "knight",
            "valkyrie",
            "goblin_cage",
            "skeletons",
        ]
    )
    available_indices = check_which_cards_are_available(emulator, False, True)
    if not available_indices:
        logger.change_status("Policy: no cards ready yet")
        return False

    hand_cards = {idx: identify_hand_cards(emulator, idx) for idx in range(4)}
    wincon_in_hand = any(
        card_id == "giant" or get_card_group(card_id) == "big_win_con"
        for card_id in hand_cards.values()
    )

    _, side_preference = switch_side()
    elapsed_time = battle_strategy.get_elapsed_time()
    elixir = estimate_elixir(emulator)
    if not allow_hold and elixir < max(0, int(min_elixir)):
        logger.change_status(f"Policy: waiting for elixir (have {elixir}, need {min_elixir})")
        return False
    snapshot = emulator.screenshot()
    bounds = safe_bounds_for_image(snapshot)
    detector = getattr(logger, "_field_detector", None)
    detector_summary = None
    detector_enemy = None
    detector_ally = None
    detector_enemy_dets = None
    detector_ally_dets = None
    detector_unknown_dets = None
    if detector is not None and snapshot is not None:
        try:
            detections = _detect_field_with_roi(detector, snapshot, bounds)
            enemy_dets, ally_dets, unknown_dets = split_detections_by_side(snapshot, detections)
            detector_summary = summarize_detections(detections, snapshot.shape)
            detector_enemy = summarize_detections(enemy_dets, snapshot.shape)
            detector_ally = summarize_detections(ally_dets, snapshot.shape)
            detector_enemy_dets = enemy_dets
            detector_ally_dets = ally_dets
            detector_unknown_dets = unknown_dets
        except Exception:
            detector_summary = None
            detector_enemy = None
            detector_ally = None
            detector_enemy_dets = None
            detector_ally_dets = None
            detector_unknown_dets = None
    crowns_now = detect_crown_counts_in_battle(snapshot) if snapshot is not None else None
    enemy_hp_ratio = estimate_enemy_hp_ratio(snapshot) if snapshot is not None else None
    player_hp_ratio = estimate_player_hp_ratio(snapshot) if snapshot is not None else None
    if snapshot is not None:
        enemy_left_hp, enemy_right_hp, enemy_king_hp = estimate_enemy_tower_hps(snapshot)
        player_left_hp, player_right_hp, player_king_hp = estimate_player_tower_hps(snapshot)
    else:
        enemy_left_hp, enemy_right_hp, enemy_king_hp = (None, None, None)
        player_left_hp, player_right_hp, player_king_hp = (None, None, None)
    prev_frame = getattr(logger, "_prev_battle_frame", None)
    motion_activity = detect_enemy_activity(snapshot, prev_frame)
    redbar_activity = detect_enemy_red_bar_activity(snapshot)
    template_activity = detect_enemy_templates(snapshot, region="top")
    _motion_top, _motion_mid = detect_enemy_activity_zone(snapshot, prev_frame)
    _red_top, red_mid = detect_enemy_red_bar_zone(snapshot)
    enemy_activity = max(
        a for a in [motion_activity, redbar_activity, template_activity] if a is not None
    ) if (motion_activity is not None or redbar_activity is not None or template_activity is not None) else None
    if detector_enemy is not None:
        total = max(1, detector_enemy.total)
        det_activity = min(1.0, detector_enemy.top / total + 0.15)
        enemy_activity = max(enemy_activity or 0.0, det_activity)
    lane_enemy_left = False
    lane_enemy_right = False
    lane_enemy_count_left = 0
    lane_enemy_count_right = 0
    enemy_air_present = False
    enemy_ground_present = False
    enemy_left_total = 0
    enemy_right_total = 0
    ally_left_total = 0
    ally_right_total = 0
    ally_giant_present = False
    if detector_enemy is not None and detector_enemy.total > 0:
        lane_enemy_left = detector_enemy.left > 0
        lane_enemy_right = detector_enemy.right > 0
    else:
        lane_enemy_left = False
        lane_enemy_right = False
    if detector_enemy_dets:
        x_min, x_max, y_min, y_max = bounds
        mid_x = (x_min + x_max) / 2.0
        mid_y = (y_min + y_max) / 2.0
        for det in detector_enemy_dets:
            bot_id = getattr(det, "bot_id", None)
            if bot_id in AIR_UNIT_IDS:
                enemy_air_present = True
            else:
                enemy_ground_present = True
            cy = (det.y1 + det.y2) / 2.0
            cx = (det.x1 + det.x2) / 2.0
            if cx <= mid_x:
                enemy_left_total += 1
            else:
                enemy_right_total += 1
            if cy < mid_y - 5:
                continue
            if cx <= mid_x:
                lane_enemy_count_left += 1
            else:
                lane_enemy_count_right += 1
    if detector_ally_dets:
        x_min, x_max, _y_min, _y_max = bounds
        mid_x = (x_min + x_max) / 2.0
        for det in detector_ally_dets:
            bot_id = getattr(det, "bot_id", None)
            if bot_id == "giant" and getattr(det, "conf", 0.0) >= 0.5:
                ally_giant_present = True
            cx = (det.x1 + det.x2) / 2.0
            if cx <= mid_x:
                ally_left_total += 1
            else:
                ally_right_total += 1
    def _clamp01(val: float | None) -> float | None:
        if val is None:
            return None
        return float(max(0.0, min(1.0, val)))
    # Disable pressure/ranged signals (use detections only).
    setattr(logger, "_prev_battle_frame", snapshot)
    prev_enemy_hp_state = getattr(logger, "_prev_enemy_hp_state", None)
    prev_player_hp_state = getattr(logger, "_prev_player_hp_state", None)
    enemy_hp_delta = 0.0
    player_hp_delta = 0.0
    if enemy_hp_ratio is not None:
        if prev_enemy_hp_state is not None:
            enemy_hp_delta = enemy_hp_ratio - prev_enemy_hp_state
        setattr(logger, "_prev_enemy_hp_state", enemy_hp_ratio)
    if player_hp_ratio is not None:
        if prev_player_hp_state is not None:
            player_hp_delta = player_hp_ratio - prev_player_hp_state
        setattr(logger, "_prev_player_hp_state", player_hp_ratio)
    prev_enemy_towers = getattr(logger, "_prev_enemy_tower_hps", None)
    prev_player_towers = getattr(logger, "_prev_player_tower_hps", None)
    enemy_left_delta = 0.0
    enemy_right_delta = 0.0
    enemy_king_delta = 0.0
    player_left_delta = 0.0
    player_right_delta = 0.0
    player_king_delta = 0.0
    if prev_enemy_towers is not None:
        prev_left, prev_right, prev_king = prev_enemy_towers
        if enemy_left_hp is not None and prev_left is not None:
            enemy_left_delta = enemy_left_hp - prev_left
        if enemy_right_hp is not None and prev_right is not None:
            enemy_right_delta = enemy_right_hp - prev_right
        if enemy_king_hp is not None and prev_king is not None:
            enemy_king_delta = enemy_king_hp - prev_king
    if prev_player_towers is not None:
        prev_left, prev_right, prev_king = prev_player_towers
        if player_left_hp is not None and prev_left is not None:
            player_left_delta = player_left_hp - prev_left
        if player_right_hp is not None and prev_right is not None:
            player_right_delta = player_right_hp - prev_right
        if player_king_hp is not None and prev_king is not None:
            player_king_delta = player_king_hp - prev_king
    setattr(logger, "_prev_enemy_tower_hps", (enemy_left_hp, enemy_right_hp, enemy_king_hp))
    setattr(logger, "_prev_player_tower_hps", (player_left_hp, player_right_hp, player_king_hp))

    if debug_hp_log:
        logger.log(
            "[HP] enemy "
            f"L={enemy_left_hp} R={enemy_right_hp} K={enemy_king_hp} "
            f"dL={enemy_left_delta:.4f} dR={enemy_right_delta:.4f} dK={enemy_king_delta:.4f} "
            "player "
            f"L={player_left_hp} R={player_right_hp} K={player_king_hp} "
            f"dL={player_left_delta:.4f} dR={player_right_delta:.4f} dK={player_king_delta:.4f}"
        )

    enemy_present = detector_enemy is not None and detector_enemy.total > 0
    target_side = choose_target_side(enemy_left_hp, enemy_right_hp)
    # Stick to a chosen weak-side target for a short window unless enemies are present.
    now_ts = time.time()
    last_target = getattr(logger, "_target_side", None)
    last_target_ts = getattr(logger, "_target_side_ts", 0.0)
    if target_side:
        setattr(logger, "_target_side", target_side)
        setattr(logger, "_target_side_ts", now_ts)
    else:
        if last_target and (now_ts - last_target_ts) < 12 and not enemy_present:
            target_side = last_target
    lane_lock_active = False
    enemy_on_our_side = (lane_enemy_count_left + lane_enemy_count_right) > 0
    enemy_heavy_side = None
    if enemy_on_our_side:
        if lane_enemy_count_left > lane_enemy_count_right:
            enemy_heavy_side = "left"
        elif lane_enemy_count_right > lane_enemy_count_left:
            enemy_heavy_side = "right"
    # Lane-aware push lock: stick to last win-con lane unless heavy pressure flips.
    last_wincon_ts = getattr(logger, "_last_wincon_ts", None)
    last_wincon_side = getattr(logger, "_last_wincon_side", None)
    wincon_lock = False
    if last_wincon_ts is not None and last_wincon_side is not None:
        if (now_ts - last_wincon_ts) < 12:
            if enemy_heavy_side and enemy_heavy_side != last_wincon_side and enemy_on_our_side:
                wincon_lock = False
            else:
                wincon_lock = True
                target_side = last_wincon_side
    if not wincon_lock and last_target and (now_ts - last_target_ts) < 15:
        if not enemy_on_our_side:
            lane_lock_active = True
            target_side = last_target
        elif enemy_heavy_side and enemy_heavy_side != last_target:
            # Break lane lock if heavy pressure appears on the other side.
            lane_lock_active = False
        else:
            lane_lock_active = True
            target_side = last_target
    elif wincon_lock:
        lane_lock_active = True
    prefer_attack_side = target_side if target_side and not enemy_present else None
    lane_bias = 0.5
    if target_side == "left":
        lane_bias = 0.0
    elif target_side == "right":
        lane_bias = 1.0
    defend_mode = 0.0
    available_mask = [1 if i in available_indices else 0 for i in range(4)]
    crown_diff = 0.0
    if crowns_now is not None:
        crown_diff = (crowns_now[0] - crowns_now[1]) / 3.0

    # Always-on vision line (rate-limited), independent of debug toggles.
    if detector is not None:
        now_ts = time.time()
        last_ts = getattr(logger, "_vision_stats_ts", 0.0)
        if now_ts - last_ts > 2.0:
            enemy_total = detector_enemy.total if detector_enemy is not None else (enemy_left_total + enemy_right_total)
            ally_total = detector_ally.total if detector_ally is not None else (ally_left_total + ally_right_total)
            logger.log(
                "[Vision] Enemy units: "
                f"{enemy_total} (left={enemy_left_total}, right={enemy_right_total}) | "
                f"Ally units: {ally_total} (left={ally_left_total}, right={ally_right_total})"
            )
            setattr(logger, "_vision_stats_ts", now_ts)
    actions: list[PolicyAction] = []
    wincon_available = False
    side_for_coords = prefer_attack_side or side_preference
    if debug_detection or debug_hand_conf:
        hand_info = ", ".join(
            f"{idx}:{card_id}/{get_card_group(card_id)}" for idx, card_id in hand_cards.items()
        )
        logger.log(f"[Detect] Hand cards: {hand_info}")
        if debug_detection:
            if detector_summary is not None:
                logger.log(
                    "[Detect] Field det "
                    f"total={detector_summary.total} top={detector_summary.top} mid={detector_summary.mid} "
                    f"left={detector_summary.left} right={detector_summary.right} max_conf={detector_summary.max_conf:.2f}"
                )
            if detector_enemy is not None and detector_ally is not None:
                logger.log(
                    "[Detect] Field split "
                    f"enemy={detector_enemy.total} ally={detector_ally.total} "
                    f"enemy_top={detector_enemy.top} enemy_mid={detector_enemy.mid}"
                )
            if bounds and (detector_enemy_dets or detector_ally_dets):
                x_min, x_max, _y_min, _y_max = bounds
                mid_x = (x_min + x_max) // 2
                enemy_left = enemy_right = 0
                ally_left = ally_right = 0
                if detector_enemy_dets:
                    for det in detector_enemy_dets:
                        cx = (det.x1 + det.x2) / 2.0
                        if cx <= mid_x:
                            enemy_left += 1
                        else:
                            enemy_right += 1
                if detector_ally_dets:
                    for det in detector_ally_dets:
                        cx = (det.x1 + det.x2) / 2.0
                        if cx <= mid_x:
                            ally_left += 1
                        else:
                            ally_right += 1
                enemy_total = detector_enemy.total if detector_enemy is not None else (enemy_left + enemy_right)
                ally_total = detector_ally.total if detector_ally is not None else (ally_left + ally_right)
                logger.log(
                    "[Detect] Enemy units: "
                    f"{enemy_total} (left={enemy_left}, right={enemy_right}) | "
                    f"Ally units: {ally_total} (left={ally_left}, right={ally_right})"
                )
            # Record raw signatures for UNKNOWN cards (for calibration).
            unknown_slots = [idx for idx, card_id in hand_cards.items() if card_id == "UNKNOWN"]
            if unknown_slots:
                sig_path = os.path.join(log_dir, "unknown_card_signatures.jsonl")
                crops_dir = os.path.join(log_dir, "unknown_cards")
                now_ts = time.time()
                last_sig_ts = getattr(logger, "_unknown_sig_ts", 0.0)
                if now_ts - last_sig_ts > 2.0:
                    with suppress(Exception):
                        os.makedirs(crops_dir, exist_ok=True)
                        import json
                        hand_img = emulator.screenshot()
                        for idx in unknown_slots:
                            sig = get_all_pixel_data(emulator, idx)
                            payload = {"ts": now_ts, "slot": idx, "signature": sig}
                            with open(sig_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(payload) + "\n")
                            # Also save a crop of the card art so we can identify it.
                            if hand_img is not None and hand_img.size > 0:
                                crop = get_hand_card_crop(emulator, idx)
                                if crop.size > 0:
                                    out_path = os.path.join(
                                        crops_dir,
                                        f"unknown_slot{idx}_{int(now_ts)}.png",
                                    )
                                    cv2.imwrite(out_path, crop)
                        setattr(logger, "_unknown_sig_ts", now_ts)
        if debug_hand_conf:
            now_ts = time.time()
            last_ts = getattr(logger, "_hand_conf_ts", 0.0)
            if now_ts - last_ts > 2.0:
                preds = get_hand_card_predictions(emulator, topk=3)
                if preds:
                    parts = []
                    for idx, entries in preds.items():
                        formatted = ", ".join(f"{name}:{conf:.2f}" for name, conf in entries)
                        parts.append(f"{idx}[{formatted}]")
                    logger.log("[Hand] top3 " + " ".join(parts))
                else:
                    logger.log("[Hand] top3 unavailable (classifier missing)")
                setattr(logger, "_hand_conf_ts", now_ts)
    for card_index in available_indices:
        card_id = hand_cards.get(card_index, identify_hand_cards(emulator, card_index))
        card_group = get_card_group(card_id)
        if card_group == "big_win_con":
            wincon_available = True
        if free_placement:
            coords = candidate_coords_free(bounds)
            if prefer_attack_side:
                coords = filter_coords_by_side(coords, bounds, prefer_attack_side)
            # When no enemies are present, keep troops off the bridge.
            if not enemy_present and card_group not in SPELL_GROUPS:
                coords = filter_coords_back(coords, bounds, frac=0.6)
        else:
            group_side = side_for_coords
            if card_id == "giant" and target_side:
                group_side = target_side
            coords = candidate_coords_for_group(card_group, group_side, elapsed_time)
            if not enemy_present and card_group not in SPELL_GROUPS:
                coords = filter_coords_back(coords, bounds, frac=0.6)
        actions.extend(
            PolicyAction(card_index=card_index, card_group=card_group, coord=coord, card_id=card_id) for coord in coords
        )

    force_hold = False
    if elixir < max(0, int(min_elixir)):
        force_hold = True
        logger.change_status(
            f"Policy: waiting for elixir (have {elixir}, need {min_elixir})"
        )

    # Track "enemy just spent" heuristic from detection spikes.
    recent_enemy_spent = False
    if detector_enemy is not None:
        prev_total = getattr(logger, "_prev_enemy_total", None)
        if prev_total is not None:
            if detector_enemy.total >= prev_total + 2:
                setattr(logger, "_enemy_spent_ts", elapsed_time)
        elif detector_enemy.total >= 2:
            setattr(logger, "_enemy_spent_ts", elapsed_time)
        setattr(logger, "_prev_enemy_total", detector_enemy.total)
    enemy_spent_ts = getattr(logger, "_enemy_spent_ts", None)
    if enemy_spent_ts is not None and (elapsed_time - enemy_spent_ts) <= 6.0:
        recent_enemy_spent = True

    override_action = None
    if detector_enemy_dets:
        override_action = _pick_defense_override(
            detector_enemy_dets,
            bounds,
            hand_cards,
            available_indices,
            elixir,
            logger=logger,
        )
    if override_action is None:
        override_action = _pick_escort_override(
            bounds,
            hand_cards,
            available_indices,
            elixir,
            getattr(logger, "_last_wincon_ts", None),
            getattr(logger, "_last_wincon_side", None),
            getattr(logger, "_last_escort_ts", None),
            getattr(logger, "_last_escort_stage", 0),
            elapsed_time,
            enemy_air_present=enemy_air_present,
            ally_giant_present=ally_giant_present,
            logger=logger,
        )

    state_vec = build_state_vector(
        elapsed_time,
        elixir,
        side_for_coords,
        available_mask,
        enemy_activity=enemy_activity or 0.0,
        wincon_available=1.0 if wincon_available else 0.0,
        enemy_red_top=_red_top or 0.0,
        enemy_red_mid=red_mid or 0.0,
        enemy_template_top=template_activity or 0.0,
        enemy_template_mid=0.0,
        lane_bias=lane_bias,
        defend_mode=defend_mode,
        player_hp=player_hp_ratio if player_hp_ratio is not None else 1.0,
        enemy_hp=enemy_hp_ratio if enemy_hp_ratio is not None else 1.0,
        player_hp_delta=player_hp_delta,
        enemy_hp_delta=enemy_hp_delta,
        player_left_delta=player_left_delta,
        player_right_delta=player_right_delta,
        player_king_delta=player_king_delta,
        enemy_left_delta=enemy_left_delta,
        enemy_right_delta=enemy_right_delta,
        enemy_king_delta=enemy_king_delta,
        crown_diff=crown_diff,
    )
    if allow_hold:
        actions.append(hold_action())

    if force_hold:
        if allow_hold:
            actions = [hold_action()]
        else:
            return False

    enemy_ids = set()
    enemy_has_hog = False
    enemy_has_giant = False
    enemy_has_battle_ram = False
    enemy_has_mini_pekka = False
    enemy_has_knight = False
    enemy_has_barbarians = False
    if detector_enemy_dets:
        mid_y = (bounds[2] + bounds[3]) / 2.0

        def _enemy_crossing_strict(target_id: str, min_conf: float = 0.5) -> bool:
            for det in detector_enemy_dets:
                bot_id = getattr(det, "bot_id", None)
                if bot_id != target_id:
                    continue
                if getattr(det, "conf", 0.0) < min_conf:
                    continue
                cy = (det.y1 + det.y2) / 2.0
                if cy < mid_y - 5:
                    continue
                return True
            return False

        for det in detector_enemy_dets:
            bot_id = getattr(det, "bot_id", None)
            if bot_id:
                enemy_ids.add(bot_id)
        enemy_has_hog = _enemy_crossing_strict("hog_rider")
        enemy_has_giant = _enemy_crossing_strict("giant")
        enemy_has_battle_ram = _enemy_crossing_strict("battle_ram")
        enemy_has_mini_pekka = _enemy_crossing_strict("mini_pekka")
        enemy_has_knight = _enemy_crossing_strict("knight")
        enemy_has_barbarians = _enemy_crossing_strict("barbarians")
    actions = mask_actions_for_context(
        actions,
        bounds,
        elixir,
        elapsed_time,
        prefer_attack_side,
        defend_mode,
        target_side,
        wincon_in_hand,
        spell_gate=spell_gate,
        enemy_activity=enemy_activity or 0.0,
        enemy_present=enemy_present,
        lane_enemy_left=lane_enemy_left,
        lane_enemy_right=lane_enemy_right,
        recent_enemy_spent=recent_enemy_spent,
        lane_lock_active=lane_lock_active,
        lane_lock_side=target_side,
        enemy_air_present=enemy_air_present,
        enemy_ground_present=enemy_ground_present,
        enemy_has_hog=enemy_has_hog,
        enemy_has_giant=enemy_has_giant,
        enemy_has_battle_ram=enemy_has_battle_ram,
        enemy_has_mini_pekka=enemy_has_mini_pekka,
        enemy_has_knight=enemy_has_knight,
        enemy_has_barbarians=enemy_has_barbarians,
        ally_giant_present=ally_giant_present,
        lane_enemy_count_left=lane_enemy_count_left,
        lane_enemy_count_right=lane_enemy_count_right,
        last_wincon_ts=getattr(logger, "_last_wincon_ts", None),
        last_wincon_side=getattr(logger, "_last_wincon_side", None),
    )
    if debug_detection:
        logger.log(f"[Policy] Candidate actions: {len(actions)}")
    # Epsilon decay: reduce exploration after a few games (policy updates).
    base_eps = getattr(logger, "_policy_base_epsilon", None)
    if base_eps is not None and policy is not None:
        games = getattr(logger, "policy_games", 0)
        decay_steps = max(0, games // 10)
        min_eps = float(getattr(logger, "_policy_min_epsilon", 0.05))
        decayed = max(min_eps, float(base_eps) * (0.98 ** decay_steps))
        if abs(policy.epsilon - decayed) > 1e-3:
            policy.epsilon = decayed
    action = override_action if override_action is not None else policy.select_action(state_vec, actions)
    # Lightweight "what the model sees" trace for debugging.
    now_ts = time.time()
    last_think = getattr(logger, "_think_ts", 0.0)
    if now_ts - last_think > 2.0:
        think_line = (
            "[Think] "
            f"elixir={elixir} min={min_elixir} hold={allow_hold} force_hold={force_hold} "
            f"strict={strict_elixir} spell_gate={spell_gate} "
            f"target={target_side} "
            f"wincon={wincon_available}"
        )
        if action is not None:
            if action.card_index < 0:
                think_line += " action=hold"
            else:
                think_line += f" action={action.card_id}@{action.coord}"
        logger.log(think_line)
        setattr(logger, "_think_ts", now_ts)
    if action is None:
        logger.change_status("Policy: no valid action; falling back")
        return False
    if action.card_index < 0:
        logger.change_status("Policy: holding (no play)")
        if trainer is not None:
            trainer.record(
                state_vec,
                action,
                elapsed_time=elapsed_time,
                crowns=crowns_now,
                enemy_hp=enemy_hp_ratio,
                player_hp=player_hp_ratio,
                enemy_activity=enemy_activity,
                elixir_after=elixir,
                target_side=prefer_attack_side,
                action_side=None,
                wincon_available=wincon_available,
                player_left_delta=player_left_delta,
                player_right_delta=player_right_delta,
                player_king_delta=player_king_delta,
                enemy_left_delta=enemy_left_delta,
                enemy_right_delta=enemy_right_delta,
                enemy_king_delta=enemy_king_delta,
                lane_bias=lane_bias,
                defend_mode=defend_mode,
                card_id=None,
            )
        return True

    if debug_detection:
        last_ts = getattr(logger, "_debug_detect_ts", 0.0)
        if time.time() - last_ts > 2.0:
            logger.log(
                "[Detect] tower HP ratios "
                f"enemy(L={enemy_left_hp}, R={enemy_right_hp}, K={enemy_king_hp}) "
                f"player(L={player_left_hp}, R={player_right_hp}, K={player_king_hp}) "
                f"bars(enemy={enemy_hp_ratio}, player={player_hp_ratio})"
            )
            fname = f"detect_{int(time.time())}.png"
            _save_detection_debug(
                snapshot,
                enemy_hp_ratio,
                player_hp_ratio,
                enemy_activity,
                player_hp_delta,
                enemy_hp_delta,
                fname,
                enemy_dets=detector_enemy_dets,
                ally_dets=detector_ally_dets,
                unknown_dets=detector_unknown_dets,
                show_window=debug_overlay,
            )
            setattr(logger, "_debug_detect_ts", time.time())

    if _consume_hp_probe_flag():
        probe_name = f"hp_probe_{int(time.time())}.png"
        _save_detection_debug(
            snapshot,
            enemy_hp_ratio,
            player_hp_ratio,
            enemy_activity,
            player_hp_delta,
            enemy_hp_delta,
            probe_name,
            enemy_dets=detector_enemy_dets,
            ally_dets=detector_ally_dets,
            unknown_dets=detector_unknown_dets,
            show_window=debug_overlay,
        )
        logger.log(f"[HP] Probe saved: {probe_name}")

    if _consume_hand_probe_flag():
        ts = int(time.time())
        preds = get_hand_card_predictions(emulator, topk=3)
        if preds:
            for idx, entries in preds.items():
                formatted = ", ".join(f"{name}:{conf:.2f}" for name, conf in entries)
                logger.log(f"[HandProbe] slot{idx} {formatted}")
        else:
            if not HAND_USE_CLASSIFIER:
                tmpl = get_hand_template_scores(emulator, topk=3)
                if tmpl:
                    for idx, entries in tmpl.items():
                        formatted = ", ".join(f"{name}:{conf:.2f}" for name, conf in entries)
                        logger.log(f"[HandProbe] slot{idx} {formatted}")
                    logger.log("[HandProbe] template-only mode (classifier disabled)")
                else:
                    logger.log("[HandProbe] template-only mode (no templates loaded)")
            else:
                logger.log("[HandProbe] classifier unavailable")
        for idx in range(4):
            crop = get_hand_card_crop(emulator, idx)
            if crop is None or crop.size == 0:
                continue
            out_path = os.path.join(log_dir, f"hand_probe_{ts}_slot{idx}.png")
            with suppress(Exception):
                cv2.imwrite(out_path, crop)
        logger.log(f"[HandProbe] saved crops hand_probe_{ts}_slot*.png")

    forced_coord = action.coord
    if bounds and action.card_id == "goblin_cage":
        x_min, x_max, y_min, y_max = bounds
        mid_x = (x_min + x_max) // 2
        center_y = int(y_min + (y_max - y_min) * 0.55)
        forced_coord = (mid_x, center_y)
    if bounds and action.card_id == "skeletons":
        # Only force center when the choice wasn't already targeting an enemy.
        x_min, x_max, y_min, y_max = bounds
        mid_x = (x_min + x_max) // 2
        center_y = int(y_min + (y_max - y_min) * 0.55)
        if abs(action.coord[0] - mid_x) > 40 or action.coord[1] < center_y - 12:
            forced_coord = (mid_x, center_y)
    safe_coord = clamp_coord(forced_coord, bounds)
    clamped = safe_coord != action.coord
    edge = is_near_edge(safe_coord, bounds)
    action_side = "left"
    if bounds:
        x_min, x_max, _y_min, _y_max = bounds
        mid_x = (x_min + x_max) // 2
        action_side = "left" if safe_coord[0] <= mid_x else "right"
    # Track Giant escort chain stages (only when Giant is detected on board).
    last_wincon_ts = getattr(logger, "_last_wincon_ts", None)
    last_wincon_side = getattr(logger, "_last_wincon_side", None)
    last_escort_stage = getattr(logger, "_last_escort_stage", 0)
    last_escort_ts = getattr(logger, "_last_escort_ts", None)
    if action.card_group == "big_win_con" or action.card_id == "giant":
        setattr(logger, "_last_wincon_ts", elapsed_time)
        setattr(logger, "_last_wincon_side", action_side)
        setattr(logger, "_last_escort_stage", 0)
        setattr(logger, "_last_escort_ts", None)
    elif ally_giant_present and action.card_id == "wizard":
        if last_wincon_ts is not None and last_wincon_side == action_side and (elapsed_time - last_wincon_ts) <= 8.0:
            setattr(logger, "_last_escort_stage", 1)
            setattr(logger, "_last_escort_ts", elapsed_time)
    elif ally_giant_present and action.card_id in {"archers", "minions"}:
        if last_escort_stage == 1 and last_escort_ts is not None and (elapsed_time - last_escort_ts) <= 8.0:
            setattr(logger, "_last_escort_stage", 2)
            setattr(logger, "_last_escort_ts", elapsed_time)
    if clamped:
        logger.change_status(f"Policy: clamped coord {action.coord} -> {safe_coord}")
    logger.change_status(f"Policy: playing card {action.card_index} at {safe_coord}")
    elixir_before = elixir
    pre_card_id = identify_hand_cards(emulator, action.card_index)
    emulator.click(HAND_CARDS_COORDS[action.card_index][0], HAND_CARDS_COORDS[action.card_index][1])
    emulator.click(safe_coord[0], safe_coord[1])
    if recording_flag:
        save_play(safe_coord, action.card_index)
    interruptible_sleep(0.25)
    elixir_after = estimate_elixir(emulator)
    available_after = check_which_cards_are_available(emulator, False, False)
    post_card_id = identify_hand_cards(emulator, action.card_index)
    post_snapshot = emulator.screenshot()
    post_enemy_hp_ratio = estimate_enemy_hp_ratio(post_snapshot) if post_snapshot is not None else None
    post_player_hp_ratio = estimate_player_hp_ratio(post_snapshot) if post_snapshot is not None else None
    # Keep post-activity neutral.
    post_enemy_activity = 0.0
    invalid = (
        action.card_index in available_after
        and elixir_after >= elixir_before
        and post_card_id == pre_card_id
    )
    if trainer is not None:
        # Immediate per-play reward signal
        immediate_reward = 0.0
        if invalid:
            immediate_reward -= 0.4
        if clamped:
            immediate_reward -= 0.08
        if edge:
            immediate_reward -= 0.08
        if enemy_hp_ratio is not None and post_enemy_hp_ratio is not None:
            immediate_reward += 1.2 * (enemy_hp_ratio - post_enemy_hp_ratio)
        if player_hp_ratio is not None and post_player_hp_ratio is not None:
            immediate_reward += 1.5 * (post_player_hp_ratio - player_hp_ratio)
        elixir_cost = max(0, elixir_before - elixir_after)
        # Strongly discourage spending in calm, low-elixir states.
        if not enemy_present and elixir_before < 8.5:
            immediate_reward -= 0.06 * max(1, elixir_cost)
        # Placement shaping: prefer middle for defense, back for win-cons when calm.
        x_min, x_max, y_min, y_max = bounds
        mid_y = (y_min + y_max) // 2
        back_cut = int(y_min + (y_max - y_min) * 0.6)
        mid_low = int(y_min + (y_max - y_min) * 0.45)
        mid_high = int(y_min + (y_max - y_min) * 0.72)
        calm = not enemy_present
        under_pressure = enemy_on_our_side
        if action.card_group not in SPELL_GROUPS:
            if calm:
                if action.card_group == "big_win_con":
                    if safe_coord[1] >= back_cut:
                        immediate_reward += 0.08
                    else:
                        immediate_reward -= 0.08
                else:
                    if safe_coord[1] < mid_low:
                        immediate_reward -= 0.14
                    elif mid_low <= safe_coord[1] <= mid_high:
                        immediate_reward += 0.06
                    elif safe_coord[1] > mid_high:
                        immediate_reward += 0.02
            elif under_pressure:
                if mid_low <= safe_coord[1] <= mid_high:
                    immediate_reward += 0.06
                elif safe_coord[1] < mid_low:
                    immediate_reward -= 0.06
        # If Giant is in hand and board is calm, reward playing Giant and penalize other troop spends.
        if wincon_in_hand and calm and action.card_group not in SPELL_GROUPS:
            if action.card_id == "giant":
                immediate_reward += 0.14
            else:
                immediate_reward -= 0.08
        # Goblin Cage shaping: reward defensive usage, discourage calm-board spam.
        if action.card_id == "goblin_cage":
            if enemy_present:
                immediate_reward += 0.08
        # Tower focus: reward staying on target side when calm.
        if prefer_attack_side and action_side:
            if calm and action.card_group not in SPELL_GROUPS:
                if action_side == prefer_attack_side:
                    immediate_reward += 0.04
                else:
                    immediate_reward -= 0.06
            if not calm and action.card_group == "big_win_con":
                if action_side == prefer_attack_side:
                    immediate_reward += 0.03
                else:
                    immediate_reward -= 0.05
        # Counter-push shaping: support after a defense on same lane.
        last_defense_ts = getattr(logger, "_last_defense_ts", None)
        last_defense_side = getattr(logger, "_last_defense_side", None)
        if last_defense_ts is not None and last_defense_side == action_side:
            if (elapsed_time - last_defense_ts) <= 8.0 and calm and action.card_group not in SPELL_GROUPS:
                if safe_coord[1] >= mid_low:
                    immediate_reward += 0.05
        # Lane-aware shaping removed (pressure signals disabled).
        # Spell discipline: only reward spells that actually help.
        if action.card_group in SPELL_GROUPS:
            if enemy_hp_ratio is not None and post_enemy_hp_ratio is not None and post_enemy_hp_ratio < enemy_hp_ratio - 0.005:
                immediate_reward += 0.14
            else:
                immediate_reward -= 0.08
                # Extra penalty for spell spam when the board is calm.
                if not enemy_present:
                    immediate_reward -= 0.09
                # Prefer spells onto the weaker tower side.
                if prefer_attack_side and action_side and action_side != prefer_attack_side:
                    immediate_reward -= 0.05
                elif prefer_attack_side and action_side and action_side == prefer_attack_side:
                    immediate_reward += 0.03
            # Extra arrows penalty if no visible enemies when cast.
            if action.card_id == "arrows":
                if not enemy_present:
                    immediate_reward -= 0.08
        # Win condition shaping: reward calm, back-lane win-cons.
        if action.card_group == "big_win_con":
            if not enemy_present and safe_coord[1] >= 340:
                immediate_reward += 0.1
            if prefer_attack_side and action_side and action_side == prefer_attack_side and not enemy_present:
                immediate_reward += 0.05
        # Defensive pull awareness (centered defense when threatened).
        if enemy_on_our_side and action.card_group not in SPELL_GROUPS:
            if abs(safe_coord[0] - 210) <= 22 and safe_coord[1] >= 330:
                immediate_reward += 0.06
            elif safe_coord[1] >= 330 and abs(safe_coord[0] - 210) > 35:
                immediate_reward -= 0.05
        # Extra reward for center defense when threatened.
        if enemy_on_our_side and action.card_group not in SPELL_GROUPS:
            if mid_low <= safe_coord[1] <= mid_high and abs(safe_coord[0] - mid_x) <= 28:
                immediate_reward += 0.04
        if action.card_group not in FORWARD_ALLOWED_GROUPS and not enemy_present:
            if safe_coord[1] < 320:
                immediate_reward -= 0.10
        # Mark a defensive play for counter-push logic.
        if enemy_on_our_side and action.card_group not in SPELL_GROUPS:
            if safe_coord[1] >= mid_low:
                setattr(logger, "_last_defense_ts", elapsed_time)
                setattr(logger, "_last_defense_side", action_side)
        trainer.update_step(state_vec, action, immediate_reward * REWARD_SCALE)
        trainer.record(
            state_vec,
            action,
            elapsed_time=elapsed_time,
            invalid=invalid,
            clamped=clamped,
            edge=edge,
            crowns=crowns_now,
            enemy_hp=enemy_hp_ratio,
            player_hp=player_hp_ratio,
            enemy_activity=enemy_activity,
            enemy_activity_after=post_enemy_activity,
            enemy_hp_after=post_enemy_hp_ratio,
            player_hp_after=post_player_hp_ratio,
            elixir_after=elixir_after,
            target_side=prefer_attack_side,
            action_side=action_side,
            wincon_available=wincon_available,
            player_left_delta=player_left_delta,
            player_right_delta=player_right_delta,
            player_king_delta=player_king_delta,
            enemy_left_delta=enemy_left_delta,
            enemy_right_delta=enemy_right_delta,
            enemy_king_delta=enemy_king_delta,
            lane_bias=lane_bias,
            defend_mode=defend_mode,
            card_id=action.card_id,
        )
    logger.add_card_played()
    # Emotes disabled
    return True


class BattleStrategy:
    """Manages battle timing and elixir selection strategy.

    Encapsulates the sophisticated elixir selection logic that changes
    based on battle phase, eliminating the need for global variables.
    """

    def __init__(self):
        self.start_time = None
        self.elixir_amounts = [3, 4, 5, 6, 7, 8, 9]

        # Strategy weights for each battle phase
        self.phase_strategies = {
            "early": [
                0,
                0,
                0,
                0,
                0.3,
                0.3,
                0.4,
            ],  # 0-7s: Conservative, wait for more elixir
            "single": [
                0.05,
                0.05,
                0.1,
                0.15,
                0.15,
                0.3,
                0.2,
            ],  # 7-90s: Balanced distribution
            "double": [
                0.05,
                0.05,
                0.1,
                0.15,
                0.25,
                0.3,
                0.1,
            ],  # 90-200s: Favor 7-8 elixir
            "triple": [
                0.05,
                0.05,
                0.1,
                0.1,
                0.3,
                0.4,
                0,
            ],  # 200s+: Heavy favor 7-8, never 9
        }

        # Wait/play thresholds for each phase
        self.phase_thresholds = {
            "early": (6000, 9000),
            "single": (6000, 9000),
            "double": (7000, 10000),
            "triple": (8000, 11000),
        }

    def start_battle(self):
        """Call when battle begins to start timing."""
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Get seconds elapsed since battle start."""
        return time.time() - self.start_time if self.start_time else 0

    def get_battle_phase(self):
        """Determine current battle phase based on elapsed time."""
        elapsed = self.get_elapsed_time()
        if elapsed < 7:
            return "early"
        elif elapsed < 90:
            return "single"
        elif elapsed < 200:
            return "double"
        else:
            return "triple"

    def select_elixir_amount(self):
        """Select elixir amount to wait for based on current battle phase."""
        phase = self.get_battle_phase()
        weights = self.phase_strategies[phase]
        return random.choices(self.elixir_amounts, weights=weights, k=1)[0]

    def get_thresholds(self):
        """Get (WAIT_THRESHOLD, PLAY_THRESHOLD) for current battle phase."""
        phase = self.get_battle_phase()
        return self.phase_thresholds[phase]


def _fight_loop(emulator, logger: Logger, recording_flag: bool, policy_cfg: dict | None = None) -> bool:
    """Method for handling dynamically timed fight"""
    create_default_bridge_iar(emulator)
    collections.deque(maxlen=3)
    prev_cards_played = logger.get_cards_played()
    battle_detection_lost_count = 0

    # Initialize battle strategy and start timing
    battle_strategy = BattleStrategy()
    battle_strategy.start_battle()

    policy = None
    trainer = None
    if policy_cfg and policy_cfg.get("enabled"):
        if not hasattr(logger, "_field_detector"):
            logger._field_detector = load_detector_from_env()
        logger.log(
            "[PolicyCfg] "
            f"min_elixir={policy_cfg.get('min_elixir')} "
            f"strict={policy_cfg.get('strict_elixir')} "
            f"spell_gate={policy_cfg.get('spell_gate')}"
        )
        model = load_torch_model(
            policy_cfg.get("model_path"),
            logger=logger,
            trainable=policy_cfg.get("train", False),
        )
        policy = EpsilonGreedyPolicy(model=model, epsilon=policy_cfg.get("epsilon", 0.1))
        setattr(logger, "_policy_base_epsilon", policy_cfg.get("epsilon", 0.1))
        setattr(logger, "_policy_min_epsilon", 0.05)
        if policy_cfg.get("train", False):
            trainer = OnlineTrainer(
                model=model,
                lr=policy_cfg.get("lr", 5e-4),
                model_path=policy_cfg.get("model_path"),
                logger=logger,
            )
            logger.log(f"[Policy] Enabled with online training (epsilon={policy.epsilon})")
        else:
            logger.log(f"[Policy] Enabled (epsilon={policy.epsilon})")
        setattr(logger, "policy_trainer", trainer)

    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}); waiting it out.",
            )

            # If we've lost detection several times in a row, assume the battle
            # ended even if we couldn't confirm it (prevents infinite loops if UI changes).
            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly; assuming battle ended.",
                )
                break

            interruptible_sleep(1)
            continue

        battle_detection_lost_count = 0
        # debug screenshot saving removed from production

        # Get elixir amount and thresholds based on current battle phase
        elixir_amount = battle_strategy.select_elixir_amount()
        wait_threshold, play_threshold = battle_strategy.get_thresholds()

        wait_output = wait_for_elixer(
            emulator,
            logger,
            elixir_amount,
            wait_threshold,
            play_threshold,
            recording_flag,
        )

        if wait_output == "restart":
            logger.change_status("Failure while waiting for elixir")
            return False

        if wait_output == "no battle":
            logger.change_status("Not in battle anymore!")
            break

        if not check_if_in_battle(emulator):
            if check_if_battle_has_ended(emulator):
                logger.change_status("Not in a battle anymore (confirmed)")
                break

            logger.change_status("Lost battle detection; continuing fight loop.")
            continue

        play_start_time = time.time()
        played = False
        if policy is not None:
            played = play_a_card_with_policy(
                emulator,
                logger,
                recording_flag,
                battle_strategy,
                policy,
                trainer,
                free_placement=bool(policy_cfg.get("free_placement", False)),
                min_elixir=int(policy_cfg.get("min_elixir", 0)),
                allow_hold=bool(policy_cfg.get("allow_hold", True)),
                strict_elixir=bool(policy_cfg.get("strict_elixir", False)),
                spell_gate=bool(policy_cfg.get("spell_gate", False)),
                debug_detection=bool(policy_cfg.get("debug_detection", False)),
                debug_overlay=bool(policy_cfg.get("debug_overlay", False)),
                debug_hp_log=bool(policy_cfg.get("debug_hp_log", False)),
                debug_hand_conf=bool(policy_cfg.get("debug_hand_conf", False)),
            )
        if not played:
            if play_a_card(emulator, logger, recording_flag, battle_strategy) is False:
                logger.change_status("Failed to play a card, retrying...")
        # play_time_taken = str(time.time() - play_start_time)[:4]
        logger.change_status(
            f"Made a play in {str(time.time() - play_start_time)[:4]}s",
        )

    logger.change_status("End of the fight!")
    interruptible_sleep(2.13)
    cards_played = logger.get_cards_played()
    logger.change_status(f"Played ~{cards_played - prev_cards_played} cards this fight")

    return True


def _random_fight_loop(emulator, logger) -> bool:
    """Method for handling dynamically timed fight with random plays"""
    logger.change_status(status="Starting battle with random plays")
    fight_timeout = 5 * 60  # 5 minutes
    start_time = time.time()
    battle_detection_lost_count = 0

    # while in battle:
    while True:
        if not check_for_in_battle_with_delay(emulator):
            if check_if_battle_has_ended(emulator):
                break

            battle_detection_lost_count += 1
            logger.change_status(
                f"Lost battle detection mid-fight ({battle_detection_lost_count}); waiting it out.",
            )

            if battle_detection_lost_count >= 4:
                logger.change_status(
                    "Lost battle detection repeatedly; assuming battle ended.",
                )
                break

            interruptible_sleep(1)
            continue

        battle_detection_lost_count = 0
        if time.time() - start_time > fight_timeout:
            logger.change_status("_random_fight_loop() timed out. Breaking")
            return False

        mag_dump(emulator, logger)
        for _ in range(random.randint(1, 3)):
            logger.add_card_played()

        interruptible_sleep(8)

    logger.change_status("Finished with battle with random plays...")
    return True


if __name__ == "__main__":
    pass
