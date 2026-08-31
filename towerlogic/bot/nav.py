import random
import time
from typing import Literal

import cv2

from towerlogic.bot.constants import CLASH_MAIN_DEADSPACE_COORD as CLASH_MAIN_MENU_DEADSPACE_COORD
from towerlogic.detection.image_rec import (
    all_pixels_are_equal,
    find_image,
    pixel_is_equal,
)
from towerlogic.utils.cancellation import interruptible_sleep
from towerlogic.utils.logger import Logger

CLASH_MAIN_OPTIONS_BURGER_BUTTON = (390, 62)
BATTLE_LOG_BUTTON = (241, 43)
CARD_PAGE_ICON_FROM_CLASH_MAIN = (108, 598)
CARD_PAGE_ICON_FROM_CARD_PAGE = (147, 598)
OK_BUTTON_COORDS_IN_TROPHY_REWARD_PAGE = (209, 599)
CLASH_MAIN_WAIT_TIMEOUT = 240  # s


def wait_for_battle_start(emulator, logger, timeout: int = 120) -> bool:
    """Waits for any battle to start (1v1 or 2v2).

    Args:
    ----
        emulator: The emulator controller.
        logger: The logger object.
        timeout: Maximum time to wait in seconds

    Returns:
    -------
        bool: True if battle started, False if timed out.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        time_taken = str(time.time() - start_time)[:4]
        logger.change_status(
            status=f"Waiting for battle to start for {time_taken}s",
        )

        # NOTE: Debug screenshot saving was intentionally removed from
        # the production flow. If you need screenshots for debugging,
        # use the recorder helpers directly in a temporary script or
        # enable a local-only change — do not commit such changes.

        battle_result = check_if_in_battle(emulator)

        if battle_result:  # True for any battle type
            logger.change_status("Detected an ongoing battle!")
            return True

        emulator.click(x_coord=20, y_coord=200)

    return False


def check_for_in_battle_with_delay(emulator) -> bool:
    """Checks if the virtual machine is in any battle with a delay.

    Args:
    ----
        emulator: The emulator controller.

    Returns:
    -------
        bool: True if the virtual machine is in any battle, False otherwise.

    """
    timeout = 3  # s
    start_time = time.time()
    while time.time() - start_time < timeout:
        battle_result = check_if_in_battle(emulator)
        if battle_result:  # True for any battle type ("1v1", "2v2")
            return True
        interruptible_sleep(0.1)
    return False


def check_if_in_battle(emulator, image=None):
    iar_bgr = image if image is not None else emulator.screenshot()
    if iar_bgr is None:
        return False

    # Convert to RGB for easier reasoning about expected colors.
    iar = iar_bgr[..., ::-1]

    def get_pixel(y: int, x: int) -> list[int] | None:
        if y >= iar.shape[0] or x >= iar.shape[1]:
            return None
        return iar[y][x].tolist()

    def is_bright(pixel: list[int] | None, threshold: int = 180) -> bool:
        if pixel is None:
            return False
        return all(channel >= threshold for channel in pixel) or (
            150 <= pixel[0] <= 195 and 40 <= pixel[1] <= 60 and 40 <= pixel[2] <= 60
        )

    def is_filled_crown(pixel: list[int] | None) -> bool:
        """Check if pixel is a gold/yellow filled crown or UI accent."""
        if pixel is None:
            return False
        r, g, b = pixel
        return r >= 170 and g >= 130 and b <= 140

    def is_scoreboard_purple(pixel: list[int] | None) -> bool:
        if pixel is None:
            return False
        r, g, b = pixel
        return r >= 200 and b >= 200 and g <= 140

    def check_mode(coords: list[tuple[int, int]]) -> bool:
        pixels = [get_pixel(y, x) for y, x in coords]
        # Allow one of the "bright" UI pixels to change (crowns filling, overlays,
        # small rendering differences) while still requiring the purple scoreboard.
        bright_required = max(1, len(coords) - 2)
        bright_count = sum(1 for pixel in pixels[:-1] if is_bright(pixel) or is_filled_crown(pixel))
        return bright_count >= bright_required and is_scoreboard_purple(pixels[-1])

    # When the emote is closed these pixels are not considered bright:
    # (533, 80) RGB=[157, 44, 44] and (532, 77) RGB=[187, 55, 55]
    coords_1v1 = [(528, 49), (532, 77), (546, 52), (546, 77), (618, 115)]
    coords_2v2 = [(534, 53), (533, 80), (548, 52), (548, 76), (615, 114)]

    if check_mode(coords_1v1):
        return True
    if check_mode(coords_2v2):
        return True

    return False


def check_for_post_battle_button(emulator) -> bool:
    """Checks for the post-battle OK/Exit buttons (battle-end screen)."""
    image = emulator.screenshot()
    if image is None:
        return False

    if find_image(image, "ok_post_battle_button", tolerance=0.85) is not None:
        return True

    if find_image(image, "exit_battle_button", tolerance=0.9) is not None:
        return True

    return False


def check_if_battle_has_ended(emulator) -> bool:
    """Best-effort confirmation that a battle ended (avoid false positives mid-fight)."""
    if check_if_on_clash_main_menu(emulator):
        return True

    if check_for_trophy_reward_menu(emulator):
        return True

    if check_for_post_battle_button(emulator):
        return True

    return False


def check_for_trophy_reward_menu(emulator) -> bool:
    iar = emulator.screenshot()

    pixels = [
        iar[592][172],
        iar[617][180],
        iar[607][190],
        iar[603][200],
        iar[596][210],
        iar[593][220],
        iar[600][230],
        iar[610][235],
        iar[623][246],
    ]
    colors = [
        [255, 184, 68],
        [255, 175, 78],
        [255, 175, 78],
        [248, 239, 227],
        [255, 187, 104],
        [255, 176, 79],
        [255, 187, 104],
        [255, 175, 78],
        [253, 135, 39],
    ]

    for i, pixel in enumerate(pixels):
        if not pixel_is_equal(pixel, colors[i], tol=25):
            return False

    return True


def check_for_trophy_box_reward(emulator, image=None) -> bool:
    """Detect the modern blue trophy-box reward reveal screen."""
    image = image if image is not None else emulator.screenshot()
    if image is None or image.shape[0] < 625 or image.shape[1] < 419:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    top = hsv[0:220, 0:419]
    center = hsv[220:410, 110:310]
    bottom = hsv[535:625, 80:330]

    blue_top = float(
        ((top[..., 0] >= 95) & (top[..., 0] <= 125) & (top[..., 1] >= 100) & (top[..., 2] >= 40)).mean()
    )
    purple_center = float(
        ((center[..., 0] >= 125) & (center[..., 0] <= 170) & (center[..., 1] >= 70) & (center[..., 2] >= 80)).mean()
    )
    orange_bottom = float(
        ((bottom[..., 0] >= 5) & (bottom[..., 0] <= 30) & (bottom[..., 1] >= 100) & (bottom[..., 2] >= 120)).mean()
    )
    return blue_top >= 0.85 and purple_center >= 0.10 and orange_bottom >= 0.08


def clear_trophy_box_reward(
    emulator,
    logger: Logger,
    timeout: float = 20.0,
) -> Literal["battle", "main_menu", "timeout", "not_detected"]:
    """Click through a confirmed trophy-box flow and report its resulting state."""
    if not check_for_trophy_box_reward(emulator):
        return "not_detected"

    logger.change_status("Trophy box reward detected; clearing reward screens...")
    deadline = time.time() + timeout
    attempts = 0
    while time.time() < deadline:
        image = emulator.screenshot()
        if check_if_in_battle(emulator, image=image):
            logger.change_status(f"Battle restored after clearing trophy box ({attempts} clicks)")
            return "battle"
        legacy_menu = inspect_clash_main_menu(image)[0]
        current_menu = detect_current_clash_main_menu(image)[0]
        if legacy_menu or current_menu:
            logger.change_status(f"Main menu reached after clearing trophy box ({attempts} clicks)")
            return "main_menu"
        attempts += 1
        logger.log(f"Clearing trophy box reward (click {attempts})")
        emulator.click(209, 316)
        interruptible_sleep(0.75)

    logger.change_status(f"Timed out clearing trophy box after {attempts} clicks")
    return "timeout"


def handle_trophy_reward_menu(
    emulator,
    logger: Logger,
    printmode=False,
) -> Literal["good"]:
    if printmode:
        logger.change_status(status="Handling trophy reward menu")
    else:
        logger.log("Handling trophy reward menu")
    emulator.click(
        OK_BUTTON_COORDS_IN_TROPHY_REWARD_PAGE[0],
        OK_BUTTON_COORDS_IN_TROPHY_REWARD_PAGE[1],
    )
    interruptible_sleep(1)

    return "good"


def wait_for_clash_main_menu(emulator, logger: Logger, deadspace_click=True) -> bool:
    """Waits for the user to be on the clash main menu.
    Returns True if on main menu, prints the pixels if False then return False
    """
    start_time: float = time.time()
    while check_if_on_clash_main_menu(emulator) is not True:
        # timeout check
        if time.time() - start_time > CLASH_MAIN_WAIT_TIMEOUT:
            logger.change_status("Timed out waiting for clash main")
            break

        # handle geting stuck on trophy road screen
        if check_for_trophy_reward_menu(emulator):
            print("Handling trophy reward menu")
            handle_trophy_reward_menu(emulator, logger)
            interruptible_sleep(2)
            continue

        # click deadspace
        if deadspace_click and random.randint(0, 1) == 0:
            emulator.click(
                CLASH_MAIN_MENU_DEADSPACE_COORD[0],
                CLASH_MAIN_MENU_DEADSPACE_COORD[1],
            )
        interruptible_sleep(1)

    interruptible_sleep(1)
    if check_if_on_clash_main_menu(emulator) is not True:
        print("Failed to get to clash main! Saw these pixels before restarting:")
        return False

    return True


def inspect_clash_main_menu(image) -> tuple[bool, list[list[int]], list[list[bool]]]:
    """Evaluate the legacy main-menu pixel signature and expose diagnostics."""
    if image is None or getattr(image, "ndim", 0) < 2:
        return False, [], []
    coords = [(14, 209), (14, 325), (19, 298), (17, 399),
              (581, 261), (584, 166), (621, 166)]
    try:
        if image.shape[0] <= max(y for y, _ in coords) or image.shape[1] <= max(x for _, x in coords):
            return False, [], []
        pixels = [image[y][x].tolist() for y, x in coords]
    except (AttributeError, IndexError, TypeError):
        return False, [], []
    # google play colors
    colors_1 = [
        [255, 255, 255],
        [255, 255, 255],
        [53, 199, 233],
        [25, 198, 65],
        [138, 105, 71],
        [139, 105, 72],
        [155, 120, 82],
    ]

    # memu colors
    colors_2 = [
        [255, 255, 255],
        [255, 255, 255],
        [53, 200, 233],
        [24, 199, 65],
        [138, 105, 71],
        [139, 105, 72],
        [155, 120, 81],
    ]

    # print("{:^15} | {:^15} | {:^15}".format("Seen", "Google", "Memu"))
    # for seen_pixel, google_play_color, memu_color in zip(pixels, colors_1, colors_2):
    #     seen_pixel =str(seen_pixel[0])+ ' '+ str(seen_pixel[1])+ ' '+ str(seen_pixel[2])
    #     google_play_color = (
    #         str(google_play_color[0]) + ' ' +
    #         str(google_play_color[1]) + ' ' +
    #         str(google_play_color[2])
    #     )
    #     memu_color = str(memu_color[0]) + ' ' + str(memu_color[1]) + ' ' + str(memu_color[2])
    #     print(
    #         "{:^15} | {:^15} | {:^15}".format(seen_pixel, google_play_color, memu_color)
    #     )

    matches = [[pixel_is_equal(pixel, expected, tol=25) for pixel, expected in zip(pixels, colors)]
               for colors in (colors_1, colors_2)]
    return any(all(row) for row in matches), pixels, matches


def check_if_on_clash_main_menu(emulator) -> bool:
    """Checks if the user is on the clash main menu."""
    image = emulator.screenshot()
    legacy = inspect_clash_main_menu(image)[0]
    current, _ = detect_current_clash_main_menu(image)
    return legacy or current


def detect_current_clash_main_menu(image) -> tuple[bool, dict[str, float | bool]]:
    """Detect the current home screen independently of the selected mode.

    The large Battle button and selected bottom Battle tab are shared by
    Trophy Road, Classic 1v1, and Classic 2v2.  Existing selected-mode
    templates remain useful compatibility cues, but current game art can make
    an individual mode template stale.  The legacy seven-pixel signature
    remains a separate fallback in ``inspect_clash_main_menu``.
    """
    if image is None or getattr(image, "ndim", 0) < 2:
        return False, {
            "selected_mode_template_match": False,
            "trophy_template_match": False,
            "classic_1v1_template_match": False,
            "classic_2v2_template_match": False,
            "battle_button_score": 0.0,
            "battle_tab_score": 0.0,
        }

    mode_matches = {
        "classic_1v1_template_match": find_image(
            image, "selected_1v1_on_main", tolerance=0.80,
            subcrop=(250, 430, 419, 575),
        ) is not None,
        "classic_2v2_template_match": find_image(
            image, "selected_2v2_on_main", tolerance=0.80,
            subcrop=(250, 430, 419, 575),
        ) is not None,
        "trophy_template_match": find_image(
            image, "selected_trophy_road_on_main", tolerance=0.80,
            subcrop=(250, 430, 419, 575),
        ) is not None,
    }
    selected_mode_match = any(mode_matches.values())

    # The central Battle button has a large, stable yellow/orange surface.
    region = image[455:535, 135:285]
    if region.size == 0:
        battle_score = 0.0
    else:
        # Screenshots are BGR; yellow/orange pixels have high G and R.
        battle_score = float(((region[..., 1] > 120) & (region[..., 2] > 170)).mean())

    # The selected Battle navigation tab has a broad blue/cyan surface across
    # all three modes.  Requiring it with the Battle button avoids accepting a
    # loading, reward, or battle frame based on one yellow region alone.
    tab_region = image[572:633, 145:279]
    if tab_region.size == 0:
        battle_tab_score = 0.0
    else:
        battle_tab_score = float(
            (
                (tab_region[..., 0] > 90)
                & (tab_region[..., 1] > 70)
                & (tab_region[..., 0] > tab_region[..., 2] * 1.15)
            ).mean()
        )

    details = {
        "selected_mode_template_match": selected_mode_match,
        **mode_matches,
        "battle_button_score": battle_score,
        "battle_tab_score": battle_tab_score,
    }
    supporting_cue = selected_mode_match or battle_tab_score >= 0.45
    return battle_score >= 0.20 and supporting_cue, details


def get_to_card_page_from_clash_main(
    emulator,
    logger: Logger,
) -> Literal["restart", "good"]:
    start_time = time.time()

    logger.change_status(status="Getting to card page from clash main")

    # click card page icon
    emulator.click(
        CARD_PAGE_ICON_FROM_CLASH_MAIN[0],
        CARD_PAGE_ICON_FROM_CLASH_MAIN[1],
    )
    interruptible_sleep(2.5)

    # while not on the card page, cycle the card page
    while not check_if_on_card_page(emulator):
        time_taken = time.time() - start_time
        if time_taken > 30:
            return "restart"

        emulator.click(
            CARD_PAGE_ICON_FROM_CARD_PAGE[0],
            CARD_PAGE_ICON_FROM_CARD_PAGE[1],
        )
        interruptible_sleep(3)

    logger.change_status(status="Made it to card page")

    return "good"


def check_if_on_card_page(emulator) -> bool:
    iar = emulator.screenshot()

    pixels = [
        iar[433][58],
        iar[116][59],
        iar[58][82],
        iar[64][179],
        iar[62][108],
        iar[67][146],
        iar[77][185],
        iar[77][84],
    ]

    colors1 = [
        [222, 0, 235],
        [255, 255, 255],
        [203, 137, 44],
        [195, 126, 34],
        [255, 255, 255],
        [255, 255, 255],
        [177, 103, 15],
        [178, 104, 15],
    ]

    colors2 = [
        [220, 0, 234],
        [255, 255, 255],
        [209, 68, 41],
        [202, 64, 41],
        [255, 255, 255],
        [255, 255, 255],
        [185, 52, 41],
        [185, 52, 41],
    ]

    def pixel_to_string(pixel):
        return f"[{pixel[0]},{pixel[1]},{pixel[2]}],"

    # print("{:^17} {:^17} {:^17}".format("pixel", "color1", "color2"))
    # for pixel, color1, color2 in zip(pixels, colors1, colors2):
    #     print(
    #         "{:^17} {:^17} {:^17}".format(
    #             pixel_to_string(pixel), pixel_to_string(color1), pixel_to_string(color2)
    #         )
    #     )

    if all_pixels_are_equal(pixels, colors1, tol=25):
        return True

    if all_pixels_are_equal(pixels, colors2, tol=25):
        return True

    return False


def get_to_activity_log(
    emulator,
    logger: Logger,
    printmode: bool = False,
) -> Literal["restart", "good"]:
    if printmode:
        logger.change_status(status="Getting to activity log")
    else:
        logger.log("Getting to activity log")

    # if not on main return restart
    if check_if_on_clash_main_menu(emulator) is not True:
        logger.change_status(
            status="Eror 08752389 Not on clash main menu, restarting vm",
        )
        return "restart"

    # click clash main burger options button
    if printmode:
        logger.change_status(status="Opening clash main options menu")
    else:
        logger.log("Opening clash main options menu")
    emulator.click(
        CLASH_MAIN_OPTIONS_BURGER_BUTTON[0],
        CLASH_MAIN_OPTIONS_BURGER_BUTTON[1],
    )
    if wait_for_clash_main_burger_button_options_menu(emulator, logger) == "restart":
        logger.change_status(
            status="Error 99993 Waited too long for clash main options menu, restarting vm",
        )
        return "restart"

    # click battle log button
    if printmode:
        logger.change_status(status="Clicking activity log button")
    else:
        logger.log("Clicking activity log button")
    emulator.click(BATTLE_LOG_BUTTON[0], BATTLE_LOG_BUTTON[1])
    if wait_for_battle_log_page(emulator, logger, printmode) == "restart":
        logger.change_status(
            status="Error 923593 Waited too long for battle log page, restarting vm",
        )
        return "restart"

    return "good"


def wait_for_battle_log_page(
    emulator,
    logger: Logger,
    printmode=False,
) -> Literal["restart", "good"]:
    start_time = time.time()
    if printmode:
        logger.change_status(status="Waiting for battle log page to appear")
    else:
        logger.log("Waiting for battle log page to appear")
    while not check_if_on_battle_log_page(emulator):
        time_taken = time.time() - start_time
        if time_taken > 20:
            logger.change_status(
                status="Error 2457245645 Waiting too long for battle log page",
            )
            return "restart"

    if printmode:
        logger.change_status(status="Done waiting for battle log page to appear")
    else:
        logger.log("Done waiting for battle log page to appear")

    return "good"


def check_if_on_battle_log_page(emulator) -> bool:
    iar = emulator.screenshot()

    pixels = [
        iar[72][160],
        iar[71][187],
        iar[71][197],
        iar[72][231],
        iar[73][258],
        iar[64][366],
        iar[79][365],
        iar[70][365],
        iar[62][92],
        iar[77][316],
    ]
    colors = [
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [147, 135, 254],
        [38, 38, 240],
        [255, 255, 255],
        [138, 122, 115],
        [124, 106, 99],
    ]

    for i, p in enumerate(pixels):
        if not pixel_is_equal(p, colors[i], tol=25):
            return False
    return True


def check_if_on_clash_main_burger_button_options_menu(emulator) -> bool:
    iar = emulator.screenshot()
    pixels = [
        iar[42][256],
        iar[41][275],
        iar[41][282],
        iar[42][293],
        iar[44][325],
        iar[32][239],
        iar[34][336],
        iar[50][248],
        iar[49][336],
    ]
    colors = [
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 254],
        [255, 255, 255],
        [255, 187, 105],
        [255, 187, 105],
        [255, 175, 78],
        [255, 175, 78],
    ]
    for i, color in enumerate(colors):
        if not pixel_is_equal(pixels[i], color, tol=25):
            return False
    return True


def wait_for_clash_main_burger_button_options_menu(
    emulator,
    logger: Logger,
    printmode: bool = False,
) -> Literal["restart", "good"]:
    """Waits for the virtual machine to be on the clash main burger button options menu.

    Args:
    ----
        emulator (int): The index of the virtual machine.
        logger (Logger): The logger object to use for logging.
        printmode (bool, optional): Whether to print status messages. Defaults to False.

    Returns:
    -------
        Literal["restart", "good"]: "restart" if the function timed
        out and needs to be restarted, "good" otherwise.

    """
    start_time = time.time()

    if printmode:
        logger.change_status(status="Waiting for clash main options menu to appear")
    else:
        logger.log("Waiting for clash main options menu to appear")
    while not check_if_on_clash_main_burger_button_options_menu(emulator):
        time_taken = time.time() - start_time
        if time_taken > 20:
            logger.change_status(
                status="Error 57245645362 Waiting too long for clash main options menu to appear",
            )
            return "restart"
    if printmode:
        logger.change_status(
            status="Done waiting for clash main options menu to appear",
        )
    else:
        logger.log("Done waiting for clash main options menu to appear")
    return "good"


def check_if_battle_mode_is_selected(emulator, mode: str):
    """Checks if the given battle mode is selected on the clash main menu.

    Args:
        emulator: The emulator controller.
        mode: The battle mode to check for.

    Returns:
        True if the mode is selected, False otherwise.
    """
    expected_mode_types = ["Classic 1v1", "Classic 2v2", "Trophy Road"]

    # Check if the mode is valid
    if mode not in expected_mode_types:
        print(f'[!] Fatal error: Mode "{mode}" is not a valid mode type. Expected one of {expected_mode_types}.')
        return None

    mode2folder = {
        "Classic 1v1": "selected_1v1_on_main",
        "Classic 2v2": "selected_2v2_on_main",
        "Trophy Road": "selected_trophy_road_on_main",
    }

    look_folder = mode2folder[mode]

    print(f"[DEBUG] Checking if {mode} is selected...")
    print(f"[DEBUG] Looking in folder: {look_folder}")
    print("[DEBUG] Subcrop: (270, 455, 350, 533)")

    # find image on screen
    coord = find_image(
        emulator.screenshot(),
        look_folder,
        tolerance=0.9,
        subcrop=(270, 455, 350, 533),
    )

    print(f"[DEBUG] Found at: {coord}")

    return coord is not None


def find_fight_mode_icon(emulator, mode: str):
    expected_mode_types = ["Classic 1v1", "Classic 2v2", "Trophy Road"]

    # Check if the mode is valid
    if mode not in expected_mode_types:
        print(f'[!] Fatal error: Mode "{mode}" is not a valid mode type. Expected one of {expected_mode_types}.')
        return None

    mode2folder = {
        "Classic 1v1": "fight_mode_1v1",
        "Classic 2v2": "fight_mode_2v2",
        "Trophy Road": "fight_mode_trophy_road",
    }

    look_folder = mode2folder[mode]

    image = emulator.screenshot()

    # os.makedirs('select_mode_images', exist_ok=True)
    # file_name = f'{random.randint(0,100000)}.png'
    # file_path = os.path.join('select_mode_images', file_name)
    # cv2.imwrite(file_path, image)

    fight_mode_1v1_button_location = find_image(
        image,
        look_folder,
        tolerance=0.9,
        show_image=False,
    )
    if fight_mode_1v1_button_location is not None:
        return fight_mode_1v1_button_location
    return None


def select_mode(emulator, mode: str):
    # Check if the mode is valid
    expected_mode_types = ["Classic 1v1", "Classic 2v2", "Trophy Road"]
    if type(mode) is not str:
        print(f'[!] Warning: Mode "{mode}" is not a string. Expected a string.')
        return False

    # Check if the mode is valid
    if mode not in expected_mode_types:
        print(f'[!] Warning: Mode "{mode}" is not a valid mode type. Expected one of {expected_mode_types}.')
        return False

    # must be on clash main
    if not check_if_on_clash_main_menu(emulator):
        print("[!] Not on clash main menu, cannot select a fight mode")
        return False

    # open fight type selection menu
    game_mode_coord = [308, 485]

    # click select mode button
    print("Clicking mode selection button")
    emulator.click(game_mode_coord[0], game_mode_coord[1])
    interruptible_sleep(2)

    def scroll_down_in_fight_mode_panel(emulator):
        start_y = 400
        end_y = 350
        x = 400
        emulator.swipe(x, start_y, x, end_y)
        interruptible_sleep(1)

    # scroll and search, until we find the mode in question
    search_timeout = 15  # s
    start_time = time.time()

    # Use a fixed start time so we actually time out correctly instead of
    # comparing two moving time.time() values (which would never time out).
    while time.time() - start_time < search_timeout:
        coord = find_fight_mode_icon(emulator, mode)
        if coord is not None:
            print(f'Located the "{mode}" button, clicking it.')
            emulator.click(*coord)
            interruptible_sleep(3)

            # After choosing a mode, the mode panel may remain open on some
            # devices/emulators. Click a safe deadspace coord to ensure the
            # selection panel closes and the main menu is active again so
            # subsequent actions (like pressing Start) work reliably.
            try:
                emulator.click(CLASH_MAIN_MENU_DEADSPACE_COORD[0], CLASH_MAIN_MENU_DEADSPACE_COORD[1])
            except Exception:
                # Don't fail if the click doesn't work; best-effort only.
                pass

            return True

        scroll_down_in_fight_mode_panel(emulator)

    return False


if __name__ == "__main__":
    # from towerlogic.emulators.memu import MemuEmulatorController
    # from towerlogic.utils.logger import Logger
    # import cv2
    # import os

    # print("Creating logger...")
    # logger = Logger()

    # print("Creating MEmu emulator controller in DEBUG mode (no restart)...")
    # emulator = MemuEmulatorController(logger, render_mode="directx", debug_mode=True)

    # # Save a screenshot for debugging
    # print("\nSaving screenshot for debugging...")
    # os.makedirs("debug_screenshots", exist_ok=True)
    # screenshot = emulator.screenshot()
    # cv2.imwrite("debug_screenshots/current_screen.png", screenshot)
    # print(f"Screenshot saved to: debug_screenshots/current_screen.png")
    # print(f"Screenshot size: {screenshot.shape}")

    # print("\nTesting find_fight_mode_icon...")
    # x = check_if_battle_mode_is_selected(emulator, "Classic 2v2")
    # print(f"Result: Is Classic 2v2 selected? {x}")

    # select_mode(emulator, "Classic 1v1")

    # here matt if you want to test this and see debug info
    pass
