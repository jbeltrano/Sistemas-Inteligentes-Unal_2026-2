from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"

TEMPLATES_DIR = ASSETS_DIR / "templates"
LEVELS_DIR = ASSETS_DIR / "levels"

DEBUG_DIR = BASE_DIR / "debug-results"

PLAYER_NAME = "player"
TEMPLATE_NAMES = [
    "player",
    "diamond",
    "door_key",
    "hole",
    "key",
    "spike",
    "rock_in_hole",
    "rock",
    "scape_block",
    "scape_enable",
    "no_spike",
    "button",
    "lava",
    "closed_gate",
    "open_gate",
    "totem",
]

THRESHOLD = 0.90
MAX_OVERLAP = 0.4
MIN_TILE = 50
MAX_TILE = 70
DEFAULT_TILE = 60
ANALYSIS_RANGE = 8
AMBIGUITY_THRESHOLD = 0.15
MIN_PATH_FRACTION = 0.5
HUD_HEIGHT = 150
