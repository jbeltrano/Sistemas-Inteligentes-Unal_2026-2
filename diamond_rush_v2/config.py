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
    "closed_gate",
    "open_gate",
    "totem",
]
TEMPLATE_VARIANTS = {
    "player": ["player", "player_key"],
}

THRESHOLD = 0.90
PLAYER_THRESHOLD = 0.75
MAX_OVERLAP = 0.4
# Tablero: 15 filas x 10 columnas de celdas cuadradas. El tamaño de
# celda se deriva de esto y del tamaño del frame (no se estima).
BOARD_ROWS = 15
BOARD_COLS = 10
# Rango razonable de tamaño de celda en píxeles; fuera de él algo anda
# mal con el frame (captura recortada, zoom distinto, etc.).
MIN_TILE = 40
MAX_TILE = 100
MIN_PATH_FRACTION = 0.5
# Fracción de cada lado de la celda que se ignora al medir cuánto suelo
# tiene (los bordes mezclan suelo y pared).
CELL_MARGIN = 0.2
# El HUD se dibuja ENCIMA del tablero (no lo desplaza). Se usa para
# descartar detecciones del HUD y celdas tapadas por él.
HUD_HEIGHT = 150

GAME_URL = "https://www.minijuegos.com/embed/diamond-rush"
IFRAME_ID = "game-player"
DEBUGGER_ADDRESS = "127.0.0.1:9222"
WINDOW_SIZE = "900,1050"

# ---------------------------------------------------------------
# Planificación (búsqueda en el espacio de mundos)
# ---------------------------------------------------------------
MAX_NODES_ASTAR = 400_000      # presupuesto de nodos para A*
MAX_NODES_BACKTRACK = 120_000  # presupuesto de nodos para el retroceso (DFS)
MAX_DEPTH = 900                # tope de profundidad del retroceso
# A* ponderado: f = g + WEIGHT * h. Con WEIGHT>1 el sesgo hacia la meta
# es mas fuerte (encuentra plan antes, a cambio de no ser optimo). Ayuda
# en los puzzles tipo Sokoban de los niveles finales.
A_STAR_WEIGHT = 2.0
# Presupuesto de A* para subobjetivos de relleno de hoyo (locales).
MAX_NODES_HOLE = 60_000

# ---------------------------------------------------------------
# Control (teclado vía Selenium)
# ---------------------------------------------------------------
# El juego se maneja con las flechas. Cada dirección del plan es una
# flecha que se manda al canvas por Selenium (no hay scraping: solo
# pulsaciones).
STEP_SECONDS = 0.40            # pausa entre pasos del plan (en segundos)
PRESS_SECONDS = 0.08           # cuánto se mantiene la flecha presionada (el juego exige un mínimo)
VERIFY_EACH_STEP = True        # re-percibir tras cada paso y re-planificar si hay desvío
MAX_REPLANS = 50               # tope de re-planificaciones seguidas antes de abortar
# Asentado tras cada paso: se re-percibe hasta que la celda del personaje
# deje de cambiar, para no leer un frame a mitad de la animación y creer
# que hubo desvío (lo que reenviaría la misma flecha = doble movimiento).
SETTLE_TIMEOUT = 1.0           # tiempo máximo de espera del asentado (segundos)
SETTLE_POLL = 0.05             # pausa entre lecturas del asentado (segundos)
