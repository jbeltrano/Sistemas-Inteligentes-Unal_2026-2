"""
Configuración central del agente de Diamond Rush.

Todo lo que hay que calibrar (umbrales, geometría del tablero,
mapeo template -> objeto, teclas, tiempos) vive acá para no tenerlo
esparcido por el resto de módulos.
"""
from pathlib import Path

# ---------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
IMAGEN_DEFECTO = BASE_DIR / "captura.png"

# ---------------------------------------------------------------
# Geometría del tablero
# ---------------------------------------------------------------
# El tablero de Diamond Rush es una retícula de FILAS x COLUMNAS
# celdas cuadradas. El tamaño de celda se deriva del frame (dividir
# ancho/alto entre las dimensiones del tablero); si la captura no es
# del canvas sino de la ventana entera, conviene recortarla o
# setear TILE_SIZE a mano (0 = sacarlo automáticamente).
FILAS = 15
COLUMNAS = 10
TILE_SIZE = 0          # 0 -> geom. automática desde el frame
ORIGEN = None          # (x, y) manual; None -> centrado con TILE_SIZE
MIN_TILE = 30
MAX_TILE = 110
CELDA_MARGEN = 0.2     # fracción del lado de la celda que se ignora al medir suelo
MIN_FRACCION_CAMINO = 0.5
ALTURA_HUD = 150       # el HUD se dibuja encima del tablero; se descarta

# ---------------------------------------------------------------
# Reconocimiento de objetos (template matching)
# ---------------------------------------------------------------
# Mapeo nombre lógico -> archivo del template. Ver README/AGENTS: los
# templates son recortes a mano y pueden dejar de matchear si el juego
# se ve distinto; hay que recalibrarlos por nivel.
TEMPLATES = {
    "personaje": "personaje.png",
    "diamante": "diamond.png",
    "puerta_llave": "door_key.png",
    "hueco": "hueco.png",
    "llave": "key.png",
    "pico": "pic.png",            # picos/púa: peligro (por calibrar)
    "roca_en_hoyo": "rock_in.png",
    "roca": "rock.png",
    "salida_bloqueada": "scape_block.png",
    "salida_abierta": "scape_enable.png",
    "piso_seguro": "no_picos.png",
    "interruptor": "button.png",
    "lava": "lava.png",
    "reja": "reja.png",
    "totem": "totem.png",
}
# Objetos con más de un template que cuentan como el mismo objeto
# lógico (p. ej. personaje con llave en la mano).
VARIANTES = {"personaje": ["personaje.png"]}

UMBRAL = 0.90
UMBRAL_JUGADOR = 0.75     # el personaje se anima; umbral más laxo para él
SOLAPE_MAX = 0.4

# ---------------------------------------------------------------
# Control (teclado)
# ---------------------------------------------------------------
TECLAS = {
    "arriba": "Up",
    "abajo": "Down",
    "izquierda": "Left",
    "derecha": "Right",
}
BACKEND_TECLADO = None     # None = autodetección (pynput -> xdotool)
PASO_SEG = 0.40            # pausa entre pasos del plan (en segundos)
PRENSA_SEG = 0.08          # cuánto mantener la tecla presionada
VENTANA_ID = None          # id xdotool de la ventana del juego (None = foco actual)

# ---------------------------------------------------------------
# Verificación de progreso
# ---------------------------------------------------------------
# El plan se calcula con UNA captura. La verificación ES opcional:
# si se provee un capturador en vivo, se re-percibe y se compara la
# celda del personaje contra la esperada; si no, la ejecución se hace
# solo por tiempos.
VERIFICAR_A_CADA_PASO = False
VERIFICAR_MILESTONES = True

# ---------------------------------------------------------------
# Planificación
# ---------------------------------------------------------------
MAX_NODOS_A = 150_000      # presupuesto de nodos para A*
MAX_NODOS_RETROCESO = 80_000
MAX_PROFUNDIDAD = 600

# ---------------------------------------------------------------
# Depuración / salida
# ---------------------------------------------------------------
DEBUG_FRAME = None         # guardar el frame percibido como debug_*.png