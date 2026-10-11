"""
Captura de un único screenshot del juego.

Dos formas de obtenerlo:

- leer_imagen(ruta): archivo PNG/JPG (el modo por defecto; los niveles
  estaticos de levels/ y captura.png entran por aca).
- capturar_pantalla(region): un solo frame de la pantalla actual via
  mss (BGRA -> BGR). El PLAN se calcula con UNA captura y no requiere
  repetir esta toma: la captura en vivo solo se reutiliza como
  verificacion opcional de progreso.

Regla anti-scraping: aca solo se leen pixeles del juego, nunca se toca
HTTP/DOM/JS del sitio.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import IMAGEN_DEFECTO, BASE_DIR


def leer_imagen(ruta) -> np.ndarray:
    """Lee una imagen a BGR. Acepta str o Path (None -> captura.png)."""
    if ruta is None:
        ruta = IMAGEN_DEFECTO
    ruta = Path(ruta)
    frame = cv2.imread(str(ruta))
    if frame is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return frame


def capturar_pantalla(region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Un solo frame de la pantalla (mss). `region` = (x, y, ancho, alto)
    o None para la pantalla completa. BGR como cv2.imread.
    """
    try:
        import mss
    except ImportError:
        raise RuntimeError(
            "capturar_pantalla necesita 'mss' (pip install mss). "
            "O pasale --imagen con una captura."
        ) from None

    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[0]
        else:
            x, y, w, h = region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        shot = sct.grab(monitor)
        frame = np.asarray(shot, dtype=np.uint8)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def guardar_debug(frame: np.ndarray, nombre: str = "debug_frame.png") -> Path:
    """Guarda el frame percibido (solo para revisar calibraciones)."""
    ruta = BASE_DIR / nombre
    cv2.imwrite(str(ruta), frame)
    return ruta