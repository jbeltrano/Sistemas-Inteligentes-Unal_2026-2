"""
Reconocimiento de un frame: matching de templates, máscara de caminos
y grilla. Punto de entrada: reconocer(frame) -> EstadoJuego.

Es el refactor de Search_objects.py en funciones reutilizables: la
lógica es la misma (matchTemplate + NMS + HSV del suelo), pero aca no
escribe archivos en disco ni imprime — devuelve el estado que va a
consumir el planificador.
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import config
from percepcion.estado import EstadoJuego, ObjetoDetectado, construir_estado

Celda = Tuple[int, int]


class TemplateMatcher:
    """Busca un template dentro de un frame y devuelve sus coincidencias."""

    def __init__(self, template_path: Path, frame: np.ndarray, nombre: str = None, voltear: bool = False):
        """
        `nombre` es el nombre lógico del objeto (p. ej. "diamante"),
        separado del nombre del archivo del template. Asi varios
        templates distintos pueden contar como el mismo objeto.
        `voltear` prueba tambien el template espejado (para el personaje).
        """
        self.frame = frame
        self.template_path = template_path
        self.voltear = voltear
        self.nombre = nombre or Path(template_path).stem

        self.template_image = cv2.imread(str(template_path))
        if self.template_image is None:
            raise FileNotFoundError(f"No se pudo cargar el template: {template_path}")

    @property
    def tamano(self) -> Tuple[int, int]:
        """(alto, ancho) del template."""
        return self.template_image.shape[:2]

    def match(self, umbral: float, solape_max: float) -> List[Tuple[int, int, float]]:
        """
        Coincidencias como (x, y, score) ordenadas de mejor a peor,
        con NMS solo dentro de este template (colapsa la meseta de
        pixeles sobre el umbral); la competencia entre templates
        distintos la resuelve _resolver_solapes().
        """
        resultado = cv2.matchTemplate(self.frame, self.template_image, cv2.TM_CCOEFF_NORMED)

        if self.voltear:
            volteado = cv2.matchTemplate(
                self.frame, cv2.flip(self.template_image, 1), cv2.TM_CCOEFF_NORMED
            )
            resultado = np.maximum(resultado, volteado)

        ys, xs = (resultado >= umbral).nonzero()
        scores = resultado[ys, xs]
        alto, ancho = self.tamano

        candidatos = sorted(zip(xs, ys, scores), key=lambda p: p[2], reverse=True)

        elegidos: List[Tuple[int, int, float]] = []
        for x, y, score in candidatos:
            if all(
                _fraccion_solape(x, y, ex, ey, ancho, alto) <= solape_max
                for ex, ey, _ in elegidos
            ):
                elegidos.append((int(x), int(y), float(score)))

        return elegidos


def _fraccion_solape(x, y, ox, oy, ancho, alto) -> float:
    """Fraccion del area de una caja ancho x alto compartida por dos coincidencias."""
    ancho_ol = max(0, min(x + ancho, ox + ancho) - max(x, ox))
    alto_ol = max(0, min(y + alto, oy + alto) - max(y, oy))
    return (ancho_ol * alto_ol) / (ancho * alto)


def _resolver_solapes(candidatos, solape_max) -> List[ObjetoDetectado]:
    """
    Suprime coincidencias que se solapan fuerte sin importar de que
    template vinieron. Necesario porque varios templates representan
    estados alternativos de un mismo objeto fisico (roca/roca_en_hoyo,
    salida_bloqueada/salida_abierta, pico/piso_seguro): si ambos
    matchean el mismo lugar, queda el de mejor score en este frame.
    """
    ordenados = sorted(candidatos, key=lambda c: c[5], reverse=True)
    elegidos: List[ObjetoDetectado] = []

    for nombre, x, y, a, h, _score in ordenados:
        if all(
            _fraccion_solape(x, y, obj.x, obj.y, a, h) <= solape_max
            for obj in elegidos
        ):
            elegidos.append(ObjetoDetectado(nombre, x, y, a, h))

    return elegidos


def detectar_caminos(frame: np.ndarray) -> np.ndarray:
    """
    Mascara binaria de suelo/camino (blanco = camino, negro = obstaculo).
    El suelo del juego es marron oscuro; los valores HSV son un punto
    de partida y conviene ajustarlos viendo la mascara resultante.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 40, 20])
    upper = np.array([30, 220, 150])
    mascara = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    return mascara


def cambiar_geometria_tablero(frame: np.ndarray) -> Tuple[int, int, int]:
    """
    (tile_size, origin_x, origin_y) a partir del frame.

    El tablero es de FILAS x COLUMNAS celdas cuadradas; el tamao de
    celda sale de dividir el frame y la grilla se centra en el
    sobrante. Si config.TILE_SIZE no es 0 se usa ese (y ORIGEN si esta
    seteado), para capturas de ventana completa que no son el canvas.
    """
    alto, ancho = frame.shape[:2]

    if config.TILE_SIZE:
        tile = config.TILE_SIZE
    else:
        tile = min(ancho // config.COLUMNAS, alto // config.FILAS)

    if not config.MIN_TILE <= tile <= config.MAX_TILE:
        raise RuntimeError(
            f"Celda de {tile}px no plausible para frame {ancho}x{alto} con "
            f"tablero {config.FILAS}x{config.COLUMNAS}. "
            f"Ajusta config.TILE_SIZE si la captura no es el canvas."
        )

    if config.ORIGEN is not None:
        origin_x, origin_y = config.ORIGEN
    else:
        origin_x = (ancho - tile * config.COLUMNAS) // 2
        origin_y = (alto - tile * config.FILAS) // 2

    return tile, origin_x, origin_y


def _fraccion_camino(mascara: np.ndarray, x: int, y: int, tile: int) -> float:
    """
    Proporcion de suelo en la zona central de una celda (ignora un
    margen CELDA_MARGEN por lado). Fuera de la imagen devuelve 0.0.
    """
    alto, ancho = mascara.shape[:2]
    if x < 0 or y < 0 or x + tile > ancho or y + tile > alto:
        return 0.0

    margen = int(tile * config.CELDA_MARGEN)
    interior = mascara[y + margen : y + tile - margen, x + margen : x + tile - margen]
    if interior.size == 0:
        return 0.0
    return float(np.mean(interior > 0))


def crear_grid(mascara: np.ndarray, tile: int, origin_x: int, origin_y: int) -> np.ndarray:
    """
    Grilla FILAS x COLUMNAS (1 = suelo, 0 = obstaculo) a partir de la
    mascara de caminos. Vertebra la conversion de Search_objects.py:
    una celda es suelo si su zona central supera MIN_FRACCION_CAMINO.
    Las celdas cuyo centro queda bajo el HUD nunca son suelo.
    """
    grilla = np.zeros((config.FILAS, config.COLUMNAS), dtype=np.uint8)

    for i in range(config.FILAS):
        for j in range(config.COLUMNAS):
            x = origin_x + j * tile
            y = origin_y + i * tile

            if y + tile // 2 < config.ALTURA_HUD:
                continue

            if _fraccion_camino(mascara, x, y, tile) > config.MIN_FRACCION_CAMINO:
                grilla[i, j] = 1

    return grilla


def _cargar_matchers(frame: np.ndarray) -> List[TemplateMatcher]:
    matchers: List[TemplateMatcher] = []
    for nombre, archivo in config.TEMPLATES.items():
        for variante in config.VARIANTES.get(nombre, [archivo]):
            matchers.append(
                TemplateMatcher(
                    config.TEMPLATES_DIR / variante,
                    frame,
                    nombre=nombre,
                    voltear=(nombre == "personaje"),
                )
            )
    return matchers


def reconocer(frame: np.ndarray) -> EstadoJuego:
    """
    Punto de entrada: corre todo el pipeline de reconocimiento sobre
    un frame y arma el EstadoJuego que consume la decision.
    """
    candidatos: List[Tuple[str, int, int, int, int, float]] = []
    for matcher in _cargar_matchers(frame):
        alto, ancho = matcher.tamano
        umbral = config.UMBRAL_JUGADOR if matcher.nombre == "personaje" else config.UMBRAL
        for x, y, score in matcher.match(umbral, config.SOLAPE_MAX):
            candidatos.append((matcher.nombre, x, y, ancho, alto, score))

    # El personaje se resuelve aparte y primero: no puede perder su lugar
    # contra otro template (p. ej. el item que lleva en la mano).
    del_personaje = [c for c in candidatos if c[0] == "personaje"]
    otros = [c for c in candidatos if c[0] != "personaje"]

    personajes = _resolver_solapes(del_personaje, config.SOLAPE_MAX)
    if not personajes:
        raise RuntimeError("No se encontró el personaje en el frame")

    personaje = personajes[0]
    detecciones = [personaje] + _resolver_solapes(otros, config.SOLAPE_MAX)

    # El contador de diamantes del HUD matchea el template de diamante;
    # se descarta cualquier deteccion por arriba de la franja del HUD.
    detecciones = [obj for obj in detecciones if obj.centro[1] >= config.ALTURA_HUD]

    tile, origin_x, origin_y = cambiar_geometria_tablero(frame)
    mascara = detectar_caminos(frame)
    grilla = crear_grid(mascara, tile, origin_x, origin_y)

    def en_tablero(celda: Celda) -> bool:
        return 0 <= celda[0] < config.FILAS and 0 <= celda[1] < config.COLUMNAS

    detecciones = [
        obj
        for obj in detecciones
        if en_tablero(
            (
                (obj.centro[1] - origin_y) // tile,
                (obj.centro[0] - origin_x) // tile,
            )
        )
    ]

    player_cell: Celda = (
        (personaje.centro[1] - origin_y) // tile,
        (personaje.centro[0] - origin_x) // tile,
    )

    return construir_estado(
        grilla,
        detecciones,
        player_cell,
        origin_x,
        origin_y,
        tile,
    )


def imprimir_estado(estado: EstadoJuego) -> None:
    """Vuelca el estado para verificar la percepcion a ojo (calibracion)."""
    print("=" * 62)
    print("EstadoJuego")
    print("=" * 62)

    print(f"grid ({estado.grid.shape[0]}x{estado.grid.shape[1]}):")
    for fila in estado.grid.tolist():
        print("  " + "".join("#" if c else "." for c in fila))

    print(f"\npersonaje en celda: {estado.player_cell}")

    print(f"\ndetecciones ({len(estado.objects)}):")
    for obj in estado.objects:
        print(
            f"  {obj.nombre:<18} pixel=({obj.x},{obj.y}) "
            f"tamano={obj.ancho}x{obj.alto} centro={obj.centro}"
        )

    print("\nobjetos por celda:")
    for celda, objs in sorted(estado.objects_by_cell.items()):
        print(f"  {celda}: {[o.nombre for o in objs]}")

    print(
        f"\norigin=({estado.origin_x},{estado.origin_y}) tile={estado.tile_size}"
    )
    print("=" * 62)