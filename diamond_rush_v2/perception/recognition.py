"""
Reconocimiento de un frame del juego: detección de objetos por
template matching, máscara de caminos y grafo de casillas transitables.

Refactor de Search_objects.py: la lógica es la misma, pero ahora
recognize() es el único punto de entrada y devuelve un GameState en
vez de imprimir y escribir archivos en disco.
"""
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

import config
from perception.state import DetectedObject, GameState


class TemplateMatcher:
    """Busca un template dentro de un frame y devuelve sus coincidencias."""

    def __init__(self, template_path: Path, frame: np.ndarray, flip: bool = False):
        self.frame = frame
        self.template_path = template_path
        self.flip = flip
        self.name = Path(template_path).stem

        self.template_image = cv2.imread(str(self.template_path))
        self._check_image_exists(self.template_image, self.template_path)

    @staticmethod
    def _check_image_exists(image, path):
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")

    @property
    def template_size(self) -> Tuple[int, int]:
        """(alto, ancho) del template."""
        return self.template_image.shape[:2]

    def match(self, threshold: float, max_overlap: float) -> List[Tuple[int, int, float]]:
        """
        Coincidencias de este template, como (x, y, score), ordenadas
        de mejor a peor. Solo aplica NMS dentro del propio template
        (colapsa la "meseta" de píxeles sobre el umbral alrededor de
        una misma coincidencia); la resolución entre templates
        distintos que compiten por el mismo lugar la hace
        _resolve_overlaps() más abajo, con el score de por medio.
        """
        result = cv2.matchTemplate(self.frame, self.template_image, cv2.TM_CCOEFF_NORMED)

        # Si el objeto puede estar espejado, probar también el template
        # reflejado y quedarse con el mejor score de ambas orientaciones.
        if self.flip:
            flipped_result = cv2.matchTemplate(
                self.frame, cv2.flip(self.template_image, 1), cv2.TM_CCOEFF_NORMED
            )
            result = np.maximum(result, flipped_result)

        ys, xs = (result >= threshold).nonzero()
        scores = result[ys, xs]
        height, width = self.template_size

        candidates = sorted(zip(xs, ys, scores), key=lambda p: p[2], reverse=True)

        chosen: List[Tuple[int, int, float]] = []
        for x, y, score in candidates:
            if all(
                _overlap_fraction(x, y, cx, cy, width, height) <= max_overlap
                for cx, cy, _ in chosen
            ):
                chosen.append((int(x), int(y), float(score)))

        return chosen


def _overlap_fraction(x, y, other_x, other_y, width, height) -> float:
    """Fracción del área de una caja de `width`x`height` que comparten dos coincidencias."""
    overlap_width = max(0, min(x + width, other_x + width) - max(x, other_x))
    overlap_height = max(0, min(y + height, other_y + height) - max(y, other_y))
    return (overlap_width * overlap_height) / (width * height)


def _resolve_overlaps(
    candidates: List[Tuple[str, int, int, int, int, float]], max_overlap: float
) -> List[DetectedObject]:
    """
    Suprime coincidencias que se superponen fuertemente, sin importar
    de qué template vinieron.

    Esto es necesario porque varios pares de templates representan
    estados alternativos de un mismo objeto físico (rock/rock_in,
    scape_block/scape_enable, spike/no_spike): si ambos matchean el
    mismo lugar, no pueden ser dos objetos distintos, solo uno de los
    dos. En vez de intentar adivinar cuál es con reglas aparte, nos
    quedamos con el que dio mejor score en ESTE frame — es la misma
    evidencia que ya tenemos, sin necesitar memoria de frames
    anteriores.
    """
    ordered = sorted(candidates, key=lambda c: c[5], reverse=True)
    chosen: List[DetectedObject] = []

    for name, x, y, width, height, _ in ordered:
        if all(
            _overlap_fraction(x, y, obj.x, obj.y, width, height) <= max_overlap
            for obj in chosen
        ):
            chosen.append(DetectedObject(name, x, y, width, height))

    return chosen


def detect_paths(frame: np.ndarray) -> np.ndarray:
    """
    Máscara binaria de suelo/camino del nivel (blanco = camino, negro
    = obstáculo). El suelo del juego es marrón oscuro; los valores de
    HSV son un punto de partida y conviene ajustarlos viendo la
    máscara resultante.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 40, 20])
    upper = np.array([30, 220, 150])
    mask = cv2.inRange(hsv, lower, upper)

    # Limpiar pequeños agujeros/ruido.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def _path_fraction(mask: np.ndarray, x: int, y: int, tile: int) -> float:
    """
    Proporción de suelo de una celda. Si se sale de la imagen devuelve
    0.0 para que no cuente como transitable.
    """
    height, width = mask.shape[:2]

    if x < 0 or y < 0 or x + tile > width or y + tile > height:
        return 0.0

    return float(np.mean(mask[y : y + tile, x : x + tile] > 0))


def _cell_origin(center: Tuple[int, int], tile: int) -> Tuple[int, int]:
    """
    Esquina superior izquierda de la celda que contiene un punto. El
    punto queda a media celda de esa esquina.
    """
    cx, cy = center
    return (
        int(round((cx - tile / 2) / tile)) * tile,
        int(round((cy - tile / 2) / tile)) * tile,
    )


def _tiles_around(center: Tuple[int, int], tile: int, reach: int, width: int, height: int):
    """Celdas completas alrededor de un punto; descarta las que se salen de la imagen."""
    x0, y0 = _cell_origin(center, tile)
    cells = []

    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            x = x0 + j * tile
            y = y0 + i * tile

            if x >= 0 and y >= 0 and x + tile <= width and y + tile <= height:
                cells.append((x, y))

    return cells


def detect_tile_size(mask: np.ndarray, player_center: Tuple[int, int]) -> Tuple[int, float]:
    """
    Estima el paso de la retícula probando cada tamaño candidato y
    puntuándolo por qué proporción de las celdas cercanas al personaje
    queda en zona ambigua (ni suelo ni vacío). Un paso bien alineado
    deja esa proporción cerca de cero.
    """
    height, width = mask.shape[:2]

    best_tile = None
    best_score = None

    for tile in range(config.MIN_TILE, config.MAX_TILE + 1):
        cells = _tiles_around(player_center, tile, config.ANALYSIS_RANGE, width, height)

        if not cells:
            continue

        score = float(np.mean([
            0.25 < _path_fraction(mask, x, y, tile) < 0.75
            for x, y in cells
        ]))

        if best_score is None or score < best_score:
            best_tile = tile
            best_score = score

    if best_tile is None:
        return config.DEFAULT_TILE, 1.0

    if best_score > config.AMBIGUITY_THRESHOLD:
        best_tile = config.DEFAULT_TILE

    return best_tile, best_score


def create_grid(mask: np.ndarray, tile: int) -> Tuple[np.ndarray, int, int]:
    """
    Convierte la máscara de caminos en una grilla de celdas de lado
    `tile`.

    La grilla se ancla a la esquina superior izquierda de la propia
    imagen (pixel (0, 0)), no a la posición del personaje: esto
    funciona porque la captura ya viene recortada justo al borde del
    tablero. Devuelve (grid, origin_x, origin_y) para mantener la
    misma interfaz que el resto del código usa al convertir entre
    celda y píxel; hoy origin_x y origin_y siempre son 0, pero quedan
    explícitos por si en el futuro hace falta compensar un desfase
    real entre la captura y el tablero (ver nota más abajo).
    """
    height, width = mask.shape[:2]
    origin_x, origin_y = 0, 0

    rows = int((height - origin_y) // tile)
    cols = int((width - origin_x) // tile)

    grid = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            x = origin_x + j * tile
            y = origin_y + i * tile

            if _path_fraction(mask, x, y, tile) > config.MIN_PATH_FRACTION:
                grid[i, j] = 1

    return grid, origin_x, origin_y


def cell_at(point: Tuple[int, int], origin_x: int, origin_y: int, tile: int) -> Tuple[int, int]:
    """Fila y columna de la celda que contiene un punto de la imagen."""
    return (int((point[1] - origin_y) // tile), int((point[0] - origin_x) // tile))


def neighbors(cell: Tuple[int, int]):
    """Las cuatro celdas vecinas: arriba, abajo, izquierda y derecha."""
    i, j = cell
    return ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))


def build_graph(grid: np.ndarray):
    """
    Arma el grafo de las celdas transitables. Cada nodo es una celda
    (i, j) de la grilla; las aristas solo unen vecinos de cuatro
    direcciones, así que no se pueden cortar esquinas por el aire.
    """
    rows, cols = grid.shape

    nodes: Set[Tuple[int, int]] = {
        (i, j) for i in range(rows) for j in range(cols) if grid[i, j]
    }

    edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for node in nodes:
        for neighbor in neighbors(node):
            if neighbor in nodes:
                edges.add(tuple(sorted((node, neighbor))))

    return nodes, edges


def reachable_component(nodes: Set[Tuple[int, int]], origin: Tuple[int, int]):
    """
    Celdas alcanzables desde `origin` moviéndose por vecinos de cuatro
    direcciones. Descarta el piso suelto que el jugador nunca pisaría.
    """
    if origin not in nodes:
        return set()

    reached = {origin}
    queue = deque([origin])

    while queue:
        for neighbor in neighbors(queue.popleft()):
            if neighbor in nodes and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)

    return reached


def _load_matchers(frame: np.ndarray) -> List[TemplateMatcher]:
    return [
        TemplateMatcher(
            config.TEMPLATES_DIR / f"{name}.png",
            frame,
            flip=(name == config.PLAYER_NAME),
        )
        for name in config.TEMPLATE_NAMES
    ]


def recognize(frame: np.ndarray) -> GameState:
    """
    Punto de entrada del módulo: corre todo el pipeline de
    reconocimiento sobre un frame y arma el GameState que consume el
    resto del agente.
    """
    matchers = _load_matchers(frame)

    candidates: List[Tuple[str, int, int, int, int, float]] = []
    for matcher in matchers:
        height, width = matcher.template_size
        for x, y, score in matcher.match(threshold=config.THRESHOLD, max_overlap=config.MAX_OVERLAP):
            candidates.append((matcher.name, x, y, width, height, score))

    detections = _resolve_overlaps(candidates, config.MAX_OVERLAP)

    # El contador de diamantes del HUD, arriba del tablero, matchea el
    # template de "diamond" igual que uno real. No es parte del
    # tablero, así que se descarta cualquier detección por encima de
    # esa franja.
    detections = [obj for obj in detections if obj.center[1] >= config.HUD_HEIGHT]

    player_matches = [obj for obj in detections if obj.name == config.PLAYER_NAME]
    if not player_matches:
        raise RuntimeError("No se encontró el personaje en el frame")

    # _resolve_overlaps ordena por score, así que el primero es la
    # mejor coincidencia del personaje.
    player_center = player_matches[0].center

    path_mask = detect_paths(frame)
    tile_size, _ = detect_tile_size(path_mask, player_center)
    grid, origin_x, origin_y = create_grid(path_mask, tile_size)

    player_cell = cell_at(player_center, origin_x, origin_y, tile_size)

    rows, cols = grid.shape
    if 0 <= player_cell[0] < rows and 0 <= player_cell[1] < cols:
        # El sprite tapa su propia celda, así que nunca pasa el filtro
        # de suelo por sí sola.
        grid[player_cell] = 1

    nodes, edges = build_graph(grid)
    reachable = reachable_component(nodes, player_cell)
    edges = {edge for edge in edges if edge[0] in reachable and edge[1] in reachable}

    return GameState(
        grid=grid,
        graph_nodes=reachable,
        graph_edges=edges,
        player_cell=player_cell,
        objects=detections,
        origin_x=origin_x,
        origin_y=origin_y,
        tile_size=tile_size,
    )