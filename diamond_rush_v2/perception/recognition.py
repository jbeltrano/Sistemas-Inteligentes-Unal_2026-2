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

    def __init__(self, template_path: Path, frame: np.ndarray, name: str = None, flip: bool = False):
        """
        `name` es el nombre lógico del objeto (ej. "player"), separado
        del nombre del archivo. Por defecto es el nombre del archivo
        (como antes), pero se puede pasar explícito para que varios
        templates distintos (ej. "player.png" y "player_key.png")
        cuenten como el mismo objeto a la hora de reportar detecciones
        y de resolver superposiciones entre templates.
        """
        self.frame = frame
        self.template_path = template_path
        self.flip = flip
        self.name = name or Path(template_path).stem

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


def board_geometry(frame: np.ndarray) -> Tuple[int, int, int]:
    """
    Geometría del tablero a partir del tamaño del frame: (tile,
    origin_x, origin_y).

    El tablero es de BOARD_ROWS x BOARD_COLS celdas cuadradas y ocupa
    el canvas; el HUD solo se dibuja encima de las filas superiores.
    Por eso el tamaño de celda sale de dividir el frame entre las
    dimensiones conocidas del tablero, y la grilla se centra en el
    sobrante (si lo hay).
    """
    height, width = frame.shape[:2]
    tile = min(width // config.BOARD_COLS, height // config.BOARD_ROWS)

    if not config.MIN_TILE <= tile <= config.MAX_TILE:
        raise RuntimeError(
            f"Tamaño de celda inesperado ({tile}px) para un frame de "
            f"{width}x{height} y un tablero de "
            f"{config.BOARD_ROWS}x{config.BOARD_COLS}"
        )

    origin_x = (width - tile * config.BOARD_COLS) // 2
    origin_y = (height - tile * config.BOARD_ROWS) // 2
    return tile, origin_x, origin_y


def _path_fraction(mask: np.ndarray, x: int, y: int, tile: int) -> float:
    """
    Proporción de suelo en la zona central de una celda (se ignora un
    margen CELL_MARGIN por lado). Fuera de la imagen devuelve 0.0.
    """
    height, width = mask.shape[:2]

    if x < 0 or y < 0 or x + tile > width or y + tile > height:
        return 0.0

    margin = int(tile * config.CELL_MARGIN)
    inner = mask[y + margin : y + tile - margin, x + margin : x + tile - margin]
    if inner.size == 0:
        return 0.0

    return float(np.mean(inner > 0))


def create_grid(mask: np.ndarray, tile: int, origin_x: int, origin_y: int) -> np.ndarray:
    """
    Grilla de BOARD_ROWS x BOARD_COLS (1 = suelo, 0 = obstáculo).

    Una celda es suelo si su zona central supera MIN_PATH_FRACTION en
    la máscara de caminos. Las celdas cuyo centro queda bajo el HUD
    nunca son suelo: lo que se ve ahí es interfaz, no tablero.
    """
    grid = np.zeros((config.BOARD_ROWS, config.BOARD_COLS), dtype=np.uint8)

    for i in range(config.BOARD_ROWS):
        for j in range(config.BOARD_COLS):
            x = origin_x + j * tile
            y = origin_y + i * tile

            if y + tile // 2 < config.HUD_HEIGHT:
                continue

            if _path_fraction(mask, x, y, tile) > config.MIN_PATH_FRACTION:
                grid[i, j] = 1

    return grid


def cell_at(point: Tuple[int, int], origin_x: int, origin_y: int, tile: int) -> Tuple[int, int]:
    """Fila y columna de la celda que contiene un punto de la imagen."""
    return (int((point[1] - origin_y) // tile), int((point[0] - origin_x) // tile))


def neighbors(cell: Tuple[int, int]):
    """Las cuatro celdas vecinas: arriba, abajo, izquierda y derecha."""
    i, j = cell
    return ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))


def build_graph(grid: np.ndarray) -> Set[Tuple[int, int]]:
    """
    Nodos del grafo: cada celda transitable de la grilla. La lista de
    adyacencia (quién es vecino de quién) se arma más adelante, en
    recognize(), ya filtrada a lo alcanzable -- no tiene sentido
    calcularla acá para después descartar la mitad.
    """
    rows, cols = grid.shape
    return {(i, j) for i in range(rows) for j in range(cols) if grid[i, j]}


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
    matchers = []
    for name in config.TEMPLATE_NAMES:
        # La mayoría de los objetos tienen un solo archivo de template
        # (el propio nombre). Algunos, como "player", tienen más de
        # uno porque el objeto cambia de aspecto según el estado del
        # juego (con una llave en la mano, por ejemplo) -- ver
        # config.TEMPLATE_VARIANTS.
        for variant in config.TEMPLATE_VARIANTS.get(name, [name]):
            matchers.append(
                TemplateMatcher(
                    config.TEMPLATES_DIR / f"{variant}.png",
                    frame,
                    name=name,
                    flip=(name == config.PLAYER_NAME),
                )
            )
    return matchers


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
        # El personaje cambia de pose mientras se desliza entre
        # celdas (la animación de caminar), así que su parecido con el
        # template estático baja más que el de objetos quietos. Se le
        # da un umbral más permisivo solo a él.
        threshold = config.PLAYER_THRESHOLD if matcher.name == config.PLAYER_NAME else config.THRESHOLD
        for x, y, score in matcher.match(threshold=threshold, max_overlap=config.MAX_OVERLAP):
            candidates.append((matcher.name, x, y, width, height, score))

    # El personaje es el único objeto obligatorio: no puede perder su
    # lugar frente a otro template que matchee mejor en el mismo sitio
    # (por ejemplo, el ítem que lleva en la mano puede parecerse al de
    # un objeto suelto). Por eso se resuelve aparte y primero, en su
    # propia competencia -- solo contra sus propias variantes
    # (player.png, player_key.png, ...) -- en vez de competir a la par
    # de todo lo demás.
    player_candidates = [c for c in candidates if c[0] == config.PLAYER_NAME]
    other_candidates = [c for c in candidates if c[0] != config.PLAYER_NAME]

    player_matches = _resolve_overlaps(player_candidates, config.MAX_OVERLAP)
    if not player_matches:
        raise RuntimeError("No se encontró el personaje en el frame")

    player_box = player_matches[0]
    player_center = player_box.center

    other_detections = _resolve_overlaps(other_candidates, config.MAX_OVERLAP)
    detections = [player_box] + other_detections

    # El contador de diamantes del HUD, arriba del tablero, matchea el
    # template de "diamond" igual que uno real. No es parte del
    # tablero, así que se descarta cualquier detección por encima de
    # esa franja.
    detections = [obj for obj in detections if obj.center[1] >= config.HUD_HEIGHT]

    tile_size, origin_x, origin_y = board_geometry(frame)
    path_mask = detect_paths(frame)
    grid = create_grid(path_mask, tile_size, origin_x, origin_y)
    rows, cols = grid.shape

    def in_board(cell):
        return 0 <= cell[0] < rows and 0 <= cell[1] < cols

    # Una detección fuera del tablero no puede ser un objeto del juego.
    detections = [
        obj for obj in detections
        if in_board(cell_at(obj.center, origin_x, origin_y, tile_size))
    ]

    # Todo objeto está parado sobre una celda del tablero, así que su
    # celda es parte del suelo aunque el sprite tape el color y la
    # máscara no la vea. Qué objetos bloquean el paso (roca, hoyo,
    # lava, puertas...) NO se decide acá: eso cambia durante el juego
    # y lo resuelve el planificador con objects_by_cell.
    for obj in detections:
        grid[cell_at(obj.center, origin_x, origin_y, tile_size)] = 1

    player_cell = cell_at(player_center, origin_x, origin_y, tile_size)
    if not in_board(player_cell):
        raise RuntimeError(f"El personaje quedó fuera del tablero: {player_cell}")
    grid[player_cell] = 1

    nodes = build_graph(grid)
    reachable = reachable_component(nodes, player_cell)

    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
        node: [n for n in neighbors(node) if n in reachable] for node in reachable
    }

    objects_by_cell: Dict[Tuple[int, int], List[DetectedObject]] = {}
    for obj in detections:
        cell = cell_at(obj.center, origin_x, origin_y, tile_size)
        objects_by_cell.setdefault(cell, []).append(obj)

    return GameState(
        grid=grid,
        graph_nodes=reachable,
        adjacency=adjacency,
        player_cell=player_cell,
        objects=detections,
        objects_by_cell=objects_by_cell,
        origin_x=origin_x,
        origin_y=origin_y,
        tile_size=tile_size,
    )