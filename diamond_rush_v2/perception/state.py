"""
Estado que produce la percepción y consume el resto del agente.

Este es el "contrato" entre percepcion/ y computation/ + play/: esos
dos módulos solo deberían importar de aquí, nunca de cv2 ni de los
detalles internos de recognition.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

Cell = Tuple[int, int]


@dataclass
class DetectedObject:
    """Una detección de un tipo de objeto sobre el frame."""

    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class GameState:
    """
    Estado percibido de un frame del juego.

    El grafo es `graph_nodes` + `adjacency` -- nada más de acá abajo
    es "otra versión del grafo", son datos de apoyo:

    - grid: la retícula completa (1 = transitable, 0 = obstáculo),
        sin filtrar. El grafo se construye a partir de esto, pero una
        vez construido no hace falta volver a mirar `grid`.
    - graph_nodes: las celdas transitables Y alcanzables desde el
        personaje (ya sin piso suelto inaccesible).
    - adjacency: lista de adyacencia del grafo -- para cada celda de
        graph_nodes, sus vecinos transitables (hasta 4, arriba/abajo/
        izquierda/derecha). Es lo que se recorre para pathfinding.
    - player_cell: celda (fila, columna) donde está el personaje.
    - objects: todas las detecciones (diamantes, rocas, llaves...),
        con posición en píxeles.
    - objects_by_cell: lo mismo que `objects`, pero agrupado por
        celda -- para saber qué hay parado en un nodo del grafo sin
        tener que convertir píxeles a mano.
    - origin_x / origin_y / tile_size: la conversión píxel<->celda,
        por si hace falta ubicar algo que no esté ya en
        objects_by_cell.
    """

    grid: np.ndarray
    graph_nodes: Set[Cell]
    adjacency: Dict[Cell, List[Cell]]
    player_cell: Cell
    objects: List[DetectedObject]
    objects_by_cell: Dict[Cell, List[DetectedObject]]
    origin_x: int
    origin_y: int
    tile_size: int

    def objects_named(self, name: str) -> List[DetectedObject]:
        """Atajo para filtrar detecciones por tipo, ej. "diamond"."""
        return [obj for obj in self.objects if obj.name == name]

    def cell_of(self, point: Tuple[int, int]) -> Cell:
        """Convierte un punto en píxeles (ej. obj.center) a su celda (fila, columna)."""
        x, y = point
        return (
            (y - self.origin_y) // self.tile_size,
            (x - self.origin_x) // self.tile_size,
        )
