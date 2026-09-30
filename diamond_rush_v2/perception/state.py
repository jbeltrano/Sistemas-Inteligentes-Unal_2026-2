"""
Estado que produce la percepción y consume el resto del agente.

Este es el "contrato" entre percepcion/ y computation/ + play/: esos
dos módulos solo deberían importar de aquí, nunca de cv2 ni de los
detalles internos de recognition.py.
"""

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np


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

    grid: matriz de la retícula (1 = transitable, 0 = obstáculo).
    graph_nodes / graph_edges: subgrafo ya filtrado a lo alcanzable
        desde el personaje (ver reachable_component en recognition.py).
    player_cell: celda (fila, columna) donde está el personaje.
    objects: todas las detecciones (diamantes, rocas, llaves, etc.).
    origin_x / origin_y: esquina superior izquierda de grid[0, 0],
        en píxeles del frame.
    tile_size: lado de cada celda, en píxeles.
    """

    grid: np.ndarray
    graph_nodes: Set[Tuple[int, int]]
    graph_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]
    player_cell: Tuple[int, int]
    objects: List[DetectedObject]
    origin_x: int
    origin_y: int
    tile_size: int

    def objects_named(self, name: str) -> List[DetectedObject]:
        """Atajo para filtrar detecciones por tipo, ej. "diamond"."""
        return [obj for obj in self.objects if obj.name == name]
