"""
Modelo de estado del juego: el "contrato" entre la percepción y la
decisión.

La percepción (reconocimiento.py) produce un EstadoJuego a partir de
un frame; la decisión (planificacion.py) lo consume. Por eso este
módulo no importa OpenCV: la conversión píxel <-> celda y el armado
del estado están separados del matching de templates para poder
probarlos sin cámara ni imágenes.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Celda = Tuple[int, int]


@dataclass
class ObjetoDetectado:
    """Una detección de un tipo de objeto sobre el frame (nombre lógico)."""

    nombre: str
    x: int
    y: int
    ancho: int
    alto: int

    @property
    def centro(self) -> Tuple[int, int]:
        """Punto central de la caja, en píxeles (x, y)."""
        return (self.x + self.ancho // 2, self.y + self.alto // 2)


@dataclass
class EstadoJuego:
    """
    Estado percibido de UNA captura del juego.

    - grid: retícula FILAS x COLUMNAS (1 = transitable, 0 = obstáculo).
        Sale de crear_grid() (máscara de caminos) con las celdas que
        tienen un objeto forzadas a transitable (un objeto siempre está
        parado sobre suelo).
    - player_cell: celda (fila, columna) del personaje.
    - objects: todas las detecciones con su caja en píxeles.
    - objects_by_cell: las mismas detecciones agrupadas por celda, para
        que la decisión sepa qué hay parado en cada casilla sin
        convertir píxeles a mano.
    - origin_x / origin_y / tile_size: la conversión píxel <-> celda.
    """

    grid: np.ndarray
    player_cell: Celda
    origin_x: int
    origin_y: int
    tile_size: int
    objects: List[ObjetoDetectado] = field(default_factory=list)
    objects_by_cell: Dict[Celda, List[ObjetoDetectado]] = field(default_factory=dict)

    def objetos_nombrados(self, nombre: str) -> List[ObjetoDetectado]:
        """Atajo: todas las detecciones de cierto tipo, ej. "diamante"."""
        return [obj for obj in self.objects if obj.nombre == nombre]

    def objetos_por_celda(self, celda: Celda) -> List[ObjetoDetectado]:
        """Detectados parados sobre una celda (lista vacía si no hay nada)."""
        return self.objects_by_cell.get(celda, [])

    def celda_de(self, punto: Tuple[int, int]) -> Celda:
        """Punto en píxeles (x, y) -> celda (fila, columna)."""
        x, y = punto
        return (
            int((y - self.origin_y) // self.tile_size),
            int((x - self.origin_x) // self.tile_size),
        )

    def en_tablero(self, celda: Celda) -> bool:
        filas, columnas = self.grid.shape
        return 0 <= celda[0] < filas and 0 <= celda[1] < columnas


def construir_estado(
    grid: np.ndarray,
    objetos: List[ObjetoDetectado],
    player_cell: Celda,
    origin_x: int,
    origin_y: int,
    tile_size: int,
) -> EstadoJuego:
    """
    Arma el EstadoJuego a partir de la grilla y las detecciones.

    Separado de reconocimiento.py para que sea puro (sin cv2): recibe
    una grilla ya calculada y la lista de detecciones, agrupa las
    detecciones por celda y fuerza a transitable toda celda que tenga
    un objeto parado encima (un objeto no puede estar sobre una pared).
    """
    filas, columnas = grid.shape

    objetos_por_celda: Dict[Celda, List[ObjetoDetectado]] = {}
    for obj in objetos:
        celda = (
            int((obj.centro[1] - origin_y) // tile_size),
            int((obj.centro[0] - origin_x) // tile_size),
        )
        if 0 <= celda[0] < filas and 0 <= celda[1] < columnas:
            objetos_por_celda.setdefault(celda, []).append(obj)
            grid[celda] = 1

    if not (0 <= player_cell[0] < filas and 0 <= player_cell[1] < columnas):
        raise RuntimeError(f"El personaje quedó fuera del tablero: {player_cell}")
    grid[player_cell] = 1

    return EstadoJuego(
        grid=grid,
        player_cell=player_cell,
        objects=objetos,
        objects_by_cell=objetos_por_celda,
        origin_x=origin_x,
        origin_y=origin_y,
        tile_size=tile_size,
    )