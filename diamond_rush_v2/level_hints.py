"""
Guia de subobjetivos por nivel (pistas para el planificador).

El planificador normalmente resuelve el nivel solo (busqueda global o
composicion automatica de subobjetivos). Pero en algunos niveles eso es
lento (el 6 gasta ~80 s) o no converge dentro del presupuesto (el 10 no
logra rellenar su segundo hoyo). Aqui declaras, por nivel, una
SECUENCIA ORDENADA de subobjetivos "gruesos": el planificador los
resuelve PRIMERO, uno por A*, y luego completa el nivel con la
composicion automatica.

Se activa con el mismo `--level N` que usa button_gate.py:

    python diamond_rush_v2/main.py --level N

Formato:

    LEVEL_SEQUENCE[nivel] = [descriptor, descriptor, ...]

Celdas en orden (fila, columna); tablero 15x10 (config.BOARD_ROWS/COLS).

Descriptores:

  ("llave",    (f, c))          recoger esa llave
  ("puerta",   (f, c))          abrir esa puerta (consume la llave)
  ("diamante", (f, c))          recoger ese diamante
  ("hoyo",     (f, c))          rellenar ese hoyo con una roca
  ("roca", (fo, co), (fd, cd))  empujar una roca hasta (fd, cd).
                                El DESTINO decide el exito: hoyo ->
                                relleno, boton -> roca encima, otra
                                celda -> roca ahi. El origen es SOLO
                                informativo (para el trazo): el A* guia
                                con `rock_goal` (la celda destino) y usa
                                la roca mas cercana que pueda empujar.

Reglas:

  - Son PREFERENCIAS, no un guion estricto. En cada planificacion se
    revalidan contra el estado percibido: una entrada cuyas celdas ya no
    existen (objeto recogido/usado) se ignora, y una meta ya cumplida se
    salta. Si un subobjetivo falla, el planificador sigue con la
    composicion automatica.
  - Por eso el tester puede dejar el personaje en cualquier parte del
    mapa (nivel a medias): la secuencia se recalcula desde ese estado.
  - Los waypoints de roca se pueden ENCADENAR para partir un Sokoban:

        ("roca", (4, 7), (11, 7)),
        ("roca", (11, 7), (11, 1)),

    (ojo: una roca que rellena un hoyo se consume, asi que los waypoints
    intermedios deben ser piso transitable, no hoyos).
  - Los niveles sin entrada no cambian: se resuelven solos.

Ejemplo (nivel 6):

    LEVEL_SEQUENCE[6] = [
        ("llave", (4, 8)),
        ("puerta", (10, 2)),
        ("roca", (4, 7), (11, 1)),
    ]
"""
from typing import Dict, List, Tuple

Cell = Tuple[int, int]

LEVEL_SEQUENCE: Dict[int, List[tuple]] = {
    # ------------------------------------------------------------------
    # Nivel 6: llave (4,8) -> puerta (10,2) -> rellenar el hoyo (11,1)
    # empujando la roca (4,7). Antes, el A* global se quedaba sin
    # presupuesto y el relleno sin guia costaba ~80 s.
    6: [
        ("llave", (4, 8)),
        ("puerta", (10, 2)),
        ("roca", (4, 7), (11, 1)),
        ("llave", (13, 4)),
        ("puerta", (8, 5)),
    ],
    # ------------------------------------------------------------------
    # Nivel 10: PENDIENTE de calibrar. El solver actual no rellena (8,7)
    # (detras de los picos (9,7)/(8,5)) y deja el diamante (8,8)
    # inalcanzable. Completar tras revisar el nivel real con `--debug`
    # o `recognize` sobre assets/levels/level10.png. Plantilla:
    10: [
        ("roca", (12, 3), (8, 2)),
        ("roca", (12, 4), (8, 7)),
        
    ],
    11: [
        ("roca", (12, 3), (8, 2)),
        ("roca", (12, 4), (8, 7)),
        
    ],
    # ------------------------------------------------------------------
}
