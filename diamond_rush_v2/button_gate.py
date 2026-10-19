"""
Mapeo explicito boton -> reja por nivel.

La percepcion no dice que reja abre cada boton; con varios pares el
agente no puede adivinarlo con fiabilidad (la heuristica por cercania
falla cuando los pares estan entrecruzados). Aqui lo declaras a mano,
por nivel: el numero que se pasa con

    python diamond_rush_v2/main.py --level N

Formato:

    BUTTON_GATE[nivel] = {(fila_boton, col_boton): (fila_reja, col_reja), ...}

Reglas:
  - Celdas en orden (fila, columna); tablero 15x10 (config.BOARD_ROWS/COLS).
  - Emparejamiento 1:1: cada boton abre UNA reja.
  - Los niveles con un solo par no necesitan entrada (se resuelven solos).
  - Si una celda no coincide con lo percibido, esa entrada se ignora y se
    cae a la heuristica por cercania (el lazo cerrado corrige tras re-percibir).
  - El efecto de la reja es TEMPORAL: se abre solo mientras su boton este
    cubierto (personaje o roca encima).

Ejemplo (nivel 6: el boton de la fila 3, columna 2 abre la reja de la fila
3, columna 7):

    BUTTON_GATE[6] = {(3, 2): (3, 7)}
"""
from typing import Dict, Tuple

Cell = Tuple[int, int]

BUTTON_GATE: Dict[int, Dict[Cell, Cell]] = {
    # ------------------------------------------------------------------
    # Pistas detectadas con recognize() sobre assets/levels/level*.png
    # (celdas (fila, columna), solo orientativas: revisar contra el nivel
    # real). Los niveles con un solo par resuelven solos y no necesitan
    # entrada.
    #
    #
    #   Un solo par (opcional): 11: (3,7)->(4,6); 12: (12,5)->(8,2);
    #     13: (7,5)->(7,4); 15: (6,7)->(12,3)
    #
    # Ejemplo de como completar un nivel multi-par:
    8: {
        (7, 2): (8, 7),
        (12, 6): (9, 6),
    },
    9: {
            (5, 1): (5, 4),
            (7, 8): (8, 2),
            (12, 5): (11, 7),
        },
    14: {
        (7, 8): (4, 2),
        (8, 4): (5, 7),
        (11, 1): (7, 3),
        (11, 4): (11, 6),
    },
    17: {
        (5, 4): (5, 2),
        (5, 7): (4, 8),
        (9, 8): (5, 5),
    },
    18: {
        (6, 3) : (12, 7),
        (10, 6) : (4, 8),
        (12, 3): (7, 4),
    },
    19: {
        (5, 2) : (5, 1) ,
        (10, 8) : (5, 5),
    },
    13: {(7, 5): (7, 4)},
    # ------------------------------------------------------------------
}


# ----------------------------------------------------------------------
# Fallo del juego: colarse por la reja contigua al boton.
#
# En estos pares el boton esta pegado a su reja: pisando el boton la reja
# se abre, y si desde ahi se manda al personaje hacia la reja, entra antes
# de que se cierre (no hace falta dejar una roca encima del boton).
#
# Formato (igual que BUTTON_GATE, pero SOLO los pares donde el fallo se
# haya comprobado en el juego):
#
#     GATE_GLITCH[nivel] = {(fila_boton, col_boton): (fila_reja, col_reja)}
#
# El planificador solo lo usa en los niveles listados aqui y, si en
# ejecucion el personaje no termina en la reja, lo desactiva para el resto
# de la corrida y re-planifica sin el.
#
# `computation.gates.glitch_candidates()` lista los pares contiguos de
# BUTTON_GATE como pista para buscar el "otro nivel" con el mismo fallo.
# ----------------------------------------------------------------------
GATE_GLITCH: Dict[int, Dict[Cell, Cell]] = {
    19: {(5, 2): (5, 1)},
    13: {(7, 5): (7, 4)},
}