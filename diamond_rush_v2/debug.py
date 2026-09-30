"""
Herramientas de depuración visual del pipeline de percepción.

Nada de este archivo se llama desde agent.py: es solo para revisar a
mano, sobre una imagen puntual, si el reconocimiento dio el resultado
esperado.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

import config
from perception.recognition import detect_paths, recognize
from perception.state import GameState


def _color_for(name: str) -> tuple:
    """
    Color determinístico por nombre de objeto (a partir de un hash),
    para que el mismo tipo de objeto se vea siempre igual entre
    corridas, a diferencia del color aleatorio del script original.
    """
    digest = hashlib.md5(name.encode()).digest()
    return (int(digest[0]), int(digest[1]), int(digest[2]))


def save_path_mask(frame: np.ndarray, output_path: Path) -> None:
    """Guarda la máscara de suelo/camino (blanco = transitable)."""
    mask = detect_paths(frame)
    cv2.imwrite(str(output_path), mask)


def draw_detections(frame: np.ndarray, state: GameState, output_path: Path) -> None:
    """Dibuja sobre una copia del frame el recuadro de cada objeto detectado."""
    image = frame.copy()

    for obj in state.objects:
        color = _color_for(obj.name)
        cv2.rectangle(
            image, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), color, 2
        )
        cv2.putText(
            image,
            f"{obj.name} ({obj.x}, {obj.y})",
            (obj.x, obj.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)


def draw_graph(frame: np.ndarray, state: GameState, output_path: Path) -> None:
    """
    Dibuja el grafo sobre una copia atenuada del frame: celdas
    alcanzables en verde, el resto en rojo, aristas entre los centros
    de celdas conectadas, la celda del personaje resaltada y los
    objetos detectados encima.
    """
    background = cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0.65, 0)
    rows, cols = state.grid.shape
    tile = state.tile_size

    def cell_center(cell):
        i, j = cell
        return (
            state.origin_x + j * tile + tile // 2,
            state.origin_y + i * tile + tile // 2,
        )

    for i in range(rows):
        for j in range(cols):
            x = state.origin_x + j * tile
            y = state.origin_y + i * tile
            color = (0, 200, 0) if (i, j) in state.graph_nodes else (0, 0, 200)
            cv2.rectangle(background, (x, y), (x + tile, y + tile), color, 1)

    for a, b in state.graph_edges:
        cv2.line(
            background, cell_center(a), cell_center(b), (0, 255, 255), 1, cv2.LINE_AA
        )

    for cell in state.graph_nodes:
        x = state.origin_x + cell[1] * tile
        y = state.origin_y + cell[0] * tile
        cv2.putText(
            background,
            f"{cell}",
            (x + 4, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for obj in state.objects:
        color = _color_for(obj.name)
        cv2.rectangle(
            background,
            (obj.x, obj.y),
            (obj.x + obj.width, obj.y + obj.height),
            color,
            2,
        )
        cv2.putText(
            background,
            f"{obj.name} ({obj.x}, {obj.y})",
            (obj.x, obj.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    # El recuadro del personaje va al final para que no quede tapado
    # por su propia celda del grid.
    player_x = state.origin_x + state.player_cell[1] * tile
    player_y = state.origin_y + state.player_cell[0] * tile
    cv2.rectangle(
        background,
        (player_x, player_y),
        (player_x + tile, player_y + tile),
        (0, 255, 255),
        2,
    )
    cv2.circle(background, cell_center(state.player_cell), tile // 3, (0, 255, 255), 2)
    cv2.putText(
        background,
        "PLAYER",
        (player_x + 4, player_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), background)


def _write_summary(state: GameState, write) -> None:
    """
    `write` es una función que recibe una línea de texto (print, o
    file.write con salto de línea agregado) — así este resumen sirve
    tanto para consola como para el reporte por nivel.
    """
    write(f"Personaje en la celda {state.player_cell} de {state.grid.shape}")
    write(f"Casillas transitables: {int(state.grid.sum())}")
    write(f"Nodos del grafo: {len(state.graph_nodes)}")
    write(f"Aristas del grafo: {len(state.graph_edges)}")
    write("Detecciones por tipo:")
    for name in config.TEMPLATE_NAMES:
        count = len(state.objects_named(name))
        write(f"  {name}: {count}")


def test_recognition(level_path: Path) -> None:
    frame = cv2.imread(str(level_path))
    if frame is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {level_path}")

    state = recognize(frame)

    _write_summary(state, print)

    draw_graph(frame, state, config.BASE_DIR / "debug_graph.png")
    draw_detections(frame, state, config.BASE_DIR / "debug_result.png")
    save_path_mask(frame, config.BASE_DIR / "debug_paths.png")
    print("Imágenes de depuración guardadas en la raíz del proyecto")


def test_recognition_all_levels() -> None:
    """
    Corre el reconocimiento sobre los 20 niveles. Si alguno falla (por
    ejemplo, si no se detecta al personaje), lo anota y sigue con el
    resto — una imagen problemática no debería tapar el resultado de
    las otras diecinueve.
    """
    failures = []

    for i in range(1, 21):
        level_path = config.LEVELS_DIR / f"level{i}.png"

        frame = cv2.imread(str(level_path))
        if frame is None:
            failures.append((i, f"no se pudo cargar {level_path}"))
            continue

        try:
            state = recognize(frame)
        except (
            Exception
        ) as error:  # queremos capturar cualquier falla y seguir con el resto
            failures.append((i, str(error)))
            continue

        level_debug_dir = config.DEBUG_DIR / f"level{i}"
        level_debug_dir.mkdir(parents=True, exist_ok=True)

        with open(
            level_debug_dir / f"debug_level{i}.txt", "w", encoding="utf-8"
        ) as file:
            _write_summary(state, lambda line: file.write(line + "\n"))

        draw_graph(frame, state, level_debug_dir / f"graph_level{i}.png")
        draw_detections(frame, state, level_debug_dir / f"result_level{i}.png")
        save_path_mask(frame, level_debug_dir / f"paths_level{i}.png")

    print("Imágenes y reportes de depuración guardados correctamente.")
    if failures:
        print(f"Niveles con error ({len(failures)}):")
        for level, reason in failures:
            print(f"  level{level}: {reason}")
