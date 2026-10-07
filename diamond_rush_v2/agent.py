"""
La clase del agente.

Por ahora el agente hace una sola cosa: ubicar el canvas, tomar UNA
captura, reconocerla, y reportar el GameState completo -- es el paso
de verificación antes de construir el plan de movimientos sobre ese
grafo. El loop continuo (percibir repetidamente) y la planificación en
sí todavía no existen a propósito.
"""

import time
from pathlib import Path

import cv2

import config
from perception.capture import BrowserCapturer
from perception.recognition import recognize
from perception.state import GameState


class Agent:
    """Ubica el canvas, percibe un frame una sola vez, y reporta el GameState resultante."""

    def __init__(self, capturer: BrowserCapturer):
        self.capturer = capturer

    def setup(self) -> None:
        self.capturer.locate_canvas()

    def run(self) -> GameState:
        """
        Captura y reconoce un único frame. Devuelve el GameState (por
        si algo más adelante quiere encadenar la planificación sobre
        el resultado), y de paso lo imprime completo para poder
        verificarlo a ojo.
        """
        self.setup()
        frame = self.capturer.capture()

        try:
            state = recognize(frame)
        except RuntimeError as error:
            self._report_miss(frame, error)
            raise

        self._print_state(state)
        return state

    @staticmethod
    def _print_state(state: GameState) -> None:
        """Vuelca cada campo del GameState, para confirmar que el reconocimiento anduvo bien."""
        print("=" * 60)
        print("GameState")
        print("=" * 60)

        print(f"\ngrid: {state.grid.shape[0]} filas x {state.grid.shape[1]} columnas")
        print(state.grid)

        print(f"\ngraph_nodes ({len(state.graph_nodes)} celdas alcanzables):")
        print(sorted(state.graph_nodes))

        edge_count = sum(len(neighbors) for neighbors in state.adjacency.values()) // 2
        print(f"\nadjacency ({len(state.adjacency)} nodos, {edge_count} aristas):")
        for cell, neighbors in sorted(state.adjacency.items()):
            print(f"  {cell}: {neighbors}")

        print(f"\nplayer_cell: {state.player_cell}")

        print(f"\nobjects ({len(state.objects)} detecciones):")
        for obj in state.objects:
            print(
                f"  {obj.name:<14} pixel=({obj.x},{obj.y}) "
                f"tamaño={obj.width}x{obj.height} centro={obj.center}"
            )

        print(f"\nobjects_by_cell ({len(state.objects_by_cell)} celdas con contenido):")
        for cell, objs in sorted(state.objects_by_cell.items()):
            names = [obj.name for obj in objs]
            print(f"  {cell}: {names}")

        print(
            f"\norigin_x={state.origin_x}  origin_y={state.origin_y}  tile_size={state.tile_size}"
        )
        print("=" * 60)

    @staticmethod
    def _report_miss(frame, error: RuntimeError) -> None:
        """Guarda el frame que no se pudo reconocer, para poder revisarlo después."""
        misses_dir = config.BASE_DIR / "debug_misses"
        misses_dir.mkdir(exist_ok=True)

        path = misses_dir / f"miss_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), frame)
        print(f"{error} -- frame guardado en {path}")
