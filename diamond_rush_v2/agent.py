"""
La clase del agente: cierra el lazo percepcion -> decision -> control
-> verificacion de progreso.

  - compute(): PERCEPCION + DECISION. Ubica el canvas, captura UN
    frame, lo reconoce en un GameState, lo traduce a un World y define
    el camino solucion completo (A* + retroceso) SIN necesitar repetir
    capturas.
  - execute(plan): CONTROL + VERIFICACION. Manda las flechas del plan
    al canvas por Selenium y, si se pide, re-percibe para comparar la
    celda del personaje contra la esperada.

Tipo de agente: deliberativo basado en modelos, sin aprendizaje.
"""
import time
from typing import List, Optional

import cv2

import config
from computation.compute import Plan, Planner, World, simulate, world_from_state
from perception.capture import BrowserCapturer
from perception.recognition import recognize
from perception.state import GameState
from play.control import Keyboard


class Agent:
    """
    Percepcion (un frame) -> decision (plan) -> control (flechas)
    -> verificacion de progreso.
    """

    def __init__(
        self,
        capturer: BrowserCapturer,
        keyboard: Optional[Keyboard] = None,
        planner: Optional[Planner] = None,
        verify_each_step: bool = config.VERIFY_EACH_STEP,
        print_state: bool = False,
    ):
        # `keyboard` es opcional y SOLO se usa al ejecutar. El plan se
        # calcula con la unica captura; no se vuelve a mirar la pantalla
        # para decidir salvo que se pida verificacion en vivo.
        self.capturer = capturer
        self.keyboard = keyboard
        self.planner = planner or Planner()
        self.verify_each_step = verify_each_step
        self.print_state = print_state

        self.state: Optional[GameState] = None
        self.world: Optional[World] = None
        self.plan: Optional[Plan] = None

    # ---------------------------------------------------------------
    # 1) Percepcion
    # ---------------------------------------------------------------
    def setup(self) -> None:
        self.capturer.locate_canvas()

    def perceive(self) -> GameState:
        """Captura un frame y lo reconoce; guarda y devuelve el GameState."""
        frame = self.capturer.capture()
        try:
            state = recognize(frame)
        except RuntimeError as error:
            self._report_miss(frame, error)
            raise

        self.state = state
        if self.print_state:
            self._print_state(state)
        return state

    # ---------------------------------------------------------------
    # 2) Decision
    # ---------------------------------------------------------------
    def compute(self) -> Plan:
        """
        Percibe (si hace falta), construye el World y devuelve el camino
        solucion completo. Con la unica percepcion se resuelve todo el
        problema.
        """
        if self.state is None:
            self.perceive()
        self.world = world_from_state(self.state)
        self.plan = self.planner.plan(self.world)
        return self.plan

    # ---------------------------------------------------------------
    # 3) Control + 4) verificacion de progreso
    # ---------------------------------------------------------------
    def execute(
        self,
        plan: Optional[Plan] = None,
        pause: Optional[float] = None,
        verify: bool = True,
    ) -> Plan:
        """
        Envia por teclado las acciones del plan (control) y verifica
        progreso re-percibiendo. Si no hay teclado, solo simula.
        """
        plan = plan or self.plan
        if plan is None:
            plan = self.compute()

        self._verify_simulation(plan)

        if self.keyboard is None:
            print("No hay teclado configurado: no se ejecuta (solo plan).")
            return plan

        expected_cells = self._expected_cells(plan)
        self.keyboard.focus()

        for i, action in enumerate(plan.actions, 1):
            print(f"[{i:>4}/{plan.length}] {action.text()}")
            self.keyboard.move(action.direction, pause=pause)
            if verify and self.verify_each_step:
                self._verify_live(expected_cells[i - 1], plan)

        if verify:
            self._verify_final(plan)

        return plan

    def run(self, execute: bool = True) -> Plan:
        """Lazo completo: setup -> compute -> execute."""
        self.setup()
        plan = self.compute()
        self._print_plan(plan)
        if execute:
            self.execute(plan)
        return plan

    # ---------------------------------------------------------------
    # Apoyo
    # ---------------------------------------------------------------
    def _expected_cells(self, plan: Plan) -> List[tuple]:
        """Celda esperada del personaje tras cada accion del plan."""
        cells = []
        current = plan.initial_world
        if current is None:
            return cells
        for action in plan.actions:
            current = simulate(current, [action])
            cells.append(current.player)
        return cells

    def _verify_simulation(self, plan: Plan) -> None:
        """Rejuego del plan sobre el modelo: si no reproduce el final, hay bug."""
        if plan.initial_world is None:
            return
        final = simulate(plan.initial_world, plan.actions)
        if plan.complete and plan.final_world is not None:
            match = (
                final.player == plan.final_world.player
                and final.diamonds == plan.final_world.diamonds
            )
            if not match:
                print("ADVERTENCIA: la simulacion del plan difiere de lo esperado.")

    def _verify_live(self, expected_cell, plan: Plan) -> None:
        """Re-percibe (1 frame) y compara la celda real con la esperada."""
        try:
            state = self.perceive()
        except RuntimeError:
            return  # pantalla de transicion/nivel terminado: no es fallo
        if state.player_cell != expected_cell:
            print(
                f"PROGRESO INESPERADO: personaje en {state.player_cell}, "
                f"se esperaba {expected_cell}."
            )
            if plan.complete:
                print("Se re-planifica sobre el estado percibido.")
                self.world = world_from_state(state)
                self.plan = self.planner.plan(self.world)

    def _verify_final(self, plan: Plan) -> None:
        print("-" * 60)
        if not plan.complete:
            print(f"Plan incompleto; metas pendientes: {plan.pending}")
            return

        goal_was_exit = (
            plan.final_world is not None
            and plan.final_world.exit == plan.final_world.player
        )

        if self.keyboard is None:
            return
        try:
            state = self.perceive()
        except RuntimeError:
            # No se encuentra al personaje tras ejecutar: si la meta era
            # la salida, es la pantalla de nivel completado -> exito.
            if goal_was_exit:
                print("Nivel completado (pantalla de victoria).")
            else:
                print("No se pudo re-percibir para confirmar el nivel.")
            return

        if (
            self.world is not None
            and self.world.exit is not None
            and state.player_cell == self.world.exit
        ):
            print("Verificacion: personaje en la salida.")
        else:
            print(f"Verificacion: personaje quedo en {state.player_cell}.")

    def _print_plan(self, plan: Plan) -> None:
        print("=" * 62)
        print("PLAN RESUELTO")
        print("=" * 62)
        if self.state is not None:
            print(f"personaje inicial: {self.state.player_cell}")
        print(f"acciones: {plan.length} pasos  |  completo: {plan.complete}")
        if plan.pending:
            print(f"metas pendientes (inalcanzables con el modelo): {plan.pending}")
        if plan.final_world is not None:
            print(f"personaje final (predicho): {plan.final_world.player}")
            print(f"diamantes restantes (predichos): {len(plan.final_world.diamonds)}")
        if plan.actions:
            print("\npasos:")
            for i, action in enumerate(plan.actions, 1):
                print(f"  {i:>4}. {action.text()}")
        print("=" * 62)

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
