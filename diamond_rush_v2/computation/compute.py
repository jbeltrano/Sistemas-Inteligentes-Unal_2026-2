"""
Planificador deliberativo del agente (sin aprendizaje).

Tipo de agente: deliberativo basado en modelos (Russell & Norvig).
Percibe UNA vez (GameState), construye una representacion del mundo
(World) y resuelve el problema con busqueda en espacio de estados:

- A* (a_star): busca en el espacio de mundos completos. Un estado es la
  posicion del personaje + el conjunto de rocas (movibles) + los
  diamantes restantes. Cada accion mueve una casilla o empuja una roca
  (empuje estilo Sokoban / Boulder Dash).
- Retroceso (backtracking_search): si A* no halla camino dentro de su
  presupuesto de nodos (tipicamente porque una roca bloquea el unico
  corredor y hay que empujarla lateralmente), se explora el espacio de
  estados con DFS + backtracking: se deshacen empujes que no llevan a
  ningun lado y se priorizan las acciones que empujan rocas.

Heuristica ("pensar fuera de la caja"): la heuristica mira el nivel
COMPLETO y no la casilla vecina. Es un campo de distancias (BFS
multi-fuente desde las metas) calculado sobre la grilla "pelada" (sin
rocas). Al ignorar las rocas, el campo subestima el costo real (las
rocas solo pueden entorpecer), asi que la heuristica es ADMISIBLE y
CONSISTENTE -> A* devuelve la solucion optima dentro de cada subobjetivo;
y al ser distancias reales (no Manhattan) el agente ya "sabe" esquivar
los muros del laberinto.

La planificacion se compone en subobjetivos:
  1) recolectar todos los diamantes restantes (una busqueda por cada uno)
  2) si hay una llave sin recoger, recogerla antes de la salida
  3) alcanzar la salida

Este modulo importa SOLO de perception.state (el contrato) y config;
nunca de cv2 ni de recognition.py.
"""
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import config
from perception.state import GameState

Cell = Tuple[int, int]

DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PLAYER_ORDER = ("up", "down", "left", "right")


@dataclass(frozen=True)
class Action:
    """Un paso del plan: moverse o empujar una roca en una direccion."""

    kind: str          # "move" | "push"
    direction: str     # una clave de DELTA

    def text(self) -> str:
        verb = "empujar roca hacia" if self.kind == "push" else "mover hacia"
        return f"{verb} {self.direction}"


@dataclass(frozen=True)
class World:
    """
    Modelo estatico del nivel para la busqueda.

    Todo son conjuntos de celdas (fila, columna). `walls` son celdas
    intransitables (pared, hoyo, picos: los peligros del juego se
    modelan como obstaculos); `rocks` son celdas ocupadas por una roca
    empujable; `diamonds` son los diamantes por recolectar.
    """

    rows: int
    cols: int
    player: Cell
    walls: FrozenSet[Cell] = frozenset()
    rocks: FrozenSet[Cell] = frozenset()
    diamonds: FrozenSet[Cell] = frozenset()
    key: Optional[Cell] = None
    has_key: bool = False
    exit: Optional[Cell] = None

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols


@dataclass
class Plan:
    """Solucion completa: la secuencia de acciones mas el mundo final."""

    actions: List[Action] = field(default_factory=list)
    final_world: Optional[World] = None
    complete: bool = False
    pending: List[str] = field(default_factory=list)
    initial_world: Optional[World] = None

    @property
    def length(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------
# Semantica de los objetos detectados -> rol en el mundo
# ---------------------------------------------------------------
# Mapeo nombre logico del objeto de v2 (perception.state) -> categoria.
# `closed_gate` bloquea; `open_gate`/`scape_*`/`door_key` son la salida.
ROLES = {
    "diamond": "goal",
    "key": "key",
    "door_key": "exit",
    "scape_block": "exit",
    "scape_enable": "exit",
    "open_gate": "exit",
    "rock": "rock",
    "rock_in_hole": "fixed",
    "closed_gate": "fixed",
    "hole": "hazard",
    "spike": "hazard",
    "totem": "hazard",
    "no_spike": "floor",
    "button": "floor",
    "player": "player",
}
_ROLE_TO_WORLD = {
    "goal": ("diamonds",),
    "key": ("key",),
    "exit": ("exit",),
    "rock": ("rocks",),
    "fixed": ("walls",),
    "hazard": ("walls",),
}


def world_from_state(state: GameState, roles: Optional[Dict[str, str]] = None) -> World:
    """
    Traduce el GameState percibido (grid + detecciones) al World de
    busqueda. Los peligros del juego (hoyo, picos...) se tratan como
    muros por defecto: el agente no entra a casillas letales.
    """
    roles = roles or ROLES
    grid = state.grid.tolist()
    rows, cols = len(grid), (len(grid[0]) if grid else 0)

    walls = {(i, j) for i in range(rows) for j in range(cols) if not grid[i][j]}
    rocks: Set[Cell] = set()
    diamonds: Set[Cell] = set()
    key = None
    exit_cell = None

    # Cada deteccion agrega su rol al mundo (una celda puede tener
    # varios objetos: el primero manda sobre la transitabilidad).
    for obj in state.objects:
        role = roles.get(obj.name, "floor")
        if role in _ROLE_TO_WORLD:
            cell = state.cell_of(obj.center)
            targets = _ROLE_TO_WORLD[role]
            if "walls" in targets:
                walls.add(cell)
            if "rocks" in targets:
                rocks.add(cell)
            if "diamonds" in targets:
                diamonds.add(cell)
            if role == "key":
                key = cell
            if role == "exit":
                exit_cell = cell

    return World(
        rows=rows,
        cols=cols,
        walls=frozenset(walls),
        rocks=frozenset(rocks),
        player=state.player_cell,
        diamonds=frozenset(diamonds),
        key=key,
        has_key=False,
        exit=exit_cell,
    )


# ---------------------------------------------------------------
# Sucesores
# ---------------------------------------------------------------
def _step(world: World, direction: str) -> Optional[Tuple[Action, World]]:
    """
    Aplica un movimiento o un empuje desde el estado actual.
    Devuelve (action, world) o None si la celda destino esta bloqueada.
    """
    d = DELTA[direction]
    origin = world.player
    target = (origin[0] + d[0], origin[1] + d[1])

    if not world.in_bounds(target) or target in world.walls:
        return None

    if target in world.rocks:
        # Empujar la roca a la celda de mas alla.
        push = (target[0] + d[0], target[1] + d[1])
        if (
            not world.in_bounds(push)
            or push in world.walls
            or push in world.rocks
            # No empujar una roca sobre recolectables ni sobre la salida.
            or push in world.diamonds
            or push == world.key
            or push == world.exit
        ):
            return None
        rocks = frozenset((world.rocks - {target}) | {push})
        player = target
        kind = "push"
    else:
        rocks = world.rocks
        player = target
        kind = "move"

    new = World(
        rows=world.rows,
        cols=world.cols,
        walls=world.walls,
        rocks=rocks,
        player=player,
        diamonds=world.diamonds,
        key=world.key,
        has_key=world.has_key,
        exit=world.exit,
    )
    return Action(kind, direction), new


def _collect(world: World) -> World:
    """Ultima transicion: si el personaje piso un diamante o la llave."""
    diamonds = world.diamonds - {world.player}
    key = world.key
    has_key = world.has_key
    if world.player == world.key:
        key = None
        has_key = True
    if diamonds == world.diamonds and key == world.key:
        return world
    return World(
        rows=world.rows,
        cols=world.cols,
        walls=world.walls,
        rocks=world.rocks,
        player=world.player,
        diamonds=frozenset(diamonds),
        key=key,
        has_key=has_key,
        exit=world.exit,
    )


def _successors(world: World):
    """Todos los (action, new_world) validos desde `world`."""
    for direction in PLAYER_ORDER:
        result = _step(world, direction)
        if result is None:
            continue
        action, new = result
        yield action, _collect(new)


# ---------------------------------------------------------------
# Heuristica: campo de distancias multi-fuente
# ---------------------------------------------------------------
def wall_distance_field(world: World, goals: Iterable[Cell]) -> Dict[Cell, int]:
    """
    BFS multi-fuente desde las metas sobre la grilla "pelada" (sin
    rocas). Distancia real en casillas para cada celda alcanzable;
    no Manhattan: ya toma en cuenta los muros del laberinto.
    """
    dist: Dict[Cell, int] = {}
    queue = deque()
    for goal in goals:
        if world.in_bounds(goal) and goal not in world.walls:
            dist.setdefault(goal, 0)
            queue.append((goal, 0))

    while queue:
        cell, d = queue.popleft()
        for df, dc in DELTA.values():
            neighbor = (cell[0] + df, cell[1] + dc)
            if (
                world.in_bounds(neighbor)
                and neighbor not in world.walls
                and neighbor not in dist
            ):
                dist[neighbor] = d + 1
                queue.append((neighbor, d + 1))
    return dist


def heuristic(world: World, goals: FrozenSet[Cell]) -> float:
    """
    Distancia del personaje a la meta mas cercana segun el campo de
    distancias. Ignora las rocas a proposito: asumir que el pasaje
    siempre esta libre subestima el costo real (admisible) y cumple la
    desigualdad triangular (consistente), con lo que A* conserva su
    garantia de optimalidad dentro del subobjetivo.
    """
    field = wall_distance_field(world, goals)
    return field.get(world.player, float("inf"))


# ---------------------------------------------------------------
# A*
# ---------------------------------------------------------------
def a_star(
    world: World,
    goals: FrozenSet[Cell],
    max_nodes: int = config.MAX_NODES_ASTAR,
) -> Optional[Tuple[List[Action], World]]:
    """
    A* sobre el espacio de mundos completos. Devuelve (actions,
    final_world) o None si no encuentra solucion dentro del presupuesto
    de nodos. Mueve, empuja rocas y queda parado sobre la meta
    (recolectandola si es diamante/llave).
    """
    if world.player in goals:
        return [], _collect(world)

    field = wall_distance_field(world, goals)
    if world.player not in field:
        return None

    open_heap: List[Tuple[float, float, int, World, Tuple[Action, ...]]] = []
    heapq.heappush(open_heap, (field[world.player], 0.0, 0, world, ()))
    visited: Set[Tuple[Cell, FrozenSet[Cell]]] = set()
    counter = 1
    nodes = 0

    while open_heap and nodes < max_nodes:
        _f, g, _c, current, plan = heapq.heappop(open_heap)
        key = (current.player, current.rocks)
        if key in visited:
            continue
        visited.add(key)
        nodes += 1

        if current.player in goals:
            return list(plan), current

        for action, successor in _successors(current):
            state_key = (successor.player, successor.rocks)
            if state_key in visited:
                continue
            h = field.get(successor.player, float("inf"))
            if h == float("inf"):
                continue
            counter += 1
            heapq.heappush(
                open_heap,
                (g + 1 + h, h, counter, successor, plan + (action,)),
            )

    return None


# ---------------------------------------------------------------
# Retroceso (DFS con backtracking)
# ---------------------------------------------------------------
def backtracking_search(
    world: World,
    goals: FrozenSet[Cell],
    max_nodes: int = config.MAX_NODES_BACKTRACK,
    max_depth: int = config.MAX_DEPTH,
) -> Optional[Tuple[List[Action], World]]:
    """
    DFS con backtracking sobre el espacio de mundos. Prioriza las
    acciones que empujan rocas (para desbloquear corredores) y se
    deshace de ramas sin salida con `visited` + tope de profundidad.
    """
    if world.player in goals:
        return [], _collect(world)

    visited: Set[Tuple[Cell, FrozenSet[Cell], FrozenSet[Cell], bool]] = set()
    stack: List[Tuple[World, Tuple[Action, ...], int]] = [(world, (), 0)]
    nodes = 0

    while stack and nodes < max_nodes:
        current, plan, depth = stack.pop()
        key = (current.player, current.rocks, current.diamonds, current.has_key)
        if key in visited:
            continue
        visited.add(key)
        nodes += 1
        if depth >= max_depth:
            continue
        if current.player in goals:
            return list(plan), current

        # Empujes primero: es lo que desbloquea el camino bloqueado.
        successors = sorted(_successors(current), key=lambda s: s[0].kind != "push")
        for action, successor in reversed(successors):
            skey = (
                successor.player,
                successor.rocks,
                successor.diamonds,
                successor.has_key,
            )
            if skey in visited:
                continue
            stack.append((successor, plan + (action,), depth + 1))

    return None


# ---------------------------------------------------------------
# Planificador de alto nivel (subobjetivos)
# ---------------------------------------------------------------
def _goals_of(world: World):
    """
    Secuencia de subobjetivos segun el contenido del nivel. Cada
    diamante es un subobjetivo individual: si uno queda inalcanzable
    se marca pendiente sin perder la planificacion del resto, la llave
    y la salida.
    """
    for cell in sorted(world.diamonds):
        yield f"diamante {cell}", frozenset({cell})
    if world.key is not None and not world.has_key:
        yield "llave", frozenset({world.key})
    if world.exit is not None:
        yield "salida", frozenset({world.exit})


@dataclass
class Planner:
    """
    Orquesta los subobjetivos. Para cada meta intenta A* y, si se queda
    sin presupuesto o no hay camino, cae en la busqueda con retroceso.
    Tras cada etapa se recalculan las metas sobre el mundo avanzado
    (diamantes recolectados se van quitando de la lista).

    Devuelve un Plan (posiblemente incompleto si queda algo
    inalcanzable).
    """

    max_nodes_astar: int = config.MAX_NODES_ASTAR
    max_nodes_backtrack: int = config.MAX_NODES_BACKTRACK
    max_depth: int = config.MAX_DEPTH

    def plan(self, world: World) -> Plan:
        pending: List[str] = []
        actions: List[Action] = []
        # Metas que ya fracasaron (inalcanzables con el modelo): no se
        # vuelven a intentar en cada re-encolado.
        failed: Set[FrozenSet[Cell]] = set()
        current = world

        def enqueue() -> deque:
            # Metas pendientes del mundo actual, SIN las que el
            # personaje ya esta pisando ni las que ya fracasaron.
            return deque(
                (name, goals)
                for name, goals in _goals_of(current)
                if current.player not in goals and goals not in failed
            )

        queue = enqueue()
        while queue:
            name, goals = queue.popleft()
            result = a_star(current, goals, self.max_nodes_astar)
            if result is None:
                result = backtracking_search(
                    current, goals, self.max_nodes_backtrack, self.max_depth
                )
            if result is None:
                pending.append(name)
                failed.add(frozenset(goals))
                continue  # pasar a la siguiente meta; el mundo no cambia
            steps, current = result
            actions.extend(steps)
            # El mundo avanzo (p. ej. se recolecto un diamante):
            # recomputar las metas restantes sobre el estado nuevo.
            queue = enqueue()

        return Plan(
            actions=actions,
            final_world=current,
            complete=not pending,
            pending=pending,
            initial_world=world,
        )


# ---------------------------------------------------------------
# Simulacion (para verificacion)
# ---------------------------------------------------------------
def simulate(world: World, actions: Iterable[Action]) -> World:
    """Reproduce el plan sobre el mundo inicial; valida cada paso."""
    current = world
    for action in actions:
        result = _step(current, action.direction)
        if result is None or result[0] != action:
            raise ValueError(f"Paso invalido en el plan: {action}")
        current = _collect(result[1])
    return current
