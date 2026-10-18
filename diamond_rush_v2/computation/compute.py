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

La planificacion ocurre en fases:
  1) recoleccion GLOBAL: una sola busqueda con meta = todos los
     diamantes (las llaves, puertas, rejas y botones quedan
     instrumentales: el A* los usa solo si hacen falta);
  2) alcanzar la salida (solo con 0 diamantes);
  3) si la busqueda global se queda sin presupuesto, respaldo con la
     composicion clasica de subobjetivos (_plan_fallback): llaves,
     botones, diamantes uno por uno y salida.

Este modulo importa SOLO de perception.state (el contrato) y config;
nunca de cv2 ni de recognition.py.
"""
import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field, replace
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

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
    Modelo del nivel para la busqueda (estatico + estado dinamico).

    Todo son conjuntos de celdas (fila, columna):

    - `walls`: celdas fijas intransitables (pared, picos activos).
    - `spikes`: picos RETRAIDOS (piso hasta ahora). Son dinamicos: en el
      juego real se activan permanentemente cuando el personaje sale de la
      celda; ver `_step`. Una vez activados pasan a `walls`.
    - `rocks`: celdas ocupadas por una roca empujable.
    - `diamonds`: diamantes por recolectar.
    - `holes`: hoyos abiertos (bloquean salvo que se empuje una roca
      dentro; al rellenarse el hoyo pasa a piso transitable).
    - `gates`: rejas cerradas (bloquean hasta que se pisa un boton).
    - `buttons`: botones (piso; placas de presion que abren una reja).
    - `pressed`: botones ya activados (por el personaje o por una roca).
    - `doors`: puertas con cerradura (bloquean hasta tener una llave).
    - `keys`: llaves sueltas por recoger.
    - `has_key`: el personaje lleva una llave (habilita `doors`). Lleva
      como maximo UNA: abrir una puerta la consume (vuelve a `False`) y
      pisar otra llave mientras se lleva una no la recoge.
    - `exit`: celda de la salida (escape/totem); bloquea hasta que no
      queden diamantes (`exit_unlocked`).
    - `exit_unlocked`: la salida ya es transitable (0 diamantes).

    `nodes` y `adjacency` son el grafo de piso que aporta la percepcion
    (GameState): la topologia "pelada" (ignora objetos) de celdas de
    suelo alcanzables desde el personaje. Se usan para generar vecinos
    y para el campo de distancias, pero NO reemplazan a `walls`: un
    peligro es piso en `adjacency` y aun asi debe bloquear. Si el grafo
    viene vacio (p. ej. fixtures de test), se cae al vecindario por
    `DELTA` sobre la grilla.
    """

    rows: int
    cols: int
    player: Cell
    walls: FrozenSet[Cell] = frozenset()
    spikes: FrozenSet[Cell] = frozenset()
    rocks: FrozenSet[Cell] = frozenset()
    diamonds: FrozenSet[Cell] = frozenset()
    holes: FrozenSet[Cell] = frozenset()
    gates: FrozenSet[Cell] = frozenset()
    buttons: FrozenSet[Cell] = frozenset()
    # Botones ya activados (por el personaje o por una roca encima). Es
    # monotono: una reja abierta no se vuelve a cerrar.
    pressed: FrozenSet[Cell] = frozenset()
    # Emparejamiento boton -> reja que abre (la percepcion no lo dice; se
    # infiere por cercania global, 1:1).
    button_gate: Dict[Cell, Cell] = field(default_factory=dict)
    doors: FrozenSet[Cell] = frozenset()
    keys: FrozenSet[Cell] = frozenset()
    has_key: bool = False
    exit: Optional[Cell] = None
    exit_unlocked: bool = False
    nodes: FrozenSet[Cell] = frozenset()
    adjacency: Dict[Cell, List[Cell]] = field(default_factory=dict)

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
# El rol decide a que capa del World se agrega la celda (ver _ROLE_TO_WORLD).
ROLES = {
    "diamond": "goal",
    "key": "key",
    "door_key": "door",
    "scape_block": "exit",
    "scape_enable": "exit",
    "totem": "exit",
    "open_gate": "floor",
    "rock": "rock",
    "rock_in_hole": "floor",
    "closed_gate": "gate",
    "hole": "hole",
    "spike": "hazard",
    "no_spike": "retracted_spike",
    "button": "button",
    "player": "player",
}
# Rol -> capas del World a las que aporta la celda. Los roles "special"
# (key/exit) se manejan aparte por ser singulares / con estado.
_ROLE_TO_WORLD = {
    "goal": ("diamonds",),
    "rock": ("rocks",),
    "fixed": ("walls",),
    "hazard": ("walls",),
    "retracted_spike": ("spikes",),
    "hole": ("holes",),
    "gate": ("gates",),
    "button": ("buttons",),
    "door": ("doors",),
}


def world_from_state(state: GameState, roles: Optional[Dict[str, str]] = None) -> World:
    """
    Traduce el GameState percibido (grid + detecciones) al World de
    busqueda. Los peligros estaticos (picos activos) se tratan como
    muros; los picos retraidos (`no_spike`) van a la capa `spikes` para
    que el planificador simule su activacion; los estados dinamicos
    (hoyos, rejas, puertas, salida) se guardan como capas propias para
    que el planificador los simule.
    """
    roles = roles or ROLES
    grid = state.grid.tolist()
    rows, cols = len(grid), (len(grid[0]) if grid else 0)

    walls = {(i, j) for i in range(rows) for j in range(cols) if not grid[i][j]}
    # Una celda que el grafo de piso de la percepcion considera transitable
    # NO es muro, aunque el grid la marque como no-camino (las rejas,
    # puertas o la salida se dibujan como tile aparte y pueden caer fuera
    # de la mascara de caminos). Sin esto, una reja quedaria atrapada en
    # `walls` y no se podria pisar ni al abrirla.
    if state.graph_nodes:
        walls -= set(state.graph_nodes)
    rocks: Set[Cell] = set()
    diamonds: Set[Cell] = set()
    spikes: Set[Cell] = set()
    holes: Set[Cell] = set()
    gates: Set[Cell] = set()
    buttons: Set[Cell] = set()
    doors: Set[Cell] = set()
    keys: Set[Cell] = set()
    exit_cell = None

    # Cada deteccion agrega su rol al mundo (una celda puede tener
    # varios objetos: el primero manda sobre la transitabilidad).
    for obj in state.objects:
        role = roles.get(obj.name, "floor")
        cell = state.cell_of(obj.center)
        if role in _ROLE_TO_WORLD:
            for target in _ROLE_TO_WORLD[role]:
                {
                    "walls": walls,
                    "rocks": rocks,
                    "diamonds": diamonds,
                    "spikes": spikes,
                    "holes": holes,
                    "gates": gates,
                    "buttons": buttons,
                    "doors": doors,
                }[target].add(cell)
        elif role == "key":
            keys.add(cell)
        elif role == "exit":
            exit_cell = cell

    return World(
        rows=rows,
        cols=cols,
        walls=frozenset(walls),
        spikes=frozenset(spikes),
        rocks=frozenset(rocks),
        player=state.player_cell,
        diamonds=frozenset(diamonds),
        holes=frozenset(holes),
        gates=frozenset(gates),
        buttons=frozenset(buttons),
        button_gate=_match_buttons_to_gates(buttons, gates),
        doors=frozenset(doors),
        keys=frozenset(keys),
        # Capacidad 1 + consumo: como el nivel tiene igual numero de
        # llaves y puertas (K0 == D0), si quedan mas puertas cerradas que
        # llaves en el piso, la que falta la lleva el personaje. Esto
        # reconstruye has_key en cualquier re-plan sin mirar la mano del
        # personaje (ver _collect).
        has_key=(len(doors) - len(keys) == 1),
        exit=exit_cell,
        exit_unlocked=False,
        nodes=frozenset(state.graph_nodes),
        adjacency=state.adjacency,
    )


def _match_buttons_to_gates(
    buttons: Iterable[Cell], gates: Iterable[Cell]
) -> Dict[Cell, Cell]:
    """
    Empareja botones con rejas 1:1 minimizando la distancia Manhattan
    total (la percepcion no indica que reja abre cada boton). Es una
    conjetura: el lazo cerrado la corrige si el juego abre otra reja.
    """
    bs = sorted(buttons)
    gs = sorted(gates)
    if not bs or not gs:
        return {}
    distance = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    if len(gs) <= 6:
        best = None
        best_cost = None
        for perm in itertools.permutations(gs, min(len(bs), len(gs))):
            cost = sum(distance(b, g) for b, g in zip(bs, perm))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = perm
        return dict(zip(bs, best))
    pairs = sorted(
        (distance(b, g), b, g) for b in bs for g in gs
    )
    used_b: Set[Cell] = set()
    used_g: Set[Cell] = set()
    mapping: Dict[Cell, Cell] = {}
    for _d, b, g in pairs:
        if b in used_b or g in used_g:
            continue
        used_b.add(b)
        used_g.add(g)
        mapping[b] = g
    return mapping


# ---------------------------------------------------------------
# Topologia: el grafo de la percepcion, con fallback a la grilla
# ---------------------------------------------------------------
def _neighbors(world: World, cell: Cell) -> Iterable[Cell]:
    """
    Vecinos de piso de una celda. Si el World trae el grafo de la
    percepcion, se usan sus aristas; si no, se derivan de la grilla con
    `DELTA` (comportamiento previo, para fixtures sin grafo).
    """
    if world.adjacency:
        return world.adjacency.get(cell, ())
    return (neighbor for neighbor in _grid_neighbors(world, cell))


def _grid_neighbors(world: World, cell: Cell) -> Iterable[Cell]:
    """Vecinos de 4 direcciones derivados de la grilla (sin el grafo)."""
    for df, dc in DELTA.values():
        neighbor = (cell[0] + df, cell[1] + dc)
        if world.in_bounds(neighbor) and neighbor not in world.walls:
            yield neighbor


def _blocked(world: World, cell: Cell) -> bool:
    """
    La celda bloquea el paso por su estado dinamico actual (aunque sea
    piso en la topologia): hoyo abierto, reja cerrada, puerta sin llave,
    o salida aun bloqueada. No incluye rocas (se empujan aparte) ni los
    muros fijos (esos estan en `walls`).
    """
    if cell in world.holes:
        return True
    if cell in world.gates:
        return True
    if cell in world.doors and not world.has_key:
        return True
    if cell == world.exit and not world.exit_unlocked:
        return True
    return False


def _is_floor(world: World, cell: Cell) -> bool:
    """La celda existe como piso transitable (no es muro ni esta fuera)."""
    if not world.in_bounds(cell) or cell in world.walls:
        return False
    if world.nodes and cell not in world.nodes:
        return False
    if _blocked(world, cell):
        return False
    return True


def _connected(world: World, origin: Cell, target: Cell) -> bool:
    """Hay un paso directo de piso entre dos celdas: son vecinas en la topologia."""
    if world.adjacency:
        return target in world.adjacency.get(origin, ())
    return target in set(_grid_neighbors(world, origin))


# ---------------------------------------------------------------
# Sucesores
# ---------------------------------------------------------------
def _leave_spike(world: World, origin: Cell) -> World:
    """
    Al SALIR de una celda de pico retraido, el pico se activa para
    siempre: la celda pasa de `spikes` a `walls` (el camino queda cortado
    en esa direccion). Activar solo se hace al salir, nunca al entrar.
    """
    if origin not in world.spikes:
        return world
    return replace(
        world,
        spikes=frozenset(world.spikes - {origin}),
        walls=frozenset(world.walls | {origin}),
    )


def _step(world: World, direction: str) -> Optional[Tuple[Action, World]]:
    """
    Aplica un movimiento o un empuje desde el estado actual.
    Devuelve (action, world) o None si la celda destino esta bloqueada.
    """
    d = DELTA[direction]
    origin = world.player
    target = (origin[0] + d[0], origin[1] + d[1])

    # La topologia la manda el grafo de piso (o la grilla si no hay):
    # el destino tiene que ser una celda de piso conectada y no un
    # peligro/fijo (`walls`).
    if not _connected(world, origin, target) or not _is_floor(world, target):
        return None

    if target in world.rocks:
        # Empujar la roca a la celda de mas alla. Hacia un hoyo lo
        # rellena (la roca queda fija); hacia un diamante solo lo TAPA:
        # las rocas no recolectan diamantes, solo el jugador al pisar la
        # celda (ver _collect), asi que el diamante sigue en `diamonds`.
        push = (target[0] + d[0], target[1] + d[1])
        filling = push in world.holes
        if not _connected(world, target, push):
            return None
        if not filling and not _is_floor(world, push):
            return None
        # No empujar una roca sobre otras rocas, llaves, rejas, puertas ni
        # la salida (sobre botones si: a veces es la unica salida, nivel 14).
        if (
            push in world.rocks
            or push in world.keys
            or push in world.gates
            or push in world.doors
            or push == world.exit
        ):
            return None
        # Tapa temporal si; tapa permanente no: una roca aterrada sobre un
        # diamante solo se permite si la celda de mas alla (misma
        # direccion) le deja salir (piso u hoyo rellenable). Una roca
        # atrapada sobre un diamante lo destruiria irrevocablemente (p. ej.
        # nivel 4, donde el A* miope de un diamante empujaba la roca contra
        # el ultimo diamante rodeado de muros y el nivel quedaba sin
        # solucion). La tapa temporal sigue permitida porque a veces es el
        # unico modo de atravesar un corredor de una casilla.
        if push in world.diamonds:
            beyond = (push[0] + d[0], push[1] + d[1])
            can_leave = _connected(world, push, beyond) and (
                _is_floor(world, beyond) or beyond in world.holes
            )
            if not can_leave:
                return None
        holes = world.holes
        if filling:
            holes = frozenset(world.holes - {push})
            rocks = frozenset(world.rocks - {target})
        else:
            rocks = frozenset((world.rocks - {target}) | {push})
        return Action("push", direction), _leave_spike(
            replace(world, rocks=rocks, holes=holes, player=target), origin
        )
    else:
        return Action("move", direction), _leave_spike(
            replace(world, player=target), origin
        )


def _open_nearest_gate(world: World, button: Cell) -> FrozenSet[Cell]:
    """
    Activar un boton abre su reja emparejada (si sigue cerrada); si no
    hay emparejamiento (fixtures sin `button_gate`), la reja mas cercana.
    Al pisar todos los botones alcanzables se abren todas las rejas; el
    lazo cerrado corrige si el juego abrio otra.
    """
    if not world.gates:
        return world.gates
    gate = world.button_gate.get(button)
    if gate is None or gate not in world.gates:
        gate = min(
            world.gates,
            key=lambda g: (abs(g[0] - button[0]) + abs(g[1] - button[1]), g),
        )
    return frozenset(world.gates - {gate})


def _collect(world: World) -> World:
    """
    Ultima transicion tras pisar una celda: recoge diamantes y llaves,
    abre puertas, activa botones (por el personaje o por una roca encima)
    y desbloquea la salida al no quedar diamantes.

    El personaje lleva UNA sola llave: pisar una puerta la abre (se quita
    de `doors`) y consume la llave; pisar una llave solo la recoge si no
    se lleva ninguna (si ya se lleva, la del piso queda intacta).
    """
    player = world.player
    diamonds = world.diamonds - {player}
    keys = world.keys
    doors = world.doors
    has_key = world.has_key
    # Abrir la puerta pisada consume la llave que se lleva.
    if player in doors and has_key:
        doors = doors - {player}
        has_key = False
    # Recoger una llave solo si no se lleva ninguna (capacidad 1).
    if player in keys and not has_key:
        keys = keys - {player}
        has_key = True
    # Un boton es una placa de presion: se activa si lo pisa el personaje
    # o si queda una roca encima. Cada boton recien activado abre la reja
    # mas cercana (la percepcion no dice que reja abre cual boton).
    covered = world.buttons & (world.rocks | {player})
    new_pressed = covered - world.pressed
    gates = world.gates
    pressed = world.pressed
    if new_pressed:
        pressed = frozenset(world.pressed | new_pressed)
        for button in sorted(new_pressed):
            gates = _open_nearest_gate(replace(world, gates=gates), button)
    exit_unlocked = world.exit_unlocked or (
        world.exit is not None and not diamonds
    )

    if (
        diamonds == world.diamonds
        and keys == world.keys
        and doors == world.doors
        and has_key == world.has_key
        and pressed == world.pressed
        and gates == world.gates
        and exit_unlocked == world.exit_unlocked
    ):
        return world
    return replace(
        world,
        diamonds=frozenset(diamonds),
        keys=frozenset(keys),
        doors=frozenset(doors),
        has_key=has_key,
        pressed=pressed,
        gates=frozenset(gates),
        exit_unlocked=exit_unlocked,
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
    BFS multi-fuente desde las metas sobre la topologia "pelada":
    ignora rocas Y bloqueos dinamicos (hoyos, rejas, puertas, salida
    bloqueada). Distancia real en casillas (no Manhattan): ya toma en
    cuenta los muros del laberinto. Al ignorar lo que puede despejarse
    (rellenar un hoyo, abrir una reja), subestima el costo real ->
    heuristica ADMISIBLE; y permite que A* encuentre metas detras de un
    bloqueo resoluble. Los vecinos salen del grafo de la percepcion (o
    de la grilla si no hay grafo).
    """
    dist: Dict[Cell, int] = {}
    queue = deque()
    for goal in goals:
        if world.in_bounds(goal) and goal not in world.walls:
            dist.setdefault(goal, 0)
            queue.append((goal, 0))

    while queue:
        cell, d = queue.popleft()
        for neighbor in _neighbors(world, cell):
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
def _state_key(world: World) -> tuple:
    """
    Clave de deduplicacion de un estado de busqueda. Incluye TODO el
    estado dinamico relevante (no solo posicion y rocas): dos mundos con
    el mismo personaje y rocas pero distinto inventario/rejas/hoyos NO
    son el mismo estado.
    """
    return (
        world.player,
        world.rocks,
        world.diamonds,
        world.spikes,
        world.keys,
        world.has_key,
        world.buttons,
        world.pressed,
        world.gates,
        world.doors,
        world.holes,
        world.exit_unlocked,
    )


def _reached(world: World, goals: FrozenSet[Cell]) -> bool:
    """
    El subobjetivo esta resuelto: o el personaje esta encima, o la meta
    es un boton ya activado (basta con que lo cubra una roca).
    """
    return world.player in goals or bool(goals & world.pressed)


def _successes_for(name: str, goals: FrozenSet[Cell]) -> List[Callable[[World], bool]]:
    """
    Predicados de exito de un subobjetivo, en orden de preferencia.

    Los diamantes solo se recolecta pisando la celda: una roca puede
    taparlos pero nunca recogerlos, asi que el unico exito posible es
    que el personaje este encima. Para llaves solo vale la DESAPARICION:
    como el personaje lleva una sola, pisar la celda de otra llave no la
    recoge, asi que "pisar" daria un falso positivo. Botones y salida
    requieren al personaje encima (o el boton activado).
    """
    if name.startswith("diamante"):
        return [lambda world: world.player in goals]
    if name.startswith("llave"):
        return [lambda world: not (goals & world.keys)]
    if name.startswith("hoyo"):
        # Un hoyo esta "resuelto" cuando se rellena (desaparece de la capa).
        return [lambda world: not (goals & world.holes)]
    return [lambda world: _reached(world, goals)]


def _bare_path(world: World, target: Cell) -> Optional[List[Cell]]:
    """
    Camino mas corto del personaje a `target` sobre la topologia "pelada"
    (ignora rocas y bloqueos dinamicos, igual que la heuristica). Se usa
    para descubrir que hoyos cortan el paso a una meta. Devuelve la lista
    de celdas desde el personaje (inclusive) hasta la meta, o None.
    """
    prev: Dict[Cell, Optional[Cell]] = {world.player: None}
    queue = deque([world.player])
    while queue:
        cell = queue.popleft()
        if cell == target:
            break
        for neighbor in _neighbors(world, cell):
            if (
                world.in_bounds(neighbor)
                and neighbor not in world.walls
                and neighbor not in prev
            ):
                prev[neighbor] = cell
                queue.append(neighbor)
    if target not in prev:
        return None
    path = []
    cell: Optional[Cell] = target
    while cell is not None:
        path.append(cell)
        cell = prev[cell]
    path.reverse()
    return path


def a_star(
    world: World,
    goals: FrozenSet[Cell],
    max_nodes: int = config.MAX_NODES_ASTAR,
    weight: float = config.A_STAR_WEIGHT,
    success: Optional[Callable[[World], bool]] = None,
) -> Optional[Tuple[List[Action], World]]:
    """
    A* ponderado sobre el espacio de mundos completos. Devuelve
    (actions, final_world) o None si no encuentra solucion dentro del
    presupuesto. Con `weight>1` el sesgo hacia la meta es mas fuerte:
    encuentra plan antes (a cambio de perder optimalidad), util en los
    puzzles Sokoban de los niveles finales. Mueve, empuja rocas, rellena
    hoyos y alcanza la meta. `success` permite definir cuando el
    subobjetivo esta logrado (por defecto, pisar la meta).
    """
    reached = success or (lambda w: _reached(w, goals))
    if reached(world):
        return [], _collect(world)

    field = wall_distance_field(world, goals)
    if world.player not in field:
        return None

    open_heap: List[Tuple[float, float, int, World, Tuple[Action, ...]]] = []
    heapq.heappush(open_heap, (weight * field[world.player], 0.0, 0, world, ()))
    visited: Set[tuple] = set()
    counter = 1
    nodes = 0

    while open_heap and nodes < max_nodes:
        _f, g, _c, current, plan = heapq.heappop(open_heap)
        key = _state_key(current)
        if key in visited:
            continue
        visited.add(key)
        nodes += 1

        if reached(current):
            return list(plan), current

        for action, successor in _successors(current):
            state_key = _state_key(successor)
            if state_key in visited:
                continue
            h = field.get(successor.player, float("inf"))
            if h == float("inf"):
                continue
            counter += 1
            heapq.heappush(
                open_heap,
                (g + 1 + weight * h, g + 1, counter, successor, plan + (action,)),
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
    success: Optional[Callable[[World], bool]] = None,
) -> Optional[Tuple[List[Action], World]]:
    """
    DFS con backtracking sobre el espacio de mundos. Prioriza las
    acciones que empujan rocas (para desbloquear corredores) y se
    deshace de ramas sin salida con `visited` + tope de profundidad.
    """
    reached = success or (lambda w: _reached(w, goals))
    if reached(world):
        return [], _collect(world)

    visited: Set[tuple] = set()
    stack: List[Tuple[World, Tuple[Action, ...], int]] = [(world, (), 0)]
    nodes = 0

    while stack and nodes < max_nodes:
        current, plan, depth = stack.pop()
        key = _state_key(current)
        if key in visited:
            continue
        visited.add(key)
        nodes += 1
        if depth >= max_depth:
            continue
        if reached(current):
            return list(plan), current

        # Empujes primero: es lo que desbloquea el camino bloqueado.
        successors = sorted(_successors(current), key=lambda s: s[0].kind != "push")
        for action, successor in reversed(successors):
            if _state_key(successor) in visited:
                continue
            stack.append((successor, plan + (action,), depth + 1))

    return None


# ---------------------------------------------------------------
# Planificador de alto nivel (subobjetivos)
# ---------------------------------------------------------------
def _goals_of(world: World):
    """
    Secuencia de subobjetivos segun el contenido del nivel. Solo la usa
    el respaldo _plan_fallback (la busqueda global no los necesita):

    1) llaves (abren las puertas, que pueden tapar el resto);
    2) botones (abren las rejas, que pueden tapar el resto);
    3) cada diamante, uno por uno;
    4) la salida, SOLO cuando ya no quedan diamantes.

    Cada subobjetivo es individual: si uno queda inalcanzable se marca
    pendiente sin perder la planificacion del resto.
    """
    for cell in sorted(world.keys):
        yield f"llave {cell}", frozenset({cell})
    for cell in sorted(world.buttons):
        if cell in world.pressed:
            continue
        yield f"boton {cell}", frozenset({cell})
    for cell in sorted(world.diamonds):
        yield f"diamante {cell}", frozenset({cell})
    if world.exit is not None and world.exit_unlocked:
        yield "salida", frozenset({world.exit})


@dataclass
class Planner:
    """
    Planifica en fases:

    1) recoleccion GLOBAL: una sola busqueda con meta = todos los
       diamantes. Elegir el orden de visita con vision de todo el nivel
       evita decisiones irreversibles que destruyan metas posteriores
       (cruzar un pico en mal orden, tapar el ultimo diamante con una
       roca, etc.). Llaves, puertas, rejas y botones quedan
       instrumentales: el A* los usa solo si hacen falta.
    2) salida: una busqueda a la celda de la salida con 0 diamantes.
    3) si la global se queda sin presupuesto, respaldo con la
       composicion clasica de subobjetivos (_plan_fallback): llaves,
       botones, diamantes uno por uno y salida, con A* y respaldo de
       retroceso por meta.

    Devuelve un Plan (posiblemente incompleto si queda algo
    inalcanzable).
    """

    max_nodes_astar: int = config.MAX_NODES_ASTAR
    max_nodes_backtrack: int = config.MAX_NODES_BACKTRACK
    max_depth: int = config.MAX_DEPTH

    def plan(self, world: World) -> Plan:
        # El mundo del planificador arranca tras recolectar lo que el
        # personaje ya pisa (y desbloqueando la salida si no hay gemas).
        current = _collect(world)
        actions: List[Action] = []
        if current.diamonds:
            result = a_star(
                current,
                frozenset(current.diamonds),
                self.max_nodes_astar,
                success=lambda w: not w.diamonds,
            )
            if result is None:
                return self._plan_fallback(world)
            steps, current = result
            actions.extend(steps)
        # La salida solo es meta con 0 diamantes (ya desbloqueada).
        pending: List[str] = []
        if current.exit is not None and not current.diamonds:
            exit_cell = current.exit
            goals = frozenset({exit_cell})
            result = a_star(current, goals, self.max_nodes_astar)
            if result is None:
                # Mismo respaldo que el resolve del fallback: puzzles de
                # salida con rocas necesitan el retroceso (p. ej. nivel 20).
                result = backtracking_search(
                    current,
                    goals,
                    self.max_nodes_backtrack,
                    self.max_depth,
                )
            if result is None:
                # Sin diamantes que recolectar, el fallback entero solo
                # persigue la salida (incluye el macro de hoyos).
                if not world.diamonds:
                    return self._plan_fallback(world)
                pending.append("salida")
            else:
                steps, current = result
                actions.extend(steps)
        return Plan(
            actions=actions,
            final_world=current,
            complete=not pending,
            pending=pending,
            initial_world=world,
        )

    def _plan_fallback(self, world: World) -> Plan:
        # Composicion clasica de subobjetivos (respaldo de la busqueda
        # global). Para cada meta intenta A* y, si se queda sin
        # presupuesto o no hay camino, cae en la busqueda con retroceso.
        # Tras cada etapa se recalculan las metas sobre el mundo avanzado
        # (diamantes recolectados se van quitando de la lista).
        actions: List[Action] = []
        # El mundo del planificador arranca tras recolectar lo que el
        # personaje ya pisa (y desbloqueando la salida si no hay gemas).
        current = _collect(world)
        # Metas que ya fracasaron en el mundo actual: no se reintentan
        # mientras el mundo no avance.
        failed: Set[FrozenSet[Cell]] = set()
        pending: List[str] = []

        def mark_pending(name: str) -> None:
            if name not in pending:
                pending.append(name)

        def enqueue() -> deque:
            # Metas del mundo actual, SIN las que ya estan logradas ni
            # las que ya fracasaron.
            return deque(
                (name, goals)
                for name, goals in _goals_of(current)
                if not any(s(current) for s in _successes_for(name, goals))
                and goals not in failed
            )

        def resolve(name: str, goals: FrozenSet[Cell], astar_budget: int):
            # Prueba los predicados de exito en orden: primero "pisar la
            # meta", luego "el objeto desaparecio". Para cada uno intenta
            # A* y, si no hay camino, el retroceso.
            for success in _successes_for(name, goals):
                result = a_star(current, goals, astar_budget, success=success)
                if result is None:
                    result = backtracking_search(
                        current,
                        goals,
                        self.max_nodes_backtrack,
                        self.max_depth,
                        success=success,
                    )
                if result is not None:
                    return result
            return None

        queue = enqueue()
        while queue:
            name, goals = queue.popleft()
            # Pruning con el grafo de la percepcion: una meta que no es
            # nodo de piso alcanzable jamas se puede resolver, asi que
            # se marca pendiente sin gastar presupuesto de busqueda.
            if current.nodes and not goals <= current.nodes:
                mark_pending(name)
                failed.add(frozenset(goals))
                continue
            # Hoyos abiertos que cortan el camino "pelado" a esta meta
            # (macro de Sokoban: habra que rellenarlos). Si los hay, se
            # intenta resolver con presupuesto pequeno y, si falla, se
            # expande en subobjetivos de relleno.
            target = next(iter(goals), None)
            path = (
                _bare_path(current, target)
                if target is not None
                and current.holes
                and not name.startswith("hoyo")
                else None
            )
            blockers = [c for c in path if c in current.holes] if path else []
            budget = (
                config.MAX_NODES_HOLE
                if (blockers or name.startswith("hoyo"))
                else self.max_nodes_astar
            )
            result = resolve(name, goals, budget)
            if result is None:
                new_blockers = [
                    h for h in blockers if frozenset({h}) not in failed
                ]
                if new_blockers:
                    items = [
                        (f"hoyo {h}", frozenset({h})) for h in new_blockers
                    ] + [(name, goals)]
                    for item in reversed(items):
                        queue.appendleft(item)
                    continue
                mark_pending(name)
                failed.add(frozenset(goals))
                continue  # pasar a la siguiente meta; el mundo no cambia
            steps, current = result
            actions.extend(steps)
            if name in pending:
                pending.remove(name)
            # El mundo avanzo de forma monotona (se recolecto algo, se
            # abrio una reja...): reintentar lo que antes habia fallado.
            failed.clear()
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
