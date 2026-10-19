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

import button_gate
import config
import level_hints
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
    - `gates`: rejas CERRADAS en este instante (bloquean el paso). La
      apertura es TEMPORAL: la reja se abre solo mientras su boton este
      cubierto y vuelve a entrar en esta capa al descubrirlo (ver
      `_gates_actuales`).
    - `all_gates`: todas las rejas del nivel, abiertas o cerradas
      (constante durante las transiciones; hace de memoria para poder
      recerrar una reja temporal).
    - `buttons`: botones (piso; placas de presion que abren una reja).
    - `pressed`: botones cubiertos (por el personaje o por una roca).
      Es derivado de la cobertura actual: NO es monotonico (al descubrir
      el boton se vacia y su reja se cierra).
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
    # Todas las rejas del nivel (abiertas o cerradas). Constante durante
    # las transiciones: es la memoria que permite RECERRAR una reja
    # temporal al descubrir su boton. Se rellena en world_from_state y
    # se auto-sana en _collect (all_gates | gates) para fixtures de test.
    all_gates: FrozenSet[Cell] = frozenset()
    buttons: FrozenSet[Cell] = frozenset()
    # Botones cubiertos (personaje o roca encima). Derivado y NO monotono
    # (una reja abierta se cierra al soltar su boton).
    pressed: FrozenSet[Cell] = frozenset()
    # Emparejamiento boton -> reja que abre (la percepcion no lo dice). Se
    # declara por nivel en button_gate.py; lo no declarado se infiere por
    # cercania global, 1:1.
    button_gate: Dict[Cell, Cell] = field(default_factory=dict)
    # Pistas de subobjetivos por nivel (level_hints.py): una lista
    # ordenada de (nombre, meta, rock_goal). Son PREFERENCIAS que el
    # respaldo resuelve primero; se revalidan contra lo percibido en
    # world_from_state. No es parte de la dinamica, asi que no entra en
    # `_state_key` ni la tocan las transiciones.
    hints: List[Tuple[str, FrozenSet[Cell], Optional[Cell]]] = field(
        default_factory=list
    )
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


def world_from_state(
    state: GameState,
    roles: Optional[Dict[str, str]] = None,
    level: Optional[int] = None,
) -> World:
    """
    Traduce el GameState percibido (grid + detecciones) al World de
    busqueda. Los peligros estaticos (picos activos) se tratan como
    muros; los picos retraidos (`no_spike`) van a la capa `spikes` para
    que el planificador simule su activacion; los estados dinamicos
    (hoyos, rejas, puertas, salida) se guardan como capas propias para
    que el planificador los simule.

    `level` (1..20) habilita el mapeo explicito boton->reja de
    button_gate.py; sin el se empareja por cercania (ver
    _button_gate_map). Tambien habilita la secuencia de subobjetivos de
    level_hints.py (`World.hints`), validada contra lo percibido.
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
        all_gates=frozenset(gates),
        buttons=frozenset(buttons),
        button_gate=_button_gate_map(level, buttons, gates),
        hints=_hints_from_level(level, grid, holes, buttons, keys, doors, diamonds),
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


def _button_gate_map(
    level: Optional[int], buttons: Set[Cell], gates: Set[Cell]
) -> Dict[Cell, Cell]:
    """
    Emparejamiento boton->reja 1:1.

    Si `level` trae un mapeo explicito en button_gate.py, se usan las
    entradas cuyas DOS celdas existan en lo percibido; el resto de
    botones/rejas se completa por cercania. Sin `level` (o sin entrada
    para ese nivel) todo se resuelve por cercania. La validacion de
    celdas evita aplicar un mapeo si el nivel percibido no coincide con
    el declarado.
    """
    explicito = (
        button_gate.BUTTON_GATE.get(level, {}) if level is not None else {}
    )
    mapping = {b: g for b, g in explicito.items() if b in buttons and g in gates}
    pendientes_b = buttons - set(mapping)
    pendientes_g = gates - set(mapping.values())
    mapping.update(_match_buttons_to_gates(pendientes_b, pendientes_g))
    return mapping


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


def _hints_from_level(
    level: Optional[int],
    grid: List[List[int]],
    holes: Set[Cell],
    buttons: Set[Cell],
    keys: Set[Cell],
    doors: Set[Cell],
    diamonds: Set[Cell],
) -> List[Tuple[str, FrozenSet[Cell], Optional[Cell]]]:
    """
    Traduce la secuencia declarada en level_hints.py a subobjetivos
    validos para ESTE estado percibido.

    Cada entrada se revalida: si la celda ya no existe como el objeto
    esperado (p. ej. el tester ya recogio la llave o relleno el hoyo), la
    entrada se ignora y el planificador cae a la composicion automatica;
    asi la secuencia funciona tambien con el nivel a medias. Un
    `("roca", origen, destino)` se normaliza segun el destino: hoyo ->
    `"hoyo ..."`, boton -> `"boton ..."`, otra celda de piso ->
    `"roca ..."`, todos con `rock_goal=destino` (el origen es informativo:
    el A* usa la roca mas cercana que pueda empujar). Un destino que no
    sea hoyo, boton ni piso transitable se descarta.
    """
    if level is None:
        return []
    spec = level_hints.LEVEL_SEQUENCE.get(level, [])
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    hints: List[Tuple[str, FrozenSet[Cell], Optional[Cell]]] = []
    for item in spec:
        if not item:
            continue
        kind = item[0]
        if kind in ("llave", "puerta", "diamante", "hoyo"):
            if len(item) != 2:
                continue
            cell = tuple(item[1])
            existe = (
                (kind == "llave" and cell in keys)
                or (kind == "puerta" and cell in doors)
                or (kind == "diamante" and cell in diamonds)
                or (kind == "hoyo" and cell in holes)
            )
            if not existe:
                continue
            hints.append((f"{kind} {cell}", frozenset({cell}), None))
            continue
        if kind == "roca":
            if len(item) != 3:
                continue
            dest = tuple(item[2])
            r, c = dest
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if dest in holes:
                hints.append((f"hoyo {dest}", frozenset({dest}), dest))
            elif dest in buttons:
                hints.append((f"boton {dest}", frozenset({dest}), dest))
            elif grid[r][c]:
                hints.append((f"roca {dest}", frozenset({dest}), dest))
            # destino muro / inexistente -> se ignora
    return hints


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


def _estado_valido(world: World) -> bool:
    """
    True si el mundo resultante de un paso es fisicamente coherente: ni
    el personaje ni una roca pueden quedar sobre una reja CERRADA. Con
    rejas temporales al salir de un boton la reja se cierra, asi que no
    vale "colarse" a la casilla de la reja en el mismo paso (entrar
    exigia tener el boton pisado, que el propio movimiento ya no deja).
    """
    collected = _collect(world)
    if collected.player in collected.gates:
        return False
    if collected.rocks & collected.gates:
        return False
    return True


def _step(
    world: World,
    direction: str,
    allow_spike_activation: bool = True,
    allow_rock_on_diamond: bool = True,
) -> Optional[Tuple[Action, World]]:
    """
    Aplica un movimiento o un empuje desde el estado actual.
    Devuelve (action, world) o None si la celda destino esta bloqueada.

    El A* miope de un subobjetivo podria tomar decisiones irreversibles
    sin ver las consecuencias para el resto del nivel. Dos banderas las
    restringen cuando se planifican subobjetivos:

    - `allow_spike_activation=False`: no se sale de un pico retraido
      (activarlo lo dejaria como muro para siempre, cortando el unico
      camino de retorno).
    - `allow_rock_on_diamond=False`: no se empuja una roca sobre un
      diamante (la perception ya no lo detectaria en re-verificaciones).
    """
    d = DELTA[direction]
    origin = world.player
    target = (origin[0] + d[0], origin[1] + d[1])

    # En modo no destructivo no se sale de un pico: activarlo es
    # irreversible y puede cortar el camino de vuelta.
    if not allow_spike_activation and origin in world.spikes:
        return None

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
            if not allow_rock_on_diamond:
                return None
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
        nuevo = _leave_spike(
            replace(world, rocks=rocks, holes=holes, player=target), origin
        )
        if not _estado_valido(nuevo):
            return None
        return Action("push", direction), nuevo
    else:
        nuevo = _leave_spike(replace(world, player=target), origin)
        if not _estado_valido(nuevo):
            return None
        return Action("move", direction), nuevo


def _es_temporal(world: World) -> bool:
    """
    True cuando la apertura de la reja es TEMPORAL: hay que mantener el
    boton cubierto (personaje o roca encima) para que la reja no se
    cierre. Vale para CUALQUIER numero de pares boton-reja (el
    emparejamiento es 1:1 via `button_gate`). No hay rejas sin boton.
    """
    return bool(world.buttons) and bool(world.all_gates or world.gates)


def _gates_actuales(world: World, covered: FrozenSet[Cell]) -> FrozenSet[Cell]:
    """
    Rejas cerradas tras cubrir/descubrir botones.

    Cada reja esta ABIERTA solo mientras SU boton esta cubierto ahora
    (personaje o roca encima) y se cierra al descubrirlo. El
    emparejamiento es 1:1 (`button_gate`); sin el (fixtures sin
    `button_gate`) se usa la reja mas cercana. La semantica es temporal
    para todos los pares.
    """
    all_gates = world.all_gates | world.gates
    if not all_gates:
        return frozenset()
    distance = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reja_del(button: Cell) -> Cell:
        # Emparejamiento 1:1; sin el (fixtures sin `button_gate`), la
        # reja mas cercana.
        gate = world.button_gate.get(button)
        if gate is not None and gate in all_gates:
            return gate
        return min(all_gates, key=lambda g: (distance(button, g), g))

    abiertas = frozenset(reja_del(b) for b in covered)
    return frozenset(all_gates - abiertas)


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
    # Un boton es una placa de presion: esta cubierto si lo pisa el
    # personaje o si queda una roca encima. Se recalcula CADA paso (no
    # solo los "nuevos") porque la reja es temporal: se cierra al
    # descubrir su boton.
    covered = world.buttons & (world.rocks | {player})
    all_gates = world.all_gates | world.gates
    gates = _gates_actuales(world, covered)
    pressed = covered
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
        and all_gates == world.all_gates
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
        all_gates=frozenset(all_gates),
        exit_unlocked=exit_unlocked,
    )


def _successors(
    world: World,
    allow_spike_activation: bool = True,
    allow_rock_on_diamond: bool = True,
):
    """Todos los (action, new_world) validos desde `world`."""
    for direction in PLAYER_ORDER:
        result = _step(
            world,
            direction,
            allow_spike_activation=allow_spike_activation,
            allow_rock_on_diamond=allow_rock_on_diamond,
        )
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


def _reachable_bare(world: World) -> FrozenSet[Cell]:
    """
    Celdas alcanzables desde el personaje por la topologia "pelada"
    (ignora hoyos, rejas, puertas y rocas, que pueden despejarse), pero
    respetando los muros fijos y los picos YA activados (que viven en
    `walls`). Se usa como seguridad de picos: un movimiento irreversible
    que activa un pico solo se permite si no deja aislada ninguna meta
    pendiente en esta topologia.
    """
    visited: Set[Cell] = {world.player}
    queue = deque([world.player])
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors(world, cell):
            if (
                world.in_bounds(neighbor)
                and neighbor not in world.walls
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(neighbor)
    return frozenset(visited)


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


def _successes_for(
    name: str, goals: FrozenSet[Cell], world: World
) -> List[Callable[[World], bool]]:
    """
    Predicados de exito de un subobjetivo, en orden de preferencia.

    Los diamantes solo se recolecta pisando la celda: una roca puede
    taparlos pero nunca recogerlos, asi que el unico exito posible es
    que el personaje este encima. Para llaves solo vale la DESAPARICION:
    como el personaje lleva una sola, pisar la celda de otra llave no la
    recoge, asi que "pisar" daria un falso positivo. Puertas: tambien la
    desaparicion (abrirlas al pisarlas con la llave).

    Botones: la reja es TEMPORAL (se cierra al soltar el boton), asi que
    solo una ROCA encima deja la reja abierta de forma util (pisar y
    salir no sirve para cruzar la propia reja); el personaje encima
    queda como ultimo recurso.
    """
    if name.startswith("roca"):
        # Pista "empujar una roca a esta celda de piso": exito cuando la
        # roca queda encima. Si el destino fuera hoyo o boton, la pista
        # se habria normalizado como "hoyo"/"boton" y se usan sus
        # predicados (relleno / roca sobre la placa).
        return [lambda world: bool(goals & world.rocks)]
    if name.startswith("diamante"):
        return [lambda world: world.player in goals]
    if name.startswith("llave"):
        return [lambda world: not (goals & world.keys)]
    if name.startswith("hoyo"):
        # Un hoyo esta "resuelto" cuando se rellena (desaparece de la capa).
        return [lambda world: not (goals & world.holes)]
    if name.startswith("puerta"):
        # Una puerta se "resuelve" al abrirla: pisarla con la llave la
        # quita de `doors` (y consume la llave). Vale la desaparicion,
        # igual que las llaves, por si el plan se mueve tras abrirla.
        return [lambda world: not (goals & world.doors)]
    if name.startswith("boton") and _es_temporal(world):
        # Con reja temporal solo una ROCA encima la deja abierta de forma
        # util. El personaje encima queda como ultimo recurso SOLO si la
        # reja no es instrumental: si bloquea una meta pendiente, pisar y
        # salir la cierra, asi que aceptar "el personaje encima" seria un
        # exito falso que cicla el fallback (nivel 11: el boton (3,7) con
        # la reja (4,6) tapando el diamante (4,5)).
        preds: List[Callable[[World], bool]] = [
            lambda world: bool(goals & world.rocks)
        ]
        button = next(iter(goals), None)
        if button is None or not _reja_bloquea(world, button):
            preds.append(lambda world: world.player in goals)
        return preds
    return [lambda world: _reached(world, goals)]


def _path_avoiding(
    world: World,
    source: Cell,
    target: Cell,
    avoid: FrozenSet[Cell] = frozenset(),
) -> Optional[List[Cell]]:
    """
    Camino mas corto de `source` a `target` sobre la topologia "pelada"
    (ignora rocas y bloqueos dinamicos, igual que la heuristica) sin
    atravesar las celdas de `avoid`. Devuelve la lista de celdas desde
    `source` (inclusive) hasta `target`, o None si no hay camino.
    """
    if source == target:
        return [source]
    prev: Dict[Cell, Optional[Cell]] = {source: None}
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors(world, cell):
            if (
                world.in_bounds(neighbor)
                and neighbor not in world.walls
                and neighbor not in avoid
                and neighbor not in prev
            ):
                prev[neighbor] = cell
                if neighbor == target:
                    queue.clear()
                    break
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


def _bare_path(world: World, target: Cell) -> Optional[List[Cell]]:
    """
    Camino mas corto del personaje a `target` sobre la topologia "pelada"
    (ignora rocas y bloqueos dinamicos, igual que la heuristica). Se usa
    para descubrir que hoyos cortan el paso a una meta. Devuelve la lista
    de celdas desde el personaje (inclusive) hasta la meta, o None.
    """
    return _path_avoiding(world, world.player, target)


def _dedup(cells: Iterable[Cell]) -> List[Cell]:
    """Celdas sin repetir, conservando el orden."""
    seen: List[Cell] = []
    for cell in cells:
        if cell not in seen:
            seen.append(cell)
    return seen


def holes_blocking(world: World, target: Cell, back: bool = True) -> List[Cell]:
    """
    Hoyos que hay que rellenar antes de ir a `target` y poder volver
    (sin repetir). Analisis conjunto IDA/VUELTA sobre la topologia
    pelada:

    - IDA: primero se busca un camino que evite hoyos Y picos (lo
      ideal). Si existe, la vuelta es su inverso y no hay nada que
      rellenar. Si no, se admite cruzar picos y se registra `cruce`
      (los picos de esa ida); si ni siquiera hay camino ignorando los
      picos, los hoyos del camino pelado son bloqueadores reales.
    - VUELTA: un pico cruzado queda como muro para siempre, asi que el
      retorno puede quedar cortado. Se prueban dos niveles: sin pisar
      hoyos ni los picos del `cruce` (si existe, el hoyo no importa);
      y solo evitando el `cruce` (picos frescos valen, se cruzan en un
      solo sentido). En el segundo nivel el retorno depende de un hoyo:
      hay que rellenarlo ANTES de cruzar ningun pico. Es el caso del
      nivel 6: la ida al bolsillo norte cruza (10,7)/(12,8) y la
      vuelta pasa por (11,1).

    `back=False` (salida) salta el analisis de vuelta: el plan termina
    en `target` y no hay que regresar.
    """
    if not world.holes:
        return []
    ideal = _path_avoiding(
        world, world.player, target, avoid=world.holes | world.spikes
    )
    if ideal is not None:
        return []
    ida = _path_avoiding(world, world.player, target, avoid=world.holes)
    if ida is None:
        out = _bare_path(world, target)
        found = [c for c in out if c in world.holes] if out else []
        return _dedup(found)
    cruce = frozenset(c for c in ida if c in world.spikes)
    if not back:
        return []
    # `cruce` es un subconjunto de los picos, asi que esta vuelta es mas
    # restrictiva que la inversa de `ideal` (ya descartada arriba).
    if (
        _path_avoiding(world, target, world.player, avoid=world.holes | cruce)
        is not None
    ):
        return []
    libre = _path_avoiding(world, target, world.player, avoid=cruce)
    if libre is None:
        return []
    return _dedup([c for c in libre if c in world.holes])


def _puerta_bloqueante(
    world: World, target: Cell, failed: Set[FrozenSet[Cell]]
) -> Optional[Cell]:
    """
    Puerta con llave disponible que hay que abrir para que algo llegue a
    `target` (uso: un hoyo cuyo relleno exige cruzar una puerta, p. ej.
    nivel 6: la roca hacia (11,1) pasa por la puerta (10,2)). Es
    candidata si alguna ruta pelada (del personaje o de una roca) hacia
    `target` la atraviesa y no fallo antes. Devuelve la mas cercana a
    `target` (la que hay que cruzar antes), o None.
    """
    caminos: List[List[Cell]] = []
    player_path = _path_avoiding(world, world.player, target)
    if player_path:
        caminos.append(player_path)
    for rock in sorted(world.rocks):
        rock_path = _path_avoiding(world, rock, target)
        if rock_path:
            caminos.append(rock_path)
    candidatas = {
        cell for path in caminos for cell in path if cell in world.doors
    }
    candidatas = {d for d in candidatas if frozenset({d}) not in failed}
    if not candidatas:
        return None
    field = wall_distance_field(world, {target})
    return min(candidatas, key=lambda d: field.get(d, float("inf")))


def a_star(
    world: World,
    goals: FrozenSet[Cell],
    max_nodes: int = config.MAX_NODES_ASTAR,
    weight: float = config.A_STAR_WEIGHT,
    success: Optional[Callable[[World], bool]] = None,
    allow_spike_activation: bool = True,
    allow_rock_on_diamond: bool = True,
    rock_goal: Optional[Cell] = None,
    keep_reachable: FrozenSet[Cell] = frozenset(),
) -> Optional[Tuple[List[Action], World]]:
    """
    A* ponderado sobre el espacio de mundos completos. Devuelve
    (actions, final_world) o None si no encuentra solucion dentro del
    presupuesto. Con `weight>1` el sesgo hacia la meta es mas fuerte:
    encuentra plan antes (a cambio de perder optimalidad), util en los
    puzzles Sokoban de los niveles finales. Mueve, empuja rocas, rellena
    hoyos y alcanza la meta. `success` permite definir cuando el
    subobjetivo esta logrado (por defecto, pisar la meta).
    Con `allow_spike_activation=False` no activa picos; con
    `allow_rock_on_diamond=False` no deja rocas sobre diamantes
    (ver `_step`).

    `rock_goal` (una celda meta que hay que CUBRIR con una roca, p. ej.
    colocar una roca sobre un boton) añade a la heuristica la distancia
    de la roca mas cercana a esa celda, guiando el empuje y acelerando la
    busqueda de los puzzles Sokoban. `keep_reachable` lista celdas de
    metas todavia pendientes que deben seguir siendo alcanzables: se
    descartan los sucesores que activan un pico y dejan alguna de esas
    celdas aislada (seguridad generica de picos: no cortar el camino a
    otro objetivo con un movimiento irreversible).
    """
    reached = success or (lambda w: _reached(w, goals))
    if reached(world):
        return [], _collect(world)

    field = wall_distance_field(world, goals)
    if world.player not in field:
        return None
    reverse = (
        wall_distance_field(world, {rock_goal})
        if rock_goal is not None
        else None
    )

    def h_of(state: World) -> float:
        base = field.get(state.player, float("inf"))
        if base == float("inf") or reverse is None:
            return base
        toca = min(
            (reverse[r] for r in state.rocks if r in reverse),
            default=0,
        )
        return base + toca

    open_heap: List[Tuple[float, float, int, World, Tuple[Action, ...]]] = []
    heapq.heappush(open_heap, (weight * h_of(world), 0.0, 0, world, ()))
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

        for action, successor in _successors(
            current,
            allow_spike_activation=allow_spike_activation,
            allow_rock_on_diamond=allow_rock_on_diamond,
        ):
            state_key = _state_key(successor)
            if state_key in visited:
                continue
            if (
                keep_reachable
                and successor.spikes != current.spikes
                and not keep_reachable <= _reachable_bare(successor)
            ):
                continue
            h = h_of(successor)
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
    allow_spike_activation: bool = True,
    allow_rock_on_diamond: bool = True,
    keep_reachable: FrozenSet[Cell] = frozenset(),
) -> Optional[Tuple[List[Action], World]]:
    """
    DFS con backtracking sobre el espacio de mundos. Prioriza las
    acciones que empujan rocas (para desbloquear corredores) y se
    deshace de ramas sin salida con `visited` + tope de profundidad.
    Con `allow_spike_activation=False` no activa picos; con
    `allow_rock_on_diamond=False` no deja rocas sobre diamantes
    (ver `_step`). `keep_reachable` lista metas pendientes que deben
    seguir siendo alcanzables (seguridad generica de picos, igual que en
    `a_star`).
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
        successors = sorted(
            _successors(
                current,
                allow_spike_activation=allow_spike_activation,
                allow_rock_on_diamond=allow_rock_on_diamond,
            ),
            key=lambda s: s[0].kind != "push",
        )
        for action, successor in reversed(successors):
            if _state_key(successor) in visited:
                continue
            if (
                keep_reachable
                and successor.spikes != current.spikes
                and not keep_reachable <= _reachable_bare(successor)
            ):
                continue
            stack.append((successor, plan + (action,), depth + 1))

    return None


# ---------------------------------------------------------------
# Planificador de alto nivel (subobjetivos)
# ---------------------------------------------------------------
def _es_profunda(world: World, cell: Cell) -> bool:
    """
    True si la ida a `cell` exige cruzar un pico: no hay camino que
    evite hoyos Y picos, pero si uno que solo evite hoyos. Tales metas
    quedan "dentro" del bolsillo tras el cruce (los picos actuan en un
    solo sentido), asi que _goals_of las pospone para barrerlas en una
    sola ida. Aplica tambien sin hoyos: un pico sigue siendo un cruce
    irreversible que conviene hacer al final.
    """
    if _path_avoiding(
        world, world.player, cell, avoid=world.holes | world.spikes
    ) is not None:
        return False
    return _path_avoiding(world, world.player, cell, avoid=world.holes) is not None


def _reja_bloquea(world: World, button: Cell) -> bool:
    """
    True si la reja de `button` corta el camino pelado del personaje a
    algun diamante o a la salida pendientes.

    Con rejas temporales solo interesa empujar una roca a los botones
    cuya reja estorba; los botones decorativos no generan subobjetivo
    (evita perseguir rocas hacia placas inutiles). Sin emparejamiento
    conocido se devuelve True (conservador: no se descarta la meta).
    """
    gate = world.button_gate.get(button)
    if gate is None:
        return True
    metas = set(world.diamonds)
    if world.exit is not None:
        metas.add(world.exit)
    for target in metas:
        path = _path_avoiding(
            world, world.player, target, avoid=world.holes | world.spikes
        )
        if path is not None and gate in path:
            return True
    return False


def _goals_of(world: World):
    """
    Secuencia de subobjetivos segun el contenido del nivel. Solo la usa
    el respaldo _plan_fallback (la busqueda global no los necesita):

    1) llaves seguras (no exigen cruzar picos);
    2) pre-fase: los hoyos que bloquean la ida o la vuelta de las
       metas profundas (ver holes_blocking). Van aqui, con la llave
       recien cogida y antes de que ningun A* consuma la llave en otra
       puerta: nivel 6 rellena (11,1) antes de cualquier diamante;
    3) botones (abren las rejas);
    4) diamantes seguros, uno por uno;
    5) llaves profundas y 6) diamantes profundos: el bolsillo que se
       alcanza cruzando picos se barre en una sola ida, tras lo seguro,
       para no gastar los picos de entrada/salida a medias;
    7) la salida, SOLO cuando ya no quedan diamantes.

    Los hoyos que cortan la IDA o la VUELTA a un objetivo se emiten
    PROACTIVAMENTE justo antes de ese objetivo (el relleno de Sokoban
    va antes que lo que depende de el). Cada subobjetivo es individual:
    si uno queda inalcanzable se marca pendiente sin perder la
    planificacion del resto. Los hoyos repetidos entre metas se
    deduplican con `seen_holes`.

    Si `world.hints` trae la secuencia declarada por nivel (level_hints),
    esos subobjetivos van PRIMERO, en su orden: son preferencias del
    usuario y se resuelven uno por A* antes de la composicion
    automatica. `world_from_state` ya revalido sus celdas contra lo
    percibido; aqui solo se evita repetir una meta ya emitida.
    """
    seen_holes: Set[Cell] = set()
    emitted: Set[FrozenSet[Cell]] = set()

    # Pistas por nivel primero (preferencias), en el orden declarado.
    for name, goals, _rock in world.hints:
        if goals and goals not in emitted:
            emitted.add(goals)
            yield name, goals

    def holes_for(cell: Cell):
        """Hoyos que cortan la ida o el retorno a `cell` (sin repetir)."""
        for hole in holes_blocking(world, cell):
            if hole not in seen_holes:
                seen_holes.add(hole)
                goals = frozenset({hole})
                if goals not in emitted:
                    emitted.add(goals)
                    yield f"hoyo {hole}", goals

    def emit(name: str, goals: FrozenSet[Cell]):
        """Un subobjetivo, salvo que su meta ya se haya emitido."""
        if goals and goals not in emitted:
            emitted.add(goals)
            return name, goals
        return None

    llaves = sorted(world.keys)
    diamantes = sorted(world.diamonds)
    llaves_p = [c for c in llaves if _es_profunda(world, c)]
    diam_p = [c for c in diamantes if _es_profunda(world, c)]

    for cell in (c for c in llaves if c not in llaves_p):
        yield from holes_for(cell)
        item = emit(f"llave {cell}", frozenset({cell}))
        if item is not None:
            yield item
    for cell in llaves_p + diam_p:
        yield from holes_for(cell)
    for cell in sorted(world.buttons):
        if cell in world.pressed:
            continue
        # Con reja temporal la reja solo se abre mientras la placa este
        # cubierta: solo vale la pena perseguir botones cuya reja corta
        # el paso a una meta pendiente.
        if _es_temporal(world) and not _reja_bloquea(world, cell):
            continue
        item = emit(f"boton {cell}", frozenset({cell}))
        if item is not None:
            yield item
    for cell in (c for c in diamantes if c not in diam_p):
        yield from holes_for(cell)
        item = emit(f"diamante {cell}", frozenset({cell}))
        if item is not None:
            yield item
    for cell in llaves_p:
        yield from holes_for(cell)
        item = emit(f"llave {cell}", frozenset({cell}))
        if item is not None:
            yield item
    for cell in diam_p:
        yield from holes_for(cell)
        item = emit(f"diamante {cell}", frozenset({cell}))
        if item is not None:
            yield item
    if world.exit is not None and world.exit_unlocked:
        item = emit("salida", frozenset({world.exit}))
        if item is not None:
            yield item


def _hint_rock_goal(world: World, name: str) -> Optional[Cell]:
    """
    Celda destino declarada en una pista de `world.hints` con ese `name`,
    o None si no hay pista (o no lleva `rock_goal`). Se usa para guiar el
    empuje Sokoban con la heuristica rock-aware.
    """
    for hint_name, _goals, rock_goal in world.hints:
        if hint_name == name and rock_goal is not None:
            return rock_goal
    return None


def _reachable_avoiding_holes(world: World, target: Cell) -> bool:
    """True si `target` es alcanzable sin pisar ningun hoyo."""
    visited: Set[Cell] = {world.player}
    queue = deque([world.player])
    while queue:
        cell = queue.popleft()
        if cell == target:
            return True
        for neighbor in _neighbors(world, cell):
            if (
                world.in_bounds(neighbor)
                and neighbor not in world.walls
                and neighbor not in world.holes
                and neighbor not in visited
            ):
                visited.add(neighbor)
                queue.append(neighbor)
    return target in visited


def _is_solvable(world: World, goals: List[FrozenSet[Cell]]) -> bool:
    """
    Heuristica para abortar _plan_fallback cuando el mundo avanzo a un
    estado ya insoluble (evita el bucle de re-cola de hoyos cuya roca
    quedo inalcanzable). Devuelve False si alguna meta quedo sin
    ninguna ruta posible:

    - ninguna celda del goal es alcanzable en el grafo pelado, o
    - todas las rutas al goal pasan por un hoyo y no quedan rocas
      con las que rellenarlo.
    """
    for goal in goals:
        if not goal:
            continue
        field = wall_distance_field(world, goal)
        if not any(cell in field for cell in goal):
            return False
        if world.holes and not world.rocks:
            target = next(iter(goal))
            if not _reachable_avoiding_holes(world, target):
                return False
    return True


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

    Si el World trae `hints` (secuencia declarada por nivel en
    level_hints.py), se salta la fase global y se va directo al respaldo
    dirigido, con los subobjetivos de la pista primero.

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
        # Con pistas por nivel (level_hints), el camino principal es la
        # composicion dirigida: un A* por subobjetivo en el orden
        # declarado, en vez del A* global (que en el nivel 6 se queda sin
        # presupuesto y en el 10 no converge). Si TODAS las pistas ya
        # estan cumplidas (nivel a medias), se sigue con la busqueda
        # global normal para el resto.
        if any(
            not any(s(current) for s in _successes_for(name, goals, current))
            for name, goals, _rock in current.hints
        ):
            return self._plan_fallback(world)
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
        #
        # El A* miope de un subobjetivo puede tomar decisiones
        # irreversibles (activar un pico, dejar una roca sobre un
        # diamante) sin ver las consecuencias para el resto del nivel.
        # Por eso se intenta PRIMERO con destructive=False y solo se
        # recurre a movimientos irreversibles como ultimo recurso.
        actions: List[Action] = []
        # El mundo del planificador arranca tras recolectar lo que el
        # personaje ya pisa (y desbloqueando la salida si no hay gemas).
        current = _collect(world)
        # Metas que ya fracasaron en el mundo actual: no se reintentan
        # mientras el mundo no avance.
        failed: Set[FrozenSet[Cell]] = set()
        # Pistas de nivel ya logradas. Un waypoint de roca es NO monotonico
        # (la roca puede abandonar la celda objetivo en un tramo posterior
        # del Sokoban), asi que una vez cumplido no debe re-encolarse.
        hint_ids = {(name, goals) for name, goals, _rock in current.hints}
        done_hints: Set[FrozenSet[Cell]] = set()
        pending: List[str] = []
        global_retries = 0

        def mark_pending(name: str) -> None:
            if name not in pending:
                pending.append(name)

        def enqueue() -> deque:
            # Metas del mundo actual, SIN las que ya estan logradas, las
            # que ya fracasaron ni las pistas ya cumplidas.
            return deque(
                (name, goals)
                for name, goals in _goals_of(current)
                if not any(s(current) for s in _successes_for(name, goals, current))
                and goals not in failed
                and goals not in done_hints
            )

        def resolve(
            name: str,
            goals: FrozenSet[Cell],
            astar_budget: int,
            solo_seguro: bool = False,
        ):
            # Prueba los predicados de exito en orden de preferencia. Para
            # cada predicado se intenta A* y, si no hay camino, el
            # retroceso, en tres pasadas de seguridad:
            #   1) sin activar picos ni rocas sobre diamantes (seguro);
            #   2) activando picos si hace falta (el pico es la unica
            #      via) pero sin rocas sobre diamantes;
            #   3) todo permitido (ultimo recurso).
            # El bucle EXTERNO es el de predicados: cada uno se agota en
            # los tres modos antes de pasar al siguiente. Con el orden
            # inverso, un predicado barato en modo seguro (p. ej. "el
            # personaje sobre el boton") ganaba al predicado correcto que
            # solo es alcanzable activando un pico ("roca sobre el
            # boton"), dando un plan falso (nivel 11).
            # Con `solo_seguro=True` (hay hoyos abiertos que cortan la
            # ida/vuelta a la meta) se queda en la pasada 1: activar un
            # pico antes de rellenar esos hoyos dejaria el retorno
            # cortado (nivel 6: subir a la fila 13 sin rellenar (11,1)).
            modes = (
                ((False, False),) if solo_seguro else
                (
                    (False, False),
                    (True, False),
                    (True, True),
                )
            )
            success_preds = _successes_for(name, goals, current)
            # Seguridad generica de picos: celdas de las DEMAS metas
            # PENDIENTES que no deben quedar aisladas por activar un pico.
            # Se excluyen las metas ya cumplidas: `_goals_of` emite las
            # pistas aunque ya esten resueltas (p. ej. un hoyo rellenado),
            # y exigir que sigan alcanzables metia una celda ya cumplida
            # en `keep_reachable`, podando el unico camino valido (nivel
            # 10: rellenar (8,2) exige cruzar el pico (8,4) yendo al sur,
            # que "aisla" el hoyo (8,7) ya rellenado).
            keep_reachable = frozenset(
                celda
                for _n, g in _goals_of(current)
                if g != goals
                and not any(s(current) for s in _successes_for(_n, g, current))
                for celda in g
            )
            # Guia rock-aware: una pista de nivel trae su celda destino
            # (una roca al hoyo/boton/celda); si no, el primer predicado
            # de un boton temporal usa el propio boton como destino.
            hint_rock = _hint_rock_goal(current, name)
            for indice, success in enumerate(success_preds):
                rock_goal = hint_rock
                if rock_goal is None and name.startswith("boton") and indice == 0:
                    rock_goal = next(iter(goals), None)
                for spikes, rocks in modes:
                    result = a_star(
                        current,
                        goals,
                        astar_budget,
                        success=success,
                        allow_spike_activation=spikes,
                        allow_rock_on_diamond=rocks,
                        rock_goal=rock_goal,
                        keep_reachable=keep_reachable,
                    )
                    if result is None:
                        result = backtracking_search(
                            current,
                            goals,
                            self.max_nodes_backtrack,
                            self.max_depth,
                            success=success,
                            allow_spike_activation=spikes,
                            allow_rock_on_diamond=rocks,
                            keep_reachable=keep_reachable,
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
                print(f"[trace] PRUNE {name} (fuera del grafo)")
                mark_pending(name)
                failed.add(frozenset(goals))
                continue
            # Hoyos abiertos que cortan la ida o la vuelta a esta meta
            # (macro de Sokoban: hay que rellenarlos ANTES; ver
            # holes_blocking). Si los hay, la meta se resuelve con
            # presupuesto de hoyo y SOLO en modo seguro, y si falla se
            # expande en subobjetivos de relleno.
            target = next(iter(goals), None)
            es_hoyo = name.startswith("hoyo")
            es_puerta = name.startswith("puerta")
            # Recuperacion ANTES de gastar presupuesto: un hoyo sin
            # rellenar cuya ruta (del personaje o de una roca) cruza una
            # puerta con llave (nivel 6: la roca hacia (11,1) pasa por
            # la puerta (10,2)). Abrirla primero despeja el A* de
            # relleno sin quemar los tres modos de resolve. Al abrirla
            # se consume la llave (o la puerta fallida entra en
            # `failed`), asi que no se vuelve a emitir: el bucle termina.
            if (
                es_hoyo
                and target is not None
                and current.has_key
                and current.doors
            ):
                puerta = _puerta_bloqueante(current, target, failed)
                if puerta is not None:
                    queue.appendleft((name, goals))
                    queue.appendleft(
                        (f"puerta {puerta}", frozenset({puerta}))
                    )
                    continue
            blockers: List[Cell] = []
            if target is not None and current.holes and not (es_hoyo or es_puerta):
                blockers = holes_blocking(
                    current, target, back=not name.startswith("salida")
                )
            budget = (
                config.MAX_NODES_HOLE
                if (blockers or es_hoyo)
                else self.max_nodes_astar
            )
            previous = current
            result = resolve(name, goals, budget, solo_seguro=bool(blockers))
            print(
                f"[trace] pop={name} blockers={blockers} budget={budget} "
                f"seguro={bool(blockers)} player={current.player} "
                f"has_key={current.has_key} doors={sorted(current.doors)} "
                f"-> {'OK ' + str(len(result[0])) if result else 'FALLA'}"
            )
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
            # Una pista ya resuelta no se vuelve a perseguir (su roca
            # puede haberse movido a otro tramo).
            if (name, goals) in hint_ids:
                done_hints.add(goals)
            # Reintentar lo fallido solo si el mundo avanzo de verdad
            # (pasos nuevos o cambio de estado). Con rejas temporales un
            # boton puede "resolverse" sin dejar progreso persistente (el
            # personaje lo pisa y lo suelta al ir a la meta siguiente):
            # limpiar `failed` sin progreso real re-encolaria la misma
            # pareja de metas en un ciclo infinito.
            if steps or current != previous:
                failed.clear()
            else:
                failed.add(frozenset(goals))
            queue = enqueue()
            # Reintento global tras progreso (puerta abierta o llave
            # recogida): el espacio de busqueda se ha reducido y el A*
            # global puede caber ahora en el presupuesto. El objetivo
            # incluye la salida si ya esta desbloqueada, para no quedar
            # atrapado en una region tras recolectar todo.
            if (
                global_retries < 2
                and current.diamonds
                and (
                    len(current.doors) < len(world.doors)
                    or len(current.keys) < len(world.keys)
                )
            ):
                global_retries += 1
                exit_cell = current.exit
                if exit_cell is not None and current.exit_unlocked:
                    goals_global = frozenset(
                        current.diamonds | {exit_cell}
                    )
                else:
                    goals_global = frozenset(current.diamonds)

                def _global_done(w: World) -> bool:
                    if w.diamonds:
                        return False
                    if w.exit is None:
                        return True
                    return w.player == w.exit or (
                        not w.exit_unlocked and not w.diamonds
                    )

                result = a_star(
                    current,
                    goals_global,
                    self.max_nodes_astar,
                    success=_global_done,
                )
                if result is not None:
                    steps, current = result
                    actions.extend(steps)
                    pending = []
                    break
            # Si alguna meta pendiente quedo sin ninguna ruta posible,
            # abortar (evita el bucle de re-cola de hoyos cuya roca
            # quedo inalcanzable).
            remaining = [
                g
                for n, g in _goals_of(current)
                if g not in failed
                and not any(s(current) for s in _successes_for(n, g, current))
            ]
            if remaining and not _is_solvable(current, remaining):
                for name_r, goals_r in _goals_of(current):
                    if goals_r not in failed and not any(
                        s(current) for s in _successes_for(name_r, goals_r, current)
                    ):
                        mark_pending(name_r)
                break

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
