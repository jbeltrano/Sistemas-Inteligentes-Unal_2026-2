"""
Planificador de empujes (Sokoban) para los subobjetivos de roca.

El A* de `computation.compute` busca en el espacio de celdas: cada paso
del personaje es un nodo, y para rellenar un hoyo o dejar una roca sobre
un boton explora miles de caminatas por casillas hasta dar con la
secuencia de empujes (el nivel 6 gasta ~15 s en rellenar (11,1)). Este
modulo planifica a nivel de EMPUJES, que es como se piensa un Sokoban:

- Estado: posicion de las rocas + region ALCANZABLE del personaje. El
  caminar es implicito: en vez de simular cada casilla, un BFS de
  alcanzabilidad dice a que celdas puede llegar el personaje sin mover
  rocas.
- Accion: un empuje factible. Solo cuentan los empujes; la caminata
  hasta colocarse detras de la roca se reconstruye al final.
- Busqueda: A* con heuristica admisible. Un empuje mueve una roca a lo
  sumo una casilla, asi que la distancia "de rejilla, ignorando rocas"
  de la roca mas cercana a la meta subestima los empujes que faltan.

El espacio colapsa de ~200k nodos de celda a unos pocos miles de empujes.
Se usa SOLO para subobjetivos cuyo exito es "poner una roca en la celda
X" (hoyos, botones, waypoints de roca); el A* de celdas queda como
respaldo cuando este modulo devuelve None.

Alcance CONSERVADOR (seguridad): se aplica en modo seguro, igual que la
primera pasada del fallback:

- Los picos (`world.spikes`) se tratan como muros: nunca se activan (y
  en la realidad tampoco se pueden pisar, por eso no se cruzan).
- Rejas y botones temporales se tratan como muros: este modulo no simula
  su apertura temporal (romper el invariante de rejas podria dar un plan
  invalido). Los niveles que lo exijan caen al A* de celdas.
- Las puertas abiertas (`has_key`) se pueden pisar y se cruzan sin
  consumir la llave; con la llave ya usada no se pisan.
- No se empuja ninguna roca sobre un diamante.

Este modulo importa SOLO de `computation.compute` (World, Action y la
topologia) y de `config`; no conoce la percepcion.
"""
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import config
from computation.compute import Action, Cell, World, _world_with

# Orden estable de vecinos (independiente de la direccion del empuje).
_NEIGHBOR_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Direccion -> nombre, para reconstruir la caminata y el empuje.
_DELTA_TO_NAME = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}


def _static_floor(world: World, include_exit: bool = False) -> FrozenSet[Cell]:
    """
    Celdas por las que caminar en modo seguro: muros, picos (no se
    cruzan), rejas temporales (no se simula su apertura), hoyos (bloquean
    al personaje) y la salida aun bloqueada son intransitables; una puerta
    solo si se lleva llave. Los BOTONES si son piso: el personaje y la roca
    pueden estar encima (de hecho dejar una roca sobre la placa es el
    objetivo de algunos subobjetivos); la reja que abre sigue siendo muro
    porque este modulo no modela su apertura temporal.

    Con `include_exit=True` la celda de salida cuenta como piso: es el caso
    del modo "el personaje alcanza la salida" (nivel 20), donde la salida
    bloqueada por rocas es justamente la meta.
    """
    blocked = set(world.walls) | set(world.spikes) | set(world.holes)
    blocked |= set(world.all_gates) | set(world.gates)
    if world.exit is not None and not world.exit_unlocked and not include_exit:
        blocked.add(world.exit)
    if not world.has_key:
        blocked |= set(world.doors)
    if world.nodes:
        return frozenset(world.nodes - blocked)
    return frozenset(
        (r, c)
        for r in range(world.rows)
        for c in range(world.cols)
        if (r, c) not in blocked
    )


def _compute_reach(
    floor: FrozenSet[Cell], rocks: FrozenSet[Cell], start: Cell
) -> FrozenSet[Cell]:
    """
    Region alcanzable del personaje desde `start` sobre `floor`, con las
    rocas como obstaculos (no se puede ocupar ni atravesar una roca).
    """
    if start not in floor:
        return frozenset()
    seen: Set[Cell] = {start}
    queue = deque([start])
    rocks_set = set(rocks)
    while queue:
        cell = queue.popleft()
        for dr, dc in _NEIGHBOR_DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in floor and nxt not in rocks_set and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return frozenset(seen)


def _rock_dest_ok(world: World, floor: FrozenSet[Cell], dest: Cell) -> bool:
    """
    Una roca puede terminar en `dest`: una celda de `floor` libre, o un
    hoyo abierto (lo rellena). Nunca sobre un diamante (lo taparia) ni
    sobre una salida aun bloqueada (el modo "alcanzar la salida" la mete
    en `floor` para el personaje, pero una roca encima taparia la meta).
    """
    if dest in world.diamonds:
        return False
    if dest in world.holes:
        return True
    if dest == world.exit and not world.exit_unlocked:
        return False
    return dest in floor


def _push_successors(
    world: World,
    floor: FrozenSet[Cell],
    rocks: FrozenSet[Cell],
    reach: FrozenSet[Cell],
) -> List[Tuple[Cell, Cell, Cell, FrozenSet[Cell]]]:
    """
    Empujes factibles. Devuelve (celda_roca, celda_destino,
    celda_del_personaje_al_empujar, nuevas_rocas).

    Un empuje: el personaje se para en `player_side` (alcanzable), empuja
    la roca de `rock` a `dest` (= rock + direccion) y queda en `rock`.
    """
    out: List[Tuple[Cell, Cell, Cell, FrozenSet[Cell]]] = []
    rocks_set = set(rocks)
    for rock in sorted(rocks):
        for dr, dc in _NEIGHBOR_DELTAS:
            dest = (rock[0] + dr, rock[1] + dc)
            player_side = (rock[0] - dr, rock[1] - dc)
            if player_side not in reach or dest in rocks_set:
                continue
            if not _rock_dest_ok(world, floor, dest):
                continue
            if dest in world.holes:
                nuevas = frozenset(rocks_set - {rock})
            else:
                nuevas = frozenset((rocks_set - {rock}) | {dest})
            out.append((rock, dest, player_side, nuevas))
    return out


def _bare_field(world: World) -> Dict[Cell, int]:
    """
    Campo de distancias "pelado" desde hoyos y picos del nivel, sobre la
    topologia de piso sin objetos. Lo usa la heuristica como DESCUENTO:
    si entre la roca y la meta hay un hoyo (que este modulo trata como
    piso para el empuje, porque la roca lo rellena), el A* de celdas lo
    veria como un rodeo y sobreestimaria; el descuento mantiene la
    heuristica admisible.
    """
    free = set(world.nodes) if world.nodes else {
        (r, c)
        for r in range(world.rows)
        for c in range(world.cols)
        if (r, c) not in world.walls
    }
    free |= world.holes | world.spikes
    dist: Dict[Cell, int] = {}
    queue: deque = deque()
    for src in list(world.holes) + list(world.spikes):
        if src not in dist:
            dist[src] = 0
            queue.append(src)
    while queue:
        cell = queue.popleft()
        for dr, dc in _NEIGHBOR_DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in free and nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _walk_path(
    floor: FrozenSet[Cell],
    rocks: FrozenSet[Cell],
    source: Cell,
    target: Cell,
) -> Optional[List[Cell]]:
    """
    Camino mas corto de `source` a `target` sobre `floor` esquivando
    `rocks` (BFS). Devuelve la lista desde `source` (inclusive) hasta
    `target`, o None.
    """
    if source == target:
        return [source]
    prev: Dict[Cell, Optional[Cell]] = {source: None}
    queue = deque([source])
    rocks_set = set(rocks)
    while queue:
        cell = queue.popleft()
        for dr, dc in _NEIGHBOR_DELTAS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt not in floor or nxt in rocks_set or nxt in prev:
                continue
            prev[nxt] = cell
            if nxt == target:
                queue.clear()
                break
            queue.append(nxt)
    if target not in prev:
        return None
    path: List[Cell] = []
    cell: Optional[Cell] = target
    while cell is not None:
        path.append(cell)
        cell = prev[cell]
    path.reverse()
    return path


def _cells_to_actions(path: List[Cell]) -> List[Action]:
    """Convierte un camino de celdas en acciones `move`."""
    actions: List[Action] = []
    for a, b in zip(path, path[1:]):
        delta = (b[0] - a[0], b[1] - a[1])
        name = _DELTA_TO_NAME.get(delta)
        if name is None:
            return []
        actions.append(Action("move", name))
    return actions


@dataclass(frozen=True)
class _Node:
    """Estado de la busqueda: rocas + region alcanzable y piso vigentes."""

    rocks: FrozenSet[Cell]
    reach: FrozenSet[Cell]
    floor: FrozenSet[Cell]


@dataclass
class _Edge:
    """Como se llego a un nodo: nodo previo + el empuje realizado."""

    parent: int
    rock: Cell
    dest: Cell
    player_side: Cell


def push_a_star(
    world: World,
    rock_target: Cell,
    max_nodes: int = config.MAX_NODES_PUSH,
    weight: float = 1.0,
    goal_player_reaches: bool = False,
) -> Optional[Tuple[List[Action], World]]:
    """
    Planifica por empujes hasta que una roca quede sobre `rock_target`
    (un hoyo, un boton o una celda de piso). Devuelve (actions,
    final_world) con acciones en el mismo formato que `compute.a_star`
    (pasos `move` de la caminata + `push` del empuje), o None si no hay
    solucion dentro del presupuesto o el nivel exige algo que este modulo
    no modela (rejas/botones temporales instrumentales).

    Con `goal_player_reaches=True` la meta cambia: en vez de mover una roca
    a `rock_target`, basta con que el PERSONAJE pueda alcanzar esa celda
    (tipico de una salida bloqueada por rocas, nivel 20). El plan termina
    con la caminata del personaje hasta `rock_target`.

    `final_world` es un `World` con las rocas ya movidas (y los hoyos
    rellenados), para encadenar el resto de la cadena de subobjetivos.
    """
    if not goal_player_reaches and rock_target in world.rocks:
        return [], world

    floor0 = _static_floor(world, include_exit=goal_player_reaches)
    if world.player not in floor0 and world.player not in world.rocks:
        return None

    bare = _bare_field(world)

    target_is_hole = rock_target in world.holes

    def goal(node: _Node) -> bool:
        # Meta "el personaje alcanza `rock_target`": basta con que este en
        # la zona alcanzable (la caminata final la agrega `_reconstruct`).
        if goal_player_reaches:
            return rock_target in node.reach
        # Un hoyo se resuelve cuando se RELLENA: la roca que lo cubre se
        # consume (desaparece de `rocks`), asi que se detecta porque el
        # hoyo entro a `floor` (ver `_push_successors`). Para boton/piso,
        # el exito es una roca encima.
        if target_is_hole:
            return rock_target in node.floor
        return rock_target in node.rocks

    def h(node: _Node) -> float:
        # Meta cumplida (hoyo relleno o roca encima): distancia 0.
        if goal_player_reaches:
            if rock_target in node.reach:
                return 0.0
            # Cota inferior: distancia del personaje a la meta en el piso
            # estatico menos lo que ya avanzo (heuristica admisible).
            return float(bare.get(rock_target, 0))
        if target_is_hole:
            if rock_target in node.floor:
                return 0.0
        elif rock_target in node.rocks:
            return 0.0
        rocks = node.rocks
        best = min((_manhattan(r, rock_target) for r in rocks), default=-1)
        if best < 0:
            return float("inf")
        return max(0.0, best - bare.get(rock_target, 0))

    start = _Node(
        rocks=frozenset(world.rocks),
        reach=_compute_reach(floor0, world.rocks, world.player),
        floor=floor0,
    )
    if goal(start):
        return [], _world_from_node(world, start)

    counter = 1
    open_heap: List[Tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (weight * h(start), 0.0, 0, 0))
    visited: Set[Tuple[FrozenSet[Cell], FrozenSet[Cell]]] = set()
    states: List[_Node] = [start]
    edges: List[Optional[_Edge]] = [None]
    best_g: Dict[Tuple[FrozenSet[Cell], FrozenSet[Cell]], float] = {}
    nodes_count = 0

    while open_heap and nodes_count < max_nodes:
        _f, g, _c, idx = heapq.heappop(open_heap)
        node = states[idx]
        key = (node.rocks, node.reach)
        if key in visited:
            continue
        visited.add(key)
        nodes_count += 1

        if goal(node):
            return _reconstruct(
                world, states, edges, idx, goal_player_reaches, rock_target
            )

        ng = g + 1
        for rock, dest, player_side, nuevas_rocas in _push_successors(
            world, node.floor, node.rocks, node.reach
        ):
            fills = dest in world.holes
            new_floor = node.floor | {dest} if fills else node.floor
            nueva_reach = _compute_reach(new_floor, nuevas_rocas, rock)
            child = _Node(rocks=nuevas_rocas, reach=nueva_reach, floor=new_floor)
            child_key = (child.rocks, child.reach)
            if child_key in visited or best_g.get(child_key, float("inf")) <= ng:
                continue
            best_g[child_key] = ng
            child_idx = len(states)
            states.append(child)
            edges.append(_Edge(idx, rock, dest, player_side))
            counter += 1
            heapq.heappush(
                open_heap, (ng + weight * h(child), ng, counter, child_idx)
            )

    return None


def _reconstruct(
    world: World,
    states: List[_Node],
    edges: List[Optional[_Edge]],
    idx: int,
    goal_player_reaches: bool = False,
    final_target: Optional[Cell] = None,
) -> Tuple[List[Action], World]:
    """
    Arma el plan: por cada empuje (de atras hacia adelante) reconstruye
    la caminata del personaje hasta `player_side` y luego el `push`.
    Con `goal_player_reaches`, termina con la caminata del personaje
    hasta `final_target` (la celda meta, p. ej. la salida).
    """
    chain: List[_Edge] = []
    while edges[idx] is not None:
        edge = edges[idx]  # type: ignore[assignment]
        chain.append(edge)
        idx = edge.parent
    chain.reverse()

    actions: List[Action] = []
    player = world.player
    rocks = frozenset(world.rocks)
    floor = _static_floor(world, include_exit=goal_player_reaches)
    for edge in chain:
        path = _walk_path(floor, rocks, player, edge.player_side)
        if path is None:
            return [], world
        actions.extend(_cells_to_actions(path))
        delta = (edge.dest[0] - edge.rock[0], edge.dest[1] - edge.rock[1])
        actions.append(Action("push", _DELTA_TO_NAME[delta]))
        player = edge.rock
        if edge.dest in world.holes:
            rocks = frozenset(rocks - {edge.rock})
            floor = floor | {edge.dest}
        else:
            rocks = frozenset((rocks - {edge.rock}) | {edge.dest})
    # La caminata final a la meta, si el objetivo es que el personaje
    # llegue (salida). El piso incluye los hoyos ya rellenados.
    if goal_player_reaches and final_target is not None and player != final_target:
        final_floor = floor | set(rocks) | {player}
        path = _walk_path(final_floor, rocks, player, final_target)
        if path is None:
            return [], world
        actions.extend(_cells_to_actions(path))
        player = final_target
    # El personaje termina sobre la celda de la roca del ultimo empuje
    # (`player` ya quedo ahi). Se refleja en `final_world` para que la
    # cadena de subobjetivos parta de la posicion correcta.
    return actions, _world_with(
        world, rocks=rocks, holes=_holes_left(world, floor), player=player
    )


def _holes_left(world: World, floor: FrozenSet[Cell]) -> FrozenSet[Cell]:
    """Hoyos que siguen abiertos: los rellenados entraron a `floor`."""
    return frozenset(h for h in world.holes if h not in floor)


def _world_from_node(world: World, node: _Node) -> World:
    """World con las rocas/hoyos del nodo (para encadenar subobjetivos)."""
    return _world_with(
        world, rocks=node.rocks, holes=_holes_left(world, node.floor)
    )
