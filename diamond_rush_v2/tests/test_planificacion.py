"""
Tests del planificador (computation/compute.py).

Cubren las dos vías de topología del World:
  - fallback por grilla (World sin `adjacency`): el comportamiento previo.
  - grafo de la percepción (`nodes` + `adjacency`): la topología explícita.
Además, el pruning de metas fuera del grafo y la integración con
`world_from_state` / `recognize`.
"""
from collections import defaultdict

import numpy as np
import pytest

from computation.compute import (
    Planner,
    World,
    _collect,
    _goals_of,
    a_star,
    simulate,
    world_from_state,
)
from perception.state import DetectedObject, GameState


def _corredor(walls_filas, cols=5):
    """Paredes de las filas indicadas (para forzar un corredor en la fila 1)."""
    return frozenset((r, c) for r in walls_filas for c in range(cols))


def _bidireccional(edges):
    """Arma una lista de adyacencia simétrica a partir de pares de celdas."""
    adjacency = defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    return {cell: sorted(neighbors) for cell, neighbors in adjacency.items()}


def test_plan_completo_sin_grafo():
    """Vía fallback (sin adjacency): recolecta el diamante y llega a la salida."""
    world = World(
        rows=5,
        cols=5,
        player=(0, 0),
        diamonds=frozenset({(0, 2)}),
        exit=(4, 4),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert plan.final_world.player == (4, 4)
    assert not plan.final_world.diamonds
    assert simulate(world, plan.actions).player == (4, 4)


def test_empuja_roca():
    """Con una roca en el corredor, el plan debe incluir al menos un empuje."""
    world = World(
        rows=5,
        cols=5,
        player=(2, 0),
        rocks=frozenset({(2, 1)}),
        exit=(2, 4),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert any(action.kind == "push" for action in plan.actions)
    assert plan.final_world.player == (2, 4)


def test_grafo_respeta_topologia():
    """
    Con el grafo de la percepción, el camino debe seguir SUS aristas: un
    atajo recto que la grilla permitiría pero el grafo no conecta queda
    descartado.
    """
    # Pasillo en "C": obliga a rodear en vez de ir recto por la fila 0.
    edges = [
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (2, 1)),
        ((2, 1), (2, 2)),
        ((2, 2), (1, 2)),
        ((1, 2), (0, 2)),
    ]
    adjacency = _bidireccional(edges)

    world = World(
        rows=3,
        cols=3,
        player=(0, 0),
        exit=(0, 2),
        nodes=frozenset(adjacency),
        adjacency=adjacency,
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert plan.final_world.player == (0, 2)
    # Rodeo de 6 pasos, no los 2 rectos por (0,1) (que no está en el grafo).
    assert plan.length == 6
    assert (0, 1) not in adjacency


def test_wall_distance_field_usa_grafo():
    """La heurística solo alcanza las celdas conectadas en el grafo."""
    from computation.compute import wall_distance_field

    adjacency = _bidireccional([((0, 0), (1, 0)), ((1, 0), (1, 1))])
    world = World(
        rows=3,
        cols=3,
        player=(0, 0),
        exit=(1, 1),
        exit_unlocked=True,
        nodes=frozenset(adjacency),
        adjacency=adjacency,
    )
    field = wall_distance_field(world, frozenset({(1, 1)}))

    assert field[(1, 1)] == 0
    assert field[(0, 0)] == 2
    assert (0, 1) not in field  # no es nodo del grafo
    assert (2, 2) not in field


def test_pruning_meta_fuera_de_grafo():
    """Un diamante que no es nodo se marca pendiente sin buscarlo."""
    world = World(
        rows=3,
        cols=3,
        player=(0, 0),
        diamonds=frozenset({(0, 2)}),
        nodes=frozenset({(0, 0)}),
        adjacency={(0, 0): []},
    )
    plan = Planner().plan(world)

    assert not plan.complete
    assert plan.pending == ["diamante (0, 2)"]
    assert plan.actions == []


def test_world_from_state_copia_grafo():
    """world_from_state traslada nodes/adjacency del GameState al World."""
    grid = np.ones((3, 3), dtype=np.uint8)
    adjacency = _bidireccional([((0, 0), (1, 0))])
    state = GameState(
        grid=grid,
        graph_nodes=set(adjacency),
        adjacency=adjacency,
        player_cell=(0, 0),
        objects=[],
        objects_by_cell={},
        origin_x=0,
        origin_y=0,
        tile_size=1,
    )
    world = world_from_state(state)

    assert world.nodes == frozenset(adjacency)
    assert world.adjacency == adjacency


def test_mundo_desde_estado_roles():
    """Los nombres de objeto de v2 se traducen a las capas del World."""
    grid = np.ones((3, 3), dtype=np.uint8)
    # tile_size=1 y origen 0 => celda == (y, x) del centro.
    objetos = [
        DetectedObject("diamond", 2, 0, 1, 1),   # celda (0, 2)
        DetectedObject("rock", 1, 1, 1, 1),      # celda (1, 1)
        DetectedObject("spike", 0, 2, 1, 1),     # celda (2, 0), peligro
        DetectedObject("scape_enable", 2, 2, 1, 1),  # celda (2, 2), salida
    ]
    state = GameState(
        grid=grid,
        graph_nodes=set(),
        adjacency={},
        player_cell=(0, 0),
        objects=objetos,
        objects_by_cell={},
        origin_x=0,
        origin_y=0,
        tile_size=1,
    )
    world = world_from_state(state)

    assert world.diamonds == frozenset({(0, 2)})
    assert world.rocks == frozenset({(1, 1)})
    assert (2, 0) in world.walls       # spike -> peligro -> muro
    assert world.exit == (2, 2)


def test_recognize_y_plan_consistente():
    """
    Integración: reconocer un nivel real, planificar y rejugar el plan
    sobre el modelo. `simulate` valida cada paso, así que si no lanza,
    el plan es internamente consistente.
    """
    cv2 = pytest.importorskip("cv2")
    import config
    from perception.recognition import recognize

    path = config.LEVELS_DIR / "level5.png"
    if not path.exists():
        pytest.skip(f"no está {path}")

    frame = cv2.imread(str(path))
    try:
        state = recognize(frame)
    except RuntimeError as error:
        pytest.skip(f"reconocimiento falló sobre {path.name}: {error}")

    world = world_from_state(state)
    plan = Planner().plan(world)

    # No importa si queda incompleto por templates/peligros: lo que se
    # exige es que el plan sea reproducible sobre el modelo.
    if plan.initial_world is not None:
        simulate(plan.initial_world, plan.actions)
    assert plan.final_world is not None


# ---------------------------------------------------------------
# Mecanicas del juego (hoyo+roca, boton/reja, llave/puerta, salida)
# ---------------------------------------------------------------
def test_hoyo_se_rellena_con_roca():
    """
    Corredor bloqueado por una roca delante de un hoyo: la unica forma
    de pasar es empujar la roca DENTRO del hoyo (rellenandolo). Sin esta
    mecánica el nivel era irresoluble.
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        rocks=frozenset({(1, 1)}),
        holes=frozenset({(1, 2)}),
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2]),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert any(action.kind == "push" for action in plan.actions)
    assert plan.final_world.player == (1, 4)
    assert not plan.final_world.diamonds
    assert (1, 2) not in plan.final_world.holes  # hoyo rellenado
    assert simulate(world, plan.actions).player == (1, 4)


def test_roca_sobre_diamante_lo_recolecta():
    """Empujar una roca hacia una casilla con diamante lo recolecta."""
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        rocks=frozenset({(1, 1)}),
        diamonds=frozenset({(1, 2)}),
        walls=_corredor([0, 2]),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert not plan.final_world.diamonds
    final = simulate(world, plan.actions)
    assert not final.diamonds          # el diamante quedo recolectado
    assert (1, 1) not in final.rocks   # la roca se movio de su sitio inicial


def test_boton_abre_reja():
    """Un boton en el camino abre la reja que bloquea el diamante."""
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        buttons=frozenset({(1, 1)}),
        gates=frozenset({(1, 2)}),
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2]),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert plan.final_world.player == (1, 4)
    assert not plan.final_world.gates  # reja abierta
    assert simulate(world, plan.actions).player == (1, 4)


def test_roca_sobre_boton_abre_reja():
    """Un boton es placa de presion: una roca encima tambien lo activa."""
    muros = _corredor([0, 3], cols=6) | {(1, 0), (2, 0)}
    world = World(
        rows=4,
        cols=6,
        player=(1, 1),
        rocks=frozenset({(2, 2)}),
        buttons=frozenset({(2, 3)}),
        gates=frozenset({(1, 4)}),
        diamonds=frozenset({(1, 5)}),
        walls=muros,
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert not plan.final_world.gates
    assert (2, 3) in plan.final_world.pressed
    assert simulate(world, plan.actions).player == (1, 5)


def test_llave_abre_puerta():
    """Recoger la llave habilita la door_key que bloquea el diamante."""
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        keys=frozenset({(1, 1)}),
        doors=frozenset({(1, 2)}),
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2]),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert plan.final_world.player == (1, 4)
    # Abrir la puerta consume la llave: al final ya no se lleva ninguna.
    assert not plan.final_world.has_key
    assert (1, 2) not in plan.final_world.doors  # puerta abierta
    assert simulate(world, plan.actions).player == (1, 4)


def test_no_lleva_dos_llaves_a_la_vez():
    """
    Capacidad 1: con dos llaves y ninguna puerta, el personaje recoge
    una y no puede recoger la segunda (la del piso queda intacta).
    """
    world = World(
        rows=1,
        cols=5,
        player=(0, 0),
        keys=frozenset({(0, 1), (0, 2)}),
    )
    plan = Planner().plan(world)

    final = plan.final_world
    assert final.has_key                      # se lleva una
    assert final.keys == frozenset({(0, 2)})  # la otra quedo en el piso
    # No hay forma de resolver la segunda llave: queda pendiente.
    assert not plan.complete
    assert "llave (0, 2)" in plan.pending


def test_dos_puertas_requieren_dos_llaves():
    """
    Dos puertas y dos llaves: hay que recoger una llave, abrir su puerta
    (consume), recoger la otra y abrir la segunda. El plan queda completo.
    """
    world = World(
        rows=1,
        cols=7,
        player=(0, 0),
        keys=frozenset({(0, 1), (0, 3)}),
        doors=frozenset({(0, 2), (0, 4)}),
        diamonds=frozenset({(0, 6)}),
    )
    plan = Planner().plan(world)

    assert plan.complete
    final = plan.final_world
    assert final.player == (0, 6)
    assert not final.doors     # ambas puertas abiertas
    assert not final.keys      # ambas llaves consumidas
    assert not final.has_key
    assert simulate(world, plan.actions).player == (0, 6)


def test_salida_no_es_meta_hasta_reunir_diamantes():
    """La salida solo entra como meta con 0 diamantes."""
    world = World(
        rows=1,
        cols=5,
        player=(0, 0),
        diamonds=frozenset({(0, 2)}),
        exit=(0, 4),
    )
    nombres = [name for name, _ in _goals_of(world)]
    assert "salida" not in nombres

    world_sin_gemas = _collect(World(rows=1, cols=5, player=(0, 0), exit=(0, 4)))
    nombres = [name for name, _ in _goals_of(world_sin_gemas)]
    assert "salida" in nombres

    plan = Planner().plan(world)
    assert plan.complete
    assert plan.final_world.exit_unlocked
    assert plan.final_world.player == (0, 4)


def test_a_star_devuelve_camino_optimo():
    """Regresion del bug de g: en terreno abierto A* debe dar la distancia real."""
    world = World(rows=5, cols=5, player=(0, 0), exit=(4, 4), exit_unlocked=True)
    result = a_star(world, frozenset({(4, 4)}))

    assert result is not None
    actions, _final = result
    assert len(actions) == 8  # Manhattan exacta (4+4)
