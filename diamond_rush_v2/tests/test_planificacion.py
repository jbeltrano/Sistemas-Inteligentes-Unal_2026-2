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
    _state_key,
    _step,
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
# Picos dinamicos (una sola cruzada permanente por celda)
# ---------------------------------------------------------------
def test_pico_se_activa_al_salir():
    """
    Un pico retraido (`spikes`) se activa permanentemente cuando el
    personaje SALE de la celda: pasa a ser muro y ya no se puede volver
    a cruzar. Entrar a la celda no la activa. La clave de deduplicacion
    distingue el mundo con el pico activado.
    """
    # Salir de un pico lo convierte en muro.
    w = World(rows=3, cols=3, player=(1, 1), spikes=frozenset({(1, 1)}))
    result = _step(w, "left")
    assert result is not None
    _action, salido = result
    assert salido.player == (1, 0)
    assert (1, 1) in salido.walls
    assert (1, 1) not in salido.spikes
    assert _step(salido, "right") is None  # el pico activado ya bloquea
    assert _state_key(w) != _state_key(salido)

    # Entrar a un pico NO lo activa; recien al salir de vuelta.
    w2 = World(rows=3, cols=3, player=(1, 0), spikes=frozenset({(1, 1)}))
    _a, entrado = _step(w2, "right")
    assert entrado.player == (1, 1)
    assert (1, 1) not in entrado.walls and (1, 1) in entrado.spikes
    _b, salio = _step(entrado, "left")
    assert (1, 1) in salio.walls
    assert (1, 1) not in salio.spikes

    # Empujar desde un pico tambien lo activa (el jugador sale de la celda).
    w3 = World(
        rows=3,
        cols=5,
        player=(1, 1),
        spikes=frozenset({(1, 1)}),
        rocks=frozenset({(1, 2)}),
    )
    _c, empujado = _step(w3, "right")
    assert _c is not None
    assert (1, 1) in empujado.walls


def test_nivel5_picos_sin_recrucex():
    """
    Integracion nivel 5: los picos se activan al cruzarlos, asi que un
    plan valido no puede visitar dos veces la misma celda de pico. El
    planificador debe resolverlo completo sin recrucex.
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

    assert plan.complete
    assert not plan.final_world.diamonds
    assert plan.final_world.player == plan.final_world.exit

    picos = {state.cell_of(o.center) for o in state.objects_named("no_spike")}
    assert picos, "el nivel 5 debe tener picos retraidos detectados"
    secuencia = [world.player]
    actual = world
    for action in plan.actions:
        result = _step(actual, action.direction)
        assert result is not None
        actual = _collect(result[1])
        secuencia.append(actual.player)
    assert actual == plan.final_world
    for pico in sorted(picos):
        assert secuencia.count(pico) <= 1, (
            f"el pico {pico} se visitó {secuencia.count(pico)} veces"
        )


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


def test_roca_tapa_diamante_y_el_jugador_lo_recoge():
    """
    Empujar una roca hacia un diamante solo lo TAPA: las rocas no
    recolectan diamantes. El plan debe seguir completo empujando la
    roca de nuevo y pisando la celda para que el jugador lo recoja.
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        rocks=frozenset({(1, 1)}),
        diamonds=frozenset({(1, 2)}),
        walls=_corredor([0, 2]),
    )

    # Empuje directo sobre el diamante: queda tapado, no recolectado.
    pushed = _step(world, "right")
    assert pushed is not None
    _action, tapado = pushed
    assert tapado.diamonds == frozenset({(1, 2)})  # el diamante sigue ahi
    assert tapado.rocks == frozenset({(1, 2)})     # la roca lo tapa

    plan = Planner().plan(world)

    assert plan.complete
    assert not plan.final_world.diamonds
    final = simulate(world, plan.actions)
    assert not final.diamonds          # el jugador lo recolecto al pisar
    assert final.player == (1, 2)
    assert (1, 1) not in final.rocks   # la roca se movio de su sitio inicial


def test_step_no_deja_roca_atrapada_sobre_diamante():
    """Tapa permanente de un diamante: prohibida. Tapa temporal: permitida."""
    def muros(piso):
        return frozenset(
            (r, c) for r in range(5) for c in range(5) if (r, c) not in piso
        )

    # Muro mas alla del diamante: la roca quedaria atrapada encima.
    # (solo la columna del diamante es piso; (4,3) es muro)
    world = World(
        rows=5,
        cols=5,
        player=(1, 3),
        rocks=frozenset({(2, 3)}),
        diamonds=frozenset({(3, 3)}),
        walls=muros({(1, 3), (2, 3), (3, 3)}),
    )
    assert _step(world, "down") is None

    # Piso mas alla: tapa temporal (la roca puede salir despues).
    abierto = World(
        rows=5,
        cols=5,
        player=(1, 3),
        rocks=frozenset({(2, 3)}),
        diamonds=frozenset({(3, 3)}),
    )
    result = _step(abierto, "down")
    assert result is not None
    _action, tapado = result
    assert tapado.rocks == frozenset({(3, 3)})    # roca sobre el diamante
    assert tapado.diamonds == frozenset({(3, 3)})  # el diamante sigue ahi

    # Hoyo mas alla (piso, no muro): la roca sale rellenandolo.
    con_hoyo = World(
        rows=5,
        cols=5,
        player=(1, 3),
        rocks=frozenset({(2, 3)}),
        diamonds=frozenset({(3, 3)}),
        holes=frozenset({(4, 3)}),
        walls=muros({(1, 3), (2, 3), (3, 3), (4, 3)}),
    )
    assert _step(con_hoyo, "down") is not None


def test_no_destruye_diamante_sin_salida():
    """
    Regresion del nivel 4: sin la mirada un paso, el A* miope de un
    diamante empujaba la roca del corredor hasta el ultimo diamante
    rodeado de muros; el plan quedaba incompleto con ese diamante
    destruido. Con la regla debe resolver por la ruta del hoyo.
    """
    libs = {(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)}
    world = World(
        rows=6,
        cols=5,
        player=(1, 1),
        rocks=frozenset({(1, 3), (2, 1)}),
        diamonds=frozenset({(2, 3), (3, 3)}),
        holes=frozenset({(3, 2)}),
        walls=frozenset(
            (r, c) for r in range(6) for c in range(5) if (r, c) not in libs
        ),
    )
    plan = Planner().plan(world)

    assert plan.complete
    assert not plan.final_world.diamonds
    final = simulate(world, plan.actions)
    assert not final.diamonds


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


def test_no_persigue_llaves_sueltas():
    """
    Sin diamantes ni salida que requieran la llave, el planificador no
    persigue llaves sueltas (la busqueda global solo mete diamantes; las
    llaves son instrumentales). La capacidad 1 de _collect sigue
    existiendo: recoge una y al pisar otra con el inventario lleno la
    deja en el piso.
    """
    world = World(
        rows=1,
        cols=5,
        player=(0, 0),
        keys=frozenset({(0, 1), (0, 2)}),
    )
    plan = Planner().plan(world)

    assert plan.complete            # nada que hacer: no se persiguen
    assert plan.length == 0
    final = plan.final_world
    assert final.keys == frozenset({(0, 1), (0, 2)})
    assert not final.has_key

    # Capacidad 1 directo en _collect: una sola llave en el inventario.
    cargada = _collect(World(rows=1, cols=5, player=(0, 1), keys=frozenset({(0, 1), (0, 2)})))
    assert cargada.has_key
    assert cargada.keys == frozenset({(0, 2)})
    # Con el inventario lleno, pisar la segunda llave no la recoge.
    _paso, movida = _step(cargada, "right")
    recogida = _collect(movida)
    assert recogida.has_key
    assert recogida.keys == frozenset({(0, 2)})  # la del piso queda intacta


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
