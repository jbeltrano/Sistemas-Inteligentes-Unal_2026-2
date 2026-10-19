"""
Tests del planificador (computation/compute.py).

Cubren las dos vías de topología del World:
  - fallback por grilla (World sin `adjacency`): el comportamiento previo.
  - grafo de la percepción (`nodes` + `adjacency`): la topología explícita.
Además, el pruning de metas fuera del grafo y la integración con
`world_from_state` / `recognize`.
"""
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

import button_gate
import level_hints
from computation.compute import (
    Action,
    Plan,
    Planner,
    World,
    _collect,
    _es_profunda,
    _goals_of,
    _state_key,
    _step,
    _successes_for,
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


def test_reja_temporal_sin_roca_no_es_solucionable():
    """
    Con un solo par boton-reja la reja se cierra al soltar el boton: si el
    diamante esta detras y no hay roca para dejarla pisada, el nivel no
    tiene solucion. El plan queda incompleto (el agente no se "cuela" por
    la reja pisando el boton).
    """
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

    assert not plan.complete
    assert plan.final_world.diamonds == frozenset({(1, 4)})


def test_reja_temporal_se_cierra_al_salir_del_boton():
    """
    Modelo temporal 1:1: pisar el boton abre la reja; el movimiento
    siguiente la vuelve a cerrar. No se puede atravesar la propia reja sin
    una roca que mantenga la placa pisada.
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        buttons=frozenset({(1, 1)}),
        gates=frozenset({(1, 2)}),
        walls=_corredor([0, 2]),
    )
    # Pisar el boton abre la reja.
    _paso, sobre_boton = _step(world, "right")
    sobre_boton = _collect(sobre_boton)
    assert sobre_boton.player == (1, 1)
    assert not sobre_boton.gates
    # Colarse a la casilla de la reja es invalido: al salir del boton la
    # reja se cerraria con el personaje dentro.
    assert _step(sobre_boton, "right") is None
    # Alrededor: salir del boton la vuelve a cerrar.
    _paso2, salido = _step(sobre_boton, "left")
    assert _collect(salido).gates == frozenset({(1, 2)})


def test_roca_sobre_boton_abre_reja():
    """
    Un boton es placa de presion: una roca encima la mantiene pisada y, con
    reja temporal, la deja abierta de forma permanente. El diamante solo es
    alcanzable cruzando la reja (la fila 2 esta cerrada al final).
    """
    muros = _corredor([0, 3], cols=6) | {(1, 0), (2, 0), (2, 4), (2, 5)}
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


def test_varios_pares_temporal_derivado_y_no_monotono():
    """
    Con varios pares la semantica tambien es temporal y el emparejamiento
    es 1:1: `_collect` deriva las rejas de la cobertura actual. Con las
    dos rocas encima de sus botones, ambas rejas quedan abiertas; al
    retirar una roca SOLO su reja se cierra (no es monotono).
    """
    world = World(
        rows=3,
        cols=7,
        player=(1, 0),
        rocks=frozenset({(1, 1), (1, 5)}),
        buttons=frozenset({(1, 1), (1, 5)}),
        gates=frozenset({(1, 2), (1, 6)}),
        walls=_corredor([0, 2], cols=7),
    )
    ambas = _collect(world)
    assert ambas.pressed == frozenset({(1, 1), (1, 5)})
    assert not ambas.gates

    una = _collect(replace(world, rocks=frozenset({(1, 1)})))
    assert una.pressed == frozenset({(1, 1)})
    assert una.gates == frozenset({(1, 6)})   # se cierra la del boton descubierto
    assert (1, 2) not in una.gates


def test_varios_pares_temporal_sin_rocas_no_es_solucionable():
    """
    Con varios pares y sin rocas no se puede mantener ninguna reja
    abierta al cruzar: el diamante tras la primera reja es inalcanzable.
    """
    world = World(
        rows=3,
        cols=7,
        player=(1, 0),
        buttons=frozenset({(1, 1), (1, 5)}),
        gates=frozenset({(1, 2), (1, 6)}),
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2], cols=7),
    )
    plan = Planner().plan(world)

    assert not plan.complete
    assert (1, 6) in plan.final_world.gates
    assert plan.final_world.player != (1, 4)


def test_planifica_dos_pares_temporales_con_dos_rocas():
    """
    Dos pares temporales independientes: cada diamante esta detras de su
    reja y solo se abre dejando una roca en el boton correspondiente. El
    personaje no basta (cubre un boton a la vez y al moverse lo suelta).
    """
    muros = _corredor([0, 6], cols=6) | {(r, 0) for r in range(1, 6)}
    muros |= {
        (2, 4), (2, 5),          # paredes tras el boton de arriba
        (3, 2), (3, 3), (3, 4), (3, 5),  # fila 3: solo el pasillo (3, 1)
        (5, 4), (5, 5),          # paredes tras el boton de abajo
    }
    world = World(
        rows=7,
        cols=6,
        player=(1, 1),
        rocks=frozenset({(2, 2), (5, 2)}),
        buttons=frozenset({(2, 3), (5, 3)}),
        gates=frozenset({(1, 4), (4, 4)}),
        diamonds=frozenset({(1, 5), (4, 5)}),
        walls=frozenset(muros),
    )
    plan = Planner().plan(world)

    assert plan.complete
    final = plan.final_world
    assert not final.diamonds
    assert not final.gates
    assert final.pressed == frozenset({(2, 3), (5, 3)})
    simulado = simulate(world, plan.actions)
    assert simulado.pressed == frozenset({(2, 3), (5, 3)})
    assert simulado.player in {(1, 5), (4, 5)}


def test_world_from_state_usa_mapeo_explicito(monkeypatch):
    """
    Con `level`, el mapeo declarado en button_gate.py sobreescribe la
    heuristica por cercania. Aqui el mapeo explicito cruza los pares que
    la cercania emparejaria al reves.
    """
    grid = np.ones((1, 6), dtype=np.uint8)
    # tile_size=1 y origen 0 => celda == (y, x) del centro.
    objetos = [
        DetectedObject("button", 2, 0, 1, 1),  # celda (0, 2)
        DetectedObject("button", 5, 0, 1, 1),  # celda (0, 5)
        DetectedObject("closed_gate", 4, 0, 1, 1),    # celda (0, 4)
        DetectedObject("closed_gate", 1, 0, 1, 1),    # celda (0, 1)
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

    # Sin `level` el emparejamiento es por cercania.
    por_cercania = world_from_state(state)
    assert por_cercania.button_gate == {(0, 2): (0, 1), (0, 5): (0, 4)}

    # Con `level` manda la tabla explicita (pares cruzados).
    explicito = {(0, 2): (0, 4), (0, 5): (0, 1)}
    monkeypatch.setitem(button_gate.BUTTON_GATE, 99, explicito)
    con_nivel = world_from_state(state, level=99)
    assert con_nivel.button_gate == explicito


def test_world_from_state_ignora_mapeo_con_celdas_ausentes(monkeypatch):
    """
    Si el mapeo explicito menciona celdas que no se perciben, esas
    entradas se ignoran y lo demas se completa por cercania (evita
    aplicar un mapeo de otro nivel).
    """
    grid = np.ones((1, 6), dtype=np.uint8)
    objetos = [
        DetectedObject("button", 2, 0, 1, 1),  # celda (0, 2)
        DetectedObject("button", 5, 0, 1, 1),  # celda (0, 5)
        DetectedObject("closed_gate", 4, 0, 1, 1),    # celda (0, 4)
        DetectedObject("closed_gate", 1, 0, 1, 1),    # celda (0, 1)
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

    # (0, 2)->(0, 4) es valido; (0, 0)->(0, 5) menciona un boton ausente.
    monkeypatch.setitem(
        button_gate.BUTTON_GATE, 98, {(0, 2): (0, 4), (0, 0): (0, 5)}
    )
    world = world_from_state(state, level=98)

    assert world.button_gate[(0, 2)] == (0, 4)
    # El boton restante (0, 5) toma la reja sobrante (0, 1).
    assert world.button_gate[(0, 5)] == (0, 1)
    assert (0, 0) not in world.button_gate


def test_fallback_temporal_termina_y_completa():
    """
    Con reja temporal el subobjetivo "boton" puede resolverse sin progreso
    persistente (pisar y soltar).     Forzando el respaldo con presupuesto de
    A* diminuto, el fallback debe terminar (sin re-encolar en ciclo) y, con
    roca disponible, completar el nivel.
    """
    muros = _corredor([0, 3], cols=6) | {(1, 0), (2, 0), (2, 4), (2, 5)}
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
    plan = Planner(max_nodes_astar=1).plan(world)

    assert plan.complete
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


# ---------------------------------------------------------------
# Nivel 6 (fixture manual): huecos, picos de un solo sentido y
# la secuencia llave -> puerta -> hoyo -> barrido del bolsillo norte.
# ---------------------------------------------------------------
def _nivel6() -> World:
    """
    Nivel 6 capturado de assets/levels/level6.png, como World manual
    (no depende de la percepcion). La grilla derivada de `walls`
    reproduce exactamente el grafo de la percepcion (adyacencia y
    nodos), asi que basta con los muros y las capas de objetos.
    """
    ceros = {(r, c) for r in (0, 1, 2, 14) for c in range(10)}
    bordes = {(r, c) for r in range(15) for c in (0, 9)}
    internos = {
        (5, 2), (5, 4), (5, 5), (5, 7),
        (8, 3), (8, 4), (8, 6),
        (9, 3), (9, 6),
        (10, 1), (10, 3), (10, 4), (10, 5), (10, 6), (10, 8),
        (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7),
    }
    walls = frozenset(ceros | bordes | internos)
    piso = frozenset(
        (r, c)
        for r in range(15)
        for c in range(10)
        if (r, c) not in walls
    )
    return World(
        rows=15,
        cols=10,
        player=(3, 5),
        walls=walls,
        spikes=frozenset({(10, 7), (12, 8), (13, 1)}),
        rocks=frozenset({(4, 2), (4, 7)}),
        diamonds=frozenset(
            {
                (4, 4), (4, 5), (9, 2), (9, 5), (9, 8),
                (11, 3), (11, 4), (11, 5), (13, 3), (13, 5),
            }
        ),
        holes=frozenset({(5, 1), (5, 8), (11, 1)}),
        doors=frozenset({(8, 5), (10, 2)}),
        keys=frozenset({(4, 8), (13, 4)}),
        exit=(9, 4),
        exit_unlocked=False,
        nodes=piso,
    )


@pytest.fixture(scope="module")
def plan_nivel6():
    """Plan del nivel 6 calculado una sola vez (cuesta ~80 s)."""
    world = _nivel6()
    return world, Planner().plan(world)


def test_nivel6_plan_completo(plan_nivel6):
    """
    El fallback del nivel 6 compone: llave (4,8) -> puerta (10,2) ->
    hoyo (11,1) -> diamantes -> salida. Antes de corregir el orden,
    la llave (13,4) sellaba el bolsillo norte y el plan quedaba
    incompleto (32 pasos y jugador atrapado en (13,3)).
    """
    world, plan = plan_nivel6

    assert plan.complete, plan.pending
    assert not plan.pending
    final = simulate(world, plan.actions)
    assert final.player == (9, 4)
    assert not final.diamonds
    assert not final.holes <= world.holes  # al menos (11,1) rellenado
    assert (11, 1) not in final.holes
    assert final == plan.final_world


def test_nivel6_roca_no_tapa_diamante(plan_nivel6):
    """
    En ningun momento del plan queda una roca sobre un diamante sin
    recolectar (taparlo lo volveria intratable: las rocas no
    recolectan diamantes).
    """
    world, plan = plan_nivel6
    actual = world
    for action in plan.actions:
        resultado = _step(actual, action.direction)
        assert resultado is not None, f"paso invalido: {action}"
        _accion, actual = resultado
        actual = _collect(actual)
        assert not (actual.rocks & actual.diamonds), (
            f"roca sobre diamante tras {action}: "
            f"{sorted(actual.rocks & actual.diamonds)}"
        )
    assert not actual.diamonds


def test_fallback_no_bucle_hoyo():
    """
    El nivel 6 emite UN solo subobjetivo de hoyo, (11,1): ni (5,1) ni
    (5,8) (se desperdiciarian las rocas y (5,8) mata el nivel), y va
    tras la llave (4,8) y antes de cualquier diamante y de las metas
    profundas del bolsillo norte. Un orden distinto re-encolaria hoyos
    o dejaria metas intratables.
    """
    nombres = [name for name, _ in _goals_of(_nivel6())]

    assert [n for n in nombres if n.startswith("hoyo")] == ["hoyo (11, 1)"]
    assert nombres.index("llave (4, 8)") < nombres.index("hoyo (11, 1)")
    assert nombres.index("hoyo (11, 1)") < nombres.index("diamante (4, 4)")
    assert nombres.index("diamante (4, 4)") < nombres.index("diamante (13, 3)")
    assert nombres.index("llave (13, 4)") < nombres.index("diamante (13, 3)")
    assert "salida" not in nombres  # bloqueada hasta juntar diamantes


def test_destructive_false_no_activa_picos(plan_nivel6):
    """
    Mientras (11,1) siga abierto no se activa NI UN pico (los picos
    activados pasan a `walls`): cruzarlos antes rellenar el hueco
    sellaria el bolsillo norte con diamantes dentro.
    """
    world, plan = plan_nivel6
    assert (11, 1) in world.holes
    actual = world
    for action in plan.actions:
        resultado = _step(actual, action.direction)
        assert resultado is not None, f"paso invalido: {action}"
        _accion, actual = resultado
        actual = _collect(actual)
        if (11, 1) in actual.holes:
            assert actual.spikes == world.spikes, (
                f"pico activado con (11,1) abierto tras {action}"
            )
    assert (11, 1) not in actual.holes  # el hueco se relleno al final


# ---------------------------------------------------------------
# Boton temporal instrumental: la roca precede al personaje.
# ---------------------------------------------------------------
def test_boton_temporal_no_acepta_personaje_sobre_boton():
    """
    Con reja temporal que tapa un diamante pendiente, "el personaje
    encima del boton" NO es exito: al salir la reja se cierra. El
    predicado de roca debe ir primero y el de personaje no ofrecerse
    (era el falso exito que ciclaba el fallback del nivel 11).
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        buttons=frozenset({(1, 1)}),
        all_gates=frozenset({(1, 2)}),
        gates=frozenset({(1, 2)}),
        button_gate={(1, 1): (1, 2)},
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2]),
    )
    preds = _successes_for("boton (1, 1)", frozenset({(1, 1)}), world)
    assert len(preds) == 1  # solo la roca encima vale como exito

    sobre_el_boton = _collect(replace(world, player=(1, 1)))
    assert not any(pred(sobre_el_boton) for pred in preds)

    con_roca = _collect(replace(world, rocks=frozenset({(1, 1)})))
    assert preds[0](con_roca)


def test_boton_temporal_no_instrumental_acepta_personaje():
    """
    Si la reja NO corta el paso a ninguna meta pendiente, pisar el boton
    sigue siendo un exito valido (no se pierde el ultimo recurso).
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        buttons=frozenset({(1, 1)}),
        all_gates=frozenset({(1, 2)}),
        gates=frozenset({(1, 2)}),
        button_gate={(1, 1): (1, 2)},
        walls=_corredor([0, 2]),
    )
    preds = _successes_for("boton (1, 1)", frozenset({(1, 1)}), world)
    assert len(preds) == 2  # roca y, de ultimo, el personaje

    sobre_el_boton = _collect(replace(world, player=(1, 1)))
    assert any(pred(sobre_el_boton) for pred in preds)


def test_es_profunda_sin_hoyos():
    """
    Un pico sigue siendo un cruce irreversible aunque el mundo no tenga
    hoyos: la meta que solo se alcanza pisandolo queda "profunda" (antes
    se devolvia False sin hoyos y se cruzaba el pico demasiado pronto).
    """
    walls = frozenset(
        (r, c) for r in range(3) for c in range(3) if c != 0
    )  # solo el pasillo de la columna 0
    world = World(
        rows=3,
        cols=3,
        player=(0, 0),
        spikes=frozenset({(1, 0)}),
        walls=walls,
    )
    assert not world.holes
    assert _es_profunda(world, (2, 0)) is True


# ---------------------------------------------------------------
# Guarda del agente: no ejecutar planes incompletos.
# ---------------------------------------------------------------
def test_execute_no_envia_teclas_si_el_plan_es_incompleto():
    """
    Un plan incompleto no se ejecuta: no se manda ninguna flecha y se
    devuelve el mismo plan (con las metas pendientes intactas).
    """
    import agent

    class TecladoFalso:
        def __init__(self):
            self.llamadas = 0

        def press(self, *args, **kwargs):  # pragma: no cover - no debe llamarse
            self.llamadas += 1

    teclado = TecladoFalso()
    agente = agent.Agent(capturer=object(), keyboard=teclado, level=11)
    plan = Plan(
        actions=[Action("move", "right")],
        complete=False,
        pending=["diamante (4, 5)"],
    )

    resultado = agente.execute(plan)

    assert resultado is plan
    assert teclado.llamadas == 0


class _TecladoEjecutable:
    """Teclado falso que registra las flechas enviadas (y no falla)."""

    def __init__(self):
        self.movimientos = []

    def focus(self):
        pass

    def move(self, direction, pause=None):
        self.movimientos.append(direction)


class _CapturerInexistente:
    """Capturer falso: cualquier frame es un RuntimeError (personaje no visto)."""

    def locate_canvas(self):
        pass

    def capture(self):
        raise RuntimeError("sin personaje en el frame de prueba")


def _plan_un_paso(world):
    """Plan de un solo movimiento a partir de `world` (sin final a validar)."""
    return Plan(
        actions=[Action("move", "right")],
        complete=True,
        initial_world=world,
    )


def test_verify_every_cero_no_toma_pantallazos(monkeypatch):
    """
    Con verify_every=0 NO se toman pantallazos intermedios: tampoco los
    forzados de botones (verify_every <= 0 silencia toda verificacion,
    no solo la periodica). Las flechas si se mandan.
    """
    import agent

    settle_calls = []

    def _settle_spy(self, prev_cell, expected_cell):
        settle_calls.append((prev_cell, expected_cell))
        return None

    monkeypatch.setattr(agent.Agent, "_settle", _settle_spy)

    world = World(rows=1, cols=3, player=(0, 0))
    teclado = _TecladoEjecutable()
    agente = agent.Agent(
        capturer=_CapturerInexistente(),
        keyboard=teclado,
        verify_every=0,
    )
    plan = Plan(
        actions=[Action("move", "right"), Action("move", "right")],
        complete=True,
        initial_world=world,
    )

    resultado = agente.execute(plan)

    assert resultado is plan
    assert teclado.movimientos == ["right", "right"]
    assert settle_calls == []  # ni periódicos ni forzados de botones


def test_verify_every_positivo_si_verifica_botones(monkeypatch):
    """
    Con verify_every >= 1 la verificacion FORZADA sigue activa: pisar un
    boton cuya reja no esta confirmada dispara un pantallazo (_settle).
    """
    import agent

    settle_calls = []

    def _settle_spy(self, prev_cell, expected_cell):
        settle_calls.append((prev_cell, expected_cell))
        return None

    monkeypatch.setattr(agent.Agent, "_settle", _settle_spy)

    world = World(
        rows=1,
        cols=3,
        player=(0, 0),
        buttons=frozenset({(0, 1)}),
        button_gate={(0, 1): (0, 2)},
        all_gates=frozenset({(0, 2)}),
    )
    teclado = _TecladoEjecutable()
    agente = agent.Agent(
        capturer=_CapturerInexistente(),
        keyboard=teclado,
        verify_every=1,
    )

    resultado = agente.execute(_plan_un_paso(world))

    assert resultado.complete
    assert teclado.movimientos == ["right"]
    assert settle_calls == [((0, 0), (0, 1))]  # el boton pisado se verifico


# ---------------------------------------------------------------
# Pistas de subobjetivos por nivel (level_hints.py)
# ---------------------------------------------------------------
def test_world_from_state_inyecta_hints(monkeypatch):
    """
    Con `level`, world_from_state traduce LEVEL_SEQUENCE a World.hints,
    normaliza ("roca", origen, destino) segun el destino y descarta las
    entradas cuyas celdas no existen en lo percibido (nivel a medias).
    """
    grid = np.ones((3, 3), dtype=np.uint8)
    # tile_size=1 y origen 0 => celda == (y, x) del centro.
    objetos = [
        DetectedObject("rock", 0, 1, 1, 1),     # celda (1, 0)
        DetectedObject("hole", 2, 1, 1, 1),     # celda (1, 2)
        DetectedObject("diamond", 0, 2, 1, 1),  # celda (2, 0)
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
    monkeypatch.setitem(
        level_hints.LEVEL_SEQUENCE,
        77,
        [
            ("roca", (1, 0), (1, 2)),   # destino hoyo -> "hoyo (1, 2)"
            ("diamante", (2, 0)),
            ("llave", (9, 9)),           # ausente -> se ignora
            ("roca", (1, 0), (9, 9)),    # destino fuera del tablero -> ignorado
            ("hoyo", (9, 9)),            # ausente -> se ignora
        ],
    )
    world = world_from_state(state, level=77)

    assert world.hints == [
        ("hoyo (1, 2)", frozenset({(1, 2)}), (1, 2)),
        ("diamante (2, 0)", frozenset({(2, 0)}), None),
    ]


def test_hint_roca_a_boton_normaliza_y_guiar(monkeypatch):
    """Un destino boton se normaliza como 'boton ...' con su rock_goal."""
    grid = np.ones((3, 3), dtype=np.uint8)
    objetos = [
        DetectedObject("rock", 0, 1, 1, 1),      # celda (1, 0)
        DetectedObject("button", 2, 1, 1, 1),    # celda (1, 2)
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
    monkeypatch.setitem(
        level_hints.LEVEL_SEQUENCE, 80, [("roca", (1, 0), (1, 2))]
    )
    world = world_from_state(state, level=80)

    assert world.hints == [
        ("boton (1, 2)", frozenset({(1, 2)}), (1, 2)),
    ]


def test_hint_roca_descartada_si_destino_no_existe(monkeypatch):
    """Un destino que no es hoyo, boton ni piso se descarta."""
    grid = np.ones((3, 3), dtype=np.uint8)
    grid[1][2] = 0  # muro
    state = GameState(
        grid=grid,
        graph_nodes=set(),
        adjacency={},
        player_cell=(0, 0),
        objects=[],
        objects_by_cell={},
        origin_x=0,
        origin_y=0,
        tile_size=1,
    )
    monkeypatch.setitem(
        level_hints.LEVEL_SEQUENCE, 79, [("roca", (1, 0), (1, 2))]
    )
    world = world_from_state(state, level=79)

    assert world.hints == []


def test_hint_roca_en_hoyo_es_objetivo_principal():
    """
    La pista va primero en _goals_of (sin duplicar el hoyo que emite la
    composicion automatica) y el plan completo la resuelve como su primer
    A*: rellenar el hoyo empujando la roca, y luego los diamantes.
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        rocks=frozenset({(1, 1)}),
        holes=frozenset({(1, 2)}),
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2]),
        hints=[("hoyo (1, 2)", frozenset({(1, 2)}), (1, 2))],
    )
    nombres = [name for name, _ in _goals_of(world)]
    assert nombres[0] == "hoyo (1, 2)"        # la pista va primero
    assert nombres.count("hoyo (1, 2)") == 1  # y no se repite

    plan = Planner().plan(world)

    assert plan.complete
    assert (1, 2) not in plan.final_world.holes
    assert not plan.final_world.diamonds
    assert simulate(world, plan.actions).player == (1, 4)


def test_hint_roca_waypoints_encadenados():
    """
    Dos waypoints de la misma roca en orden: primero dejarla en (0,3) y
    luego meterla al hoyo (0,5). Cada tramo es un A* chico.
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 1)}),
        holes=frozenset({(0, 5)}),
        hints=[
            ("roca (0, 3)", frozenset({(0, 3)}), (0, 3)),
            ("hoyo (0, 5)", frozenset({(0, 5)}), (0, 5)),
        ],
    )
    plan = Planner().plan(world)

    assert plan.complete
    final = simulate(world, plan.actions)
    assert not final.holes
    assert (0, 5) not in final.rocks  # la roca se consumio en el hoyo


def test_hint_ya_cumplida_no_se_reintenta():
    """
    Nivel a medias: la pista "roca al hoyo" ya esta cumplida (el hoyo no
    esta en `holes`). Su predicado de exito da True, asi que el
    planificador no rehace la pista y completa el resto normalmente.
    """
    world = World(
        rows=3,
        cols=5,
        player=(1, 0),
        diamonds=frozenset({(1, 4)}),
        walls=_corredor([0, 2]),
        hints=[("hoyo (1, 2)", frozenset({(1, 2)}), (1, 2))],
    )
    assert any(
        pred(world)
        for pred in _successes_for("hoyo (1, 2)", frozenset({(1, 2)}), world)
    )

    plan = Planner().plan(world)

    assert plan.complete
    assert simulate(world, plan.actions).player == (1, 4)


def test_sin_level_no_hay_hints():
    """Sin `level` (o sin entrada) no se inyectan pistas."""
    grid = np.ones((3, 3), dtype=np.uint8)
    state = GameState(
        grid=grid,
        graph_nodes=set(),
        adjacency={},
        player_cell=(0, 0),
        objects=[],
        objects_by_cell={},
        origin_x=0,
        origin_y=0,
        tile_size=1,
    )

    assert world_from_state(state).hints == []
    assert world_from_state(state, level=999).hints == []


def test_hint_cumplida_no_bloquea_keep_reachable():
    """
    Regresion (nivel 10): una pista YA CUMPLIDA no debe entrar en
    `keep_reachable`.

    `_goals_of` emite las pistas aunque ya esten resueltas (p. ej. un
    hoyo rellenado). Si `resolve` las contaba como metas pendientes, su
    celda quedaba en `keep_reachable` y podaba el unico camino valido en
    cuanto exigia activar un pico.

    Mundo minimo: corredor (0,0) personaje -> (0,1) pico -> (0,2)
    diamante. La pista `diamante (0,2)` (pendiente) fuerza el fallback;
    la pista `hoyo (0,0)` ya esta cumplida ((0,0) no es un hoyo). Cruzar
    el pico activa (0,1) y deja (0,0) aislado: con el bug, la celda
    (0,0) de la pista cumplida bloquea el paso y el plan queda
    incompleto.
    """
    world = World(
        rows=1,
        cols=3,
        player=(0, 0),
        spikes=frozenset({(0, 1)}),
        diamonds=frozenset({(0, 2)}),
        hints=[
            ("hoyo (0,0)", frozenset({(0, 0)}), None),      # cumplida
            ("diamante (0,2)", frozenset({(0, 2)}), None),  # pendiente
        ],
    )
    # La pista "hoyo (0,0)" esta satisfecha: (0,0) no esta en `holes`.
    assert not world.holes

    plan = Planner().plan(world)

    assert plan.complete, plan.pending
    assert not simulate(world, plan.actions).diamonds


# ---------------------------------------------------------------
# Llave en la mano del personaje (variante player_key del sprite)
# ---------------------------------------------------------------
# Si el tester coge la llave a mano antes de correr el agente, el sprite
# del personaje cambia a player_key.png y la llave pintada dentro de el
# matchea TAMBIEN el template de la llave suelta. Sin player_has_key,
# world_from_state contaba esa deteccion como llave del nivel y has_key
# quedaba en False: el planificador creia que el personaje no tenia
# llave y el plan salia incompleto.


def _frame_suelo(color_bgr):
    """Frame 724x1080 (tablero 15x10, tile=72) relleno de suelo macizo."""
    return np.full((1080, 724, 3), color_bgr, dtype=np.uint8)


def _pegar(frame, template, cell):
    """Pega `template` (numpy BGR) centrado en `cell` (fila, col)."""
    tile = 72
    origin_x = (frame.shape[1] - tile * 10) // 2
    origin_y = (frame.shape[0] - tile * 15) // 2
    i, j = cell
    x = origin_x + j * tile + (tile - template.shape[1]) // 2
    y = origin_y + i * tile + (tile - template.shape[0]) // 2
    h, w = template.shape[:2]
    frame[y : y + h, x : x + w] = template


def _color_suelo_de_template(path):
    """Color de fondo (esquina) del template, para igualar el del frame."""
    import cv2

    image = cv2.imread(str(path))
    return tuple(int(v) for v in image[0, 0])


def test_player_key_marca_player_has_key():
    """Con el sprite de la variante con llave, player_has_key queda True."""
    cv2 = pytest.importorskip("cv2")
    import config
    from perception.recognition import recognize

    templates = config.TEMPLATES_DIR
    frame = _frame_suelo(_color_suelo_de_template(templates / "player_key.png"))
    _pegar(frame, cv2.imread(str(templates / "player_key.png")), (5, 2))

    state = recognize(frame)

    assert state.player_cell == (5, 2)
    assert state.player_has_key is True


def test_player_sin_llave_no_marca_player_has_key():
    """Con el sprite normal, player_has_key queda False."""
    cv2 = pytest.importorskip("cv2")
    import config
    from perception.recognition import recognize

    templates = config.TEMPLATES_DIR
    frame = _frame_suelo(_color_suelo_de_template(templates / "player.png"))
    _pegar(frame, cv2.imread(str(templates / "player.png")), (5, 2))

    state = recognize(frame)

    assert state.player_cell == (5, 2)
    assert state.player_has_key is False


def test_world_has_key_desde_sprite_y_descarta_llave_fantasma():
    """
    El sprite con llave manda: has_key es True aunque el conteo no lo
    sugiera, y la deteccion `key` montada sobre el personaje se descarta
    de `world.keys` (era la llave de la mano, no una suelta).
    """
    grid = np.ones((3, 3), dtype=np.uint8)
    jugador = DetectedObject("player", 0, 0, 2, 2)   # caja (0,0)-(2,2)
    llave_en_mano = DetectedObject("key", 0, 0, 1, 1)  # centro (0,0): dentro
    state = GameState(
        grid=grid,
        graph_nodes=set(),
        adjacency={},
        player_cell=(0, 0),
        objects=[jugador, llave_en_mano],
        objects_by_cell={(0, 0): [jugador, llave_en_mano]},
        origin_x=0,
        origin_y=0,
        tile_size=1,
        player_has_key=True,
    )

    world = world_from_state(state)

    assert world.has_key is True
    assert (0, 0) not in world.keys


def test_world_has_key_fallback_por_conteo():
    """
    Sin la senal visual (fixtures de test, sprite indeterminado) sigue
    valiendo el conteo: mas puertas cerradas que llaves sueltas => el
    personaje lleva la que falta.
    """
    grid = np.ones((3, 3), dtype=np.uint8)
    state = GameState(
        grid=grid,
        graph_nodes=set(),
        adjacency={},
        player_cell=(0, 0),
        objects=[DetectedObject("door_key", 2, 2, 1, 1)],
        objects_by_cell={(2, 2): [DetectedObject("door_key", 2, 2, 1, 1)]},
        origin_x=0,
        origin_y=0,
        tile_size=1,
    )

    world = world_from_state(state)

    assert world.has_key is True
    assert world.doors == frozenset({(2, 2)})
