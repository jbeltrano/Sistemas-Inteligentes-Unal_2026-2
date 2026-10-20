"""
Tests del planificador de empujes (computation/push_planner.py).

Cubren la planificacion a nivel de Sokoban (acciones = empujes, caminar
implicito), su integracion con la cadena de subobjetivos del Planner y
los casos en que NO debe aplicarse (seguridad: picos, alcanzabilidad).
"""
import time

import pytest

from computation.compute import Planner, World, simulate
from computation.push_planner import push_a_star


def _corredor(filas, cols):
    return frozenset((r, c) for r in filas for c in range(cols))


def test_empuja_roca_a_hoyo_en_corredor():
    """Un empuje directo rellena el hoyo y la roca se consume."""
    world = World(
        rows=1,
        cols=8,
        player=(0, 0),
        rocks=frozenset({(0, 1)}),
        holes=frozenset({(0, 7)}),
    )
    result = push_a_star(world, (0, 7))

    assert result is not None
    actions, final = result
    assert len(actions) == 6  # 6 empujes hacia la derecha
    assert (0, 7) not in final.holes  # hoyo rellenado
    assert final.rocks == frozenset()  # la roca se consumio
    # El plan rejugado sobre el mundo real coincide.
    replayed = simulate(world, actions)
    assert (0, 7) not in replayed.holes


def test_empuje_desde_lejos_exige_caminata():
    """
    Si el personaje no esta pegado a la roca, el plan incluye primero los
    `move` de la caminata y luego el empuje.
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 4)}),
        holes=frozenset({(0, 5)}),
    )
    result = push_a_star(world, (0, 5))

    assert result is not None
    actions, _final = result
    kinds = [a.kind for a in actions]
    assert "push" in kinds
    # Los movimientos van antes del empuje que mete la roca.
    assert kinds.index("push") == kinds.count("move")
    simulate(world, actions)


def test_roca_a_boton_no_se_consume():
    """
    Una roca dejada sobre un boton (celda de piso) NO se consume: queda
    encima de la placa.
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 1)}),
        buttons=frozenset({(0, 3)}),
        all_gates=frozenset({(0, 5)}),
        gates=frozenset({(0, 5)}),
        button_gate={(0, 3): (0, 5)},
    )
    result = push_a_star(world, (0, 3))

    assert result is not None
    _actions, final = result
    assert (0, 3) in final.rocks


def test_hoyo_inalcanzable_devuelve_none():
    """
    Si entre la roca y el hoyo hay un muro infranqueable, no hay plan.
    """
    world = World(
        rows=3,
        cols=6,
        player=(1, 0),
        rocks=frozenset({(1, 1)}),
        holes=frozenset({(1, 5)}),
        walls=frozenset({(1, 3)}),  # muro en medio del corredor
    )
    assert push_a_star(world, (1, 5)) is None


def test_no_cruza_picos():
    """
    En modo seguro los picos son muros: un hoyo detras de un pico no se
    puede rellenar (no se activa ningun pico).
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 1)}),
        spikes=frozenset({(0, 3)}),
        holes=frozenset({(0, 5)}),
    )
    assert push_a_star(world, (0, 5)) is None


def test_no_empuja_roca_sobre_diamante():
    """
    Una celda de piso con diamante no es destino valido de una roca (la
    taparia y el diamante solo lo recolecta el personaje).
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 1)}),
        diamonds=frozenset({(0, 4)}),
    )
    assert push_a_star(world, (0, 4)) is None


def test_dos_rocas_encadenan_hoyo_y_boton():
    """
    El estado incluye TODAS las rocas: el planificador puede usar una roca
    para el hoyo y otra para el boton sin confundirlas.
    """
    world = World(
        rows=3,
        cols=8,
        player=(0, 0),
        rocks=frozenset({(0, 1), (2, 1)}),
        holes=frozenset({(0, 4)}),
        buttons=frozenset({(2, 5)}),
        all_gates=frozenset({(1, 5)}),
        gates=frozenset({(1, 5)}),
        button_gate={(2, 5): (1, 5)},
        walls=frozenset({(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)}),
    )
    # Rellenar el hoyo con la roca de (0,1).
    result = push_a_star(world, (0, 4))
    assert result is not None
    _actions, final = result
    assert (0, 4) not in final.holes
    assert (2, 1) in final.rocks  # la otra roca quedo intacta


def test_push_planner_resuelve_hoyo_del_nivel6():
    """
    El fixture del nivel 6 tiene hoyos laterales alcanzables: el
    planificador de empujes los rellena en milisegundos (antes, un A* de
    celdas por cada uno). El hoyo (11,1), en cambio, exige cruzar el pico
    (10,7), asi que en modo seguro devuelve None.
    """
    import test_planificacion as t
    from dataclasses import replace

    from computation.compute import _collect

    world = t._nivel6()
    current = replace(
        world,
        keys=world.keys - {(4, 8)},
        doors=world.doors - {(10, 2)},
        player=(10, 2),
        has_key=False,
    )
    current = _collect(current)

    t0 = time.time()
    result = push_a_star(current, (5, 1))
    dt = time.time() - t0
    assert result is not None
    assert dt < 2.0  # muy por debajo del A* de celdas
    assert (5, 1) not in result[1].holes

    # El hoyo del bolsillo norte exige cruzar un pico: no aplica.
    assert push_a_star(current, (11, 1)) is None


def test_planner_usa_push_en_waypoint_encadenado():
    """
    Integracion: dos waypoints de roca (piso -> hoyo) los resuelve la
    cadena del Planner, con el push_planner por waypoint. El plan final
    deja el hoyo rellenado y la roca consumida.
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
    assert (0, 5) not in final.rocks


def test_salida_ya_alcanzable_no_necesita_empuje():
    """
    Con `goal_player_reaches`, si la salida ya esta en la zona alcanzable
    el plan es vacio (no se mueve ninguna roca).
    """
    world = World(rows=1, cols=4, player=(0, 0), exit=(0, 3))
    result = push_a_star(world, (0, 3), goal_player_reaches=True)
    assert result is not None
    assert result[0] == []


def test_salida_bloqueada_por_roca_apartada_a_nicho():
    """
    Salida tapada: el personaje empuja la roca a un nicho lateral y luego
    alcanza la salida. El plan es valido al rejugarlo con `simulate`.
    """
    walls = frozenset(
        {(0, 0), (0, 1), (2, 0), (2, 1), (2, 3), (2, 4),
         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)}
    )
    world = World(
        rows=4,
        cols=5,
        player=(1, 4),
        rocks=frozenset({(1, 2)}),
        exit=(1, 0),
        walls=walls,
    )
    result = push_a_star(world, (1, 0), goal_player_reaches=True)

    assert result is not None
    actions, final = result
    assert any(a.kind == "push" for a in actions)
    assert final.player == (1, 0)
    # La roca queda en el nicho (2,2), no sobre la salida.
    assert (1, 0) not in final.rocks
    replayed = simulate(world, actions)
    assert replayed.player == (1, 0)


def test_no_empuja_roca_sobre_salida_bloqueada():
    """
    Aunque el modo salida meta la salida en el piso caminable, una roca NO
    puede quedar encima de la salida bloqueada (taparia la meta).
    """
    walls = frozenset(
        {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
         (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)}
    )
    # Salida en el bolsillo (1,1); unica entrada lateral (1,2), tapada por
    # una roca. Empujarla hacia la salida la taparia: no hay solucion.
    world = World(
        rows=4,
        cols=5,
        player=(1, 3),
        rocks=frozenset({(1, 2)}),
        exit=(1, 1),
        walls=walls,
    )
    assert push_a_star(world, (1, 1), goal_player_reaches=True) is None


def test_planner_resuelve_salida_con_roca():
    """
    Integracion: el Planner resuelve una salida tapada por una roca; la
    fase de salida de `_solve` delega en el planificador de empujes.
    """
    walls = frozenset(
        {(0, 0), (0, 1), (2, 0), (2, 1), (2, 3), (2, 4),
         (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)}
    )
    world = World(
        rows=4,
        cols=5,
        player=(1, 4),
        rocks=frozenset({(1, 2)}),
        exit=(1, 0),
        walls=walls,
    )
    plan = Planner().plan(world)

    assert plan.complete
    final = simulate(world, plan.actions)
    assert final.player == (1, 0)


def test_pico_puente_permite_caminata_pero_no_por_defecto():
    """
    El unico camino al lado del empuje cruza un pico (0,1). En modo seguro
    el pico es muro y no hay plan; con `allow_bridge_spikes` el personaje
    lo cruza (queda activado) y rellena el hoyo.
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 3)}),
        spikes=frozenset({(0, 1)}),
        holes=frozenset({(0, 5)}),
    )
    # Seguro: el pico bloquea el paso al lado izquierdo de la roca.
    assert push_a_star(world, (0, 5)) is None
    # Con picos puente: cruza el pico y empuja la roca al hoyo.
    result = push_a_star(world, (0, 5), allow_bridge_spikes=True)
    assert result is not None
    actions, final = result
    assert (0, 5) not in final.holes  # hoyo rellenado
    replayed = simulate(world, actions)
    assert (0, 5) not in replayed.holes


def test_pico_puente_nunca_empuja_una_roca_sobre_un_pico():
    """
    Aunque el personaje pueda cruzar un pico, una ROCA no puede quedar
    encima de uno (activaria el pico con algo que no puede salir).
    """
    # La roca (0,1) tendria que ir al hoyo (0,4) cruzando el pico (0,3):
    # al empujarla sobre (0,3) quedaria atrapada ahi -> sin plan.
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 1)}),
        spikes=frozenset({(0, 3)}),
        holes=frozenset({(0, 4)}),
    )
    assert push_a_star(world, (0, 4), allow_bridge_spikes=True) is None


def test_pico_puente_rechaza_si_aisla_una_meta():
    """
    Con `keep_reachable`, cruzar un pico que aislaria una meta pendiente se
    rechaza (None); si la meta sigue alcanzable, el plan vale.
    """
    world = World(
        rows=1,
        cols=6,
        player=(0, 0),
        rocks=frozenset({(0, 3)}),
        spikes=frozenset({(0, 1)}),
        holes=frozenset({(0, 5)}),
    )
    # Cruzar (0,1) deja (0,0), la meta pendiente, inalcanzable.
    assert (
        push_a_star(
            world,
            (0, 5),
            allow_bridge_spikes=True,
            keep_reachable=frozenset({(0, 0)}),
        )
        is None
    )
    # Una meta al otro lado del pico sigue alcanzable: el plan vale.
    assert (
        push_a_star(
            world,
            (0, 5),
            allow_bridge_spikes=True,
            keep_reachable=frozenset({(0, 4)}),
        )
        is not None
    )
