"""
Tests del planificador: A*, empuje de rocas (Sokoban/Boulder), retroceso
y heuristica. Toda la logica es pura (sin cv2/numpy), corre sobre
Mundo/Plan.
"""
import pytest

from planificacion import (
    Planificador,
    Mundo,
    a_estrella,
    busqueda_con_retroceso,
    simular,
)


def _bordes(filas, columnas):
    """Muros del borde de un tablero filas x columnas."""
    return {
        (i, j)
        for i in range(filas)
        for j in range(columnas)
        if i in (0, filas - 1) or j in (0, columnas - 1)
    }


def _complemento(filas, columnas, abierto):
    """Todas las celdas que no estan en `abierto`."""
    return {
        (i, j)
        for i in range(filas)
        for j in range(columnas)
        if (i, j) not in abierto
    }


def test_camino_simple():
    """Corredor sin rocas: A* llega al diamante por el camino optimo."""
    mundo = Mundo(
        filas=5,
        columnas=5,
        muros=frozenset(_bordes(5, 5)),
        rocas=frozenset(),
        personaje=(3, 1),
        diamantes=frozenset({(1, 3)}),
    )
    plan = Planificador().planificar(mundo)
    assert plan.completo
    assert not plan.pendientes
    final = simular(mundo, plan.acciones)
    assert final.personaje == (1, 3)
    assert not final.diamantes
    # Distancia de Manhattan: |3-1| + |1-3| = 4 -> con heuristica admisible
    # y consistente el plan es optimo y tiene 4 pasos.
    assert plan.longitud == 4


def test_heuristica_esquiva_muros():
    """
    El campo de distancias (no Manhattan) tin un laberinto: para llegar
    al diamante hay que rodear los muros centrales, no ir en linea recta.
    """
    mundo = Mundo(
        filas=5,
        columnas=5,
        # Muros que obligan a rodear: (1,2) y (2,2) cierran el paso directo.
        muros=frozenset(_bordes(5, 5) | {(1, 2), (2, 2)}),
        rocas=frozenset(),
        personaje=(1, 1),
        diamantes=frozenset({(1, 3)}),
    )
    plan = Planificador().planificar(mundo)
    assert plan.completo
    final = simular(mundo, plan.acciones)
    assert final.personaje == (1, 3)
    # El rodeo optimo baja, cruza y sube: 6 pasos (Manhattan decia 2).
    assert plan.longitud == 6


def test_recolectar_varios_diamantes_y_salir():
    """Multiples diamantes mas salida: plan completo los recoge a todos."""
    abierto = {
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 3), (4, 4),
        (5, 1), (5, 2), (5, 3), (5, 4),
        (6, 1), (6, 2), (6, 3), (6, 4),
    }
    mundo = Mundo(
        filas=7,
        columnas=6,
        muros=frozenset(_complemento(7, 6, abierto)),
        rocas=frozenset(),
        personaje=(5, 4),
        diamantes=frozenset({(1, 2), (6, 1)}),
        salida=(2, 3),
    )
    plan = Planificador().planificar(mundo)
    assert plan.completo
    final = simular(mundo, plan.acciones)
    assert not final.diamantes
    assert final.personaje == mundo.salida


def test_empujar_roca_para_despejar_paso():
    """
    Sokoban-like: una roca cunde el unico acceso al diamante; el plan
    debe empujarla hacia el hueco lateral (pocket) para pasar.
    """
    abierto = {
        (0, 3),             # pocket arriba donde sacar la roca
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 2), (2, 3),     # pasillo inferior para ubicarse bajo la roca
    }
    mundo = Mundo(
        filas=3,
        columnas=6,
        muros=frozenset(_complemento(3, 6, abierto)),
        rocas=frozenset({(1, 3)}),
        personaje=(1, 1),
        diamantes=frozenset({(1, 4)}),
    )
    plan = Planificador().planificar(mundo)

    assert plan.completo
    empujes = [a for a in plan.acciones if a.tipo == "empujar"]
    assert empujes, "deberia haberse empujado la roca"
    assert any(a.direccion == "arriba" for a in empujes)

    final = simular(mundo, plan.acciones)
    assert final.personaje == (1, 4)
    assert not final.diamantes
    assert final.rocas == frozenset({(0, 3)})


def test_retroceso_resuelve_cuando_aestrella_no(monkeypatch):
    """
    El Planificador cae en la busqueda con retroceso si A* no entrega
    solucion (se fuerza con monkeypatch el fallo de A*).
    """
    import planificacion as mod

    abierto = {
        (0, 3),
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 2), (2, 3),
    }
    mundo = Mundo(
        filas=3,
        columnas=6,
        muros=frozenset(_complemento(3, 6, abierto)),
        rocas=frozenset({(1, 3)}),
        personaje=(1, 1),
        diamantes=frozenset({(1, 4)}),
    )

    monkeypatch.setattr(mod, "a_estrella", lambda *a, **k: None)

    plan = Planificador().planificar(mundo)
    assert plan.completo
    assert any(a.tipo == "empujar" for a in plan.acciones)
    final = simular(mundo, plan.acciones)
    assert final.personaje == (1, 4)
    assert not final.diamantes


def test_llave_antes_de_la_salida():
    """Con llave y salida (sin diamantes): la recoge y luego sale."""
    abierto = {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)}
    mundo = Mundo(
        filas=3,
        columnas=6,
        muros=frozenset(_complemento(3, 6, abierto)),
        rocas=frozenset(),
        personaje=(1, 0),
        diamantes=frozenset(),
        llave=(1, 2),
        salida=(1, 4),
    )
    plan = Planificador().planificar(mundo)
    assert plan.completo
    final = simular(mundo, plan.acciones)
    assert final.tiene_llave
    assert final.personaje == mundo.salida


def test_diamante_inalcanzable_marca_incompleto():
    """
    Un diamante aislado por muros (lava/pared) no se puede recoger con
    el modelo: el plan queda incompleto pero igual busca la salida.
    """
    abierto_central = {
        (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 1), (3, 2), (3, 3), (3, 4),
        (4, 1), (4, 2), (4, 3), (4, 4),
    }
    # Un diamante interior totalmente rodeado de muros (como detras de lava).
    muros = _complemento(5, 5, abierto_central) | {(0, 2), (2, 2), (1, 1), (1, 3)}
    mundo = Mundo(
        filas=5,
        columnas=5,
        muros=frozenset(muros),
        rocas=frozenset(),
        personaje=(4, 2),
        diamantes=frozenset({(1, 2)}),
        salida=(1, 4),
    )
    plan = Planificador().planificar(mundo)
    assert not plan.completo
    assert any(p.startswith("diamante") for p in plan.pendientes)
    final = simular(mundo, plan.acciones)
    assert final.personaje == mundo.salida


def test_simular_detecta_pasos_invalidos():
    """La simulacion tiene que tirar error ante un plan que no reproduce."""
    import planificacion
    from planificacion import Accion

    mundo = Mundo(
        filas=3,
        columnas=3,
        muros=frozenset(_bordes(3, 3)),
        personaje=(1, 1),
    )
    # Empujar sin roca: Accion invalida.
    with pytest.raises(ValueError):
        planificacion.simular(mundo, [Accion("empujar", "arriba")])


def test_a_estrella_y_retroceso_directos():
    """Los motores sueltos tambien funcionan por fuera del Planificador."""
    mundo = Mundo(
        filas=4,
        columnas=4,
        muros=frozenset(_bordes(4, 4)),
        personaje=(1, 1),
        diamantes=frozenset({(2, 2)}),
    )
    meta = mundo.diamantes
    resultado = a_estrella(mundo, meta)
    assert resultado is not None
    acciones, final = resultado
    assert final.personaje == (2, 2)
    assert simular(mundo, acciones).personaje == (2, 2)

    resultado_retro = busqueda_con_retroceso(mundo, meta)
    assert resultado_retro is not None
    assert resultado_retro[1].personaje == (2, 2)