"""
Tests del modelo de estado: armado del EstadoJuego desde detecciones
(construir_estado) y traduccion a Mundo (mundo_desde_estado). Puros,
sin OpenCV.
"""
import numpy as np
import pytest

from percepcion.estado import ObjetoDetectado, construir_estado
from planificacion import Mundo, mundo_desde_estado

ORIGEN_X, ORIGEN_Y, TILE = 0, 0, 60


def _deteccion(nombre, centro_x, centro_y, ancho=40, alto=40):
    return ObjetoDetectado(nombre, centro_x - ancho // 2, centro_y - alto // 2, ancho, alto)


def _grid_vacia():
    return np.zeros((15, 10), dtype=np.uint8)


def test_construir_estado_agrupa_y_fuerza_transitables():
    grid = _grid_vacia()
    diamante = _deteccion("diamante", 180, 240)   # celda (4, 3)
    roca = _deteccion("roca", 300, 120)           # celda (2, 5)

    estado = construir_estado(
        grid,
        [diamante, roca],
        player_cell=(1, 1),
        origin_x=ORIGEN_X,
        origin_y=ORIGEN_Y,
        tile_size=TILE,
    )

    assert estado.player_cell == (1, 1)
    assert estado.grid[4, 3] == 1
    assert estado.grid[2, 5] == 1
    assert estado.grid[1, 1] == 1
    assert estado.grid[0, 0] == 0

    assert (4, 3) in estado.objects_by_cell
    assert [o.nombre for o in estado.objects_by_cell[(4, 3)]] == ["diamante"]
    assert [o.nombre for o in estado.objetos_por_celda((2, 5))] == ["roca"]
    assert estado.objetos_por_celda((0, 0)) == []


def test_deteccion_fuera_del_tablero_se_ignora():
    grid = _grid_vacia()
    estado = construir_estado(
        grid,
        [_deteccion("diamante", 600, -50)],
        player_cell=(1, 1),
        origin_x=ORIGEN_X,
        origin_y=ORIGEN_Y,
        tile_size=TILE,
    )
    assert estado.objects_by_cell == {}


def test_personaje_fuera_del_tablero_error():
    with pytest.raises(RuntimeError):
        construir_estado(
            _grid_vacia(),
            [],
            player_cell=(50, 50),
            origin_x=ORIGEN_X,
            origin_y=ORIGEN_Y,
            tile_size=TILE,
        )


def test_celda_de_y_en_tablero():
    estado = construir_estado(
        _grid_vacia(),
        [],
        player_cell=(1, 1),
        origin_x=ORIGEN_X,
        origin_y=ORIGEN_Y,
        tile_size=TILE,
    )
    assert estado.celda_de((180, 240)) == (4, 3)
    assert estado.en_tablero((2, 5))
    assert not estado.en_tablero((15, 0))
    assert not estado.en_tablero((0, -1))


def _estado_con(objetos, player_cell=(2, 2)):
    """Estado listo para traducir a Mundo."""
    return construir_estado(
        _grid_vacia(),
        objetos,
        player_cell=player_cell,
        origin_x=ORIGEN_X,
        origin_y=ORIGEN_Y,
        tile_size=TILE,
    )


def test_mundo_desde_estado_roles():
    estado = _estado_con(
        [
            _deteccion("diamante", 180, 240),
            _deteccion("roca", 300, 120),
            _deteccion("llave", 60, 240),
            _deteccion("lava", 480, 600),
            _deteccion("salida_bloqueada", 360, 720),
        ],
        player_cell=(2, 2),
    )

    mundo = mundo_desde_estado(estado)
    assert isinstance(mundo, Mundo)
    assert (4, 3) in mundo.diamantes          # diamante
    assert (2, 5) in mundo.rocas              # roca
    assert mundo.llave == (4, 1)              # llave en celda (4, 1)
    assert mundo.salida == (12, 6)            # salida en celda (12, 6)
    assert (10, 8) in mundo.muros             # lava -> peligro -> muro

    # El personaje arranca en su celda.
    assert mundo.personaje == (2, 2)
    assert not mundo.tiene_llave


def test_mundo_peligro_es_muro_y_roca_empujable():
    estado = _estado_con(
        [_deteccion("hueco", 60, 60), _deteccion("roca", 120, 60)],
        player_cell=(2, 2),
    )
    mundo = mundo_desde_estado(estado)
    assert (1, 1) in mundo.muros   # hueco
    assert (1, 2) in mundo.rocas
    assert (1, 2) not in mundo.muros