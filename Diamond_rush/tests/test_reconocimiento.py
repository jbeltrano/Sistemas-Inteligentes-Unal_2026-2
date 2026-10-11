"""
Tests del reconocimiento: geometria del tablero, crear_grid (mascara ->
grilla) y matching de templates. Dependen de OpenCV + numpy (presentes
en el venv) y de los templates reales de Diamond_rush/templates.
"""
import numpy as np
import pytest

import config
from percepcion.reconocimiento import (
    TemplateMatcher,
    _resolver_solapes,
    cambiar_geometria_tablero,
    crear_grid,
    reconocer,
)


def test_geometria_desde_canvas():
    """Frame 606x915 (nivel 'captura'): celda cuadrada derivada de 15x10."""
    frame = np.zeros((915, 606, 3), dtype=np.uint8)
    tile, origin_x, origin_y = cambiar_geometria_tablero(frame)
    assert tile == 60          # min(606//10, 915//15)
    assert (origin_x, origin_y) == (3, 7)


def test_geometria_con_override():
    """Config permite forzar el tile cuando la captura no es el canvas."""
    frame = np.zeros((1079, 1919, 3), dtype=np.uint8)
    viejo_tile = config.TILE_SIZE
    try:
        config.TILE_SIZE = 90
        tile, origin_x, origin_y = cambiar_geometria_tablero(frame)
        assert tile == 90
        assert origin_x == (1919 - 10 * 90) // 2
        assert origin_y == (1079 - 15 * 90) // 2
    finally:
        config.TILE_SIZE = viejo_tile


def test_geometria_invalida_error():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        cambiar_geometria_tablero(frame)


def test_crear_grid_desde_mascara():
    """Celda = suelo si su zona central supera Min fraccion en la mascara."""
    tile, origin_x, origin_y = 60, 3, 7

    alto, ancho = 915, 606
    mascara = np.zeros((alto, ancho), dtype=np.uint8)
    # Piso de las filas 5..9 (descontando el borde que se ignora).
    mascara[300:602, 0:ancho] = 255

    grid = crear_grid(mascara, tile, origin_x, origin_y)
    assert grid.shape == (config.FILAS, config.COLUMNAS)

    # Filas 5..9 transitables; mas arriba obstaculo; HUD (filas 0..1) nuca.
    assert grid[5:10, :].all() == 1
    assert grid[:5, :].any() == 0
    assert grid[10:, :].any() == 0


def test_template_sin_coincidencias():
    """Un frame limpio (sin el juego) no matchea ningun template."""
    template = config.TEMPLATES_DIR / config.TEMPLATES["diamante"]
    frame = np.full((600, 600, 3), 255, dtype=np.uint8)
    matcher = TemplateMatcher(template, frame, nombre="diamante")
    assert matcher.match(umbral=config.UMBRAL, solape_max=config.SOLAPE_MAX) == []


def test_resolver_solapes_queda_el_mejor():
    """Dos coincidencias del mismo lugar: solo la de mejor score."""
    candidatos = [
        ("roca", 100, 100, 40, 40, 0.95),
        ("roca", 105, 105, 40, 40, 0.82),
        ("diamante", 300, 100, 40, 40, 0.91),
    ]
    resultado = _resolver_solapes([tuple(c) for c in candidatos], config.SOLAPE_MAX)
    assert [o.nombre for o in resultado] == ["roca", "diamante"]
    assert resultado[0].x == 100 and resultado[0].y == 100


def test_reconocer_frame_vacio_error_sin_personaje():
    """Sin personaje reconocible, reconocer() falla de forma clara."""
    frame = np.full((600, 915, 3), 200, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="personaje"):
        reconocer(frame)


def test_reconocer_captura_real():
    """End-to-end sobre captura.png: personaje ubicado, tablero plausible."""
    from percepcion.captura import leer_imagen

    frame = leer_imagen(config.IMAGEN_DEFECTO)
    estado = reconocer(frame)
    assert estado.grid.shape[0] == config.FILAS
    assert estado.grid.shape[1] == config.COLUMNAS
    assert estado.en_tablero(estado.player_cell)
    # En la captura del repo hay al menos el personaje.
    assert estado.player_cell is not None