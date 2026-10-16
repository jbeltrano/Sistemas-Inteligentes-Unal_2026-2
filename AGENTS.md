# AGENTS.md

Repositorio del curso Sistemas Inteligentes (UNAL 2026-2). Proyectos independientes, una carpeta por proyecto. Sin tooling compartido: no hay CI, ni linter, ni `requirements` raíz, ni `pytest.ini`/`setup.cfg`.

**Idioma:** READMEs, comentarios, docstrings, prints y mensajes de commit van en español (aunque algunos nombres de módulos en `diamond_rush_v2/` estén en inglés).

## Estado del working tree (revisar `git status` antes de commitear)

- `AGENTS.md` y `Picas_y_Fijas/` aparecen **borrados** en el working tree (siguen en `HEAD`): no los restaures ni los commitees sin preguntar.
- `Diamond_rush/*.py`, `Diamond_rush/percepcion/`, `Diamond_rush/tests/` están **sin trackear**. El working tree suele estar sucio; no commitear salvo que se pida.

## Proyectos

### `Diamond_rush/` — agente v1 (el más completo/activo)

Agente deliberativo basado en modelos (sin aprendizaje) para el juego "Diamond Rush". Cierra el lazo percepción → decisión → control → verificación. Lógica y contrato de estado en español (`percepcion/estado.py`, `percepcion/reconocimiento.py`, `planificacion.py`, `control.py`, `agente.py`). `Search_objects.py` es el script monolítico original: legado, no lo uses como fuente de verdad.

```
# desde dentro de Diamond_rush/
python main.py                         # plan sobre captura.png (por defecto)
python main.py --imagen levels/level1.png
python main.py --capturar              # 1 screenshot de pantalla (mss)
python main.py --enviar                # además suelta el plan por teclado
python main.py --auto                  # = --capturar --enviar
```

- Deps en `Diamond_rush/requirements.txt` (numpy, opencv-python, mss, pynput, pytest); instalar dentro de un venv.
- El plan se calcula con **UNA sola captura**; `--enviar` recién ahí manda teclas. `--capturar` usa `mss` sobre la pantalla, no el canvas vía DOM.
- Backend de teclado: `pynput` o `xdotool` (autodetectado en `control.py`). Sin ninguno de los dos, `Teclado()` lanza `RuntimeError`.

### `diamond_rush_v2/` — refactor (parcial)

Reorganización modular del v1: `perception/` (implementado), `computation/` y `play/` son **stubs vacíos** (`compute.py` y `control.py` con 0 líneas). Solo existe la percepción + un agente de una sola pasada que imprime el `GameState`.

```
# desde dentro de diamond_rush_v2/
python main.py        # run_agent(); requiere Chrome con remote debugging (ver abajo)
```

- **Contrato de percepción:** `perception/state.py` define `GameState`, el único contrato entre percepción y el resto. Regla: `computation/` y `play/` importan **solo** de `perception.state`, nunca de `cv2` ni de `recognition.py`. Detalle completo en `diamond_rush_v2/perception/README.md` (leerlo antes de tocar percepción).
- Captura con **Selenium** sobre un Chrome ya abierto: `capture.attach_driver()` se conecta por CDP a `127.0.0.1:9222` sin navegar. **No llamar `driver.quit()`** sobre ese driver: cierra el navegador abierto a mano con el progreso del profesor.
- Lanzar el Chrome con debugging (comando en `Nuevo Documento de texto.txt`):
  `chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\Users\juanp\chrome-agente"`. Hay que pasar el tutorial a mano y dejar el nivel en pantalla.
- **No hay `requirements.txt`** para v2; necesita `selenium` (además de opencv/numpy), que no está listado en ningún lado.
- `main.py` tiene comentadas las llamadas a depuración; para revisar reconocimiento sobre imágenes estáticas usa `debug.test_recognition(path)` / `debug.test_recognition_all_levels()` (descomentar en `main.py`). `debug.py` no se usa desde `agent.py`.

## Hechos compartidos

- **Geometría del tablero:** siempre 15 filas × 10 columnas. `tile_size` se deriva del frame (`min(ancho//10, alto//15)`), no se estima. El HUD se dibuja **encima** del tablero (no lo desplaza); las celdas bajo `HUD_HEIGHT` valen 0. Cuidado con el orden: celdas `(fila, col)`, píxeles `(x, y)`.
- **Templates:** `templates/`/`assets/templates/` son recortes a mano de un frame concreto; si el juego se ve distinto dejan de matchear y hay que recalibrar `THRESHOLD`/`MIN_PATH_FRACTION` por nivel. Primera cosa a revisar ante suelo/objetos mal detectados: la máscara de caminos (HSV).
- **Sin scraping:** el estado del juego se lee solo de píxeles (templates/máscara). No leer datos del juego por HTTP/DOM/JS. En v2 Selenium se usa únicamente para encuadrar el canvas, no para extraer datos.

## Tests

Solo `Diamond_rush/` tiene tests (pytest). Correr **desde dentro de `Diamond_rush/`** (no desde la raíz; `tests/conftest.py` mete `Diamond_rush` en `sys.path`):

```
# desde Diamond_rush/
pytest                                   # toda la suite
pytest tests/test_planificacion.py       # un módulo
pytest tests/test_reconocimiento.py::test_reconocer_captura_real  # un test
```

Los tests end-to-end dependen de OpenCV + numpy + los templates reales y de `captura.png`; no son puros de lógica.
