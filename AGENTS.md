# AGENTS.md

Repositorio del curso Sistemas Inteligentes (UNAL 2026-2). Un solo proyecto activo: `diamond_rush_v2/` (agente que juega Diamond Rush en el navegador). Sin tooling compartido: no hay CI, ni linter, ni configuración de tipos.

**Idioma:** READMEs, comentarios, docstrings, prints y mensajes de commit van en español.

## `diamond_rush_v2/` — único proyecto

Agente deliberativo basado en modelos (sin aprendizaje). Ciclo: percepción (1 frame vía Selenium) → decisión (A* ponderado + retroceso) → control (flechas por Selenium) → verificación con re-planificación.

```
# desde dentro de diamond_rush_v2/
python main.py                       # agente completo (requiere Chrome con debugging, ver abajo)
python main.py --no-execute          # solo percibir y planificar, sin mandar teclas
python main.py --debug               # imprime el GameState completo
python main.py --verify-every 1      # re-percibe cada paso (por defecto: cada 10; 0 = solo al final)
```

- Deps en `diamond_rush_v2/requirements.txt` (numpy, opencv-python, selenium, pytest); instalar dentro de un venv. No hay requirements en la raíz.
- Ejecutar siempre desde dentro de `diamond_rush_v2/`: no hay paquete instalable, los imports son planos (`from agent import Agent`) y dependen del CWD. `tests/conftest.py` mete la raíz del proyecto en `sys.path` para que pytest funcione desde cualquier lado.

### Chrome con debugging (requisito para correr el agente)

El agente se conecta por CDP a un Chrome ya abierto (`perception/capture.py:attach_driver()`, `127.0.0.1:9222`) sin navegar. El comando está en `Ejecucion.md` (raíz del repo):

```
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\juanp\chrome-agente"
```

Hay que pasar el tutorial a mano y dejar el nivel en pantalla antes de correr `main.py`. **Nunca llamar `driver.quit()`** sobre el driver de `attach_driver()`: cierra el navegador abierto a mano con el progreso del nivel.

### Arquitectura por capas (contrato fuerte)

`GameState` (`perception/state.py`) es el **único** contrato entre percepción y el resto. Regla: `computation/` y `play/` importan **solo** de `perception.state` y `config`, nunca de `cv2` ni de `recognition.py`. Detalle completo del contrato en `perception/README.md` (leerlo antes de tocar percepción ni planificación).

- `perception/` — captura (Selenium sobre el canvas) y reconocimiento (templates + máscara HSV de suelo). Produce un `GameState` por frame.
- `computation/` — modelo (`World`) y planificación: A* ponderado (`A_STAR_WEIGHT = 2.0`) sobre el espacio de mundos (posición + rocas + diamantes + picos), con `backtracking_search` (DFS) de respaldo si A* se queda sin presupuesto. Heurística: campo de distancias BFS multi-fuente sobre la grilla sin rocas (admisible). Planifica en fases: **búsqueda global de recolección** (todos los diamantes en un solo A*; llaves/rejas/botones instrumentales) → salida; si la global revienta el presupuesto, respaldo con la composición clásica de subobjetivos (llaves → botones → diamantes → salida). Los picos retraídos se activan permanentemente al salir de la celda (una sola cruzada).
- `play/` — control: cada dirección del plan es una flecha enviada al canvas por Selenium (`ActionChains` key_down → pausa → key_up en un solo `perform()`; dos `perform()` separados mueven dos casillas).
- `agent.py` — cierra el lazo: ejecuta el plan y, cada `VERIFY_EVERY` pasos, re-percibe ("settle" hasta que la celda del personaje se asiente) y re-planifica si hay desvío (tope `MAX_REPLANS`).

### Tests

Solo pytest, dentro de `diamond_rush_v2/`:

```
# desde diamond_rush_v2/
pytest                                   # toda la suite (~21 tests)
pytest tests/test_planificacion.py       # un módulo
pytest tests/test_planificacion.py::test_plan_completo_sin_grafo   # un test
```

Los tests del planificador son de lógica pura (usan `World` directo, sin OpenCV ni navegador). No hay tests de percepción ni end-to-end.

### Depuración de percepción

- `debug.py` no se usa desde `agent.py`; es solo para revisar a mano: descomentar las llamadas en `main.py` (`test_recognition(path)`, `test_recognition_all_levels()`).
- Si `recognize` falla, `agent._report_miss` guarda el frame en `debug_misses/` (gitignoreado) para analizarlo con `test_recognition`.

## Hechos compartidos (geometría y percepción)

- **Tablero:** siempre 15 filas × 10 columnas (`config.BOARD_ROWS/COLS`). `tile_size = min(ancho//10, alto//15)`, no se estima. El HUD se dibuja **encima** del tablero (no lo desplaza); las celdas bajo `HUD_HEIGHT` valen 0. Celdas `(fila, col)`, píxeles `(x, y)` — cuidado con el orden.
- **Templates:** `assets/templates/` son recortes a mano de un frame concreto; si el juego se ve distinto dejan de matchear y hay que recalibrar `THRESHOLD`/`MIN_PATH_FRACTION` (`config.py`) por nivel. Primera cosa a revisar ante suelo/objetos mal detectados: la máscara de caminos HSV (`debug_paths.png`).
- **Sin scraping:** el estado del juego se lee solo de píxeles (templates/máscara). Selenium se usa únicamente para encuadrar el canvas y pulsar flechas, nunca para extraer datos del juego por HTTP/DOM/JS.
