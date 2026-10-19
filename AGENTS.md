# AGENTS.md

Curso Sistemas Inteligentes (UNAL 2026-2). Un solo proyecto activo: `diamond_rush_v2/` — agente deliberativo (sin aprendizaje) que juega Diamond Rush en el navegador: percepción (1 frame por Selenium) → decisión (A* ponderado + retroceso) → control (flechas por Selenium) → verificación con re-planificación. Sin CI, ni linter, ni typecheck; la única verificación es pytest.

**Idioma:** READMEs, comentarios, docstrings, prints y mensajes de commit van en español.

## Setup

```bash
pip install -r diamond_rush_v2/requirements.txt   # pinned; en un venv
pip install pytest                                # NO está en requirements.txt
```

## Comandos

```bash
# El agente (funciona desde la raíz o desde diamond_rush_v2/;
# los imports son planos, por eso no sirve `import agent` en un REPL
# sin tener diamond_rush_v2/ en sys.path)
python diamond_rush_v2/main.py                    # agente completo (requiere Chrome, ver abajo)
python diamond_rush_v2/main.py --no-execute       # solo percibir y planificar, sin mandar teclas
python diamond_rush_v2/main.py --debug            # imprime el GameState completo
python diamond_rush_v2/main.py --verify-every 1   # re-percibe cada paso; 0 = solo al final;
                                                  # por defecto: cada 10 (config.VERIFY_EVERY)
python diamond_rush_v2/main.py --level 6          # nivel actual; activa el mapeo botón→reja
                                                  # de button_gate.py (varios pares) y las
                                                  # pistas de subobjetivos de level_hints.py

# Tests: solo pytest, sin navegador (~36 tests, ~2 min). Sirve desde
# diamond_rush_v2/ o desde la raíz (tests/conftest.py mete la raíz del
# proyecto en sys.path)
pytest
pytest tests/test_planificacion.py
pytest tests/test_planificacion.py::test_a_star_devuelve_camino_optimo
```

**Ojo con los skips:** dos tests de integración (`test_recognize_y_plan_consistente`, `test_nivel5_picos_sin_recrucex`) corren `recognize` sobre `assets/levels/level5.png` y hacen `pytest.skip` si el reconocimiento falla. Un resultado esencialmente verde puede esconder una regresión de percepción — mirar si alguien pasó a skip.

## Chrome con debugging (requisito para correr el agente)

`perception/capture.py:attach_driver()` se conecta por CDP a un Chrome ya abierto (`127.0.0.1:9222`) sin navegar. El comando está en `Ejecucion.md` (raíz del repo):

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\juanp\chrome-agente"
```

Abrir `https://www.minijuegos.com/embed/diamond-rush`, pasar el tutorial a mano y dejar el nivel en pantalla antes de correr `main.py`. **Nunca llamar `driver.quit()`** sobre el driver de `attach_driver()`: cierra el navegador con el progreso del nivel. Cerrar Chrome a mano.

## Arquitectura por capas (contrato fuerte)

`GameState` (`perception/state.py`) es el **único** contrato entre percepción y el resto. Regla: `computation/` y `play/` importan **solo** de `perception.state` y `config`, nunca de `cv2` ni de `recognition.py` (verificado por grep; `agent.py` es el único que cierra el lazo). **Leer `perception/README.md` antes de tocar percepción ni planificación**: documenta campo por campo el contrato, cómo se usa en A\* y las limitaciones conocidas.

- `perception/` — captura (Selenium sobre el canvas) y reconocimiento (templates + máscara HSV de suelo). Produce un `GameState` por frame.
- `computation/` — modelo (`World`) y planificación: A* ponderado (`A_STAR_WEIGHT = 2.0`) sobre el espacio de mundos, con `backtracking_search` (DFS) de respaldo si A* se queda sin presupuesto. Planifica en fases: **búsqueda global de recolección** (todos los diamantes en un solo A*; llaves/rejas/botones instrumentales) → salida; si la global revienta el presupuesto, `_plan_fallback` compone subobjetivos (llaves → botones → diamantes → salida). Presupuestos y pesos en `config.py`. Para niveles que la búsqueda global no resuelve (6 lento, 10 hoy) se pueden declarar **pistas de subobjetivos** en `level_hints.py` (misma opción `--level N`): una secuencia ordenada que el fallback persigue primero (una pista por A\*, con waypoints de roca encadenables) y luego completa el resto automáticamente; son preferencias revalidadas contra lo percibido, no un guion. Las rejas son **temporales** (abiertas solo mientras su botón esté cubierto por el personaje o una roca): el emparejamiento botón→reja es 1:1 y, con varios pares, se declara por nivel en `button_gate.py` (se pasa con `--level N`; lo no declarado se empareja por cercanía). Para rejas temporales **instrumentales** (tapan una meta pendiente) el subobjetivo `boton` solo acepta **una roca encima**, nunca al personaje (pisar y salir la cierra): era un falso éxito que ciclaba el fallback. A* admite además `rock_goal` (heurística rock-aware para empujes Sokoban) y `keep_reachable` (seguridad de picos: no permite activar un pico que aísle una meta pendiente). **Limitación conocida:** el nivel 11 necesita un Sokoban de **dos rocas por la columna 7** (una rellena el hoyo (5,7) y se consume, otra llega al botón (3,7)); el fallback, que resuelve un subobjetivo por A*, no lo compone y el plan queda incompleto.
- `play/` — control: cada dirección es una flecha enviada al canvas en **una sola** `ActionChains.perform()` (key_down → pausa → key_up). Dos `perform()` separados = dos casillas de movimiento (bug ya corregido, ver docstring de `Keyboard.press`).
- `agent.py` — cierra el lazo: ejecuta el plan y, cada `VERIFY_EVERY` pasos, re-percibe con "settle" (`_settle`: espera a que la celda del personaje se asiente) y re-planifica si hay desvío (tope `MAX_REPLANS = 50`). Si el plan queda **incompleto** no se ejecuta: imprime las metas pendientes y se detiene sin mandar teclas (ejecutarlo dejaría el nivel peor sin resolverlo).

## Depuración de percepción

- `debug.py` no se usa desde `agent.py`; es manual: descomentar las llamadas en `main.py` (`test_recognition(path)`, `test_recognition_all_levels()`). Escribe `debug_paths.png`, `debug_graph.png`, `debug_result.png` (gitignoreados).
- Si `recognize` falla, `agent._report_miss` guarda el frame en `debug_misses/` (gitignoreado) para analizarlo con `test_recognition`.

## Geometría y percepción

- **Tablero:** siempre 15 filas × 10 columnas (`config.BOARD_ROWS/COLS`). `tile_size = min(ancho//10, alto//15)`, no se estima. El HUD se dibuja **encima** del tablero (no lo desplaza); las celdas bajo `HUD_HEIGHT` valen 0. Celdas `(fila, col)`, píxeles `(x, y)` — cuidado con el orden.
- **Templates:** `assets/templates/` son recortes a mano de un frame concreto; si el juego se ve distinto dejan de matchear y hay que recalibrar `THRESHOLD`/`MIN_PATH_FRACTION` (`config.py`) por nivel. Ante suelo/objetos mal detectados, revisar primero la máscara de caminos HSV (`debug_paths.png`).
- **Sin scraping:** el estado se lee solo de píxeles (templates/máscara). Selenium se usa únicamente para encuadrar el canvas y pulsar flechas, nunca para extraer datos del juego por HTTP/DOM/JS.
