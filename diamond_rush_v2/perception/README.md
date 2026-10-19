# `GameState`: el grafo del agente

`GameState` (definido en `perception/state.py`) es **el único contrato entre la percepción y el resto del agente**. Lo produce `recognize(frame)` en `perception/recognition.py` y lo consumen el cálculo (A\*, planificación) y la actuación.

```
percibir ──► recognize(frame) ──► GameState ──► calcular (A*) ──► plan ──► actuar
 (cv2)                           (este doc)
```

Regla: `computation/` y `play/` importan **solo** de `perception.state`, nunca de `cv2` ni de `recognition.py`.

Un `GameState` es una **foto de un único frame**. No guarda historia ni predice nada. Si el planificador necesita simular movimientos (empujar una roca, recoger una llave), debe llevar su propio estado simulado y usar el `GameState` como punto de partida.

---

## 1. Convenciones

| Concepto | Convención |
|---|---|
| **Celda** | Tupla `(fila, columna)`. Alias `Cell = Tuple[int, int]`. |
| **Origen** | `(0, 0)` es la celda superior izquierda del tablero. La fila crece hacia abajo, la columna hacia la derecha. |
| **Píxel** | Tupla `(x, y)`: `x` horizontal, `y` vertical. **Ojo con el orden**: las celdas son `(fila, col)` y los píxeles son `(x, y)`. |
| **Tablero** | Siempre `config.BOARD_ROWS × config.BOARD_COLS` = **15 filas × 10 columnas**, sin importar el nivel. |
| **Vecindad** | 4 direcciones (arriba, abajo, izquierda, derecha). Sin diagonales. |

Conversión píxel ↔ celda (con los campos de `GameState`):

```
fila    = (y - origin_y) // tile_size
columna = (x - origin_x) // tile_size

x_izq_de_celda = origin_x + columna * tile_size
y_sup_de_celda = origin_y + fila    * tile_size
centro_celda   = (x_izq + tile_size // 2, y_sup + tile_size // 2)
```

---

## 2. Campos

### Resumen

| Campo | Tipo | Qué es |
|---|---|---|
| `grid` | `np.ndarray (15, 10)`, `uint8` | Mapa de suelo completo, sin filtrar |
| `graph_nodes` | `Set[Cell]` | Celdas de suelo **alcanzables** desde el personaje |
| `adjacency` | `Dict[Cell, List[Cell]]` | Lista de adyacencia del grafo |
| `player_cell` | `Cell` | Celda del personaje |
| `objects` | `List[DetectedObject]` | Todas las detecciones, con posición en píxeles |
| `objects_by_cell` | `Dict[Cell, List[DetectedObject]]` | Las mismas detecciones, agrupadas por celda |
| `origin_x`, `origin_y` | `int` | Esquina superior izquierda del tablero en el frame (px) |
| `tile_size` | `int` | Lado de una celda en píxeles |
| `player_has_key` | `bool` | El personaje lleva **llave en la mano** (variante `player_key` de su sprite) |

**El grafo son `graph_nodes` + `adjacency`.** El resto son datos de apoyo.

---

### `grid`

Matriz de `15 × 10` con `grid[fila, columna]`:

- `1`: la celda es **suelo** (parte del tablero por donde se puede estar).
- `0`: la celda es **pared / obstáculo fijo / HUD**.

Una celda vale `1` si ocurre alguna de estas tres cosas:

1. Su zona central (sin el margen `CELL_MARGIN`) tiene más de `MIN_PATH_FRACTION` de píxeles de suelo en la máscara de color (`detect_paths`).
2. Hay un objeto detectado en ella (un objeto solo puede estar sobre suelo, aunque su sprite tape el color).
3. Es la celda del personaje.

Las celdas cuyo centro queda bajo el HUD (`y < HUD_HEIGHT`) siempre valen `0`.

Importante:

- **`grid` no está filtrado por alcanzabilidad.** Puede haber suelo aislado que el personaje nunca pisará. Eso lo resuelve `graph_nodes`.
- **`grid` no dice qué bloquea.** Una celda con una roca, un hoyo o lava vale `1`. Que se pueda pisar depende del objeto (ver §4).
- Sirve para consultar "¿esta celda existe como suelo?" sin pasar por el grafo, y para depurar (`draw_graph` lo dibuja).

---

### `graph_nodes`

Conjunto de celdas **de suelo y alcanzables** desde `player_cell` mediante vecinos de 4 direcciones (BFS en `reachable_component`). Siempre contiene `player_cell`.

Es el conjunto de **nodos** del grafo. Todo nodo cumple `grid[nodo] == 1`, pero no al revés.

La alcanzabilidad se calcula **ignorando los objetos**. Una celda detrás de una roca, un hoyo o una puerta cerrada sigue siendo nodo. Que sea *realmente* alcanzable lo decide el planificador.

---

### `adjacency`

Lista de adyacencia: para cada celda de `graph_nodes`, la lista de sus vecinos que también están en `graph_nodes`.

```python
state.adjacency[(4, 1)]   # p. ej. [(5, 1), (4, 2)]  (hasta 4 vecinos)
```

- Las claves son exactamente `graph_nodes` (misma cantidad de entradas).
- Es **simétrica**: si `b in adjacency[a]` entonces `a in adjacency[b]`.
- Sin pesos: todas las aristas cuestan lo mismo (un movimiento).
- Es lo que expande A\*: `for vecino in state.adjacency[nodo]`.
- Número de aristas = `sum(len(v) for v in adjacency.values()) // 2`.

---

### `player_cell`

Celda `(fila, columna)` donde está el personaje, calculada desde el **centro** de su detección. Siempre está dentro del tablero y siempre pertenece a `graph_nodes`. Es el nodo de **inicio** del A\*.

Si el personaje no se detecta, `recognize` lanza `RuntimeError`: es el único objeto obligatorio.

---

### `player_has_key`

`True` cuando el sprite del personaje corresponde a la variante **con llave en la mano** (`player_key.png`). Lo decide `recognize` comparando el mejor score de `player.png` y `player_key.png` sobre la región del frame donde está el jugador detectado: gana el de mayor score. `False` si no hay templates o no se pudo determinar.

Sirve para que `world_from_state` sepa si el personaje **ya lleva una llave** (por ejemplo, si un tester la recogió a mano antes de correr el agente) sin tener que mirar la mano del personaje. Además, al traducir el estado, la detección `key` que cae dentro de la caja del personaje **se descarta de `world.keys`**: esa llave es la pintada dentro de su propio sprite, no una llave suelta del nivel (contarla anularía `World.has_key`).

---

### `objects`

Lista de `DetectedObject` con **todo** lo que se detectó por template matching, ya filtrado:

- Se descartan detecciones sobre el HUD (el contador de diamantes del encabezado).
- Se descartan detecciones fuera del tablero.
- Si dos templates se superponen más de `MAX_OVERLAP`, gana el de mayor score (rock/rock_in_hole, spike/no_spike, etc.).
- **El personaje va primero** (`objects[0].name == "player"`) y también aparece aquí.

Cada `DetectedObject` tiene:

| Atributo | Significado |
|---|---|
| `name` | Nombre lógico (`config.TEMPLATE_NAMES`). Las variantes de un mismo objeto (`player`, `player_key`) comparten nombre. |
| `x`, `y` | Esquina superior izquierda de la detección, en píxeles |
| `width`, `height` | Tamaño del template, en píxeles |
| `center` | Propiedad: `(x + width // 2, y + height // 2)` |

Las coordenadas son de **píxeles del frame**, no de celdas. Para celdas, usar `objects_by_cell` o `state.cell_of(obj.center)`.

---

### `objects_by_cell`

Las mismas detecciones que `objects`, agrupadas por la celda que contiene su **centro**.

```python
state.objects_by_cell.get((5, 4), [])   # objetos en esa celda; la clave puede no existir
```

- Es un diccionario normal: **usa `.get(celda, [])`**, las celdas sin objetos no tienen clave.
- El valor es una lista, porque en teoría una celda puede tener más de un objeto (por ejemplo, el personaje sobre un botón).
- Incluye al personaje en `player_cell`.
- Es la herramienta principal del planificador: al expandir un nodo, mirar qué hay en él.

---

### `origin_x`, `origin_y`, `tile_size`

Parámetros de la retícula, para convertir entre píxeles y celdas.

- `tile_size`: lado de la celda, `min(ancho // 10, alto // 15)`. Con un frame de 724×1080 es **72**.
- `origin_x`, `origin_y`: esquina superior izquierda de la celda `(0, 0)`. Con ese mismo frame: `(2, 0)`. El HUD **no** desplaza el origen: se dibuja encima de las filas superiores.

Se necesitan para **actuar**: convertir una celda destino en un punto de clic/arrastre sobre el canvas.

---

## 3. Métodos

| Método | Hace |
|---|---|
| `objects_named("diamond")` | Lista de detecciones de ese tipo (en píxeles). |
| `cell_of((x, y))` | Convierte un punto en píxeles a su celda `(fila, columna)`. Ej.: `state.cell_of(obj.center)`. |

---

## 4. Tipos de objeto

Los nombres válidos son los de `config.TEMPLATE_NAMES`. **El grafo no codifica cuáles bloquean el paso**: eso depende del estado del juego y lo decide el planificador. La columna "Rol probable" es una guía y las filas marcadas con `?` hay que confirmarlas jugando.

| `name` | Rol probable |
|---|---|
| `player` | El personaje (variantes `player` y `player_key`, con llave en la mano) |
| `diamond` | Coleccionable; objetivo del nivel |
| `key` | Llave suelta que se recoge |
| `door_key` | Puerta con cerradura que se abre con la llave |
| `rock` | Roca empujable; bloquea hasta que se empuja |
| `hole` | Hoyo; no se puede cruzar salvo que se rellene con una roca |
| `rock_in_hole` | Hoyo ya rellenado (ahora transitable) |
| `lava` | Peligro mortal |
| `spike` / `no_spike` | Pinchos activos / pinchos retraídos (estados del mismo objeto) |
| `button` | Placa de presión: mantiene abierta su reja mientras el personaje o una roca la cubra (emparejamiento 1:1) |
| `closed_gate` / `open_gate` | Reja cerrada / abierta (estados del mismo objeto) |
| `scape_block` / `scape_enable` | Bloque de escape y su estado activado (`?`) |
| `totem` | Estatua (`?`) |

---

## 5. Ejemplo: nivel 5

Frame de 724×1080 → `tile_size = 72`, `origin = (2, 0)`, `grid.shape == (15, 10)`. Valores **aproximados**, leídos de la captura:

```
player_cell            = (4, 1)
objects_named("diamond") → celdas (3,7) (3,8) (4,7) (4,8) (6,1) (7,1) (13,5) (13,6)   # 8/8
key                    = (4, 5)
door_key               = (5, 8)
rock                   = (5, 4), (11, 8)
hole                   = (8, 1), (6, 8)
no_spike               = (4, 2) (5, 1) (12, 5) (12, 6)
```

Consultas típicas:

```python
# ¿Qué hay en la celda de la roca?
state.objects_by_cell.get((5, 4), [])        # [DetectedObject(name="rock", ...)]

# Vecinos del personaje en el grafo
state.adjacency[state.player_cell]

# Celdas de todos los diamantes (objetivos de A*)
goals = [state.cell_of(d.center) for d in state.objects_named("diamond")]

# Celda destino -> píxel donde actuar
def cell_to_pixel(state, cell):
    i, j = cell
    return (state.origin_x + j * state.tile_size + state.tile_size // 2,
            state.origin_y + i * state.tile_size + state.tile_size // 2)
```

---

## 6. Cómo usarlo en A\*

```python
def neighbors_for_search(state, node, inventory):
    for nxt in state.adjacency[node]:
        if is_passable(state, nxt, inventory):   # decide el planificador
            yield nxt

def is_passable(state, cell, inventory):
    for obj in state.objects_by_cell.get(cell, []):
        if obj.name in ("lava", "spike", "hole", "rock"):
            return False
        if obj.name == "door_key" and "key" not in inventory:
            return False
    return True
```

El grafo da la **topología** (qué celdas existen y qué conecta con qué) y `objects_by_cell` da las **reglas** (qué se puede hacer en cada una). Mantenerlos separados permite que las reglas cambien sin tocar la percepción.

---

## 7. Limitaciones conocidas

1. **Alcanzable ≠ factible.** `graph_nodes` ignora los objetos. Una zona tras una puerta o una roca aparece como alcanzable aunque requiera resolver un puzzle.
2. **Objetos de varias celdas.** La lava del nivel 5 ocupa 3×2 celdas, pero el template detecta una sola. Las demás quedan como suelo sin objeto y A\* las atravesaría. Hay que detectarlas todas, por ejemplo con templates de una celda o expandiendo la detección a las celdas que cubre.
3. **Asignación por centro.** Un objeto se asigna a la celda de su centro. Si un sprite está desplazado (animación, template recortado), puede caer en la celda contigua.
4. **Personaje en movimiento.** Mientras camina entre celdas el centro puede quedar en el límite entre dos. Conviene percibir con el personaje quieto.
5. **Máscara de suelo.** `grid` depende del rango HSV de `detect_paths`. Es la primera cosa a revisar (`debug_paths.png`) si aparece suelo donde hay pared o al revés.
6. **Foto de un frame.** Tras cada acción que cambie el tablero (empujar roca, abrir puerta, recoger llave) hay que volver a percibir, o simular el cambio en el planificador.
7. **Diamante tapado por una roca.** Una roca empujada sobre un diamante lo tapa sin recolectarlo (solo el jugador recolecta al pisar la celda), pero el sprite de la roca oculta el del diamante: en una re-verificación intermedia `recognize` ya no lo detecta y el planificador cree que no quedan gemas, desbloqueando la salida antes de tiempo. En el peor caso el lazo de control lo detecta como desvío y se queda re-planificando hasta agotar `MAX_REPLANS`.
8. **Picos dinámicos.** Un pico retraído (`no_spike`) se activa **permanentemente** cuando el personaje sale de la celda: una sola cruzada por celda. El modelo lo simula en `World.spikes` (`computation/compute.py`): al salir de la celda esta pasa a `walls` y el planificador nunca vuelve a cruzarla. Tras la activación, una re-percepción la debería ver como `spike` activo (muro) y confirmar el modelo; si el sprite no matchea, el plan sigue siendo válido porque el modelo ya asumió el corte.
9. **Rejas temporales (emparejamiento 1:1).** La reja está **temporal**: abierta solo mientras el personaje o una roca cubra su botón (para atravesarla hay que dejar una roca encima; pisar y salir la vuelve a cerrar). El emparejamiento botón→reja es siempre 1:1 y la percepción no dice cuál botón abre cuál reja: por defecto se infiere por cercanía (`_match_buttons_to_gates`), pero con **varios pares** esa heurística puede fallar y conviene declararlo por nivel en `button_gate.py` (se pasa con `python main.py --level N`); lo no declarado se completa por cercanía y, si las celdas no coinciden con lo percibido, la entrada se ignora. `World.all_gates` recuerda todas las rejas para poder recerrarlas y `World.gates` guarda solo las cerradas ahora mismo; `World.pressed` es derivado de la cobertura actual (no monótono). Si el juego real difiere, el lazo cerrado lo corrige: la reja reaparece como `closed_gate` en la re-percepción y se re-planifica. Caso especial: si el nivel **ya arranca** con el personaje sobre el botón (tester que lo dejó así a mano), el botón no se detecta (tapado por su sprite) y la reja se ve abierta; `computation/gates.py` (`GateKnowledge._infer_pressed`) reconstruye ese par a partir de la reja abierta — una reja abierta implica su botón cubierto — usando el par declarado en `button_gate.py` (o, sin declaración, solo cuando es inequívoco: una reja abierta y una celda cubierta).
10. **Sokoban de dos rocas (nivel 11).** El fallback resuelve **un subobjetivo por A\*** y no compone cadenas de empujes. El nivel 11 exige llevar **dos rocas** por la columna 7: la primera rellena el hoyo (5,7) y se consume (el hoyo es de una sola casilla, así que la roca desaparece); la segunda pasa por el hueco ya relleno hasta el botón (3,7) para dejar la reja (4,6) abierta. Ningún A\* de un solo `rock_goal` arma las dos etapas dentro del presupuesto, así que el plan queda **incompleto**. Para que el fallo no degrade el nivel, el planificador aplica tres guardas: (a) el subobjetivo `boton` temporal solo acepta una **roca** encima (nunca al personaje, que al salir cierra la reja: era un falso éxito que ciclaba el fallback); (b) `a_star(..., keep_reachable=...)` poda los sucesores que activan un pico y aíslan metas **pendientes** (seguridad genérica de picos), y lo mismo hace `backtracking_search`; ojo: las metas **ya cumplidas** se excluyen (aunque `_goals_of` siga emitiendo una pista ya resuelta, p. ej. un hoyo rellenado que permanece en `world.hints`), porque su celda en `keep_reachable` podaba el único camino válido (era la causa de que el nivel 10 se bloqueara al rellenar (8,2) cruzando el pico (8,4)); (c) al botón se le pasa `rock_goal` para guiar el empuje Sokoban. Además `agent.execute` **no ejecuta** un plan incompleto: imprime `plan.pending` y se detiene sin mandar flechas, para no dejar el nivel peor (picos activados, rocas consumidas) sin resolverlo.
11. **Pistas de subobjetivos por nivel (`level_hints.py`).** Cuando la búsqueda global es lenta (nivel 6, ~80 s) o no converge (nivel 10), se puede declarar una **secuencia ordenada de subobjetivos** por nivel, en `level_hints.py`, activada con el mismo `--level N` de `button_gate.py`. El planificador los resuelve **primero**, uno por A\* (`World.hints` → `_goals_of` los emite al frente; `Planner.plan` salta la fase global si hay pistas pendientes) y luego completa el nivel con la composición automática. Descriptores: `("llave"|"puerta"|"diamante"|"hoyo", celda)` y `("roca", (origen), (destino))` — este último empuja una roca hasta `destino` (hoyo → relleno, botón → roca encima, otra celda → roca ahí) guiando con `rock_goal=destino`; los **waypoints de roca se encadenan** para partir un Sokoban. Son **preferencias, no un guion**: en cada planificación se revalidan contra lo percibido (una celda ya resuelta se ignora) y una pista ya cumplida no se reintenta, así que funcionan con el nivel a medias (el tester puede dejar al personaje en cualquier parte).
12. **Llave en la mano y doble detección.** El personaje con llave usa el sprite `player_key.png`, pero la llave pintada dentro de él **también** matchea el template de la llave suelta (`key`). Si el personaje ya cogió la llave a mano antes de que arranque el agente, esa detección `key` cae dentro de su propia caja y `world_from_state` la descarta de `world.keys` (ver `player_has_key`); sin ese filtro, el conteo `puertas − llaves` se anulaba y `World.has_key` quedaba en `False` aunque el personaje llevara la llave, dejando el plan incompleto.
