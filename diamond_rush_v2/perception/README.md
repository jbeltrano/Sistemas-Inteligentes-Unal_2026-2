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
| `button` | Se pisa para activar algo (`?`) |
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
