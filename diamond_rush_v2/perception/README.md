# `perception/`

Convierte un frame crudo del juego (una imagen) en un `GameState`: la
información ya estructurada que el resto del agente necesita para
decidir qué hacer. Es la única parte del proyecto que sabe de OpenCV,
templates y umbrales — todo lo que está fuera de esta carpeta debería
poder ignorar por completo cómo se percibe el juego.

```
frame (imagen cruda) → recognize() → GameState → computation/ → play/
```

## Archivos

### `state.py` — el contrato

Define los dos tipos de datos que salen de la percepción:

- **`DetectedObject`**: una detección individual (un diamante, una
  roca, la llave...). Tiene nombre, posición, tamaño, y una propiedad
  `.center` ya calculada.
- **`GameState`**: el estado completo de un frame — la grilla de
  celdas transitables (`grid`), el grafo ya filtrado a lo alcanzable
  desde el personaje (`graph_nodes`, `graph_edges`), la celda del
  personaje (`player_cell`), todas las detecciones (`objects`), y los
  datos para convertir entre píxel y celda (`origin_x`, `origin_y`,
  `tile_size`). También trae `objects_named(name)`, un atajo para
  filtrar detecciones por tipo (ej. `state.objects_named("diamond")`).

Ningún otro módulo del proyecto (`computation/`, `play/`) debería
importar nada de `perception/` que no sea esto. `recognize()` es la
única función que arma un `GameState`, y `state.py` es la única
definición de qué campos tiene.

### `recognition.py` — el pipeline

Contiene todo el reconocimiento en sí, organizado en etapas:

1. **`TemplateMatcher`**: busca un template en el frame con
   `matchTemplate` y devuelve sus coincidencias como `(x, y, score)`,
   ya sin duplicados del mismo template encima del mismo objeto.
2. **`_resolve_overlaps`**: junta las coincidencias de *todos* los
   templates y descarta las que se superponen fuertemente con otra de
   mejor score, sin importar el nombre. Resuelve los casos donde dos
   templates distintos representan estados alternativos del mismo
   objeto físico (`rock`/`rock_in`, por ejemplo).
3. **`detect_paths`**: máscara binaria de suelo/camino por color (HSV).
4. **`detect_tile_size`** / **`create_grid`**: estiman el tamaño de
   celda y arman la grilla 0/1 de transitable, anclada al pixel
   `(0, 0)` del frame (ver nota abajo).
5. **`build_graph`** / **`reachable_component`**: arman el grafo de
   celdas transitables y lo filtran a lo alcanzable desde el
   personaje.
6. **`recognize(frame)`**: el único punto de entrada público. Corre
   todo lo anterior y devuelve un `GameState`.

Detalles de comportamiento que conviene tener presentes:

- El **HUD se descarta por coordenada**: cualquier detección cuyo
  centro quede por encima de `config.HUD_HEIGHT` se filtra, porque el
  contador de diamantes del HUD matchea el template de "diamond"
  igual que uno real.
- `origin_x`/`origin_y` siempre valen `(0, 0)` hoy — la grilla asume
  que la captura ya viene recortada justo al borde del tablero. Si
  algún día la captura deja un margen antes del primer tile, esto
  hay que revisarlo (implicaría detectar la fase real del tablero,
  no asumirla).
- Todas las constantes (umbrales, rango de tile, nombres de
  templates, `HUD_HEIGHT`) viven en `config.py`, no acá.

### `capture.py` — pendiente

Todavía no lo discutimos en detalle (eso es lo que sigue). Cuando lo
definamos, esta sección debería explicar cómo se obtiene el frame que
alimenta a `recognize()` y qué garantías da sobre su contenido (por
ejemplo: resolución fija, siempre recortado al tablero, sin ventanas
tapando encima).

## Cómo trabajar con `state.py` de acá en adelante

- **`state.py` describe qué se vio, no qué significa.** Si
  `computation/` necesita un dato derivado — por ejemplo, la
  distancia al diamante más cercano — esa cuenta va en
  `computation/`, no como un método nuevo de `GameState`. Mezclar
  lógica de decisión ahí rompería la separación que buscamos.
- **Todo campo nuevo se agrega en pareja**: si `recognize()` necesita
  producir un dato que hoy no existe en `GameState`, se agrega el
  campo acá y se lo llena allá en el mismo cambio — son dos mitades
  de un mismo contrato, no deberían quedar desincronizadas.
- **`computation/` y `play/` no importan `cv2`, `numpy` (salvo para
  tipos, ej. `np.ndarray` de `grid`) ni nada de `recognition.py`
  directamente.** Si alguna vez sienten la tentación de hacerlo, es
  señal de que falta un campo en `GameState` — se agrega el campo en
  vez de perforar el límite.
- Antes de agregar un campo, preguntarse: *¿esto es algo que la
  percepción observó, o es una conclusión que alguien saca a partir
  de lo observado?* Lo primero va en `GameState`; lo segundo, en
  quien lo necesite.
