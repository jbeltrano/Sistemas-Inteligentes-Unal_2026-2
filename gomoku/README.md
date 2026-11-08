# JBVCAgent — Agente de Gomoku (Swap 2)

Agente de Gomoku para la variante **Swap 2**. Implementación **síncrona**, sin
paralelismo (sin Workers ni Promises), pensada para **10 s por jugada** y
tableros de tamaño variable.

Combina heurísticas de patrones con búsqueda **negamax alpha-beta/PVS**, tabla
de transposición con Zobrist, búsqueda **VCF** (victoria por cuatros
consecutivos) y control de tiempo adaptativo.

## Archivos

| Archivo | Descripción |
|---|---|
| `gomoku/JBVCAgent.js` | Implementación del agente |
| `gomoku/gomoku.js` | Interfaz `Agent`, tablero y reglas (no modificado) |
| `gomoku/index.html` | Interfaz gráfica (no modificado) |

## Uso

```html
<script src="JBVCAgent.js"></script>
```

```js
const agent = new JBVCAgent();
const move = agent.compute(board, move_state, time);
```

- `board[y][x]` contiene `' '`, `'B'` o `'W'`.
- Las jugadas se devuelven como `[x, y]`.
- `time` puede ser un número (milisegundos restantes) o un arreglo cuyo primer
  elemento son los milisegundos restantes.

## Estados de Swap 2

| `move_state` | Rol | Respuesta |
|---|---|---|
| `'1'` | Primer jugador | `[[x1,y1],[x2,y2],[x3,y3]]`: 2 negras + 1 blanca |
| `'2'` | Segundo jugador | `'BLACK'` o `[x,y]` (blanca) |
| `'3'` | Primer jugador tras la opción (iii) | `'BLACK'` o `[x,y]` (blanca) |
| `'B'` | Juego normal | `[x,y]` negra |
| `'W'` | Juego normal | `[x,y]` blanca |

La opción (iii) del estado `'2'` (poner una blanca y una negra y pasar la
elección al primer jugador) **no se usa**; ver la sección
[¿Por qué no se usa la opción (iii)?](#por-qué-no-se-usa-la-opción-iii).

## Flujo de decisión

```
compute()
  ├─ loadBoard()                     estado incremental
  ├─ '1'  → playOpening()            heurística pura
  ├─ '2'/'3' → decideSwap()          minimax superficial (~1 s)
  └─ 'B'/'W' → playNormal()
        ├─ findWinningMove(color)    ganar ahora
        ├─ findWinningMove(opp)      bloquear derrota inmediata
        ├─ vcf(color, 12)            búsqueda de cuatros forzados
        └─ iterative deepening       negamax + alpha-beta/PVS
              └─ searchRoot(depth)   hasta agotar el tiempo
  └─ validateResult()                legalidad + fallback
```

La idea no es "heurísticas primero y minimax después" de forma estricta: las
heurísticas se usan **antes** (tácticas y VCF), **durante** (generación,
ordenación, límites de ramas y extensiones) y **al final** (evaluación de las
hojas) de la búsqueda.

## Heurísticas

### Evaluación de posiciones

Se mantiene el nivel de cada línea (fila, columna y diagonales) de forma
incremental. En cada línea se calcula el mejor patrón de cada color:

| Nivel | Patrón | Peso |
|---|---|---|
| 1 | débil (1–2 piedras) | 1 |
| 2 | tres | 20 |
| 3 | tres abierto (`01110`) | 1 500 |
| 4 | cuatro | 20 000 |
| 5 | cuatro abierto (`011110`) | 500 000 |
| 6 | cinco | 5 000 000 |

Además se suman bonos por:

- doble cuatro: `+400 000`
- doble tres: `+100 000`
- cuatro + tres: `+250 000`
- tempo: `+5`

La evaluación es desde la perspectiva del jugador que mueve (negamax).

### Generación y ordenación de jugadas

- Solo se consideran celdas cercanas a piedras: radio 2 (Chebyshev) en los
  primeros ply y radio 1 en los demás.
- Cada celda se puntúa con `analyzeMove` para ataque y defensa: rachas,
  extremos abiertos y huecos.
- Se añaden bonos por: victoria del rival, cuatro propio, tres abierto propio,
  cuatro rival, tres abierto rival, jugada de la tabla de transposición,
  killer, history y cercanía al centro.
- Se ordenan y se conservan las mejores `K` según `getBranchLimit`:

| Condición | Límite |
|---|---|
| `ply == 0` | 24 |
| `ply == 1` | 16 |
| `depth >= 6` | 14 |
| `depth >= 4` | 12 |
| resto | 8 |

Si `N > 15`, el límite se reduce en 2 (mínimo 6).

### Búsqueda

- Negamax con alpha-beta y **PVS** (primera jugada con ventana completa, resto
  con ventana nula y re-búsqueda si mejora `alpha`).
- **Tabla de transposición** con Zobrist (2¹⁹ entradas): profundidad, cota,
  valor y mejor jugada.
- **Killer moves** (2 por ply) e **history heuristic**.
- **Extensiones**: `+1` ply si la jugada crea un cuatro; `+1` ply si crea un
  tres abierto.
- Chequeo de tiempo cada 256 nodos.

### VCF (Victory by Continuous Fours)

Búsqueda de victoria por cuatros consecutivos:

1. Si hay una jugada ganadora, se devuelve.
2. Se generan jugadas que crean un cuatro.
3. Si tras el cuatro hay dos puntos ganadores, es victoria forzada.
4. Si hay uno, el rival está obligado a bloquear y se recurre.
5. Presupuesto propio: hasta 1 500 ms; chequeo cada 127 nodos.

## Gestión del tiempo

El agente detecta automáticamente si `time` es tiempo **por jugada** o tiempo
**total**, comparando cuánto bajó el reloj con el tiempo que consumió la
jugada anterior.

| Modo | Presupuesto |
|---|---|
| Por jugada | `min(8200, remaining - 600)` ms |
| Tiempo total | `min(8200, remaining / max(6, 24 - moveCount))` ms |
| Desconocido | `min(2000, remaining - 600)` ms |
| Swap (`'2'`/`'3'`) | 1 000–1 500 ms |
| Apertura (`'1'`) | 250 ms |

El margen de seguridad es de 600 ms, por lo que el agente no debería superar
los 10 s por jugada.

## Apertura y swap

### Apertura (`'1'`)

Busca dos negras adyacentes cerca del centro y una blanca en el entorno 5×5,
minimizando el desbalance entre ambos colores. Es una heurística pura, sin
búsqueda; tarda del orden de 10 ms.

### Decisión de swap (`'2'` y `'3'`)

Búsqueda superficial desde la perspectiva de blancas. Si el valor de blancas
es menor que `-100`, devuelve `'BLACK'` (toma negras); si no, juega blanca.

En el estado `'3'` el agente es el primer jugador y elige entre tomar negras
(`'BLACK'`) o tomar blancas y colocar una blanca. La lógica es la misma:
comparar el valor de blancas con el umbral.

## ¿Por qué no se usa la opción (iii)?

La opción (iii) del estado `'2'` permite al segundo jugador colocar una blanca
y una negra, y luego el primer jugador elige con qué color quedarse.

En un juego de suma cero con una evaluación simétrica, esa opción está
**dominada** para el segundo jugador:

- Con las opciones (i) y (ii), el segundo jugador elige el **mejor** color y
  obtiene un valor `|V| >= 0`.
- Con la opción (iii), el primer jugador elige el mejor color y el segundo se
  queda con el **peor**, obteniendo un valor `<= 0`.

Por tanto, la opción (iii) no puede mejorar el resultado del segundo jugador.
Su utilidad en el protocolo Swap2 real es evitar líneas preparadas por el
primer jugador (aspecto psicológico/de preparación), algo que no aplica en un
bot que calcula en el momento.

El agente **sí responde correctamente si el rival usa la opción (iii)**: en el
estado `'3'` elige el color que más le conviene.

## Simetría de color

La búsqueda, la evaluación, el VCF y la generación de jugadas usan `color` y
`opp = 3 - color`; no hay lógica separada para negras o blancas. En pruebas de
simetría (posiciones con `B↔W` y turno invertido) el agente eligió la misma
celda a profundidades fijas 3–6, por lo que no tiene sesgo hacia un color.

## Rendimiento medido

| Tablero | Profundidad | Nodos en 8.2 s |
|---|---|---|
| 15×15 | 8–9 | ~540 000 |
| 19×19 | 7–8 | ~400 000–500 000 |
| 30×30 | 6 | ~230 000 |

El límite de 10 s se respeta en todos los tamaños; en tableros grandes baja la
profundidad porque se escanea `N²` celdas y hay más líneas.

## Limitaciones

- No usa la opción (iii) del estado `'2'` (ver arriba).
- La búsqueda es selectiva: no es un minimax exhaustivo, por lo que "óptimo"
  no está garantizado.
- En tableros grandes la profundidad baja.
- El umbral de swap (`-100`) es heurístico.

## Pruebas

Las pruebas de legalidad, tiempos, táctica y partidas contra un oponente
greedy se ejecutaron fuera del repositorio. Para una comprobación rápida de
sintaxis:

```bash
node --check gomoku/JBVCAgent.js
```
