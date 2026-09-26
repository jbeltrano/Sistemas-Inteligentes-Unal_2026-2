import cv2
from pathlib import Path
from random import random
import numpy as np
from collections import deque

class MatchTemplate:
    
    def __init__(self, template_path, imagen, reflejar=False):
        self.imagen = imagen
        self.template_path = template_path
        self.reflejar = reflejar
        self.nombre = Path(template_path).stem
        
        self.template_image = cv2.imread(str(self.template_path))
        self.check_images_exist(self.template_image, self.template_path)
        
        self.get_template_size()
        
        
    @staticmethod
    def check_images_exist(imagen, path):
        if imagen is None:
            raise FileNotFoundError(
                f"No se pudo cargar la imagen: {path}"
            )
    
    def get_template_size(self):
        h, w = self.template_image.shape[:2]
        print(f"Template: {w}x{h}")
        return h, w
    
    def match_template(self, threshold=0.80, solape_max=None):
        if solape_max is None:
            solape_max = SOLAPE_MAX
        resultado = cv2.matchTemplate(
            self.imagen,
            self.template_image,
            cv2.TM_CCOEFF_NORMED
        )
        
        # Si el objeto puede estar girado, probar también el template reflejado
        # y quedarse con el mejor score de ambas orientaciones
        if self.reflejar:
            resultado_flip = cv2.matchTemplate(
                self.imagen,
                cv2.flip(self.template_image, 1),
                cv2.TM_CCOEFF_NORMED
            )
            resultado = np.maximum(resultado, resultado_flip)
        
        ys, xs = (resultado >= threshold).nonzero()
        scores = resultado[ys, xs]
        h, w = self.get_template_size()
        
        # NMS: quedarse con una sola caja por objeto
        # (matchTemplate deja una "meseta" de píxeles sobre el umbral alrededor de cada coincidencia)
        candidatos = sorted(
            zip(xs, ys, scores),
            key=lambda p: p[2],
            reverse=True
        )
        
        elegidos = []
        for x, y, _ in candidatos:
            if all(
                self._sobrelape(x, y, ex, ey, w, h) <= solape_max
                for ex, ey in elegidos
            ):
                elegidos.append((x, y))
        
        xs, ys = zip(*elegidos) if elegidos else ([], [])
        print(f"Coincidencias encontradas: {len(elegidos)}")
        
        return list(xs), list(ys)
    
    def _sobrelape(self, x, y, dx, dy, w, h):
        ancho = max(0, min(x + w, dx + w) - max(x, dx))
        alto  = max(0, min(y + h, dy + h) - max(y, dy))
        return (ancho * alto) / (w * h)


def detectar_caminos(imagen):
    """
    Detecta las zonas de suelo/camino del nivel.
    Devuelve una máscara binaria:
        blanco = camino
        negro = obstáculo
    """

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # El suelo del juego es marrón oscuro.
    #
    # Estos valores son un punto de partida.
    # Conviene ajustarlos viendo la máscara resultante.
    lower = np.array([5, 40, 20])
    upper = np.array([30, 220, 150])

    mask = cv2.inRange(hsv, lower, upper)

    # Limpiar pequeños agujeros/ruido
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask

def _fraccion_camino(mask, x, y, tile):
    """
    Proporción de suelo de un tile. Si el tile se sale de la imagen
    devuelve 0.0 para que no cuente como transitable.
    """
    alto, ancho = mask.shape[:2]

    if x < 0 or y < 0 or x + tile > ancho or y + tile > alto:
        return 0.0

    return float(np.mean(mask[y:y + tile, x:x + tile] > 0))


def _origen_celda(centro, tile):
    """
    Esquina superior izquierda del tile que contiene al personaje.
    El centro del personaje queda a media celda de esa esquina.
    """
    cx, cy = centro

    return (
        int(round((cx - tile / 2) / tile)) * tile,
        int(round((cy - tile / 2) / tile)) * tile
    )


def _tiles_around(centro, tile, alcance, ancho, alto):
    """
    Tiles completos alrededor del personaje. Los que se salen de la
    imagen se descartan.
    """
    x0, y0 = _origen_celda(centro, tile)
    celdas = []

    for i in range(-alcance, alcance + 1):
        for j in range(-alcance, alcance + 1):
            x = x0 + j * tile
            y = y0 + i * tile

            if x >= 0 and y >= 0 and x + tile <= ancho and y + tile <= alto:
                celdas.append((x, y))

    return celdas


def detectar_paso(mask, centro, paso_min=None, paso_max=None,
                  paso_defecto=None, alcance=None):
    """
    Estima el paso de la retícula de tiles probando cada tamaño
    candidato y puntúandolo por qué proporción de los tiles cercanos
    al personaje queda en zona ambigua (ni suelo ni vacío). Un paso
    bien alineado deja esa proporción cerca de cero; un paso
    desplazado hace que los tiles caigan sobre los bordes del suelo y
    las fracciones queden a mitad de camino.
    """
    paso_min = PASO_MIN if paso_min is None else paso_min
    paso_max = PASO_MAX if paso_max is None else paso_max
    paso_defecto = PASO_POR_DEFECTO if paso_defecto is None else paso_defecto
    alcance = RANGO_ANALISIS if alcance is None else alcance

    alto, ancho = mask.shape[:2]

    mejor_paso = None
    mejor_puntaje = None

    for paso in range(paso_min, paso_max + 1):
        celdas = _tiles_around(centro, paso, alcance, ancho, alto)

        if not celdas:
            continue

        puntaje = float(np.mean([
            0.25 < _fraccion_camino(mask, x, y, paso) < 0.75
            for x, y in celdas
        ]))

        if mejor_puntaje is None or puntaje < mejor_puntaje:
            mejor_paso = paso
            mejor_puntaje = puntaje

    if mejor_paso is None:
        return paso_defecto, 1.0

    if mejor_puntaje > UMBRAL_AMBIGUEDAD:
        print(f"Atención: el paso no es confiable"
              f" ({mejor_puntaje:.0%} de celdas ambiguas),"
              f" se usa {paso_defecto}")
        mejor_paso = paso_defecto

    print(f"Paso de la retícula: {mejor_paso} px"
          f" ({mejor_puntaje:.0%} de celdas ambiguas)")

    return mejor_paso, mejor_puntaje


def crear_grid(mask, centro, tile, fraccion_minima=None):
    """
    Convierte la máscara de caminos en una grilla de tiles de lado
    `tile`, alineada con el tile donde está el personaje.

    1 = transitable
    0 = obstáculo

    La grilla cubre toda la imagen, así que el personaje puede caer en
    cualquier fila. Devuelve (grid, x_ini, y_ini), donde (x_ini, y_ini)
    es la esquina superior izquierda de la celda grid[0, 0].
    """
    fraccion_minima = FRACCION_MINIMA if fraccion_minima is None else fraccion_minima

    alto, ancho = mask.shape[:2]
    x_personaje, y_personaje = _origen_celda(centro, tile)

    # La retícula se extiende en todas las direcciones desde el
    # personaje, así que el origen de la grilla puede quedar en
    # negativo respecto de él.
    x_ini = x_personaje + int(np.floor(-x_personaje / tile)) * tile
    y_ini = y_personaje + int(np.floor(-y_personaje / tile)) * tile
    filas = int(np.ceil((alto - y_ini) / tile))
    columnas = int(np.ceil((ancho - x_ini) / tile))

    grid = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(filas):
        for j in range(columnas):

            x = x_ini + j * tile
            y = y_ini + i * tile

            # Si más de la mitad de la casilla
            # corresponde al suelo, consideramos que se puede caminar.
            if _fraccion_camino(mask, x, y, tile) > fraccion_minima:
                grid[i, j] = 1

    return grid, x_ini, y_ini


def celda_de(centro, x_ini, y_ini, tile):
    """
    Fila y columna de la celda que contiene un punto de la imagen.
    """
    return (
        int((centro[1] - y_ini) // tile),
        int((centro[0] - x_ini) // tile)
    )


def centro_celda(celda, x_ini, y_ini, tile):
    """
    Punto medio de una celda, en píxeles de la imagen.
    """
    i, j = celda

    return (
        x_ini + j * tile + tile // 2,
        y_ini + i * tile + tile // 2
    )


def vecinos(celda):
    """
    Las cuatro casillas vecinas: arriba, abajo, izquierda y derecha.
    """
    i, j = celda

    return ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))


def construir_grafo(grid):
    """
    Arma el grafo de las casillas transitablees. Cada nodo es una
    celda (i, j) de la grilla y las aristas unen solo vecinos de
    cuatro direcciones, así que no se pueden cortar esquinas por el
    aire.
    """
    filas, columnas = grid.shape

    nodos = {
        (i, j)
        for i in range(filas)
        for j in range(columnas)
        if grid[i, j]
    }

    aristas = set()

    for nodo in nodos:
        for vecino in vecinos(nodo):
            if vecino in nodos:
                aristas.add(tuple(sorted((nodo, vecino))))

    return nodos, aristas


def componente(nodos, origen):
    """
    Casillas alcanzables desde `origen` moviéndose por los vecinos de
    cuatro direcciones, ya sea de forma directa o indirecta. Sirve
    para tirar las casillas sueltas que el jugador nunca va a pisar.
    """
    if origen not in nodos:
        return set()

    alcanzables = {origen}
    cola = deque([origen])

    while cola:
        for vecino in vecinos(cola.popleft()):
            if vecino in nodos and vecino not in alcanzables:
                alcanzables.add(vecino)
                cola.append(vecino)

    return alcanzables


def pintar_grafo(imagen, grid, x_ini, y_ini, tile, nodos, aristas,
                 celda_personaje, objetos, camino_salida):
    """
    Dibuja el grafo sobre una copia atenuada de la captura: casillas
    transitablees en verde, bloqueadas en rojo, aristas entre los
    centros de las casillas conectadas, las coordenadas de cada
    casilla, la del personaje resaltada y los objetos detectados
    encima.
    """
    fondo = cv2.addWeighted(imagen, 0.35, np.zeros_like(imagen), 0.65, 0)

    filas, columnas = grid.shape

    for i in range(filas):
        for j in range(columnas):
            x = x_ini + j * tile
            y = y_ini + i * tile
            color = (0, 200, 0) if (i, j) in nodos else (0, 0, 200)

            cv2.rectangle(
                fondo,
                (x, y),
                (x + tile, y + tile),
                color,
                1
            )

    for a, b in aristas:
        cv2.line(
            fondo,
            centro_celda(a, x_ini, y_ini, tile),
            centro_celda(b, x_ini, y_ini, tile),
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    for i in range(filas):
        for j in range(columnas):
            if (i, j) not in nodos:
                continue

            x = x_ini + j * tile
            y = y_ini + i * tile

            cv2.putText(
                fondo,
                f"({i},{j})",
                (x + 4, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    for nombre, xs, ys, h, w, r, g, b in objetos:
        for x, y in zip(xs, ys):

            x = int(x)
            y = int(y)

            cv2.rectangle(
                fondo,
                (x, y),
                (x + w, y + h),
                (r, g, b),
                2
            )

            cv2.putText(
                fondo,
                f"{nombre} ({x},{y})",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (r, g, b),
                1,
                cv2.LINE_AA
            )

    # El recuadro del personaje cae casi encima de su propia celda, así
    # que el resaltado va al final para que no quede tapado.
    x = x_ini + celda_personaje[1] * tile
    y = y_ini + celda_personaje[0] * tile

    cv2.rectangle(
        fondo,
        (x, y),
        (x + tile, y + tile),
        (0, 255, 255),
        2
    )

    cv2.circle(
        fondo,
        centro_celda(celda_personaje, x_ini, y_ini, tile),
        tile // 3,
        (0, 255, 255),
        2
    )

    cv2.putText(
        fondo,
        "PERSONAJE",
        (x + 4, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.imwrite(str(camino_salida), fondo)
    print(f"Grafo guardado en: {camino_salida}")

# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

imagen_path = BASE_DIR / "captura.png"
# imagen_path = BASE_DIR / "levels" / "level19.png"
template_path = BASE_DIR / "templates" / "personaje.png"
personaje_path = BASE_DIR / "templates" / "personaje.png"
diamond_path = BASE_DIR / "templates" / "diamond.png"
door_key_path = BASE_DIR / "templates" / "door_key.png"
hueco_path = BASE_DIR / "templates" / "hueco.png"
key_path = BASE_DIR / "templates" / "key.png"
pic_path = BASE_DIR / "templates" / "pic.png"
rock_in_path = BASE_DIR / "templates" / "rock_in.png"
rock_path = BASE_DIR / "templates" / "rock.png"
scape_block_path = BASE_DIR / "templates" / "scape_block.png"
space_enable_path = BASE_DIR / "templates" / "scape_enable.png"
no_picos_path = BASE_DIR / "templates" / "no_picos.png"
button_path = BASE_DIR / "templates" / "button.png"
lava_path = BASE_DIR / "templates" / "lava.png"
reja_path = BASE_DIR / "templates" / "reja.png"

UMBRAL = 0.90
# UMBRAL = 0.96
# Fracción máxima del área del template que dos coincidencias pueden compartir
# antes de considerarse la misma meseta (0 = nada, 1 = todo)
SOLAPE_MAX = 0.4
# Tamaños de tile (en píxeles) que se prueban al buscar la retícula
PASO_MIN = 50
PASO_MAX = 70
PASO_POR_DEFECTO = 60
# Cuántos tiles a la redonda del personaje se usan para medir la retícula
RANGO_ANALISIS = 8
# Si más de esta proporción de casillas queda ambigua, el paso no sirve
UMBRAL_AMBIGUEDAD = 0.15
# Proporción de suelo mínima para que una casilla sea transitable
FRACCION_MINIMA = 0.5
PERSONAJE = "personaje"
grafo_path = BASE_DIR / "grafo.png"

imagen = cv2.imread(str(imagen_path))
MatchTemplate.check_images_exist(imagen, imagen_path)

objects = [
    MatchTemplate(personaje_path, imagen, reflejar=True), 
    MatchTemplate(diamond_path, imagen), 
    MatchTemplate(door_key_path, imagen), 
    MatchTemplate(hueco_path, imagen), 
    MatchTemplate(key_path, imagen), 
    MatchTemplate(pic_path, imagen), 
    MatchTemplate(rock_in_path, imagen), 
    MatchTemplate(rock_path, imagen), 
    MatchTemplate(scape_block_path, imagen), 
    MatchTemplate(space_enable_path, imagen),
    MatchTemplate(no_picos_path, imagen),
    MatchTemplate(button_path, imagen),
    MatchTemplate(lava_path, imagen),
    MatchTemplate(reja_path, imagen)
    ]


# Obteiene los resultados 
data = []
for obj in objects:
    
    xs, ys = obj.match_template(threshold=UMBRAL)
    h, w = obj.get_template_size()
    
    r = int(random() * 255)
    g = int(random() * 255)
    b = int(random() * 255)
    data.append((obj.nombre,xs,ys,h,w,r,g,b))

caminos = detectar_caminos(imagen)




cv2.imwrite(
    str(BASE_DIR / "caminos.png"),
    caminos
)


# ==========================================
# GRAFO DE CAMINOS
# ==========================================

# El primer match del personaje es el de mayor puntaje, y viene en la
# posición 0 de la lista porque match_template ordena por score.
centro = None

for nombre, xs, ys, h, w, r, g, b in data:
    if nombre == PERSONAJE and xs:
        centro = (int(xs[0]) + w // 2, int(ys[0]) + h // 2)
        break

if centro is None:
    print(f"No se encontró el personaje, no se puede armar el grafo")

else:

    paso, puntaje = detectar_paso(caminos, centro)

    grid, x_ini, y_ini = crear_grid(caminos, centro, paso)

    celda_personaje = celda_de(centro, x_ini, y_ini, paso)
    filas, columnas = grid.shape

    if 0 <= celda_personaje[0] < filas and 0 <= celda_personaje[1] < columnas:
        # El sprite tapa su propio tile, así que la casilla del
        # personaje nunca pasa el filtro de suelo.
        grid[celda_personaje] = 1

    nodos, aristas = construir_grafo(grid)

    # Al jugador solo le sirve la parte del grafo que puede alcanzar
    # caminando desde donde está.
    alcanzables = componente(nodos, celda_personaje)
    aristas = {
        arista
        for arista in aristas
        if arista[0] in alcanzables and arista[1] in alcanzables
    }

    print(f"Personaje en la celda {celda_personaje} de"
          f" {filas}x{columnas}")
    print(f"Casillas transitablees: {int(grid.sum())}")
    print(f"Nodos del grafo: {len(alcanzables)}")
    print(f"Aristas del grafo: {len(aristas)}")

    pintar_grafo(
        imagen,
        grid,
        x_ini,
        y_ini,
        paso,
        alcanzables,
        aristas,
        celda_personaje,
        data,
        grafo_path
    )


# Dibuja los resultados en la imagen
for nombre, xs, ys, h, w, r, g, b in data:
    for x, y in zip(xs, ys):

        x = int(x)
        y = int(y)

        cv2.rectangle(
            imagen,
            (x, y),
            (x + w, y + h),
            (r, g, b),
            2
        )

        cv2.putText(
            imagen,
            f"({x}, {y})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (r, g, b),
            1
        )


resultado_path = BASE_DIR / "resultado.png"

cv2.imwrite(str(resultado_path), imagen)

print(f"Resultado guardado en:")
print(resultado_path)
