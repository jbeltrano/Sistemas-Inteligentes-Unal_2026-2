"""
Planificador deliberativo del agente (sin aprendizaje).

Tipo de agente: deliberativo basado en modelos (Russell & Norvig).
Percibe UNA vez (EstadoJuego), construye una representacion del mundo
(Mundo) y resuelve el problema con busqueda en espacio de estados:

- A* (a_estrella): busca en el espacio de mundos completos. Un estado
  es la posicion del personaje + el conjunto de rocas (movibles) +
  los diamantes restantes. Cada accion mueve una casilla o empuja una
  roca (empuje estilo Sokoban / Boulder Dash).
- Retroceso (busqueda_con_retroceso): si A* no halla camino dentro de
  su presupuesto de nodos (tipicamente porque una roca bloquea el unico
  corredor y hay que empujarla lateralmente), se explora el espacio de
  estados con DFS + backtracking: se deshacen empujes que no llevan a
  ningun lado y se priorizan las acciones que empujan rocas.

Heuristica ("pensar fuera de la caja"): la heuristica mira el nivel
COMPLETO y no la casilla vecina. Es un campo de distancias (BFS
multi-fuente desde las metas) calculado sobre la grilla "pelada"
(sin rocas). Al ignorar las rocas, el campo subestima el costo real
(las rocas solo pueden entorpecer), asi que la heuristica es ADMISIBLE
y CONSISTENTE -> A* devuelve la solucion optima dentro de cada
subobjetivo; y al ser distancias reales (no Manhattan) el agente ya
"sabe" esquivar los muros del laberinto.

La planificacion se compone en subobjetivos:
  1) recolectar todos los diamantes restantes (una busqueda por cada uno)
  2) si hay una llave sin recoger, recogerla antes de la salida
  3) alcanzar la salida
"""
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import config

Celda = Tuple[int, int]

DELTA = {
    "arriba": (-1, 0),
    "abajo": (1, 0),
    "izquierda": (0, -1),
    "derecha": (0, 1),
}
ORDEN_JUGADOR = ("arriba", "abajo", "izquierda", "derecha")


@dataclass(frozen=True)
class Accion:
    """Un paso del plan: moverse o empujar una roca en una direccion."""

    tipo: str          # "mover" | "empujar"
    direccion: str     # una clave de DELTA

    def texto(self) -> str:
        verbo = "empujar roca hacia" if self.tipo == "empujar" else "mover hacia"
        return f"{verbo} {self.direccion}"


@dataclass(frozen=True)
class Mundo:
    """
    Modelo estatico del nivel para la busqueda.

    Todo son conjuntos de celdas (fila, columna). `muros` son celdas
    intransitables (pared, lava, hoyo, picos: los peligros del juego se
    modelan como obstaculos); `rocas` son celdas ocupadas por una roca
    empujable; `diamantes` son los diamantes por recolectar.
    """

    filas: int
    columnas: int
    personaje: Celda
    muros: FrozenSet[Celda] = frozenset()
    rocas: FrozenSet[Celda] = frozenset()
    diamantes: FrozenSet[Celda] = frozenset()
    llave: Optional[Celda] = None
    tiene_llave: bool = False
    salida: Optional[Celda] = None

    def en_rango(self, celda: Celda) -> bool:
        return 0 <= celda[0] < self.filas and 0 <= celda[1] < self.columnas


@dataclass
class Plan:
    """Solucion completa: la secuencia de acciones mas el mundo final."""

    acciones: List[Accion] = field(default_factory=list)
    mundo_final: Optional[Mundo] = None
    completo: bool = False
    pendientes: List[str] = field(default_factory=list)
    mundo_inicial: Optional[Mundo] = None

    @property
    def longitud(self) -> int:
        return len(self.acciones)


# ---------------------------------------------------------------
# Semantica de los objetos detectados -> rol en el mundo
# ---------------------------------------------------------------
# Mapeo nombre logico del objeto -> categoría. "por calibrar": los
# templates pueden cambiar por nivel, y el rol exacto de reja/pico/
# roca_en_hoyo es la parte mas especulativa.
ROLES = {
    "diamante": "objetivo",
    "llave": "llave",
    "salida_bloqueada": "salida",
    "salida_abierta": "salida",
    "puerta_llave": "salida",
    "roca": "roca",
    "roca_en_hoyo": "inmovil",      # roca que ya tapa un hoyo: fija (por calibrar)
    "reja": "inmovil",              # reja/barras: bloquea (por calibrar)
    "lava": "peligro",
    "hueco": "peligro",
    "pico": "peligro",
    "totem": "peligro",
    "interruptor": "piso",
    "piso_seguro": "piso",
    "personaje": "jugador",
}
_ROL_A_MUNDO = {
    "objetivo": ("diamantes",),
    "llave": ("llave",),
    "salida": ("salida",),
    "roca": ("rocas",),
    "inmovil": ("muros",),
    "peligro": ("muros",),
}


def mundo_desde_estado(estado, roles: Optional[Dict[str, str]] = None) -> Mundo:
    """
    Traduce el EstadoJuego percibido (grid + detecciones) al Mundo de
    busqueda. Los peligros del juego (lava, hoyo, picos...) se tratan
    como muros por defecto: el agente no entra a casillas letales.
    """
    roles = roles or ROLES
    grilla = estado.grid.tolist()
    filas, columnas = len(grilla), (len(grilla[0]) if grilla else 0)

    muros = {(i, j) for i in range(filas) for j in range(columnas) if not grilla[i][j]}
    rocas: Set[Celda] = set()
    diamantes: Set[Celda] = set()
    llave = None
    salida = None

    # Cada deteccion agrega su rol al mundo (una celda puede tener
    # varios objetos: el primero manda sobre la transitabilidad).
    for obj in estado.objects:
        rol = roles.get(obj.nombre, "piso")
        if rol in _ROL_A_MUNDO:
            celda = estado.celda_de(obj.centro)
            destinos = _ROL_A_MUNDO[rol]
            if "muros" in destinos:
                muros.add(celda)
            if "rocas" in destinos:
                rocas.add(celda)
            if "diamantes" in destinos:
                diamantes.add(celda)
            if rol == "llave":
                llave = celda
            if rol == "salida":
                salida = celda

    return Mundo(
        filas=filas,
        columnas=columnas,
        muros=frozenset(muros),
        rocas=frozenset(rocas),
        personaje=estado.player_cell,
        diamantes=frozenset(diamantes),
        llave=llave,
        tiene_llave=False,
        salida=salida,
    )


# ---------------------------------------------------------------
# Sucesores
# ---------------------------------------------------------------
def _paso(mundo: Mundo, direccion: str) -> Optional[Tuple[Accion, Mundo]]:
    """
    Aplica un movimiento o un empuje desde el estado actual.
    Devuelve (accion, mundo) o None si la celda destino esta bloqueada.
    """
    d = DELTA[direccion]
    origen = mundo.personaje
    destino = (origen[0] + d[0], origen[1] + d[1])

    if not mundo.en_rango(destino) or destino in mundo.muros:
        return None

    if destino in mundo.rocas:
        # Empujar la roca a la celda de mas alla.
        empuje = (destino[0] + d[0], destino[1] + d[1])
        if (
            not mundo.en_rango(empuje)
            or empuje in mundo.muros
            or empuje in mundo.rocas
            # No empujar una roca sobre recolectables ni sobre la salida.
            or empuje in mundo.diamantes
            or empuje == mundo.llave
            or empuje == mundo.salida
        ):
            return None
        rocas = frozenset((mundo.rocas - {destino}) | {empuje})
        personaje = destino
        tipo = "empujar"
    else:
        rocas = mundo.rocas
        personaje = destino
        tipo = "mover"

    nuevo = Mundo(
        filas=mundo.filas,
        columnas=mundo.columnas,
        muros=mundo.muros,
        rocas=rocas,
        personaje=personaje,
        diamantes=mundo.diamantes,
        llave=mundo.llave,
        tiene_llave=mundo.tiene_llave,
        salida=mundo.salida,
    )
    return Accion(tipo, direccion), nuevo


def _recolectar(mundo: Mundo) -> Mundo:
    """Client ultima transicion: si el personaje piso un diamante o la llave."""
    diamantes = mundo.diamantes - {mundo.personaje}
    llave = mundo.llave
    tiene_llave = mundo.tiene_llave
    if mundo.personaje == mundo.llave:
        llave = None
        tiene_llave = True
    if diamantes == mundo.diamantes and llave == mundo.llave:
        return mundo
    return Mundo(
        filas=mundo.filas,
        columnas=mundo.columnas,
        muros=mundo.muros,
        rocas=mundo.rocas,
        personaje=mundo.personaje,
        diamantes=frozenset(diamantes),
        llave=llave,
        tiene_llave=tiene_llave,
        salida=mundo.salida,
    )


def _sucesores(mundo: Mundo):
    """Todos los (accion, mundo_nuevo) validos desde `mundo`."""
    for direccion in ORDEN_JUGADOR:
        resultado = _paso(mundo, direccion)
        if resultado is None:
            continue
        accion, nuevo = resultado
        yield accion, _recolectar(nuevo)


# ---------------------------------------------------------------
# Heuristica: campo de distancias multi-fuente
# ---------------------------------------------------------------
def campo_distancias(mundo: Mundo, metas: Iterable[Celda]) -> Dict[Celda, int]:
    """
    BFS multi-fuente desde las metas sobre la grilla "pelada" (sin
    rocas). Distancia real en casillas para cada celda alcanzable;
    no Manhattan: ya toma en cuenta los muros del laberinto.
    """
    dist: Dict[Celda, int] = {}
    cola = deque()
    for meta in metas:
        if mundo.en_rango(meta) and meta not in mundo.muros:
            dist.setdefault(meta, 0)
            cola.append((meta, 0))

    while cola:
        celda, d = cola.popleft()
        for df, dc in DELTA.values():
            vecina = (celda[0] + df, celda[1] + dc)
            if (
                mundo.en_rango(vecina)
                and vecina not in mundo.muros
                and vecina not in dist
            ):
                dist[vecina] = d + 1
                cola.append((vecina, d + 1))
    return dist


def heuristica(mundo: Mundo, metas: FrozenSet[Celda]) -> float:
    """
    Distancia del personaje a la meta mas cercana segun el campo de
    distancias. Ignora las rocas a proposito: asumir que el pasaje
    siempre esta libre subestima el costo real (admisible) y cumple la
    desigualdad triangular (consistente), con lo que A* conserva su
    garantia de optimalidad dentro del subobjetivo.
    """
    campo = campo_distancias(mundo, metas)
    return campo.get(mundo.personaje, float("inf"))


# ---------------------------------------------------------------
# A*
# ---------------------------------------------------------------
def a_estrella(
    mundo: Mundo,
    metas: FrozenSet[Celda],
    max_nodos: int = config.MAX_NODOS_A,
) -> Optional[Tuple[List[Accion], Mundo]]:
    """
    A* sobre el espacio de mundos completos. Devuelve (acciones,
    mundo_final) o None si no encuentra solucion dentro del
    presupuesto de nodos. Mueve, empuja rocas y queda parado sobre la
    meta (recolectandola si es diamante/llave).
    """
    if mundo.personaje in metas:
        return [], _recolectar(mundo)

    campo = campo_distancias(mundo, metas)
    if mundo.personaje not in campo:
        return None

    abiertos: List[Tuple[float, float, int, Mundo, Tuple[Accion, ...]]] = []
    heapq.heappush(abiertos, (campo[mundo.personaje], 0.0, 0, mundo, ()))
    visitados: Set[Tuple[Celda, FrozenSet[Celda]]] = set()
    contador = 1
    nodos = 0

    while abiertos and nodos < max_nodos:
        _f, g, _c, mundo_act, plan = heapq.heappop(abiertos)
        clave = (mundo_act.personaje, mundo_act.rocas)
        if clave in visitados:
            continue
        visitados.add(clave)
        nodos += 1

        if mundo_act.personaje in metas:
            return list(plan), mundo_act

        for accion, sucesor in _sucesores(mundo_act):
            camino = (sucesor.personaje, sucesor.rocas)
            if camino in visitados:
                continue
            h = campo.get(sucesor.personaje, float("inf"))
            if h == float("inf"):
                continue
            contador += 1
            heapq.heappush(
                abiertos,
                (g + 1 + h, h, contador, sucesor, plan + (accion,)),
            )

    return None


# ---------------------------------------------------------------
# Retroceso (DFS con backtracking)
# ---------------------------------------------------------------
def busqueda_con_retroceso(
    mundo: Mundo,
    metas: FrozenSet[Celda],
    max_nodos: int = config.MAX_NODOS_RETROCESO,
    max_prof: int = config.MAX_PROFUNDIDAD,
) -> Optional[Tuple[List[Accion], Mundo]]:
    """
    DFS con backtracking sobre el espacio de mundos. Prioriza las
    acciones que empujan rocas (para desbloquear corredores) y se
    deshace de ramas sin salida con `visitados` + tope de profundidad.
    """
    if mundo.personaje in metas:
        return [], _recolectar(mundo)

    visitados: Set[Tuple[Celda, FrozenSet[Celda], FrozenSet[Celda], bool]] = set()
    pila: List[Tuple[Mundo, Tuple[Accion, ...], int]] = [(mundo, (), 0)]
    nodos = 0

    while pila and nodos < max_nodos:
        m, plan, prof = pila.pop()
        clave = (m.personaje, m.rocas, m.diamantes, m.tiene_llave)
        if clave in visitados:
            continue
        visitados.add(clave)
        nodos += 1
        if prof >= max_prof:
            continue
        if m.personaje in metas:
            return list(plan), m

        # Empujes primero: es lo que desbloquea el camino bloqueado.
        sucesores = sorted(_sucesores(m), key=lambda s: s[0].tipo != "empujar")
        for accion, sucesor in reversed(sucesores):
            skey = (sucesor.personaje, sucesor.rocas, sucesor.diamantes, sucesor.tiene_llave)
            if skey in visitados:
                continue
            pila.append((sucesor, plan + (accion,), prof + 1))

    return None


# ---------------------------------------------------------------
# Planificador de alto nivel (subobjetivos)
# ---------------------------------------------------------------
def _metas_de(mundo: Mundo):
    """
    Secuencia de subobjetivos segun el contenido del nivel. Cada
    diamante es un subobjetivo individual: si uno queda inalcanzable
    (p. ej. detras de lava) se marca pendiente sin perder la
    planificacion del resto, la llave y la salida.
    """
    for celda in sorted(mundo.diamantes):
        yield f"diamante {celda}", frozenset({celda})
    if mundo.llave is not None and not mundo.tiene_llave:
        yield "llave", frozenset({mundo.llave})
    if mundo.salida is not None:
        yield "salida", frozenset({mundo.salida})


@dataclass
class Planificador:
    """
    Orquesta los subobjetivos. Para cada meta intenta A* y, si se
    queda sin presupuesto o no hay camino, cae en la busqueda con
    retroceso. Tras cada etapa se recalculan las metas sobre el mundo
    avanzado (diamantes recolectados se van quitando de la lista).

    Devuelve un Plan (posiblemente incompleto si queda algo
    inalcanzable, p. ej. un diamante detras de lava).
    """

    max_nodos_a: int = 150_000
    max_nodos_retroceso: int = 80_000
    max_profundidad: int = 600

    def planificar(self, mundo: Mundo) -> Plan:
        pendientes: List[str] = []
        acciones: List[Accion] = []
        # Metas que ya fracasaron (inalcanzables con el modelo): no se
        # vuelven a intentar en cada re-encolado.
        fracasadas: Set[FrozenSet[Celda]] = set()
        m = mundo

        def encolar() -> deque:
            # Metas pendientes del mundo actual, SIN las que el personaje
            # ya esta pisando (evita re-encolar la misma meta si un paso
            # no cambio nada, p. ej. empezar parado sobre la salida) ni
            # las que ya fracasaron.
            return deque(
                (nombre, metas)
                for nombre, metas in _metas_de(m)
                if m.personaje not in metas and metas not in fracasadas
            )

        cola = encolar()
        while cola:
            nombre, metas = cola.popleft()
            resultado = a_estrella(m, metas, self.max_nodos_a)
            if resultado is None:
                resultado = busqueda_con_retroceso(
                    m, metas, self.max_nodos_retroceso, self.max_profundidad
                )
            if resultado is None:
                pendientes.append(nombre)
                fracasadas.add(frozenset(metas))
                continue  # pasar a la siguiente meta; el mundo no cambia
            pasos, m = resultado
            acciones.extend(pasos)
            # El mundo avanzo (p. ej. se recolecto un diamante):
            # recomputar las metas restantes sobre el estado nuevo.
            cola = encolar()

        return Plan(
            acciones=acciones,
            mundo_final=m,
            completo=not pendientes,
            pendientes=pendientes,
            mundo_inicial=mundo,
        )


# ---------------------------------------------------------------
# Simulacion (para tests)
# ---------------------------------------------------------------
def simular(mundo: Mundo, acciones: Iterable[Accion]) -> Mundo:
    """Reproduce el plan sobre el mundo inicial; valida cada paso."""
    m = mundo
    for accion in acciones:
        resultado = _paso(m, accion.direccion)
        if resultado is None or resultado[0] != accion:
            raise ValueError(f"Paso invalido en el plan: {accion}")
        m = _recolectar(resultado[1])
    return m