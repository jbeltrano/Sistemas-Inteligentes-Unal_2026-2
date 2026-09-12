import numpy as np
from itertools import permutations
from collections import defaultdict
from math import log2


# ============================================================
# CONFIGURACIÓN
# ============================================================

DIGITOS = "0123456789"
LONGITUD = 4

# Generamos los 5040 números posibles
NUMEROS = [
    ''.join(p)
    for p in permutations(DIGITOS, LONGITUD)
]

N = len(NUMEROS)

print(f"Se generaron {N} números posibles.")


# ============================================================
# REPRESENTACIÓN NUMÉRICA
# ============================================================

# Convertimos cada número a una matriz:
#
# 0123 -> [0, 1, 2, 3]
# 5831 -> [5, 8, 3, 1]

DIGITOS_NUM = np.array(
    [[int(d) for d in numero] for numero in NUMEROS],
    dtype=np.int8
)


# ============================================================
# PRECALCULAR RESULTADOS
# ============================================================

print("Precalculando tabla de resultados...")

# codificación:
#
# resultado = picas * 5 + fijas
#
# Ejemplos:
#
# 0 picas, 0 fijas -> 0
# 1 pica, 0 fijas  -> 5
# 0 picas, 1 fija  -> 1
# 2 picas, 1 fija  -> 11
# 0 picas, 4 fijas -> 4

RESULTADOS = np.zeros((N, N), dtype=np.uint8)


def calcular_resultados_para_intento(intento):

    """
    Calcula simultáneamente el resultado del intento
    contra todos los posibles secretos.
    """

    intento_digitos = np.array(
        [int(d) for d in intento],
        dtype=np.int8
    )

    # Fijas
    fijas = np.sum(
        DIGITOS_NUM == intento_digitos,
        axis=1
    )

    # Coincidencias totales
    coincidencias = np.zeros(N, dtype=np.int8)

    for digito in intento_digitos:

        coincidencias += np.sum(
            DIGITOS_NUM == digito,
            axis=1
        )

    picas = coincidencias - fijas

    return (picas * 5 + fijas).astype(np.uint8)


# Precalculamos toda la matriz
for i, intento in enumerate(NUMEROS):

    RESULTADOS[i] = calcular_resultados_para_intento(
        intento
    )

    if i % 500 == 0:
        print(f"Precalculado: {i}/{N}")


print("Tabla terminada.")


# ============================================================
# DECODIFICAR RESULTADO
# ============================================================

def decodificar_resultado(codigo):

    picas = codigo // 5
    fijas = codigo % 5

    return picas, fijas


def codificar_resultado(picas, fijas):

    return picas * 5 + fijas


# ============================================================
# CALCULAR ENTROPÍA
# ============================================================

def entropia(grupos, total):

    """
    Calcula la entropía de la distribución.
    """

    resultado = 0.0

    for cantidad in grupos:

        if cantidad == 0:
            continue

        p = cantidad / total

        resultado -= p * log2(p)

    return resultado


# ============================================================
# MINIMAX + ENTROPÍA
# ============================================================

def mejor_intento(candidatos):

    """
    Busca el mejor intento.

    Primero minimiza el peor grupo.

    Si dos intentos tienen un peor caso similar,
    utilizamos la entropía como desempate.
    """

    indices_candidatos = np.flatnonzero(candidatos)

    cantidad_candidatos = len(indices_candidatos)

    # Si queda solamente uno
    if cantidad_candidatos == 1:
        return indices_candidatos[0]

    mejor_indice = None
    mejor_peor = float("inf")
    mejor_entropia = -float("inf")

    # ========================================================
    # IMPORTANTE:
    #
    # Para pocos candidatos conviene considerar TODOS los
    # números como posibles intentos.
    #
    # Para muchos candidatos utilizamos también todos,
    # pero numpy permite hacerlo relativamente rápido.
    # ========================================================

    for intento in range(N):

        resultados = RESULTADOS[
            intento,
            indices_candidatos
        ]

        # Contamos cuántos candidatos producen
        # cada posible respuesta
        grupos = np.bincount(
            resultados,
            minlength=25
        )

        peor = grupos.max()

        # No necesitamos evaluar la entropía si
        # ya es peor que nuestro mejor resultado.
        if peor > mejor_peor:
            continue

        ent = entropia(
            grupos,
            cantidad_candidatos
        )

        # Minimax:
        # menor peor caso
        #
        # Desempate:
        # mayor entropía

        if (
            peor < mejor_peor
            or
            (
                peor == mejor_peor
                and ent > mejor_entropia
            )
        ):

            mejor_peor = peor
            mejor_entropia = ent
            mejor_indice = intento

    return mejor_indice


# ============================================================
# JUEGO
# ============================================================

def jugar():

    # Al comienzo todos son posibles
    candidatos = np.ones(
        N,
        dtype=bool
    )

    # Primer intento.
    #
    # 0123 es un buen comienzo porque prueba
    # cuatro dígitos distintos.
    intento = NUMEROS.index("0123")

    turno = 1

    print()
    print("=" * 55)
    print("        PICAS Y FIJAS - SOLVER")
    print("=" * 55)
    print()
    print("Piensa un número de 4 dígitos SIN repetir.")
    print()
    print("Cuando haga un intento responde:")
    print()
    print("    picas,fijas")
    print()
    print("Ejemplo:")
    print("    2,1")
    print()
    print("=" * 55)

    while True:

        numero_intento = NUMEROS[intento]

        print()
        print(f"Intento #{turno}: {numero_intento}")

        # ====================================================
        # PEDIR RESPUESTA
        # ====================================================

        while True:

            entrada = input(
                "Resultado (picas,fijas): "
            ).strip()

            try:

                partes = entrada.split(",")

                if len(partes) != 2:
                    raise ValueError

                picas = int(partes[0])
                fijas = int(partes[1])

                if not (
                    0 <= picas <= 4
                    and
                    0 <= fijas <= 4
                ):
                    raise ValueError

                if picas + fijas > 4:
                    raise ValueError

                break

            except ValueError:

                print(
                    "Formato inválido."
                    " Ejemplo: 1,2"
                )

        # ====================================================
        # GANAMOS
        # ====================================================

        if fijas == 4:

            print()
            print("=" * 55)
            print(f"🎯 ¡Encontré tu número!")
            print(f"    {numero_intento}")
            print(f"Intentos: {turno}")
            print("=" * 55)

            return numero_intento

        # ====================================================
        # FILTRAR CANDIDATOS
        # ====================================================

        codigo = codificar_resultado(
            picas,
            fijas
        )

        # Solamente mantenemos números que producirían
        # exactamente la misma respuesta.
        candidatos &= (
            RESULTADOS[intento] == codigo
        )

        cantidad = np.count_nonzero(candidatos)

        print(
            f"Candidatos restantes: {cantidad}"
        )

        # ====================================================
        # COMPROBAR CONSISTENCIA
        # ====================================================

        if cantidad == 0:

            print()
            print("❌ No quedan números posibles.")
            print()
            print(
                "Probablemente alguna respuesta "
                "introducida anteriormente fue incorrecta."
            )

            return None

        # ====================================================
        # SIGUIENTE INTENTO
        # ====================================================

        print("Calculando mejor intento...")

        intento = mejor_intento(
            candidatos
        )

        turno += 1


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    jugar()