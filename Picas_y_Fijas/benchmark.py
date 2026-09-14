import random
import statistics
import sys
import time
from itertools import permutations

from picas_fijas4 import PicasFijasAgentCJBVC

SEMILLA = 12345


def puntuar(intento, secreto):
    fijas = 0
    matches = 0
    for i in range(4):
        if intento[i] == secreto[i]:
            fijas += 1
        if intento[i] in secreto:
            matches += 1
    return (matches - fijas, fijas)


def main():

    completo = False
    muestra = 500
    estrategia = "entropia"
    solo = False
    umbral = 30
    sin_cache = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--completo":
            completo = True
        elif a == "--muestra":
            muestra = int(args[i + 1])
            i += 1
        elif a == "--estrategia":
            estrategia = args[i + 1]
            i += 1
        elif a == "--solo-candidatos":
            solo = True
        elif a == "--umbral":
            umbral = int(args[i + 1])
            i += 1
        elif a == "--sin-cache":
            sin_cache = True
        i += 1

    universo = list(permutations(range(10), 4))

    if completo:
        secretos = universo
    else:
        random.seed(SEMILLA)
        secretos = random.sample(universo, muestra)

    t0 = time.perf_counter()

    agente = PicasFijasAgentCJBVC(
        estrategia=estrategia,
        solo_candidatos=solo,
        umbral=umbral,
        sin_cache=sin_cache,
    )

    intentos = []

    for secreto in secretos:
        agente.start()
        n = 0
        while True:
            n += 1
            guess = agente.try_attempt()
            if tuple(guess) == secreto:
                break
            agente.feedBack(list(puntuar(guess, secreto)))
        intentos.append(n)

    total_time = time.perf_counter() - t0

    prom = sum(intentos) / len(intentos)
    mediana = statistics.median(intentos)
    peor = max(intentos)

    print(f"Estrategia: {estrategia}")
    if solo:
        print(f"Solo candidatos: si (umbral {umbral})")
    if sin_cache:
        print("Cache: desactivada")
    print(f"Partidas: {len(intentos)}")
    print(f"Promedio: {prom:.4f}")
    print(f"Mediana:  {mediana}")
    print(f"Peor caso:{peor}")

    hist = {}
    for d in intentos:
        hist[d] = hist.get(d, 0) + 1
    print("Histograma:", "  ".join(
        f"{k}:{hist[k]}" for k in sorted(hist)
    ))
    print(f"Tiempo: {total_time:.1f} s")


if __name__ == "__main__":
    main()