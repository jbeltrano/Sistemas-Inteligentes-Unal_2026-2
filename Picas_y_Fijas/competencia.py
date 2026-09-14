import random
import sys
import time
from itertools import permutations

from dinoAgent import AgenteCompetidor
from picas_fijas4 import PicasFijasAgentCJBVC

SEMILLA = 12345


def evaluar(intento, secreto):
    fijas = 0
    picas = 0
    for i in range(4):
        if intento[i] == secreto[i]:
            fijas += 1
        elif intento[i] in secreto:
            picas += 1
    return (picas, fijas)


def jugar_adivinando(secret, reset_fn, intento_fn, feedback_fn):
    """Juega UNA partida completa para el agente: el bucle solo termina
    cuando EL MISMO agente produce (en try_attempt) el número secreto.
    Devuelve (intentos, traza) donde traza = [(intento, feedback_juez, acierto)].
    Cada agente agota su propia partida sin depender del otro."""
    reset_fn()
    intentos = 0
    traza = []
    while True:
        intentos += 1
        guess = intento_fn()
        respuesta = evaluar(guess, secret)
        acierto = tuple(guess) == secret
        traza.append((list(guess), respuesta, acierto))
        if acierto:
            return intentos, traza
        feedback_fn(list(respuesta))


def verificar(secreto, na, traza_a, nb, traza_b):
    """Verifica que el conteo de intentos de cada agente termina exactamente
    en el turno donde EL agente acierta, y que el feedback del juez coincide
    con el scoring interno de ambos agentes (sin picas/fijas invertidas)."""
    assert len(traza_a) == na, f"dino: {len(traza_a)} trazas vs {na} intentos"
    assert len(traza_b) == nb, f"pyf4: {len(traza_b)} trazas vs {nb} intentos"
    assert traza_a[-1][2], "dino no termino con acierto"
    assert traza_b[-1][2], "pyf4 no termino con acierto"
    assert traza_a[-1][1] == (0, 4), f"dino acierto con feedback raro: {traza_a[-1][1]}"
    assert traza_b[-1][1] == (0, 4), f"pyf4 acierto con feedback raro: {traza_b[-1][1]}"
    for g, r, ac in traza_a:
        esperado = AgenteCompetidor._evaluar_rapido(tuple(g), tuple(secreto))
        assert r == esperado, f"dino/juez mismatch: {g} {r} vs {esperado}"
    for g, r, ac in traza_b:
        esperado = PicasFijasAgentCJBVC.check_option_static(g, list(secreto))
        assert r == esperado, f"pyf4/juez mismatch: {g} {r} vs {esperado}"


def main():

    partidas = 100
    detalle = False
    if "--partidas" in sys.argv:
        partidas = int(sys.argv[sys.argv.index("--partidas") + 1])
    if "--detalle" in sys.argv:
        detalle = True

    #random.seed(SEMILLA)
    random.seed()
    universo = list(permutations(range(10), 4))
    secretos = random.sample(universo, partidas)

    agente_a = AgenteCompetidor()
    agente_b = PicasFijasAgentCJBVC()

    t0 = time.perf_counter()

    intentos_a = []
    intentos_b = []
    trazas_a = []
    trazas_b = []

    for secreto in secretos:
        n_a, traza_a = jugar_adivinando(
            secreto,
            agente_a.start,
            agente_a.try_attempt,
            agente_a.feedBack,
        )
        n_b, traza_b = jugar_adivinando(
            secreto,
            agente_b.start,
            agente_b.try_attempt,
            agente_b.feedBack,
        )
        verificar(secreto, n_a, traza_a, n_b, traza_b)
        intentos_a.append(n_a)
        intentos_b.append(n_b)
        trazas_a.append(traza_a)
        trazas_b.append(traza_b)

    total_time = time.perf_counter() - t0

    if detalle:
        for i, (secreto, ta, tb) in enumerate(zip(secretos, trazas_a, trazas_b)):
            print(f"--- Partida {i + 1}: secreto {list(secreto)}")
            print(f"  dinoAgent   (gana en {intentos_a[i]} intentos)")
            for g, r, ac in ta:
                print(f"    intento {g} -> {r}  {'<-- ACIERTO' if ac else ''}")
            print(f"  picas_fijas4 (gana en {intentos_b[i]} intentos)")
            for g, r, ac in tb:
                print(f"    intento {g} -> {r}  {'<-- ACIERTO' if ac else ''}")

    def resumen(nombre, datos):
        prom = sum(datos) / len(datos)
        mediana = sorted(datos)[len(datos) // 2]
        peor = max(datos)
        print(f"{nombre}:")
        print(f"  Promedio: {prom:.4f}")
        print(f"  Mediana:  {mediana}")
        print(f"  Peor caso:{peor}")
        hist = {}
        for d in datos:
            hist[d] = hist.get(d, 0) + 1
        print("  Histograma:", "  ".join(
            f"{k}:{hist[k]}" for k in sorted(hist)
        ))

    ganan_a = 0
    ganan_b = 0
    empates = 0
    ventaja_total = 0

    for na, nb in zip(intentos_a, intentos_b):
        if na < nb:
            ganan_a += 1
        elif nb < na:
            ganan_b += 1
        else:
            empates += 1
        ventaja_total += na - nb

    print("=" * 50)
    resumen("dinoAgent (AgenteCompetidor)", intentos_a)
    print()
    resumen("picas_fijas4 (PicasFijasAgentCJBVC)", intentos_b)
    print("=" * 50)
    print(f"Victorias dinoAgent:      {ganan_a}")
    print(f"Victorias picas_fijas4:   {ganan_b}")
    print(f"Empates:                  {empates}")
    print(f"Ventaja media dino - pyf4: {ventaja_total / partidas:+.4f} intentos")
    print(f"Partidas: {partidas}  Tiempo: {total_time:.1f} s")


if __name__ == "__main__":
    main()