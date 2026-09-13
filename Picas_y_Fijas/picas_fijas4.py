import math
import random
import time
from itertools import permutations

import numpy as np


class PicasFijasAgent:

    # ==========================================================
    # VARIABLES ESTÁTICAS (tablas compartidas entre instancias)
    # ==========================================================

    ALL_NUMBERS = None      # lista de tuplas (d1, d2, d3, d4) -> 5040 códigos
    NUM_MATRIX = None       # numpy int8 (5040, 4)
    RESULT_TABLE = None     # numpy uint8 (5040, 5040): código = picas*5+fijas
    FIRST_GUESS = (5, 6, 7, 8)

    _N = 0
    _FIRST_GUESS_IDX = 0

    def __init__(self, solo_candidatos=False):

        if PicasFijasAgent.ALL_NUMBERS is None:
            PicasFijasAgent.initialize_game_data()

        self.solo_candidatos = solo_candidatos

        self.candidate_idxs = []      # índices sobre ALL_NUMBERS
        self.my_number = None         # tupla secreta
        self.my_guess_idx = None      # índice sobre ALL_NUMBERS

    # ==========================================================
    # PRECALCULAR TODOS LOS DATOS (una sola vez)
    # ==========================================================

    @classmethod
    def initialize_game_data(cls):

        print("Inicializando datos...")

        # Todos los números válidos (5040 permutaciones)
        cls.ALL_NUMBERS = [
            tuple(int(d) for d in p)
            for p in permutations("0123456789", 4)
        ]

        cls.NUM_MATRIX = np.array(cls.ALL_NUMBERS, dtype=np.int8)

        cls._N = len(cls.ALL_NUMBERS)

        print(f"Números posibles: {cls._N}")

        # Tabla de resultados vectorizada por fila:
        #
        #   RESULT_TABLE[guess_idx, candidate_idx] = picas*5+fijas
        #
        cls.RESULT_TABLE = np.zeros(
            (cls._N, cls._N),
            dtype=np.uint8
        )

        digitos = cls.NUM_MATRIX

        for i in range(cls._N):

            guess = digitos[i]

            # Fijas: dígitos en posición correcta
            fijas = np.sum(
                digitos == guess,
                axis=1,
                dtype=np.int16
            )

            # Coincidencias totales de dígito (sin posición)
            coincidencias = np.zeros(cls._N, dtype=np.int16)

            for d in guess:
                coincidencias += np.sum(
                    digitos == d,
                    axis=1,
                    dtype=np.int16
                )

            picas = coincidencias - fijas

            cls.RESULT_TABLE[i] = (
                (picas * 5 + fijas).astype(np.uint8)
            )

        cls._FIRST_GUESS_IDX = cls.ALL_NUMBERS.index(cls.FIRST_GUESS)

        print("Tabla de resultados creada.")

    # ==========================================================
    # START (el medio ambiente inicia un juego)
    # ==========================================================

    def start(self):

        # Todos los números son inicialmente posibles
        self.candidate_idxs = list(range(self._N))

        # Número secreto
        self.my_number = random.choice(self.ALL_NUMBERS)

        # Primer intento
        self.my_guess_idx = self._FIRST_GUESS_IDX

    # ==========================================================
    # INTENTO (Acción del agente: jugar)
    # ==========================================================

    def try_attempt(self):

        return list(self.ALL_NUMBERS[self.my_guess_idx])

    # ==========================================================
    # RESPONDER AL OPONENTE (el agente también es medio ambiente)
    # ==========================================================

    def discover(self, enemy_guess):

        enemy_guess = tuple(enemy_guess)

        return list(
            self.check_option_static(
                enemy_guess,
                self.my_number
            )
        )

    # ==========================================================
    # RECIBIR FEEDBACK (Percepción del agente)
    # ==========================================================

    def feedBack(self, enemy_answer):

        enemy_answer = tuple(enemy_answer)

        codigo_objetivo = (
            enemy_answer[0] * 5 + enemy_answer[1]
        )

        # Solamente mantenemos números que producirían
        # exactamente la misma respuesta.
        resultados = self.RESULT_TABLE[self.my_guess_idx]

        self.candidate_idxs = [
            idx
            for idx in self.candidate_idxs
            if resultados[idx] == codigo_objetivo
        ]

        # ------------------------------------------------------
        # Si encontramos el número
        # ------------------------------------------------------

        if len(self.candidate_idxs) == 1:

            self.my_guess_idx = self.candidate_idxs[0]

            return

        # ------------------------------------------------------
        # Elegir siguiente intento (Computación del agente)
        # ------------------------------------------------------

        self.my_guess_idx = self.get_best_guess()

    # ==========================================================
    # MINIMAX + ENTROPÍA + PREFERENCIA POR CANDIDATO
    # ==========================================================

    def get_best_guess(self):

        candidatos = self.candidate_idxs

        if len(candidatos) == 1:

            # Si queda únicamente un candidato, esa es la respuesta
            return candidatos[0]

        # ------------------------------------------------------
        # Posibles intentos a evaluar.
        #
        # Por defecto se evalúan TODOS los números (5040) para
        # conseguir el minimax global.
        #
        # Con solo_candidatos=True (solo para benchmark) se restringe
        # la búsqueda a los candidatos reales cuando quedan pocos.
        # ------------------------------------------------------

        if self.solo_candidatos and len(candidatos) <= 30:

            posibles = list(candidatos)

        else:

            posibles = range(self._N)

        indices = np.array(candidatos, dtype=np.intp)
        candidato_set = set(candidatos)

        total = len(candidatos)

        tabla = self.RESULT_TABLE

        mejor_guess = None
        mejor_peor = float("inf")
        mejor_entropia = -float("inf")
        mejor_es_candidato = False

        for guess in posibles:

            grupos = np.bincount(
                tabla[guess, indices],
                minlength=25
            )

            peor = int(grupos.max())

            # Poda: si el peor caso ya es peor que el mejor
            # encontrado, no vale la pena calcular entropía.
            if peor > mejor_peor:
                continue

            # Entropía de la distribución
            entropia = 0.0

            for cantidad in grupos:

                if cantidad == 0:
                    continue

                p = cantidad / total

                entropia -= p * math.log2(p)

            es_candidato = guess in candidato_set

            # --------------------------------------------------
            # SELECCIÓN
            #
            # 1. Menor peor caso (MINIMAX)
            # 2. Desempate: mayor entropía
            # 3. Desempate final: preferir un candidato real
            # --------------------------------------------------

            if (
                peor < mejor_peor
                or (
                    peor == mejor_peor
                    and entropia > mejor_entropia
                )
                or (
                    peor == mejor_peor
                    and entropia == mejor_entropia
                    and es_candidato
                    and not mejor_es_candidato
                )
            ):

                mejor_guess = guess
                mejor_peor = peor
                mejor_entropia = entropia
                mejor_es_candidato = es_candidato

        return mejor_guess

    # ==========================================================
    # CHECK OPTION
    # ==========================================================

    @staticmethod
    def check_option_static(option, guessing_number):

        picas = 0
        fijas = 0

        for i in range(4):

            if option[i] == guessing_number[i]:

                fijas += 1

            elif option[i] in guessing_number:

                picas += 1

        return (picas, fijas)


# ============================================================
# DRIVER DE JUEGO
# ============================================================

if __name__ == "__main__":

    t0 = time.perf_counter()

    agente = PicasFijasAgent()

    t_init = time.perf_counter() - t0

    agente.start()

    intentos = 0

    while True:

        intentos += 1

        intento = agente.try_attempt()

        if tuple(intento) == agente.my_number:

            print(f"Número encontrado: {''.join(map(str, agente.my_number))}")
            print(f"Intentos: {intentos}")
            print(f"Tiempo de inicialización: {t_init:.1f} s")
            print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")

            break

        respuesta = agente.discover(intento)

        agente.feedBack(respuesta)