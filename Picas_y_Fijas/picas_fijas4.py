import math
import random
import time
from collections import defaultdict
from itertools import permutations

import numpy as np


class PicasFijasAgentCJBVC:

    ALL_NUMBERS = None
    NUM_MATRIX = None
    RESULT_TABLE = None
    FIRST_GUESS = (5, 6, 7, 8)

    CACHE = {}

    _N = 0
    _FIRST_GUESS_IDX = 0

    # Parametros de busqueda profunda
    _BEAM = 20
    _BEAM_D1 = 8
    _EXACTO_UMBRAL = 32
    _PROF_UMBRAL = 600
    _MUESTRA_UNIV = None      # indices fijos de universo para continuar la busqueda
    _MUESTRA_D1 = None        # muestreo reducido para nivel 2 de la busqueda

    ESTRATEGIAS = ("entropia", "minimax_entropia", "esperado", "profundidad2")

    def __init__(self, estrategia="entropia", solo_candidatos=False,
                 umbral=30, sin_cache=False):

        if PicasFijasAgentCJBVC.ALL_NUMBERS is None:
            PicasFijasAgentCJBVC.initialize_game_data()

        if estrategia not in self.ESTRATEGIAS:
            raise ValueError(
                f"Estrategia '{estrategia}' invalida. Opciones: {self.ESTRATEGIAS}"
            )

        self.estrategia = estrategia
        self.solo_candidatos = solo_candidatos
        self.umbral = umbral
        self.sin_cache = sin_cache

        if sin_cache:
            self.cache = {}
        else:
            self.cache = PicasFijasAgentCJBVC.CACHE

        self.candidate_idxs = []
        self.my_number = None
        self.my_guess_idx = None

    @classmethod
    def initialize_game_data(cls):

        cls.ALL_NUMBERS = [
            tuple(int(d) for d in p)
            for p in permutations("0123456789", 4)
        ]

        cls.NUM_MATRIX = np.array(cls.ALL_NUMBERS, dtype=np.int8)

        cls._N = len(cls.ALL_NUMBERS)

        cls.RESULT_TABLE = np.zeros(
            (cls._N, cls._N),
            dtype=np.uint8
        )

        digitos = cls.NUM_MATRIX

        for i in range(cls._N):

            guess = digitos[i]

            fijas = np.sum(
                digitos == guess,
                axis=1,
                dtype=np.int16
            )

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

        cls._MUESTRA_UNIV = random.Random(987).sample(range(cls._N), 400)

        cls._MUESTRA_D1 = random.Random(988).sample(range(cls._N), 120)

    def start(self):

        self.candidate_idxs = list(range(self._N))

        self.my_number = random.choice(self.ALL_NUMBERS)

        self.my_guess_idx = self._FIRST_GUESS_IDX

    def try_attempt(self):

        return list(self.ALL_NUMBERS[self.my_guess_idx])

    def discover(self, enemy_guess):

        enemy_guess = tuple(enemy_guess)

        return list(
            self.check_option_static(
                enemy_guess,
                self.my_number
            )
        )

    def feedBack(self, enemy_answer):

        enemy_answer = tuple(enemy_answer)

        codigo_objetivo = (
            enemy_answer[0] * 5 + enemy_answer[1]
        )

        resultados = self.RESULT_TABLE[self.my_guess_idx]

        self.candidate_idxs = [
            idx
            for idx in self.candidate_idxs
            if resultados[idx] == codigo_objetivo
        ]

        if len(self.candidate_idxs) == 1:

            self.my_guess_idx = self.candidate_idxs[0]

            return

        self.my_guess_idx = self.get_best_guess()

    # ------------------------------------------------------------------
    # SELECCION DE INTENTO
    # ------------------------------------------------------------------
    def get_best_guess(self):

        candidatos = self.candidate_idxs

        if len(candidatos) == 1:

            return candidatos[0]

        if self.estrategia == "profundidad2":

            mejor, _ = self._profundidad2(candidatos, 2)

            return mejor

        return self._best_ply1(candidatos)

    def _best_ply1(self, candidatos):

        mask = self._mascara(candidatos)

        key = ("ply", self.estrategia, mask)

        if key in self.cache:
            return self.cache[key]

        mejor = self._best_ply1_bruto(candidatos)

        if not self.sin_cache:
            self.cache[key] = mejor

        return mejor

    def _best_ply1_bruto(self, candidatos):

        indices = np.array(candidatos, dtype=np.intp)
        candidato_set = set(candidatos)

        total = len(candidatos)

        tabla = self.RESULT_TABLE

        mejor_guess = None
        mejor_peor = float("inf")
        mejor_entropia = -float("inf")
        mejor_esperado = float("inf")
        mejor_es_candidato = False

        for guess in self._posibles(candidatos):

            grupos = np.bincount(
                tabla[guess, indices],
                minlength=25
            )

            entropia = 0.0
            suma_cuadrados = 0
            peor = 0

            for cantidad in grupos:

                if cantidad == 0:
                    continue

                if cantidad > peor:
                    peor = int(cantidad)

                suma_cuadrados += cantidad * cantidad

                p = cantidad / total

                entropia -= p * math.log2(p)

            es_candidato = guess in candidato_set

            if self._mejor_que_ply1(
                guess,
                peor,
                entropia,
                suma_cuadrados,
                es_candidato,
                mejor_guess,
                mejor_peor,
                mejor_entropia,
                mejor_esperado,
                mejor_es_candidato,
            ):

                mejor_guess = guess
                mejor_peor = peor
                mejor_entropia = entropia
                mejor_esperado = suma_cuadrados
                mejor_es_candidato = es_candidato

        return mejor_guess

    def _mejor_que_ply1(self, guess, peor, entropia, suma_cuadrados,
                        es_candidato, m_guess, m_peor, m_entropia, m_esperado,
                        m_es_candidato):

        if self.estrategia == "minimax_entropia":
            return (
                peor < m_peor
                or (peor == m_peor and entropia > m_entropia)
                or (
                    peor == m_peor
                    and entropia == m_entropia
                    and es_candidato
                    and not m_es_candidato
                )
            )

        if self.estrategia == "esperado":
            return (
                suma_cuadrados < m_esperado
                or (suma_cuadrados == m_esperado and entropia > m_entropia)
                or (
                    suma_cuadrados == m_esperado
                    and entropia == m_entropia
                    and es_candidato
                    and not m_es_candidato
                )
            )

        # entropia (default)
        return (
            entropia > m_entropia
            or (entropia == m_entropia and peor < m_peor)
            or (
                entropia == m_entropia
                and peor == m_peor
                and es_candidato
                and not m_es_candidato
            )
        )

    def _posibles(self, candidatos, chico=False):

        if self.solo_candidatos and len(candidatos) <= self.umbral:
            return candidatos

        # Determinista: candidatos (si son pocos) + muestra fija del universo
        lista = []
        vistos = set()

        muestra = (
            self._MUESTRA_D1 if chico else self._MUESTRA_UNIV
        )

        for fuente in ((candidatos if len(candidatos) <= 300 else ()),
                       muestra):

            for v in fuente:
                if v not in vistos:
                    vistos.add(v)
                    lista.append(v)

        return lista

    # ------------------------------------------------------------------
    # BUSQUEDA PROFUNDA (profundidad2): costo esperado recursivo
    # ------------------------------------------------------------------
    def _profundidad2(self, candidatos, depth, chico=False):

        n = len(candidatos)

        if n == 1:
            return candidatos[0], 1.0

        mask = self._mascara(candidatos)
        cache = self.cache

        if n <= self._EXACTO_UMBRAL:

            key = ("exact", mask)

            if key in cache:
                return cache[key]

            r = self._exacto(candidatos, mask)

            if not self.sin_cache:
                cache[key] = r

            return r

        if depth <= 0:

            key = ("base", mask)

            if key in cache:
                return cache[key]

            r = self._base_ply1(candidatos, chico)

            if not self.sin_cache:
                cache[key] = r

            return r

        # Si el conjunto es enorme, la recursion profunda cuesta demasiado:
        # se corta con la evaluacion de 1 nivel
        if n > self._PROF_UMBRAL and depth < 2:

            key = ("base", mask)

            if key in cache:
                return cache[key]

            r = self._base_ply1(candidatos, chico)

            if not self.sin_cache:
                cache[key] = r

            return r

        key = ("prof", depth, mask)

        if key in cache:
            return cache[key]

        r = self._beam(candidatos, mask, depth, chico)

        if not self.sin_cache:
            cache[key] = r

        return r

    def _exacto(self, candidatos, mask):

        n = len(candidatos)

        if n == 1:
            return candidatos[0], 1.0

        # Resultado parcial ya cacheado en _profundidad2 (nivel exacto)
        tabla = self.RESULT_TABLE
        mejor_guess = None
        mejor_exp = float("inf")

        for g in candidatos:

            buckets = defaultdict(list)

            for i in candidatos:
                buckets[tabla[g, i]].append(i)

            exp_futuro = 0.0

            for sub in buckets.values():

                k = len(sub)

                if k == n:
                    continue  # guess no separa nada

                _, sub_exp = self._exacto(sub, self._mascara(sub))

                exp_futuro += (k / n) * sub_exp

            total = 1 + exp_futuro

            if total < mejor_exp:
                mejor_exp = total
                mejor_guess = g

        return mejor_guess, mejor_exp

    def _base_ply1(self, candidatos, chico=False):

        # Continuacion de 1 nivel por minimo esperado de candidatos restantes
        indices = np.array(candidatos, dtype=np.intp)
        n = len(candidatos)
        tabla = self.RESULT_TABLE

        mejor_guess = None
        mejor_esperado = float("inf")
        mejor_entropia = -float("inf")

        for g in self._posibles(candidatos, chico):

            grupos = np.bincount(tabla[g, indices], minlength=25)

            suma_cuadrados = 0
            entropia = 0.0

            for cantidad in grupos:

                if cantidad == 0:
                    continue

                suma_cuadrados += cantidad * cantidad

                p = cantidad / n
                entropia -= p * math.log2(p)

            if (
                suma_cuadrados < mejor_esperado
                or (
                    suma_cuadrados == mejor_esperado
                    and entropia > mejor_entropia
                )
            ):

                mejor_esperado = suma_cuadrados
                mejor_entropia = entropia
                mejor_guess = g

        return mejor_guess, 1.0 + mejor_esperado / n

    def _beam(self, candidatos, mask, depth, chico):

        n = len(candidatos)
        indices = np.array(candidatos, dtype=np.intp)
        tabla = self.RESULT_TABLE

        beam_size = self._BEAM_D1 if chico else self._BEAM

        # 1er nivel: puntuar todos los posibles, quedarse con el beam
        key_scoring = ("sc", chico, mask)

        if key_scoring in self.cache:
            scoring = self.cache[key_scoring]
        else:
            scoring = []
            for g in self._posibles(candidatos, chico):
                grupos = np.bincount(tabla[g, indices], minlength=25)
                suma_cuadrados = 0
                entropia = 0.0
                for cantidad in grupos:
                    if cantidad == 0:
                        continue
                    suma_cuadrados += cantidad * cantidad
                    p = cantidad / n
                    entropia -= p * math.log2(p)
                scoring.append((suma_cuadrados, -entropia, g))
            scoring.sort()
            if not self.sin_cache:
                self.cache[key_scoring] = scoring

        beam = [g for _, _, g in scoring[:beam_size]]

        mejor_guess = None
        mejor_exp = float("inf")

        for g in beam:

            buckets = defaultdict(list)

            for i in candidatos:
                buckets[tabla[g, i]].append(i)

            exp_futuro = 0.0

            for sub in buckets.values():

                k = len(sub)

                _, sub_exp = self._profundidad2(sub, depth - 1, chico or depth - 1 == 1)

                exp_futuro += (k / n) * sub_exp

            total = 1 + exp_futuro

            if total < mejor_exp:
                mejor_exp = total
                mejor_guess = g

        return mejor_guess, mejor_exp

    @staticmethod
    def _mascara(candidatos):

        m = 0

        for i in candidatos:
            m |= 1 << i

        return m

    # ------------------------------------------------------------------
    # OTROS
    # ------------------------------------------------------------------
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


if __name__ == "__main__":

    import sys as _sys

    estrategia = "entropia"

    if len(_sys.argv) > 1:
        estrategia = _sys.argv[1]

    t0 = time.perf_counter()

    agente = PicasFijasAgentCJBVC(estrategia=estrategia)

    t_init = time.perf_counter() - t0

    agente.start()

    intentos = 0

    while True:

        intentos += 1

        intento = agente.try_attempt()

        if tuple(intento) == agente.my_number:

            print(f"Numero encontrado: {''.join(map(str, agente.my_number))}")
            print(f"Intentos: {intentos}")
            print(f"Tiempo de inicializacion: {t_init:.1f} s")
            print(f"Tiempo total: {time.perf_counter() - t0:.1f} s")

            break

        respuesta = agente.discover(intento)

        agente.feedBack(respuesta)