from abc import ABC, abstractmethod
from itertools import permutations
from collections import defaultdict



class AgentePicasFijas_generic():

    def __init__(self):
        # Todos los números posibles de 4 dígitos
        # sin repetir y sin 0 al inicio.
        self.todos = self._generar_numeros()

        # Estado actual del agente
        self.candidatos = []
        self.ultimo_intento = None
        self.intentos = 0
        self.terminado = False

        # Cache para no calcular repetidamente
        self._cache_particiones = {}

    # --------------------------------------------------------
    # GENERACIÓN DE POSIBLES SECRETOS
    # --------------------------------------------------------

    def _generar_numeros(self):
        return [
            "".join(p)
            for p in permutations("0123456789", 4)
            if p[0] != "0"
        ]

    # --------------------------------------------------------
    # CALCULAR PICAS Y FIJAS
    # --------------------------------------------------------

    @staticmethod
    def _score(intento, secreto):

        fijas = sum(
            a == b
            for a, b in zip(intento, secreto)
        )

        comunes = len(
            set(intento) & set(secreto)
        )

        picas = comunes - fijas

        return picas, fijas

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):
        """
        Reinicia completamente el agente para una nueva ronda.
        """

        self.candidatos = self.todos.copy()
        self.ultimo_intento = None
        self.intentos = 0
        self.terminado = False

        self._cache_particiones.clear()

    # --------------------------------------------------------
    # TRY ATTEMPT
    # --------------------------------------------------------

    def try_attempt(self):
        """
        Calcula y devuelve el mejor siguiente intento.
        """

        if self.terminado:
            return None

        # Primera jugada.
        #
        # 1234 es una buena jugada inicial porque prueba
        # cuatro dígitos diferentes.
        if self.ultimo_intento is None:

            self.ultimo_intento = "1234"
            self.intentos += 1

            return self.ultimo_intento

        # Si solamente queda un candidato,
        # no necesitamos hacer minimax.
        if len(self.candidatos) == 1:

            self.ultimo_intento = self.candidatos[0]
            self.intentos += 1

            return self.ultimo_intento

        # Buscar la mejor jugada.
        intento = self._mejor_intento()

        self.ultimo_intento = intento
        self.intentos += 1

        return intento

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    def feedBack(self, retroalimentacionLista):
        """
        Recibe:

            [picas, fijas]

        Ejemplo:

            [2, 1]

        significa:
            2 picas
            1 fija
        """

        if self.ultimo_intento is None:
            raise RuntimeError(
                "No se puede proporcionar feedback "
                "antes de realizar un intento."
            )

        if len(retroalimentacionLista) != 2:
            raise ValueError(
                "El feedback debe tener la forma [picas, fijas]."
            )

        picas, fijas = retroalimentacionLista

        if not (
            isinstance(picas, int)
            and isinstance(fijas, int)
        ):
            raise ValueError(
                "Picas y fijas deben ser enteros."
            )

        if picas < 0 or fijas < 0 or picas + fijas > 4:
            raise ValueError(
                "Feedback inválido."
            )

        # Si tenemos 4 fijas, hemos ganado.
        if fijas == 4:
            self.terminado = True
            self.candidatos = [self.ultimo_intento]
            return

        # Filtrar candidatos compatibles.
        nuevo_conjunto = []

        objetivo = (picas, fijas)

        for candidato in self.candidatos:

            if self._score(
                self.ultimo_intento,
                candidato
            ) == objetivo:

                nuevo_conjunto.append(candidato)

        self.candidatos = nuevo_conjunto

        if not self.candidatos:
            raise ValueError(
                "El feedback recibido es inconsistente "
                "con los intentos anteriores."
            )

    # --------------------------------------------------------
    # MEJOR INTENTO
    # --------------------------------------------------------

    def _mejor_intento(self):

        candidatos = self.candidatos
        conjunto_candidatos = set(candidatos)

        mejor_intento = None
        mejor_clave = None

        # Cuando hay muchos candidatos, probar todos los 5040
        # posibles intentos es innecesariamente costoso.
        #
        # Cuando quedan pocos candidatos, sí vale la pena
        # considerar todos los posibles números.
        if len(candidatos) <= 150:
            intentos_posibles = self.todos
        else:
            intentos_posibles = candidatos

        for intento in intentos_posibles:

            grupos = defaultdict(int)

            for candidato in candidatos:

                resultado = self._score(
                    intento,
                    candidato
                )

                grupos[resultado] += 1

            # Peor cantidad de candidatos que puede quedar
            peor_caso = max(grupos.values())

            # Desempate:
            # favorecemos una distribución uniforme.
            tamanos = sorted(
                grupos.values(),
                reverse=True
            )

            # Preferimos candidatos reales en caso de empate.
            no_candidato = (
                0
                if intento in conjunto_candidatos
                else 1
            )

            clave = (
                peor_caso,
                *tamanos[:5],
                no_candidato
            )

            if (
                mejor_clave is None
                or clave < mejor_clave
            ):
                mejor_clave = clave
                mejor_intento = intento

        return mejor_intento
