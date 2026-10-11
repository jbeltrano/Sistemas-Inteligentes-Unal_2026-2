"""
La clase del agente: cierra el lazo percepcion -> decision -> control
-> verificacion de progreso.

  - compute(entrada): PERCEPCION + DECISION. Recibe UN unico screenshot
    (un path o un frame), reconoce el EstadoJuego, arma el Mundo y
    define el camino solucion completo (A* + retroceso) SIN necesitar
    repetir capturas.
  - ejecutar(plan): CONTROL + VERIFICACION. Envia las teclas del plan
    y, si se le provee un capturador en vivo, re-percibe para
    comparar la celda del personaje contra la esperada.

Tipos de agente: deliberativo basado en modelos, sin aprendizaje.
"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import config
from control import Teclado
from percepcion.captura import leer_imagen
from percepcion.estado import EstadoJuego
from percepcion.reconocimiento import imprimir_estado, reconocer
from planificacion import (
    Mundo,
    Plan,
    Planificador,
    mundo_desde_estado,
    simular,
)


class Agent:
    """
    Percepcion (un screenshot) -> decision (plan) -> control (teclas)
    -> verificacion de progreso.
    """

    def __init__(
        self,
        planificador: Optional[Planificador] = None,
        teclado: Optional[Teclado] = None,
        capturador: Optional[Callable[[], np.ndarray]] = None,
        verificar_a_cada_paso: bool = config.VERIFICAR_A_CADA_PASO,
        imprimir_estado: bool = False,
    ):
        # `capturador` es opcional y SOLO se usa para verificacion de
        # progreso en vivo. El plan nunca depende de recapturar:
        # se calcula con la unica imagen que recibe compute().
        self.planificador = planificador or Planificador()
        # El teclado se crea bajo demanda (solo hace falta al ejecutar).
        self._teclado = teclado
        self.capturador = capturador
        self.verificar_a_cada_paso = verificar_a_cada_paso
        self.quiero_imprimir = imprimir_estado

        self.estado: Optional[EstadoJuego] = None
        self.mundo: Optional[Mundo] = None
        self.plan: Optional[Plan] = None

    @property
    def teclado(self) -> Teclado:
        if self._teclado is None:
            self._teclado = Teclado()
        return self._teclado

    # ---------------------------------------------------------------
    # 1 + 2) Percepcion + decision
    # ---------------------------------------------------------------
    def compute(self, entrada=None) -> Plan:
        """
        Recibe UN screenshot (ruta, Path o frame BGR) y devuelve el
        camino solucion completo. Con la unica percepcion se resuelve
        todo el problema; no se vuelve a mirar la pantalla para decidir.
        """
        frame = self._leer_entrada(entrada)
        self.estado = reconocer(frame)
        self.mundo = mundo_desde_estado(self.estado)
        self.plan = self.planificador.planificar(self.mundo)

        if self.quiero_imprimir:
            imprimir_estado(self.estado)

        return self.plan

    def _leer_entrada(self, entrada) -> np.ndarray:
        if isinstance(entrada, np.ndarray):
            return entrada
        if isinstance(entrada, (str, Path)):
            return leer_imagen(entrada)
        if entrada is None:
            # Sin entrada: el unico screenshot por defecto es captura.png.
            return leer_imagen(config.IMAGEN_DEFECTO)
        raise TypeError(f"Entrada de screenshot no entendida: {type(entrada)}")

    # ---------------------------------------------------------------
    # 3 + 4) Control + verificacion de progreso
    # ---------------------------------------------------------------
    def ejecutar(
        self,
        plan: Optional[Plan] = None,
        pausa_paso: Optional[float] = None,
        verificar: bool = True,
    ) -> Plan:
        """
        Envia por teclado las acciones del plan (control) y verifica
        progreso. Si no hay capturador en vivo, la verificacion se
        reduce a la reproduccion simulada del plan.
        """
        plan = plan or self.plan
        if plan is None:
            plan = self.compute()

        self._verificar_simulacion(plan)

        profundidad_objetivo = self._celda_objetivo(plan)
        paso = 0
        for accion in plan.acciones:
            paso += 1
            print(f"[{paso:>4}/{plan.longitud}] {accion.texto()}")
            self.teclado.mover(accion.direccion, pausa=pausa_paso)
            if verificar and self.verificar_a_cada_paso and self.capturador:
                self._verificar_en_vivo(profundidad_objetivo, plan)

        if verificar:
            self._verificar_final(plan)

        return plan

    def _celda_objetivo(self, plan: Plan):
        """Celda esperada del personaje al ejecutar el plan completo."""
        m = simular(plan.mundo_inicial, plan.acciones)
        return m.personaje

    def _verificar_simulacion(self, plan: Plan) -> None:
        """Rejuego del plan sobre el modelo: si no reproduce el final, hay bug."""
        m = simular(plan.mundo_inicial, plan.acciones)
        if plan.completo:
            coincidencia = (
                plan.mundo_final is not None
                and m.personaje == plan.mundo_final.personaje
                and m.diamantes == plan.mundo_final.diamantes
            )
            if not coincidencia:
                print("ADVERTENCIA: la simulacion del plan difiere de lo esperado.")

    def _repercibir(self):
        try:
            return reconocer(self.capturador())
        except RuntimeError:
            return None

    def _verificar_en_vivo(self, celda_esperada, plan: Plan) -> None:
        """Re-percibe (1 frame) y compara la celda real con la esperada."""
        if not self.capturador:
            return
        estado = self._repercibir()
        if estado is None:
            return  # pantalla de transicion/nivel terminado: no es fallo
        if estado.player_cell != celda_esperada:
            print(
                f"PROGRESO INESPERADO: personaje en {estado.player_cell}, "
                f"se esperaba {celda_esperada}."
            )
            if plan.completo:
                print("Se re-planifica sobre el estado percibido.")
                self.estado = estado
                self.mundo = mundo_desde_estado(estado)
                self.plan = self.planificador.planificar(self.mundo)

    def _verificar_final(self, plan: Plan) -> None:
        print("-" * 60)
        if not plan.completo:
            print(f"Plan incompleto; metas pendientes: {plan.pendientes}")
            return

        meta_final_era_salida = (
            plan.mundo_final is not None
            and plan.mundo_final.salida == plan.mundo_final.personaje
        )

        estado = self._repercibir() if self.capturador else self.estado
        if estado is None:
            # No se encuentra al personaje tras ejecutar: si la meta era
            # la salida, es la pantalla de nivel completado -> exito.
            if meta_final_era_salida and self.capturador is not None:
                print("Nivel completado (pantalla de victoria).")
            else:
                print("No se pudo re-percibir para confirmar el nivel.")
            return

        if self.mundo is not None and self.mundo.salida is not None and estado.player_cell == self.mundo.salida:
            print("Verificacion: personaje en la salida.")
        else:
            print(f"Verificacion: personaje quedó en {estado.player_cell}.")