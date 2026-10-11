"""
Control de entrada: simular el teclado para jugar la partida.

Backends posibles, en orden de preferencia:
  - pynput (si esta instalado en el venv): controla el teclado del
    sistema; funciona sobre X11/Wayland segun la config.
  - xdotool (binario del sistema): `xdotool key --clearmodifiers`.

El juego web (minijuegos) se juega con las flechas, asi que cada
direccion del plan es una tecla. Backend no accede a la pagina: solo
envia pulsaciones.
"""
import shutil
import subprocess
import time
from typing import Optional

import config


class Teclado:
    """Traduce una direccion del plan a una pulsacion de tecla."""

    def __init__(
        self,
        backend: Optional[str] = None,
        pausa_paso: Optional[float] = None,
        prensa_seg: Optional[float] = None,
        window_id: Optional[str] = None,
    ):
        self.pausa_paso = config.PASO_SEG if pausa_paso is None else pausa_paso
        self.prensa_seg = config.PRENSA_SEG if prensa_seg is None else prensa_seg
        self.window_id = window_id if window_id is not None else config.VENTANA_ID

        self.backend = backend or self._detectar_backend()
        if self.backend == "pynput":
            from pynput.keyboard import Controller, Key

            self._control = Controller()
            self._keys = Key
        elif self.backend == "xdotool":
            if shutil.which("xdotool") is None:
                raise RuntimeError("xdotool no esta instalado")
        else:
            raise RuntimeError(f"Backend de teclado desconocido: {backend}")

    @staticmethod
    def _detectar_backend() -> str:
        try:
            import pynput  # noqa: F401
        except ImportError:
            if shutil.which("xdotool"):
                return "xdotool"
            raise RuntimeError(
                "No hay backend de teclado: instala 'pynput' o ten xdotool."
            ) from None
        return "pynput"

    def presionar(self, direccion: str) -> None:
        """Pulsacion de la flecha correspondiente a `direccion`."""
        tecla = config.TECLAS[direccion]
        if self.backend == "pynput":
            self._pulsar_pynput(tecla)
        else:
            self._pulsar_xdotool(tecla)

    def mover(self, direccion: str, pausa: Optional[float] = None) -> None:
        """Presiona la flecha y espera el tiempo del paso."""
        self.presionar(direccion)
        time.sleep(self.pausa_paso if pausa is None else pausa)

    # -- backends -----------------------------------------------------
    def _pulsar_pynput(self, tecla: str) -> None:
        simbolo = getattr(self._keys, tecla)
        self._control.press(simbolo)
        time.sleep(self.prensa_seg)
        self._control.release(simbolo)

    def _pulsar_xdotool(self, tecla: str) -> None:
        cmd = ["xdotool", "key", "--clearmodifiers"]
        if self.window_id:
            cmd += ["--window", self.window_id]
        cmd.append(tecla)
        subprocess.run(cmd, check=False)