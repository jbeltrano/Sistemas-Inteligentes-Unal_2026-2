"""
Control de entrada: traduce una direccion del plan a una flecha que se
manda al canvas por Selenium.

El juego se juega con las flechas, asi que cada direccion del plan es
una tecla. Se reutiliza el MISMO driver que la percepcion (ver
capture.py): cuando `locate_canvas()` corrio, el driver ya esta
posicionado dentro del iframe, asi que el canvas es un elemento valido
y el contexto de teclado correcto. Este modulo no crea ni cierra el
navegador, y no lee datos de la pagina: solo envia pulsaciones.
"""
import time
from typing import Optional

from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

import config

# Direccion del plan -> flecha de Selenium.
KEYS = {
    "up": Keys.ARROW_UP,
    "down": Keys.ARROW_DOWN,
    "left": Keys.ARROW_LEFT,
    "right": Keys.ARROW_RIGHT,
}


class Keyboard:
    """Traduce una direccion del plan a una pulsacion de flecha sobre el canvas."""

    def __init__(
        self,
        driver: WebDriver,
        canvas: WebElement,
        step_seconds: Optional[float] = None,
        press_seconds: Optional[float] = None,
    ):
        self.driver = driver
        self.canvas = canvas
        self.step_seconds = (
            config.STEP_SECONDS if step_seconds is None else step_seconds
        )
        self.press_seconds = (
            config.PRESS_SECONDS if press_seconds is None else press_seconds
        )

    def focus(self) -> None:
        """Da foco al canvas para que el juego reciba el teclado."""
        self.canvas.click()

    def press(self, direction: str) -> None:
        """Pulsacion de la flecha correspondiente a `direction`."""
        key = KEYS[direction]
        ActionChains(self.driver).key_down(key, self.canvas).perform()
        time.sleep(self.press_seconds)
        ActionChains(self.driver).key_up(key, self.canvas).perform()

    def move(self, direction: str, pause: Optional[float] = None) -> None:
        """Presiona la flecha y espera el tiempo del paso."""
        self.press(direction)
        time.sleep(self.step_seconds if pause is None else pause)
