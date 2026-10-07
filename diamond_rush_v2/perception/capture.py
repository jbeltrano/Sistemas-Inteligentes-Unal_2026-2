"""
Captura del área de juego de Diamond Rush vía Selenium.

Este módulo no crea ni cierra el navegador -- eso es responsabilidad
de agent.py, que decide cuál de las dos funciones de más abajo usar y
le pasa el driver resultante a BrowserCapturer. Es la misma instancia
que después va a necesitar play/control.py para mandar teclas, así
que no tiene sentido que cada módulo abra su propio navegador.

Dos formas de obtener el driver:

- new_driver(): abre un Chrome nuevo y navega a config.GAME_URL. Sirve
  para pruebas sueltas de este archivo, pero arranca siempre desde el
  principio del juego (perfil vacío, sin tutorial pasado).
- attach_driver(): se conecta a un Chrome que YA está abierto (con
  --remote-debugging-port activo) sin navegar a ningún lado. Es el
  modo real de uso: el navegador se abre a mano una vez, se pasa el
  tutorial y se deja parado en el nivel que haga falta: el agente solo
  mira lo que ya está en pantalla, sin tocarlo.
"""
import time
from typing import Iterator

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

import config


def new_driver() -> webdriver.Chrome:
    """Chrome nuevo, navegando directo al juego. Para pruebas sueltas de este archivo."""
    options = webdriver.ChromeOptions()
    options.add_argument(f"--window-size={config.WINDOW_SIZE}")
    driver = webdriver.Chrome(options=options)
    driver.get(config.GAME_URL)
    return driver


def attach_driver() -> webdriver.Chrome:
    """
    Se conecta a un Chrome que ya está corriendo con
    --remote-debugging-port (ver config.DEBUGGER_ADDRESS), sin navegar
    a ningún lado.

    No llamar a driver.quit() sobre lo que devuelve esto: cerraría el
    navegador que se dejó abierto a mano con el progreso del profesor.
    """
    options = webdriver.ChromeOptions()
    options.add_experimental_option("debuggerAddress", config.DEBUGGER_ADDRESS)
    return webdriver.Chrome(options=options)


class BrowserCapturer:
    """Entrega frames del canvas del juego a partir de un driver ya creado."""

    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver
        self.canvas: WebElement = None

    def locate_canvas(self) -> None:
        """
        Ubica el canvas dentro del iframe del juego. No navega a
        ningún lado -- asume que el navegador ya está mostrando la
        página correcta, para no perder el progreso de una partida ya
        abierta a mano.
        """
        self.driver.switch_to.default_content()
        iframe = self.driver.find_element(By.ID, config.IFRAME_ID)
        self.driver.switch_to.frame(iframe)
        self.canvas = self.driver.find_element(By.TAG_NAME, "canvas")

        size = self.canvas.size
        print(
            f"Canvas renderizando a {size['width']}x{size['height']} "
            f"(esperado: cerca de 640x960)"
        )

    def capture(self) -> np.ndarray:
        """Un frame del canvas, en BGR (mismo formato que cv2.imread)."""
        if self.canvas is None:
            raise RuntimeError("Llamá a locate_canvas() antes de capturar")

        png_bytes = self.canvas.screenshot_as_png
        buffer = np.frombuffer(png_bytes, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def stream(self, fps: int = 5) -> Iterator[np.ndarray]:
        """Generador de frames a una frecuencia aproximada de `fps`."""
        period = 1 / fps
        while True:
            start = time.perf_counter()
            yield self.capture()
            remaining = period - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    # Prueba suelta de este archivo: acá sí tiene sentido crear y
    # cerrar el navegador en el mismo lugar, porque es un Chrome
    # nuevo que no comparte estado con nadie más.
    driver = new_driver()
    capturer = BrowserCapturer(driver)
    capturer.locate_canvas()

    frame = capturer.capture()
    cv2.imwrite("captura_selenium.png", frame)
    print(f"Guardado captura_selenium.png ({frame.shape[1]}x{frame.shape[0]})")

    driver.quit()