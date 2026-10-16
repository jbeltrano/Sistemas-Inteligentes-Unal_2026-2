from agent import Agent
from debug import test_recognition, test_recognition_all_levels
from perception.capture import BrowserCapturer, attach_driver
from play.control import Keyboard
import argparse

import config


def run_agent(execute: bool = True, debug: bool = False) -> None:
    driver = attach_driver()
    capturer = BrowserCapturer(driver)

    # El driver se ubica en el canvas una vez y se comparte entre la
    # percepcion (capturer) y el control (keyboard): Selenium manda las
    # flechas al mismo elemento que se captura.
    capturer.locate_canvas()
    keyboard = Keyboard(driver, capturer.canvas)

    agent = Agent(capturer, keyboard=keyboard, print_state=debug)
    agent.run(execute=execute)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="solo percibir y planificar, sin mandar teclas",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="imprimir el GameState completo",
    )
    args = parser.parse_args()

    # test_recognition(config.BASE_DIR / "test.png")
    # test_recognition(config.LEVELS_DIR / "level19.png")
    # test_recognition_all_levels()
    run_agent(execute=not args.no_execute, debug=args.debug)


if __name__ == "__main__":
    main()
