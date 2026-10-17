from agent import Agent
from debug import test_recognition, test_recognition_all_levels
from perception.capture import BrowserCapturer, attach_driver
from play.control import Keyboard
import argparse

import config


def run_agent(
    execute: bool = True, debug: bool = False, verify_every: int = None
) -> None:
    driver = attach_driver()
    capturer = BrowserCapturer(driver)

    # El driver se ubica en el canvas una vez y se comparte entre la
    # percepcion (capturer) y el control (keyboard): Selenium manda las
    # flechas al mismo elemento que se captura.
    capturer.locate_canvas()
    keyboard = Keyboard(driver, capturer.canvas)

    kwargs = {} if verify_every is None else {"verify_every": verify_every}
    agent = Agent(capturer, keyboard=keyboard, print_state=debug, **kwargs)
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
    parser.add_argument(
        "--verify-every",
        type=int,
        default=None,
        help="re-percibir cada N pasos (0 = solo al final); por defecto, config.VERIFY_EVERY",
    )
    args = parser.parse_args()

    # test_recognition(config.BASE_DIR / "test.png")
    # test_recognition(config.LEVELS_DIR / "level19.png")
    # test_recognition_all_levels()
    run_agent(
        execute=not args.no_execute,
        debug=args.debug,
        verify_every=args.verify_every,
    )


if __name__ == "__main__":
    main()
