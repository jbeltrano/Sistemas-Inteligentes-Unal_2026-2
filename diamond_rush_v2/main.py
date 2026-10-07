from agent import Agent
from debug import test_recognition, test_recognition_all_levels
from perception.capture import BrowserCapturer, attach_driver
import config


def run_agent():
    driver = attach_driver()
    capturer = BrowserCapturer(driver)

    agent = Agent(capturer)
    agent.run()


def main():
    # test_recognition(config.BASE_DIR / "test.png")
    # test_recognition(config.LEVELS_DIR / "level19.png")
    # test_recognition_all_levels()
    run_agent()


if __name__ == "__main__":
    main()
