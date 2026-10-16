"""Adapters for user-owned browser pages and desktop game windows.

These adapters intentionally require an explicit configuration. They do not
guess game state or send input until the caller constructs and permits them.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np

from games.base import GameEnv


class BrowserGame(GameEnv):
    """Drive a user-owned browser game with Playwright screenshots and keys."""

    def __init__(
        self,
        url: str,
        *,
        actions: list[list[str]],
        capture_size: tuple[int, int] = (84, 84),
        step_hz: int = 15,
        headless: bool = False,
    ) -> None:
        self.url = url
        self.actions = actions
        self.capture_size = capture_size
        self.step_interval = 1 / step_hz
        self.headless = headless
        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*capture_size, 3), dtype=np.uint8
        )
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Open or reload the configured page and capture its first frame."""
        del seed, options
        self._ensure_page()
        self._page.reload()
        return self._capture(), {"url": self.url}

    def step(self, action: int):
        """Press one configured key combination and capture the next frame."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be between 0 and {len(self.actions) - 1}")
        started = time.perf_counter()
        for key in self.actions[int(action)]:
            self._page.keyboard.press(key)
        time.sleep(self.step_interval)
        observation = self._capture()
        return (
            observation,
            0.0,
            False,
            False,
            {"capture_to_input_seconds": time.perf_counter() - started},
        )

    def render(self) -> np.ndarray:
        """Return the current browser capture."""
        return self._capture()

    def close(self) -> None:
        """Close the browser session owned by this adapter."""
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = None
        self._playwright = None
        self._page = None

    def _ensure_page(self) -> None:
        """Start Playwright lazily so importing the app stays lightweight."""
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError(
                "Browser games require the optional 'browser' dependencies."
            ) from error
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page(
            viewport={"width": self.capture_size[0], "height": self.capture_size[1]}
        )
        self._page.goto(self.url)

    def _capture(self) -> np.ndarray:
        """Capture and normalize the page screenshot to the observation shape."""
        image = self._page.screenshot(type="png")
        try:
            from io import BytesIO

            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Browser games require Pillow.") from error
        return np.asarray(
            Image.open(BytesIO(image)).convert("RGB").resize(self.capture_size)
        )


class ScreenGame(GameEnv):
    """Capture a configured desktop rectangle and send explicit key presses."""

    def __init__(
        self,
        region: tuple[int, int, int, int],
        *,
        actions: list[list[str]],
        capture_size: tuple[int, int] = (84, 84),
        step_hz: int = 15,
    ) -> None:
        self.region = region
        self.actions = actions
        self.step_interval = 1 / step_hz
        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*capture_size, 3), dtype=np.uint8
        )
        self.capture_size = capture_size
        self._mss: Any = None
        self._keyboard: Any = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Capture the configured screen area without sending an automatic key."""
        del seed, options
        self._ensure_tools()
        return self._capture(), {"region": self.region}

    def step(self, action: int):
        """Send only a configured action, then capture the next screen frame."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be between 0 and {len(self.actions) - 1}")
        started = time.perf_counter()
        for key in self.actions[int(action)]:
            self._keyboard.press(key)
        time.sleep(self.step_interval)
        return (
            self._capture(),
            0.0,
            False,
            False,
            {"capture_to_input_seconds": time.perf_counter() - started},
        )

    def render(self) -> np.ndarray:
        """Return the current screen capture."""
        return self._capture()

    def close(self) -> None:
        """Release screen capture and keyboard resources."""
        if self._mss is not None:
            self._mss.close()
        self._mss = None
        self._keyboard = None

    def _ensure_tools(self) -> None:
        """Load desktop automation libraries only when the adapter is used."""
        try:
            import mss
            import pyautogui
        except ImportError as error:
            raise RuntimeError(
                "Desktop games require the optional 'desktop' dependencies."
            ) from error
        self._mss = mss.mss()
        self._keyboard = pyautogui

    def _capture(self) -> np.ndarray:
        """Capture the configured rectangle and normalize its size."""
        from PIL import Image

        left, top, width, height = self.region
        raw = np.asarray(
            self._mss.grab({"left": left, "top": top, "width": width, "height": height})
        )
        return np.asarray(Image.fromarray(raw[:, :, :3]).resize(self.capture_size))
