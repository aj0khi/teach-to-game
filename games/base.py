"""Shared interface for game environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class GameEnv(gym.Env[Any, int]):
    """Base class for games exposed to the learner."""

    metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"]}

    def render(self) -> Any:
        """Return the current frame or display it in the configured view."""
        raise NotImplementedError
