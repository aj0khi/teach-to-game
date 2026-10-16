"""A compact numeric-observation Snake environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from games.base import GameEnv


class SnakeEnv(GameEnv):
    """Grid Snake with four directional actions and a score reward."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        *,
        grid_size: int = 10,
        target_score: int = 0,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if target_score < 0:
            raise ValueError("target_score must be non-negative")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.grid_size = grid_size
        self.target_score = target_score
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._snake: list[tuple[int, int]] = []
        self._food = (0, 0)
        self._direction = 0
        self._steps = 0
        self._score = 0
        self._terminated = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new game with a seeded snake and food position."""
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        center = self.grid_size // 2
        self._snake = [(center, center), (center - 1, center)]
        self._direction = 0
        self._steps = 0
        self._score = 0
        self._terminated = False
        self._place_food()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Move the snake one cell and return the standard game contract."""
        if self._terminated:
            raise RuntimeError("step() called after the game ended; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be between 0 and 3, got {action}")
        if not self._is_reverse(int(action)):
            self._direction = int(action)
        head_x, head_y = self._snake[0]
        delta = ((1, 0), (0, 1), (-1, 0), (0, -1))[self._direction]
        head = (head_x + delta[0], head_y + delta[1])
        self._steps += 1
        ate_food = head == self._food
        collision = (
            not (0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size)
            or head in self._snake[:-1]
        )
        if collision:
            self._terminated = True
            return self._observation(), -1.0, True, False, self._info()
        self._snake.insert(0, head)
        if ate_food:
            self._score += 1
            self._place_food()
        else:
            self._snake.pop()
        truncated = self._steps >= self.grid_size * self.grid_size * 4
        target_reached = self.target_score > 0 and self._score >= self.target_score
        self._terminated = truncated or target_reached
        reward = 1.0 if ate_food else 0.01
        if target_reached:
            reward += 1.0
        return self._observation(), reward, target_reached, truncated, self._info()

    def render(self) -> np.ndarray:
        """Return a simple RGB board suitable for the dashboard."""
        frame = np.full(
            (self.grid_size * 20, self.grid_size * 20, 3), 24, dtype=np.uint8
        )
        for x, y in self._snake:
            frame[y * 20 + 2 : y * 20 + 18, x * 20 + 2 : x * 20 + 18] = (70, 210, 120)
        food_x, food_y = self._food
        frame[
            food_y * 20 + 4 : food_y * 20 + 16,
            food_x * 20 + 4 : food_x * 20 + 16,
        ] = (
            240,
            90,
            90,
        )
        return frame

    def close(self) -> None:
        """Release resources; Snake does not own an external window."""

    def _is_reverse(self, action: int) -> bool:
        """Prevent an immediate 180-degree turn into the snake body."""
        return (action - self._direction) % 2 == 0 and action != self._direction

    def _place_food(self) -> None:
        """Place food on an unoccupied cell."""
        available = [
            (x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if (x, y) not in self._snake
        ]
        self._food = available[int(self._rng.integers(len(available)))]

    def _observation(self) -> np.ndarray:
        """Expose danger, food direction, and current heading as normalized values."""
        head_x, head_y = self._snake[0]
        directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        dx, dy = directions[self._direction]
        next_cell = (head_x + dx, head_y + dy)
        danger = [
            float(
                not (
                    0 <= next_cell[0] < self.grid_size
                    and 0 <= next_cell[1] < self.grid_size
                )
                or next_cell in self._snake
            )
        ]
        return np.array(
            danger
            + [
                float(self._food[0] < head_x),
                float(self._food[0] > head_x),
                float(self._food[1] < head_y),
                float(self._food[1] > head_y),
                float(self._direction == 0),
                float(self._direction == 1),
                float(self._direction == 2),
                float(self._direction == 3),
                self._score / (self.grid_size * self.grid_size),
                len(self._snake) / (self.grid_size * self.grid_size),
            ],
            dtype=np.float32,
        )

    def _info(self) -> dict[str, Any]:
        """Return viewer-friendly game metrics."""
        return {
            "score": self._score,
            "length": len(self._snake),
            "steps": self._steps,
            "target_score": self.target_score,
            "target_reached": self.target_score > 0
            and self._score >= self.target_score,
        }
