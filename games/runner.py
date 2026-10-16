"""A small deterministic one-button runner for fast reinforcement learning."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from games.base import GameEnv


@dataclass(frozen=True)
class Obstacle:
    """An obstacle's horizontal position and width."""

    x: float
    width: float


class RunnerEnv(GameEnv):
    """Side-scrolling runner with a jump action and numeric observations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        fast_mode: bool = True,
        seed: int | None = None,
    ) -> None:
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.fast_mode = fast_mode
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng(seed)
        self._obstacles: list[Obstacle] = []
        self._player_y = 0.0
        self._velocity = 0.0
        self._distance = 0.0
        self._steps = 0
        self._terminated = False
        self._pygame: Any = None
        self._screen: Any = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an attempt and generate its obstacle layout."""
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._obstacles = self._generate_obstacles()
        self._player_y = 0.0
        self._velocity = 0.0
        self._distance = 0.0
        self._steps = 0
        self._terminated = False
        return self._observation(), {"obstacles": tuple(self._obstacles)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance physics by one frame and return the learner contract."""
        if self._terminated:
            raise RuntimeError("step() called after the attempt ended; call reset()")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be 0 or 1, got {action}")
        if action == 1 and self._player_y == 0.0:
            self._velocity = 0.42
        self._velocity -= 0.08
        self._player_y = max(0.0, self._player_y + self._velocity)
        self._distance += 0.08
        self._steps += 1
        collision = self._collides()
        self._terminated = collision
        truncated = self._steps >= 2_000
        reward = -1.0 if collision else 0.08
        if truncated:
            self._terminated = True
        if not self.fast_mode:
            time.sleep(1 / self.metadata["render_fps"])
        info = {
            "score": self._distance,
            "distance": self._distance,
            "attempt_steps": self._steps,
            "collision": collision,
        }
        return self._observation(), reward, collision, truncated, info

    def render(self) -> np.ndarray | None:
        """Return an RGB frame, or draw the frame in a pygame window."""
        frame = self._frame()
        if self.render_mode == "human":
            self._render_human(frame)
            return None
        return frame

    def close(self) -> None:
        """Release the optional visible renderer."""
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None

    def obstacle_positions(self) -> tuple[float, ...]:
        """Return obstacle positions for deterministic-layout checks."""
        return tuple(obstacle.x for obstacle in self._obstacles)

    def _generate_obstacles(self) -> list[Obstacle]:
        """Generate a seeded obstacle layout far enough apart to learn."""
        positions: list[Obstacle] = []
        x = 5.0
        for _ in range(40):
            x += float(self._rng.uniform(2.5, 4.5))
            positions.append(Obstacle(x=x, width=0.35))
        return positions

    def _next_obstacle(self) -> Obstacle:
        """Return the first obstacle the player has not passed."""
        for obstacle in self._obstacles:
            if obstacle.x + obstacle.width >= self._distance:
                return obstacle
        return Obstacle(self._distance + 10.0, 0.35)

    def _collides(self) -> bool:
        """Check whether the player is on the ground when an obstacle arrives."""
        obstacle = self._next_obstacle()
        horizontal_distance = obstacle.x - self._distance
        return 0.0 <= horizontal_distance <= 0.12 and self._player_y < 0.22

    def _observation(self) -> np.ndarray:
        """Build normalized numeric features for the learner."""
        obstacle = self._next_obstacle()
        return np.array(
            [
                self._player_y,
                np.clip(self._velocity / 0.42, -1.0, 1.0),
                np.clip((obstacle.x - self._distance) / 4.5, 0.0, 1.0),
                obstacle.width,
            ],
            dtype=np.float32,
        )

    def _frame(self) -> np.ndarray:
        """Draw a compact RGB representation without requiring a window."""
        frame = np.full((180, 320, 3), 245, dtype=np.uint8)
        frame[145:148, :, :] = (45, 45, 45)
        player_x = 55
        player_bottom = 145 - int(self._player_y * 100)
        frame[max(0, player_bottom - 20) : player_bottom, player_x : player_x + 16] = (
            40,
            100,
            220,
        )
        for obstacle in self._obstacles:
            relative_x = int((obstacle.x - self._distance) * 40 + player_x)
            if -10 < relative_x < frame.shape[1]:
                frame[125:145, max(0, relative_x) : min(320, relative_x + 14)] = (
                    220,
                    60,
                    50,
                )
        return frame

    def _render_human(self, frame: np.ndarray) -> None:
        """Display a frame using pygame, imported only for visible mode."""
        if self._pygame is None:
            try:
                import pygame
            except ImportError as error:
                raise RuntimeError(
                    "Visible mode requires pygame; install the project dependencies."
                ) from error
            self._pygame = pygame
            self._pygame.init()
            self._screen = self._pygame.display.set_mode((320, 180))
            self._pygame.display.set_caption("Teach to Game - Runner")
        surface = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        self._pygame.display.flip()
