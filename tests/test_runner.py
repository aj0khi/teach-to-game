"""Acceptance checks for the M1 practice runner."""

from __future__ import annotations

import time

import numpy as np
import pytest

from games.runner import RunnerEnv


def test_random_agent_can_play_1000_steps() -> None:
    """The runner can be restarted repeatedly without contract errors."""
    env = RunnerEnv()
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert "obstacles" in info

    steps = 0
    while steps < 1_000:
        observation, _, terminated, truncated, info = env.step(
            int(env.action_space.sample())
        )
        assert env.observation_space.contains(observation)
        assert isinstance(info["score"], float)
        steps += 1
        if terminated or truncated:
            env.reset()
    env.close()


def test_seed_reproduces_obstacle_layout() -> None:
    """The same seed produces the same obstacle positions."""
    first = RunnerEnv()
    second = RunnerEnv()
    first.reset(seed=123)
    second.reset(seed=123)
    assert first.obstacle_positions() == second.obstacle_positions()
    first.close()
    second.close()


def test_fast_mode_is_faster_than_realtime_mode() -> None:
    """Fast mode skips frame pacing used by the visible real-time mode."""
    fast = RunnerEnv(fast_mode=True)
    realtime = RunnerEnv(fast_mode=False)
    fast.reset(seed=1)
    realtime.reset(seed=1)

    start = time.perf_counter()
    for _ in range(30):
        fast.step(0)
    fast_duration = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(30):
        realtime.step(0)
    realtime_duration = time.perf_counter() - start

    assert fast_duration < realtime_duration / 3
    fast.close()
    realtime.close()


def test_rgb_array_render_has_expected_shape() -> None:
    """The no-window renderer supplies frames for later viewer milestones."""
    env = RunnerEnv(render_mode="rgb_array")
    env.reset(seed=4)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (180, 320, 3)
    env.close()


def test_invalid_action_is_rejected() -> None:
    """Actions outside the declared action space cannot reach game physics."""
    env = RunnerEnv()
    env.reset()
    with pytest.raises(ValueError, match="Action must be 0 or 1"):
        env.step(2)
    env.close()
