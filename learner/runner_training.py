"""PPO training and evaluation for the numeric runner environment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from games.runner import RunnerEnv


class RewardLogCallback(BaseCallback):
    """Write completed episode rewards to a CSV file during training."""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self.log_path = log_path
        self._file: Any = None
        self._writer: csv.writer | None = None

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["episode", "reward", "length"])

    def _on_step(self) -> bool:
        if self._writer is None:
            return True
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode = info["episode"]
                self._writer.writerow([self.num_timesteps, episode["r"], episode["l"]])
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()


def make_runner(seed: int | None = None) -> gym.Env[Any, Any]:
    """Create a fast runner wrapped for PPO episode statistics."""
    return gym.wrappers.RecordEpisodeStatistics(RunnerEnv(seed=seed))


def train_runner(
    output_dir: str | Path,
    *,
    total_timesteps: int = 20_000,
    seed: int = 0,
) -> Path:
    """Train PPO on the numeric runner and return its checkpoint path."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    environment = make_runner(seed=seed)
    environment.reset(seed=seed)
    environment.action_space.seed(seed)
    model = PPO(
        "MlpPolicy",
        environment,
        seed=seed,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        verbose=0,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=RewardLogCallback(destination / "rewards.csv"),
    )
    checkpoint = destination / "runner_ppo"
    model.save(checkpoint)
    environment.close()
    return checkpoint.with_suffix(".zip")


def evaluate_runner(
    checkpoint: str | Path,
    *,
    episodes: int = 10,
    seed: int = 10_000,
) -> list[int]:
    """Return time-alive steps for deterministic evaluation episodes."""
    model = PPO.load(checkpoint)
    results: list[int] = []
    for episode in range(episodes):
        environment = RunnerEnv(seed=seed + episode)
        observation, _ = environment.reset(seed=seed + episode)
        steps = 0
        while steps < 2_000:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(int(action))
            steps += 1
            if terminated or truncated:
                break
        results.append(steps)
        environment.close()
    return results


def evaluate_random(
    *,
    episodes: int = 10,
    seed: int = 10_000,
) -> list[int]:
    """Return time-alive steps for the seeded random baseline."""
    results: list[int] = []
    for episode in range(episodes):
        environment = RunnerEnv(seed=seed + episode)
        environment.reset(seed=seed + episode)
        environment.action_space.seed(seed + episode)
        steps = 0
        while steps < 2_000:
            _, _, terminated, truncated, _ = environment.step(
                int(environment.action_space.sample())
            )
            steps += 1
            if terminated or truncated:
                break
        results.append(steps)
        environment.close()
    return results
