"""Acceptance checks for the M2 numeric PPO learning loop."""

from __future__ import annotations

from learner.runner_training import (
    evaluate_random,
    evaluate_runner,
    train_runner,
)


def test_ppo_runner_beats_random_baseline(tmp_path) -> None:
    """A short training run learns to jump substantially longer than random."""
    checkpoint = train_runner(tmp_path, total_timesteps=20_000, seed=21)
    learned = evaluate_runner(checkpoint, episodes=5, seed=40)
    random = evaluate_random(episodes=5, seed=40)

    assert checkpoint.exists()
    assert sum(learned) / len(learned) >= 3 * sum(random) / len(random)
