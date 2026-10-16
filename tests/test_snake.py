"""Acceptance checks for the built-in Snake game."""

from games.snake import SnakeEnv


def test_snake_follows_game_contract_for_1000_steps() -> None:
    """A random policy can play and restart Snake without adapter errors."""
    env = SnakeEnv()
    observation, _ = env.reset(seed=4)
    assert env.observation_space.contains(observation)
    for _ in range(1_000):
        observation, _, terminated, truncated, info = env.step(
            int(env.action_space.sample())
        )
        assert env.observation_space.contains(observation)
        assert "score" in info
        if terminated or truncated:
            env.reset()
    env.close()


def test_snake_seed_reproduces_first_food() -> None:
    """The same seed creates the same initial board."""
    first = SnakeEnv()
    second = SnakeEnv()
    first.reset(seed=12)
    second.reset(seed=12)
    assert first.render().tolist() == second.render().tolist()
