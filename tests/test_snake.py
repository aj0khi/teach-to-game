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


def test_snake_target_score_ends_episode_after_growth() -> None:
    """A configured target makes the requested score a real game objective."""
    env = SnakeEnv(target_score=1)
    env.reset(seed=12)
    env._food = (env._snake[0][0] + 1, env._snake[0][1])

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == 2.0
    assert terminated is True
    assert truncated is False
    assert info["score"] == 1
    assert info["length"] == 3
    assert info["target_reached"] is True
