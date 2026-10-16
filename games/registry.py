"""Factories for games selected through the setup screen."""

from __future__ import annotations

from typing import Any

from games.runner import RunnerEnv
from games.snake import SnakeEnv

BUILTIN_GAMES = {
    "runner": {
        "name": "Practice Runner",
        "description": "Learn when to jump over approaching obstacles.",
    },
    "snake": {
        "name": "Snake",
        "description": "Collect food, grow longer, and avoid the walls.",
    },
}


def create_game(game_id: str, *, fast_mode: bool = True) -> Any:
    """Create a configured built-in game by its stable identifier."""
    if game_id == "runner":
        return RunnerEnv(fast_mode=fast_mode)
    if game_id == "snake":
        return SnakeEnv()
    raise ValueError(f"Unknown built-in game: {game_id}")
