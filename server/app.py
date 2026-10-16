"""FastAPI application for the local runner viewer."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from stable_baselines3 import PPO

from games.registry import BUILTIN_GAMES, create_game


class TrainingController:
    """Run a small PPO loop separately from the web request handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=100)
        self._latest_frame: dict[str, Any] | None = None
        self._attempt = 0
        self._status = "stopped"
        self._setup: dict[str, Any] = {
            "game": "snake",
            "algorithm": "ppo",
            "goal": "Collect as much food as possible without crashing.",
            "baseline": 0,
        }

    @property
    def status(self) -> str:
        """Return the current controller status."""
        with self._lock:
            return self._status

    def start(self) -> None:
        """Start training if it is not already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._pause.clear()
            self._stop.clear()
            self._status = "running"
            self._thread = threading.Thread(target=self._train, daemon=True)
            self._thread.start()

    def pause(self) -> None:
        """Pause the training loop at its next step."""
        self._pause.set()
        with self._lock:
            if self._status == "running":
                self._status = "paused"

    def resume(self) -> None:
        """Resume a paused training loop."""
        self._pause.clear()
        with self._lock:
            if self._status == "paused":
                self._status = "running"

    def stop(self) -> None:
        """Request a prompt stop and wait briefly for the worker."""
        self._stop.set()
        self._pause.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            self._status = "stopped"

    def snapshot(self) -> list[dict[str, Any]]:
        """Return and clear queued events for a WebSocket client."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
            if self._latest_frame is not None:
                events.append(self._latest_frame)
                self._latest_frame = None
        return events

    def setup(self, values: dict[str, Any]) -> dict[str, Any]:
        """Validate and store the setup wizard selections."""
        if (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop.is_set()
        ):
            raise RuntimeError("Stop training before changing the setup")
        game = str(values.get("game", "snake"))
        algorithm = str(values.get("algorithm", "ppo"))
        if game not in BUILTIN_GAMES:
            raise ValueError(f"Unknown built-in game: {game}")
        if algorithm not in {"ppo", "random"}:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self._setup = {
            "game": game,
            "algorithm": algorithm,
            "goal": str(values.get("goal", "")).strip() or "Improve the score.",
            "baseline": max(0, float(values.get("baseline", 0))),
        }
        return self._setup.copy()

    def get_setup(self) -> dict[str, Any]:
        """Return the active setup selections."""
        return self._setup.copy()

    def _publish(self, event: dict[str, Any]) -> None:
        """Append an event while dropping stale events when clients lag."""
        with self._lock:
            if event["type"] == "frame":
                self._latest_frame = event
            else:
                self._events.append(event)

    def _train(self) -> None:
        """Train one controller-owned model and publish attempt metrics."""
        setup = self.get_setup()
        environment = create_game(setup["game"])
        model = (
            PPO("MlpPolicy", environment, n_steps=128, batch_size=32, verbose=0)
            if setup["algorithm"] == "ppo"
            else None
        )
        try:
            for _ in range(100):
                if self._stop.is_set():
                    break
                while self._pause.is_set() and not self._stop.is_set():
                    self._stop.wait(0.05)
                observation, _ = environment.reset()
                reward_total = 0.0
                steps = 0
                info: dict[str, Any] = {"score": 0.0}
                while steps < 600 and not self._stop.is_set():
                    action = (
                        environment.action_space.sample()
                        if model is None
                        else model.predict(observation, deterministic=False)[0]
                    )
                    observation, reward, terminated, truncated, info = environment.step(
                        int(action)
                    )
                    reward_total += reward
                    steps += 1
                    if steps % 3 == 0 or terminated or truncated:
                        frame = environment.render()
                        if frame.shape[1] > 160:
                            frame = frame[::2, ::2]
                        self._publish(
                            {
                                "type": "frame",
                                "width": int(frame.shape[1]),
                                "height": int(frame.shape[0]),
                                "pixels": np.asarray(frame, dtype=np.uint8)
                                .tobytes()
                                .hex(),
                                "action": int(action),
                                "reward": reward,
                                "score": info["score"],
                            }
                        )
                    if terminated or truncated:
                        break
                environment.reset()
                if model is not None:
                    model.learn(
                        total_timesteps=max(1, steps), reset_num_timesteps=False
                    )
                self._attempt += 1
                self._publish(
                    {
                        "type": "attempt",
                        "attempt": self._attempt,
                        "steps": steps,
                        "reward": reward_total,
                        "score": info["score"],
                        "goal": setup["goal"],
                    }
                )
        finally:
            environment.close()
            with self._lock:
                self._status = "stopped"


app = FastAPI(title="Teach to Game")
controller = TrainingController()
UI_PATH = Path(__file__).parents[1] / "ui" / "index.html"


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page local dashboard."""
    return FileResponse(UI_PATH)


@app.get("/api/status")
def status() -> dict[str, Any]:
    """Return the training controller status."""
    return {"status": controller.status, "setup": controller.get_setup()}


@app.get("/api/games")
def games() -> dict[str, Any]:
    """List built-in games available from a fresh installation."""
    return {"games": [{"id": key, **value} for key, value in BUILTIN_GAMES.items()]}


@app.post("/api/setup")
def configure(values: dict[str, Any]) -> dict[str, Any]:
    """Save the setup wizard configuration."""
    return {"setup": controller.setup(values)}


@app.post("/api/{command}")
def command(command: str) -> dict[str, str]:
    """Handle start, pause, resume, and stop controls."""
    handlers = {
        "start": controller.start,
        "pause": controller.pause,
        "resume": controller.resume,
        "stop": controller.stop,
    }
    if command not in handlers:
        return {"status": "unknown command"}
    handlers[command]()
    return {"status": controller.status}


@app.websocket("/ws")
async def metrics(websocket: WebSocket) -> None:
    """Stream queued attempt metrics without blocking the learner."""
    await websocket.accept()
    try:
        while True:
            for event in controller.snapshot():
                await websocket.send_json(event)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        return
