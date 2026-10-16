"""FastAPI application for the local runner viewer."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from stable_baselines3 import PPO

from games.runner import RunnerEnv


class TrainingController:
    """Run a small PPO loop separately from the web request handlers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=100)
        self._attempt = 0
        self._status = "stopped"

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
        return events

    def _publish(self, event: dict[str, Any]) -> None:
        """Append an event while dropping stale events when clients lag."""
        with self._lock:
            self._events.append(event)

    def _train(self) -> None:
        """Train one controller-owned model and publish attempt metrics."""
        environment = RunnerEnv()
        model = PPO("MlpPolicy", environment, n_steps=128, batch_size=32, verbose=0)
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
                    action, _ = model.predict(observation, deterministic=False)
                    observation, reward, terminated, truncated, info = environment.step(
                        int(action)
                    )
                    reward_total += reward
                    steps += 1
                    if terminated or truncated:
                        break
                environment.reset()
                model.learn(total_timesteps=max(1, steps), reset_num_timesteps=False)
                self._attempt += 1
                self._publish(
                    {
                        "type": "attempt",
                        "attempt": self._attempt,
                        "steps": steps,
                        "reward": reward_total,
                        "score": info["score"],
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
def status() -> dict[str, str]:
    """Return the training controller status."""
    return {"status": controller.status}


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
