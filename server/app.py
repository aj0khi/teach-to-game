"""FastAPI application for the local runner viewer."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from games.registry import BUILTIN_GAMES, create_game


class LiveTrainingCallback(BaseCallback):
    """Publish the exact PPO transitions being trained on."""

    def __init__(self, controller: TrainingController) -> None:
        super().__init__()
        self.controller = controller
        self.attempt = 0
        self.attempt_steps = 0
        self.attempt_reward = 0.0

    def _on_step(self) -> bool:
        """Stream a rendered observation and stop/pause at a safe step boundary."""
        if self.controller._stop.is_set():
            return False
        while self.controller._pause.is_set() and not self.controller._stop.is_set():
            self.controller._stop.wait(0.05)
        if self.controller._stop.is_set():
            return False

        action = int(np.asarray(self.locals["actions"]).reshape(-1)[0])
        reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
        info = self.locals["infos"][0]
        self.attempt_steps += 1
        self.controller._game_steps += 1
        self.attempt_reward += reward
        if self.attempt_steps % 2 == 0 or self.locals["dones"][0]:
            frame = self.training_env.env_method("render")[0]
            self.controller.publish_frame(
                frame,
                action,
                reward,
                float(info.get("score", 0.0)),
                int(info.get("length", 0)),
                self.attempt_steps,
            )
            time.sleep(self.controller.frame_delay())
        if self.locals["dones"][0]:
            self.attempt += 1
            self.controller.publish_attempt(
                self.attempt,
                self.attempt_steps,
                self.attempt_reward,
                float(info.get("score", 0.0)),
                int(info.get("length", 2)),
            )
            time.sleep(0.8 / self.controller.watch_speed)
            self.attempt_steps = 0
            self.attempt_reward = 0.0
        return True


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
        self._started_at: float | None = None
        self._game_steps = 0
        self._best_score = 0.0
        self._target_beaten = False
        self._status = "stopped"
        self._watch_speed = 1.0
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
            self._started_at = time.monotonic()
            self._game_steps = 0
            self._best_score = 0.0
            self._target_beaten = False
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

    def elapsed_seconds(self) -> float:
        """Return real time elapsed in the active or most recent session."""
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def session_metrics(self) -> dict[str, Any]:
        """Return live experiment metrics for the dashboard header."""
        setup = self.get_setup()
        target = float(setup["baseline"])
        return {
            "attempt": self._attempt,
            "game_steps": self._game_steps,
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "best_score": self._best_score,
            "target_score": target,
            "target_beaten": self._target_beaten,
        }

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

    def set_watch_speed(self, speed: float) -> float:
        """Set the human-facing playback pace without changing game physics."""
        if speed not in {0.5, 1.0, 2.0}:
            raise ValueError("Watch speed must be 0.5, 1.0, or 2.0")
        with self._lock:
            self._watch_speed = speed
        return speed

    @property
    def watch_speed(self) -> float:
        """Return the current dashboard playback pace."""
        with self._lock:
            return self._watch_speed

    def frame_delay(self) -> float:
        """Return the delay between visible frames at the selected watch speed."""
        return 0.12 / self.watch_speed

    def _publish(self, event: dict[str, Any]) -> None:
        """Append an event while dropping stale events when clients lag."""
        with self._lock:
            if event["type"] == "frame":
                self._latest_frame = event
            else:
                self._events.append(event)

    def publish_frame(
        self,
        frame: np.ndarray,
        action: int,
        reward: float,
        score: float,
        length: int = 0,
        steps: int = 0,
    ) -> None:
        """Publish a current RGB frame and the transition that produced it."""
        if frame.shape[1] > 160:
            frame = frame[::2, ::2]
        self._publish(
            {
                "type": "frame",
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "pixels": np.asarray(frame, dtype=np.uint8).tobytes().hex(),
                "action": action,
                "reward": reward,
                "score": score,
                "length": length,
                "steps": steps,
                "target_score": self.get_setup()["baseline"],
            }
        )

    def publish_attempt(
        self, attempt: int, steps: int, reward: float, score: float, length: int
    ) -> None:
        """Publish a completed attempt from the actual environment rollout."""
        self._attempt = attempt
        self._best_score = max(self._best_score, score)
        target = float(self.get_setup()["baseline"])
        self._target_beaten = target > 0 and self._best_score >= target
        self._publish(
            {
                "type": "attempt",
                "attempt": attempt,
                "steps": steps,
                "reward": reward,
                "score": score,
                "length": length,
                "target_score": self.get_setup()["baseline"],
                "elapsed_seconds": round(self.elapsed_seconds(), 1),
                "game_steps": self._game_steps,
                "best_score": self._best_score,
                "target_beaten": self._target_beaten,
                "goal": self.get_setup()["goal"],
            }
        )

    def _train(self) -> None:
        """Train one controller-owned model on the same frames being streamed."""
        setup = self.get_setup()
        environment = create_game(setup["game"], target_score=int(setup["baseline"]))
        try:
            observation, info = environment.reset()
            self.publish_frame(
                environment.render(),
                action=-1,
                reward=0.0,
                score=float(info.get("score", 0.0)),
                length=int(info.get("length", 0)),
                steps=int(info.get("steps", 0)),
            )
            if setup["algorithm"] == "ppo":
                model = PPO(
                    "MlpPolicy", environment, n_steps=128, batch_size=32, verbose=0
                )
                model.learn(
                    total_timesteps=100_000,
                    callback=LiveTrainingCallback(self),
                )
            else:
                self._run_random(environment)
        finally:
            environment.close()
            with self._lock:
                self._status = "stopped"

    def _run_random(self, environment: Any) -> None:
        """Run the explicit random baseline with the same live stream."""
        for _ in range(100):
            if self._stop.is_set():
                break
            observation, _ = environment.reset()
            reward_total = 0.0
            steps = 0
            info: dict[str, Any] = {"score": 0.0, "length": 0, "steps": 0}
            while steps < 600 and not self._stop.is_set():
                action = int(environment.action_space.sample())
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = environment.step(action)
                reward_total += reward
                steps += 1
                self._game_steps += 1
                if steps % 2 == 0 or terminated or truncated:
                    self.publish_frame(
                        environment.render(),
                        action,
                        reward,
                        info["score"],
                        int(info.get("length", 0)),
                        steps,
                    )
                    time.sleep(self.frame_delay())
                if terminated or truncated:
                    break
            self.publish_attempt(
                self._attempt + 1,
                steps,
                reward_total,
                info["score"],
                int(info.get("length", 0)),
            )
            time.sleep(0.8 / self.watch_speed)


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
    return {
        "status": controller.status,
        "setup": controller.get_setup(),
        "watch_speed": controller.watch_speed,
        "session": controller.session_metrics(),
    }


@app.get("/api/games")
def games() -> dict[str, Any]:
    """List built-in games available from a fresh installation."""
    return {"games": [{"id": key, **value} for key, value in BUILTIN_GAMES.items()]}


@app.post("/api/setup")
def configure(values: dict[str, Any]) -> dict[str, Any]:
    """Save the setup wizard configuration."""
    return {"setup": controller.setup(values)}


@app.post("/api/watch-speed")
def watch_speed(values: dict[str, float]) -> dict[str, float]:
    """Change the visible playback pace."""
    return {"watch_speed": controller.set_watch_speed(float(values["speed"]))}


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
