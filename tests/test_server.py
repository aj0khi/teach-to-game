"""Acceptance checks for the M3 local viewer API."""

from fastapi.testclient import TestClient

from server.app import app, controller


def test_dashboard_and_status_are_available() -> None:
    """The local app serves the dashboard and reports its initial state."""
    with TestClient(app) as client:
        page = client.get("/")
        status = client.get("/api/status")

    assert page.status_code == 200
    assert "Teach to Game" in page.text
    assert status.json()["status"] == "stopped"
    assert status.json()["setup"]["game"] == "snake"


def test_training_controls_change_status() -> None:
    """Start, pause, resume, and stop are exposed through the local API."""
    controller.stop()
    with TestClient(app) as client:
        assert client.post("/api/start").json()["status"] == "running"
        assert client.post("/api/pause").json()["status"] == "paused"
        assert client.post("/api/resume").json()["status"] == "running"
        assert client.post("/api/stop").json()["status"] == "stopped"


def test_setup_can_select_snake_and_random_baseline() -> None:
    """The setup API persists game, algorithm, goal, and baseline selections."""
    controller.stop()
    with TestClient(app) as client:
        response = client.post(
            "/api/setup",
            json={
                "game": "snake",
                "algorithm": "random",
                "goal": "Beat five points",
                "baseline": 5,
            },
        )
    assert response.status_code == 200
    assert response.json()["setup"] == {
        "game": "snake",
        "algorithm": "random",
        "goal": "Beat five points",
        "baseline": 5.0,
    }


def test_frame_events_keep_only_the_newest_frame() -> None:
    """A slow browser receives a current frame without blocking training."""
    controller._publish({"type": "frame", "pixels": "old"})
    controller._publish({"type": "frame", "pixels": "new"})
    events = controller.snapshot()
    assert events == [{"type": "frame", "pixels": "new"}]
