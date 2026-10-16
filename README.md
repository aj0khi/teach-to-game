# Teach to Game

Teach to Game is a local reinforcement-learning lab. It lets you choose a
game, describe the goal, set a target score, and watch an agent improve without
sending training data to a cloud service.

## Quick start

```bash
git clone <repository-url>
cd teach-to-game
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

Open <http://127.0.0.1:8765>. Choose **Snake**, save the goal, and press
**Start learning**. Snake and the runner are self-contained practice games, so
they work on a fresh laptop without a separate game installation.

For Snake, the target score is a real episode objective: each apple adds one
segment, and reaching the target completes the episode successfully. A target
of zero means there is no score cutoff; it does not add segments automatically.

## Connecting a real game

Real games need a user-provided adapter configuration: the game URL or screen
rectangle, the keys the agent is allowed to press, how the game ends, and the
reward signal. The app must not guess these details because a wrong reward can
teach an agent to exploit the setup instead of playing well.

For a browser game, install the optional adapter dependencies and the browser
runtime:

```bash
.venv/bin/python -m pip install -e '.[browser]'
.venv/bin/python -m playwright install chromium
```

For a downloaded desktop game:

```bash
.venv/bin/python -m pip install -e '.[desktop]'
```

Desktop automation still requires a visible desktop session and a game the
user owns. Online or anti-cheat games are intentionally out of scope.
