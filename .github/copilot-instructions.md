# Project rules for Copilot

Project: a local app that lets a person connect a game, set a goal, and watch a
reinforcement-learning agent learn to play it live. The full plan is in
docs/PLAN.md.

- Work on one milestone at a time. Never start the next milestone unprompted.
- Python 3.11+, type hints everywhere, ruff for formatting, pytest for tests.
- Every game is a Gymnasium environment. The learner never imports game-specific code.
- Keep the learner, the game adapters and the web UI in separate packages that
  talk only through the interfaces in docs/PLAN.md.
- Add or update tests with every change. A milestone is done only when its
  acceptance checks pass.
- Do not add a dependency without saying why.
- Give every non-obvious function a short docstring. I am learning, so prefer clear
  code over clever code.
- Only automate single-player or offline games the user owns. Never target online
  or anti-cheat games.
