# Game-Learning AI Bot Plan

Oct 15, 2026 · @Aj

## Goal and scope

Build a local app that makes machine learning fun: pick a single-player game you enjoy, give an AI permission to play it, and watch live as it learns and how long it takes to beat you.

First user: you, as a learning project. Later users: anyone who downloads it and teaches it their own game. The agent learns by trial and error (reinforcement learning), so the app is a training room with a window into it.

Version 1 is done when all four of these are true:

- [ ] Two practice games built into the app, a one-button runner like Geometry Dash and a top-down racer, can be learned from scratch while you watch.
- [ ] The app records your own score as a baseline and shows how many attempts and how much time the AI needed to beat it.
- [ ] One real browser game can be connected and played by the bot.
- [ ] A new game can be added from a setup screen, with no code edits.
- [ ] Someone else can install it from a download and run it. Windows first, since most PC games and input tools target it.

Out of scope for v1: online multiplayer or anti-cheat games, console games, learning a game from a text description alone, and cloud training.

## The player journey

The whole product is one loop: tell it the game, let it play, and find out how long it takes to beat you.

1. **Tell it the game.** The app asks for the game's name and one sentence about the goal, such as "get the highest score" or "finish the track without crashing".
2. **Give permission.** A screen states exactly what the AI will do: which game window, which keys it may press, and for how long. Nothing starts until you click Allow, and a Stop button and a hotkey stay available the whole time.
3. **Set the bar.** You play a few rounds yourself, or type your best score. That becomes the baseline the AI has to beat.
4. **Watch it learn.** The AI starts clumsy and improves. The dashboard shows each attempt, the AI's score against your baseline, and a running clock.
5. **See the verdict.** When the AI's average over its last 10 attempts passes your baseline, the app shows a card with three numbers: attempts, time played inside the game, and real time elapsed. Game time and real time differ when a practice game is sped up. The first single attempt that beat you is noted too.
6. **Raise the bar or stop.** Set a harder target, such as 10 percent above your best, or let it keep going and watch where it levels off.

**Fair play.** Limit the AI to a human-plausible action rate, around 10 to 15 actions a second as in the config example, so beating you means playing better and not just reacting faster. Offer a "no limits" switch for people who want to see the ceiling.

**Some games will not fall.** Not every game is beatable in a reasonable time. Give each run a time budget, and if the AI has not caught up when it ends, show the learning curve and the remaining gap. "How long does it take?" is the question people come for, so an honest answer, including "longer than you gave it", is part of the product.

## Working with Copilot

Copilot does its best work on one small, well-specified milestone at a time, with this plan and a short rules file always in view.

**Set up the repo once.**

- [ ] Export this doc as Markdown and save it as `docs/PLAN.md`.
- [ ] Create `.github/copilot-instructions.md` with the starter below so Copilot reads the rules in every chat.
- [ ] Make a git repo and commit both files before writing any code.

Starter for `.github/copilot-instructions.md`:

```markdown
# Project rules for Copilot

Project: a local app that lets a person connect a game, set a goal, and watch a
reinforcement-learning agent learn to play it live. The full plan is in docs/PLAN.md.

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
```

Prompt to paste for each milestone:

```text
Read docs/PLAN.md and .github/copilot-instructions.md.
Implement Milestone <N>: <name>. Do only this milestone.
1. List the files you will create or change before writing code.
2. Write the tests for the acceptance checks first.
3. Implement until the tests pass.
4. Run the tests and show me the output.
5. Tell me how to check it by eye, and list anything you did not do.
```

**Habits that keep it on track.**

- Start a fresh chat for each milestone so old context does not leak in.
- Use agent mode to scaffold and ask mode when you want something explained.
- Read every diff. If Copilot names a library function you have not seen, ask it to show the docs or run it.
- Commit when a milestone passes and tag it `m1`, `m2`, and so on.

## Reality check and risks

The bot will learn, but real games run in real time, so the plan is built around that limit.

| Risk | What it means | How the plan handles it |
| --- | --- | --- |
| Real-time speed limit | A real game at 30 frames per second gives about 100,000 frames an hour, and learning from pixels often needs millions | Train first on built-in practice games that run far faster than real time; jump-start real games with recorded human play |
| Reward the bot can cheat | It learns only what the reward measures, so "survive" can produce a bot that stalls | Reward presets, plus a reward chart you can watch for odd behavior |
| "Any game" | Playing any game from a plain-English goal is unsolved research | The app handles any game it can capture and send keys to, once a person sets up its reward; plain-English goals only help fill in that setup |
| Vision breaks | A new resolution or theme confuses a bot trained on pixels | Fixed capture size, normalized frames, capture settings saved with each game |
| Input delay | Screen capture plus key presses add lag that a human player does not have | Measure the round trip in the screen-capture milestone and show it in the viewer |
| Games that fight automation | Online games and anti-cheat can ban accounts, and some single-player games forbid bots in their terms | Single-player and offline only, stated on the setup screen; game packs are config files and never include game files |
| Hardware | Training a pixel-based network on a CPU alone is slow | Check for a CUDA GPU in the first milestone; the plan works on CPU with small networks and the built-in games |

Rough expectation: a one-button runner can be learned in minutes to hours on a laptop. A 3D racer learned from pixels in real time can take days.

## Website or app

Build a local app with a web interface first, then wrap it in an installer. A pure website cannot control other games.

| Option | Can it play other games? | Learning speed | What users do to run it | Build effort |
| --- | --- | --- | --- | --- |
| Pure website | No. A page can ask to record the screen but cannot press keys in other apps. It can only play games embedded in the page, and only ones it is allowed to read. | Limited to what the browser allows for in-browser machine learning | Open a link | Medium |
| Local app with a web interface (Python backend, dashboard at localhost) | Yes. Python can capture a window and send key presses. | Full: uses the person's own CPU or GPU | Install Python and run a command at first | Medium |
| Packaged desktop app (the same app in an installer) | Yes | Full | Download, double-click | Higher: packaging and updates |

**Recommendation.** Build the local app first, with the dashboard written as a normal web page. That keeps the door open: the same dashboard can later sit inside an installer, and a small public website can show recorded replays and host game packs without running any AI itself.

## Architecture

Every game goes through one adapter, so the learner never touches a game directly.

&#91;embedded content: architecture · 7 parts, 1 shared adapter\]

The play loop runs along the top. The adapter shows the game to the learner as frames and rewards, and turns the learner's chosen action into key presses. The learner only ever sees that one shape, so a new game needs a new adapter and config, not new learning code.

The live link carries frames and numbers out to the dashboard and carries pause, resume and speed commands back in. The learner and the dashboard run as separate processes, so a slow browser cannot slow training. The runs folder holds checkpoints, replays and logs, and the dashboard reads replays from it. The setup wizard writes the per-game config that the adapter reads.

Suggested folders for Copilot to create in M0:

```text
learner/    training loop, policies, checkpoints
games/      GameEnv base class, built-in games, browser and screen adapters
server/     FastAPI app, live link, run storage
ui/         dashboard and setup wizard (web pages)
configs/    game.yaml files and template images
tests/      acceptance checks for each milestone
```

## Tech stack

Use existing, well-tested libraries for the hard parts so your effort goes into the game connection and the viewer.

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11 or newer | The machine learning ecosystem lives here |
| Learning | PyTorch with Stable-Baselines3 (PPO to start) | Ready-made, tested algorithms, so you do not write them yourself |
| Game interface | Gymnasium | The standard `reset` and `step` interface every reinforcement-learning library understands |
| Browser games | Playwright | Drives a real browser, takes screenshots, presses keys, and can read the game's state with JavaScript |
| Desktop games | `mss` for screen capture, plus a key-sending library such as `pydirectinput` | Fast window capture; many games ignore ordinary simulated keys, so Copilot must test which input method works |
| Image processing | OpenCV and NumPy | Resize, grayscale, and match a template image to spot a game-over screen |
| Live stream | FastAPI with WebSockets | Pushes frames and numbers to the dashboard as they happen |
| Dashboard | Plain HTML and JavaScript with a canvas and a small chart library | The viewer is mostly a picture and a few charts; React is optional |
| Per-game settings | YAML files | Easy for people to read, edit and share |
| Tests | pytest | Runs the acceptance checks for each milestone |
| Packaging (later) | PyInstaller, or a desktop shell around the local server | One-download install; decided in the last milestone |

Ask Copilot to pin the current stable version of each library in `requirements.txt` when it first adds it.

## Game contract

Every game, built-in or real, is wrapped so it looks identical to the learner. That one rule is what lets the app support new games without touching the learning code.

```python
class GameEnv(gymnasium.Env):
    """Wraps one game so the learner can play it."""

    def reset(self, *, seed=None, options=None):
        """Start a new attempt. Returns (observation, info)."""

    def step(self, action):
        """Press the keys for this action, let the game advance a little, and
        return (observation, reward, terminated, truncated, info)."""

    def render(self):
        """Return the current frame as an image array for the dashboard."""
```

- **Observation:** the last 4 frames, grayscale and shrunk to 84 by 84, stacked together. Built-in games can also offer a plain list of numbers (speed, distance to the next obstacle) for fast early testing.
- **Action:** a number picking one of the game's key combinations, for example 0 means do nothing and 1 means jump.
- **Reward:** a number from the game's reward preset (see Teaching a new game).
- **Terminated:** the attempt ended (crash, game over). **Truncated:** it hit the time limit.
- **Info:** extras for the viewer, such as score, attempt number and frame time.

Three adapters produce a `GameEnv`: `BuiltinGame` for the practice games, `BrowserGame` for pages driven by Playwright, and `ScreenGame` for desktop windows captured with `mss`.

Each game is described by one YAML file. This sketch is the shape Copilot should build toward:

```yaml
name: Dino clone
kind: browser              # builtin | browser | screen
source:
  url: http://localhost:8000/dino/index.html
capture:
  region: [0, 0, 600, 150]  # x, y, width, height of the game area
  frame_size: [84, 84]
  frame_stack: 4
actions:
  - name: nothing
    keys: []
  - name: jump
    keys: [space]
game_over:
  detect: template          # template | score_unchanged | page_flag
  template: assets/game_over.png
  restart_keys: [space]
reward:
  preset: survive           # survive | score | progress | speed
  score_region: [450, 10, 140, 30]   # optional, read with OCR
  penalty_on_death: -1.0
limits:
  max_episode_seconds: 120
  step_hz: 15               # how many times per second the bot acts
```

The adapter test for every game is the same: a random agent plays 1,000 steps without errors, and every `step` returns the five values above.

## Live viewer

The dashboard is the product people will remember, so it ships early and grows with every milestone. It is one web page served from the local app.

**Panels**

- **Game view.** The real game on the left, or the bot's own view (the small grayscale frames it actually sees) on a toggle.
- **Attempt strip.** Attempt number, time alive this run, best run so far, and your baseline score with a marker for the moment the AI passes it.
- **Reward chart.** Reward per attempt as a line, with a smoothed average on top. This is how you watch it learn and spot cheating.
- **Action display.** The key being pressed now, and bars showing how confident the bot is in each action.
- **Replay shelf.** Every new best run is saved and can be replayed.
- **Health.** Frames per second, delay from capture to key press, CPU or GPU load.

**Controls.** Start, pause, resume, stop and reset. Save and load a checkpoint. A speed selector (1x, 4x, as fast as possible) that is offered only for built-in games, since real games cannot be sped up.

**Two watch modes.** *Training* shows the bot exploring, which looks clumsy because it deliberately tries random things. *Show best* turns the exploring off so you see the best it can do.

**What streams to the page.** Small JPEG frames about 10 to 15 times a second, plus short JSON messages for each step and each finished attempt. Keep this stream separate from the training loop and let it drop old frames when the browser falls behind, so a slow page never slows the learning.

## Teaching a new game

A setup wizard turns "I want it to play this game" into a `game.yaml` file in about five minutes, with no code.

1. **Pick the kind of game:** built-in, a web page, or a desktop window.
2. **Point at it.** Enter the page address, or choose the window and drag a box over the play area.
3. **Choose the controls.** List the keys the bot may press and name each action, such as "jump".
4. **Show what game over looks like.** The app starts the game, you click "capture" when the game-over screen appears, and the app saves it as the template to watch for. Pick the key that restarts.
5. **Pick a reward preset** (table below), and optionally drag a box over the score or progress readout.
6. **Test it.** A random bot plays for 30 seconds and the page shows the rewards it earned, so you can check they make sense.
7. **Save and start training.**

| Preset | The bot is rewarded for | Needs from the game | Good for |
| --- | --- | --- | --- |
| survive | Each moment it stays alive, with a penalty at game over | Game-over detection only | Runners and Geometry Dash-style games |
| score | Increases in an on-screen score | A score readout the app can read | Arcade games |
| progress | Moving forward through the level or track | A progress bar or distance value | Racers and platformers |
| speed | Going fast without crashing | A speed readout, plus game-over detection | Racers |

**Shortcut for real games: learn from you first.** A "record me" button saves a few minutes of your own play, frames and key presses together. The bot first copies you, then reinforcement learning improves on it. This matters because real games cannot be sped up.

**Where plain-English goals fit.** The setup always asks for a one-sentence goal, such as "survive as long as possible and finish the lap", and shows it on the dashboard. At first it is only a label, and the person still picks the reward preset from the list. Later, a language model can read the sentence and propose the preset and what to watch for, and the person always reviews it. Treat that as an experiment for the last milestones, since the wizard works without it.

## Milestones

Build in this order. Each milestone ends with checks Copilot can run, so "done" is never a matter of opinion. Give Copilot one milestone per chat using the prompt in Working with Copilot.

- [x] **M0: Project setup.** Folders `learner/`, `games/`, `server/`, `ui/`, `configs/`, `tests/`; a virtual environment; ruff and pytest; a `tools/check_hardware.py` script. Done when: `pytest` runs with one passing test, and the script prints the Python version and whether a CUDA GPU is available.
- [x] **M1: Practice game 1, a one-button runner.** A simple side-scrolling runner with random obstacles, one jump button and a crash that ends the attempt, wrapped as a `GameEnv`. It has a fast mode with no window and a visible mode. Done when: a random agent plays 1,000 steps, the same seed gives the same obstacles, and fast mode runs far faster than real time.
- [x] **M2: First learning loop, on numbers.** Train PPO on the runner using the plain number observation. Save checkpoints and log rewards. Done when: after training, the average time alive is at least 3 times a random agent's, checked by a test.
- [ ] **M3: Live viewer v1.** A FastAPI server streams frames and metrics to a web page showing the game, the attempt counter and the reward chart, with start, pause and stop. Done when: you can watch the runner learn in the browser, pause and resume work, and training speed with the page open is within 10 percent of training speed without it.
- [ ] **M4: Your baseline and the verdict.** A "play a few rounds" mode records your own scores, or accepts a typed best score, as the baseline. The viewer draws it as a line on the reward chart. Done when: with a made-up baseline on the runner, a verdict card appears once the AI's average over its last 10 attempts passes it, and the card's attempts, game time and real time match the logs, checked by a test.
- [ ] **M5: Permission gate.** Before any game is touched, a screen lists the game, the window, the keys the AI may press and the time budget, plus the action-rate limit. Nothing is sent until the person clicks Allow, and a Stop button and a hotkey are always available. Done when: a test shows no key press is sent before Allow, and Stop halts key sending within one second.
- [ ] **M6: Learn from pixels.** Frame preprocessing (grayscale, 84 by 84, stack of 4) and a small convolutional network. Done when: preprocessing has a unit test on a known image, and the pixel-based bot beats a random agent by a clear margin on the runner.
- [ ] **M7: Practice game 2, a top-down racer.** A simple track with steering and acceleration and a progress reward from checkpoints along the track. Done when: the learner code trains on it with no changes, and only the game file and its config are new. This proves the contract works.
- [ ] **M8: Browser game adapter.** `BrowserGame` using Playwright, for a game you host locally (your own, or an open-source HTML5 game). Done when: the bot plays 20 attempts unattended with automatic restarts, and the capture-to-key delay is measured and shown in the viewer.
- [ ] **M9: Desktop game adapter.** `ScreenGame` using `mss` capture of a chosen window and key sending, with game-over detection by template match and automatic restart. Test on a single-player game you own, such as Geometry Dash. Done when: 100 unattended attempts finish without help, and the notes record which key-sending method worked.
- [ ] **M10: Setup wizard.** The web screens for the seven steps above, writing `game.yaml`. It opens by asking for the game's name and a one-sentence goal, which the dashboard shows, and it reuses the permission screen from M5. Done when: a new browser game is added through the wizard with no code changes, and its 30-second test run shows sensible rewards.
- [ ] **M11: Learn from demonstrations.** A "record me" button that reuses the recorder from M4, behavior cloning from the recording, then reinforcement learning on top. Done when: on one real game, a bot started from a 5-minute recording beats a from-scratch bot given the same training time, shown as two lines on one chart.
- [ ] **M12: Show-best mode and replay shelf.** Exploration off for the best policy, new best runs saved, replays playable from the dashboard. Done when: a new best run appears on the shelf automatically and replays correctly after a restart.
- [ ] **M13: Plain-English goals (experimental).** A language model suggests the reward preset and game-over cues from the typed goal, and the person confirms. Done when: with the feature off the app behaves exactly as before, and with it on nothing is saved until the person confirms.
- [ ] **M14: Package and share.** Installer, README, license, example game packs. Done when: a fresh Windows user account can install from the download and run the runner demo.

## Packaging and sharing

Ship in this order, each step only after the one before it works: source code first, an installer second, game packs third.

- [ ] **Source release.** Put the repo on GitHub with a README that has a screenshot, a three-command install, and a "teach it your first game" walkthrough. Add an open-source license so people may use and change it.
- [ ] **Installer.** Wrap the local server and dashboard so a person downloads one file, double-clicks, and a browser window opens at the dashboard. PyInstaller is the simplest start; a desktop shell such as Tauri or Electron is the polished option. Let Copilot compare them when you reach M14.
- [ ] **Game packs.** A pack is a `game.yaml` plus its template images, shared as a folder or zip. A small public site can list packs, show recorded replays, and link to the download. The site never runs any AI itself.
- [ ] **Safe defaults.** The app runs only on the person's own computer, sends nothing out, and shows the single-player and offline notice on the setup screen. Packs never contain game files.
- [ ] **Updates.** Version the config format so old packs keep working, and tag each release.
