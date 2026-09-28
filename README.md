# Adversarial Backgammon

Backgammon played by LLM agents running on a local [Ollama](https://ollama.com) model, drawn on a
3D board ray-traced with NVIDIA OptiX, with every move annotated: what it does, the strategy
behind it, and why it is legal. You can watch the AIs play each other, learn from an AI coach,
or play an AI that is trying to beat you.

```
backgammon/
  engine/       rules: board, dice, legal moves, hitting, bar entry, bearing off, cube, scoring
  render/       3D board: scene geometry + camera/picking in Python, ray tracing in C++/OptiX
    cpp/        libbgrender.so (C API, host code) + OptiX device programs (CUDA -> OptiX IR)
  agents/       evaluator, 1-ply search, Ollama client, LLM / heuristic / random agents
  annotation/   move annotations, strategy themes, legality explanations, tutor (hints, critique)
  interface/    game sessions (teacher / adversary / watch), text UI and Qt + OptiX GUI
```

Each module has its own `tests/` folder.

## How the pieces fit

- **Engine** (`backgammon.engine`) is pure Python and has no dependencies on the rest.
  `legal_plays()` enforces every movement rule: bar first, blocked points, hitting blots, bearing
  off (exact or higher die from the highest point), use both dice if possible, the larger die
  when only one can be used, and four moves on doubles. It is cross-checked against a brute-force
  reference generator over random positions and all 21 rolls. `Game` adds the opening roll, turn
  flow, the doubling cube, and single / gammon / backgammon scoring.
- **Agents** (`backgammon.agents`). LLMs are weak at backgammon arithmetic, so a hand-written
  evaluator plus a one-ply lookahead (averaging the opponent's best reply over all 21 rolls)
  shortlists the strongest plays. The LLM then picks one and explains why, using structured JSON
  output. If Ollama is down or gives an invalid answer, the agent falls back to the engine's
  choice, so a game never stalls.
  - Cube decisions use a win-probability model fitted by logistic regression on self-play
    (`python -m backgammon.agents.calibrate`).
- **Annotation** (`backgammon.annotation`) compares the position before and after a move to detect
  themes (hits, points made, anchors, primes, escapes, blots and shot counts, bear-offs, breaking
  contact). It explains each checker move against the rule that allows it, and ranks the play
  against the engine's alternatives. `Tutor` gives hints and grades a learner's moves.
- **Render** (`backgammon.render`) builds the board as a triangle mesh in numpy: wooden frame,
  felt, points, bevelled checkers, dice with pips, highlights. It passes the mesh through ctypes to
  `libbgrender.so`, which builds an OptiX acceleration structure and traces primary, shadow and
  reflection rays with soft area lights and progressive anti-aliasing (64 samples per pixel at
  1280×800 takes about 30 ms on an RTX A6000). The Qt `BoardView` widget orbits the camera on
  left-drag, zooms on the wheel, and turns a click into a board point with a ray-plane pick.
- **Interface** (`backgammon.interface`). `GameSession` is UI-independent. The text UI and the Qt
  GUI both drive it.

## Setup

Requirements:
- Python ≥ 3.11
- For the 3D board: CUDA toolkit, CMake ≥ 3.27, and an NVIDIA RTX GPU whose driver includes OptiX 9.x
- For the LLM agents: Ollama

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,gui]'        # gui = PySide6 (Qt) for the 3D window

# Build the OptiX renderer. The OptiX headers are fetched from github.com/NVIDIA/optix-dev
# (v9.1.0) unless you pass -DOPTIX_INCLUDE_DIR=/path/to/optix/include or set OptiX_INSTALL_DIR.
cmake -S backgammon/render/cpp -B backgammon/render/cpp/build -G Ninja
cmake --build backgammon/render/cpp/build

ollama pull qwen3.8:27b                       # or any chat model; pass it with --model
```

The Python side finds `backgammon/render/cpp/build/lib/libbgrender.so` automatically. Set
`BGRENDER_LIB` to use a library somewhere else.

## Playing

```bash
.venv/bin/python -m backgammon.interface --mode teacher     # the AI coaches you as you play
.venv/bin/python -m backgammon.interface --mode adversary   # the AI plays to win, no help
.venv/bin/python -m backgammon.interface --mode watch       # AI vs AI, fully annotated
```

- **Teacher mode.**
  - The opponent ("Coach") explains each of its moves: the strategy, its reasoning, and the rules
    that make the move legal.
  - After each of your moves it grades the play and names a stronger one if there was one.
  - **Hint** suggests the top plays with their ideas, and gives cube advice (with win chances)
    before you roll or when you're doubled.
  - Illegal moves are rejected with the rule they break.
- **Adversary mode.** The opponent ("Rival") uses a deeper search and a tighter shortlist, keeps
  its reasoning to itself, and hints are off. After the game, **Review** annotates every move,
  including its own.
- **Watch mode.** Two LLM agents with different personalities play each other: "Aggressor" loves
  to hit and blitz, "Strategist" plays for primes and safety. Use `--model2` to put a different
  model on the Black side.

In the 3D window, click a checker and then its destination. Legal sources glow yellow, legal
destinations green, and the tray lights up when bearing off is legal. You can also type
notation such as `13/8 6/5`, `bar/22`, `6/off` or `8/5(2)`.
- **Mouse:** left-drag rotates the board, the wheel zooms.
- **Keys:** `R` resets the view, arrow keys spin it.

Useful flags:
- `--ui text` plays in the terminal; this is chosen automatically when PySide6 or the renderer
  is missing.
- `--color black`
- `--no-llm` uses the built-in engine instead of Ollama.
- `--no-cube`
- `--llm-commentary` adds LLM teaching prose to each annotation.
- `--seed N` makes the dice repeatable.

To render only the board: `python -m backgammon.render` (window) or
`python -m backgammon.render --snapshot board.png`.

## Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/pytest            # all five modules
.venv/bin/pytest backgammon/engine                    # one module
.venv/bin/pytest -m "not gpu and not ollama"          # no GPU / no Ollama needed
ctest --test-dir backgammon/render/cpp/build          # C++ smoke test of the OptiX library
```

- GPU tests skip automatically when `libbgrender.so` isn't built.
- The live Ollama test skips when the model isn't available.
- Agent tests use a fake Ollama client, so they need no server.

## Limitations

- The evaluator is hand-tuned, not a trained neural network like GNU Backgammon's. It plays
  solidly but not at expert level, and its equity figures are cruder than a bot's. The grading
  thresholds are widened to allow for that.
- The doubling cube's value appears in the side panel, not printed on the 3D cube.
- Games are money-play style with a running score; match play (Crawford rule) isn't implemented.
