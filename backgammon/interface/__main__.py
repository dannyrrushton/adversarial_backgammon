"""Play backgammon against (or watch) LLM agents.

  python -m backgammon.interface --mode teacher      # learn: the AI coaches you
  python -m backgammon.interface --mode adversary    # the AI plays to beat you
  python -m backgammon.interface --mode watch        # AI vs AI, fully annotated
"""

from __future__ import annotations

import argparse
import logging
import sys

from backgammon.agents import DEFAULT_MODEL
from backgammon.engine import Player

from .factory import Options, build_session
from .session import Mode


def gui_available() -> tuple[bool, str]:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False, "PySide6 is not installed (pip install -e '.[gui]')"
    from backgammon.render import available

    if not available():
        return False, "the OptiX renderer is not built (see README)"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default="teacher")
    parser.add_argument("--ui", choices=["auto", "gui", "text"], default="auto")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="your side")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for the AI")
    parser.add_argument("--model2", help="second Ollama model (watch mode Black)")
    parser.add_argument("--host", help="Ollama server URL (default $OLLAMA_HOST or localhost:11434)")
    parser.add_argument("--no-llm", action="store_true", help="use the built-in engine instead of Ollama")
    parser.add_argument("--no-cube", action="store_true", help="play without the doubling cube")
    parser.add_argument("--llm-commentary", action="store_true", help="ask the LLM for extra teaching commentary")
    parser.add_argument("--games", type=int, default=1, help="games to play in watch mode (text UI)")
    parser.add_argument("--delay", type=float, default=0.7, help="seconds between AI actions")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-legality", action="store_true", help="hide the rule explanations (text UI)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    options = Options(
        mode=Mode(args.mode),
        human=Player.WHITE if args.color == "white" else Player.BLACK,
        model=args.model,
        model2=args.model2,
        host=args.host,
        use_llm=not args.no_llm,
        use_cube=not args.no_cube,
        llm_commentary=args.llm_commentary,
        seed=args.seed,
    )
    session, warnings = build_session(options)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    ui = args.ui
    if ui != "text":
        ok, why = gui_available()
        if not ok:
            if ui == "gui":
                print(f"error: cannot start the 3D interface: {why}", file=sys.stderr)
                return 1
            print(f"note: using the text interface because {why}", file=sys.stderr)
            ui = "text"
        else:
            ui = "gui"

    if ui == "gui":
        from .gui import run_gui

        return run_gui(session, ai_delay_ms=int(args.delay * 1000))

    from .cli import TextUI

    TextUI(session, show_legality=not args.no_legality, delay=args.delay if options.mode is Mode.WATCH else 0).run(games=args.games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
