"""Render the board: ``python -m backgammon.render [--snapshot out.png]``.

Without ``--snapshot`` it opens an interactive window showing the starting position.
"""

from __future__ import annotations

import argparse
import sys

from backgammon.engine import Board, Player, Roll

from .camera import OrbitCamera
from .image import write_png
from .optix import OptixRenderer
from .scene import SceneState, build_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", help="write a PNG instead of opening a window")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--yaw", type=float, default=0.0, help="camera yaw in degrees")
    parser.add_argument("--pitch", type=float, default=58.0, help="camera pitch in degrees")
    args = parser.parse_args(argv)

    board = Board.initial()
    state = SceneState(roll=Roll(3, 1), roll_player=Player.WHITE)
    if args.snapshot:
        import math

        camera = OrbitCamera(yaw=math.radians(args.yaw), pitch=math.radians(args.pitch))
        with OptixRenderer() as renderer:
            renderer.set_scene(build_scene(board, state))
            write_png(args.snapshot, renderer.render(camera, args.width, args.height, samples=args.samples))
        print(f"wrote {args.snapshot}")
        return 0

    from PySide6 import QtWidgets

    from .viewer import BoardView

    app = QtWidgets.QApplication(sys.argv)
    view = BoardView()
    view.setWindowTitle(f"Backgammon board - OptiX on {view.renderer.device_name}")
    view.resize(args.width, args.height)
    view.show_position(board, state)
    view.picked.connect(lambda p: print("picked", p))
    view.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
