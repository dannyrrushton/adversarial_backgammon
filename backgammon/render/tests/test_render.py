import math
import os

import numpy as np
import pytest

from backgammon.engine import Board, Player, Roll
from backgammon.render import M, MATERIALS, OrbitCamera, SceneState, build_scene, pick, pick_board
from backgammon.render import geometry, scene
from backgammon.render.image import write_png
from backgammon.render.optix import OptixRenderer, RenderError, available

W, B = Player.WHITE, Player.BLACK


# --- geometry -------------------------------------------------------------------------------------


def test_box_and_concat():
    a = geometry.box((0, 0, 0), (1, 1, 1), 3)
    b = geometry.box((2, 0, 0), (3, 1, 1), 4)
    m = geometry.Mesh.concat([a, b])
    assert a.num_triangles == 12 and m.num_triangles == 24
    assert m.indices.max() == len(m.positions) - 1
    assert set(m.material_ids.tolist()) == {3, 4}


def test_checker_normals_are_unit_and_bounds_match():
    c = geometry.checker((1, 0, 2), 0.5, 0.2, 7)
    assert np.allclose(np.linalg.norm(c.normals, axis=1), 1, atol=1e-5)
    lo, hi = c.bounds()
    assert np.allclose(lo, (0.5, 0, 1.5), atol=1e-3) and np.allclose(hi, (1.5, 0.2, 2.5), atol=1e-3)


def test_die_faces_have_all_pips():
    for value in range(1, 7):
        d = geometry.die((0, 0, 0), 1.0, value, M.DIE, M.PIP)
        pip_tris = int((d.material_ids == M.PIP).sum())
        assert pip_tris == 21 * 12  # 1+2+...+6 pips, 12 triangles each
        # The top face carries `value` pips.
        centroids = d.positions[d.indices].mean(axis=1)
        on_top = (d.material_ids == M.PIP) & (centroids[:, 1] > 0.99)
        assert int(on_top.sum()) == value * 12


def test_die_orientations_are_consistent():
    for top, (front, right) in geometry.DIE_ORIENTATION.items():
        faces = {top, 7 - top, front, 7 - front, right, 7 - right}
        assert faces == {1, 2, 3, 4, 5, 6}


def test_disc_winding_matches_normal():
    d = geometry.disc((0, 0, 0), 1.0, (0, 0, 1), 0)
    a, b, c = d.positions[d.indices[0]]
    assert np.dot(np.cross(b - a, c - a), (0, 0, 1)) > 0


# --- layout -----------------------------------------------------------------------------------------


def test_points_are_laid_out_like_a_real_board():
    xs = [scene.point_x(i) for i in range(24)]
    # White's 1-point is bottom right, 12 bottom left, 13 top left, 24 top right.
    assert xs[0] == max(xs[:12]) and xs[11] == min(xs[:12])
    assert xs[12] == min(xs[12:]) and xs[23] == max(xs[12:])
    assert len({(round(x, 3), scene.point_side(i)) for i, x in enumerate(xs)}) == 24


def test_checker_slots_stay_on_their_point():
    for i in (0, 11, 12, 23):
        for slot in range(15):
            x, y, z = scene.checker_position(i, slot)
            assert abs(x - scene.point_x(i)) < 1e-9
            assert abs(z) <= scene.EDGE_Z and np.sign(z) == scene.point_side(i)
            assert y >= 0


def test_scene_contains_every_checker():
    board = Board.from_relative({6: 5, 5: 5}, {6: 5, 5: 5}, bar=(1, 2), off=(4, 3))
    mesh = build_scene(board, SceneState(roll=Roll(6, 6)))
    per_checker = geometry.checker((0, 0, 0), 1, 1, 0).num_triangles
    per_off = geometry.checker((0, 0, 0), 1, 1, 0, segments=24).num_triangles
    whites = int((mesh.material_ids == M.CHECKER_WHITE).sum())
    assert whites == 11 * per_checker + 4 * per_off
    assert int((mesh.material_ids == M.PIP).sum()) == 2 * 21 * 12


def test_highlights_change_materials():
    mesh = build_scene(Board.initial(), SceneState(source_points={5}, target_points={2, 3}, selected=5))
    assert (mesh.material_ids == M.HIGHLIGHT_SOURCE).sum() == 1
    assert (mesh.material_ids == M.HIGHLIGHT_TARGET).sum() == 2
    assert (mesh.material_ids == M.SELECTED).sum() > 0


def test_material_table_complete():
    assert len(MATERIALS) == len(M)


# --- picking and camera -------------------------------------------------------------------------------


def test_pick_board_regions():
    for i in range(24):
        z = scene.point_side(i) * (scene.EDGE_Z - 1.0)
        assert pick_board(scene.point_x(i), z).index == i
    assert pick_board(0.0, 2.0).kind == "bar" and pick_board(0.0, 2.0).side is W
    assert pick_board((scene.TRAY_X0 + scene.TRAY_X1) / 2, -3).side is B
    assert pick_board(0.0, 99).kind == "none"


def project(camera, point, width, height):
    forward, u, v = camera.basis(width, height)
    d = np.asarray(point) - camera.eye
    depth = np.dot(d, forward)
    sx = np.dot(d, u) / np.dot(u, u) / depth
    sy = np.dot(d, v) / np.dot(v, v) / depth
    return (sx + 1) / 2 * width - 0.5, (1 - sy) / 2 * height - 0.5


@pytest.mark.parametrize("yaw", [0.0, 0.7, math.pi, -2.0])
def test_click_on_projected_point_picks_it(yaw):
    camera = OrbitCamera(yaw=yaw)
    for i in range(24):
        centre = (scene.point_x(i), 0.0, scene.point_side(i) * (scene.EDGE_Z - 1.2))
        px, py = project(camera, centre, 1280, 800)
        assert pick(camera, px, py, 1280, 800).index == i


def test_camera_limits():
    cam = OrbitCamera()
    cam.orbit(0, 10_000)
    assert cam.pitch == cam.MAX_PITCH
    cam.orbit(0, -10_000)
    assert cam.pitch == cam.MIN_PITCH
    cam.zoom(100)
    assert cam.distance == cam.MIN_DISTANCE
    cam.face(B)
    assert cam.eye[2] < 0


def test_write_png(tmp_path):
    img = np.zeros((4, 5, 4), np.uint8)
    img[..., 0] = 255
    write_png(tmp_path / "x.png", img)
    data = (tmp_path / "x.png").read_bytes()
    assert data.startswith(b"\x89PNG") and b"IEND" in data


# --- GPU ------------------------------------------------------------------------------------------

gpu = pytest.mark.skipif(not available(), reason="libbgrender.so not built")


@pytest.fixture(scope="module")
def renderer():
    if not available():
        pytest.skip("libbgrender.so not built")
    try:
        r = OptixRenderer()
    except RenderError as exc:
        pytest.skip(f"no OptiX device: {exc}")
    yield r
    r.close()


@gpu
@pytest.mark.gpu
def test_render_board(renderer):
    renderer.set_scene(build_scene(Board.initial(), SceneState(roll=Roll(5, 2))))
    cam = OrbitCamera()
    img = renderer.render(cam, 320, 200, samples=4)
    assert img.shape == (200, 320, 4) and img[..., 3].min() == 255
    assert img[..., :3].std() > 20  # not a flat colour
    renderer.render(cam, 320, 200, samples=4, accumulate=True)
    assert renderer.accumulated_samples == 8
    cam.orbit(50, 0)
    renderer.render(cam, 320, 200, samples=2, accumulate=True)
    assert renderer.accumulated_samples == 2  # camera moved: accumulation restarts


@gpu
@pytest.mark.gpu
def test_white_checker_is_bright_where_it_projects(renderer):
    board = Board.initial()
    renderer.set_scene(build_scene(board))
    cam = OrbitCamera(pitch=math.radians(89))
    img = renderer.render(cam, 640, 400, samples=16)
    x, y, z = scene.checker_position(5, 0)  # White's 6-point, first checker
    px, py = project(cam, (x, y + scene.CHECKER_HEIGHT, z), 640, 400)
    white = img[int(py), int(px), :3].astype(int)
    x, y, z = scene.checker_position(18, 0)  # Black's 6-point
    px, py = project(cam, (x, y + scene.CHECKER_HEIGHT, z), 640, 400)
    black = img[int(py), int(px), :3].astype(int)
    assert white.mean() > 150 and black.mean() < 90


@gpu
@pytest.mark.gpu
def test_invalid_scene_raises(renderer):
    mesh = build_scene(Board.initial())
    mesh.material_ids[0] = 999
    with pytest.raises(RenderError, match="material id"):
        renderer.set_scene(mesh)


@gpu
@pytest.mark.gpu
def test_viewer_widget_click_emits_pick(renderer):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtCore
    from PySide6.QtTest import QTest

    from backgammon.render.viewer import BoardView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = BoardView(renderer)
    view.resize(800, 500)
    view.show()
    view.show_position(Board.initial())
    picks = []
    view.picked.connect(picks.append)
    centre = (scene.point_x(7), 0.0, scene.point_side(7) * (scene.EDGE_Z - 1.2))
    px, py = project(view.camera, centre, view.width(), view.height())
    QTest.mouseClick(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(int(px), int(py)))
    assert picks and picks[0].index == 7
    # A drag rotates instead of picking.
    yaw = view.camera.yaw
    QTest.mousePress(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(100, 100))
    QTest.mouseMove(view, QtCore.QPoint(200, 110))
    QTest.mouseRelease(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(200, 110))
    assert view.camera.yaw != yaw and len(picks) == 1
    assert view.snapshot() is not None
    view.close()
    app.processEvents()
