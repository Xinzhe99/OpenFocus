"""Headless replay of retouch checkpoints against explicitly supplied pixels."""
import json

import cv2
import numpy as np
import pytest

from core.cli import run_cli
from core.render_context import RenderContext, SourceFrame
from utils.retouch import RetouchDocument


@pytest.fixture(params=[np.uint8, np.uint16])
def replay_case(tmp_path, request):
    dtype = request.param
    base_value, first_value, second_value = ((120, 200, 240) if dtype == np.uint8
                                               else (12000, 40000, 52000))
    base = np.full((32, 32, 3), base_value, dtype)
    first = np.full((20, 20, 3), first_value, dtype)
    first[:2] = 0  # aligned registration fill is excluded from painting
    second = np.full((32, 32, 3), second_value, dtype)
    frames = (SourceFrame("roi", "roi", first, 4, 5, True),
              SourceFrame("full", "full", second))
    context = RenderContext("original", "test", (), 32, 32, base.dtype, frames)
    doc = RetouchDocument(base, context)
    doc.set_brush(diameter=6, hardness=1, opacity=1)
    assert doc.begin_stroke(12, 13)
    assert doc.end_stroke() == "committed"
    doc.set_frame(1)
    assert doc.begin_stroke(27, 27)
    assert doc.end_stroke() == "committed"
    assert doc.undo()  # the saved cursor must exclude the second stroke
    checkpoint = tmp_path / "edits.json"
    doc.save_checkpoint(str(checkpoint))
    paths = [tmp_path / name for name in ("base.png", "first.png", "second.png")]
    for path, image in zip(paths, (base, first, second)):
        assert cv2.imwrite(str(path), image)
    return checkpoint, paths, doc.result().copy()


def _replay(checkpoint, paths, output):
    return run_cli(["retouch", "replay", str(checkpoint), "--base", str(paths[0]),
                    "--sources", *(str(path) for path in paths[1:]),
                    "--output", str(output)])


def test_replay_preserves_pixels_depth_geometry_mask_and_cursor(replay_case, tmp_path):
    checkpoint, paths, expected = replay_case
    output = tmp_path / "replayed.tif"
    assert _replay(checkpoint, paths, output) == 0
    actual = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual[13, 12] != actual[0, 0])
    assert np.array_equal(actual[27, 27], actual[0, 0])


def test_replay_rejects_wrong_order_pixels_and_base(replay_case, tmp_path):
    checkpoint, paths, expected = replay_case
    output = tmp_path / "replayed.png"
    assert _replay(checkpoint, [paths[0], paths[2], paths[1]], output) != 0
    assert _replay(checkpoint, [paths[1], paths[1], paths[2]], output) != 0
    changed = np.full((20, 20, 3), 100, expected.dtype)
    assert cv2.imwrite(str(paths[1]), changed)
    assert _replay(checkpoint, paths, output) != 0
    assert not output.exists()


def test_replay_rejects_invalid_geometry_and_unreproducible_mask(replay_case, tmp_path):
    checkpoint, paths, _ = replay_case
    original = json.loads(checkpoint.read_text(encoding="utf-8"))
    for field, value in (("offset_x", 100),
                         ("mask", {"shape": [20, 20], "dtype": "bool", "sha256": "0" * 64})):
        altered = json.loads(json.dumps(original))
        altered["frames"][0][field] = value
        checkpoint.write_text(json.dumps(altered), encoding="utf-8")
        output = tmp_path / f"{field}.png"
        assert _replay(checkpoint, paths, output) != 0
        assert not output.exists()


def test_replay_never_overwrites_inputs_or_uses_lossy_output(replay_case, tmp_path):
    checkpoint, paths, _ = replay_case
    for output in [checkpoint, *paths]:
        before = output.read_bytes()
        assert _replay(checkpoint, paths, output) != 0
        assert output.read_bytes() == before
    output = tmp_path / "lossy.jpg"
    assert _replay(checkpoint, paths, output) != 0
    assert not output.exists()
