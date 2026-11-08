"""Headless replay of retouch checkpoints against explicitly supplied pixels."""
import json

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

from core.cli import run_cli
from core.render_context import RenderContext, SourceFrame
from utils.image_utils import imwrite_auto
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
    doc.initialize_checkpoint()
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


def test_replay_reports_low_memory_without_overwriting_output(replay_case, tmp_path, monkeypatch):
    from core import resource_policy

    checkpoint, paths, _ = replay_case
    output = tmp_path / 'existing.tif'
    output.write_bytes(b'existing')
    snapshot = resource_policy.ResourceSnapshot(
        total_ram=2 * 1024 ** 3, available_ram=1024 ** 3)
    monkeypatch.setattr(resource_policy, 'probe_resources', lambda: snapshot)
    assert _replay(checkpoint, paths, output) == 1
    assert output.read_bytes() == b'existing'


@pytest.mark.parametrize("extension", [".png", ".tif"])
@pytest.mark.parametrize("profile_kind", ["rgb", "invalid", "missing"])
def test_replay_exports_base_profile(replay_case, tmp_path, extension, profile_kind):
    checkpoint, paths, expected = replay_case
    rgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    profile = {"rgb": rgb, "invalid": b"invalid profile", "missing": b""}[profile_kind]
    base = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    assert imwrite_auto(str(paths[0]), base, icc_profile=profile)
    source = cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED)
    assert imwrite_auto(str(paths[1]), source, icc_profile=rgb)
    output = tmp_path / f"profiled{extension}"

    assert _replay(checkpoint, paths, output) == 0
    np.testing.assert_array_equal(cv2.imread(str(output), cv2.IMREAD_UNCHANGED), expected)
    with Image.open(output) as saved:
        assert saved.info.get("icc_profile", b"") == (rgb if profile_kind == "rgb" else b"")


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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_replay_preserves_standalone_grayscale_layout(tmp_path, dtype):
    """A checkpoint from a grayscale standalone session replays from its files."""
    from utils.standalone_retouch import load_standalone_retouch

    high = 200 if dtype == np.uint8 else 50000
    base_path, source_path = tmp_path / "base.png", tmp_path / "source.png"
    assert cv2.imwrite(str(base_path), np.full((32, 32), 40, dtype))
    assert cv2.imwrite(str(source_path), np.full((32, 32), high, dtype))
    base, context = load_standalone_retouch(str(base_path), [str(source_path)])
    assert base.ndim == 2
    doc = RetouchDocument(base, context)
    doc.initialize_checkpoint()
    doc.set_brush(diameter=8, hardness=1, opacity=1)
    assert doc.begin_stroke(16, 16)
    assert doc.end_stroke() == "committed"
    checkpoint = tmp_path / "edits.json"
    doc.save_checkpoint(str(checkpoint))
    output = tmp_path / "replayed.tif"

    assert _replay(checkpoint, [base_path, source_path], output) == 0
    actual = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert actual.dtype == dtype and actual.ndim == 2
    np.testing.assert_array_equal(actual, doc.result())


@pytest.fixture(params=['edited', 'analysis'])
def exact_archive_case(tmp_path, monkeypatch, request):
    from core.background_fill import BackgroundFillResult, MissingData
    base = np.full((64, 64, 3), 12001, np.uint16)
    first, second = np.full_like(base, 19001), np.full_like(base, 43003)
    frames = (SourceFrame('near', 'near', first), SourceFrame('far', 'far', second))
    context = RenderContext('exact', 'test', (), 64, 64, base.dtype, frames)
    doc = RetouchDocument(base, context)
    doc.initialize_checkpoint()
    def background(sources, fused, **kwargs):
        image = fused.copy()
        if request.param == 'edited':
            image[5:59, 7:57] = 23003
            image[31, 31] = (23001, 23002, 23003)
        layer = np.full((64, 64), 93, np.uint8)
        return BackgroundFillResult(image, request.param == 'edited', 0, 1, 0.4,
                                    MissingData(*(layer.copy() for _ in range(4))))
    monkeypatch.setattr('core.background_fill.replace_background', background)
    assert doc.auto_background('soft') == ('committed' if request.param == 'edited' else 'empty')
    assert doc.adjust_tones(0.1, 0) == 'committed'
    assert doc.undo()
    checkpoint = tmp_path / 'exact.ofretouch'
    doc.save_checkpoint(checkpoint)
    paths = [tmp_path / name for name in ('exact-base.png', 'near.png', 'far.png')]
    for path, image in zip(paths, (base, first, second)):
        assert cv2.imwrite(str(path), image)
    def forbidden(*args, **kwargs):
        pytest.fail('exact archive replay recomputed the background')
    monkeypatch.setattr('core.background_fill.replace_background', forbidden)
    return checkpoint, paths, doc.working.copy()


def test_replay_exact_archive_preserves_low_bits_and_saved_cursor(exact_archive_case, tmp_path):
    checkpoint, paths, expected = exact_archive_case
    output = tmp_path / 'exact-output.tif'
    assert _replay(checkpoint, paths, output) == 0
    actual = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)


def test_replay_exact_archive_verifies_sources_before_publishing(exact_archive_case, tmp_path):
    checkpoint, paths, _ = exact_archive_case
    changed = cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED)
    changed[8, 8, 0] += 1
    assert cv2.imwrite(str(paths[1]), changed)
    output = tmp_path / 'keep-output.tif'
    output.write_bytes(b'existing output')
    assert _replay(checkpoint, paths, output) != 0
    assert output.read_bytes() == b'existing output'


def test_replay_exact_archive_rejects_corrupt_payload_without_output(exact_archive_case, tmp_path):
    import zipfile
    checkpoint, paths, _ = exact_archive_case
    with zipfile.ZipFile(checkpoint) as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(checkpoint, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in members:
            if name == 'working.npy':
                data = data[:-1] + bytes([data[-1] ^ 1])
            archive.writestr(name, data)
    output = tmp_path / 'corrupt-output.tif'
    assert _replay(checkpoint, paths, output) != 0
    assert not output.exists()


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16])
def test_replay_extended_tones(tmp_path, dtype):
    base = np.empty((32, 32, 3), dtype)
    base[:] = (40, 100, 180) if dtype == np.uint8 else (10000, 26000, 46000)
    source = base.copy()
    context = RenderContext('tones', 'test', (), 32, 32, base.dtype,
                            (SourceFrame('source', 'source', source),))
    doc = RetouchDocument(base, context)
    doc.initialize_checkpoint()
    assert doc.adjust_tones(.1, -.1, highlights=.8, shadows=-.6,
                            saturation=.5, temperature=-.7) == 'committed'
    checkpoint = tmp_path / 'tones.json'
    doc.save_checkpoint(checkpoint)
    paths = [tmp_path / 'base.png', tmp_path / 'source.png']
    assert cv2.imwrite(str(paths[0]), base)
    assert cv2.imwrite(str(paths[1]), source)
    output = tmp_path / 'tones.tif'
    assert _replay(checkpoint, paths, output) == 0
    np.testing.assert_array_equal(cv2.imread(str(output), cv2.IMREAD_UNCHANGED), doc.working)
