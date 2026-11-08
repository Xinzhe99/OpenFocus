"""EXIF belongs to the published result, never to the currently loaded stack."""
import numpy as np


def test_render_context_uses_supplied_exif_and_never_reads_paths(monkeypatch):
    import core.image_loader as image_loader
    from core.render_context import build_render_context
    monkeypatch.setattr(image_loader, "_read_exif_bytes",
                        lambda path: (_ for _ in ()).throw(AssertionError(path)))
    frames = [np.zeros((2, 2, 3), np.uint8) for _ in range(3)]
    context = build_render_context(
        frames, frames[0], filenames=["a", "b", "a"],
        algorithm="openfocus_c", source_paths=("/a.jpg", "", "/a.jpg"),
        frame_exif=(b"A", b"", b"A"))
    assert context is not None
    assert context.source_exif == b"A"
    # Empty paths keep their position; they do not shift later metadata.
    assert context.frame_exif == (b"A", b"", b"A")


def test_output_record_copies_exif_snapshot():
    from core.render_context import build_render_context
    from controllers.output_manager import OutputRecord
    frames = [np.zeros((2, 2, 3), np.uint8)]
    context = build_render_context(
        frames, frames[0], source_paths=(), frame_exif=(b"X",))
    record = OutputRecord.create(frames[0], "result", context=context)
    assert record.source_exif == b"X"
    assert record.frame_exif == (b"X",)


def _make_jpeg_with_make(path, make):
    from PIL import Image
    image = Image.new("RGB", (8, 8), (40, 80, 120))
    exif = Image.Exif()
    exif[271] = make  # Make
    image.save(path, exif=exif.tobytes())


def test_real_exif_survives_cross_stack_history_write(tmp_path):
    from PIL import Image
    from core.image_loader import ImageStackLoader, _read_exif_bytes
    from core.render_context import build_render_context
    from utils.image_utils import imwrite_auto

    stack_a = tmp_path / "a"
    stack_b = tmp_path / "b"
    stack_a.mkdir()
    stack_b.mkdir()
    _make_jpeg_with_make(stack_a / "a0.jpg", "StackA")
    _make_jpeg_with_make(stack_a / "a1.jpg", "StackA")
    _make_jpeg_with_make(stack_b / "b0.jpg", "StackB")
    _make_jpeg_with_make(stack_b / "b1.jpg", "StackB")

    loader = ImageStackLoader()
    assert loader.load_from_filepaths([str(stack_a / "a0.jpg"), str(stack_a / "a1.jpg")])[0]
    frames_a = [np.asarray(Image.open(path)) for path in sorted(stack_a.glob("*.jpg"))]
    assert all(b"StackA" in data for data in loader.source_exifs)
    context_a = build_render_context(
        frames_a, frames_a[0], source_paths=(), frame_exif=loader.source_exifs)

    # Loading another stack overwrites the global cache; the historical result
    # still writes A's bytes, not B's.
    assert loader.load_from_filepaths([str(stack_b / "b0.jpg"), str(stack_b / "b1.jpg")])[0]
    out_a = tmp_path / "a.png"
    assert imwrite_auto(str(out_a), frames_a[0], exif=context_a.source_exif)
    with Image.open(out_a) as written:
        assert written.getexif().get(271) == "StackA"
    assert b"StackB" in _read_exif_bytes(str(stack_b / "b0.jpg"))


def _exif_bytes(make):
    from PIL import Image
    exif = Image.Exif()
    exif[271] = make
    return exif.tobytes()


class _Labels:
    def snapshot(self):
        # _save_frames_as expects the snapshot itself to expose
        # prepare_bgr_image (that is the LabelManager contract).
        return self

    def prepare_bgr_image(self, _target, image, _index, **_kwargs):
        return image


class _Status:
    def showMessage(self, *_args, **_kwargs):
        pass


class _ExportWindow:
    def __init__(self, tmp_path):
        self.current_folder_path = str(tmp_path)
        self.image_icc_profiles = ()
        self.raw_images = []
        self.image_filenames = []
        self.image_source_exifs = ()
        self.registered_source_indices = ()
        self.registration_results = []
        self.linear_dng_metadata = None
        self.label_manager = _Labels()

    def statusBar(self):
        return _Status()


def test_output_record_write_uses_result_exif_after_cross_stack_render(tmp_path, monkeypatch):
    from PIL import Image
    from core.render_context import build_render_context
    from controllers.output_manager import OutputManager, OutputRecord
    import controllers.output_manager as output_manager

    monkeypatch.setattr(output_manager.scalebar, "burn", lambda image, _cfg: image)
    frames = [np.zeros((8, 8, 3), np.uint8)]
    context_a = build_render_context(
        frames, frames[0], source_paths=("/a.jpg",),
        frame_exif=(_exif_bytes("StackA"),))
    record_a = OutputRecord.create(frames[0], "A result", context=context_a)
    # Simulate loading B before saving A's historical record.
    context_b = build_render_context(
        frames, frames[0], source_paths=("/b.jpg",),
        frame_exif=(_exif_bytes("StackB"),))
    assert context_b.source_exif != context_a.source_exif
    manager = OutputManager(_ExportWindow(tmp_path))
    path = tmp_path / "history-a.png"
    manager._write_record_pixels(
        record_a, str(path), _Labels(), None, None,
        {'labels': [], 'scale': None, 'watermark': None})
    with Image.open(path) as written:
        assert written.getexif().get(271) == "StackA"


def test_aligned_frame_export_uses_frame_exif(tmp_path):
    from PIL import Image
    from controllers.export_manager import ExportManager
    from controllers.output_manager import OutputRecord

    images = [np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)]
    record = OutputRecord.create(None, "aligned", kind="aligned")
    record.annotations_baked = True
    record.watermark_baked = True
    record.frame_exif = (_exif_bytes("FrameA0"), _exif_bytes("FrameA1"))
    record.source_exif = record.frame_exif[0]
    exporter = ExportManager(_ExportWindow(tmp_path))
    exporter._save_frames_as(
        images, ["a0.tif", "a1.tif"], "registered", label="A",
        baked=False, destination=str(tmp_path), extension="png", record=record)
    written = sorted((tmp_path / "A_aligned").glob("*.png"))
    assert len(written) == 2
    assert [Image.open(path).getexif().get(271) for path in written] == [
        "FrameA0", "FrameA1"]


def test_same_path_reload_gets_new_snapshot(tmp_path):
    from core.image_loader import ImageStackLoader
    path = tmp_path / "same.jpg"
    _make_jpeg_with_make(path, "OldMake")
    loader = ImageStackLoader()
    assert loader.load_from_filepaths([str(path)])[0]
    old = loader.source_exifs
    _make_jpeg_with_make(path, "NewMake")
    assert loader.load_from_filepaths([str(path)])[0]
    assert old[0] != loader.source_exifs[0]
    assert b"NewMake" in loader.source_exifs[0]


def _current_source_export_fixture(tmp_path, monkeypatch):
    import controllers.export_manager as export_manager
    from utils import watermark
    monkeypatch.setattr(export_manager.scalebar, "export_cfg", lambda *a, **k: None)
    monkeypatch.setattr(export_manager.scalebar, "burn", lambda image, _cfg: image)
    monkeypatch.setattr(watermark, "export_cfg", lambda *a, **k: None)
    window = _ExportWindow(tmp_path)
    frames = [np.zeros((8, 8, 3), np.uint8), np.ones((8, 8, 3), np.uint8)]
    window.raw_images = frames
    window.image_source_exifs = (_exif_bytes("InputA"), _exif_bytes("InputB"))
    window.image_filenames = ["a0.tif", "a1.tif"]
    return window, frames, export_manager.ExportManager(window)


def test_current_input_export_keeps_frame_order(tmp_path, monkeypatch):
    from PIL import Image
    window, frames, exporter = _current_source_export_fixture(tmp_path, monkeypatch)
    exporter._save_frames_as(
        frames, window.image_filenames, 'input', label='InputTest',
        destination=str(tmp_path), extension='png')
    files = sorted((tmp_path / 'InputTest_processed').glob('*.png'))
    assert [Image.open(path).getexif().get(271) for path in files] == [
        'InputA', 'InputB']


def test_current_registered_export_uses_reordered_source_indices(tmp_path, monkeypatch):
    from PIL import Image
    window, frames, exporter = _current_source_export_fixture(tmp_path, monkeypatch)
    # Registered frames are in source order 1, 0; EXIF must follow the map,
    # not the current raw-frame positions.
    window.registered_source_indices = (1, 0)
    window.registration_results = [frames[1], frames[0]]
    exporter._save_frames_as(
        window.registration_results, window.image_filenames, 'registered',
        label='RegTest', destination=str(tmp_path), extension='png')
    files = sorted((tmp_path / 'RegTest_aligned').glob('*.png'))
    assert [Image.open(path).getexif().get(271) for path in files] == [
        'InputB', 'InputA']
