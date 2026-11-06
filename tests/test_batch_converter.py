from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from core import batch_converter as converter


@pytest.fixture(autouse=True)
def memory_budget(monkeypatch):
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 8 * 1024**3)


def test_unicode_conflicts_and_16bit(tmp_path):
    source = tmp_path / '花.png'
    original = np.array([[0, 1000, 65535], [25, 20000, 45000]], dtype=np.uint16)
    cv2.imencode('.png', original)[1].tofile(str(source))
    before = source.read_bytes()
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    assert result.endswith('花_1.png')
    assert source.read_bytes() == before
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, original)
    second = converter.convert_one(str(source), str(tmp_path), '.png')
    assert second.endswith('花_2.png')


def test_multipage_tiff_and_jpeg(tmp_path):
    source = tmp_path / 'stack.tif'
    with tifffile.TiffWriter(source) as tif:
        tif.write(np.full((8, 9), 65535, dtype=np.uint16), photometric='minisblack')
        tif.write(np.full((8, 9), 0, dtype=np.uint16), photometric='minisblack')
    assert converter.page_count(str(source)) == 2
    first = converter.convert_one(str(source), str(tmp_path), '.jpg', 0, 2)
    second = converter.convert_one(str(source), str(tmp_path), '.jpg', 1, 2)
    assert first.endswith('_page_0001.jpg') and second.endswith('_page_0002.jpg')
    assert np.asarray(Image.open(first)).min() == 255
    assert np.asarray(Image.open(second)).max() == 0


def test_transparency_is_not_discarded(tmp_path):
    source = tmp_path / 'alpha.png'
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 100
    Image.fromarray(rgba).save(source)
    with pytest.raises(ValueError, match='transparency'):
        converter.convert_one(str(source), str(tmp_path), '.jpg')
    assert not (tmp_path / 'alpha.jpg').exists()
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    assert np.asarray(Image.open(result))[0, 0, 3] == 100


def test_opaque_rgba_can_convert_to_jpeg(tmp_path):
    source = tmp_path / 'opaque.png'
    Image.new('RGBA', (8, 8), (120, 80, 40, 255)).save(source)
    output = converter.convert_one(str(source), str(tmp_path), '.jpg')
    with Image.open(output) as image:
        assert image.mode == 'RGB' and image.size == (8, 8)


def test_high_depth_heif_to_png_retains_16bit(tmp_path):
    pillow_heif = pytest.importorskip('pillow_heif')
    source = tmp_path / 'high-depth.heic'
    pixels = np.full((32, 32, 3), 32000, dtype=np.uint16)
    try:
        pillow_heif.from_bytes('RGB;16', (32, 32), pixels.tobytes()).save(
            str(source), quality=100)
    except (RuntimeError, OSError) as exc:
        pytest.skip(f'HEIF encoder unavailable: {exc}')
    output = converter.convert_one(str(source), str(tmp_path), '.png')
    decoded = cv2.imdecode(np.fromfile(output, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint16 and decoded.shape == pixels.shape
    assert np.any(decoded % 257)


def test_cancellation_after_encoding_removes_reserved_file(tmp_path):
    source = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(source)
    calls = iter([False, False, True])
    with pytest.raises(InterruptedError):
        converter.convert_one(str(source), str(tmp_path), '.jpg', cancelled=lambda: next(calls))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['image.png']


def test_insufficient_memory_and_existing_file(tmp_path, monkeypatch):
    source = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(source)
    existing = tmp_path / 'image.jpg'
    existing.write_bytes(b'keep')
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 1)
    with pytest.raises(MemoryError):
        converter.convert_one(str(source), str(tmp_path), '.jpg')
    assert existing.read_bytes() == b'keep'


def test_dialog_starts_without_main_window(qapp):
    from dialogs.batch_converter import BatchConverterDialog
    dialog = BatchConverterDialog()
    try:
        assert not dialog.start.isEnabled()
        assert not dialog.remove.isEnabled()
        assert not dialog.delete_shortcut.isEnabled()
        assert not dialog.backspace_shortcut.isEnabled()
        assert not dialog.empty_hint.isHidden()
        assert set(dialog.format.itemText(i) for i in range(dialog.format.count())) >= {'JPEG', 'PNG', 'TIFF'}
    finally:
        dialog.close()


@pytest.fixture
def converter_dialog(qapp, monkeypatch):
    from dialogs.batch_converter import BatchConverterDialog, ThumbnailWorker, ConverterWorker
    monkeypatch.setattr(ThumbnailWorker, 'start', lambda self: None)
    monkeypatch.setattr(ConverterWorker, 'start', lambda self: None)
    dialog = BatchConverterDialog()
    dialog.show()
    qapp.processEvents()
    try:
        yield dialog
    finally:
        dialog.worker = None
        dialog.close()


def _converter_drop(qapp, target, urls):
    from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt
    from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    mime = QMimeData()
    mime.setUrls(urls)
    enter = QDragEnterEvent(QPoint(5, 5), Qt.DropAction.CopyAction, mime,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(target, enter)
    move = QDragMoveEvent(QPoint(5, 5), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(target, move)
    drop = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                      Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    qapp.sendEvent(target, drop)
    return enter, move, drop


@pytest.mark.parametrize('target', ['dialog', 'viewport'])
def test_converter_real_drop_expands_folders_orders_and_reports_skips(
        qapp, converter_dialog, tmp_path, target):
    from pathlib import Path
    from PyQt6.QtCore import QUrl
    folder = tmp_path / 'images'
    folder.mkdir()
    for name in ('frame10.PNG', 'frame2.png', 'notes.txt'):
        (folder / name).touch()
    nested = folder / 'nested'
    nested.mkdir()
    (nested / 'not-a-direct-child.png').touch()
    first = tmp_path / 'frame1.png'
    first.touch()
    dialog = converter_dialog
    dialog._append([first])
    receiver = dialog if target == 'dialog' else dialog.list.viewport()
    events = _converter_drop(qapp, receiver, [
        QUrl.fromLocalFile(str(folder)), QUrl.fromLocalFile(str(folder / 'frame2.png')),
        QUrl.fromLocalFile(str(first)), QUrl('https://example.com/remote.png')])
    assert all(event.isAccepted() for event in events)
    assert [Path(path).name for path in dialog.sources] == ['frame1.png', 'frame2.png', 'frame10.PNG']
    assert dialog.list.count() == 3
    assert dialog.empty_hint.isHidden()


def test_converter_picker_and_drop_share_order_dedup_and_empty_feedback(
        qapp, converter_dialog, tmp_path, monkeypatch):
    from PyQt6.QtCore import QUrl
    from dialogs.batch_converter import FileDialog
    dialog = converter_dialog
    folder = tmp_path / 'images'
    folder.mkdir()
    sources = [folder / 'image10.tif', folder / 'image2.tif']
    for source in sources:
        source.touch()
    monkeypatch.setattr(FileDialog, 'getOpenFileNames', lambda *a: ([str(p) for p in sources], ''))
    dialog.add_files.click()
    assert dialog.sources == [str(sources[1]), str(sources[0])]
    monkeypatch.setattr(FileDialog, 'getExistingDirectory', lambda *a: str(folder))
    dialog.add_folder.click()
    assert len(dialog.sources) == 2
    before = dialog.summary.text()
    monkeypatch.setattr(FileDialog, 'getOpenFileNames', lambda *a: ([], ''))
    dialog.add_files.click()
    assert dialog.summary.text() == before
    empty = tmp_path / 'empty'
    empty.mkdir()
    events = _converter_drop(qapp, dialog.list.viewport(), [QUrl.fromLocalFile(str(empty))])
    assert all(event.isAccepted() for event in events)


def test_converter_remote_drop_is_rejected_and_unsupported_local_drop_reports_skip(
        qapp, converter_dialog, tmp_path):
    from PyQt6.QtCore import QUrl
    dialog = converter_dialog
    events = _converter_drop(qapp, dialog.list.viewport(), [QUrl('https://example.com/photo.png')])
    assert not any(event.isAccepted() for event in events)
    assert not dialog.sources
    unsupported = tmp_path / 'notes.txt'
    unsupported.touch()
    events = _converter_drop(qapp, dialog.list.viewport(), [QUrl.fromLocalFile(str(unsupported))])
    assert all(event.isAccepted() for event in events)
    assert not dialog.empty_hint.isHidden()
    assert not dialog.start.isEnabled()


@pytest.mark.parametrize('delete_key', ['Key_Delete', 'Key_Backspace'])
def test_converter_output_path_keyboard_and_delete_scope(
        qapp, converter_dialog, tmp_path, monkeypatch, delete_key):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from dialogs.batch_converter import FileDialog
    dialog = converter_dialog
    source = tmp_path / 'photo.png'
    source.touch()
    monkeypatch.setattr(FileDialog, 'getOpenFileNames', lambda *a: pytest.fail('Enter opened Add Files'))
    QTest.keyClick(dialog, Qt.Key.Key_Return)
    dialog.directory.setText(str(tmp_path))
    assert not dialog.start.isEnabled()
    dialog._append([source])
    assert dialog.start.isEnabled() and not dialog.remove.isEnabled()
    dialog.list.setCurrentRow(0)
    assert dialog.remove.isEnabled() and dialog.delete_shortcut.isEnabled()
    assert dialog.backspace_shortcut.isEnabled()
    dialog.directory.setFocus()
    dialog.directory.selectAll()
    QTest.keyClick(dialog.directory, getattr(Qt.Key, delete_key))
    assert dialog.list.count() == 1 and not dialog.start.isEnabled()
    dialog.directory.setText(str(source))
    assert not dialog.start.isEnabled() and dialog.directory.toolTip()
    dialog.directory.setText(str(tmp_path / 'missing'))
    assert not dialog.start.isEnabled()
    monkeypatch.setenv('OPENFOCUS_CONVERTER_OUTPUT', str(tmp_path))
    dialog.directory.setText('"%OPENFOCUS_CONVERTER_OUTPUT%"')
    assert dialog.start.isEnabled() and not dialog.directory.toolTip()
    QTest.keyClick(dialog.directory, Qt.Key.Key_Return)
    assert not dialog.start.isEnabled() and not dialog.remove.isEnabled()
    dialog.worker.finished.emit()
    assert dialog.start.isEnabled() and dialog.remove.isEnabled()
    dialog.list.setFocus()
    qapp.processEvents()
    QTest.keyClick(dialog.list, getattr(Qt.Key, delete_key))
    assert not dialog.sources and dialog.list.count() == 0
    assert source.exists()
    assert not dialog.start.isEnabled() and not dialog.remove.isEnabled()
    assert not dialog.empty_hint.isHidden()


@pytest.mark.parametrize('target', ['dialog', 'viewport'])
def test_converter_busy_rejects_drop_mutation_and_repeat_start_until_finished(
        qapp, converter_dialog, tmp_path, monkeypatch, target):
    from PyQt6.QtCore import QUrl
    from dialogs.batch_converter import FileDialog
    dialog = converter_dialog
    source = tmp_path / 'photo.png'
    extra = tmp_path / 'extra.png'
    source.touch()
    extra.touch()
    dialog._append([source])
    dialog.directory.setText(str(tmp_path))
    dialog.list.setCurrentRow(0)
    dialog.start.click()
    worker = dialog.worker
    assert worker is not None and not worker.isRunning()
    monkeypatch.setattr(FileDialog, 'getOpenFileNames', lambda *a: pytest.fail('Busy file picker'))
    monkeypatch.setattr(FileDialog, 'getExistingDirectory', lambda *a: pytest.fail('Busy folder picker'))
    dialog._add_files()
    dialog._add_folder()
    dialog._browse()
    dialog._append([extra])
    dialog._remove()
    dialog._start()
    receiver = dialog if target == 'dialog' else dialog.list.viewport()
    events = _converter_drop(qapp, receiver, [QUrl.fromLocalFile(str(extra))])
    assert not any(event.isAccepted() for event in events)
    assert dialog.worker is worker and dialog.sources == [str(source)]
    assert all(not control.isEnabled() for control in dialog.controls)
    dialog.cancel.click()
    assert not dialog.cancel.isEnabled()
    assert not dialog.start.isEnabled() and not dialog.delete_shortcut.isEnabled()
    assert not dialog.backspace_shortcut.isEnabled()
    worker.complete.emit(0, 0, True)
    assert not dialog.start.isEnabled()
    worker.finished.emit()
    assert dialog.start.isEnabled() and dialog.remove.isEnabled()
    assert not dialog.cancel.isEnabled()


def test_exif_orientation_applied_and_reset(tmp_path):
    source = tmp_path / 'portrait.jpg'
    image = Image.new('RGB', (8, 4), 'red')
    exif = Image.Exif()
    exif[274] = 6
    exif[271] = 'Camera'
    image.save(source, exif=exif)
    result = converter.convert_one(str(source), str(tmp_path), '.jpg')
    with Image.open(result) as output:
        assert output.size == (4, 8)
        assert output.getexif().get(274, 1) == 1
        assert output.getexif().get(271) == 'Camera'


def test_raw_development_uses_srgb_and_embeds_profile(tmp_path):
    rawpy = pytest.importorskip('rawpy')
    from utils.image_utils import srgb_icc_profile
    from utils.linear_dng import LinearDNGMetadata, write_linear_dng

    source = tmp_path / '相机.dng'
    pixels = np.full((128, 128, 3), [7000, 11000, 14000], np.uint16)
    metadata = LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667), 'Canon', 'Canon EOS R6')
    write_linear_dng(source, pixels, metadata)
    with open(source, 'rb') as stream, rawpy.imread(stream) as raw:
        expected = raw.postprocess(use_camera_wb=True, output_bps=16,
                                   no_auto_bright=True, gamma=(2.4, 12.92))
        bt709 = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True)
    assert np.max(np.abs(expected.astype(np.int32) - bt709.astype(np.int32))) > 50
    loaded = converter.image_loader._read_raw(str(source))
    np.testing.assert_array_equal(loaded, expected[..., ::-1])
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, expected[..., ::-1])
    with Image.open(result) as output:
        assert output.info['icc_profile'] == srgb_icc_profile()
    assert converter.image_loader.common_icc_profile([str(source)]) == srgb_icc_profile()


def test_lzw_tiff_pages_and_per_page_admission(tmp_path, monkeypatch):
    source = tmp_path / '压缩.tif'
    first = np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000
    second = np.full((8, 8), 40000, dtype=np.uint16)
    Image.fromarray(first).save(source, compression='tiff_lzw', save_all=True,
                                append_images=[Image.fromarray(second)])
    # Whole-file size must not be charged for each TIFF page.
    monkeypatch.setattr(converter.os.path, 'getsize', lambda path: 100 * 1024**3)
    assert converter.page_count(str(source)) == 2
    for page, expected in enumerate((first, second)):
        result = converter.convert_one(str(source), str(tmp_path), '.png', page, 2)
        actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, expected)


def test_jpeg_writer_preserves_safe_exif_and_icc(tmp_path):
    from PIL import ImageCms
    from utils.image_utils import imwrite_auto
    exif = Image.Exif()
    exif[271] = 'Camera'
    exif[274] = 6
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    destination = tmp_path / 'out.jpg'
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    assert imwrite_auto(str(destination), pixels, exif=exif.tobytes(), icc_profile=profile)
    with Image.open(destination) as result:
        assert result.getexif()[271] == 'Camera'
        assert result.getexif()[274] == 1
        assert result.info['icc_profile'] == profile


def test_lzw_tiff_alpha_is_rejected(tmp_path):
    source = tmp_path / 'alpha.tif'
    Image.new('RGBA', (4, 4), (100, 50, 25, 128)).save(source, compression='tiff_lzw')
    with pytest.raises(ValueError, match='alpha'):
        converter.convert_one(str(source), str(tmp_path), '.png')


def test_lzw_tiff_orientation(tmp_path):
    source = tmp_path / 'rotated.tif'
    pixels = np.arange(24, dtype=np.uint8).reshape(4, 6)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(source, compression='tiff_lzw', exif=exif)
    result = converter.convert_one(str(source), str(tmp_path), '.png')
    actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, np.rot90(pixels, -1))


def test_parallel_output_reservations_never_collide(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    source = tmp_path / 'shared.png'
    Image.new('RGB', (16, 16), 'red').save(source)
    before = source.read_bytes()
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: converter.convert_one(str(source), str(tmp_path), '.jpg'), range(12)))
    assert len(set(outputs)) == 12
    assert source.read_bytes() == before
    for output in outputs:
        with Image.open(output) as image:
            assert image.size == (16, 16)


def test_dng_rejects_developed_input_without_creating_output(tmp_path):
    source = tmp_path / 'developed.png'
    Image.new('RGB', (8, 8)).save(source)
    with pytest.raises(ValueError, match='camera RAW'):
        converter.convert_one(str(source), str(tmp_path), '.dng')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['developed.png']


def test_cancelled_batch_keeps_completed_page_and_reports_unstarted(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from threading import Event
    source = tmp_path / 'pages.tif'
    with tifffile.TiffWriter(source) as tif:
        for value in (1000, 2000, 3000):
            tif.write(np.full((8, 8), value, np.uint16))
    remaining = tmp_path / 'remaining.png'
    Image.new('RGB', (8, 8)).save(remaining)
    cancelled = Event()
    real = converter.convert_one

    def convert(*args, **kwargs):
        output = real(*args, **kwargs)
        cancelled.set()
        return output

    monkeypatch.setattr(converter, 'convert_one', convert)
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=1, native_threads=1, max_in_flight=1,
                                        ram_budget=1024**3, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 1024**3)
    statuses, progress = {}, []
    result = converter.convert_batch([str(source), str(remaining)], str(tmp_path), '.png',
                                     cancelled.is_set, lambda i, text: statuses.update({i: text}), progress.append)
    assert result == (0, 0, True)
    assert (tmp_path / 'pages_page_0001.png').is_file()
    assert not (tmp_path / 'pages_page_0002.png').exists()
    assert 'Cancelled' in statuses[0] and statuses[1] == 'Cancelled'
    assert not list(tmp_path.glob('tmp*'))


def test_job_memory_reservation_bounds_parallel_admission(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    from types import SimpleNamespace
    entered, release = Event(), Event()
    calls, lock = [], Lock()
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=3, max_in_flight=3, ram_budget=150, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 150)
    monkeypatch.setattr(converter, 'page_count', lambda *a: 1)
    monkeypatch.setattr(converter, 'estimate_job', lambda *a: (100, 1))

    def convert(index, *args):
        with lock:
            calls.append(index)
        entered.set()
        assert release.wait(5)
        return True, False

    monkeypatch.setattr(converter, '_convert_source', convert)
    with ThreadPoolExecutor(max_workers=1) as owner:
        future = owner.submit(converter.convert_batch, ['a', 'b', 'c'], str(tmp_path), '.png',
                              lambda: False, lambda *a: None, lambda *a: None)
        try:
            assert entered.wait(5)
            # The held job consumes 100 of 150 bytes. No second job may enter.
            assert calls == [0]
        finally:
            release.set()
        assert future.result(timeout=5) == (3, 0, False)


def test_thumbnail_refuses_large_raster_before_loading(tmp_path, monkeypatch):
    import struct
    import zlib
    from PIL import PngImagePlugin

    source = tmp_path / 'large-header-small-idat.png'
    Image.new('RGB', (1, 1), 'red').save(source)
    encoded = bytearray(source.read_bytes())
    assert encoded[12:16] == b'IHDR'
    # Keep the tiny IDAT: only the real PNG header advertises 2500x2500.
    # Its valid IHDR CRC allows Pillow's native header reader to accept it.
    struct.pack_into('>II', encoded, 16, 2500, 2500)
    struct.pack_into('>I', encoded, 29, zlib.crc32(encoded[12:29]) & 0xffffffff)
    source.write_bytes(encoded)
    loads = []

    def load(*args, **kwargs):
        loads.append(True)
        raise AssertionError('PNG pixels were decoded before thumbnail admission')

    monkeypatch.setattr(PngImagePlugin.PngImageFile, 'load', load)
    with pytest.raises(ValueError):
        converter.thumbnail_rgb(str(source))
    assert loads == []


def test_large_raw_without_preview_never_unpacks_sensor(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class RawHeader:
        sizes = SimpleNamespace(raw_width=100000, raw_height=100000,
                                width=100000, height=100000, flip=0)
        def open_file(self, stream):
            pass
        def extract_thumb(self):
            pytest.fail('Unbounded LibRaw bitmap thumbnail decode')
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    source = tmp_path / 'giant.cr3'
    source.write_bytes(b'header')
    monkeypatch.setattr(converter.image_loader.rawpy, 'RawPy', RawHeader)
    monkeypatch.setattr(converter, '_embedded_raw_jpegs', lambda *a, **kw: iter(()))
    with pytest.raises(ValueError, match='bounded'):
        converter.thumbnail_rgb(str(source))


def test_thumbnail_stale_completion_cannot_modify_readded_row(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QImage
    from dialogs.batch_converter import BatchConverterDialog, ThumbnailWorker
    monkeypatch.setattr(ThumbnailWorker, 'start', lambda self: None)
    source = str(tmp_path / 'photo.png')
    (tmp_path / 'photo.png').touch()
    dialog = BatchConverterDialog()
    try:
        dialog._append([source])
        old_token = dialog.list.item(0).data(Qt.ItemDataRole.UserRole)
        dialog.list.item(0).setSelected(True)
        dialog._remove()
        dialog._append([source])
        item = dialog.list.item(0)
        assert item.data(Qt.ItemDataRole.UserRole) != old_token
        before = item.icon().cacheKey()
        image = QImage(20, 20, QImage.Format.Format_RGB888)
        image.fill(Qt.GlobalColor.red)
        dialog._thumbnail_ready(old_token, image, 'old failure')
        assert item.icon().cacheKey() == before and 'old failure' not in item.toolTip()
        assert item.sizeHint().height() == 88
        assert dialog.list.iconSize().width() == 96
    finally:
        dialog.close()


def test_thumbnail_queue_is_bounded_and_delete_is_disabled_while_busy(qapp, tmp_path, monkeypatch):
    from dialogs.batch_converter import BatchConverterDialog, ThumbnailWorker, ConverterWorker
    monkeypatch.setattr(ThumbnailWorker, 'start', lambda self: None)
    monkeypatch.setattr(ConverterWorker, 'start', lambda self: None)
    dialog = BatchConverterDialog()
    try:
        sources = [tmp_path / f'{index}.png' for index in range(100)]
        for source in sources:
            source.touch()
        dialog._append(sources)
        assert len(dialog.thumbnail_worker._queue) == ThumbnailWorker.capacity
        dialog.directory.setText(str(tmp_path))
        dialog._start()
        assert not dialog.delete_shortcut.isEnabled()
        assert not dialog.remove.isEnabled() and not dialog.add_files.isEnabled()
        monkeypatch.setattr(ConverterWorker, 'isRunning', lambda self: True)
        dialog.list.item(0).setSelected(True)
        dialog._remove()
        assert len(dialog.sources) == 100
    finally:
        dialog.worker = None
        dialog.close()


def test_multipage_partial_failure_retains_successful_pages(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / 'partial.tif'
    with tifffile.TiffWriter(source) as tif:
        for value in (1000, 2000, 3000):
            tif.write(np.full((8, 8), value, np.uint16))
    decode = converter._decode

    def fail_middle(path, page):
        if page == 1:
            raise ValueError('damaged middle page')
        return decode(path, page)

    monkeypatch.setattr(converter, '_decode', fail_middle)
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2, ram_budget=1024**3, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 1024**3)
    statuses = {}
    result = converter.convert_batch([str(source)], str(tmp_path), '.png', lambda: False,
                                     lambda i, text: statuses.update({i: text}), lambda *a: None)
    assert result == (0, 1, False)
    assert (tmp_path / 'partial_page_0001.png').is_file()
    assert not (tmp_path / 'partial_page_0002.png').exists()
    assert (tmp_path / 'partial_page_0003.png').is_file()
    assert 'damaged middle page' in statuses[0]


def test_batch_restores_global_native_codec_threads_after_failure(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    options = SimpleNamespace(DECODE_THREADS=7)
    monkeypatch.setitem(sys.modules, 'pillow_heif', SimpleNamespace(options=options))
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=1, native_threads=3, max_in_flight=1,
                                        ram_budget=1024**3, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 1024**3)
    monkeypatch.setattr(converter, 'page_count', lambda *a: 1)
    monkeypatch.setattr(converter, 'estimate_job', lambda *a: (100, 1))
    original_cv = converter.cv2.getNumThreads()

    def failed_job(*args):
        assert options.DECODE_THREADS == 3
        assert converter.cv2.getNumThreads() == 3
        raise ValueError('codec failed')

    monkeypatch.setattr(converter, '_convert_source', failed_job)
    result = converter.convert_batch(['image.heic'], str(tmp_path), '.png', lambda: False,
                                     lambda *a: None, lambda *a: None)
    assert result == (0, 1, False)
    assert options.DECODE_THREADS == 7
    assert converter.cv2.getNumThreads() == original_cv


def test_converter_close_waits_for_worker_and_keeps_controls_disabled(qapp, tmp_path, monkeypatch):
    from PyQt6.QtGui import QCloseEvent
    from dialogs.batch_converter import BatchConverterDialog, ThumbnailWorker, ConverterWorker
    monkeypatch.setattr(ThumbnailWorker, 'start', lambda self: None)
    monkeypatch.setattr(ConverterWorker, 'start', lambda self: None)
    dialog = BatchConverterDialog()
    (tmp_path / 'input.png').touch()
    dialog._append([str(tmp_path / 'input.png')])
    dialog.directory.setText(str(tmp_path))
    dialog._start()
    interrupted = []
    running = [True]
    monkeypatch.setattr(ConverterWorker, 'isRunning', lambda self: running[0])
    monkeypatch.setattr(ConverterWorker, 'requestInterruption', lambda self: interrupted.append(True))
    try:
        event = QCloseEvent()
        dialog.closeEvent(event)
        assert not event.isAccepted() and dialog._close_pending and interrupted
        assert not dialog.remove.isEnabled() and not dialog.delete_shortcut.isEnabled()
        running[0] = False
        dialog._finished()
        assert not dialog.start.isEnabled() and not dialog.delete_shortcut.isEnabled()
    finally:
        dialog.worker = None
        dialog.close()


def test_progressive_thumbnail_admits_original_dct_scratch_before_decode(tmp_path, monkeypatch):
    required = []

    class ProgressiveHeader:
        format = 'JPEG'
        width = height = 10000
        info = {'progressive': 1}

        def getexif(self):
            return {}

        def getbands(self):
            return ('R', 'G', 'B')

        def draft(self, *args):
            self.width = self.height = 1250

        def thumbnail(self, *args):
            pytest.fail('Decoded progressive JPEG before admitting its original DCT scratch')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def deny(required_bytes, *args, **kwargs):
        required.append(required_bytes)
        raise MemoryError('Thumbnail exceeds the available resource budget')

    monkeypatch.setattr(converter.Image, 'open', lambda *a: ProgressiveHeader())
    monkeypatch.setattr(converter.resource_policy, 'require_memory', deny)
    # Header-only simulation: no large image is created or decoded.
    with pytest.raises(MemoryError, match='resource budget'):
        converter.thumbnail_rgb(str(tmp_path / 'progressive.jpg'))
    assert len(required) == 1
    assert required[0] >= 10000 * 10000 * 3 * 2 + 1250 * 1250 * 16


def test_small_progressive_preview_preserves_geometry_for_admission(tmp_path, monkeypatch):
    source = tmp_path / 'small-progressive.jpg'
    Image.new('RGB', (320, 240), 'red').save(source, progressive=True, subsampling=0)
    required = []
    monkeypatch.setattr(converter.resource_policy, 'require_memory', lambda n, **kw: required.append(n))
    pixels = converter.thumbnail_rgb(str(source))
    assert pixels.shape == (72, 96, 3)
    assert len(required) == 1 and required[0] >= 320 * 240 * 3 * 2


def test_tiff_preflight_visits_headers_once_and_stops_before_next_page(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    cancelled = Event()
    opened, headers = [], []

    class HeaderFile:
        is_dng = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @property
        def pages(self):
            for index in range(10000):
                headers.append(index)
                yield SimpleNamespace(imagewidth=8, imagelength=8, databytecounts=(20,))

    def open_tiff(path):
        opened.append(path)
        return HeaderFile()

    real_estimate = converter._image_requirements

    def estimate(*args):
        result = real_estimate(*args)
        cancelled.set()
        return result

    monkeypatch.setattr(converter.tifffile, 'TiffFile', open_tiff)
    monkeypatch.setattr(converter, '_image_requirements', estimate)
    monkeypatch.setattr(converter, 'estimate_job', lambda *a: pytest.fail('Reopened a TIFF page'))
    with pytest.raises(InterruptedError, match='source admission'):
        converter._estimate_source(str(tmp_path / 'many-pages.tif'), cancelled.is_set)
    assert len(opened) == 1 and headers == [0]


def test_tiff_preflight_uses_one_open_for_all_page_budgets(tmp_path, monkeypatch):
    from types import SimpleNamespace
    entries = [SimpleNamespace(imagewidth=8, imagelength=9, databytecounts=(20,)),
               SimpleNamespace(imagewidth=80, imagelength=90, databytecounts=(200,))]
    opened = []

    class HeaderFile:
        is_dng = False
        pages = entries

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def open_tiff(path):
        opened.append(path)
        return HeaderFile()

    monkeypatch.setattr(converter.tifffile, 'TiffFile', open_tiff)
    monkeypatch.setattr(converter, 'estimate_job', lambda *a: pytest.fail('Reopened a TIFF page'))
    assert converter._estimate_source(str(tmp_path / 'two-pages.tif'), lambda: False) == (
        2, *converter._image_requirements(80, 90, 200))
    assert len(opened) == 1


def test_cancelled_preflight_drains_active_job_without_reporting_failure(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    cancelled, estimated, statuses = Event(), [], {}
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2, ram_budget=150, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 150)

    def estimate(source, should_cancel, ext=None):
        estimated.append(source)
        if source == 'second':
            cancelled.set()
            raise InterruptedError('Cancelled while inspecting TIFF pages')
        return 1, 10, 1

    def complete_active(*args):
        assert cancelled.wait(5)
        return True, False

    monkeypatch.setattr(converter, '_estimate_source', estimate)
    monkeypatch.setattr(converter, '_convert_source', complete_active)
    result = converter.convert_batch(['first', 'second', 'third'], str(tmp_path), '.png',
                                     cancelled.is_set, lambda i, status: statuses.update({i: status}),
                                     lambda *a: None)
    assert result == (1, 0, True)
    assert estimated == ['first', 'second']
    assert statuses == {1: 'Cancelled', 2: 'Cancelled'}


def test_png_thumbnail_reads_exif_after_idat_before_pixel_admission(tmp_path, monkeypatch):
    import struct
    import zlib
    from PIL import PngImagePlugin

    source = tmp_path / 'trailing-orientation.png'
    original = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    Image.fromarray(original).save(source)
    encoded = source.read_bytes()
    assert encoded[-8:-4] == b'IEND'
    exif = Image.Exif()
    exif[274] = 6
    payload = exif.tobytes()[6:]  # PNG eXIf contains TIFF bytes without Exif\0\0.
    chunk = b'eXIf' + payload
    encoded = (encoded[:-12] + struct.pack('>I', len(payload)) + chunk
               + struct.pack('>I', zlib.crc32(chunk) & 0xffffffff) + encoded[-12:])
    source.write_bytes(encoded)
    admitted = []
    original_memory = converter.resource_policy.require_memory
    original_load = PngImagePlugin.PngImageFile.load

    def load(image, *args, **kwargs):
        assert admitted, 'EXIF lookup decoded PNG pixels before memory admission'
        return original_load(image, *args, **kwargs)

    def admit(amount, **kwargs):
        original_memory(amount, **kwargs)
        admitted.append(amount)

    monkeypatch.setattr(converter.resource_policy, 'require_memory', admit)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, 'load', load)
    pixels = converter.thumbnail_rgb(str(source))
    np.testing.assert_array_equal(pixels, np.rot90(original, -1))


def test_mapped_rgb16_tiff_preserves_channels_orientation_and_profile(tmp_path):
    from utils.image_utils import srgb_icc_profile
    source = tmp_path / 'sensor.tif'
    rgb = (np.arange(11 * 17 * 3, dtype=np.uint16).reshape(11, 17, 3) * 101)
    profile = srgb_icc_profile()
    tifffile.imwrite(source, rgb, photometric='rgb', metadata=None,
                     extratags=[(274, 'H', 1, 6, False),
                                (271, 's', 0, 'Camera', False),
                                (34675, 'B', len(profile), profile, False)])
    output = converter.convert_one(str(source), str(tmp_path), '.tif')
    with tifffile.TiffFile(output) as tif:
        page = tif.pages[0]
        np.testing.assert_array_equal(page.asarray(), np.rot90(rgb, -1))
        assert page.bitspersample == 16
        assert page.tags[274].value == 1
        assert page.tags[271].value == 'Camera'
        assert page.tags[34675].value == profile


def test_existing_dng_copy_fits_small_budget_without_sensor_unpack(tmp_path, monkeypatch):
    from utils.linear_dng import LinearDNGMetadata, write_linear_dng
    source = tmp_path / 'large-sensor.dng'
    pixels = np.full((128, 128, 3), (1000, 5000, 12000), np.uint16)
    matrix = ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
              (-0.1013, 0.2415, 0.5509))
    write_linear_dng(source, pixels, LinearDNGMetadata(matrix, (0.5, 1., 0.7),
                                                     'Camera', 'Camera Model'))
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 256 * 1024**2)
    real_rawpy = converter.image_loader.rawpy.RawPy

    class HeaderOnly:
        def __enter__(self):
            from types import SimpleNamespace
            self.raw = real_rawpy()
            self.sizes = SimpleNamespace(raw_width=24000, raw_height=18000)
            return self

        def open_file(self, path):
            self.raw.open_file(path)

        def __exit__(self, *args):
            self.raw.close()

    # Missing unpack/postprocess methods make accidental sensor development
    # fail, while the copied DNG remains a real independently readable file.
    monkeypatch.setattr(converter.image_loader.rawpy, 'RawPy', HeaderOnly)
    output = converter.convert_one(str(source), str(tmp_path), '.dng')
    assert source.read_bytes() == Path(output).read_bytes()
    assert converter.estimate_job(source)[0] > 20 * 1024**3
    assert converter.estimate_job(source, ext='.dng')[0] <= 64 * 1024**2


def test_cancelled_metadata_tiff_retains_original_and_existing_output(tmp_path):
    from core.registration import RegistrationCancelled
    source = tmp_path / 'sensor.tif'
    rgb = np.full((300, 11, 3), (1000, 9000, 50000), np.uint16)
    tifffile.imwrite(source, rgb, photometric='rgb', metadata=None,
                     extratags=[(271, 's', 0, 'Camera', False)])
    directory = tmp_path / 'output'
    directory.mkdir()
    existing = directory / source.name
    existing.write_bytes(b'keep existing output')
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 4

    with pytest.raises((InterruptedError, RegistrationCancelled)):
        converter.convert_one(str(source), str(directory), '.tif', cancelled=cancelled)
    assert list(directory.iterdir()) == [existing]
    assert existing.read_bytes() == b'keep existing output'
    np.testing.assert_array_equal(tifffile.imread(source), rgb)
