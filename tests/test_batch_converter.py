from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image
from test_input_formats import cmyk_profile, _cmyk_pixels

from core import batch_converter as converter


@pytest.fixture(autouse=True)
def memory_budget(monkeypatch):
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 8 * 1024**3)


def _assert_batch(result, **expected):
    counts = dict.fromkeys(('success', 'failed', 'partial', 'cancelled', 'not_started'), 0)
    counts.update(expected)
    assert result['counts'] == counts
    assert sum(counts.values()) == len(result['items'])
    assert [item['status'] for item in result['items']].count('active') == 0
    assert [item['status'] for item in result['items']].count('pending') == 0


def _successful_record(index, source):
    return dict(index=index, source=source, status='success', outputs=['output'],
                errors=[], stop_reason=None)

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


@pytest.mark.parametrize('stage', ['writer', 'after_write'])
def test_cancelled_conversion_removes_reserved_and_temporary_outputs(tmp_path, monkeypatch, stage):
    source = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(source)
    before = source.read_bytes()
    stopped = False
    write = converter.imwrite_auto

    def cancel_at_write(*args, **kwargs):
        nonlocal stopped
        if stage == 'writer':
            stopped = True
        result = write(*args, **kwargs)
        stopped = True
        return result

    monkeypatch.setattr(converter, 'imwrite_auto', cancel_at_write)
    with pytest.raises(InterruptedError):
        converter.convert_one(str(source), str(tmp_path), '.jpg', cancelled=lambda: stopped)
    assert source.read_bytes() == before
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


@pytest.fixture
def clean_converter_format(qapp):
    """Start each dialog test without a remembered converter format."""
    from utils.settings_store import CONVERTER_FORMAT_KEY, get_settings
    settings = get_settings()
    settings.remove(CONVERTER_FORMAT_KEY)
    yield settings
    settings.remove(CONVERTER_FORMAT_KEY)


def test_dialog_starts_without_main_window(qapp, clean_converter_format):
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
def converter_dialog(qapp, monkeypatch, clean_converter_format):
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


def test_converter_format_risks_follow_the_selected_format(converter_dialog):
    dialog = converter_dialog
    for image_format in ('JPEG', 'BMP', 'WebP'):
        dialog.format.setCurrentText(image_format)
        assert not dialog.format_warning.isHidden()
        assert '16' in dialog.format_warning.text()
    dialog.format.setCurrentText('DNG')
    assert not dialog.format_warning.isHidden()
    assert 'DNG' in dialog.format_warning.text()
    dialog.format.setCurrentText('TIFF')
    assert not dialog.format_warning.isHidden()
    assert 'PNG' in dialog.format_warning.text()
    dialog.format.setCurrentText('PNG')
    assert dialog.format_warning.isHidden()


def test_converter_remembers_explicit_format_and_recovers_from_a_bad_value(
        qapp, clean_converter_format):
    from dialogs.batch_converter import BatchConverterDialog
    from utils.settings_store import CONVERTER_FORMAT_KEY
    first = BatchConverterDialog()
    try:
        first.format.setCurrentText('JPEG')
    finally:
        first.close()
    second = BatchConverterDialog()
    try:
        assert second.format.currentText() == 'JPEG'
        second.format.setCurrentText('DNG')
    finally:
        second.close()
    third = BatchConverterDialog()
    try:
        assert third.format.currentText() == 'DNG'
    finally:
        third.close()
    clean_converter_format.setValue(CONVERTER_FORMAT_KEY, 'JPEG2000')
    fallback = BatchConverterDialog()
    try:
        assert fallback.format.currentText() == 'PNG'
    finally:
        fallback.close()






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
    assert dialog.start.isEnabled() and not dialog.directory.toolTip()
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
    worker.complete.emit({'items': [dict(index=0, source=str(source), status='cancelled',
                                         outputs=[], errors=[], stop_reason='user_cancel')],
                          'counts': dict(success=0, failed=0, partial=0, cancelled=1, not_started=0),
                          'user_cancelled': True, 'scheduler_error': None})
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


@pytest.mark.parametrize('orientation', [1, 6])
def test_16bit_planar_lzw_pages_preserve_pixels(tmp_path, orientation):
    source = tmp_path / 'planar-lzw.tif'
    rows, cols = np.indices((32, 40), dtype=np.uint16)
    rgb = np.stack((rows * 1000 + cols, cols * 1200 + rows,
                    rows * 313 + cols * 719), axis=-1)
    pages = (rgb, np.uint16(65535) - rgb)
    with tifffile.TiffWriter(source) as tif:
        for pixels in pages:
            tif.write(np.moveaxis(pixels, -1, 0), photometric='rgb',
                      planarconfig='separate', compression='lzw', metadata=None,
                      extratags=[(274, 'H', 1, orientation, False)])
    for page, pixels in enumerate(pages):
        result = converter.convert_one(str(source), str(tmp_path), '.png', page, 2)
        actual = cv2.imdecode(np.fromfile(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        expected = pixels[..., ::-1]
        if orientation == 6:
            expected = np.rot90(expected, -1)
        assert actual.dtype == np.uint16
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
    events, progress = [], []
    result = converter.convert_batch([str(source), str(remaining)], str(tmp_path), '.png',
                                     cancelled.is_set, events.append, progress.append)
    _assert_batch(result, partial=1, not_started=1)
    assert result['user_cancelled']
    assert (tmp_path / 'pages_page_0001.png').is_file()
    assert not (tmp_path / 'pages_page_0002.png').exists()
    assert result['items'][0]['outputs'] == [str(tmp_path / 'pages_page_0001.png')]
    assert result['items'][1]['stop_reason'] == 'user_cancel'
    assert progress[-1] == 2
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
        return _successful_record(index, args[0])

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
        _assert_batch(future.result(timeout=5), success=3)


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

    def fail_middle(path, page, *args, **kwargs):
        if page == 1:
            raise ValueError('damaged middle page')
        return decode(path, page, *args, **kwargs)

    monkeypatch.setattr(converter, '_decode', fail_middle)
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2, ram_budget=1024**3, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 1024**3)
    events = []
    result = converter.convert_batch([str(source)], str(tmp_path), '.png', lambda: False,
                                     events.append, lambda *a: None)
    _assert_batch(result, partial=1)
    assert (tmp_path / 'partial_page_0001.png').is_file()
    assert not (tmp_path / 'partial_page_0002.png').exists()
    assert (tmp_path / 'partial_page_0003.png').is_file()
    assert any('damaged middle page' in str(error) for error in result['items'][0]['errors'])
    assert len(result['items'][0]['outputs']) == 2


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
    _assert_batch(result, failed=1)
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
    monkeypatch.setattr(converter, '_tiff_requirements',
                        lambda path, page: estimate(page.imagewidth, page.imagelength,
                                                    sum(page.databytecounts)))
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
    monkeypatch.setattr(converter, '_tiff_requirements',
                        lambda path, page: converter._image_requirements(
                            page.imagewidth, page.imagelength, sum(page.databytecounts)))
    monkeypatch.setattr(converter, 'estimate_job', lambda *a: pytest.fail('Reopened a TIFF page'))
    assert converter._estimate_source(str(tmp_path / 'two-pages.tif'), lambda: False) == (
        2, *converter._image_requirements(80, 90, 200))
    assert len(opened) == 1


def test_cancelled_preflight_drains_active_job_without_reporting_failure(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    cancelled, estimated, events = Event(), [], []
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, max_in_flight=2, ram_budget=150, describe=lambda: 'test'))
    monkeypatch.setattr(converter.resource_policy, 'ram_budget', lambda *a: 150)

    def estimate(source, should_cancel, ext=None):
        estimated.append(source)
        if source == 'second':
            cancelled.set()
            raise InterruptedError('Cancelled while inspecting TIFF pages')
        return 1, 10, 1

    def complete_active(index, source, *args):
        assert cancelled.wait(5)
        return _successful_record(index, source)

    monkeypatch.setattr(converter, '_estimate_source', estimate)
    monkeypatch.setattr(converter, '_convert_source', complete_active)
    result = converter.convert_batch(['first', 'second', 'third'], str(tmp_path), '.png',
                                     cancelled.is_set, events.append, lambda *a: None)
    _assert_batch(result, success=1, not_started=2)
    assert result['user_cancelled']
    assert estimated == ['first', 'second']
    assert [item['status'] for item in result['items']] == ['success', 'not_started', 'not_started']


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


def test_batch_creates_unicode_parents_and_does_not_repeat_page_admission(tmp_path, monkeypatch):
    source = tmp_path / '花.png'
    Image.new('RGB', (8, 8), 'red').save(source)
    original = source.read_bytes()
    directory = tmp_path / '新建' / '输出'
    monkeypatch.setattr(converter, '_admit', lambda *a: pytest.fail('batch re-admitted its reserved page'))
    events, progress = [], []
    result = converter.convert_batch([str(source)], str(directory), '.jpg', lambda: False,
                                     events.append, progress.append)
    _assert_batch(result, success=1)
    assert progress[-1] == 1
    assert source.read_bytes() == original
    with Image.open(directory / '花.jpg') as output:
        assert output.size == (8, 8)


def test_batch_reports_output_directory_creation_error(tmp_path, monkeypatch):
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 8)).save(source)
    before = source.read_bytes()
    directory = tmp_path / 'denied' / 'output'
    def deny(*args, **kwargs):
        raise PermissionError('directory access denied')
    monkeypatch.setattr(converter.os, 'makedirs', deny)
    events = []
    result = converter.convert_batch([str(source)], str(directory), '.jpg', lambda: False,
                                     events.append, lambda n: None)
    _assert_batch(result, not_started=1)
    assert 'directory access denied' in result['scheduler_error']
    assert result['items'][0]['stop_reason'] == 'scheduler_error'
    assert source.read_bytes() == before and not directory.exists()


def test_batch_admits_one_job_above_comfortable_budget_only_up_to_ceiling(tmp_path, monkeypatch):
    from types import SimpleNamespace
    plan = SimpleNamespace(workers=1, native_threads=1, max_in_flight=1,
                           ram_budget=100, describe=lambda: 'test')
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k: plan)
    monkeypatch.setattr(converter.resource_policy, 'ram_ceiling', lambda *a: 150)
    monkeypatch.setattr(converter, '_estimate_source', lambda *a: (1, 125, 1))
    calls = []
    monkeypatch.setattr(converter, '_convert_source',
                        lambda index, source, *a: (calls.append(source) or _successful_record(index, source)))
    result = converter.convert_batch(['one'], str(tmp_path), '.jpg', lambda: False,
                                     lambda *a: None, lambda *a: None)
    _assert_batch(result, success=1)
    assert calls == ['one']
    monkeypatch.setattr(converter.resource_policy, 'ram_ceiling', lambda *a: 124)
    events = []
    result = converter.convert_batch(['two'], str(tmp_path), '.jpg', lambda: False,
                                     events.append, lambda *a: None)
    _assert_batch(result, failed=1)
    assert calls == ['one']
    assert converter.trans.t('converter_memory_error') in str(result['items'][0]['errors'])


def test_raw_jpeg_estimate_uses_sensor_bounds_with_headroom(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / 'sensor.raf'
    source.write_bytes(b'raw')
    class Header:
        sizes = SimpleNamespace(raw_width=1000, raw_height=800)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def open_file(self, path):
            pass
    monkeypatch.setattr(converter.image_loader.rawpy, 'RawPy', Header)
    jpeg_ram, _ = converter.estimate_job(source, ext='.jpg')
    png_ram, _ = converter.estimate_job(source, ext='.png')
    assert jpeg_ram == 1000 * 800 * 32 + 6
    assert png_ram == 1000 * 800 * 64 + 6


def test_cancelled_batch_does_not_create_output_directory(tmp_path):
    directory = tmp_path / 'not-created'
    result = converter.convert_batch(['source'], str(directory), '.jpg', lambda: True,
                                     lambda *a: None, lambda *a: None)
    _assert_batch(result, not_started=1)
    assert result['user_cancelled']
    assert not directory.exists()


def test_batch_preserves_existing_output_and_same_directory_source(tmp_path):
    source = tmp_path / 'photo.png'
    Image.new('RGB', (8, 8), 'blue').save(source)
    original = source.read_bytes()
    existing = tmp_path / 'photo.jpg'
    existing.write_bytes(b'existing output')
    result = converter.convert_batch([str(source)], str(tmp_path), '.jpg', lambda: False,
                                     lambda *a: None, lambda *a: None)
    _assert_batch(result, success=1)
    assert source.read_bytes() == original
    assert existing.read_bytes() == b'existing output'
    with Image.open(tmp_path / 'photo_1.jpg') as output:
        assert output.size == (8, 8)


def test_batch_reports_unknown_memory_instead_of_claiming_shortage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    plan = SimpleNamespace(workers=1, native_threads=1, max_in_flight=1,
                           ram_budget=100, describe=lambda: 'test')
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k: plan)
    monkeypatch.setattr(converter.resource_policy, 'probe_resources', lambda: SimpleNamespace(ram_known=False))
    monkeypatch.setattr(converter.resource_policy, 'ram_ceiling', lambda *a: 0)
    monkeypatch.setattr(converter, '_estimate_source', lambda *a: (1, 10, 1))
    events = []
    result = converter.convert_batch(['source'], str(tmp_path), '.jpg', lambda: False,
                                     events.append, lambda *a: None)
    _assert_batch(result, failed=1)
    assert converter.trans.t('converter_memory_unknown') in str(result['items'][0]['errors'])


def test_single_page_admission_uses_localized_resource_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(converter, 'estimate_job', lambda *a: (100, 100))
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 0)
    with pytest.raises(MemoryError) as error:
        converter._admit('source', 0, str(tmp_path), '.jpg')
    assert str(error.value) == converter.trans.t('converter_memory_unknown')
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 100)
    with pytest.raises(MemoryError) as error:
        converter._admit('source', 0, str(tmp_path), '.jpg')
    assert str(error.value) == converter.trans.t('converter_memory_error')
    monkeypatch.setattr(converter, 'available_memory_bytes', lambda: 1000)
    monkeypatch.setattr(converter.shutil, 'disk_usage', lambda *a: SimpleNamespace(free=99))
    with pytest.raises(OSError) as error:
        converter._admit('source', 0, str(tmp_path), '.jpg')
    assert str(error.value) == converter.trans.t('converter_space_error')


def test_scheduler_failure_preserves_committed_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    sources = [tmp_path / 'first.png', tmp_path / 'second.png']
    for source in sources:
        Image.new('RGB', (8, 8), 'red').save(source)
    output_dir = tmp_path / 'output'
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=1, native_threads=1, max_in_flight=1,
                                        ram_budget=1024**3, describe=lambda: 'test'))
    disk_usage = converter.shutil.disk_usage
    calls = []

    def fail_second_probe(path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError('output volume disconnected')
        return disk_usage(path)

    monkeypatch.setattr(converter.shutil, 'disk_usage', fail_second_probe)
    events, progress = [], []
    result = converter.convert_batch([str(p) for p in sources], str(output_dir), '.jpg',
                                     lambda: False, events.append, progress.append)
    _assert_batch(result, success=1, not_started=1)
    assert not result['user_cancelled']
    assert 'output volume disconnected' in result['scheduler_error']
    assert result['items'][1]['stop_reason'] == 'scheduler_error'
    assert len(result['items'][0]['outputs']) == 1
    with Image.open(result['items'][0]['outputs'][0]) as output:
        assert output.size == (8, 8)
    assert progress[-1] == 2


def test_last_page_committed_before_cancel_is_success(tmp_path, monkeypatch):
    from threading import Event

    source = tmp_path / 'single.png'
    Image.new('RGB', (8, 8), 'blue').save(source)
    cancelled = Event()
    real_convert = converter.convert_one

    def convert(*args, **kwargs):
        output = real_convert(*args, **kwargs)
        cancelled.set()
        return output

    monkeypatch.setattr(converter, 'convert_one', convert)
    result = converter.convert_batch([str(source)], str(tmp_path), '.jpg',
                                     cancelled.is_set, lambda event: None, lambda n: None)
    _assert_batch(result, success=1)
    assert result['user_cancelled']
    assert Path(result['items'][0]['outputs'][0]).is_file()


@pytest.mark.parametrize('kind', ['many_samples', 'float', 'planar'])
def test_tiff_unsupported_layout_is_rejected_before_decode(tmp_path, monkeypatch, kind):
    source = tmp_path / 'unsupported.tif'
    if kind == 'many_samples':
        tifffile.imwrite(source, np.zeros((8, 8, 128), np.uint16), photometric='minisblack',
                         planarconfig='contig', compression='deflate')
    elif kind == 'float':
        tifffile.imwrite(source, np.zeros((8, 8), np.float32))
    else:
        tifffile.imwrite(source, np.zeros((3, 8, 8), np.uint16), photometric='rgb',
                         planarconfig='separate')
    original = source.read_bytes()
    monkeypatch.setattr(tifffile.TiffPage, 'asarray',
                        lambda *a, **k: pytest.fail('TIFF decoded before layout validation'))
    monkeypatch.setattr(converter.cv2, 'imdecodemulti',
                        lambda *a, **k: pytest.fail('TIFF decoded before layout validation'))
    with pytest.raises((ValueError, converter.image_loader.UnsupportedColorSpace)):
        converter.convert_one(str(source), str(tmp_path), '.png')
    assert source.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ['unsupported.tif']


def test_tiff_thumbnail_admission_precedes_pixel_decode(tmp_path, monkeypatch):
    source = tmp_path / 'preview.tif'
    tifffile.imwrite(source, np.zeros((8, 8, 3), np.uint8), photometric='rgb')
    monkeypatch.setattr(converter.resource_policy, 'require_memory',
                        lambda *a, **k: (_ for _ in ()).throw(MemoryError('budget')))
    monkeypatch.setattr(tifffile.TiffPage, 'asarray',
                        lambda *a, **k: pytest.fail('TIFF decoded before admission'))
    with pytest.raises(MemoryError, match='budget'):
        converter.thumbnail_rgb(str(source))


def test_page_progress_events_are_incremental_and_keep_all_outputs(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(converter, 'convert_one',
                        lambda source, directory, ext, page, *a, **k:
                        str(tmp_path / ('长路径' * 10 + f'_{page:04}.png')))

    def run(pages):
        events = []
        record = converter._convert_source(0, 'input.png', str(tmp_path), '.png', pages,
                                           lambda: False, events.append)
        payload_size = sum(len(json.dumps(event, ensure_ascii=False)) for event in events)
        return record, events, payload_size

    small, small_events, small_size = run(100)
    large, large_events, large_size = run(200)
    assert small['status'] == large['status'] == 'success'
    assert len(small['outputs']) == len(set(small['outputs'])) == 100
    assert len(large['outputs']) == len(set(large['outputs'])) == 200
    assert len(small_events) <= 201 and len(large_events) <= 401
    assert large_size < small_size * 2.5
    assert all(set(event) == {'index', 'page', 'pages', 'completed', 'failed',
                              'output', 'error', 'status'} for event in large_events)
    assert sum(event['output'] is not None for event in large_events) == 200
    assert large['outputs'][0] not in json.dumps(large_events[-1], ensure_ascii=False)


def test_scheduler_failure_drains_active_items_and_preserves_partial_output(tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    started = [Event(), Event()]
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=2, native_threads=1, max_in_flight=3,
                                        ram_budget=1024**3, describe=lambda: 'test'))
    monkeypatch.setattr(converter, '_estimate_source', lambda *a: (2, 1, 1))
    disk_usage = converter.shutil.disk_usage
    calls = []

    def fail_third_probe(path):
        calls.append(path)
        if len(calls) == 3:
            assert all(event.wait(5) for event in started)
            raise OSError('volume disconnected')
        return disk_usage(path)

    def active(index, source, directory, ext, count, cancelled, item):
        outputs = []
        if index == 0:
            output = str(tmp_path / 'committed.png')
            Image.new('RGB', (8, 8), 'red').save(output)
            outputs.append(output)
        started[index].set()
        while not cancelled():
            Event().wait(.01)
        return dict(index=index, source=source, status='partial' if outputs else 'cancelled',
                    outputs=outputs, errors=[], stop_reason='user_cancel')

    monkeypatch.setattr(converter.shutil, 'disk_usage', fail_third_probe)
    monkeypatch.setattr(converter, '_convert_source', active)
    progress = []
    result = converter.convert_batch(['first', 'second', 'third'], str(tmp_path), '.png',
                                     lambda: False, lambda event: None, progress.append)
    _assert_batch(result, partial=1, failed=1, not_started=1)
    assert not result['user_cancelled']
    assert 'volume disconnected' in result['scheduler_error']
    assert all(item['stop_reason'] == 'scheduler_error' for item in result['items'])
    with Image.open(result['items'][0]['outputs'][0]) as output:
        assert output.size == (8, 8)
    assert progress[-1] == 3


def test_cmyk_jpeg_without_profile_matches_existing_color_conversion(tmp_path):
    source = tmp_path / '颜色.jpg'
    Image.new('CMYK', (8, 6), (220, 40, 10, 20)).save(source)
    with Image.open(source) as image:
        expected = converter.image_loader.cmyk_to_bgr(image, str(source), b'')
    destination = converter.convert_one(str(source), str(tmp_path), '.png')
    actual = cv2.imdecode(np.fromfile(destination, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('pages', [4, 8])
def test_tiff_conversion_visits_source_pages_linearly(tmp_path, monkeypatch, pages):
    source = tmp_path / 'pages.tif'
    with tifffile.TiffWriter(source) as writer:
        for value in range(pages):
            writer.write(np.full((8, 8), value * 100, np.uint16))
    visits = []
    real_pages = converter.image_loader.tiff_source_pages

    def counted(tiff):
        for entry in real_pages(tiff):
            visits.append(entry[0])
            yield entry

    monkeypatch.setattr(converter.image_loader, 'tiff_source_pages', counted)
    result = converter.convert_batch([str(source)], str(tmp_path), '.png', lambda: False,
                                     lambda event: None, lambda n: None)
    _assert_batch(result, success=1)
    assert len(visits) <= 3 * pages
    assert len(result['items'][0]['outputs']) == pages
    for index, output in enumerate(result['items'][0]['outputs']):
        actual = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, np.full((8, 8), index * 100, np.uint16))


@pytest.mark.parametrize('extension', ['.png', '.tif'])
def test_profiled_cmyk_jpeg_conversion_preserves_color_and_orientation(tmp_path, extension, cmyk_profile):
    from io import BytesIO
    from PIL import ImageCms
    from utils.image_utils import srgb_icc_profile

    profile = cmyk_profile
    source = tmp_path / '颜色方向.jpg'
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(_cmyk_pixels(), 'CMYK').save(source, quality=100,
                                                  icc_profile=profile, exif=exif)
    with Image.open(source) as image:
        transformed = ImageCms.profileToProfile(
            image, ImageCms.ImageCmsProfile(BytesIO(profile)),
            ImageCms.ImageCmsProfile(BytesIO(srgb_icc_profile())), outputMode='RGB')
        expected = np.rot90(np.asarray(transformed), -1)
        assert not np.array_equal(np.asarray(transformed), np.asarray(image.convert('RGB')))
    destination = converter.convert_one(str(source), str(tmp_path), extension)
    with Image.open(destination) as output:
        np.testing.assert_array_equal(np.asarray(output), expected)
        assert output.info['icc_profile'] == srgb_icc_profile()
        assert output.getexif().get(274, 1) == 1
    thumbnail = converter.thumbnail_rgb(str(source))
    np.testing.assert_array_equal(thumbnail, expected)


@pytest.mark.parametrize('profile_kind', ['invalid', 'rgb'])
def test_cmyk_jpeg_bad_profile_does_not_publish_output(tmp_path, profile_kind):
    from utils.image_utils import srgb_icc_profile

    source = tmp_path / 'bad.jpg'
    profile = b'invalid ICC' if profile_kind == 'invalid' else srgb_icc_profile()
    Image.new('CMYK', (8, 8), (200, 30, 40, 0)).save(source, icc_profile=profile)
    original = source.read_bytes()
    existing = tmp_path / 'bad.png'
    existing.write_bytes(b'keep')
    with pytest.raises(converter.image_loader.UnsupportedColorSpace):
        converter.convert_one(str(source), str(tmp_path), '.png')
    assert source.read_bytes() == original and existing.read_bytes() == b'keep'
    assert {path.name for path in tmp_path.iterdir()} == {'bad.jpg', 'bad.png'}


def test_batch_tiff_reduced_page_preserves_physical_page_metadata(tmp_path):
    from utils.image_utils import srgb_icc_profile

    source = tmp_path / 'metadata.tif'
    first_profile = srgb_icc_profile()
    second_profile = bytearray(first_profile)
    second_profile[-1] ^= 1
    second_profile = bytes(second_profile)
    first = np.arange(4 * 6 * 3, dtype=np.uint16).reshape(4, 6, 3) * 100
    second = first + 1000
    with tifffile.TiffWriter(source) as writer:
        for pixels, reduced, orientation, camera, profile in (
                (first, False, 1, 'Camera0', first_profile),
                (np.zeros((2, 3, 3), np.uint16), True, 1, 'Preview', first_profile),
                (second, False, 6, 'Camera2', second_profile)):
            tags = [(274, 'H', 1, orientation, False), (271, 's', 0, camera, False),
                    (34675, 'B', len(profile), profile, False)]
            writer.write(pixels, photometric='rgb', subfiletype=int(reduced), extratags=tags)
    result = converter.convert_batch([str(source)], str(tmp_path / 'out'), '.tif',
                                     lambda: False, lambda event: None, lambda n: None)
    _assert_batch(result, success=1)
    outputs = result['items'][0]['outputs']
    assert len(outputs) == 2
    for path, expected, profile, camera in zip(outputs, (first, np.rot90(second, -1)),
                                               (first_profile, second_profile), ('Camera0', 'Camera2')):
        with tifffile.TiffFile(path) as output:
            page = output.pages[0]
            np.testing.assert_array_equal(page.asarray(), expected)
            assert page.dtype == np.uint16
            assert page.tags[34675].value == profile
            assert page.tags[271].value == camera
            assert page.tags.get(274) is None or page.tags[274].value == 1


def test_queued_progress_survives_cancel_and_exposes_complete_details(qapp, tmp_path, monkeypatch):
    from threading import Event
    from types import SimpleNamespace
    from dialogs.batch_converter import BatchConverterDialog, ThumbnailWorker, QMessageBox

    monkeypatch.setattr(ThumbnailWorker, 'start', lambda self: None)
    monkeypatch.setattr(converter.resource_policy, 'plan_execution', lambda *a, **k:
                        SimpleNamespace(workers=1, native_threads=1, max_in_flight=1,
                                        ram_budget=1024**3, describe=lambda: 'test'))
    monkeypatch.setattr(converter, '_estimate_source', lambda *a: (100, 1, 1))
    reached, release = Event(), Event()
    paths = []

    def convert(source, directory, ext, page, *args, **kwargs):
        path = str(tmp_path / f'output_{page:04}.png')
        Image.new('RGB', (2, 2), (page, 0, 0)).save(path)
        paths.append(path)
        if page == 9:
            reached.set()
            assert release.wait(5)
        return path

    monkeypatch.setattr(converter, 'convert_one', convert)
    source = tmp_path / 'input.png'
    Image.new('RGB', (2, 2)).save(source)
    dialog = BatchConverterDialog()
    worker = None
    events, details = [], []
    original_item = dialog._item

    def consume(event):
        events.append(event)
        original_item(event)

    dialog._item = consume
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: details.append(args[2]))
    try:
        dialog._append([str(source)])
        dialog.directory.setText(str(tmp_path))
        dialog._start()
        worker = dialog.worker
        # Do not pump the GUI queue while the worker produces its page events.
        assert reached.wait(5)
        assert not events
        dialog._cancel()
        release.set()
        assert worker.wait(5000)
        qapp.processEvents()
        result = dialog._conversion_results
        _assert_batch(result, partial=1)
        assert result['user_cancelled']
        assert result['items'][0]['outputs'] == paths and len(paths) == 10
        assert [event['output'] for event in events if event['output']] == paths
        assert all(Path(path).is_file() for path in paths)
        row = dialog.list.item(0)
        assert all(path not in row.text() and path not in row.toolTip() for path in paths)
        dialog._show_result_details(row)
        assert len(details) == 1
        assert all(details[0].count(path) == 1 for path in paths)
    finally:
        release.set()
        if worker is not None:
            worker.requestInterruption()
            worker.wait(5000)
        dialog.close()
