from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from core import batch_converter as converter
from utils.image_utils import srgb_icc_profile


def batch(sources, directory, ext, **options):
    events, progress = [], []
    result = converter.convert_batch([str(path) for path in sources], directory, ext,
                                     lambda: False, events.append, progress.append, **options)
    assert progress == list(range(1, len(sources) + 1))
    return result, events


@pytest.mark.parametrize('mode,value,relative', [
    ('sibling', None, '花_converted'), ('sibling', '_完成', '花_完成'),
    ('subfolder', '', '花/OpenFocus_Output'), ('subfolder', '输出', '花/输出'),
    ('same', 'ignored', '花'),
])
def test_preview_resolves_per_source_without_creating_folders(tmp_path, mode, value, relative):
    source = tmp_path / '花' / '图片.jpg'
    destination = converter.output_directory(source, mode, value)
    assert destination == str(tmp_path / relative)
    assert not Path(destination).exists()


@pytest.mark.parametrize('mode', ['sibling', 'subfolder'])
@pytest.mark.parametrize('name', ['..', '.', '../逃逸', r'..\逃逸', 'A/B', r'A\B', 'C:照片',
                                  '照片:stream', 'NUL.png', 'CON', 'LPT3.txt', '尾点.',
                                  '尾空 ', ' ', '问号?', 'a\x00b', 'PRN .txt', 'COM¹.log', '花' * 256])
def test_relative_output_requires_portable_single_component(tmp_path, mode, name):
    with pytest.raises(ValueError):
        converter.output_directory(tmp_path / 'input' / '花.tif', mode, name)


def test_custom_accepts_new_unicode_parents_and_rejects_files_and_invalid_components(tmp_path):
    target = tmp_path / '新' / '图像'
    assert converter.output_directory(None, 'custom', target) == str(target)
    file = tmp_path / '照片.jpg'
    file.write_bytes(b'original')
    for invalid in ('', ' ', file, file / '输出', tmp_path / 'AUX' / '输出'):
        with pytest.raises(ValueError):
            converter.output_directory(None, 'custom', invalid)
    assert list(tmp_path.iterdir()) == [file]


@pytest.mark.parametrize('source,target', [('jpeg', 'jpg'), ('JPE', 'jpg'), ('JfIf', 'jpeg'),
                                           ('TIFF', 'tif'), ('tif', 'tiff'), ('PnG', 'png')])
def test_same_format_aliases(tmp_path, source, target):
    assert converter.same_output_format(tmp_path / f'照片.{source}', '.' + target)


def test_skipped_alias_and_companion_preserve_originals(tmp_path, monkeypatch):
    folder = tmp_path / '原片'
    folder.mkdir()
    jpeg, png = (folder / name for name in ('花.JFIF', '花.png'))
    Image.new('RGB', (12, 6), 'red').save(jpeg, format='JPEG')
    Image.new('RGB', (12, 6), 'blue').save(png)
    existing = folder / '花.jpg'
    existing.write_bytes(b'existing output')
    before = {p: p.read_bytes() for p in (jpeg, png, existing)}
    decode = converter._decode
    def check_decode(path, *args):
        assert Path(path) == png
        return decode(path, *args)
    monkeypatch.setattr(converter, '_decode', check_decode)
    result, events = batch([jpeg, png], '', '.jpg', output_mode='same',
                           max_edge=2048, skip_same_format=True)
    assert result['counts']['skipped'] == result['counts']['success'] == 1
    assert [r['status'] for r in result['items']] == ['skipped', 'success']
    assert result['items'][0]['outputs'] == result['items'][0]['errors'] == []
    assert [e['status'] for e in events if e['index'] == 0] == ['skipped']
    out = result['items'][1]['outputs']
    assert out == [str(folder / '花_1.jpg')]
    with Image.open(out[0]) as image:
        assert image.size == (12, 6) and image.getpixel((0, 0))[2] > 240
    assert {p: p.read_bytes() for p in before} == before


def test_skip_checks_effective_pixels_before_allocating_output(tmp_path, monkeypatch):
    tiny = tmp_path / '微小.jpg'
    Image.new('L', (1, 1), 88).save(tiny)
    monkeypatch.setattr(converter, '_estimate_source', lambda *a: pytest.fail('admitted unchanged source'))
    dest = tmp_path / '输出'
    result, _ = batch([tiny], dest, '.jpg', scale=.1, skip_same_format=True)
    assert result['items'][0]['status'] == 'skipped'
    assert not dest.exists()


def test_mislabeled_or_missing_same_suffix_is_not_silently_skipped(tmp_path):
    mislabeled = tmp_path / '伪装.jpg'
    Image.new('RGB', (3, 4), 'red').save(mislabeled, format='PNG')
    result, _ = batch([mislabeled, tmp_path / '失踪.jpg'], tmp_path / '输出', '.jpg', skip_same_format=True)
    assert [r['status'] for r in result['items']] == ['success', 'failed']
    with Image.open(result['items'][0]['outputs'][0]) as output:
        assert output.format == 'JPEG' and output.size == (3, 4)


@pytest.mark.parametrize('fallback', [False, True])
def test_oriented_jpeg_downsamples_once_and_preserves_metadata(tmp_path, monkeypatch, fallback):
    source = tmp_path / '纵向.JPEG'
    exif = Image.Exif()
    exif[274], exif[271] = 6, 'Camera'
    profile = srgb_icc_profile()
    pixels = np.zeros((4, 12, 3), np.uint8)
    pixels[:, :6, 0] = 255
    pixels[:, 6:, 2] = 255
    Image.fromarray(pixels).save(source, exif=exif, icc_profile=profile, quality=100, subsampling=0)
    before = source.read_bytes()
    if fallback:
        monkeypatch.setattr(cv2, 'imdecode', lambda *a: None)
    calls = []
    decode = converter._decode
    def counted(*args):
        calls.append(args[0])
        return decode(*args)
    monkeypatch.setattr(converter, '_decode', counted)
    result, _ = batch([source], '_缩小', '.jpg', output_mode='subfolder', max_edge=6, skip_same_format=True)
    assert result['counts']['success'] == 1 and calls == [str(source)]
    with Image.open(result['items'][0]['outputs'][0]) as output:
        assert output.size == (2, 6)
        assert output.getpixel((0, 0))[0] > 240 and output.getpixel((0, 5))[2] > 240
        assert output.getexif()[271] == 'Camera' and output.getexif()[274] == 1
        assert output.info['icc_profile'] == profile
    assert source.read_bytes() == before


@pytest.mark.parametrize('fallback', [False, True])
def test_downsize_preserves_16bit_straight_alpha_without_hidden_color_bleed(tmp_path, monkeypatch, fallback):
    source = tmp_path / '透明.png'
    rgba = np.array([[[53001, 1001, 4023, 65535], [0, 65535, 0, 0]]], dtype=np.uint16)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    assert ok
    encoded.tofile(source)
    if fallback:
        monkeypatch.setattr(cv2, 'imdecode', lambda *a: None)
    result, _ = batch([source], tmp_path / '输出', '.tif', scale=.5)
    assert result['items'][0]['status'] == 'success'
    with tifffile.TiffFile(result['items'][0]['outputs'][0]) as tif:
        page = tif.pages[0]
        assert page.dtype == np.uint16 and page.extrasamples[0].name == 'UNASSALPHA'
        np.testing.assert_array_equal(page.asarray(), [[[53001, 1001, 4023, 32768]]])


def test_same_tiff_considers_every_nonpreview_page_and_keeps_small_pages(tmp_path):
    source = tmp_path / '页面.tiff'
    with tifffile.TiffWriter(source) as tif:
        tif.write(np.full((16, 20), 500, np.uint16), subfiletype=1)
        tif.write(np.array([[1009, 5003]], np.uint16))
        tif.write(np.arange(32, dtype=np.uint16).reshape(4, 8) * 16,
                  extratags=[(274, 'H', 1, 6, False)])
    dest = tmp_path / '输出'
    result, _ = batch([source], dest, '.tif', max_edge=10, skip_same_format=True)
    assert result['items'][0]['status'] == 'skipped' and not dest.exists()
    result, _ = batch([source], dest, '.tif', max_edge=4, skip_same_format=True)
    first, second = map(Path, result['items'][0]['outputs'])
    assert first.name == '页面_page_0001.tif' and second.name == '页面_page_0002.tif'
    np.testing.assert_array_equal(tifffile.imread(first), [[1009, 5003]])
    expected = np.rot90(np.array([[72, 104, 136, 168], [328, 360, 392, 424]], dtype=np.uint16), -1)
    np.testing.assert_array_equal(tifffile.imread(second), expected)


def test_disk_preflight_uses_each_sources_destination(tmp_path, monkeypatch):
    sources = [tmp_path / name / '图.png' for name in ('一', '二')]
    for src in sources:
        src.parent.mkdir()
        Image.new('RGB', (4, 2), 'red').save(src)
    destinations = [tmp_path / (src.parent.name + '_converted') for src in sources]
    probes = []
    def disk_usage(path):
        probes.append(Path(path))
        return SimpleNamespace(free=0 if Path(path) == destinations[0] else 10**12)
    monkeypatch.setattr(converter.shutil, 'disk_usage', disk_usage)
    result, _ = batch(sources, None, '.jpg', output_mode='sibling')
    assert [r['status'] for r in result['items']] == ['failed', 'success']
    assert all(dest in probes for dest in destinations)
    assert result['items'][1]['outputs'] == [str(destinations[1] / '图.jpg')]


@pytest.mark.parametrize('options', [{'scale': 0}, {'scale': 1.01}, {'scale': float('nan')},
                                     {'max_edge': 0}, {'max_edge': 2.5}, {'max_edge': True}])
def test_invalid_size_fails_before_decode_or_output(tmp_path, options):
    with pytest.raises(ValueError):
        converter.convert_one(tmp_path / '不存在.jpg', tmp_path, '.png', **options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('option', [{'max_edge': 2048}, {'scale': .8}])
def test_dng_refuses_raster_resizing_before_opening_raw(tmp_path, option):
    source = tmp_path / '相机.raf'
    with pytest.raises(ValueError, match=converter.trans.t('converter_dng_original_size')):
        converter.convert_one(source, tmp_path, '.dng', **option)
    dest = tmp_path / '输出'
    result, _ = batch([source], dest, '.dng', **option)
    assert result['scheduler_error'] == converter.trans.t('converter_dng_original_size')
    assert not dest.exists()
