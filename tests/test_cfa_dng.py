"""Small sensor fixtures: an independent LibRaw consumer reads real DNG bytes."""
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy
import tifffile

from utils import linear_dng


MATRIX = ((0.8293, -0.1611, -0.1132), (-0.4759, 1.2711, 0.2275),
          (-0.1013, 0.2415, 0.5509))
XTRANS = np.array([[1, 2, 1, 1, 0, 1], [0, 1, 0, 2, 1, 2],
                   [1, 2, 1, 1, 0, 1], [1, 0, 1, 1, 2, 1],
                   [2, 1, 2, 0, 1, 0], [1, 0, 1, 1, 2, 1]], dtype=np.uint8)


class Sensor:
    def __init__(self, pattern=None, bits=14, flip=0):
        self.raw_pattern = np.array([[0, 1], [3, 2]], dtype=np.uint8) if pattern is None else pattern
        self.raw_type = rawpy.RawType.Flat
        self.num_colors = 3
        self.color_desc = b'RGBG'
        extent = 512 if self.raw_pattern.shape == (6, 6) else 128
        self.sizes = SimpleNamespace(raw_width=extent + 8, raw_height=extent + 4, width=extent, height=extent,
                                     top_margin=3, left_margin=5, pixel_aspect=1., flip=flip)
        self.rgb_xyz_matrix = np.array((*MATRIX, (0., 0., 0.)))
        self.camera_whitebalance = (2., 1., 1.5, 1.)
        self.black_level_per_channel = (64, 65, 66, 67)
        self.white_level = (1 << bits) - 1
        self.camera_white_level_per_channel = None
        self.raw_image = ((np.arange((extent + 4) * (extent + 8), dtype=np.uint32).reshape(extent + 4, extent + 8) * 17)
                          % (self.white_level - 128) + 128).astype(np.uint16)

    def raw_color(self, row, col):
        return int(self.raw_pattern[row % self.raw_pattern.shape[0], col % self.raw_pattern.shape[1]])

    @property
    def raw_image_visible(self):
        s = self.sizes
        return self.raw_image[s.top_margin:s.top_margin + s.height, s.left_margin:s.left_margin + s.width]

    def extract_thumb(self):
        raise rawpy.LibRawNoThumbnailError()

    def open_file(self, path):
        assert isinstance(path, str)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def export_sensor(tmp_path, monkeypatch, sensor):
    source = tmp_path / '传感器.nef'
    source.write_bytes(b'synthetic camera container')
    output = tmp_path / '标准.dng'
    with monkeypatch.context() as context:
        context.setattr(rawpy, 'RawPy', lambda: sensor)
        context.setattr(linear_dng, '_camera_identity', lambda *args: ('Nikon', 'Nikon Synthetic'))
        linear_dng.convert_raw_to_dng(output, source)
    return output


@pytest.mark.parametrize('bits', [12, 14, 16])
@pytest.mark.parametrize('pattern', [None, XTRANS])
def test_cfa_sensor_samples_bit_depth_and_phase(tmp_path, monkeypatch, bits, pattern):
    sensor = Sensor(pattern, bits)
    output = export_sensor(tmp_path, monkeypatch, sensor)
    with tifffile.TiffFile(output) as tif:
        page = tif.pages[0]
        assert page.photometric.value == 32803
        assert page.tags[254].value == 0
        assert page.bitspersample == 16
        assert page.tags[50717].value == sensor.white_level
        assert page.tags[50829].value == (0, 0, sensor.sizes.height, sensor.sizes.width)
        # TIFF BYTE values are bytes; np.asarray(bytes) would produce one scalar.
        values = np.frombuffer(page.tags[33422].value, dtype=np.uint8).reshape(sensor.raw_pattern.shape)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                color = sensor.raw_color(y + 3, x + 5)
                assert values[y, x] == (1 if color == 3 else color)
        np.testing.assert_array_equal(page.asarray(), sensor.raw_image_visible)
    with open(output, 'rb') as stream, rawpy.imread(stream) as decoded:
        # Real decoder, not the writer's own TIFF parser.
        np.testing.assert_array_equal(decoded.raw_image, sensor.raw_image_visible)
        assert decoded.sizes.width == sensor.sizes.width and decoded.sizes.height == sensor.sizes.height
        wb = np.asarray(decoded.camera_whitebalance)[:3]
        np.testing.assert_allclose(wb / wb[1], [2., 1., 1.5], rtol=1e-5)
        rgb = decoded.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True)
        assert rgb.dtype == np.uint16 and rgb.shape == (sensor.sizes.height, sensor.sizes.width, 3)
    metadata = linear_dng._linear_dng_calibration(output)
    np.testing.assert_allclose(metadata.as_shot_neutral, [0.5, 1., 2 / 3], rtol=1e-5)


@pytest.mark.parametrize('flip,orientation', [(0, 1), (3, 3), (5, 8), (6, 6)])
def test_cfa_orientation_is_metadata_not_a_rotated_mosaic(tmp_path, monkeypatch, flip, orientation):
    output = export_sensor(tmp_path, monkeypatch, Sensor(flip=flip))
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].tags[274].value == orientation
    with open(output, 'rb') as stream, rawpy.imread(stream) as raw:
        assert raw.sizes.flip == flip


@pytest.mark.parametrize('change,error', [
    ({'raw_pattern': np.zeros((4, 4), dtype=np.uint8)}, 'CFA layout'),
    ({'camera_whitebalance': (0., 1., 1., 1.)}, 'white balance'),
    ({'white_level': 32}, 'black/white'),
    ({'camera_white_level_per_channel': (1000, 1100, 1200, 1100)}, 'unequal'),
    ({'num_colors': 4}, 'three-color'),
])
def test_unknown_calibration_fails_without_touching_source(tmp_path, monkeypatch, change, error):
    sensor = Sensor()
    for key, value in change.items():
        setattr(sensor, key, value)
    source = tmp_path / 'camera.nef'
    source.write_bytes(b'keep input')
    with monkeypatch.context() as context:
        context.setattr(rawpy, 'RawPy', lambda: sensor)
        context.setattr(linear_dng, '_camera_identity', lambda *args: ('Nikon', 'Nikon Synthetic'))
        with pytest.raises(ValueError, match=error):
            linear_dng.convert_raw_to_dng(tmp_path / 'out.dng', source)
    assert source.read_bytes() == b'keep input'


def test_existing_dng_preserves_all_profile_bytes(tmp_path, monkeypatch):
    source = export_sensor(tmp_path, monkeypatch, Sensor())
    target = tmp_path / 'copy.dng'
    linear_dng.convert_raw_to_dng(target, source)
    assert target.read_bytes() == source.read_bytes()


def test_existing_linear_dng_remains_linear_and_libraw_readable(tmp_path):
    source = tmp_path / 'linear.dng'
    output = tmp_path / 'copy.dng'
    pixels = np.full((128, 128, 3), (7000, 11000, 14000), dtype=np.uint16)
    metadata = linear_dng.LinearDNGMetadata(MATRIX, (0.5, 1., 2 / 3), 'Nikon', 'Nikon Synthetic')
    linear_dng.write_linear_dng(source, pixels, metadata)
    linear_dng.convert_raw_to_dng(output, source)
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].photometric.value == 34892
        assert tif.pages[0].tags[254].value == 0
        np.testing.assert_array_equal(tif.pages[0].asarray(), pixels[:, :, ::-1])
    with open(output, 'rb') as stream, rawpy.imread(stream) as raw:
        assert raw.postprocess(output_bps=16).shape == (128, 128, 3)


def test_cfa_strip_cancellation_removes_unpublished_outputs(tmp_path, monkeypatch):
    from core import batch_converter
    sensor = Sensor(XTRANS)
    source = tmp_path / 'original.nef'
    source.write_bytes(b'keep source')
    monkeypatch.setattr(rawpy, 'RawPy', lambda: sensor)
    monkeypatch.setattr(linear_dng, '_camera_identity', lambda *a: ('Nikon', 'Nikon Synthetic'))
    monkeypatch.setattr(batch_converter, 'available_memory_bytes', lambda: 8 * 1024**3)
    checks = 0
    writing_sensor = False
    original_metadata = linear_dng._cfa_metadata

    def metadata(*args):
        nonlocal writing_sensor
        tags = original_metadata(*args)
        writing_sensor = True
        return tags

    monkeypatch.setattr(linear_dng, '_cfa_metadata', metadata)

    def cancelled():
        nonlocal checks
        if writing_sensor:
            checks += 1
        return checks >= 3

    with pytest.raises(InterruptedError):
        batch_converter.convert_one(str(source), str(tmp_path), '.dng', cancelled=cancelled)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['original.nef']
    assert source.read_bytes() == b'keep source'


def test_cfa_default_crop_survives_masked_border_removal(tmp_path, monkeypatch):
    sensor = Sensor()
    sensor.sizes.crop_left_margin = 9
    sensor.sizes.crop_top_margin = 5
    sensor.sizes.crop_width = 120
    sensor.sizes.crop_height = 124
    output = export_sensor(tmp_path, monkeypatch, sensor)
    with tifffile.TiffFile(output) as tif:
        assert tif.pages[0].tags[50719].value == (4, 2)
        assert tif.pages[0].tags[50720].value == (120, 124)
    with rawpy.RawPy() as raw:
        raw.open_file(str(output))
        assert raw.sizes.crop_width == 120 and raw.sizes.crop_height == 124


def test_dng_source_and_destination_cannot_alias(tmp_path, monkeypatch):
    source = export_sensor(tmp_path, monkeypatch, Sensor())
    before = source.read_bytes()
    with pytest.raises(ValueError, match='differ'):
        linear_dng.convert_raw_to_dng(source, source)
    assert source.read_bytes() == before


def test_unrecognized_six_by_six_cfa_fails_even_with_rgb_counts(tmp_path, monkeypatch):
    sensor = Sensor(np.sort(XTRANS.reshape(-1)).reshape(6, 6))
    monkeypatch.setattr(linear_dng, '_camera_identity', lambda *a: ('Nikon', 'Nikon Synthetic'))
    with pytest.raises(ValueError, match='arrangement'):
        linear_dng._cfa_metadata(sensor, tmp_path / 'sensor.nef')
