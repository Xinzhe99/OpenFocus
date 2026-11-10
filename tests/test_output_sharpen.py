"""Output sharpening is opt-in and works on the fused image at full precision."""
import cv2
import numpy as np
import pytest

from utils.output_sharpen import sharpen_output
from utils.image_utils import quantize_fuse_output


def _scene():
    image = np.full((64, 80, 3), 60, np.float32)
    image[16:48, 20:60] = (120, 160, 200)
    return cv2.GaussianBlur(image, (0, 0), 1.5)


def test_sharpening_increases_edge_contrast_without_changing_flat_areas():
    image = _scene()
    result = sharpen_output(image)
    assert result[32, 20, 1] > image[32, 20, 1]
    assert result[32, 19, 1] < image[32, 19, 1]
    np.testing.assert_allclose(result[2, 2], image[2, 2], atol=1e-4)
    np.testing.assert_array_equal(image, _scene())


def test_sharpen_preserves_alpha_and_16_bit_precision():
    color = _scene() + 0.013
    alpha = np.broadcast_to(np.arange(80, dtype=np.float32), (64, 80))
    rgba = np.dstack((color, alpha))
    result = sharpen_output(rgba)
    np.testing.assert_array_equal(result[..., 3], alpha)
    output = quantize_fuse_output(result, True)
    assert output.dtype == np.uint16
    assert np.any(output[..., :3] % 257 != 0)


def test_cancel_before_sharpening():
    from core.registration import RegistrationCancelled
    with pytest.raises(RegistrationCancelled):
        sharpen_output(_scene(), lambda: True)


def test_cli_and_render_worker_use_same_opt_in_processing(qapp):
    from core.cli import build_parser, _fuse_stack
    from core.workers import RenderWorker

    args = build_parser().parse_args(['-i', 'x', '-o', 'x.png', '-m', 'guided_filter',
                                      '--align', 'none', '--cpu', '--no-exposure-match'])
    image = (_scene() * 257).round().astype(np.uint16)
    frames = [image, image.copy()]
    plain = _fuse_stack(frames, args)
    args.sharpen_output = True
    sharp = _fuse_stack(frames, args)
    assert plain.dtype == sharp.dtype == np.uint16
    assert not np.array_equal(plain, sharp)
    results, errors = [], []
    worker = RenderWorker(frames, algorithm='guided_filter', alignment='none',
                          exposure_match=False, tile_enabled=False, use_gpu=False,
                          sharpen_output=True)
    worker.finished_signal.connect(lambda _frames, result, *_: results.append(result))
    worker.error_signal.connect(errors.append)
    worker.run()
    assert not errors
    np.testing.assert_array_equal(results[0], sharp)


def test_batch_sharpening_changes_fused_output(qapp, tmp_path):
    from core.workers import BatchWorker
    source = tmp_path / 'stack'
    source.mkdir()
    image = (_scene() * 257).round().astype(np.uint16)
    for index in range(2):
        assert cv2.imwrite(str(source / f'{index}.png'), image)
    outputs = []
    for enabled in (False, True):
        settings = {'format': 'png', 'fusion_method': 'guided_filter', 'reg_methods': [],
                    'fusion_params': {}, 'runtime_settings': {
                        'exposure_match': False, 'sharpen_output': enabled}}
        worker = BatchWorker([str(source)], 'custom', str(tmp_path / str(enabled)),
                             settings, use_gpu=False, thread_count=1)
        result = {}
        worker.finished.connect(result.update)
        worker.run()
        assert result['success'] == 1
        outputs.append(cv2.imread(result['outputs'][0], cv2.IMREAD_UNCHANGED))
    assert outputs[0].dtype == outputs[1].dtype == np.uint16
    assert not np.array_equal(*outputs)


def test_sharpening_does_not_invalidate_alignment_cache():
    from types import SimpleNamespace
    from controllers.render_manager import RenderManager
    window = SimpleNamespace(raw_images=[_scene(), _scene()], source_revision=0,
        current_fusion_method=lambda: 'guided_filter',
        current_alignment_method=lambda: 'none',
        registration_parameters_for=lambda method: {},
        fusion_parameters_for=lambda method: {},
        brightness_option_for=lambda option, method: False)
    manager = RenderManager(window)
    before = manager._alignment_key([0, 1])
    window.sharpen_output = True
    assert manager._alignment_key([0, 1]) == before
    assert manager._processing_parameters('guided_filter')['sharpen_output'] is True


@pytest.mark.parametrize('method,expected', [
    ('openfocus_a', 'first'), ('openfocus_b', 'first'),
    ('openfocus_c', 'first'), ('guided_filter', 'last'),
])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_render_parameters_choose_base_in_processing_order(method, expected, enabled, reverse):
    from types import SimpleNamespace
    from controllers.render_manager import RenderManager

    window = SimpleNamespace(
        current_alignment_method=lambda: 'none',
        registration_parameters_for=lambda alignment: {},
        fusion_parameters_for=lambda selected: {},
        brightness_option_for=lambda option, selected: enabled if option == 'base_correct' else False,
        stack_reverse_order=reverse)
    settings = RenderManager(window)._processing_parameters(method)
    assert settings['base_frame'] == (expected if enabled else 'none')
    assert settings['sort_mode'] == ('desc' if reverse else 'asc')
