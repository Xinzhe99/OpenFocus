"""Output sharpening is opt-in and works on the fused image at full precision."""
import cv2
import numpy as np
import pytest

from utils.output_sharpen import (DEFAULT_SHARPEN_STRENGTH, MAX_SHARPEN_STRENGTH,
                                  sharpen_output, validate_sharpen_strength)
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


def test_default_matches_fixed_weights_and_strength_scales_edge_contrast():
    image = _scene()
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    legacy = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    np.testing.assert_array_equal(sharpen_output(image), legacy)
    low = sharpen_output(image, strength=50)
    high = sharpen_output(image, strength=MAX_SHARPEN_STRENGTH)
    assert image[32, 20, 1] < low[32, 20, 1] < legacy[32, 20, 1] < high[32, 20, 1]


@pytest.mark.parametrize('strength', [False, True, -1, 201, 10.0, '10', None])
def test_strength_requires_an_integer_in_range(strength):
    with pytest.raises(ValueError, match='sharpen_strength'):
        validate_sharpen_strength(strength)


def test_zero_strength_preserves_pixels_and_skips_memory_and_blur(monkeypatch):
    from core import resource_policy

    def unexpected(*args, **kwargs):
        pytest.fail('unneeded sharpening work at zero strength')

    image = _scene()
    monkeypatch.setattr(resource_policy, 'require_memory', unexpected)
    monkeypatch.setattr(cv2, 'GaussianBlur', unexpected)
    assert sharpen_output(image, strength=0) is image


def test_sharpen_preserves_alpha_and_16_bit_precision():
    color = _scene() + 0.013
    alpha = np.broadcast_to(np.arange(80, dtype=np.float32), (64, 80))
    rgba = np.dstack((color, alpha))
    result = sharpen_output(rgba, strength=175)
    np.testing.assert_array_equal(result[..., 3], alpha)
    output = quantize_fuse_output(result, True)
    assert output.dtype == np.uint16
    assert np.any(output[..., :3] % 257 != 0)


def test_cancel_before_sharpening():
    from core.registration import RegistrationCancelled
    with pytest.raises(RegistrationCancelled):
        sharpen_output(_scene(), lambda: True)


def test_cancel_after_blur_and_before_weighting(monkeypatch):
    from core.registration import RegistrationCancelled

    def unexpected(*args, **kwargs):
        pytest.fail('weighted combination ran after cancellation')

    cancelled = iter((False, True))
    monkeypatch.setattr(cv2, 'addWeighted', unexpected)
    with pytest.raises(RegistrationCancelled):
        sharpen_output(_scene(), lambda: next(cancelled), strength=175)


def test_cli_strength_help_and_validation(capsys):
    from core.cli import build_parser

    parser = build_parser()
    help_text = parser.format_help()
    assert '--sharpen-strength PERCENT' in help_text
    assert f'0–{MAX_SHARPEN_STRENGTH} percent' in help_text
    args = ['-i', 'x', '-o', 'x.png']
    assert parser.parse_args(args).sharpen_strength == DEFAULT_SHARPEN_STRENGTH
    for invalid in ('-1', '201', '12.0'):
        with pytest.raises(SystemExit) as error:
            parser.parse_args([*args, '--sharpen-strength', invalid])
        assert error.value.code == 2
        assert '--sharpen-strength' in capsys.readouterr().err


def test_cli_render_and_compare_use_same_opt_in_strength(qapp):
    from core.cli import build_parser, _fuse_stack
    from core.compare_worker import PreviewWorker
    from core.workers import RenderWorker

    parser = build_parser()
    argv = ['-i', 'x', '-o', 'x.png', '-m', 'guided_filter', '--align', 'none',
            '--cpu', '--no-exposure-match', '--threads', '1', '--no-tile-enabled']
    image = (_scene() * 257).round().astype(np.uint16)
    frames = [image, image.copy()]
    plain = _fuse_stack(frames, parser.parse_args(argv))
    args = parser.parse_args([*argv, '--sharpen-strength', '175'])
    assert args.sharpen_output is False
    np.testing.assert_array_equal(_fuse_stack(frames, args), plain)
    args.sharpen_output = True
    sharp = _fuse_stack(frames, args)
    args.sharpen_strength = 0
    np.testing.assert_array_equal(_fuse_stack(frames, args), plain)
    assert plain.dtype == sharp.dtype == np.uint16
    assert not np.array_equal(plain, sharp)
    results, errors = [], []
    worker = RenderWorker(frames, algorithm='guided_filter', alignment='none',
                          exposure_match=False, tile_enabled=False, use_gpu=False,
                          thread_count=1, sharpen_output=True, sharpen_strength=175)
    worker.finished_signal.connect(lambda _frames, result, *_: results.append(result))
    worker.error_signal.connect(errors.append)
    worker.run()
    assert not errors
    np.testing.assert_array_equal(results[0], sharp)

    comparison = PreviewWorker(frames, ('guided_filter',), {},
        {'tile_enabled': False, 'thread_count': 1, 'sharpen_output': True, 'sharpen_strength': 175},
        None, roi_only=False)
    results.clear()
    comparison.result_ready.connect(lambda _method, result, _bounds: results.append(result))
    comparison.method_failed.connect(lambda _method, error: errors.append(error))
    comparison.run()
    assert not errors
    np.testing.assert_array_equal(results[0], sharp)


@pytest.mark.parametrize('group_size', [0, 2])
def test_batch_sharpening_changes_only_final_fused_output(qapp, tmp_path, monkeypatch, group_size):
    from core.workers import BatchWorker
    from utils import output_sharpen

    strengths = []

    def record_strength(image, should_cancel=None, *, strength=DEFAULT_SHARPEN_STRENGTH):
        strengths.append(strength)
        return sharpen_output(image, should_cancel, strength=strength)

    monkeypatch.setattr(output_sharpen, 'sharpen_output', record_strength)
    source = tmp_path / 'stack'
    source.mkdir()
    image = (_scene() * 257).round().astype(np.uint16)
    for index in range(4):
        assert cv2.imwrite(str(source / f'{index}.png'), image)
    outputs = []
    for enabled, strength in ((False, 175), (True, 0), (True, 175)):
        settings = {'format': 'png', 'fusion_method': 'guided_filter', 'reg_methods': [],
                    'fusion_params': {}, 'group_size': group_size, 'runtime_settings': {
                        'exposure_match': False, 'sharpen_output': enabled, 'sharpen_strength': strength}}
        worker = BatchWorker([str(source)], 'custom', str(tmp_path / f'{enabled}-{strength}'),
                             settings, use_gpu=False, thread_count=1)
        result = {}
        worker.finished.connect(result.update)
        worker.run()
        assert result['success'] == 1
        outputs.append(cv2.imread(result['outputs'][0], cv2.IMREAD_UNCHANGED))
    assert all(output.dtype == np.uint16 for output in outputs)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0], outputs[2])
    assert strengths == ([175] if group_size else [0, 175])


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
