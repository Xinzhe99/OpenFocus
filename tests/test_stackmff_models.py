"""StackMFF (V1) and StackMFF V2 integration: reference parity and capabilities.

The models themselves are vendored, so these tests pin the parts OpenFocus
adds: staging, colour reconstruction, output bit depth, the declared stack
order, tiling policy and the depth/CLI surfaces.
"""
import numpy as np
import cv2
import pytest


from core.algorithms import fusion_algorithms, get_fusion, registration_algorithms
from core.depth_map import compute_focus_index
from core.multi_focus_fusion import MultiFocusFusion
from fusion_methods import stackmff_v1, stackmff_v2
from utils.image_utils import normalize_fuse_input, quantize_fuse_output
from utils.weights import weights_path


def _stack(count=5, size=(64, 80)):
    """Focus sweep: each frame is sharp on its own band."""
    h, w = size
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, (h, w), dtype=np.uint8)
    blurred = cv2.GaussianBlur(texture, (31, 31), 0)
    frames = []
    for index in range(count):
        band = np.zeros((h, w), bool)
        band[index * h // count:(index + 1) * h // count] = True
        frames.append(cv2.cvtColor(
            np.where(band, texture, blurred).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return frames


def _upstream_v1(frames):
    torch = pytest.importorskip("torch")
    from core.models.stackmff_v1.network import StackMFF
    model = StackMFF()
    checkpoint = torch.load(weights_path("stackmff_final.pth"), map_location="cpu",
                            weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     for frame in frames]).astype(np.float32) / 255.0
    with torch.no_grad():
        return model(torch.from_numpy(gray)[None, None], sigma=0.5,
                     scales=(1.0, 0.75)).squeeze().numpy() * 255.0


def _upstream_v2(frames):
    torch = pytest.importorskip("torch")
    from core.models.stackmff_v2.network import StackMFF_V2
    model = StackMFF_V2()
    checkpoint = torch.load(weights_path("stackmff_v2_final.pth"), map_location="cpu",
                            weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     for frame in frames]).astype(np.float32) / 255.0
    h, w = gray.shape[1:]
    hp, wp = h + (-h) % 64, w + (-w) % 64
    mode = "reflect" if hp - h < h and wp - w < w else "edge"
    padded = np.pad(gray, ((0, 0), (0, hp - h), (0, wp - w)), mode=mode)
    with torch.no_grad():
        fused, depth, index = model(torch.from_numpy(padded.copy())[None])
    return (fused[..., :h, :w], depth[..., :h, :w], index[..., :h, :w])


def test_default_method_stays_stackmffv3():
    assert fusion_algorithms()[0].id == "stackmffv3"
    from ui.right_panel import FUSION_DEFAULT
    assert FUSION_DEFAULT == "stackmffv3"


def test_declared_capabilities():
    v1 = get_fusion("stackmff")
    assert v1.parameters == () and v1.depth is None
    assert v1.weights == "weights/stackmff_final.onnx"
    assert v1.neural and not v1.order_sensitive and not v1.sort_frames

    v2 = get_fusion("stackmffv2")
    assert v2.parameters == ()
    assert v2.depth is not None and v2.depth_planner
    assert v2.order_sensitive and not v2.sort_frames
    assert v2.weights == "weights/stackmff_v2_final.onnx"


def test_every_method_exposes_a_hint():
    from locales import trans
    for spec in (*fusion_algorithms(), *registration_algorithms()):
        assert spec.hint_key, spec.id
        text = trans.t(spec.hint_key)
        assert text and text != spec.hint_key, spec.id


def test_v1_luma_matches_the_reference_network():
    frames = _stack()
    fused = stackmff_v1.stackmff_impl(
        list(frames), None, weights_path("stackmff_final.onnx"), False,
        thread_count=1)
    assert fused.dtype == np.uint8 and fused.shape == frames[0].shape
    luma = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expected = _upstream_v1(frames)
    assert np.abs(luma - expected).max() <= 1.0
    # Gray input must stay gray through colour reconstruction.
    assert np.array_equal(fused[:, :, 0], fused[:, :, 1])
    assert np.array_equal(fused[:, :, 1], fused[:, :, 2])

@pytest.mark.parametrize("depth", [2, 3, 7])
def test_v1_supports_dynamic_stack_depth(depth):
    frames = _stack(count=depth, size=(32, 40))
    result = stackmff_v1.stackmff_impl(
        frames, None, weights_path("stackmff_final.onnx"), False, thread_count=1)
    assert result.shape == frames[0].shape
    assert result.dtype == np.uint8
    source = np.stack(frames)
    assert np.all(result.astype(np.int16) >= source.min(axis=0).astype(np.int16) - 1)
    assert np.all(result.astype(np.int16) <= source.max(axis=0).astype(np.int16) + 1)


def test_v1_cancels_between_inference_scales(monkeypatch):
    from core.registration import RegistrationCancelled
    from fusion_methods import neural_runtime
    original = neural_runtime.logits
    completed = []

    def logits(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(True)
        return result

    monkeypatch.setattr(neural_runtime, "logits", logits)
    with pytest.raises(RegistrationCancelled):
        stackmff_v1.stackmff_impl(
            _stack(count=3, size=(32, 40)), None,
            weights_path("stackmff_final.onnx"), False, thread_count=1,
            should_cancel=lambda: bool(completed))
    assert len(completed) == 1


@pytest.mark.parametrize("size", [(64, 80), (70, 83), (8, 12)])
def test_v2_hard_matches_the_reference_network_exactly(size):
    frames = _stack(size=size)
    fused, depth, index = stackmff_v2.infer(
        list(frames), weights_path("stackmff_v2_final.onnx"), False,
        thread_count=1)
    ref_fused, ref_depth, ref_index = _upstream_v2(frames)
    expected = np.clip(np.round(ref_fused.squeeze().numpy() * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(fused[:, :, 0], expected)
    assert np.array_equal(index, ref_index.squeeze().numpy())
    assert np.abs(depth - ref_depth.squeeze().numpy()).max() < 1e-4


def test_v2_depth_follows_focus_order_while_fused_image_is_invariant():
    frames = _stack()
    path = weights_path("stackmff_v2_final.onnx")
    fused, depth, index = stackmff_v2.infer(frames, path, False, thread_count=1)
    reverse_fused, reverse_depth, reverse_index = stackmff_v2.infer(
        list(reversed(frames)), path, False, thread_count=1)
    np.testing.assert_allclose(depth + reverse_depth, 1.0, atol=2e-4)
    # Exact half-layer values can round to the same even index in both orders.
    position = depth * (len(frames) - 1)
    untied = np.abs(position - np.floor(position) - 0.5) > 1e-3
    assert untied.any()
    assert np.array_equal(index[untied] + reverse_index[untied],
                          np.full(untied.sum(), len(frames) - 1))
    np.testing.assert_array_equal(fused[untied], reverse_fused[untied])


def test_both_methods_preserve_16_bit_output():
    frames = [cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257,
                           cv2.COLOR_GRAY2BGR) for frame in _stack()]
    for name in ("stackmff", "stackmffv2"):
        fusion = MultiFocusFusion(algorithm=name, use_gpu=False)
        normalized, is_16bit = normalize_fuse_input(list(frames))
        out = quantize_fuse_output(
            fusion.fuse(input_source=normalized, thread_count=1), is_16bit)
        assert out.dtype == np.uint16
        assert np.unique(out).size > 256


def test_v2_tiled_fusion_uses_global_depth_and_source_pixels():
    from fusion_methods.stackmff_v2_tiled import _Source, tiled_depth

    frames = _stack(size=(64, 80))
    path = weights_path("stackmff_v2_final.onnx")
    depth, index = tiled_depth(_Source(frames, None, None, None), path, False,
                               tile_size=32, tile_overlap=8, thread_count=1)
    fused = stackmff_v2.stackmffv2_impl(
        frames, None, path, False, tile_size=32, tile_overlap=8, thread_count=1)
    assert 0.0 <= depth.min() <= depth.max() <= 1.0
    np.testing.assert_array_equal(index, np.rint(depth * (len(frames) - 1)))
    expected = np.stack(frames)[index, np.arange(depth.shape[0])[:, None],
                                np.arange(depth.shape[1])[None, :]]
    np.testing.assert_array_equal(fused, expected)


def test_v2_depth_reports_a_sampled_layer():
    frames = _stack()
    index = compute_focus_index(frames, "stackmffv2", thread_count=1)
    assert index.shape == frames[0].shape[:2]
    assert index.min() >= 0.0 and index.max() <= 1.0
    steps = index * (len(frames) - 1)
    assert np.allclose(steps, np.round(steps), atol=1e-5)
    with pytest.raises(ValueError, match="no depth measure"):
        compute_focus_index(frames, "stackmff")


def test_cli_runs_both_and_rejects_depth_for_v1(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from utils.image_utils import imwrite_auto

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "stack"
    source.mkdir()
    frames = _stack(count=4)
    for index, frame in enumerate(frames):
        assert imwrite_auto(str(source / f"f{index}.png"), frame)
    cli_pythonpath = os.pathsep.join(
        path for path in (str(root), os.environ.get("PYTHONPATH", "")) if path
    )
    environment = dict(os.environ, OPENFOCUS_LOG_DIR=str(tmp_path / "logs"),
                       PYTHONPATH=cli_pythonpath, KMP_DUPLICATE_LIB_OK="TRUE",
                       PYTHONIOENCODING="utf-8")

    def run(args):
        return subprocess.run(
            [sys.executable, str(root / "main.py"), *args],
            cwd=str(root), env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)

    for method in ("stackmff", "stackmffv2"):
        output = tmp_path / f"{method}.png"
        result = run(["-i", str(source), "-o", str(output), "-m", method,
                      "--cpu", "--threads", "1", "--no-exposure-match"])
        assert result.returncode == 0, result.stderr
        assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED) is not None

    depth = tmp_path / "v1_depth.png"
    rejected = run(["-i", str(source), "-o", str(depth), "-m", "stackmff",
                    "--depth-map"])
    assert rejected.returncode == 2
    assert "not available" in rejected.stderr
    assert not depth.exists()

    accepted = run(["-i", str(source), "-o", str(tmp_path / "v2.png"),
                    "-m", "stackmffv2", "--depth-map", "--cpu", "--threads", "1"])
    assert accepted.returncode == 0, accepted.stderr
    assert (tmp_path / "v2_depth.png").is_file()
