"""StackMFF (V1) and StackMFF V2 integration: reference parity and capabilities.

The models themselves are vendored, so these tests pin the parts OpenFocus
adds: staging, colour reconstruction, output bit depth, the declared stack
order, tiling policy and the depth/CLI surfaces.
"""
import numpy as np
import cv2
import pytest

torch = pytest.importorskip("torch")

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
    from core.models.stackmff_v1.network import StackMFF
    model = StackMFF()
    model.load_state_dict({name[7:] if name.startswith("module.") else name: value
                           for name, value in torch.load(
                               weights_path("stackmff_final.pth"),
                               map_location="cpu").items()})
    model.eval()
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     for frame in frames]).astype(np.float32) / 255.0
    with torch.no_grad():
        return model(torch.from_numpy(gray)[None, None]).squeeze().numpy() * 255.0


def _upstream_v2(frames):
    from core.models.stackmff_v2.network import StackMFF_V2
    model = StackMFF_V2()
    model.load_state_dict({name.replace("module.", ""): value
                           for name, value in torch.load(
                               weights_path("stackmff_v2_final.pth"),
                               map_location="cpu").items()})
    model.eval()
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     for frame in frames]).astype(np.float32) / 255.0
    h, w = gray.shape[1:]
    # The reference resizes to a multiple of 32; OpenFocus pads instead, so
    # the comparison uses the same edge padding and crops the result.
    hp, wp = h + (-h) % 32, w + (-w) % 32
    padded = np.pad(gray, ((0, 0), (0, hp - h), (0, wp - w)), mode="edge")
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
    assert v1.weights == "weights/stackmff_final.pth"
    assert v1.neural and not v1.order_sensitive and not v1.sort_frames

    v2 = get_fusion("stackmffv2")
    assert [parameter.key for parameter in v2.parameters] == ["mode"]
    assert v2.parameters[0].default == "hard"
    assert v2.depth is not None and v2.depth_planner
    assert v2.order_sensitive and not v2.sort_frames
    assert v2.weights == "weights/stackmff_v2_final.pth"


def test_every_method_exposes_a_hint():
    from locales import trans
    for spec in (*fusion_algorithms(), *registration_algorithms()):
        assert spec.hint_key, spec.id
        text = trans.t(spec.hint_key)
        assert text and text != spec.hint_key, spec.id


def test_v1_luma_matches_the_reference_network():
    frames = _stack()
    fused = stackmff_v1.stackmff_impl(
        list(frames), None, weights_path("stackmff_final.pth"), False,
        thread_count=1)
    assert fused.dtype == np.uint8 and fused.shape == frames[0].shape
    luma = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY).astype(np.float32)
    expected = _upstream_v1(frames)
    assert np.abs(luma - expected).max() <= 1.0
    # Gray input must stay gray through colour reconstruction.
    assert np.array_equal(fused[:, :, 0], fused[:, :, 1])
    assert np.array_equal(fused[:, :, 1], fused[:, :, 2])


def test_v2_hard_matches_the_reference_network_exactly():
    frames = _stack()
    fused, depth, index = stackmff_v2.infer(
        list(frames), weights_path("stackmff_v2_final.pth"), False,
        mode="hard", thread_count=1)
    ref_fused, ref_depth, ref_index = _upstream_v2(frames)
    expected = np.clip(np.round(ref_fused.squeeze().numpy() * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(fused[:, :, 0], expected)
    assert np.array_equal(index, ref_index.squeeze().numpy())
    assert np.abs(depth - ref_depth.squeeze().numpy()).max() < 1e-4


def test_v2_soft_is_a_convex_blend_and_differs_from_hard():
    frames = _stack()
    hard, _depth, _index = stackmff_v2.infer(
        list(frames), weights_path("stackmff_v2_final.pth"), False,
        mode="hard", thread_count=1)
    soft, _depth, _index = stackmff_v2.infer(
        list(frames), weights_path("stackmff_v2_final.pth"), False,
        mode="soft", thread_count=1)
    assert not np.array_equal(hard, soft)
    stack = np.stack(frames).astype(np.float32)
    low, high = stack.min(axis=0), stack.max(axis=0)
    body = soft[2:-2, 2:-2]
    assert (body >= low[2:-2, 2:-2] - 1).all() and (body <= high[2:-2, 2:-2] + 1).all()


def test_v2_keeps_the_focus_order_while_v1_is_insensitive():
    frames = _stack()
    reversed_frames = list(reversed(frames))

    def run(module, name, mode=None):
        params = {} if mode is None else {"mode": mode}
        out = MultiFocusFusion(algorithm=name, use_gpu=False,
                               fusion_parameters=params).fuse(
            input_source=list(frames), thread_count=1)
        back = MultiFocusFusion(algorithm=name, use_gpu=False,
                                fusion_parameters=params).fuse(
            input_source=list(reversed_frames), thread_count=1)
        return float(np.abs(out.astype(np.int16) - back.astype(np.int16)).mean())

    v1_delta = run(stackmff_v1, "stackmff")
    v2_delta = run(stackmff_v2, "stackmffv2", "hard")
    # Reversal changes V2's recurrent path materially; V1's per-frame 3D pass
    # is far less affected (the paper's order-insensitivity). Pixel equality
    # is not claimed for either model — only the declared capability and the
    # observed direction of the difference.
    assert v2_delta > 4.0
    assert v2_delta > v1_delta


def test_both_methods_preserve_16_bit_output():
    frames = [cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint16) * 257,
                           cv2.COLOR_GRAY2BGR) for frame in _stack()]
    for name, mode in (("stackmff", None), ("stackmffv2", "hard")):
        params = {} if mode is None else {"mode": mode}
        fusion = MultiFocusFusion(algorithm=name, use_gpu=False,
                                  fusion_parameters=params)
        normalized, is_16bit = normalize_fuse_input(list(frames))
        out = quantize_fuse_output(
            fusion.fuse(input_source=normalized, thread_count=1), is_16bit)
        assert out.dtype == np.uint16
        assert np.unique(out).size > 256


def test_v1_tiles_without_visible_seams_and_v2_never_tiles():
    frames = _stack()
    whole = MultiFocusFusion(algorithm="stackmff", use_gpu=False,
                             tile_enabled=False).fuse(
        input_source=list(frames), thread_count=1)
    tiled = MultiFocusFusion(algorithm="stackmff", use_gpu=False,
                             tile_enabled=True, tile_block_size=64,
                             tile_overlap=16, tile_threshold=10).fuse(
        input_source=list(frames), thread_count=1)
    delta = np.abs(whole.astype(np.int16) - tiled.astype(np.int16))
    assert delta.mean() < 2 and delta.max() <= 16

    v2_whole = MultiFocusFusion(algorithm="stackmffv2", use_gpu=False,
                                tile_enabled=False,
                                fusion_parameters={"mode": "hard"}).fuse(
        input_source=list(frames), thread_count=1)
    v2_tiled = MultiFocusFusion(algorithm="stackmffv2", use_gpu=False,
                                tile_enabled=True, tile_block_size=32,
                                tile_overlap=8, tile_threshold=10,
                                fusion_parameters={"mode": "hard"}).fuse(
        input_source=list(frames), thread_count=1)
    assert np.array_equal(v2_whole, v2_tiled)


def test_v2_depth_positions_stay_in_range_and_quantise_in_hard_mode():
    frames = _stack()
    soft = compute_focus_index(frames, "stackmffv2", parameters={"mode": "soft"},
                               thread_count=1)
    hard = compute_focus_index(frames, "stackmffv2", parameters={"mode": "hard"},
                               thread_count=1)
    for result in (soft, hard):
        assert result.shape == frames[0].shape[:2]
        assert result.min() >= 0.0 and result.max() <= 1.0
    steps = hard * (len(frames) - 1)
    assert np.allclose(steps, np.round(steps), atol=1e-5)
    assert not np.array_equal(soft, hard)
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
    environment = dict(os.environ, OPENFOCUS_LOG_DIR=str(tmp_path / "logs"),
                       PYTHONPATH=str(root), KMP_DUPLICATE_LIB_OK="TRUE",
                       PYTHONIOENCODING="utf-8")

    def run(args):
        return subprocess.run(
            [sys.executable, str(root / "main.py"), *args],
            cwd=str(root), env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300)

    for method in ("stackmff", "stackmffv2"):
        output = tmp_path / f"{method}.png"
        result = run(["-i", str(source), "-o", str(output), "-m", method,
                      "--threads", "1", "--no-exposure-match"])
        assert result.returncode == 0, result.stderr
        assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED) is not None

    depth = tmp_path / "v1_depth.png"
    rejected = run(["-i", str(source), "-o", str(depth), "-m", "stackmff",
                    "--depth-map"])
    assert rejected.returncode == 2
    assert "not available" in rejected.stderr
    assert not depth.exists()

    accepted = run(["-i", str(source), "-o", str(tmp_path / "v2.png"),
                    "-m", "stackmffv2", "--depth-map", "--threads", "1"])
    assert accepted.returncode == 0, accepted.stderr
    assert (tmp_path / "v2_depth.png").is_file()
