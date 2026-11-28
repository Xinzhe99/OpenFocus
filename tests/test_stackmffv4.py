"""Integration, memory-planning, and UI visibility tests for StackMFF V4 Flash."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent
V4_ONNX = ROOT / "weights" / "stackmff_v4_flash.onnx"
V4_PTH = ROOT / "weights" / "stackmff_v4_flash.pth"
V4_ONNX_SHA256 = "1623a4fd98ac75118bcbda7fb23391de2848f4b90d5654c1c8f6e0e633bb49db"
V4_PTH_SHA256 = "f113d229985bf5c71be43a8eff7d512fcbd8c1dd3463e29e35d3d1abdc9d244a"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _synthetic_stack(n: int = 3, h: int = 96, w: int = 128, dtype=np.uint8):
    rng = np.random.default_rng(2026)
    base_colors = [
        np.array([30, 120, 210], dtype=np.float32),
        np.array([200, 60, 40], dtype=np.float32),
        np.array([40, 190, 90], dtype=np.float32),
    ]
    frames = []
    yy, xx = np.mgrid[0:h, 0:w]
    for idx in range(n):
        col = base_colors[idx % len(base_colors)]
        canvas = np.zeros((h, w, 3), dtype=np.float32) + col[None, None, :] * 0.25
        cx = int(w * (idx + 0.5) / n)
        mask = np.exp(-((xx - cx) ** 2 + (yy - h // 2) ** 2) / (2 * (min(h, w) * 0.22) ** 2))
        texture = 0.5 + 0.5 * np.sin(xx * 0.55) * np.cos(yy * 0.55)
        signal = col[None, None, :] * 0.45 + texture[:, :, None] * 110.0 * mask[:, :, None]
        noise = rng.normal(0.0, 2.0, size=(h, w, 3))
        arr = np.clip(canvas * (1.0 - mask[:, :, None]) + signal + noise, 0, 255)
        if dtype == np.uint16:
            frames.append((arr * 257.0).astype(np.uint16))
        else:
            frames.append(arr.astype(np.uint8))
    return frames


def test_v4_weights_integrity_and_single_file_onnx_signature():
    import onnxruntime as ort

    assert V4_ONNX.is_file()
    assert V4_PTH.is_file()
    assert not Path(str(V4_ONNX) + ".data").exists()
    assert _sha256(V4_ONNX) == V4_ONNX_SHA256
    assert _sha256(V4_PTH) == V4_PTH_SHA256

    sess = ort.InferenceSession(str(V4_ONNX), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    assert [i.name for i in inputs] == ["input"]
    assert [o.name for o in outputs] == ["logits"]
    assert len(inputs[0].shape) == 4
    assert len(outputs[0].shape) == 4


def test_v4_pytorch_module_matches_expected_shape():
    torch = pytest.importorskip("torch")
    from core.models.stackmff_v4.network import StreamFocusNet

    model = StreamFocusNet()
    x = torch.rand(1, 3, 64, 64)
    y = model(x)
    assert y.shape == (1, 3, 64, 64)


def test_v4_registry_flags_and_legacy_separation():
    from core.algorithms import fusion_algorithms, get_fusion

    v4 = get_fusion("stackmffv4")
    assert v4.family == "stackmff"
    assert v4.experimental is True
    assert v4.legacy is False
    assert v4.order_sensitive is False
    assert v4.weights == "weights/stackmff_v4_flash.onnx"

    ids = [spec.id for spec in fusion_algorithms()]
    assert ids.index("stackmffv4") < ids.index("stackmffv3")
    for legacy_id in ("stackmffv3", "stackmffv2", "stackmff"):
        spec = get_fusion(legacy_id)
        assert spec.legacy is True


def test_v4_single_scale_inference_hard_provenance_and_permutation_equivariance(monkeypatch):
    from fusion_methods import neural_runtime, stackmffv4

    frames = _synthetic_stack(n=3, h=64, w=64, dtype=np.uint8)

    # Count ORT logits evaluations to prove single-scale scales=(1.0,) runs 1 pass per tile.
    call_count = {"n": 0}
    orig_logits = neural_runtime.logits

    def _counting_logits(*args, **kwargs):
        call_count["n"] += 1
        return orig_logits(*args, **kwargs)

    monkeypatch.setattr(neural_runtime, "logits", _counting_logits)

    soft = stackmffv4._stackmffv4_impl(
        frames, None, str(V4_ONNX), use_gpu=False, mode="soft", thread_count=1
    )
    assert call_count["n"] == 1
    assert soft.shape == frames[0].shape
    assert soft.dtype == np.uint8

    hard = stackmffv4._stackmffv4_impl(
        frames, None, str(V4_ONNX), use_gpu=False, mode="hard", thread_count=1
    )
    assert hard.shape == frames[0].shape
    assert hard.dtype == np.uint8
    matches_any = np.zeros(hard.shape[:2], dtype=bool)
    for frame in frames:
        matches_any |= np.all(hard == frame, axis=2)
    assert bool(np.all(matches_any))

    # Order-invariant: reversed stack must yield virtually identical fused output.
    soft_rev = stackmffv4._stackmffv4_impl(
        list(reversed(frames)), None, str(V4_ONNX), use_gpu=False, mode="soft", thread_count=1
    )
    max_diff = int(np.max(np.abs(soft.astype(np.int32) - soft_rev.astype(np.int32))))
    assert max_diff <= 1


def test_v4_uint16_and_depth_map(qapp):
    from core.depth_map import compute_focus_index
    from core.workers import RenderWorker

    values = (np.arange(65 * 67, dtype=np.uint16).reshape(65, 67) * 11 + 1001)
    frame = np.repeat(values[..., None], 3, axis=2)
    frames_16 = [frame, frame.copy()]
    emitted = {}
    worker = RenderWorker(
        raw_images=frames_16,
        aligned_images=[],
        is_images_aligned=False,
        algorithm="stackmffv4",
        fusion_parameters={"mode": "hard"},
        use_gpu=False,
        thread_count=1,
        exposure_match=False,
        tile_enabled=False,
    )
    worker.finished_signal.connect(lambda *args: emitted.setdefault("result", args[1]))
    worker.run()
    fused_16 = emitted["result"]
    assert fused_16.dtype == np.uint16
    np.testing.assert_array_equal(fused_16, frame)
    assert np.any(fused_16 % 257), "true 16-bit low bits must be preserved"

    depth = compute_focus_index(
        _synthetic_stack(n=3, h=64, w=64, dtype=np.uint16),
        "stackmffv4",
        use_gpu=False,
        thread_count=1,
    )
    assert depth.dtype == np.float32
    assert depth.shape == (64, 64)
    assert 0.0 <= float(depth.min()) <= float(depth.max()) <= 1.0


def test_v4_linear_memory_estimate_much_smaller_than_quadratic_v3():
    from fusion_methods.neural_common import attention_inference_bytes
    from fusion_methods.neural_runtime import (
        estimate_inference_bytes,
        estimate_linear_inference_bytes,
    )

    # For deep stacks on CPU (N=400) or streaming CUDA (N=120), V4 O(N)/O(1) is far below V3 O(N^2).
    q_cpu = estimate_inference_bytes(400, 512, 512, "CPU", linear_layers=False)
    lin_cpu = estimate_linear_inference_bytes(400, 512, 512, "CPU")
    assert estimate_inference_bytes(400, 512, 512, "CPU", linear_layers=True) == lin_cpu
    assert lin_cpu * 2 < q_cpu

    q_cuda = estimate_inference_bytes(120, 512, 512, "TORCH_CUDA", linear_layers=False)
    lin_cuda = estimate_linear_inference_bytes(120, 512, 512, "TORCH_CUDA")
    assert lin_cuda * 5 < q_cuda

    v3_total, _ = attention_inference_bytes(400, 512, 512, 3, "CPU", linear_layers=False)
    v4_total, _ = attention_inference_bytes(400, 512, 512, 3, "CPU", linear_layers=True)
    assert v4_total * 2 < v3_total


def test_legacy_features_toggle_controls_ui_visibility(qapp, monkeypatch, tmp_path):
    from PyQt6.QtCore import QSettings
    from dialogs.advanced_settings import AdvancedSettingsDialog
    from dialogs.batch import BatchProcessingDialog
    from dialogs.compare import CompareDialog
    from dialogs.depthmap import DepthMapDialog
    from dialogs.watch_folder import WatchFolderDialog
    from locales import trans
    from main import OpenFocus
    from utils import recovery, settings_store

    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("ui/first_run_done", True)
    settings.setValue("updates/last_check_epoch", 9999999999)
    monkeypatch.setattr(settings_store, "_settings_singleton", settings)
    monkeypatch.setattr(recovery, "_base_dir", lambda: str(tmp_path / "recovery"))

    w = OpenFocus()
    w._stackmff_available = True
    w.apply_experimental_features()
    w.apply_legacy_features()
    try:
        assert not w.legacy_features
        # Main window: V4 visible (experimental); V1-V3 hidden by default.
        v4_idx = w.combo_fusion.findData("stackmffv4")
        assert v4_idx >= 0
        assert not w.combo_fusion.model().item(v4_idx).isEnabled()
        assert trans.t("experimental_label") in w.combo_fusion.itemText(v4_idx)
        for legacy_id in ("stackmffv3", "stackmffv2", "stackmff"):
            assert w.combo_fusion.findData(legacy_id) < 0

        for dlg_cls, attr in (
            (BatchProcessingDialog, "fusion_combo"),
            (WatchFolderDialog, "fusion_combo"),
            (DepthMapDialog, "method_combo"),
        ):
            dlg = dlg_cls(w)
            combo = getattr(dlg, attr)
            assert combo.findData("stackmffv4") >= 0
            for legacy_id in ("stackmffv3", "stackmffv2", "stackmff"):
                assert combo.findData(legacy_id) < 0
            dlg.close()
            dlg.deleteLater()

        cmp_dlg = CompareDialog(w)
        assert "stackmffv4" in cmp_dlg._checks
        for legacy_id in ("stackmffv3", "stackmffv2", "stackmff"):
            assert legacy_id not in cmp_dlg._checks
        cmp_dlg.reject()
        cmp_dlg.deleteLater()

        # Enabling Legacy Features in Advanced Settings reveals & unlocks V1-V3 without experimental_features.
        adv = AdvancedSettingsDialog(w)
        adv.chk_legacy_features.setChecked(True)
        adv._apply()
        assert w.legacy_features is True
        assert w.experimental_features is False
        for legacy_id in ("stackmffv3", "stackmffv2", "stackmff"):
            idx = w.combo_fusion.findData(legacy_id)
            assert idx >= 0
            assert w.combo_fusion.model().item(idx).isEnabled()
            assert trans.t("legacy_label") in w.combo_fusion.itemText(idx)

        w.set_fusion_method("stackmffv3")
        assert w.current_fusion_method() == "stackmffv3"

        # Turning Legacy Features off hides V1-V3 again and falls back to OpenFocus B when experimental features are off.
        adv2 = AdvancedSettingsDialog(w)
        adv2.chk_legacy_features.setChecked(False)
        adv2._apply()
        assert w.legacy_features is False
        assert w.current_fusion_method() == "openfocus_b"
        for legacy_id in ("stackmffv3", "stackmffv2", "stackmff"):
            assert w.combo_fusion.findData(legacy_id) < 0
    finally:
        w._force_quit = True
        w.close()
        w.deleteLater()
        qapp.processEvents()
