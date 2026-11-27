"""Tests for the OpenFocus Interactive Live Color Preview engine (core/live_preview.py)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.live_preview import (
    DEFAULT_PREVIEW_SIZE,
    LIVE_PREVIEW_METHODS,
    PREVIEW_SIZES,
    LivePreviewCancelled,
    LivePreviewRequest,
    LivePreviewResult,
    LivePreviewSession,
    extract_cached_transforms,
)


def _make_synthetic_focus_stack(
    n_frames: int = 5,
    height: int = 240,
    width: int = 360,
    *,
    shifts: list[tuple[float, float]] | None = None,
) -> list[np.ndarray]:
    """Create a vivid multi-band focus stack where slice k is sharpest in vertical band k."""
    rng = np.random.default_rng(42)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    # Build high-contrast colored checkerboard + fine textured bands so each slice owns a zone.
    band_w = width // n_frames
    yy, xx = np.mgrid[0:height, 0:width]
    colors = [
        (220, 60, 40),
        (50, 210, 80),
        (40, 110, 235),
        (215, 180, 40),
        (190, 60, 210),
        (60, 200, 210),
    ]
    for k in range(n_frames):
        x0 = k * band_w
        x1 = width if k == n_frames - 1 else (k + 1) * band_w
        checker = (((yy[:, x0:x1] // 6) + (xx[:, x0:x1] // 6)) % 2).astype(np.uint8)
        fine = (((yy[:, x0:x1] // 2) + (xx[:, x0:x1] // 2)) % 2).astype(np.uint8)
        col = np.array(colors[k % len(colors)], dtype=np.uint8)
        bg = np.array([30, 30, 35], dtype=np.uint8)
        patch = np.where((checker ^ fine)[:, :, None] > 0, col, bg)
        noise = rng.integers(0, 18, size=patch.shape, dtype=np.uint8)
        base[:, x0:x1] = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    blurred_strong = cv2.GaussianBlur(base, (31, 31), 9.0)
    blurred_mid = cv2.GaussianBlur(base, (15, 15), 4.0)

    frames: list[np.ndarray] = []
    for k in range(n_frames):
        frame = blurred_strong.copy()
        x0 = k * band_w
        x1 = width if k == n_frames - 1 else (k + 1) * band_w
        # Smoothly transition focus so adjacent bands have moderate blur and band k is razor sharp.
        if k > 0:
            px0 = (k - 1) * band_w
            frame[:, px0:x0] = blurred_mid[:, px0:x0]
        if k + 1 < n_frames:
            nx1 = width if k + 1 == n_frames - 1 else (k + 2) * band_w
            frame[:, x1:nx1] = blurred_mid[:, x1:nx1]
        frame[:, x0:x1] = base[:, x0:x1]

        if shifts is not None:
            dx, dy = shifts[k]
            m = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float64)
            frame = cv2.warpAffine(
                frame, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        frames.append(frame)
    return frames


def _laplacian_var(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class TestProxyAcquisitionAndMemory:
    def test_size_presets_and_low_memory(self):
        big_stack = _make_synthetic_focus_stack(n_frames=4, height=1200, width=1800)
        for preset, edge in PREVIEW_SIZES.items():
            with LivePreviewSession(big_stack, size=preset, align_method="none") as session:
                assert session.size_name == preset
                assert session.frame_count == 4
                w, h = session.proxy_size
                assert max(w, h) == edge
                assert session.proxy_shape == (h, w, 3)
                assert session.source_shape == (1200, 1800)
                # Contiguous uint8 BGR + float32 gray must be far smaller than full 1800x1200 stack.
                full_bytes = 4 * 1200 * 1800 * 3
                assert session.approx_memory_bytes < full_bytes

    def test_reuses_source_frame_preview_without_full_decode(self, tmp_path: Path):
        from core.source_frames import SourceFrame

        frames = _make_synthetic_focus_stack(n_frames=3, height=960, width=1440)
        sources: list[SourceFrame] = []
        for idx, img in enumerate(frames):
            path = tmp_path / f"frame_{idx}.jpg"
            cv2.imwrite(str(path), img)
            stat = os.stat(path)
            preview = cv2.resize(img, (768, 512), interpolation=cv2.INTER_AREA)
            sf = SourceFrame(
                path=str(path),
                page=0,
                shape=img.shape,
                dtype=img.dtype,
                preview=preview,
                fingerprint=(stat.st_size, stat.st_mtime_ns),
            )
            sources.append(sf)

        # Monkeypatch full_resolution to fail if ever called during proxy acquisition.
        def _forbid_full(*_args, **_kwargs):
            raise AssertionError("LivePreviewSession must not full-decode when preview >= target size")

        orig = SourceFrame.full_resolution
        SourceFrame.full_resolution = _forbid_full  # type: ignore[assignment]
        try:
            with LivePreviewSession(sources, size="medium", align_method="none") as session:
                assert session.frame_count == 3
                assert max(session.proxy_size) == PREVIEW_SIZES["medium"]
                res = session.render(LivePreviewRequest(method="openfocus_b"))
                assert res.image_bgr.shape[:2] == session.proxy_shape[:2]
        finally:
            SourceFrame.full_resolution = orig  # type: ignore[assignment]

    def test_unicode_path_jpeg_and_png_loading(self, tmp_path: Path):
        unicode_dir = tmp_path / "测试目录_微距"
        unicode_dir.mkdir()
        frames = _make_synthetic_focus_stack(n_frames=3, height=640, width=960)
        paths = []
        for i, img in enumerate(frames):
            ext = ".jpg" if i < 2 else ".png"
            p = unicode_dir / f"切片_{i}{ext}"
            ok, encoded = cv2.imencode(ext, img)
            assert ok
            p.write_bytes(encoded.tobytes())
            paths.append(str(p))

        with LivePreviewSession(paths, size="small", align_method="none") as session:
            assert session.frame_count == 3
            assert max(session.proxy_size) == PREVIEW_SIZES["small"]
            res = session.render(LivePreviewRequest(method="openfocus_a"))
            assert res.image_bgr.dtype == np.uint8
            assert res.image_rgb.shape == res.image_bgr.shape


class TestCachedTransformsAndAlignment:
    def test_cached_matrix_rescaling_aligns_shifted_stack(self):
        # Create full-res stack (960x1440) with known pixel shifts, then supply exact inverse full-res matrices.
        full_shifts = [(0.0, 0.0), (24.0, -12.0), (-18.0, 16.0), (12.0, 20.0)]
        stack = _make_synthetic_focus_stack(n_frames=4, height=960, width=1440, shifts=full_shifts)
        # Forward matrices that undo the shifts in full-resolution coordinates.
        cached_mats = [
            np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]], dtype=np.float64)
            for dx, dy in full_shifts
        ]

        with LivePreviewSession(
            stack,
            size="small",
            align_method="ecc",
            cached_transforms=cached_mats,
        ) as session:
            assert session.used_cached_alignment is True
            # After inverse shift alignment, overlapping central regions should have low difference
            # compared to an unaligned session.
            res = session.render(LivePreviewRequest(method="openfocus_b"))
            assert res.used_cached_alignment is True
            # Verify alignment actually shifted frame 1 back onto frame 0's grid.
            s0 = session.get_source_frame(0)
            s1 = session.get_source_frame(1)
            # Band 0 vs Band 1 have different blur, but color centroids across the full frame stay aligned.
            diff_aligned = float(np.mean(np.abs(s0.astype(np.float32) - s1.astype(np.float32))))

        with LivePreviewSession(stack, size="small", align_method="none") as unaligned:
            assert unaligned.used_cached_alignment is False
            u0 = unaligned.get_source_frame(0)
            u1 = unaligned.get_source_frame(1)
            diff_unaligned = float(np.mean(np.abs(u0.astype(np.float32) - u1.astype(np.float32))))

        assert diff_aligned < diff_unaligned

    def test_extract_cached_transforms_from_processed_frames(self):
        from core.openfocus_alignment import SourceFingerprint
        from core.registration import PerspectiveWarp
        from core.source_frames import process_frame

        stack = _make_synthetic_focus_stack(n_frames=3, height=320, width=480)
        processed = []
        for idx, frame in enumerate(stack):
            fp = SourceFingerprint(shape=frame.shape, dtype=frame.dtype.str, sha256="0" * 64)
            mat = (1.0, 0.0, float(-idx * 2), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            warp = PerspectiveWarp(fp, mat, "<f8", (476, 320), None, method="ecc")
            processed.append(process_frame(frame, warp, shape=(320, 476, 3)))

        extracted = extract_cached_transforms(processed)
        assert extracted is not None

        with LivePreviewSession(
            stack,
            size="small",
            align_method="ecc",
            aligned_frames=processed,
        ) as session:
            assert session.used_cached_alignment is True
            res = session.render(LivePreviewRequest(method="openfocus_b"))
            assert res.used_cached_alignment is True

    def test_proxy_auto_alignment_recovers_shift(self):
        shifts = [(0.0, 0.0), (8.0, -4.0), (14.0, -8.0), (6.0, 4.0)]
        stack = _make_synthetic_focus_stack(n_frames=4, height=320, width=480, shifts=shifts)

        with LivePreviewSession(stack, size="small", align_method="ecc") as aligned:
            assert aligned.used_cached_alignment is False
            # Proxy alignment should crop borders slightly and keep sharp fused result.
            res = aligned.render(LivePreviewRequest(method="openfocus_b"))
            assert _laplacian_var(res.image_bgr) > _laplacian_var(aligned.get_source_frame(0))


class TestFusionMethodsAndParameters:
    @pytest.fixture
    def session(self):
        stack = _make_synthetic_focus_stack(n_frames=5, height=240, width=360)
        s = LivePreviewSession(stack, size="medium", align_method="none")
        yield s
        s.close()

    @pytest.mark.parametrize("method", LIVE_PREVIEW_METHODS)
    def test_all_methods_produce_sharp_valid_output_and_histogram(self, session: LivePreviewSession, method: str):
        res = session.render(LivePreviewRequest(method=method, radius=8, smoothing=4.0, feather=3))
        assert isinstance(res, LivePreviewResult)
        assert res.method == method
        assert res.image_bgr.dtype == np.uint8
        assert res.image_bgr.shape == session.proxy_shape
        assert res.image_rgb.shape == session.proxy_shape
        assert np.array_equal(res.image_rgb[:, :, 0], res.image_bgr[:, :, 2])
        assert res.proxy_size == session.proxy_size
        assert res.elapsed_ms >= 0.0

        # Fused output must be substantially sharper than any single partially-focused slice.
        single_max_sharp = max(_laplacian_var(session.get_source_frame(i)) for i in range(session.frame_count))
        assert _laplacian_var(res.image_bgr) > single_max_sharp * 1.4

        # Depth histogram validity: length N, non-negative, sums to 1.0, active across all 5 bands.
        hist = res.depth_histogram
        assert hist.shape == (session.frame_count,)
        assert hist.dtype == np.float32
        assert np.all(hist >= 0.0)
        assert float(hist.sum()) == pytest.approx(1.0, abs=1e-4)
        # All 5 vertical bands are equally sized, so every slice should contribute meaningfully.
        assert np.all(hist > 0.05)

    def test_openfocus_a_radius_sensitivity_and_inert_smoothing(self, session: LivePreviewSession):
        r1 = session.render(LivePreviewRequest(method="openfocus_a", radius=2, smoothing=0.0))
        r2 = session.render(LivePreviewRequest(method="openfocus_a", radius=24, smoothing=0.0))
        r3 = session.render(LivePreviewRequest(method="openfocus_a", radius=24, smoothing=8.0))
        assert not np.array_equal(r1.image_bgr, r2.image_bgr)
        # Smoothing is intentionally inert in OpenFocus A, matching fusion_methods/openfocus_a.py.
        assert np.array_equal(r2.image_bgr, r3.image_bgr)

    def test_openfocus_b_parameter_sensitivity(self, session: LivePreviewSession):
        base = session.render(LivePreviewRequest(method="openfocus_b", radius=4, smoothing=2.0, feather=0))
        diff_radius = session.render(LivePreviewRequest(method="openfocus_b", radius=28, smoothing=2.0, feather=0))
        diff_smooth = session.render(LivePreviewRequest(method="openfocus_b", radius=4, smoothing=24.0, feather=0))
        diff_feather = session.render(LivePreviewRequest(method="openfocus_b", radius=4, smoothing=24.0, feather=10))

        assert not np.array_equal(base.image_bgr, diff_radius.image_bgr)
        assert not np.array_equal(base.image_bgr, diff_smooth.image_bgr)
        assert not np.array_equal(diff_smooth.image_bgr, diff_feather.image_bgr)

    def test_openfocus_c_smoothing_sensitivity(self, session: LivePreviewSession):
        c0 = session.render(LivePreviewRequest(method="openfocus_c", smoothing=0.0))
        c1 = session.render(LivePreviewRequest(method="openfocus_c", smoothing=12.0))
        assert not np.array_equal(c0.image_bgr, c1.image_bgr)

    @pytest.mark.parametrize("method", LIVE_PREVIEW_METHODS)
    def test_draft_scale_and_full_refine_parity(self, method: str):
        stack = _make_synthetic_focus_stack(n_frames=5, height=320, width=480)
        with LivePreviewSession(stack, size="small", align_method="none") as session:
            draft_res = session.render(
                LivePreviewRequest(method=method, radius=10, smoothing=4.0, feather=3, draft=True)
            )
            full_res = session.render(
                LivePreviewRequest(method=method, radius=10, smoothing=4.0, feather=3, draft=False)
            )
            assert draft_res.is_draft is True
            assert full_res.is_draft is False
            assert draft_res.image_bgr.dtype == np.uint8
            assert draft_res.image_bgr.shape[0] < full_res.image_bgr.shape[0]
            assert draft_res.image_bgr.shape[1] < full_res.image_bgr.shape[1]
            assert full_res.image_bgr.shape == session.proxy_shape
            assert draft_res.depth_histogram.shape == (session.frame_count,)
            assert float(draft_res.depth_histogram.sum()) == pytest.approx(1.0, abs=1e-4)
            # Upscaled draft should visually agree closely with the full-proxy render.
            draft_up = cv2.resize(
                draft_res.image_bgr,
                (full_res.image_bgr.shape[1], full_res.image_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            mae = float(np.mean(np.abs(draft_up.astype(np.float32) - full_res.image_bgr.astype(np.float32))))
            assert mae < 35.0

    def test_cache_memory_budget_evicts_inactive_methods(self):
        stack = _make_synthetic_focus_stack(n_frames=12, height=480, width=720)
        # Tight 24 MB cap forces LRU eviction of inactive method caches when switching A -> B -> C.
        budget = 24 * 1024 * 1024
        with LivePreviewSession(
            stack,
            size="medium",
            align_method="none",
            max_cache_bytes=budget,
        ) as session:
            session.render(LivePreviewRequest(method="openfocus_a", radius=8))
            assert session._a_burt_levels is not None
            session.render(LivePreviewRequest(method="openfocus_b", radius=8))
            assert session._b_gray_half is not None
            session.render(LivePreviewRequest(method="openfocus_c", smoothing=4.0))
            assert session._c_weight_levels is not None
            # Switching through all 3 methods under a tight budget evicts older caches.
            assert session._a_burt_levels is None or session._b_gray_half is None
            assert session.approx_memory_bytes <= budget + 4 * 1024 * 1024
            # Switching back rebuilds seamlessly with accurate output.
            res_a = session.render(LivePreviewRequest(method="openfocus_a", radius=8))
            assert res_a.image_bgr.shape == session.proxy_shape


class TestSubstackSelection:
    @pytest.mark.parametrize("method", LIVE_PREVIEW_METHODS)
    def test_substack_slicing_and_histogram_alignment(self, method: str):
        stack = _make_synthetic_focus_stack(n_frames=6, height=200, width=300)
        with LivePreviewSession(stack, size="medium", align_method="none") as session:
            # Warm cache on full stack first.
            full = session.render(LivePreviewRequest(method=method))
            mem_after_warm = session.approx_memory_bytes
            proxy_id = id(session._proxy_bgr)

            # Trim to middle substack [1, 4] (inclusive).
            sub = session.render(LivePreviewRequest(method=method, substack_range=(1, 4)))
            assert sub.substack_range == (1, 4)
            assert sub.depth_histogram.shape == (6,)
            assert float(sub.depth_histogram[0]) == 0.0
            assert float(sub.depth_histogram[5]) == 0.0
            assert float(sub.depth_histogram[1:5].sum()) == pytest.approx(1.0, abs=1e-4)
            assert not np.array_equal(full.image_bgr, sub.image_bgr)

            # Zero rebuild: underlying proxy stack unchanged and memory does not explode.
            assert id(session._proxy_bgr) == proxy_id
            assert session.approx_memory_bytes <= mem_after_warm + 2 * 1024 * 1024

            # Single-frame substack returns exact slice and one-hot histogram.
            single = session.render(LivePreviewRequest(method=method, substack_range=(3, 3)))
            assert np.array_equal(single.image_bgr, session.get_source_frame(3))
            expected_hist = np.zeros(6, dtype=np.float32)
            expected_hist[3] = 1.0
            assert np.allclose(single.depth_histogram, expected_hist)

    def test_substack_validation_errors(self):
        stack = _make_synthetic_focus_stack(n_frames=4, height=120, width=180)
        with LivePreviewSession(stack, size="small", align_method="none") as session:
            with pytest.raises(ValueError):
                LivePreviewRequest(substack_range=(3, 1))
            with pytest.raises(ValueError):
                LivePreviewRequest(substack_range=(-1, 2))
            with pytest.raises(ValueError):
                session.render(LivePreviewRequest(substack_range=(0, 4)))


class TestLookupCancellationAndLifecycle:
    def test_instant_source_lookup(self):
        stack = _make_synthetic_focus_stack(n_frames=5, height=200, width=300)
        with LivePreviewSession(stack, size="medium", align_method="none") as session:
            t0 = time.perf_counter()
            frame = session.get_source_frame(2)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert frame.shape == session.proxy_shape
            assert frame.dtype == np.uint8
            assert elapsed_ms < 5.0
            rgb = session.get_source_frame_rgb(2)
            assert np.array_equal(rgb[:, :, 0], frame[:, :, 2])
            with pytest.raises(IndexError):
                session.get_source_frame(5)
            with pytest.raises(IndexError):
                session.get_source_frame(-1)

    def test_cancellation_during_prepare_and_render(self):
        stack = _make_synthetic_focus_stack(n_frames=4, height=160, width=240)
        with pytest.raises(LivePreviewCancelled):
            LivePreviewSession(stack, size="small", align_method="none", cancel_check=lambda: True)

        with LivePreviewSession(stack, size="small", align_method="none") as session:
            with pytest.raises(LivePreviewCancelled):
                session.render(LivePreviewRequest(method="openfocus_b"), cancel_check=lambda: True)

    def test_progress_callback_invoked(self):
        stack = _make_synthetic_focus_stack(n_frames=4, height=160, width=240)
        events: list[tuple[int, int, str]] = []
        with LivePreviewSession(
            stack,
            size="small",
            align_method="ecc",
            progress_cb=lambda d, t, st: events.append((d, t, st)),
        ):
            pass
        stages = {st for _, _, st in events}
        assert "load" in stages
        assert "align" in stages
        assert "warp" in stages

    def test_close_releases_memory_and_blocks_further_use(self):
        stack = _make_synthetic_focus_stack(n_frames=5, height=300, width=450)
        session = LivePreviewSession(stack, size="medium", align_method="none")
        for method in LIVE_PREVIEW_METHODS:
            session.render(LivePreviewRequest(method=method))
        assert session.approx_memory_bytes > 0
        assert not session.is_closed

        session.close()
        assert session.is_closed
        assert session.approx_memory_bytes == 0
        assert session.frame_count == 0
        # Calling close() twice is idempotent.
        session.close()

        with pytest.raises(RuntimeError):
            session.get_source_frame(0)
        with pytest.raises(RuntimeError):
            session.render(LivePreviewRequest(method="openfocus_b"))
