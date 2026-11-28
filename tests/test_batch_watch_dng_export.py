"""Regression tests for Linear DNG export in Batch Processing and Watch Folder."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from core.watch_folder import StackCandidate, WatchFolderConfig, _default_render_stack
from core.workers import BatchWorker
from dialogs.batch import BatchProcessingDialog
from dialogs.watch_folder import WatchFolderDialog
from utils.linear_dng import LinearDNGMetadata


def _camera_meta() -> LinearDNGMetadata:
    return LinearDNGMetadata(
        ((0.8293, -0.1611, -0.1132),
         (-0.4759, 1.2711, 0.2275),
         (-0.1013, 0.2415, 0.5509)),
        (1024 / 1853, 1.0, 1024 / 1667),
        "Example",
        "Example Camera",
    )


def _linear_frames(count: int = 2) -> list[np.ndarray]:
    y, x = np.indices((48, 48))
    return [
        np.stack((2000 + x * 40 + i * 200, 4000 + y * 30, 8000 + x * 20), axis=2).astype(np.uint16)
        for i in range(count)
    ]


def test_batch_and_watch_dialogs_expose_dng_format(qapp):
    batch_dlg = BatchProcessingDialog()
    watch_dlg = WatchFolderDialog()
    try:
        assert batch_dlg.format_combo.findText("DNG") >= 0
        batch_dlg.format_combo.setCurrentText("DNG")
        assert batch_dlg.get_processing_settings()["format"] == "dng"
        assert not batch_dlg.quality_slider.isVisible()

        assert watch_dlg.format_combo.findText("DNG") >= 0
        watch_dlg.format_combo.setCurrentText("DNG")
        assert watch_dlg.build_config().output_format == "dng"
    finally:
        batch_dlg.close()
        batch_dlg.deleteLater()
        watch_dlg.close()
        watch_dlg.deleteLater()


def test_batch_worker_multiple_and_split_dng_export(tmp_path, monkeypatch):
    src_dir = tmp_path / "raw_stack"
    src_dir.mkdir()
    p1 = src_dir / "IMG_0001.dng"
    p2 = src_dir / "IMG_0002.dng"
    p1.write_bytes(b"RAW1")
    p2.write_bytes(b"RAW2")
    out_dir = tmp_path / "out_multi"
    frames = _linear_frames(2)
    meta = _camera_meta()

    def fake_load_linear(self, filepaths, scale_factor=1.0, **_kwargs):
        self.images = list(frames[: len(filepaths)])
        self.image_paths = list(filepaths)
        self.source_exif = b""
        self.source_icc_profile = b""
        self.px_size_um = 2.5
        self.failed_count = 0
        self.linear_dng_metadata = meta
        return True, f"Loaded {len(filepaths)}", list(frames[: len(filepaths)]), [
            os.path.basename(p) for p in filepaths
        ]

    monkeypatch.setattr("core.image_loader.ImageStackLoader.load_linear_raw_stack", fake_load_linear)

    worker = BatchWorker(
        folder_paths=[str(src_dir)],
        output_type="custom",
        output_path=str(out_dir),
        processing_settings={
            "format": "dng",
            "fusion_method": "openfocus_a",
            "alignment": "none",
            "save_aligned": False,
        },
        import_mode="multiple_folders",
    )
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert len(results) == 1
    payload = results[0]
    assert payload["counts"]["success"] == 1
    assert len(payload["outputs"]) == 1
    assert payload["outputs"][0].lower().endswith(".dng")
    with tifffile.TiffFile(payload["outputs"][0]) as tif:
        assert tif.asarray().shape == (48, 48, 3)

    # Single-folder split mode preloads display frames first; BatchWorker must
    # upgrade those split entries to camera-linear RAW before writing DNG.
    split_out = tmp_path / "out_split"
    preview_u8 = (frames[0] >> 8).astype(np.uint8)
    split_worker = BatchWorker(
        folder_paths=[],
        output_type="custom",
        output_path=str(split_out),
        processing_settings={
            "format": "dng",
            "fusion_method": "openfocus_a",
            "alignment": "none",
            "save_aligned": False,
        },
        import_mode="single_folder",
        split_method="count",
        split_param=2,
        single_folder_images_with_times=[
            (str(p1), preview_u8.copy(), 1.0, "IMG_0001.dng"),
            (str(p2), preview_u8.copy(), 2.0, "IMG_0002.dng"),
        ],
    )
    split_results = []
    split_worker.finished.connect(split_results.append)
    split_worker.run()
    assert len(split_results) == 1
    split_payload = split_results[0]
    assert split_payload["counts"]["success"] == 1
    assert split_payload["outputs"][0].lower().endswith(".dng")
    with tifffile.TiffFile(split_payload["outputs"][0]) as tif:
        assert tif.asarray().shape == (48, 48, 3)


def test_batch_worker_dng_rejects_non_raw_and_grouped(tmp_path):
    png_dir = tmp_path / "png_stack"
    png_dir.mkdir()
    img = np.full((24, 24, 3), 120, dtype=np.uint8)
    tifffile.imwrite(str(png_dir / "a.tif"), img)
    tifffile.imwrite(str(png_dir / "b.tif"), img)

    worker = BatchWorker(
        folder_paths=[str(png_dir)],
        output_type="custom",
        output_path=str(tmp_path / "out_err"),
        processing_settings={"format": "dng", "fusion_method": "openfocus_a", "alignment": "none"},
        import_mode="multiple_folders",
    )
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert results[0]["counts"]["failed"] == 1
    assert not results[0]["outputs"]

    grouped = BatchWorker(
        folder_paths=[str(png_dir)],
        output_type="custom",
        output_path=str(tmp_path / "out_grp"),
        processing_settings={
            "format": "dng",
            "fusion_method": "openfocus_a",
            "alignment": "none",
            "group_size": 2,
            "group_overlap": 0,
        },
        import_mode="multiple_folders",
    )
    grp_results = []
    grouped.finished.connect(grp_results.append)
    grouped.run()
    assert grp_results[0]["counts"]["failed"] == 1
    assert "DNG" in grp_results[0]["failed"][0]


def test_watch_folder_render_stack_exports_linear_dng(tmp_path, monkeypatch):
    src_dir = tmp_path / "watch" / "stack_01"
    src_dir.mkdir(parents=True)
    p1 = src_dir / "f1.dng"
    p2 = src_dir / "f2.dng"
    p1.write_bytes(b"RAW1")
    p2.write_bytes(b"RAW2")
    out_dir = tmp_path / "watch_out"
    frames = _linear_frames(2)
    meta = _camera_meta()

    def fake_load_linear(self, filepaths, scale_factor=1.0, **_kwargs):
        self.images = list(frames)
        self.image_paths = list(filepaths)
        self.source_exif = b""
        self.source_icc_profile = b""
        self.px_size_um = 1.5
        self.failed_count = 0
        self.linear_dng_metadata = meta
        return True, "ok", list(frames), ["f1.dng", "f2.dng"]

    monkeypatch.setattr("core.image_loader.ImageStackLoader.load_linear_raw_stack", fake_load_linear)

    candidate = StackCandidate(
        stack_id="stack_01",
        name="stack_01",
        source_dir=str(src_dir),
        file_paths=(str(p1), str(p2)),
        signature="sig_01",
    )
    cfg = WatchFolderConfig(
        watch_dir=str(tmp_path / "watch"),
        output_dir=str(out_dir),
        fusion_method="openfocus_a",
        alignment_method="none",
        output_format="dng",
    )
    written = _default_render_stack(candidate, cfg, lambda: False)
    assert written.lower().endswith(".dng")
    assert os.path.isfile(written)
    with tifffile.TiffFile(written) as tif:
        assert tif.asarray().shape == (48, 48, 3)
