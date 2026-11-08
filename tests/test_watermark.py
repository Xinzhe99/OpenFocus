"""Watermark geometry, validation, bit-depth and scope contracts."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PIL")

from utils import watermark as wm  # noqa: E402


def _config(**overrides):
    values = {
        "text": "© OpenFocus 测试",
        "font_family": "",
        "size_percent": 6.0,
        "opacity": 0.5,
        "color": "white",
        "angle": -25.0,
        "density": "standard",
    }
    values.update(overrides)
    return wm.WatermarkConfig(**values)


class _Record:
    def __init__(self, output_id="r1", *, config=None, initialized=True,
                 baked=False, kind="fusion", image=None, context=None):
        self.output_id = output_id
        self.watermark_config = config
        self.watermark_initialized = initialized
        self.watermark_baked = baked
        self.kind = kind
        self.image = image
        self.context = context


class _Window:
    def __init__(self):
        self.raw_images = []
        self.source_revision = 0
        self.image_source_paths = []
        self.output_list = _List()
        self.output_manager = _Manager(self)
        self.watermark_text = ""
        self.watermark_font_family = ""
        self.watermark_size_percent = wm.DEFAULT_SIZE_PERCENT
        self.watermark_opacity = wm.DEFAULT_OPACITY
        self.watermark_color = wm.DEFAULT_COLOR
        self.watermark_angle = wm.DEFAULT_ANGLE
        self.watermark_density = wm.DEFAULT_DENSITY
        self.watermark_source_enabled = True
        self.watermark_registered_enabled = False
        self.watermark_output_enabled = True


class _List:
    def __init__(self):
        self.records = []

    def count(self):
        return len(self.records)


class _Manager:
    def __init__(self, window):
        self.window = window

    def record_for_row(self, row):
        return self.window.output_list.records[row]


@pytest.mark.parametrize("text", ["", "   ", "\n", "\t", "x" * 81])
def test_invalid_text_is_rejected(text):
    with pytest.raises(wm.WatermarkError):
        wm.normalize_config({"text": text})


@pytest.mark.parametrize("text", ["© 创作者", "Café — 2026", "😀 watermark"])
def test_unicode_text_is_preserved(text):
    cfg = wm.normalize_config({"text": text})
    assert cfg.text == text


@pytest.mark.parametrize("field,value", [
    ("size_percent", 0.5), ("size_percent", 13),
    ("opacity", 0.01), ("opacity", 1.0),
    ("angle", -61), ("angle", 61),
    ("color", "gray"), ("density", "wide"),
])
def test_invalid_style_is_rejected(field, value):
    values = {"text": "mark"}
    values[field] = value
    with pytest.raises(wm.WatermarkError):
        wm.normalize_config(values)


def test_safe_config_clamps_corrupt_style_but_keeps_text():
    cfg = wm.safe_config({"text": "keep", "opacity": float("nan"),
                          "size_percent": 99, "angle": "bad"})
    assert cfg is not None and cfg.text == "keep"
    assert cfg.opacity == wm.DEFAULT_OPACITY
    assert cfg.size_percent == wm.DEFAULT_SIZE_PERCENT
    assert cfg.angle == wm.DEFAULT_ANGLE
    assert wm.safe_config({"text": ""}) is None


def test_burn_changes_whole_canvas_and_is_a_copy():
    image = np.full((480, 640, 3), 180, np.uint8)
    result = wm.burn(image, _config())
    assert result is not image
    assert np.any(result != image)
    # Pattern reaches all four quadrants, including the edges.
    for ys, xs in ((slice(0, 240), slice(0, 320)),
                   (slice(0, 240), slice(320, None)),
                   (slice(240, None), slice(0, 320)),
                   (slice(240, None), slice(320, None))):
        assert np.any(result[ys, xs] != image[ys, xs])


def test_density_changes_visible_watermark_coverage():
    image = np.full((600, 800, 3), 127, np.uint8)
    coverage = {}
    for density in ("sparse", "standard", "dense"):
        result = wm.burn(image, _config(density=density))
        coverage[density] = np.count_nonzero(np.any(result != image, axis=2))
    assert coverage["sparse"] < coverage["standard"] < coverage["dense"]


def test_single_pixel_regions_match_full_canvas_including_stroke_only():
    config = _config(text="TEST", size_percent=8.0)
    image = np.full((300, 400, 3), 140, np.uint8)
    full = wm.burn(image, config)
    changed = np.argwhere(np.any(full != image, axis=2))
    step = max(1, len(changed) // 400)
    for y, x in changed[::step]:
        pixel = wm.burn(image[y:y + 1, x:x + 1].copy(), config,
                        canvas_size=(400, 300), origin=(int(x), int(y)), scale=1.0)
        assert np.array_equal(pixel[0, 0], full[y, x]), (int(x), int(y))




def test_burn_can_be_cancelled_in_the_stamp_loop():
    with pytest.raises(wm.WatermarkCancelled):
        wm.burn(np.full((200, 260, 3), 100, np.uint8), _config(),
                should_cancel=lambda: True)


def test_region_is_exactly_the_same_pattern_as_the_full_image():
    image = np.full((900, 1200, 3), 120, np.uint8)
    config = _config()
    full = wm.burn(image, config)
    region = wm.burn(image[200:520, 300:700].copy(), config,
                     canvas_size=(1200, 900), origin=(300, 200), scale=1.0)
    assert np.array_equal(region, full[200:520, 300:700])


def test_scaled_preview_uses_the_full_canvas_geometry():
    config = _config()
    small = np.full((300, 400, 3), 100, np.uint8)
    preview = wm.burn(small, config, canvas_size=(1200, 900))
    assert np.any(preview != small)
    layout = wm.plan(config, 1200, 900)
    assert layout.font_px == round(900 * config.size_percent / 100)


def test_sixteen_bit_keeps_depth_and_outside_pixels():
    image = np.full((300, 400, 3), 40000, np.uint16)
    result = wm.burn(image, _config())
    assert result.dtype == np.uint16
    changed = result != image
    assert changed.any()
    assert np.array_equal(result[~changed], image[~changed])
    assert result.max() > image.max()
    # Blending used the original 16-bit base, so anti-aliased pixels are not
    # the 8-bit * 257 palette the old scale-bar path produced.
    assert np.any(result[changed] % 257 != 0)


def test_sixteen_bit_blend_depends_on_original_values():
    a = np.full((200, 260), 10000, np.uint16)
    b = np.full((200, 260), 50000, np.uint16)
    out_a = wm.burn(a, _config(opacity=0.5))
    out_b = wm.burn(b, _config(opacity=0.5))
    changed = out_a != a
    assert changed.any()
    # Where both masks are fully opaque, the delta to white is proportional.
    both = changed & (out_b != b)
    delta_a = out_a[both].astype(np.int64) - a[both].astype(np.int64)
    delta_b = out_b[both].astype(np.int64) - b[both].astype(np.int64)
    assert np.any(delta_a != 0) and np.any(delta_b != 0)
    assert not np.array_equal(delta_a, delta_b)


def test_rgba_keeps_alpha_and_transparent_pixels():
    image = np.zeros((200, 300, 4), np.uint8)
    image[..., :3] = 40
    image[50:150, 80:220, 3] = 255
    result = wm.burn(image, _config(size_percent=10.0, opacity=0.8))
    assert np.array_equal(result[..., 3], image[..., 3])
    transparent = image[..., 3] == 0
    assert np.array_equal(result[transparent], image[transparent])


@pytest.mark.parametrize("width,height", [
    (640, 480), (2560, 1920), (1920, 2560), (800, 200), (200, 800),
])
def test_layouts_cover_common_and_extreme_aspects(width, height):
    config = _config(size_percent=3.5)
    layout = wm.plan(config, width, height)
    assert layout.font_px == round(min(width, height) * 0.035)
    assert layout.cells
    assert layout.row_pitch >= layout.stamp_height
    assert layout.col_pitch >= layout.stamp_width
    xs = [x for x, _ in layout.cells]
    ys = [y for _, y in layout.cells]
    assert min(xs) <= 0 and max(xs) >= width
    assert min(ys) <= 0 and max(ys) >= height
    assert len(layout.cells) > 4


def test_angle_controls_the_visual_slope():
    rising = wm.plan(_config(angle=-25.0), 800, 600)
    falling = wm.plan(_config(angle=25.0), 800, 600)
    assert rising.u[1] < 0 < falling.u[1]


def test_huge_stamp_is_rejected_without_allocating_the_mask():
    text = "M" * 80
    cfg = _config(text=text, size_percent=12.0)
    info = wm.analyze(cfg, 10000, 500)
    assert info["ok"]  # the ultra-wide target really fits
    info = wm.analyze(cfg, 10000, 10000)
    assert not info["ok"] and info["reason"] == "text_too_long_for_image"


def test_tiny_image_and_long_text_report_a_reason():
    with pytest.raises(wm.WatermarkError) as excinfo:
        wm.burn(np.zeros((20, 20, 3), np.uint8), _config())
    assert excinfo.value.reason == "too_small"
    long_text = _config(text="M" * 80, size_percent=12.0)
    with pytest.raises(wm.WatermarkError) as excinfo:
        wm.plan(long_text, 320, 240)
    assert excinfo.value.reason == "text_too_long_for_image"


def test_analyze_returns_a_stable_reason():
    info = wm.analyze(_config(), 20, 20)
    assert not info["ok"] and info["reason"] == "too_small"
    assert info["message_key"] == "watermark_too_small"
    ok = wm.analyze(_config(), 640, 480)
    assert ok["ok"] and ok["font_px"] > 0 and ok["cells"] > 0


def test_export_cfg_respects_scope_and_baked_records():
    window = _Window()
    window.watermark_text = "mark"
    assert wm.export_cfg(window, "source") is not None
    assert wm.export_cfg(window, "output") is not None
    assert wm.export_cfg(window, "registered") is None
    window.watermark_source_enabled = False
    assert wm.export_cfg(window, "source") is None
    record = _Record(config={"text": "own"}, initialized=True)
    assert wm.export_cfg(window, "output", record)["text"] == "own"
    baked = _Record(config={"text": "own"}, baked=True)
    assert wm.export_cfg(window, "output", baked) is None


def test_apply_to_records_updates_current_and_selected_only():
    window = _Window()
    current = _Record("current", context=_Context([object()]))
    other = _Record("other", context=_Context([object()]))
    window.output_list.records = [current, other]
    window.raw_images = current.context.frames[0].image
    window.raw_images = [window.raw_images]
    window.source_revision = 0
    changed = wm.apply_to_records(window, _config(), ("output",))
    assert changed >= 1
    assert wm.record_export_cfg(current) is not None
    assert wm.record_export_cfg(other) is None
    # Selecting a historical output explicitly edits only that record.
    selected = _Record("selected")
    window.output_list.records.append(selected)
    wm.apply_to_records(window, _config(text="explicit"), ("output",), selected)
    assert wm.record_export_cfg(selected)["text"] == "explicit"


def test_target_configs_are_independent():
    window = _Window()
    source = wm.WatermarkConfig(text="SOURCE")
    output = wm.WatermarkConfig(text="OUTPUT")
    window.watermark_source_config = source.to_dict()
    window.watermark_output_config = output.to_dict()
    window.watermark_source_enabled = True
    window.watermark_output_enabled = True
    window.watermark_text = "SOURCE"
    assert wm.config_for_target(window, "source").text == "SOURCE"
    assert wm.config_for_target(window, "output").text == "OUTPUT"
    assert wm.export_cfg(window, "source")["text"] == "SOURCE"
    assert wm.export_cfg(window, "output")["text"] == "OUTPUT"
    state = wm.capture_state(window)
    window.watermark_output_config = source.to_dict()
    wm.restore_state(window, state)
    assert wm.config_for_target(window, "output").text == "OUTPUT"


def test_project_round_trip_keeps_independent_target_texts(tmp_path):
    from utils import project_file
    from tests.test_project_file import _make_stack, _window as project_window

    stack_dir = str(tmp_path / "stack")
    paths = _make_stack(stack_dir)
    window = project_window(paths, stack_dir)
    window.watermark_source_config = wm.WatermarkConfig(text="SOURCE").to_dict()
    window.watermark_output_config = wm.WatermarkConfig(text="OUTPUT").to_dict()
    window.watermark_source_enabled = True
    window.watermark_output_enabled = True
    window.watermark_text = "OUTPUT"
    project = str(tmp_path / "targets.ofproj")
    ok, message = project_file.save_project(window, project)
    assert ok, message
    ok, message, state = project_file.validate_project(project)
    assert ok, message
    restored = project_window(paths, stack_dir)
    project_file._apply_settings(restored, state)
    assert wm.config_for_target(restored, "source").text == "SOURCE"
    assert wm.config_for_target(restored, "output").text == "OUTPUT"
    assert wm.config_for_target(restored, "registered") is None


def test_editing_one_scope_keeps_a_separately_configured_record():
    window = _Window()
    window.watermark_source_enabled = True
    window.watermark_output_enabled = False
    record = _Record("r", config={"text": "output-only"}, initialized=True)
    window.output_list.records = [record]
    wm.apply_to_records(window, _config(text="source style"), ("source",),
                        clear_targets=())
    assert wm.record_export_cfg(record)["text"] == "output-only"
    # Explicitly turning the output scope off does clear its records.
    wm.apply_to_records(window, _config(text="source style"), ("source",),
                        selected_record=record, clear_targets=("output",))
    assert wm.record_export_cfg(record) is None


def test_batch_freezes_each_target_config_independently():
    from core.workers import BatchWorker

    worker = BatchWorker([], "custom", "", {
        "watermark": {
            "source": wm.WatermarkConfig(text="SOURCE").to_dict(),
            "registered": wm.WatermarkConfig(text="REGISTERED").to_dict(),
            "output": None,
            "targets": ["source", "registered"],
        }}, use_gpu=False)
    assert worker._stack_watermark("source")["text"] == "SOURCE"
    assert worker._stack_watermark("registered")["text"] == "REGISTERED"
    assert worker._stack_watermark("output") is None


def test_removing_an_inactive_scope_changes_nothing():
    window = _Window()
    before = wm.capture_state(window)
    wm.remove_target(window, "source")
    assert wm.capture_state(window) == before


def test_remove_source_does_not_clear_the_selected_output():
    window = _Window()
    window.watermark_text = "mark"
    record = _Record("r", config={"text": "output"}, initialized=True)
    window.output_list.records = [record]
    wm.remove_target(window, "source", record)
    assert wm.record_export_cfg(record) is not None
    wm.remove_target(window, "output", record)
    assert wm.record_export_cfg(record) is None


def test_current_stack_detection_accepts_excluded_frame_subsets():
    window = _Window()
    window.raw_images = [object(), object(), object()]
    window.image_source_paths = ["/a.tif", "/b.tif", "/c.tif"]
    inside = _Record("inside", context=_Context([object()], paths=["/a.tif", "/c.tif"]))
    outside = _Record("outside", context=_Context([object()], paths=["/z.tif"]))
    reloaded = _Record("reloaded", context=_Context(
        [object()], paths=["/a.tif", "/c.tif"], revision=1))
    assert wm.record_belongs_to_current_stack(window, inside)
    assert not wm.record_belongs_to_current_stack(window, outside)
    # Same paths but a previous load/revision must not be treated as current.
    assert not wm.record_belongs_to_current_stack(window, reloaded)


class _Frame:
    def __init__(self, image):
        self.image = image


class _Context:
    def __init__(self, frames, *, paths=(), revision=0):
        self.frames = [_Frame(image) for image in frames]
        self.source_paths = tuple(paths)
        self.source_revision = revision


def test_new_stack_does_not_inherit_the_previous_text():
    window = _Window()
    window.watermark_text = "old creator"
    window.watermark_registered_enabled = True
    wm.reset_for_new_stack(window)
    assert window.watermark_text == ""
    assert wm.config_from_window(window) is None
    assert not window.watermark_registered_enabled


def test_capture_and_restore_round_trip_including_records():
    window = _Window()
    window.watermark_text = "first"
    window.watermark_output_enabled = True
    record = _Record("r", initialized=True)
    window.output_list.records = [record]
    before = wm.capture_state(window)
    window.watermark_text = "second"
    wm.set_record_config(record, wm.normalize_config({"text": "second"}))
    wm.restore_state(window, before)
    assert window.watermark_text == "first"
    assert wm.record_export_cfg(record) is None


def test_project_round_trip_keeps_watermark_settings(tmp_path):
    from utils import project_file
    from tests.test_project_file import _make_stack, _window as project_window

    stack_dir = str(tmp_path / "stack")
    paths = _make_stack(stack_dir)
    window = project_window(paths, stack_dir)
    window.watermark_text = "© Project"
    window.watermark_font_family = "Arial"
    window.watermark_size_percent = 4.5
    window.watermark_opacity = 0.33
    window.watermark_color = "black"
    window.watermark_angle = 12.0
    window.watermark_density = "dense"
    window.watermark_source_enabled = False
    window.watermark_registered_enabled = True
    window.watermark_output_enabled = True
    project = str(tmp_path / "watermark.ofproj")
    ok, message = project_file.save_project(window, project)
    assert ok, message
    ok, message, state = project_file.validate_project(project)
    assert ok, message
    saved = state["settings"]["watermark"]
    assert saved["text"] == "© Project"
    assert saved["targets"] == ["registered", "output"]

    restored = project_window(paths, stack_dir)
    restored.watermark_text = "stale"
    project_file._apply_settings(restored, state)
    assert restored.watermark_text == "© Project"
    assert restored.watermark_font_family == "Arial"
    assert restored.watermark_opacity == pytest.approx(0.33)
    assert restored.watermark_color == "black"
    assert restored.watermark_registered_enabled is True
    assert restored.watermark_source_enabled is False


def test_corrupt_project_watermark_is_disabled_without_crashing():
    window = _Window()
    window.watermark_text = "old"
    wm.apply_settings(window, {"text": "", "opacity": float("nan"),
                               "targets": ["not-a-target"]})
    assert window.watermark_text == ""
    assert window.watermark_opacity == wm.DEFAULT_OPACITY
    assert not wm.active_targets(window)


@pytest.mark.parametrize('text', ['\u3164', '\u2800', '\u200d'])
def test_invisible_glyphs_cannot_export_an_unmarked_image(text):
    source = np.full((300, 400, 3), 1001, np.uint16)
    with pytest.raises(wm.WatermarkError):
        wm.burn(source, _config(text=text))


def test_spilled_watermark_export_succeeds_when_ram_snapshot_is_unavailable(monkeypatch, tmp_path):
    from core import frame_storage
    from utils.image_utils import imwrite_auto

    source = np.full((256, 384, 3), 1001, np.uint16)
    before = source.copy()
    config = _config(text='Creator')
    expected = wm.burn(source, config)
    monkeypatch.setattr(wm, 'MAX_IN_MEMORY_BURN_BYTES', 0)
    monkeypatch.setattr(frame_storage, '_storage_directory', str(tmp_path))

    def refuse_ram(*args, **kwargs):
        raise MemoryError('RAM snapshot unavailable')

    monkeypatch.setattr(frame_storage, '_try_ram_frame', refuse_ram)
    with frame_storage.scratch_directory():
        result = wm.burn(source, config)
        path = tmp_path / 'watermarked.png'
        assert imwrite_auto(str(path), result)
        exported = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(exported, expected)
    np.testing.assert_array_equal(source, before)
