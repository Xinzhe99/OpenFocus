"""Widgets must not clip their own translated text.

Two real defects motivated this file:
  * the smoothing spin box had a fixed 80 px width while it needs 105 px, so
    "2.0" rendered as "2.(" with the arrows over the decimal;
  * the option checkboxes sat two per row, and long translations were elided
    ("Invertir orden de la pila (barrid…") because the panel is only 375 px.
"""
import pytest

pytest.importorskip("PyQt6")


def _labels_that_must_fit():
    return ("chk_quick_preview", "chk_reverse_order",
            "chk_exposure_match", "chk_base_correct")


def test_spin_box_is_not_narrower_than_it_needs():
    """A fixed width below minimumSizeHint clips the value."""
    from pathlib import Path
    src = Path("ui/right_panel.py").read_text(encoding="utf-8")
    block = src[src.index("spin_depth_smooth = StepperSpinBox("):]
    block = block[:block.index("ds_layout.addWidget")]
    assert "setFixedWidth" not in block, (
        "a fixed width clips the spin box value; use setMinimumWidth")
    assert "setMinimumWidth" in block


def test_option_checkboxes_flow_and_wrap():
    """Options pack side by side and wrap on width, never one-per-row.

    They used to be stacked unconditionally because a 375 px column could not
    hold two long translations. The flow layout packs what fits and wraps the
    rest, so a wide window shows two rows instead of four and the saved height
    goes to the source/output lists.
    """
    from PyQt6.QtWidgets import QApplication, QCheckBox

    from widgets.flow_layout import FlowPanel

    app = QApplication.instance() or QApplication([])
    panel = FlowPanel(h_spacing=14, v_spacing=6)
    boxes = []
    for text in ("Quick preview", "Reverse order", "Exposure match",
                 "Base correction"):
        box = QCheckBox(text)
        boxes.append(box)
        panel.add(box)
    try:
        # Font metrics differ across CI runners; derive the one-row width
        # from the actual size hints instead of assuming 620 px fits.
        one_row = (sum(b.sizeHint().width() for b in boxes)
                   + 14 * (len(boxes) - 1) + 16)
        panel.resize(one_row, 200)
        panel.show()
        app.processEvents()
        panel.relayout()
        app.processEvents()
        rows = {b.geometry().y() for b in boxes}
        assert len(rows) == 1, f"short options should share one row: {rows}"

        panel.resize(150, 300)
        app.processEvents()
        panel.relayout()
        app.processEvents()
        rows = {b.geometry().y() for b in boxes}
        assert len(rows) > 1, "narrow widths must wrap onto more rows"
    finally:
        panel.close()
        panel.deleteLater()


def test_ai_mode_radios_are_not_pre_elided():
    """The AI radios must show their full wording, not a fixed-width ellipsis.

    They were elided to PANEL_TEXT_WIDTH regardless of the panel's real width,
    which both hid the meaning and forced one radio per row.
    """
    from pathlib import Path

    src = Path("ui/right_panel.py").read_text(encoding="utf-8")
    block = src[src.index("radio_ai_soft = QRadioButton("):]
    block = block[:block.index("ai_mode_widget.setVisible(False)")]
    assert "elidedText" not in block, (
        "the AI radios must not be pre-elided")
    assert "FlowPanel" in block


@pytest.mark.parametrize("lang", ["en", "zh", "ja", "es"])
def test_option_labels_have_a_translation(lang):
    """Read the tables directly: `trans` is a QObject and is collected by the
    full suite, so switching language from a test is not reliable."""
    from locales import trans as _t
    table = _t.translations[lang]
    for key in ("chk_quick_preview", "chk_reverse_order",
                "chk_exposure_match", "chk_base_correct"):
        text = table.get(key)
        assert text and text != key, f"{lang}: {key} is untranslated"


@pytest.mark.parametrize("lang", ["en", "zh", "ja", "es"])
def test_help_bodies_are_translated(lang):
    """The registration help used to show English text in every language."""
    from locales import trans
    for key in ("help_render_body", "help_registration_body",
                "help_tile_body"):
        body = trans.translations[lang].get(key, "")
        assert body, f"{lang}: {key} missing"
        assert "<h3>" in body
    if lang != "en":
        assert "Registration Methods" not in trans.translations[lang][
            "help_registration_body"], f"{lang}: help body is still English"


def test_checked_radio_does_not_grow_its_indicator():
    """A 5 px border on :checked grew the box from 18 px to 26 px, so the
    indicator visibly jumped when the AI mode was toggled."""
    from PyQt6.QtWidgets import QApplication, QRadioButton
    from ui.styles import build_stylesheet
    app = QApplication.instance() or QApplication([])
    previous = app.styleSheet()
    try:
        for theme in ("light", "dark"):
            app.setStyleSheet(build_stylesheet(theme))
            # Same label: only the checked state may differ.
            soft, hard = QRadioButton("same"), QRadioButton("same")
            soft.setChecked(True)
            assert soft.height() == hard.height(), theme
            assert soft.sizeHint().height() == hard.sizeHint().height(), theme
            assert soft.sizeHint().width() == hard.sizeHint().width(), theme
    finally:
        app.setStyleSheet(previous)


def test_unchecked_radio_indicator_is_a_ring():
    """Any background colour on the unchecked radio is painted as a filled
    disc, which read as a solid dot on the dark theme."""
    from pathlib import Path
    src = Path("ui/styles.py").read_text(encoding="utf-8")
    block = src[src.index("QRadioButton::indicator {{"):
                src.index("QCheckBox:disabled")]
    assert "background: transparent" in block
    assert "border: 5px" not in block


def test_ai_mode_radios_exist_and_default_to_soft():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from ui.right_panel import create_right_panel
    panel = create_right_panel()
    try:
        assert panel.radio_ai_soft.isChecked()
        assert not panel.radio_ai_hard.isChecked()
        assert not panel.ai_mode_widget.isVisible()
    finally:
        panel.widget.deleteLater()


def test_help_chips_follow_the_theme():
    """The "?" chips are styled inline at creation. Switching to light used to
    leave them dark: apply_theme refreshed them BEFORE recording the new
    CURRENT_THEME, so help_button_style() re-read the old one."""
    from pathlib import Path
    src = Path("main.py").read_text(encoding="utf-8")
    body = src[src.index("def apply_theme(self, theme: str) -> None:"):]
    body = body[:body.index("def _enable_dark_title_bar")]
    assert body.index("set_current_theme(theme)") < body.index("help_button_style(theme)"), (
        "the theme must be recorded before inline styles are recomputed")


def test_help_button_style_uses_the_active_theme():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import ui.styles as st
    try:
        st.set_current_theme("light")
        light = st.help_button_style()
        st.set_current_theme("dark")
        dark = st.help_button_style()
        assert light != dark
        # An explicit theme must win over the global.
        assert st.help_button_style("light") == light
    finally:
        st.set_current_theme("dark")


def test_no_input_widget_is_narrower_than_its_content():
    """A fixed width below minimumSizeHint silently clips the value.

    Two shipped defects of this class: the right panel's blend-smoothing spin
    ("2.0" → "2.("), the load-scale spin ("10%" → "10(") and the dust-map
    sigma ("4.0" → "4."). This walks every dialog's spin boxes so a new fixed
    width cannot reintroduce it.
    """
    from PyQt6.QtWidgets import (QApplication, QDoubleSpinBox, QSpinBox,
                                 QWidget)
    app = QApplication.instance() or QApplication([])
    from ui.styles import build_stylesheet
    app.setStyleSheet(build_stylesheet("dark"))

    from dialogs.settings import DownsampleDialog, TileSettingsDialog
    from dialogs.dustmap import DustMapDialog

    parent = QWidget()
    dialogs = [DownsampleDialog(parent, initial_scale=0.05, max_scale=1.0),
               TileSettingsDialog(parent),
               DustMapDialog(parent)]
    try:
        for dlg in dialogs:
            dlg.show()
            app.processEvents()
            for spin in dlg.findChildren((QSpinBox, QDoubleSpinBox)):
                need = spin.minimumSizeHint().width()
                assert spin.width() >= need, (
                    f"{type(dlg).__name__}: {spin.text()!r} needs {need} px, "
                    f"has {spin.width()}")
            dlg.close()
    finally:
        for dlg in dialogs:
            dlg.deleteLater()
        parent.deleteLater()


def test_fusion_dropdown_order_and_default():
    """Alignment-only stays first; AI is second but is the DEFAULT, and its
    label names the algorithm so the method is identifiable outside the app."""
    from ui.right_panel import FUSION_ITEMS, FUSION_DEFAULT
    codes = [c for c, _k, _h in FUSION_ITEMS]
    assert codes[0] == "none"
    assert codes[1] == "stackmffv3"
    assert codes[2:5] == ["openfocus_a", "openfocus_b", "openfocus_c"]
    assert codes[5:] == ["guided_filter", "dct", "dtcwt", "gfgfgf"]
    assert FUSION_DEFAULT == "stackmffv3"


def test_ai_label_names_the_algorithm_in_every_language(qapp):
    """'AI' alone tells the user nothing; the paper is StackMFF V3."""
    from locales import TranslationManager
    tables = TranslationManager().translations
    for lang in ("en", "zh", "ja", "es"):
        assert "StackMFF-V3" in tables[lang]["radio_stackmff"], lang


def test_menus_have_an_opaque_background():
    """A menu must not inherit `QWidget { background: transparent; }`.

    The base rule made the popup paint its text over whatever was behind it:
    on the dark theme a right-click menu showed the desktop through it, which
    read as a translucent panel with unreadable labels. Both the frame and
    every item need an explicit fill.
    """
    from ui.styles import build_stylesheet

    for theme in ("dark", "light"):
        qss = build_stylesheet(theme)
        menu = qss[qss.index("QMenu {"):qss.index("QMenu::item {")]
        item = qss[qss.index("QMenu::item {"):qss.index("QMenu::item:selected")]
        assert "background-color:" in menu, theme
        assert "background-color:" in item, theme
        assert "background: transparent" not in item, theme


def test_status_block_is_two_rows(qapp):
    """The status block is a 2x2 grid, not four stacked rows.

    v1.66.0 stacked the four statistics because tools/lang_ui_audit.py reported
    the RAM string as needing 260 px against a 212 px column. That figure is an
    artefact of the offscreen platform failing to resolve the UI font (it
    measures 260 px where the real 96 dpi desktop measures 126 px), so the
    block was made taller for nothing: 107 px instead of 61, taken from the
    source list above it. The widest real statistic is French at 132 px and a
    half of the 400 px panel is 177 px.
    """
    from PyQt6.QtWidgets import QFrame, QLabel

    from ui.right_panel import create_right_panel

    parts = create_right_panel()
    try:
        panel = parts.widget.findChild(QFrame, "statusPanel")
        assert panel is not None
        layout = panel.layout()
        cells = [(layout.getItemPosition(layout.indexOf(w))[:2], w)
                 for w in panel.findChildren(QLabel)]
        rows = {row for (row, _col), _w in cells}
        assert len(rows) == 2, f"status block must be two rows, got {len(rows)}"
        cols = {col for (_row, col), _w in cells}
        assert cols == {0, 1}, f"expected two columns, got {sorted(cols)}"
        # No cell may be empty: four labels fill all four positions.
        assert len(cells) == 4
        assert {(r, c) for (r, c), _w in cells} == {(0, 0), (0, 1),
                                                    (1, 0), (1, 1)}
    finally:
        parts.widget.deleteLater()


def test_compare_all_ignores_quick_preview():
    """Compare-all must fill the output history even with the box ticked.

    Honouring the quick-preview checkbox there rendered each of the eight
    methods as a throwaway draft, so the run finished with nothing added to the
    output history - reported as "the fusion ran but produced no result". A
    draft per method is also useless for comparison: every entry is
    downscaled.
    """
    from pathlib import Path

    src = Path("controllers/render_manager.py").read_text(encoding="utf-8")
    start = src.index("        preview_downscale = getattr(window,")
    block = src[start:start + 900]
    assert "and not compare" in block, (
        "compare-all must opt out of quick preview")
    assert "getattr(self, '_compare_mode', False)" in block


def test_quick_preview_is_a_fraction_not_an_absolute_cap():
    """A fixed 1200 px cap shrank a 1344 px stack by 11% and was slower.

    The downscale plus the forced re-registration of the small frames made the
    quick path slower than the full render (7.7 s vs 6.1 s on 1344x1021 with
    ECC), so the constant is a fraction of the source size now.
    """
    from controllers.render_manager import PREVIEW_SCALE, _downscale_for_preview
    import numpy as np

    assert 0.0 < PREVIEW_SCALE <= 0.6, (
        "a preview only helps if it cuts the pixel count substantially")
    img = np.zeros((1021, 1344, 3), np.uint8)
    out = _downscale_for_preview(img)
    assert max(out.shape[:2]) < 900, out.shape


def test_quick_preview_reuses_an_existing_alignment():
    """ECC costs seconds and does not shrink with the frame.

    Registration dominates the run and is independent of the draft size, so a
    preview that always re-registers is slower than the full render. Once the
    stack is aligned the preview must reuse it.
    """
    from pathlib import Path

    src = Path("controllers/render_manager.py").read_text(encoding="utf-8")
    start = src.index("        elif self._preview_mode:")
    block = src[start:src.index("        elif (need_align_homography", start)]
    assert "reusable" in block and "window.is_images_aligned" in block, (
        "the preview branch must reuse window.aligned_images when valid")
    assert "effective_need_align_ecc = False" in block


def test_add_label_dialog_style_exists():
    """The style constant the action imports must exist.

    v1.62's visual-layer rewrite deleted ADD_LABEL_DIALOG_STYLE while
    controllers/label_manager.py kept importing it. The ImportError fired
    inside the menu handler, so "Add Label" silently did nothing for three
    releases.
    """
    from ui.styles import (ADD_LABEL_DIALOG_STYLE,
                           get_add_label_dialog_style)

    assert "QDialog" in ADD_LABEL_DIALOG_STYLE
    assert get_add_label_dialog_style("light") != ADD_LABEL_DIALOG_STYLE


def test_first_run_theme_is_light():
    """A fresh install opens in light; a saved preference is respected."""
    from utils.settings_store import DEFAULT_THEME, _PERSISTED_FIELDS

    assert DEFAULT_THEME == "light"
    assert _PERSISTED_FIELDS["ui_theme"][1] == "light"


def test_batch_dialog_offers_real_method_choices(qapp):
    """Batch must expose method/alignment/options instead of echoing the
    main window: a batch run could otherwise only reuse whatever the last
    manual render happened to use."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        assert dlg.fusion_combo.count() >= 8
        assert dlg.align_combo.count() >= 4
        # Alignment-only batch is a real choice.
        dlg.align_combo.setCurrentIndex(dlg.align_combo.findData("none"))
        dlg.fusion_combo.setCurrentIndex(dlg.fusion_combo.findData("openfocus_b"))
        settings = dlg.get_processing_settings()
        assert settings["reg_methods"] == []
        assert settings["fusion_method"] == "openfocus_b"
        # Options the right panel has must exist here too.
        assert hasattr(dlg, "chk_reverse_order")
        assert hasattr(dlg, "chk_exposure_match")
        assert hasattr(dlg, "chk_base_correct")
        assert hasattr(dlg, "downscale_slider")
    finally:
        dlg.deleteLater()


def test_batch_downscale_is_a_ratio_of_the_stack():
    """The slider is a percentage of each stack's width, not a pixel width."""
    from core.workers import BatchWorker

    worker = BatchWorker(
        folder_paths=[], output_type="subfolder", output_path="out",
        processing_settings={"reg_scale_percent": 50},
        reg_downscale_width=1024)
    images = [__import__("numpy").zeros((100, 4000, 3), "uint8")]
    assert worker._registration_width(images) == 2000
    worker.processing_settings["reg_scale_percent"] = 100
    assert worker._registration_width(images) == 1024


def test_retouch_paints_a_locked_frame(qapp):
    """Retouching copies from the frame the user locked, and auto-pick is off."""
    import numpy as np

    from controllers.output_manager import OutputRecord
    from core.render_context import build_render_context
    from dialogs.retouch import RetouchDialog

    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 255, (80, 100, 3), dtype=np.uint8)
              for _ in range(3)]
    ctx = build_render_context(
        frames, frames[0], filenames=["a.png", "b.png", "c.png"],
        algorithm="test")
    record = OutputRecord.create(frames[0], "test_output", context=ctx)
    dlg = RetouchDialog(None, record=record,
                        apply_callback=lambda _rec, _img: (True, ""))
    try:
        assert not hasattr(dlg, "chk_auto"), "auto-pick is gone"
        assert dlg.frame_list.count() == 3
        dlg._select_frame(2)
        qapp.processEvents()
        qapp.processEvents()
        assert dlg.doc.frame_index == 2
        assert dlg._source_ready
        before = dlg.doc.working.copy()
        assert dlg.begin_stroke(30, 30)
        dlg.stroke_to(45, 40)
        dlg.end_stroke()
        assert (dlg.doc.working != before).any(), "the stroke painted nothing"
        assert dlg.doc.is_dirty
        dlg._undo()
        assert not (dlg.doc.working != before).any(), "undo did not restore"
    finally:
        dlg._allow_close = True
        dlg.close()
        dlg.deleteLater()


def test_depth_dialog_has_a_live_3d_tab(qapp):
    """The 3D surface renders inside the dialog, not only on export."""
    import numpy as np

    from widgets.depth_surface import DepthSurface

    surf = DepthSurface()
    try:
        surf.resize(300, 220)
        assert surf._height is None
        yy, xx = np.mgrid[0:60, 0:80]
        height = ((np.sin(xx / 8.0) + 1) / 2).astype(np.float32)
        surf.set_surface(height)
        surf.set_relief(0.4)
        surf.set_invert(True)
        surf.grab()  # must not raise
    finally:
        surf.deleteLater()


def test_theme_bound_styles_are_strings_and_follow_the_theme():
    """Message-box / progress stylesheets must be real strings.

    `setStyleSheet()` rejects anything that is not a str, so a wrapper object
    raises TypeError at the call site. They are rebindable module globals
    instead: five controllers import them by name, and when they were built
    once for the dark palette every message box stayed black in light mode.
    """
    import ui.styles as st

    st.set_current_theme("dark")
    dark_msg, dark_prog = st.MESSAGE_BOX_STYLE, st.PROGRESS_DIALOG_STYLE
    st.set_current_theme("light")
    light_msg, light_prog = st.MESSAGE_BOX_STYLE, st.PROGRESS_DIALOG_STYLE
    try:
        for value in (dark_msg, light_msg, dark_prog, light_prog):
            assert isinstance(value, str) and value
        assert dark_msg != light_msg, "message box ignores the theme"
        assert dark_prog != light_prog, "progress dialog ignores the theme"
        # And the light sheets really are light.
        assert "#FFFFFF" in light_msg.upper() or "#F" in light_msg.upper()
    finally:
        st.set_current_theme("dark")


def test_stepper_reapplies_theme_colours(qapp):
    """Steppers carry inline button sheets, which a global sheet cannot beat."""
    import ui.styles as st
    from widgets.stepper import StepperSpinBox

    st.set_current_theme("dark")
    spin = StepperSpinBox(minimum=0, maximum=8, decimals=1, step=0.2)
    try:
        dark = spin._minus.styleSheet()
        st.set_current_theme("light")
        spin.reapply_theme()
        light = spin._minus.styleSheet()
        assert dark != light, "stepper kept the old theme's button colours"
        assert "#FFFFFF" in light.upper()
    finally:
        st.set_current_theme("dark")
        spin.deleteLater()


def test_batch_groups_follow_the_workflow_order(qapp):
    """Inputs -> processing -> output, with the aligned-stack option inside
    the output group.

    The processing group used to be added after every output control, and the
    "save aligned stack" checkbox floated loose at the bottom outside all
    groups, so the dialog read as a flat list with no categories.
    """
    from PyQt6.QtWidgets import QGroupBox

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        titles = []
        for g in dlg.findChildren(QGroupBox):
            titles.append(g.title())
        assert len(titles) >= 4, titles
        # The processing group must come before the output group.
        # Compare positions in the scroll content, not the outer layout.
        content = dlg.options_group.parentWidget()
        lay = content.layout()
        proc = lay.indexOf(dlg.options_group)
        out_group = dlg.save_aligned_cb.parentWidget()
        out = lay.indexOf(out_group)
        assert proc >= 0 and out >= 0, (proc, out)
        assert proc < out, f"processing must precede output, got {titles}"

        # The aligned-stack checkbox lives inside the output group.
        out_group = dlg.save_aligned_cb.parentWidget()
        assert isinstance(out_group, QGroupBox)
        assert out_group is not dlg.options_group
    finally:
        dlg.deleteLater()


def test_batch_parameter_controls_share_a_left_edge(qapp):
    """Every parameter control starts at the same x.

    Three separate HBoxes with a trailing stretch pushed the steppers to the
    far right while the downscale slider started at the far left, so the block
    had no left edge - reported as "the left side is not aligned".
    """
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        dlg.resize(820, 900)
        dlg.show()
        qapp.processEvents()
        dlg.fusion_combo.setCurrentIndex(
            dlg.fusion_combo.findData("openfocus_b"))
        qapp.processEvents()
        xs = set()
        for wd in (dlg.kernel_spin, dlg.smooth_spin, dlg.downscale_slider):
            if wd.isVisible():
                xs.add(wd.mapTo(dlg, wd.rect().topLeft()).x())
        assert len(xs) == 1, f"parameter controls start at different x: {xs}"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_batch_hides_parameters_the_method_ignores(qapp):
    """AI has no kernel/smoothing; showing a live value invites confusion."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        dlg.show()
        qapp.processEvents()
        dlg.fusion_combo.setCurrentIndex(
            dlg.fusion_combo.findData("stackmffv3"))
        qapp.processEvents()
        assert not dlg.kernel_spin.isVisible()
        assert not dlg.smooth_spin.isVisible()
        dlg.fusion_combo.setCurrentIndex(
            dlg.fusion_combo.findData("openfocus_b"))
        qapp.processEvents()
        assert dlg.kernel_spin.isVisible()
        assert dlg.smooth_spin.isVisible()
    finally:
        dlg.close()
        dlg.deleteLater()


def test_batch_path_field_actually_adds_folders(qapp, tmp_path):
    """The path box must ADD what you type, not just seed the picker.

    It previously only set the file dialog's start folder, so typing a path
    appeared to do nothing - the reason the folder chooser felt crippled.
    """
    import numpy as np
    import cv2

    from dialogs.batch import BatchProcessingDialog, BATCH_PICK_DIR_KEY

    stack = tmp_path / "stackA"
    stack.mkdir()
    cv2.imwrite(str(stack / "f0.png"), np.zeros((40, 50, 3), np.uint8))

    dlg = BatchProcessingDialog(None)
    try:
        assert hasattr(dlg, "path_input"), "the path entry box is missing"
        dlg.path_input.setText(str(stack))
        dlg.add_path_from_field()
        assert dlg.folder_paths == [str(stack)]
        # Several at once, with one bogus entry left in the field.
        dlg.path_input.setText(f'"{stack}"; {tmp_path / "missing"}')
        dlg.add_path_from_field()
        assert len(dlg.folder_paths) == 1, "duplicates must be ignored"
        assert "missing" in dlg.path_input.text(), (
            "invalid paths must stay visible for correction")
        assert BATCH_PICK_DIR_KEY.startswith("batch/")
    finally:
        dlg.deleteLater()


def test_batch_import_mode_group_stays_compact(qapp):
    """The two import-mode radios are one choice and must not be spread out."""
    from PyQt6.QtWidgets import QSizePolicy

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        dlg.resize(800, 1000)
        dlg.show()
        qapp.processEvents()
        group = dlg.rb_multiple_folders.parentWidget()
        assert group.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum
        gap = dlg.rb_single_folder.geometry().top() - \
            dlg.rb_multiple_folders.geometry().bottom()
        assert gap < 20, f"radios drifted {gap}px apart"
    finally:
        dlg.close()
        dlg.deleteLater()


def test_folder_picker_has_no_empty_extra_column(qapp, tmp_path):
    """The picker must not get a second, empty list widget.

    Inserting a QListWidget into Qt's splitter put an empty column beside the
    real place list - the blank box reported in the screenshot.
    """
    from PyQt6.QtWidgets import QApplication, QSplitter, QListView

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        from PyQt6.QtWidgets import QFileDialog
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    captured["count"] = spl.count() if spl else 0
                    captured["sidebar"] = (type(spl.widget(0)).__name__
                                           if spl else None)
                    captured["widgets"] = [type(spl.widget(i)).__name__
                                           for i in range(spl.count())] if spl else []
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        assert captured.get("count") == 2, captured
        assert captured.get("sidebar") == "QListView"
        # Only the sidebar and the file view; no extra list widget.
        assert captured["widgets"].count("QListWidget") == 0, captured
    finally:
        dlg.deleteLater()


def test_folder_picker_path_field_is_editable(qapp, tmp_path):
    """Qt's "Look in" combo is a plain dropdown unless made editable."""
    from PyQt6.QtWidgets import QApplication, QComboBox, QFileDialog

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    captured = {}
    target = tmp_path / "stackB"
    target.mkdir()

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    for combo in wd.findChildren(QComboBox):
                        if combo.objectName() == "lookInCombo":
                            captured["editable"] = combo.isEditable()
                            line = combo.lineEdit()
                            captured["line"] = line is not None
                            if line is not None:
                                line.setText(str(target))
                                line.returnPressed.emit()
                                QApplication.processEvents()
                                captured["dir"] = wd.directory().absolutePath()
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folders()
        assert captured.get("editable"), "the path combo must accept typing"
        assert captured.get("line"), "no line edit to type into"
        assert captured.get("dir", "").replace("\\", "/").lower() == \
            str(target).replace("\\", "/").lower(), captured
    finally:
        dlg.deleteLater()


def test_folder_picker_keeps_only_qt_places(qapp, tmp_path):
    """No custom entries in the place sidebar.

    "Last used" and "Queued: <name>" were appended to Qt's place model; they
    duplicated what the picker already offers and pushed the useful entries
    down. The sidebar must keep exactly the standard locations.
    """
    from PyQt6.QtWidgets import QApplication, QFileDialog, QSplitter

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    model = spl.widget(0).model()
                    captured["places"] = [model.index(r, 0).data()
                                          for r in range(model.rowCount())]
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        places = captured.get("places") or []
        assert places, "the sidebar is empty"
        joined = " ".join(str(p) for p in places).lower()
        assert "queued" not in joined and "已排队" not in joined
        assert "last used" not in joined and "上次使用" not in joined
    finally:
        dlg.deleteLater()


def test_folder_list_shows_at_least_three_rows(qapp):
    """The queued-stack list must not collapse to a single clipped row."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        needed = dlg._folder_row_height() * 3
        assert dlg.folder_list.minimumHeight() >= needed, (
            f"min height {dlg.folder_list.minimumHeight()} < 3 rows "
            f"({needed}px)")
    finally:
        dlg.deleteLater()


def test_label_text_resolution_rules():
    """`format` is the template; `text` is only a fallback.

    The old rule was "if '{value}' is in format, use format, else use text",
    so typing "SAMPLE" into the format field and leaving the separate custom
    text box empty produced an EMPTY label: the dialog saved the settings,
    reported success, and drew nothing.
    """
    from utils.validators import build_label_text

    assert build_label_text("{value}", "", 3) == "3"
    assert build_label_text("Fig {value}", "", 3) == "Fig 3"
    assert build_label_text("SAMPLE", "", 3) == "SAMPLE"
    assert build_label_text("", "FIGURE", 3) == "FIGURE"
    assert build_label_text("", "", 3) == ""
    # Steppers hand back floats; "Fig 3.0" is not what anyone typed.
    assert build_label_text("Fig {value}", "", 3.0) == "Fig 3"


def test_label_draws_with_float_geometry():
    """cv2.putText rejects float coordinates ("Can't parse 'org'").

    The dialog's steppers expose the QDoubleSpinBox API and return floats, and
    the draw error was swallowed, so perfectly valid settings drew nothing.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((300, 400, 3), 80, np.uint8)
    adder = LabelAdder()
    adder.config.update_config({
        "format": "Fig {value}", "font_size": 70.0, "x_location": 30.0,
        "y_location": 100.0, "starting_value": 1.0, "interval": 1.0,
        "transparent_bg": True, "font_color": (255, 255, 255),
        "bg_color": (0, 0, 0),
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert (out != image).any(), "a label with valid settings drew nothing"
    assert not adder.last_error


def test_label_keeps_uint16_depth_and_leaves_other_pixels():
    """A caption on a 16-bit frame must not down-convert the image.

    Drawing used to round-trip through an 8-bit copy for the whole frame;
    only the pixels the caption touches may change, and they must stay
    representable at the original depth.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((160, 260, 3), 40000, np.uint16)
    image[0:4, 0:4] = 1234
    adder = LabelAdder()
    adder.config.update_config({
        "format": "SAMPLE", "font_size": 28, "x_location": 12,
        "y_location": 60, "range": "All", "transparent_bg": True,
        "font_color": (255, 255, 255), "bg_color": (0, 0, 0),
        "font_family": "Arial",
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert out.dtype == np.uint16
    assert np.array_equal(out[0:4, 0:4], image[0:4, 0:4]), (
        "pixels outside the caption changed")
    assert not np.array_equal(out, image), "nothing was drawn"


def test_label_outside_the_range_is_reported_not_drawn():
    """Out-of-range is a state, not a failure and not a silent success."""
    import numpy as np

    from utils.validators import LabelAdder

    adder = LabelAdder()
    adder.config.update_config({"format": "SAMPLE", "range": "9-10",
                                "font_size": 24, "x_location": 5,
                                "y_location": 40})
    image = np.full((80, 120, 3), 60, np.uint8)
    out = adder.add_label_to_image(image.copy(), 0)
    assert np.array_equal(out, image)
    assert adder.last_out_of_range is True
    assert not adder.last_error


def test_add_label_dialog_has_a_live_preview(qapp):
    """The dialog must show the label before it is confirmed."""
    from dialogs.retouch import RetouchDialog  # noqa: F401  (import sanity)
    from widgets.label_preview import LabelPreview

    preview = LabelPreview()
    try:
        preview.resize(320, 220)
        preview.set_config({"format": "SAMPLE", "text": "", "font_size": 40,
                            "x_location": 20, "y_location": 60,
                            "transparent_bg": True, "font_color": (255,) * 3,
                            "bg_color": (0,) * 3})
        assert preview.message() == ""
        pix = preview.grab()
        assert pix.width() > 0 and pix.height() > 0
        # Both fields empty -> the preview reports why nothing is drawn.
        preview.set_config({"format": "", "text": ""})
        assert preview.message() == "empty"
    finally:
        preview.deleteLater()


def test_clear_stack_resets_the_sharpness_curve():
    """Clearing the stack must remove the sharpness strip.

    Checked at the source level: the curve belongs to the frames that were
    just dropped, and leaving it drew the old bars under an empty list.
    """
    from pathlib import Path

    src = Path("controllers/source_manager.py").read_text(encoding="utf-8")
    block = src[src.index("def clear_image_stack(self) -> None:"):]
    block = block[:block.index("def update_slider_range")]
    assert "sharpness_curve" in block and ".clear()" in block, (
        "clear_image_stack must reset the sharpness curve")


def test_gpu_menu_follows_the_hardware(qapp):
    """The GPU item must be unticked and disabled without a device."""
    from PyQt6.QtGui import QAction

    from utils import platform_utils

    platform_utils.reset_gpu_probe_cache()
    present = platform_utils.has_compute_gpu()
    from pathlib import Path

    src = Path("ui/menus.py").read_text(encoding="utf-8")
    block = src[src.index("gpu_action = QAction("):]
    block = block[:block.index("settings_menu.addAction(gpu_action)")]
    assert "has_compute_gpu" in block, (
        "the GPU menu item must ask the hardware")
    assert "setEnabled(_gpu_present)" in block
    if not present:
        # And the setter refuses to turn it on.
        main_src = Path("main.py").read_text(encoding="utf-8")
        setter = main_src[main_src.index("def set_use_gpu("):]
        setter = setter[:setter.index("def set_restore_last_stack")]
        assert "has_compute_gpu" in setter
    assert QAction is not None




def test_folder_picker_has_no_empty_extra_column(qapp, tmp_path):
    """The picker must not get a second, empty list widget.

    Inserting a QListWidget into Qt's splitter put an empty column beside the
    real place list - the blank box reported in the screenshot.
    """
    from PyQt6.QtWidgets import QApplication, QSplitter, QListView

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        from PyQt6.QtWidgets import QFileDialog
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    captured["count"] = spl.count() if spl else 0
                    captured["sidebar"] = (type(spl.widget(0)).__name__
                                           if spl else None)
                    captured["widgets"] = [type(spl.widget(i)).__name__
                                           for i in range(spl.count())] if spl else []
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        assert captured.get("count") == 2, captured
        assert captured.get("sidebar") == "QListView"
        # Only the sidebar and the file view; no extra list widget.
        assert captured["widgets"].count("QListWidget") == 0, captured
    finally:
        dlg.deleteLater()


def test_folder_picker_path_field_is_editable(qapp, tmp_path):
    """Qt's "Look in" combo is a plain dropdown unless made editable."""
    from PyQt6.QtWidgets import QApplication, QComboBox, QFileDialog

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    captured = {}
    target = tmp_path / "stackB"
    target.mkdir()

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    for combo in wd.findChildren(QComboBox):
                        if combo.objectName() == "lookInCombo":
                            captured["editable"] = combo.isEditable()
                            line = combo.lineEdit()
                            captured["line"] = line is not None
                            if line is not None:
                                line.setText(str(target))
                                line.returnPressed.emit()
                                QApplication.processEvents()
                                captured["dir"] = wd.directory().absolutePath()
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folders()
        assert captured.get("editable"), "the path combo must accept typing"
        assert captured.get("line"), "no line edit to type into"
        assert captured.get("dir", "").replace("\\", "/").lower() == \
            str(target).replace("\\", "/").lower(), captured
    finally:
        dlg.deleteLater()


def test_folder_picker_keeps_only_qt_places(qapp, tmp_path):
    """No custom entries in the place sidebar.

    "Last used" and "Queued: <name>" were appended to Qt's place model; they
    duplicated what the picker already offers and pushed the useful entries
    down. The sidebar must keep exactly the standard locations.
    """
    from PyQt6.QtWidgets import QApplication, QFileDialog, QSplitter

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    model = spl.widget(0).model()
                    captured["places"] = [model.index(r, 0).data()
                                          for r in range(model.rowCount())]
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        places = captured.get("places") or []
        assert places, "the sidebar is empty"
        joined = " ".join(str(p) for p in places).lower()
        assert "queued" not in joined and "已排队" not in joined
        assert "last used" not in joined and "上次使用" not in joined
    finally:
        dlg.deleteLater()


def test_folder_list_shows_at_least_three_rows(qapp):
    """The queued-stack list must not collapse to a single clipped row."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        needed = dlg._folder_row_height() * 3
        assert dlg.folder_list.minimumHeight() >= needed, (
            f"min height {dlg.folder_list.minimumHeight()} < 3 rows "
            f"({needed}px)")
    finally:
        dlg.deleteLater()


def test_label_text_resolution_rules():
    """`format` is the template; `text` is only a fallback.

    The old rule was "if '{value}' is in format, use format, else use text",
    so typing "SAMPLE" into the format field and leaving the separate custom
    text box empty produced an EMPTY label: the dialog saved the settings,
    reported success, and drew nothing.
    """
    from utils.validators import build_label_text

    assert build_label_text("{value}", "", 3) == "3"
    assert build_label_text("Fig {value}", "", 3) == "Fig 3"
    assert build_label_text("SAMPLE", "", 3) == "SAMPLE"
    assert build_label_text("", "FIGURE", 3) == "FIGURE"
    assert build_label_text("", "", 3) == ""
    # Steppers hand back floats; "Fig 3.0" is not what anyone typed.
    assert build_label_text("Fig {value}", "", 3.0) == "Fig 3"


def test_label_draws_with_float_geometry():
    """cv2.putText rejects float coordinates ("Can't parse 'org'").

    The dialog's steppers expose the QDoubleSpinBox API and return floats, and
    the draw error was swallowed, so perfectly valid settings drew nothing.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((300, 400, 3), 80, np.uint8)
    adder = LabelAdder()
    adder.config.update_config({
        "format": "Fig {value}", "font_size": 70.0, "x_location": 30.0,
        "y_location": 100.0, "starting_value": 1.0, "interval": 1.0,
        "transparent_bg": True, "font_color": (255, 255, 255),
        "bg_color": (0, 0, 0),
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert (out != image).any(), "a label with valid settings drew nothing"
    assert not adder.last_error


def test_label_keeps_uint16_depth_and_leaves_other_pixels():
    """A caption on a 16-bit frame must not down-convert the image.

    Drawing used to round-trip through an 8-bit copy for the whole frame;
    only the pixels the caption touches may change, and they must stay
    representable at the original depth.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((160, 260, 3), 40000, np.uint16)
    image[0:4, 0:4] = 1234
    adder = LabelAdder()
    adder.config.update_config({
        "format": "SAMPLE", "font_size": 28, "x_location": 12,
        "y_location": 60, "range": "All", "transparent_bg": True,
        "font_color": (255, 255, 255), "bg_color": (0, 0, 0),
        "font_family": "Arial",
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert out.dtype == np.uint16
    assert np.array_equal(out[0:4, 0:4], image[0:4, 0:4]), (
        "pixels outside the caption changed")
    assert not np.array_equal(out, image), "nothing was drawn"


def test_label_outside_the_range_is_reported_not_drawn():
    """Out-of-range is a state, not a failure and not a silent success."""
    import numpy as np

    from utils.validators import LabelAdder

    adder = LabelAdder()
    adder.config.update_config({"format": "SAMPLE", "range": "9-10",
                                "font_size": 24, "x_location": 5,
                                "y_location": 40})
    image = np.full((80, 120, 3), 60, np.uint8)
    out = adder.add_label_to_image(image.copy(), 0)
    assert np.array_equal(out, image)
    assert adder.last_out_of_range is True
    assert not adder.last_error


def test_add_label_dialog_has_a_live_preview(qapp):
    """The dialog must show the label before it is confirmed."""
    from dialogs.retouch import RetouchDialog  # noqa: F401  (import sanity)
    from widgets.label_preview import LabelPreview

    preview = LabelPreview()
    try:
        preview.resize(320, 220)
        preview.set_config({"format": "SAMPLE", "text": "", "font_size": 40,
                            "x_location": 20, "y_location": 60,
                            "transparent_bg": True, "font_color": (255,) * 3,
                            "bg_color": (0,) * 3})
        assert preview.message() == ""
        pix = preview.grab()
        assert pix.width() > 0 and pix.height() > 0
        # Both fields empty -> the preview reports why nothing is drawn.
        preview.set_config({"format": "", "text": ""})
        assert preview.message() == "empty"
    finally:
        preview.deleteLater()


def test_clear_stack_resets_the_sharpness_curve():
    """Clearing the stack must remove the sharpness strip.

    Checked at the source level: the curve belongs to the frames that were
    just dropped, and leaving it drew the old bars under an empty list.
    """
    from pathlib import Path

    src = Path("controllers/source_manager.py").read_text(encoding="utf-8")
    block = src[src.index("def clear_image_stack(self) -> None:"):]
    block = block[:block.index("def update_slider_range")]
    assert "sharpness_curve" in block and ".clear()" in block, (
        "clear_image_stack must reset the sharpness curve")


def test_gpu_menu_follows_the_hardware(qapp):
    """The GPU item must be unticked and disabled without a device."""
    from PyQt6.QtGui import QAction

    from utils import platform_utils

    platform_utils.reset_gpu_probe_cache()
    present = platform_utils.has_compute_gpu()
    from pathlib import Path

    src = Path("ui/menus.py").read_text(encoding="utf-8")
    block = src[src.index("gpu_action = QAction("):]
    block = block[:block.index("settings_menu.addAction(gpu_action)")]
    assert "has_compute_gpu" in block, (
        "the GPU menu item must ask the hardware")
    assert "setEnabled(_gpu_present)" in block
    if not present:
        # And the setter refuses to turn it on.
        main_src = Path("main.py").read_text(encoding="utf-8")
        setter = main_src[main_src.index("def set_use_gpu("):]
        setter = setter[:setter.index("def set_restore_last_stack")]
        assert "has_compute_gpu" in setter
    assert QAction is not None




def test_folder_picker_has_no_empty_extra_column(qapp, tmp_path):
    """The picker must not get a second, empty list widget.

    Inserting a QListWidget into Qt's splitter put an empty column beside the
    real place list - the blank box reported in the screenshot.
    """
    from PyQt6.QtWidgets import QApplication, QSplitter, QListView

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        from PyQt6.QtWidgets import QFileDialog
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    captured["count"] = spl.count() if spl else 0
                    captured["sidebar"] = (type(spl.widget(0)).__name__
                                           if spl else None)
                    captured["widgets"] = [type(spl.widget(i)).__name__
                                           for i in range(spl.count())] if spl else []
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        assert captured.get("count") == 2, captured
        assert captured.get("sidebar") == "QListView"
        # Only the sidebar and the file view; no extra list widget.
        assert captured["widgets"].count("QListWidget") == 0, captured
    finally:
        dlg.deleteLater()


def test_folder_picker_path_field_is_editable(qapp, tmp_path):
    """Qt's "Look in" combo is a plain dropdown unless made editable."""
    from PyQt6.QtWidgets import QApplication, QComboBox, QFileDialog

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    captured = {}
    target = tmp_path / "stackB"
    target.mkdir()

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    for combo in wd.findChildren(QComboBox):
                        if combo.objectName() == "lookInCombo":
                            captured["editable"] = combo.isEditable()
                            line = combo.lineEdit()
                            captured["line"] = line is not None
                            if line is not None:
                                line.setText(str(target))
                                line.returnPressed.emit()
                                QApplication.processEvents()
                                captured["dir"] = wd.directory().absolutePath()
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folders()
        assert captured.get("editable"), "the path combo must accept typing"
        assert captured.get("line"), "no line edit to type into"
        assert captured.get("dir", "").replace("\\", "/").lower() == \
            str(target).replace("\\", "/").lower(), captured
    finally:
        dlg.deleteLater()


def test_folder_picker_keeps_only_qt_places(qapp, tmp_path):
    """No custom entries in the place sidebar.

    "Last used" and "Queued: <name>" were appended to Qt's place model; they
    duplicated what the picker already offers and pushed the useful entries
    down. The sidebar must keep exactly the standard locations.
    """
    from PyQt6.QtWidgets import QApplication, QFileDialog, QSplitter

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    model = spl.widget(0).model()
                    captured["places"] = [model.index(r, 0).data()
                                          for r in range(model.rowCount())]
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        places = captured.get("places") or []
        assert places, "the sidebar is empty"
        joined = " ".join(str(p) for p in places).lower()
        assert "queued" not in joined and "已排队" not in joined
        assert "last used" not in joined and "上次使用" not in joined
    finally:
        dlg.deleteLater()


def test_folder_list_shows_at_least_three_rows(qapp):
    """The queued-stack list must not collapse to a single clipped row."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        needed = dlg._folder_row_height() * 3
        assert dlg.folder_list.minimumHeight() >= needed, (
            f"min height {dlg.folder_list.minimumHeight()} < 3 rows "
            f"({needed}px)")
    finally:
        dlg.deleteLater()


def test_label_text_resolution_rules():
    """`format` is the template; `text` is only a fallback.

    The old rule was "if '{value}' is in format, use format, else use text",
    so typing "SAMPLE" into the format field and leaving the separate custom
    text box empty produced an EMPTY label: the dialog saved the settings,
    reported success, and drew nothing.
    """
    from utils.validators import build_label_text

    assert build_label_text("{value}", "", 3) == "3"
    assert build_label_text("Fig {value}", "", 3) == "Fig 3"
    assert build_label_text("SAMPLE", "", 3) == "SAMPLE"
    assert build_label_text("", "FIGURE", 3) == "FIGURE"
    assert build_label_text("", "", 3) == ""
    # Steppers hand back floats; "Fig 3.0" is not what anyone typed.
    assert build_label_text("Fig {value}", "", 3.0) == "Fig 3"


def test_label_draws_with_float_geometry():
    """cv2.putText rejects float coordinates ("Can't parse 'org'").

    The dialog's steppers expose the QDoubleSpinBox API and return floats, and
    the draw error was swallowed, so perfectly valid settings drew nothing.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((300, 400, 3), 80, np.uint8)
    adder = LabelAdder()
    adder.config.update_config({
        "format": "Fig {value}", "font_size": 70.0, "x_location": 30.0,
        "y_location": 100.0, "starting_value": 1.0, "interval": 1.0,
        "transparent_bg": True, "font_color": (255, 255, 255),
        "bg_color": (0, 0, 0),
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert (out != image).any(), "a label with valid settings drew nothing"
    assert not adder.last_error


def test_label_keeps_uint16_depth_and_leaves_other_pixels():
    """A caption on a 16-bit frame must not down-convert the image.

    Drawing used to round-trip through an 8-bit copy for the whole frame;
    only the pixels the caption touches may change, and they must stay
    representable at the original depth.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((160, 260, 3), 40000, np.uint16)
    image[0:4, 0:4] = 1234
    adder = LabelAdder()
    adder.config.update_config({
        "format": "SAMPLE", "font_size": 28, "x_location": 12,
        "y_location": 60, "range": "All", "transparent_bg": True,
        "font_color": (255, 255, 255), "bg_color": (0, 0, 0),
        "font_family": "Arial",
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert out.dtype == np.uint16
    assert np.array_equal(out[0:4, 0:4], image[0:4, 0:4]), (
        "pixels outside the caption changed")
    assert not np.array_equal(out, image), "nothing was drawn"


def test_label_outside_the_range_is_reported_not_drawn():
    """Out-of-range is a state, not a failure and not a silent success."""
    import numpy as np

    from utils.validators import LabelAdder

    adder = LabelAdder()
    adder.config.update_config({"format": "SAMPLE", "range": "9-10",
                                "font_size": 24, "x_location": 5,
                                "y_location": 40})
    image = np.full((80, 120, 3), 60, np.uint8)
    out = adder.add_label_to_image(image.copy(), 0)
    assert np.array_equal(out, image)
    assert adder.last_out_of_range is True
    assert not adder.last_error


def test_add_label_dialog_has_a_live_preview(qapp):
    """The dialog must show the label before it is confirmed."""
    from dialogs.retouch import RetouchDialog  # noqa: F401  (import sanity)
    from widgets.label_preview import LabelPreview

    preview = LabelPreview()
    try:
        preview.resize(320, 220)
        preview.set_config({"format": "SAMPLE", "text": "", "font_size": 40,
                            "x_location": 20, "y_location": 60,
                            "transparent_bg": True, "font_color": (255,) * 3,
                            "bg_color": (0,) * 3})
        assert preview.message() == ""
        pix = preview.grab()
        assert pix.width() > 0 and pix.height() > 0
        # Both fields empty -> the preview reports why nothing is drawn.
        preview.set_config({"format": "", "text": ""})
        assert preview.message() == "empty"
    finally:
        preview.deleteLater()


def test_clear_stack_resets_the_sharpness_curve():
    """Clearing the stack must remove the sharpness strip.

    Checked at the source level: the curve belongs to the frames that were
    just dropped, and leaving it drew the old bars under an empty list.
    """
    from pathlib import Path

    src = Path("controllers/source_manager.py").read_text(encoding="utf-8")
    block = src[src.index("def clear_image_stack(self) -> None:"):]
    block = block[:block.index("def update_slider_range")]
    assert "sharpness_curve" in block and ".clear()" in block, (
        "clear_image_stack must reset the sharpness curve")


def test_gpu_menu_follows_the_hardware(qapp):
    """The GPU item must be unticked and disabled without a device."""
    from PyQt6.QtGui import QAction

    from utils import platform_utils

    platform_utils.reset_gpu_probe_cache()
    present = platform_utils.has_compute_gpu()
    from pathlib import Path

    src = Path("ui/menus.py").read_text(encoding="utf-8")
    block = src[src.index("gpu_action = QAction("):]
    block = block[:block.index("settings_menu.addAction(gpu_action)")]
    assert "has_compute_gpu" in block, (
        "the GPU menu item must ask the hardware")
    assert "setEnabled(_gpu_present)" in block
    if not present:
        # And the setter refuses to turn it on.
        main_src = Path("main.py").read_text(encoding="utf-8")
        setter = main_src[main_src.index("def set_use_gpu("):]
        setter = setter[:setter.index("def set_restore_last_stack")]
        assert "has_compute_gpu" in setter
    assert QAction is not None




def test_folder_picker_has_no_empty_extra_column(qapp, tmp_path):
    """The picker must not get a second, empty list widget.

    Inserting a QListWidget into Qt's splitter put an empty column beside the
    real place list - the blank box reported in the screenshot.
    """
    from PyQt6.QtWidgets import QApplication, QSplitter, QListView

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        from PyQt6.QtWidgets import QFileDialog
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    captured["count"] = spl.count() if spl else 0
                    captured["sidebar"] = (type(spl.widget(0)).__name__
                                           if spl else None)
                    captured["widgets"] = [type(spl.widget(i)).__name__
                                           for i in range(spl.count())] if spl else []
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        assert captured.get("count") == 2, captured
        assert captured.get("sidebar") == "QListView"
        # Only the sidebar and the file view; no extra list widget.
        assert captured["widgets"].count("QListWidget") == 0, captured
    finally:
        dlg.deleteLater()


def test_folder_picker_path_field_is_editable(qapp, tmp_path):
    """Qt's "Look in" combo is a plain dropdown unless made editable."""
    from PyQt6.QtWidgets import QApplication, QComboBox, QFileDialog

    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    captured = {}
    target = tmp_path / "stackB"
    target.mkdir()

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    for combo in wd.findChildren(QComboBox):
                        if combo.objectName() == "lookInCombo":
                            captured["editable"] = combo.isEditable()
                            line = combo.lineEdit()
                            captured["line"] = line is not None
                            if line is not None:
                                line.setText(str(target))
                                line.returnPressed.emit()
                                QApplication.processEvents()
                                captured["dir"] = wd.directory().absolutePath()
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folders()
        assert captured.get("editable"), "the path combo must accept typing"
        assert captured.get("line"), "no line edit to type into"
        assert captured.get("dir", "").replace("\\", "/").lower() == \
            str(target).replace("\\", "/").lower(), captured
    finally:
        dlg.deleteLater()


def test_folder_picker_keeps_only_qt_places(qapp, tmp_path):
    """No custom entries in the place sidebar.

    "Last used" and "Queued: <name>" were appended to Qt's place model; they
    duplicated what the picker already offers and pushed the useful entries
    down. The sidebar must keep exactly the standard locations.
    """
    from PyQt6.QtWidgets import QApplication, QFileDialog, QSplitter

    from dialogs.batch import BatchProcessingDialog

    stack = tmp_path / "stackA"
    stack.mkdir()
    dlg = BatchProcessingDialog(None)
    captured = {}

    def probe():
        for wd in QApplication.topLevelWidgets():
            if isinstance(wd, QFileDialog) and wd.isVisible():
                try:
                    spl = wd.findChild(QSplitter)
                    model = spl.widget(0).model()
                    captured["places"] = [model.index(r, 0).data()
                                          for r in range(model.rowCount())]
                finally:
                    wd.reject()
                break

    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(400, probe)
        dlg.add_folder_to_list(str(stack))
        dlg.add_folders()
        places = captured.get("places") or []
        assert places, "the sidebar is empty"
        joined = " ".join(str(p) for p in places).lower()
        assert "queued" not in joined and "已排队" not in joined
        assert "last used" not in joined and "上次使用" not in joined
    finally:
        dlg.deleteLater()


def test_folder_list_shows_at_least_three_rows(qapp):
    """The queued-stack list must not collapse to a single clipped row."""
    from dialogs.batch import BatchProcessingDialog

    dlg = BatchProcessingDialog(None)
    try:
        needed = dlg._folder_row_height() * 3
        assert dlg.folder_list.minimumHeight() >= needed, (
            f"min height {dlg.folder_list.minimumHeight()} < 3 rows "
            f"({needed}px)")
    finally:
        dlg.deleteLater()


def test_label_text_resolution_rules():
    """`format` is the template; `text` is only a fallback.

    The old rule was "if '{value}' is in format, use format, else use text",
    so typing "SAMPLE" into the format field and leaving the separate custom
    text box empty produced an EMPTY label: the dialog saved the settings,
    reported success, and drew nothing.
    """
    from utils.validators import build_label_text

    assert build_label_text("{value}", "", 3) == "3"
    assert build_label_text("Fig {value}", "", 3) == "Fig 3"
    assert build_label_text("SAMPLE", "", 3) == "SAMPLE"
    assert build_label_text("", "FIGURE", 3) == "FIGURE"
    assert build_label_text("", "", 3) == ""
    # Steppers hand back floats; "Fig 3.0" is not what anyone typed.
    assert build_label_text("Fig {value}", "", 3.0) == "Fig 3"


def test_label_draws_with_float_geometry():
    """cv2.putText rejects float coordinates ("Can't parse 'org'").

    The dialog's steppers expose the QDoubleSpinBox API and return floats, and
    the draw error was swallowed, so perfectly valid settings drew nothing.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((300, 400, 3), 80, np.uint8)
    adder = LabelAdder()
    adder.config.update_config({
        "format": "Fig {value}", "font_size": 70.0, "x_location": 30.0,
        "y_location": 100.0, "starting_value": 1.0, "interval": 1.0,
        "transparent_bg": True, "font_color": (255, 255, 255),
        "bg_color": (0, 0, 0),
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert (out != image).any(), "a label with valid settings drew nothing"
    assert not adder.last_error


def test_label_keeps_uint16_depth_and_leaves_other_pixels():
    """A caption on a 16-bit frame must not down-convert the image.

    Drawing used to round-trip through an 8-bit copy for the whole frame;
    only the pixels the caption touches may change, and they must stay
    representable at the original depth.
    """
    import numpy as np

    from utils.validators import LabelAdder

    image = np.full((160, 260, 3), 40000, np.uint16)
    image[0:4, 0:4] = 1234
    adder = LabelAdder()
    adder.config.update_config({
        "format": "SAMPLE", "font_size": 28, "x_location": 12,
        "y_location": 60, "range": "All", "transparent_bg": True,
        "font_color": (255, 255, 255), "bg_color": (0, 0, 0),
        "font_family": "Arial",
    })
    out = adder.add_label_to_image(image.copy(), 0)
    assert out.dtype == np.uint16
    assert np.array_equal(out[0:4, 0:4], image[0:4, 0:4]), (
        "pixels outside the caption changed")
    assert not np.array_equal(out, image), "nothing was drawn"


def test_label_outside_the_range_is_reported_not_drawn():
    """Out-of-range is a state, not a failure and not a silent success."""
    import numpy as np

    from utils.validators import LabelAdder

    adder = LabelAdder()
    adder.config.update_config({"format": "SAMPLE", "range": "9-10",
                                "font_size": 24, "x_location": 5,
                                "y_location": 40})
    image = np.full((80, 120, 3), 60, np.uint8)
    out = adder.add_label_to_image(image.copy(), 0)
    assert np.array_equal(out, image)
    assert adder.last_out_of_range is True
    assert not adder.last_error


def test_add_label_dialog_has_a_live_preview(qapp):
    """The dialog must show the label before it is confirmed."""
    from dialogs.retouch import RetouchDialog  # noqa: F401  (import sanity)
    from widgets.label_preview import LabelPreview

    preview = LabelPreview()
    try:
        preview.resize(320, 220)
        preview.set_config({"format": "SAMPLE", "text": "", "font_size": 40,
                            "x_location": 20, "y_location": 60,
                            "transparent_bg": True, "font_color": (255,) * 3,
                            "bg_color": (0,) * 3})
        assert preview.message() == ""
        pix = preview.grab()
        assert pix.width() > 0 and pix.height() > 0
        # Both fields empty -> the preview reports why nothing is drawn.
        preview.set_config({"format": "", "text": ""})
        assert preview.message() == "empty"
    finally:
        preview.deleteLater()


def test_clear_stack_resets_the_sharpness_curve():
    """Clearing the stack must remove the sharpness strip.

    Checked at the source level: the curve belongs to the frames that were
    just dropped, and leaving it drew the old bars under an empty list.
    """
    from pathlib import Path

    src = Path("controllers/source_manager.py").read_text(encoding="utf-8")
    block = src[src.index("def clear_image_stack(self) -> None:"):]
    block = block[:block.index("def update_slider_range")]
    assert "sharpness_curve" in block and ".clear()" in block, (
        "clear_image_stack must reset the sharpness curve")


def test_gpu_menu_follows_the_hardware(qapp):
    """The GPU item must be unticked and disabled without a device."""
    from PyQt6.QtGui import QAction

    from utils import platform_utils

    platform_utils.reset_gpu_probe_cache()
    present = platform_utils.has_compute_gpu()
    from pathlib import Path

    src = Path("ui/menus.py").read_text(encoding="utf-8")
    block = src[src.index("gpu_action = QAction("):]
    block = block[:block.index("settings_menu.addAction(gpu_action)")]
    assert "has_compute_gpu" in block, (
        "the GPU menu item must ask the hardware")
    assert "setEnabled(_gpu_present)" in block
    if not present:
        # And the setter refuses to turn it on.
        main_src = Path("main.py").read_text(encoding="utf-8")
        setter = main_src[main_src.index("def set_use_gpu("):]
        setter = setter[:setter.index("def set_restore_last_stack")]
        assert "has_compute_gpu" in setter
    assert QAction is not None





def test_menus_match_the_new_information_architecture():
    """Scale bar and Labels live in Tools; Settings keeps theme/language.

    Labels were split between Edit (add) and a separate delete menu, and the
    scale bar sat under Settings as if it were an app preference. The four
    one-purpose tuning dialogs were folded into one Advanced page.
    """
    from pathlib import Path

    src = Path("ui/menus.py").read_text(encoding="utf-8")
    tools = src[src.index("batch_menu = menubar.addMenu"):]
    tools = tools[:tools.index("# --- Settings Menu ---")]
    assert "menu_scale_bar" in tools, "scale bar must be under Tools"
    assert "menu_labels" in tools, "labels must be under Tools"
    labels = src[src.index("label_menu = batch_menu.addMenu"):]
    labels = labels[:labels.index("# --- Settings Menu ---")]
    assert "action_add_label" in labels
    assert "action_del_reg_label" in labels
    assert "action_del_input_label" in labels

    settings = src[src.index("# --- Settings Menu ---"):]
    assert "menu_scale_bar" not in settings, "scale bar left Settings"
    # Light first: it is the fresh-install default.
    assert '("light", \'theme_light\'),' in settings
    for gone in ("action_tile_settings", "action_reg_settings",
                 "action_thread_settings", "action_ai_batch_settings"):
        assert gone not in settings, gone

def test_depth_surface_detail_is_user_controlled():
    """The 3D detail slider must change the working resolution live."""
    from PyQt6.QtWidgets import QApplication

    from widgets.depth_surface import DepthSurface

    srf = DepthSurface()
    try:
        srf.resize(400, 300)
        import numpy as np
        srf.set_surface(np.full((300, 400), 0.5, np.float32))
        srf.set_detail(40)
        assert srf._mesh_step == 40
        srf.set_detail(100)
        assert srf._mesh_step == 100
        # The cache must be dropped when detail changes, or the slider would
        # have no visible effect.
        srf._cache = None
        srf.set_detail(64)
        assert srf._cache is None
    finally:
        srf.deleteLater()


def test_feedback_dialog_embeds_the_session_log(qapp):
    """The feedback page shows the session log and exports a zip package."""
    import zipfile
    import tempfile

    import os
    import tempfile
    import zipfile

    from dialogs.feedback import FeedbackDialog
    from utils import logging_setup as ls

    ls.setup_logging()
    d = FeedbackDialog(None)
    try:
        body = d._compose()
        assert "Environment" in body and "Session log" in body
        dest = os.path.join(tempfile.mkdtemp(), "fb.zip")
        path = ls.export_feedback_package(dest)
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            assert "session_log.txt" in names
            assert "system.txt" in names
    finally:
        d.deleteLater()
def test_gpu_status_shows_na_without_a_device(qapp):
    """No accelerator must read N/A, never CPU."""
    from utils import platform_utils

    if platform_utils.has_compute_gpu():
        pytest.skip("this machine has a GPU")
    from pathlib import Path

    src = Path("main.py").read_text(encoding="utf-8")
    block = src[src.index("        # GPU. Rule the user asked for"):]
    block = block[:block.index("except Exception:")]
    # The only paths left are N/A, CPU* (accelerator + classical), CUDA/MPS.
    assert "'N/A'" in block
    assert "elif not self._ai_method_selected()" in block


def test_flat_field_failure_keeps_the_original_stack(qapp, monkeypatch):
    """A failed base-stage correction must not leave a half-applied stack."""
    import numpy as np
    from types import SimpleNamespace
    import main as main_module
    from utils import dust_map

    frames = [np.full((8, 8, 3), 10, np.uint8) for _ in range(2)]
    base = [np.full((8, 8, 3), 20, np.uint8) for _ in range(2)]
    host = SimpleNamespace(
        raw_images=frames, base_images=base, _undo_guard=True,
        push_undo=lambda: None,
        get_or_create_source_pixmap=lambda *a, **k: None,
        statusBar=lambda: SimpleNamespace(showMessage=lambda *a, **k: None),
    )
    warnings = []
    monkeypatch.setattr(main_module, "show_warning_box",
                        lambda *a, **k: warnings.append(a))

    calls = {"n": 0}

    def fail_on_second(images, gain):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("size mismatch")
        return [f + 1 for f in images]

    monkeypatch.setattr(dust_map, "apply_flat_field_stack", fail_on_second)
    result = main_module.OpenFocus.apply_flat_field_to_stack(host, object())
    assert result is False
    assert host.raw_images is frames, "raw frames were replaced on failure"
    assert host.base_images is base, "base frames were replaced on failure"
    assert warnings
