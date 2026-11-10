"""Record selected desktop actions without collecting user-entered content."""
from utils.logging_setup import log_action


def install_activity_logging(window):
    """Connect controls after initialization, so restored settings are not clicks."""
    for name, control in (('combo_fusion', 'fusion_method'),
                          ('combo_align', 'alignment_method')):
        combo = getattr(window, name, None)
        if combo is not None:
            combo.activated.connect(
                lambda index, widget=combo, kind=control:
                    log_action('ui.action', control=kind, method=widget.itemData(index)))
    for name, control in (('btn_render', 'render'), ('btn_compare_all', 'compare_all'),
                          ('chk_quick_preview', 'quick_preview'),
                          ('chk_exposure_match', 'exposure_match'),
                          ('chk_base_correct', 'base_correct'),
                          ('chk_reverse_order', 'reverse_order'),
                          ('chk_sharpen_output', 'sharpen_output'),
                          ('btn_wipe', 'wipe'), ('btn_preview_roi', 'roi')):
        button = getattr(window, name, None)
        if button is not None:
            button.clicked.connect(
                lambda checked=False, kind=control:
                    log_action('ui.action', control=kind, value=bool(checked)))
    for name, control in (('stack_slider', 'source_frame'), ('result_slider', 'result_frame')):
        slider = getattr(window, name, None)
        if slider is not None:
            slider.sliderReleased.connect(
                lambda widget=slider, kind=control:
                    log_action('ui.action', control=kind, index=widget.value()))
