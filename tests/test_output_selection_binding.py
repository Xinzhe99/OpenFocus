"""Output selection is displayed once per actual row change."""
from types import SimpleNamespace

from PyQt6.QtWidgets import QWidget

from ui.right_panel import bind_right_panel, create_right_panel


def test_bound_output_selection_ignores_repeated_clicks(qapp):
    calls = []
    class Window(QWidget):
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    window = Window()
    components = create_right_panel()
    output = components.output_list
    window.output_manager = SimpleNamespace(
        show_output_context_menu=lambda *args: None,
        sync_output_slider_from_list=lambda row: None,
        display_current_output_row=lambda row: calls.append(row) if row >= 0 else None,
        display_output_image_in_result_view=lambda item: calls.append('clicked'),
        delete_selected_output_images=lambda: None)
    window.source_manager = SimpleNamespace(**{name: lambda *args: None for name in (
        'toggle_stack_review', 'show_source_context_menu', 'sync_slider_from_list',
        'on_item_changed', 'apply_list_order', 'delete_selected_source_images')})
    window.render_manager = SimpleNamespace(start_render=lambda: None)
    bind_right_panel(window, components)
    output.addItems(['A', 'B'])
    output.setCurrentRow(0)
    calls.clear()
    output.setCurrentRow(1)
    output.itemClicked.emit(output.item(1))
    output.itemClicked.emit(output.item(1))
    assert calls == [1]
    output.setCurrentRow(0)
    assert calls == [1, 0]
