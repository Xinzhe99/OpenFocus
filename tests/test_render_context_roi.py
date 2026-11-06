"""ROI outputs keep their own crops, not views that pin the full frames."""
import numpy as np

from core.render_context import build_render_context


def test_roi_render_context_does_not_keep_full_frames_alive():
    """An ROI output's provenance holds copies of the crops, not views that
    would pin every full-size aligned frame while the output exists."""
    full = [np.full((200, 300, 3), 40 * (i + 1), np.uint8) for i in range(3)]
    crop = full[0][50:150, 60:260]
    ctx = build_render_context(full, crop.copy(), filenames=["a", "b", "c"], algorithm="test",
                               roi_rect=(60, 50, 200, 100), roi_mode="crop")
    assert ctx is not None
    for frame, source in zip(ctx.frames, full):
        assert frame.image.shape == (100, 200, 3)
        assert not np.shares_memory(frame.image, source)
        np.testing.assert_array_equal(frame.image, source[50:150, 60:260])
