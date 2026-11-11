import numpy as np
import pytest

from core.stack_review import analyze_frames


@pytest.mark.parametrize("dtype,scale", [(np.uint8, 1), (np.uint16, 257)])
def test_neighbours_and_unique_focused_patch(dtype, scale):
    rng = np.random.default_rng(31)
    base = rng.integers(20, 180, (40, 50, 3), dtype=np.uint8).astype(dtype) * scale
    near = base + scale
    unique = near.copy()
    unique[10:14, 15:19] = 230 * scale
    result = analyze_frames(iter([base, base.copy(), near, unique]))
    assert [(item["index"], item["reference_index"]) for item in result] == [(1, 0), (2, 1)]
    assert result[0]["difference"] == 0
    assert result[1]["difference"] > 0


def test_tiny_mismatched_and_noncontiguous_frames():
    tiny = np.array([[12]], dtype=np.uint8)
    assert analyze_frames([tiny, tiny.copy()])[0]["index"] == 1
    assert analyze_frames([tiny, np.zeros((2, 1), np.uint8)]) == []
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)[:, ::2]
    assert len(analyze_frames([image, image.copy()])) == 1


def test_cancel_stops_during_comparison():
    calls = iter([False, False, True])
    with pytest.raises(InterruptedError):
        analyze_frames([np.zeros((2, 2), np.uint8)] * 2, lambda: next(calls))


def test_cancel_after_decode_does_not_compare_or_resolve_next(monkeypatch):
    from core import source_frames, stack_review

    cancelled = [False]
    calls = []
    def resolve(frame, should_cancel=None):
        calls.append(frame)
        assert should_cancel is not None
        cancelled[0] = True
        return np.zeros((2, 2), np.uint8)

    monkeypatch.setattr(source_frames, 'resolve_frame', resolve)
    monkeypatch.setattr(stack_review, '_difference', lambda *args: pytest.fail('compared after cancellation'))
    with pytest.raises(InterruptedError, match='Stack review cancelled'):
        analyze_frames([1, 2], lambda: cancelled[0])
    assert calls == [1]


def test_decode_cancellation_is_normalized_but_io_errors_propagate(monkeypatch):
    from core import source_frames
    from core.registration import RegistrationCancelled

    def fail(frame, should_cancel=None):
        raise RegistrationCancelled()

    monkeypatch.setattr(source_frames, 'resolve_frame', fail)
    with pytest.raises(InterruptedError) as error:
        analyze_frames([1])
    assert isinstance(error.value.__cause__, RegistrationCancelled)

    def fail_io(frame, should_cancel=None):
        raise OSError('source unavailable')

    monkeypatch.setattr(source_frames, 'resolve_frame', fail_io)
    with pytest.raises(OSError, match='source unavailable'):
        analyze_frames([1])
