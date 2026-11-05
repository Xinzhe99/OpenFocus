"""Conservative, advisory comparison of neighbouring source frames.

Only almost identical samples are suggested.  A strict per-sample limit keeps
localized focus detail from being hidden by an average over the whole image.
"""

import numpy as np


_MAX_DIFFERENCE = 2.0 / 255.0
_MEAN_DIFFERENCE = 1.0 / 255.0
_BLOCK = 128


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("Stack review cancelled")


def _difference(first, second, cancel):
    if (first.shape != second.shape or first.ndim not in (2, 3)
            or not first.size or first.dtype not in (np.uint8, np.uint16)
            or second.dtype not in (np.uint8, np.uint16)):
        return None
    first_scale = float(np.iinfo(first.dtype).max)
    second_scale = float(np.iinfo(second.dtype).max)
    total = 0.0
    for y in range(0, first.shape[0], _BLOCK):
        for x in range(0, first.shape[1], _BLOCK):
            _check_cancel(cancel)
            region = np.s_[y:y + _BLOCK, x:x + _BLOCK]
            a = first[region].astype(np.float32) / first_scale
            b = second[region].astype(np.float32) / second_scale
            delta = np.abs(a - b)
            if float(delta.max()) > _MAX_DIFFERENCE + 1e-7:
                return None
            total += float(delta.sum(dtype=np.float64))
    mean = total / first.size
    return mean if mean <= _MEAN_DIFFERENCE + 1e-7 else None


def analyze_frames(frames, cancel=None):
    """Return suggestions for nearly identical adjacent frames, never exclusions.

    Each result contains zero-based ``index`` and ``reference_index`` and the
    mean normalized sample ``difference``. Only two input frames and a small
    comparison block are retained at once. The caller owns the input arrays.
    """
    previous = None
    results = []
    for index, frame in enumerate(frames):
        _check_cancel(cancel)
        from core.source_frames import resolve_frame
        frame = resolve_frame(frame)
        if previous is not None:
            difference = _difference(previous, frame, cancel)
            if difference is not None:
                results.append({"index": index, "reference_index": index - 1,
                                "difference": difference})
        previous = frame
    _check_cancel(cancel)
    return results
