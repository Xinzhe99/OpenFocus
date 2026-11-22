"""Sample the focus positions produced during OpenFocus fusion."""
from functools import lru_cache
from time import monotonic

import numpy as np

from utils.logging_setup import get_logger


LIVE_PREVIEW_EDGE = 512
_MIN_EMIT_INTERVAL = 0.25
_LOG = get_logger(__name__)


def preview_shape(height, width):
    edge = max(height, width)
    if edge <= LIVE_PREVIEW_EDGE:
        return height, width
    scale = LIVE_PREVIEW_EDGE / edge
    return max(1, round(height * scale)), max(1, round(width * scale))


@lru_cache(maxsize=4)
def _sample_axes(height, width):
    rows, cols = preview_shape(height, width)
    y = (2 * np.arange(rows, dtype=np.intp) + 1) * height // (2 * rows)
    x = (2 * np.arange(cols, dtype=np.intp) + 1) * width // (2 * cols)
    return y[:, None], x[None, :]


def _sample(plane):
    if max(plane.shape) <= LIVE_PREVIEW_EDGE:
        return plane
    return plane[_sample_axes(*plane.shape)]


class LiveFusionPreview:
    """Publish bounded grayscale maps with a fixed scale across the stack."""

    def __init__(self, sink):
        self._sink = sink
        self._best = None
        self._layers = None
        self._seen = -1
        self._blend_seen = -1
        self._next_emit = -float('inf')

    def ready(self):
        return self._sink is not None and monotonic() >= self._next_emit

    def tracks(self):
        return self._sink is not None

    def update_best(self, best, index, total, *, sampled=False):
        """Track layer changes from the cumulative maximum used by A and C."""
        if not self.tracks():
            return
        try:
            scores = best if sampled else _sample(best)
            if self._best is None or index == 0:
                self._best = scores.copy()
                self._layers = np.zeros(scores.shape, dtype=np.uint8)
            else:
                shade = round(255 * index / max(1, total - 1))
                np.copyto(self._layers, np.uint8(shade), where=scores > self._best)
                np.copyto(self._best, scores)
            if self.ready() and index > self._seen:
                self._seen = index
                self._publish(self._layers, index, total)
        except MemoryError:
            self.unavailable()

    def show_positions(self, positions, index, total, maximum, *, sampled=False):
        if not self.ready() or index <= self._seen:
            return
        try:
            if not sampled:
                positions = _sample(positions)
            gray = positions * np.float32(255 / max(1, maximum))
            np.minimum(gray, 255, out=gray)
            np.rint(gray, out=gray)
            self._seen = index
            self._publish(gray.astype(np.uint8), index, total)
        except MemoryError:
            self.unavailable()

    def show_blend(self, depth, index, total):
        """Show how far reconstruction has reached through the depth map."""
        if not self.ready() or index <= self._blend_seen:
            return
        try:
            step = 65535 // (total - 1)
            # Pixels turn white when both bracketing layers have been merged.
            pending = _sample(depth).astype(np.int32)
            np.subtract((index + 1) * step, pending, out=pending)
            np.clip(pending, 0, step, out=pending)
            gray = np.rint(pending * np.float32(255 / step)).astype(np.uint8)
            self._blend_seen = index
            self._publish(gray, index, total, blending=True)
        except MemoryError:
            self.unavailable()

    def _publish(self, gray, index, total, *, blending=False):
        self._sink(gray, index + 1, total, blending)
        self._next_emit = monotonic() + _MIN_EMIT_INTERVAL

    def unavailable(self):
        _LOG.warning('Could not continue displaying the live focus map', exc_info=True)
        self._sink = self._best = self._layers = None
