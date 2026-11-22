"""Small color composites for the desktop fusion progress display."""
from functools import lru_cache
from time import monotonic

import cv2
import numpy as np

from utils.logging_setup import get_logger


LIVE_PREVIEW_EDGE = 512
_MIN_EMIT_INTERVAL = 0.25
_LOG = get_logger(__name__)


@lru_cache(maxsize=4)
def _sample_axes(height, width):
    scale = LIVE_PREVIEW_EDGE / max(height, width)
    rows, cols = max(1, round(height * scale)), max(1, round(width * scale))
    y = (2 * np.arange(rows, dtype=np.intp) + 1) * height // (2 * rows)
    x = (2 * np.arange(cols, dtype=np.intp) + 1) * width // (2 * cols)
    return y[:, None], x[None, :]


def _sample(pixels):
    height, width = pixels.shape[:2]
    if max(height, width) <= LIVE_PREVIEW_EDGE:
        return pixels
    # Gather only preview pixels, including from strided or mapped sources.
    return pixels[_sample_axes(height, width)]


def sample_live_frame(pixels):
    """Keep only a bounded color sample when handing a frame to a consumer."""
    try:
        sample = _sample(pixels)
        return sample.copy(order='C') if sample is pixels else sample
    except MemoryError:
        _LOG.warning('Could not sample a frame for the live fusion preview', exc_info=True)
        return None


class LiveFusionPreview:
    """Keep the color with the strongest local contrast at each preview pixel.

    The ordered fusion consumer lends prepared pixels for one call only.
    Keep a fixed sampling grid and no source references or frame history.
    """

    def __init__(self, sink, *, linear_metadata=None):
        self._sink = sink
        self._linear_metadata = linear_metadata
        self._best = None
        self._bgr = None
        self._take = None
        self._done = 0
        self._next_emit = -float('inf')

    def _sample_bgr(self, pixels, rgb, source_u8):
        pixels = _sample(pixels)
        if rgb:
            pixels = pixels[..., ::-1]
        if self._linear_metadata is not None:
            # Keep camera-linear precision in the composite. Develop only the
            # published image, using the same display transform as the result.
            if pixels.dtype == np.uint8:
                pixels = pixels.astype(np.uint16) * np.uint16(257)
            elif pixels.dtype.kind == 'f':
                pixels = np.rint(np.clip(pixels, 0, 255) * 257).astype(np.uint16)
            elif pixels.dtype != np.uint16:
                pixels = pixels.astype(np.uint16)
        elif pixels.dtype.kind == 'f':
            # Fusion floats are expressed in 0–255, unlike display floats.
            pixels = cv2.convertScaleAbs(np.clip(pixels, 0, 255))
        elif pixels.dtype != np.uint8:
            # Resize/conversion paths represent native u8 as v << 8.
            pixels = cv2.convertScaleAbs(pixels, alpha=1 / (256 if source_u8 else 257))
        if pixels.ndim == 2 or pixels.shape[2] == 1:
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        return pixels

    def accumulate(self, pixels, index, *, rgb=False, source_u8=False):
        """Add a new frame to the composite, ignoring replay after GPU fallback."""
        if self._sink is None or index < self._done:
            return False
        try:
            bgr = self._sample_bgr(pixels, rgb, source_u8)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            contrast = cv2.Laplacian(gray, cv2.CV_32F, borderType=cv2.BORDER_REPLICATE)
            np.abs(contrast, out=contrast)
            score = cv2.blur(contrast, (5, 5), borderType=cv2.BORDER_REPLICATE)
            if self._bgr is None:
                self._bgr = bgr.copy(order='C')
                self._best = score
                self._take = np.empty(score.shape, dtype=bool)
            else:
                np.greater(score, self._best, out=self._take)
                np.maximum(self._best, score, out=self._best)
                np.copyto(self._bgr, bgr, where=self._take[..., None])
        except (MemoryError, cv2.error):
            self._disable()
            return False
        self._done = index + 1
        return True

    def publish(self, total):
        """Publish the current composite after the caller processes the frame."""
        if self._sink is None or monotonic() < self._next_emit:
            return
        try:
            bgr = self._bgr
            if self._linear_metadata is not None:
                from utils.linear_dng import develop_linear
                bgr = cv2.convertScaleAbs(develop_linear(bgr, self._linear_metadata), alpha=1 / 257)
            self._sink(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self._done, total)
            self._next_emit = monotonic() + _MIN_EMIT_INTERVAL
        except (MemoryError, OSError, ValueError, cv2.error, np.linalg.LinAlgError):
            self._disable()

    def push(self, pixels, index, total, *, rgb=False, source_u8=False):
        if self.accumulate(pixels, index, rgb=rgb, source_u8=source_u8):
            self.publish(total)

    def _disable(self):
        _LOG.warning('Could not continue the live fusion preview', exc_info=True)
        self._sink = self._best = self._bgr = self._take = None
