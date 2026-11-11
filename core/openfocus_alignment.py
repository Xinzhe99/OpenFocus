"""Translation and wide similarity search with reusable Q8.8 warps.

The wide detector uses float32 evaluation order. Plans retain immutable source
identities, never decoded frame stacks.
"""
from dataclasses import dataclass
import hashlib
import math
import os

import numpy as np

from core import resource_policy
from core import _alignment_kernels as _native_kernels
from core.source_frames import resolve_frame

_F32 = np.float32
_WEIGHTS = (12, 64, 37, 2)


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def _geometry(frame, *, internal_float=False):
    shape = tuple(frame.shape)
    dtype = np.dtype(frame.dtype)
    if (len(shape) not in (2, 3) or min(shape[:2]) < 1
            or (len(shape) == 3 and shape[2] not in (1, 3, 4))):
        raise ValueError(f'Unsupported alignment shape: {shape}')
    if (dtype not in (np.dtype('uint8'), np.dtype('uint16'))
            and not (internal_float and dtype in (np.dtype('float32'), np.dtype('float64')))):
        raise ValueError(f'Alignment requires uint8 or uint16'
                         f'{" or normalized internal float32/float64" if internal_float else ""}, got {dtype}')
    return shape, dtype


def _hash_pixels(image, should_cancel):
    _check_cancel(should_cancel)
    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(image)).cast('B'))
    _check_cancel(should_cancel)
    return digest.hexdigest()


def _pixel_digest(image, should_cancel):
    _check_cancel(should_cancel)
    # A render's cached copy of a frame is hashed once: estimation records the
    # digest and applying the transform to those same stored pixels reuses it.
    # Arrays, fresh decodes and plans from an earlier render are always hashed.
    from core import frame_resolution_cache
    return frame_resolution_cache.pixel_digest(
        image, lambda: _hash_pixels(image, should_cancel))


def _source_file_identity(frame):
    while not isinstance(frame, np.ndarray) and hasattr(frame, 'source'):
        frame = frame.source
    return (os.fspath(getattr(frame, 'path', '')), int(getattr(frame, 'page', 0)),
            tuple(getattr(frame, 'fingerprint', ())))


@dataclass(frozen=True)
class SourceFingerprint:
    """Decoded pixel identity plus the descriptor's file identity, if present."""

    shape: tuple
    dtype: str
    sha256: str
    path: str = ''
    page: int = 0
    file_stamp: tuple = ()
    context: str = ''

    def validate_file(self):
        if self.path and self.file_stamp:
            stat = os.stat(self.path)
            if (stat.st_size, stat.st_mtime_ns) != self.file_stamp:
                raise OSError(f'Source changed since alignment: {self.path}')

    def validate_pixels(self, image, should_cancel=None):
        _check_cancel(should_cancel)
        self.validate_file()
        if tuple(image.shape) != self.shape or np.dtype(image.dtype).str != self.dtype:
            raise ValueError('Alignment source shape or dtype changed')
        if _pixel_digest(image, should_cancel) != self.sha256:
            raise ValueError('Alignment source pixels changed')
        self.validate_file()


@dataclass(frozen=True)
class TranslationTransform:
    """Translation in full-resolution pixels; sample at (x-dx, y-dy).

    The previous reference is already aligned to source zero, so pair_dx/dy
    and dx/dy are in that same anchor coordinate system.
    """

    source_index: int
    dx: float
    dy: float
    pair_dx: float
    pair_dy: float
    source: SourceFingerprint

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.dx, self.dy, self.pair_dx, self.pair_dy)):
            raise ValueError('Translation must be finite')

    @property
    def is_identity(self):
        return self.dx == 0.0 and self.dy == 0.0

    @property
    def valid_bounds(self):
        """Half-open (left, top, right, bottom) output rectangle, no allocation.

        Validity is the sampler's geometric in-bounds flag, including its last
        row/column clamp. Pixel darkness is never used to infer validity.
        """
        h, w = self.source.shape[:2]
        def first_at_least(length, shift, value):
            low, high = 0, length
            while low < high:
                middle = (low + high) // 2
                if _F32(middle) - _F32(shift) < _F32(value):
                    low = middle + 1
                else:
                    high = middle
            return low

        left = first_at_least(w, self.dx, 0)
        right = first_at_least(w, self.dx, w)
        top = first_at_least(h, self.dy, 0)
        bottom = first_at_least(h, self.dy, h)
        return (left, top, right, bottom) if left < right and top < bottom else (0, 0, 0, 0)


@dataclass(frozen=True)
class TranslationPlan:
    transforms: tuple
    reference_index: int
    processing_order: tuple
    max_shift_ratio: float
    estimation_bytes: int

    @property
    def reference(self):
        return self.transforms[self.reference_index].source

    def validate_sources(self, sources, *, should_cancel=None):
        _validate_plan_sources(sources, self.transforms, _geometry, should_cancel)


def _validate_plan_sources(sources, transforms, geometry, should_cancel):
    if len(sources) != len(transforms):
        raise ValueError('Alignment source count changed')
    for frame, transform in zip(sources, transforms):
        _check_cancel(should_cancel)
        expected = transform.source
        if isinstance(frame, np.ndarray):
            expected.validate_pixels(frame, should_cancel)
        else:
            if (geometry(frame) != (expected.shape, np.dtype(expected.dtype))
                    or _source_file_identity(frame) != (expected.path, expected.page, expected.file_stamp)):
                raise ValueError('Alignment source descriptor changed')
            expected.validate_file()
            if expected.context and _source_context(frame) != expected.context:
                raise ValueError('Alignment source calibration or transforms changed')
    _check_cancel(should_cancel)


def _alignment_geometry(frame):
    # File-backed source descriptors remain native unsigned integer images.
    # Arrays/memmaps and processors rooted in arrays use the existing internal
    # normalized float domain without adding another accepted file format.
    return _geometry(frame, internal_float=not _source_file_identity(frame)[0])


def _validate_internal_float(image, should_cancel):
    if image.dtype.kind != 'f':
        return
    for row in image:
        _check_cancel(should_cancel)
        if not np.isfinite(row).all() or row.min() < 0 or row.max() > 255:
            raise ValueError('Internal floating alignment pixels must be finite and in [0, 255]')


def _alignment_gray16(image, should_cancel):
    if image.dtype.kind != 'f':
        return _gray16(image, should_cancel)
    _validate_internal_float(image, should_cancel)
    gray = np.empty(image.shape[:2], np.uint16)
    for y, row in enumerate(image):
        _check_cancel(should_cancel)
        # Recover native code values from normalize_fuse_input's 0..255
        # floats before taking the same truncated equal-channel mean.
        take = min(image.shape[2], 3) if image.ndim == 3 else 1
        color = row[:, :take] if image.ndim == 3 else row
        codes = np.rint(color.astype(np.float64) * 257.).astype(np.uint16)
        gray[y] = codes.sum(axis=1, dtype=np.uint32) // take if image.ndim == 3 else codes
    return gray


def translation_resolution_bytes(frame):
    """New RAM needed by apply_translation, excluding already decoded input.

    Includes a same-dtype output and conservative one-row sampler/hash scratch.
    Add this to the source descriptor's frame_resolution_bytes in a producer.
    """
    shape, dtype = _geometry(frame)
    channels = shape[2] if len(shape) == 3 else 1
    return math.prod(shape) * dtype.itemsize + shape[1] * (channels * 64 + 256)


def _decode_bytes(frame):
    if isinstance(frame, np.ndarray):
        return 0
    # The streaming integration supplies this helper; keep the standalone core
    # usable with the pre-streaming SourceFrame implementation as well.
    from core import source_frames
    estimate = getattr(source_frames, 'frame_resolution_bytes', None)
    if estimate is not None:
        return int(estimate(frame))
    shape, dtype = _geometry(frame)
    native = tuple(getattr(frame, 'native_shape', ())) or shape
    encoded = os.path.getsize(frame.path) if getattr(frame, 'path', '') else 0
    return math.prod(native) * dtype.itemsize * 4 + math.prod(shape) * dtype.itemsize + encoded * 2


def estimate_translation_bytes(sources):
    """Conservative new RAM bound independent of stack length.

    Two native uint16 gray pyramids, one aligned gray, decoder scratch, and
    bounded (at most 256 squared pixels) scoring and row conversion scratch.
    """
    if not sources:
        raise ValueError('Alignment requires at least one source')
    shape, _ = _geometry(sources[0])
    h, w = shape[:2]
    for frame in sources:
        if _geometry(frame)[0] != shape:
            raise ValueError('Alignment sources must have equal shapes')
    return (h * w * 10 + max(_decode_bytes(frame) for frame in sources)
            + w * 1024 + 256 * 256 * 128)


def _gray16(image, should_cancel):
    _check_cancel(should_cancel)
    gray = np.empty(image.shape[:2], dtype=np.uint16)
    if _native_kernels.available() and image.ndim in (2, 3):
        kernel = (_native_kernels.gray16_u8 if image.dtype == np.uint8 else
                  _native_kernels.gray16_u16 if image.dtype == np.uint16 else None)
        if kernel is not None:
            if should_cancel is None:
                kernel(image, gray)
            else:
                # Bound native work between cancellation checks without copying rows.
                for start in range(0, len(image), 128):
                    _check_cancel(should_cancel)
                    kernel(image[start:start + 128], gray[start:start + 128])
                _check_cancel(should_cancel)
            return gray
    for y, row in enumerate(image):
        _check_cancel(should_cancel)
        if image.ndim == 2:
            gray[y] = row
        else:
            take = min(image.shape[2], 3)
            gray[y] = row[:, :take].sum(axis=1, dtype=np.uint32) // take
        if image.dtype == np.uint8:
            gray[y] <<= 8
    return gray

def _down_half(src, should_cancel):
    """Separable (12,64,37,2)/115, rounded after EACH axis, edge replicated."""
    _check_cancel(should_cancel)
    if _native_kernels.available() and src.dtype == np.uint16 and src.ndim == 2:
        return _native_kernels.down_half(src)
    h, w = src.shape
    cols = np.arange(0, w, 2)
    out = np.empty(((h + 1) // 2, (w + 1) // 2), dtype=np.uint16)
    for y in range(0, h, 2):
        _check_cancel(should_cancel)
        vertical = np.zeros(w, np.uint32)
        for i, weight in enumerate(_WEIGHTS):
            vertical += src[min(max(y - 1 + i, 0), h - 1)].astype(np.uint32) * weight
        vertical = (vertical + 57) // 115
        value = np.zeros(cols.size, np.uint32)
        for i, weight in enumerate(_WEIGHTS):
            value += vertical[np.clip(cols - 1 + i, 0, w - 1)] * weight
        out[y // 2] = (value + 57) // 115
    return out

def _pyramid(gray, should_cancel):
    levels = [gray]
    for _ in range(6):
        if min(levels[-1].shape) < 16:
            break
        levels.append(_down_half(levels[-1], should_cancel))
    while len(levels) > 1 and min(levels[-1].shape) < 64:
        levels.pop()
    return levels


def _axis_samples(coords, length, *, padding=0):
    """Sampler indices and weights with float32 clamp ordering."""
    coords = np.asarray(coords, dtype=np.float32).copy()
    extent = length + 2 * padding
    invalid = (coords >= _F32(extent)) | (coords < 0)
    coords[coords >= _F32(extent - 1)] = _F32(extent) - _F32(1.0010000467300415)
    np.maximum(coords, _F32(0), out=coords)
    indices = coords.astype(np.int32)
    frac = ((coords - indices.astype(np.float32)) * _F32(256)).astype(np.uint32)
    lo = np.clip(indices - padding, 0, length - 1)
    hi = np.clip(indices + 1 - padding, 0, length - 1)
    return lo, hi, frac, invalid


def _sample_plane(image, x, y, *, padding=0):
    """Bounded crop sampler; uint32 is sufficient for 65535 * 65536."""
    ix, jx, fx, invalid_x = _axis_samples(x, image.shape[1], padding=padding)
    iy, jy, fy, invalid_y = _axis_samples(y, image.shape[0], padding=padding)
    gx, gy = np.uint32(256) - fx, np.uint32(256) - fy
    top = image[iy[:, None], ix].astype(np.uint32) * gx
    top += image[iy[:, None], jx].astype(np.uint32) * fx
    bottom = image[jy[:, None], ix].astype(np.uint32) * gx
    bottom += image[jy[:, None], jx].astype(np.uint32) * fx
    values = (top * gy[:, None] + bottom * fy[:, None]) >> 16
    values[invalid_y[:, None] | invalid_x] = 0
    return values


def _refine_level(a, b, c_dx, c_dy, *, half, step, crop, should_cancel):
    _check_cancel(should_cancel)
    _sample_plane(a, np.zeros(1, np.float32), np.zeros(1, np.float32))
    _check_cancel(should_cancel)
    if (_native_kernels.available() and a.dtype == np.uint16
            and b.dtype == np.uint16 and a.ndim == b.ndim == 2):
        dx, dy, found = _native_kernels.refine_level(
            a, b, _F32(c_dx), _F32(c_dy), _F32(half), _F32(step), int(crop))
        return (float(dx), float(dy)) if found else None
    h, w = a.shape
    padding = int(np.ceil(max(abs(c_dx), abs(c_dy), 1.0) + half)) + 1
    n = min(crop, h, w)
    cy, cx = (h - n) // 2, (w - n) // 2
    reference = a[cy:cy + n, cx:cx + n].astype(np.int64)
    xs = np.arange(cx, cx + n, dtype=np.float32)
    ys = np.arange(cy, cy + n, dtype=np.float32)
    count = int(round(2 * half / step)) | 1
    offsets = (np.arange(count) - (count - 1) // 2) * step
    best = None
    best_score = _F32(np.finfo(np.float32).max)
    # Evaluate dy outermost and dx fastest; ties keep the first candidate.
    for ody in offsets:
        _check_cancel(should_cancel)
        dy = c_dy + ody
        y = ys + padding - _F32(dy)
        for odx in offsets:
            _check_cancel(should_cancel)
            dx = c_dx + odx
            sampled = _sample_plane(b, xs + padding - _F32(dx), y, padding=padding)
            difference = reference - sampled.astype(np.int64)
            mean = float(difference.mean())
            mean_sq = float((difference * difference).mean())
            # _refine_level compares f64 residual std against its f32 incumbent.
            score = float(np.sqrt(max(mean_sq - mean * mean, 0.0)))
            if score < float(best_score):
                best_score = _F32(score)
                best = (float(dx), float(dy))
    return best


def _estimate_pair(reference, current, should_cancel):
    a, b = _pyramid(reference, should_cancel), _pyramid(current, should_cancel)
    dx = dy = 0.0
    first = True
    for level in range(len(a) - 1, -1, -1):
        _check_cancel(should_cancel)
        if min(a[level].shape) < 16:
            continue
        scale = 0.5 ** level
        if first:
            center = _refine_level(a[level], b[level], 0.0, 0.0, half=12.0,
                                   step=1.0, crop=128, should_cancel=should_cancel)
            first = False
        else:
            center = _refine_level(a[level], b[level], dx * scale, dy * scale,
                                   half=1.5, step=0.25, crop=256,
                                   should_cancel=should_cancel)
        dx, dy = center[0] / scale, center[1] / scale
    return dx, dy


def _warp(image, dx, dy, should_cancel):
    if dx == 0.0 and dy == 0.0:
        # Returning borrowed pixels is safe for repeatable immutable producers.
        _check_cancel(should_cancel)
        return image
    _check_cancel(should_cancel)
    # Preserve the row-sampler cancellation probe used by callers that cancel
    # from inside sampling, before the vectorized/native plane path starts.
    _sample_plane(image, np.zeros(1, np.float32), np.zeros(1, np.float32))
    _check_cancel(should_cancel)
    if (_native_kernels.available() and image.dtype == np.uint16
            and image.ndim in (2, 3)):
        return _native_kernels.warp_u16(image, _F32(dx), _F32(dy))
    h, w = image.shape[:2]
    output = np.empty_like(image)
    x = np.arange(w, dtype=np.float32) - _F32(dx)
    y = np.arange(h, dtype=np.float32) - _F32(dy)
    ix, jx, fx, invalid_x = _axis_samples(x, w)
    iy, jy, fy, invalid_y = _axis_samples(y, h)
    gx, gy = np.uint32(256) - fx, np.uint32(256) - fy
    invalid = invalid_y[:, None] | invalid_x[None, :]
    if image.ndim == 2:
        top = image[iy[:, None], ix[None, :]].astype(np.uint32) * gx[None, :]
        top += image[iy[:, None], jx[None, :]].astype(np.uint32) * fx[None, :]
        bottom = image[jy[:, None], ix[None, :]].astype(np.uint32) * gx[None, :]
        bottom += image[jy[:, None], jx[None, :]].astype(np.uint32) * fx[None, :]
        values = (top * gy[:, None] + bottom * fy[:, None]) >> 16
        values[invalid] = 0
        output[...] = values
    else:
        for channel in range(image.shape[2]):
            plane = image[:, :, channel]
            top = plane[iy[:, None], ix[None, :]].astype(np.uint32) * gx[None, :]
            top += plane[iy[:, None], jx[None, :]].astype(np.uint32) * fx[None, :]
            bottom = plane[jy[:, None], ix[None, :]].astype(np.uint32) * gx[None, :]
            bottom += plane[jy[:, None], jx[None, :]].astype(np.uint32) * fx[None, :]
            values = (top * gy[:, None] + bottom * fy[:, None]) >> 16
            values[invalid] = 0
            output[:, :, channel] = values
    _check_cancel(should_cancel)
    return output


def estimate_translation_plan(sources, *, max_shift_ratio=0.05,
                              should_cancel=None, progress_callback=None):
    """Estimate once; retain only immutable transforms and source identities.

    Sources are a sized sequence of arrays or resolve_frame descriptors.
    Reference is source zero; processing order is input order. The previous
    gray reference is already aligned, so each result directly maps to the
    anchor. max_shift_ratio bounds that estimate per axis (width*x, height*y).
    Zero disables estimation and emits identity transforms. Progress is
    callback(completed_frames, total_frames).
    """
    _check_cancel(should_cancel)
    ratio = float(max_shift_ratio)
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('max_shift_ratio must be finite and in [0, 1]')
    needed = estimate_translation_bytes(sources)
    shape, _ = _geometry(sources[0])
    h, w = shape[:2]
    # One sequential owner. No nested executor/native thread pool is created.
    resource_policy.plan_execution(resource_policy.Workload(
        algorithm='openfocus_translation', frames=len(sources), height=h, width=w,
        fixed_bytes=needed, per_worker_bytes=0, outer_tasks=1), user_threads=1)
    transforms = []
    previous = None
    for index, frame in enumerate(sources):
        _check_cancel(should_cancel)
        image = resolve_frame(frame)
        _check_cancel(should_cancel)
        if not isinstance(image, np.ndarray) or _geometry(image) != _geometry(frame):
            raise ValueError('Alignment source metadata does not match decoded pixels')
        path, page, file_stamp = _source_file_identity(frame)
        identity = SourceFingerprint(shape, image.dtype.str,
                                     _pixel_digest(image, should_cancel), path, page, file_stamp)
        identity.validate_file()
        gray = _gray16(image, should_cancel)
        del image
        pair_x = pair_y = 0.0
        if index and ratio:
            pair_x, pair_y = _estimate_pair(previous, gray, should_cancel)
            if abs(pair_x) > w * ratio or abs(pair_y) > h * ratio:
                raise ValueError(f'Translation for frame {index} exceeds max_shift_ratio: '
                                 f'({pair_x}, {pair_y}) full-resolution pixels')
        transform = TranslationTransform(index, pair_x, pair_y, pair_x, pair_y, identity)
        transforms.append(transform)
        previous = _warp(gray, pair_x, pair_y, should_cancel)
        del gray
        if progress_callback is not None:
            progress_callback(index + 1, len(sources))
    _check_cancel(should_cancel)
    return TranslationPlan(tuple(transforms), 0, tuple(range(len(sources))), ratio, needed)


def apply_translation(image, transform, *, should_cancel=None):
    """Validate source identity and apply a reusable full-resolution transform.

    Identity returns the original borrowed array exactly. Other translations
    allocate one same-shape/dtype output; source pixels are never mutated.
    Q8.8 truncates interpolation; out-of-bounds is black. The final source row
    and column use the verified W/H-1.001f clamp, even for integer shifts.
    """
    _check_cancel(should_cancel)
    _geometry(image)
    if not isinstance(transform, TranslationTransform):
        raise TypeError('transform must be a TranslationTransform')
    resource_policy.require_memory(translation_resolution_bytes(image))
    transform.source.validate_pixels(image, should_cancel)
    result = _warp(image, transform.dx, transform.dy, should_cancel)
    _check_cancel(should_cancel)
    return result


F32 = np.float32

_PI_F32 = F32(math.pi)
_180_F32 = F32(180.0)
_HALF_F32 = F32(0.5)
_ONE_F32 = F32(1.0)
_FLIP = np.uint32(0x80000000)
_BIG = F32(1.0e6)

def fast_downsize(img, should_cancel=None):
    """2x2 mean (sum+2)>>2, ceil dimensions and edge tail."""
    h, w = img.shape
    out = np.empty(((h + 1) // 2, (w + 1) // 2), np.uint16)
    xs = np.arange(0, w, 2)
    xe = np.minimum(xs + 1, w - 1)
    for y in range(0, h, 2):
        _check_cancel(should_cancel)
        row = img[y].astype(np.uint32)
        other = img[min(y + 1, h - 1)].astype(np.uint32)
        out[y // 2] = (row[xs] + row[xe] + other[xs] + other[xe] + 2) >> 2
    return out


def build_pyramid(img, min_dim=16, should_cancel=None):
    """CreatePyramid semantics: while min(w,h) >= min_dim: append(half)."""
    lev = [np.ascontiguousarray(img, dtype=np.uint16)]
    while min(lev[-1].shape) >= min_dim:
        _check_cancel(should_cancel)
        lev.append(fast_downsize(lev[-1], should_cancel))
    return lev

def gpu_params(f8):
    """Candidate f8 {tx,ty,cx,cy,rot,magx,magy,bright} -> inverse-map fields.

    All float ops f32 in ctor order. Returns dict of f32 arrays broadcastable
    to the candidate shape.
    """
    f8 = np.asarray(f8, dtype=np.float32)
    single = f8.ndim == 1
    if single:
        f8 = f8[None, :]
    rot_bits = f8[..., 4].view(np.uint32) ^ _FLIP
    rot_flipped = rot_bits.view(np.float32)
    rad = rot_flipped * _PI_F32 / _180_F32
    s = np.sin(rad).astype(np.float32)              # sinf
    c = np.cos(rad).astype(np.float32)              # cosf
    magx = f8[..., 5]
    magy = f8[..., 6]
    tx = f8[..., 0]
    ty = f8[..., 1]
    cx = f8[..., 2]
    cy = f8[..., 3]

    def _div(a, m):
        out = np.full(a.shape, _BIG, dtype=np.float32)
        np.divide(a, m, out=out, where=m != F32(0.0))
        return out

    fields = dict(P08=_div(s, magx), P0C=_div(s, magy),
                  P18=_div(c, magx), P1C=_div(c, magy),
                  tx=tx, ty=ty, cx=cx, cy=cy)
    if single:
        return {k: v[0] for k, v in fields.items()}
    return fields


def map_coords(uv_fields, xs, ys):
    """Detector inverse map using float32 operations.

    xs/ys: f32 sample coordinate grids broadcastable against the candidate
    leading dims. Returns (u, v) f32 arrays of the broadcast shape.
    """
    dx_ = xs - uv_fields["cx"][..., None, None]
    dy_ = ys - uv_fields["cy"][..., None, None]
    v = dy_ * uv_fields["P1C"][..., None, None]
    v = v - dx_ * uv_fields["P08"][..., None, None]
    v = v + uv_fields["ty"][..., None, None]
    u = dx_ * uv_fields["P18"][..., None, None]
    u = u + dy_ * uv_fields["P0C"][..., None, None]
    u = u + uv_fields["tx"][..., None, None]
    return u, v

def enumerate_candidates(cfg8, cnts, steps, M, start=0, stop=None):
    """cfg8: 8 f32 {rng0,rng1,rng2,rng3,rot,magx,magy,bright}.

    Returns (N, f8 (N,8) f32), with dx fastest and rotation slowest.
    """
    n0, n1, n2, n3 = (int(c) for c in cnts)
    N = n0 * n1 * n2 * n3
    if N <= 0:
        return N, None
    i = np.arange(start, N if stop is None else min(stop, N), dtype=np.int64)
    N = len(i)
    # Truncated remainder: cnts may go negative.
    def _crem(ii, n):
        q = np.trunc(ii / n).astype(np.int64)
        return ii - q * n

    t0 = _crem(i, n0).astype(np.float32)
    i = np.trunc(i / n0).astype(np.int64)
    t1 = _crem(i, n1).astype(np.float32)
    i = np.trunc(i / n1).astype(np.int64)
    t2 = _crem(i, n2).astype(np.float32)
    t3 = _crem(np.trunc(i / n2).astype(np.int64), n3).astype(np.float32)

    step0, step1, step2, step3 = (F32(x) for x in steps)
    cfg = np.asarray(cfg8, dtype=np.float32)

    out = np.empty((N, 8), dtype=np.float32)
    out[:, 0] = cfg[0]
    out[:, 1] = cfg[1]

    # dx = 2*(M*(q*step)) + base ; q = ((1-cnt)*0.5f + t)
    q0 = F32(1 - n0) * _HALF_F32 + t0
    d = q0 * step0
    d = d * M
    d = d + d
    out[:, 2] = d + cfg[2]

    q1 = F32(1 - n1) * _HALF_F32 + t1
    d = q1 * step1
    d = d * M
    d = d + d
    out[:, 3] = d + cfg[3]

    # rot / mag: no x2, no M
    q3 = F32(1 - n3) * _HALF_F32 + t3
    out[:, 4] = q3 * step3 + cfg[4]

    q2 = F32(1 - n2) * _HALF_F32 + t2
    mag = q2 * step2 + cfg[5]
    out[:, 5] = mag
    out[:, 6] = mag
    out[:, 7] = _ONE_F32
    return N, out

def _score_chunk(img_a, img_b, xs, ys, cand):
    """img_a: ref u16 level; img_b: test u16 level; cand: (C,8) f32.

    Returns per-candidate (sum i64, sumsq i64, cnt i64) with oob pixels
    excluded (skip semantics), scale = 1.0 -> ref_scaled == ref.

    Candidates whose entire affine u/v bounding box misses the test image
    are returned as cnt=0 without sampling — identical to the in-kernel
    skip-everything outcome, at zero cost.
    """
    fields = gpu_params(cand)
    x_lo, x_hi = np.float32(xs[0]), np.float32(xs[-1])
    y_lo, y_hi = np.float32(ys[0]), np.float32(ys[-1])
    # u = dx_*P18 + dy_*P0C + tx ; dx_ = x - cx, dy_ = y - cy
    a_u, b_u = fields["P18"], fields["P0C"]
    c_u = (fields["tx"] - fields["cx"] * a_u - fields["cy"] * b_u).astype(np.float32)
    a_v, b_v = fields["P1C"], fields["P08"]
    c_v = (fields["ty"] - fields["cy"] * a_v - fields["cx"] * b_v).astype(np.float32)

    def _span(a, b, c):
        lo = (np.minimum(a * x_lo, a * x_hi) + np.minimum(b * y_lo, b * y_hi)
              + c).astype(np.float32)
        hi = (np.maximum(a * x_lo, a * x_hi) + np.maximum(b * y_lo, b * y_hi)
              + c).astype(np.float32)
        return lo, hi

    u_lo, u_hi = _span(a_u, b_u, c_u)
    v_lo, v_hi = _span(a_v, b_v, c_v)
    W = np.float32(img_b.shape[1])
    Hh = np.float32(img_b.shape[0])
    # the sampler clamps [W-1, W) to W-1.001 before flagging; the strict oob
    # domain is u >= W or u < 0 (same v)
    alive = ((u_hi >= np.float32(0.0)) & (u_lo < W)
             & (v_hi >= np.float32(0.0)) & (v_lo < Hh))

    C = cand.shape[0]
    ssum = np.zeros(C, dtype=np.int64)
    ssq = np.zeros(C, dtype=np.int64)
    cnt = np.zeros(C, dtype=np.int64)
    if not alive.any():
        return ssum, ssq, cnt
    idx = np.nonzero(alive)[0]
    u, v = map_coords({k: v[idx] for k, v in fields.items()},
                      xs[None, :], ys[:, None])
    nyy, nxx = len(ys), len(xs)
    flat_u = np.ascontiguousarray(u).reshape(-1)
    flat_v = np.ascontiguousarray(v).reshape(-1)
    t, oob = _bilinear_points(img_b, flat_u, flat_v)
    t = t.reshape(len(idx), nyy, nxx)
    oob = oob.reshape(len(idx), nyy, nxx)
    ref = img_a[ys.astype(np.int64)[:, None], xs.astype(np.int64)[None, :]].astype(np.int64)
    d = ref[None, :, :] - t
    d = np.where(oob, np.int64(0), d)
    cnt[idx] = (~oob).sum(axis=(1, 2))
    ssum[idx] = d.sum(axis=(1, 2))
    ssq[idx] = (d * d).sum(axis=(1, 2))
    return ssum, ssq, cnt


def _decide(ssum, ssq, cnt):
    """Select the lowest float-domain variance with at least two samples.

    A single sample has zero variance and would otherwise win spuriously.
    """
    cnt_f = cnt.astype(np.float32)
    denom = np.maximum(cnt_f, _ONE_F32).astype(np.float64)
    mean_sq = ssq.astype(np.float64) / denom
    mean_sum = ssum.astype(np.float64) / denom
    msq_f = mean_sq.astype(np.float32)
    msu_f = mean_sum.astype(np.float32)
    var_f = msq_f - msu_f * msu_f
    with np.errstate(invalid="ignore"):
        score = np.sqrt(var_f).astype(np.float32)
    ok = (cnt >= 2) & ~np.isnan(score)
    score = np.where(ok, score, np.float32(np.inf))
    win = int(np.argmin(score))          # first minimum wins (strict <)
    if not np.isfinite(score[win]):
        return -1, np.float32(np.inf)
    return win, score[win]

def _div_rot(coef_eff, half_dim):
    """asin(1/halfDim)*coef*180/pi in double, stored f32."""
    hd = np.float64(half_dim)
    d = math.asin(1.0 / hd) * np.float64(coef_eff) * 180.0 / math.pi
    return F32(d)


def _cnt_step(rng, div, grid_half):
    """cnt = min((2*trunc(rng/div))|1, (2*gridHalf)|1)|1; step = 2*rng/(cnt-1).

    The counter math is 32-bit, so 2*n wraps like C int."""
    n = int(np.trunc(F32(rng) / F32(div)))
    n32 = np.int32(n)
    two_n = np.int32(n32 * np.int32(2))
    a = np.int32(two_n | np.int32(1))
    cap = np.int32((2 * grid_half) | 1)
    cnt = a if a < cap else cap
    cnt = int(np.int32(cnt | np.int32(1)))
    if cnt > 1:
        step = F32(2.0) * F32(rng) / F32(cnt - 1)
    else:
        step = F32(0.0)
    return cnt, step


def grid_level_4d(img_a, img_b, base8, ranges4, coef=1.0, grid_half=8,
                  divisor=4, max_rounds=16, verbose=False, tag="", should_cancel=None):
    """Search a single pyramid level.

    base8 is the initial eight-value candidate; ranges4 contains search ranges.

    Returns (f8 (8,) f32, ranges (4,) f32, rounds). f8 = {tx,ty,cx,cy,rot,
    magx,magy,bright}.
    """
    A = np.ascontiguousarray(img_a, dtype=np.uint16)
    B = np.ascontiguousarray(img_b, dtype=np.uint16)
    H, W = A.shape
    coef = F32(coef)
    coef_eff = F32(0.2) if coef == F32(1.0) else F32(0.5)
    half_dim = F32(max(W, H) // 2)
    M = half_dim
    div_shift = F32(coef_eff * _HALF_F32 / M)
    div_mag = F32(coef_eff / M)
    div_rot = _div_rot(coef_eff, half_dim)
    divs = (div_shift, div_shift, div_mag, div_rot)

    # Clamp the full image rectangle to both images.
    x0, y0, x1, y1 = 0, 0, W, H
    x0 = min(max(x0, 0), W); x1 = min(max(x1, 0), W)
    y0 = min(max(y0, 0), H); y1 = min(max(y1, 0), H)
    if x0 > x1: x0 = x1
    if y0 > y1: y0 = y1
    area = (x1 - x0) * (y1 - y0)

    base = np.array(base8, dtype=np.float32)
    rng = [F32(r) for r in ranges4]
    f8 = base.copy()
    rounds = 0

    for rnd in range(max_rounds):
        _check_cancel(should_cancel)
        rounds = rnd + 1
        cnts, steps = [], []
        for k in range(4):
            c, st = _cnt_step(rng[k], divs[k], grid_half)
            cnts.append(c)
            steps.append(st)
        N = cnts[0] * cnts[1] * cnts[2] * cnts[3]
        if verbose:
            print(f"  [{tag}] lvl rnd{rnd}: cnt={cnts} N={N} "
                  f"rng={['%.4g' % float(r) for r in rng]}")
        if N <= 1:
            f8 = base.copy()                   # early out: out = cfg
            break
        if N > 20_000_000:
            raise AlignmentEstimationError('Alignment candidate domain is too large')
        x_step = int(max(1.0, math.sqrt(area / float(divisor))))
        row_step = int(max(1.0, (float(area) / float(divisor)) / float(x_step)))
        xs = np.arange(x0, x1, x_step, dtype=np.int64)
        ys = np.arange(y0, y1, row_step, dtype=np.int64)
        xs_f = xs.astype(np.float32)
        ys_f = ys.astype(np.float32)

        best_score = np.float32(np.inf)
        winner = None
        samples = len(xs) * len(ys)
        chunk = max(1, int(65_536 // max(samples, 1)))
        for lo in range(0, N, chunk):
            _check_cancel(should_cancel)
            _, cand = enumerate_candidates(base, cnts, steps, M, lo, lo + chunk)
            ssum, ssq, cnt = _score_chunk(A, B, xs_f, ys_f, cand)
            win, sc = _decide(ssum, ssq, cnt)
            if win >= 0 and sc < best_score:
                winner, best_score = cand[win].copy(), sc
        if winner is None:
            raise AlignmentEstimationError('Alignment has no eligible overlapping samples')
        f8 = winner
        base = f8.copy()                       # next round base = winner
        for k in range(4):
            if cnts[k] > 1:
                rng[k] = min(F32(steps[k]), F32(rng[k] * _HALF_F32))
    _check_cancel(should_cancel)
    return f8, rng, rounds

def align_pair_wide(ref_u16, test_u16, h_pct=12.0, v_pct=12.0, rot_deg=6.0,
                    mag_pct=6.0, min_dim=64, divisor=100, grid_half=8,
                    max_rounds=24, verbose=False, should_cancel=None, initial_f8=None):
    """Coarse-to-fine 4D search over the shared pyramid.

    Coarsest level: grid_level_4d with the user percentage ranges (same
    fraction semantics as the driver). Finer levels: local refinement with a
    +-1.5 full-res-px-equivalent shift radius (s/maxDim fraction), +-0.5 deg
    rot, +-1.5% mag. Found {tx,ty,cx,cy} slots are carried to the next finer
    level scaled by 2 (level px -> coarser level px); rot/mag carry as-is
    with scale composed multiplicatively and rotation additively.

    Returns (f8 full-res {tx,ty,cx,cy,rot,magx,magy,bright}, rounds_total).
    """
    if min(ref_u16.shape) < 2:
        raise AlignmentEstimationError("Alignment requires at least two pixels per axis")
    A_pyr = build_pyramid(ref_u16, min_dim, should_cancel)
    B_pyr = build_pyramid(test_u16, min_dim, should_cancel)
    n = min(len(A_pyr), len(B_pyr))
    A_pyr, B_pyr = A_pyr[:n], B_pyr[:n]
    while n > 1 and min(A_pyr[n - 1].shape) < min_dim:
        n -= 1
        A_pyr, B_pyr = A_pyr[:n], B_pyr[:n]

    base = np.array(_IDENTITY_F8 if initial_f8 is None else initial_f8, dtype=np.float32)
    fixed_center = h_pct == 0 and v_pct == 0
    if fixed_center:
        # Keep the existing map but rotate and scale around the frame center.
        cx = F32((ref_u16.shape[1] - 1) / 2)
        cy = F32((ref_u16.shape[0] - 1) / 2)
        tx, ty = map_coords(gpu_params(base), cx, cy)
        base[:4] = (tx.item(), ty.item(), cx, cy)
    rounds_total = 0
    for lvl in range(n - 1, -1, -1):
        _check_cancel(should_cancel)
        s = F32(0.5) ** lvl
        A = A_pyr[lvl]
        B = B_pyr[lvl]
        base_lvl = base.copy()
        base_lvl[0] = F32(base[0] * s)   # tx scaled into level px
        base_lvl[1] = F32(base[1] * s)
        base_lvl[2] = F32(base[2] * s)   # cx
        base_lvl[3] = F32(base[3] * s)   # cy
        if lvl == n - 1:
            shift_frac = [F32(h_pct) / F32(100.0), F32(v_pct) / F32(100.0)]
        else:
            # FIX 2026-11-07: the old expression mis-parenthesised max()
            # (max(<int>) -> TypeError) and passed 0.0 into F32();
            # only shift_frac[0] is consumed below.
            shift_frac = [F32(1.5 / max(float(A_pyr[0].shape[1]),
                                        float(A_pyr[0].shape[0])))] * 2
        # Rotation and magnification ranges scale with the level, remaining
        # tiny at coarse levels (no shift aliasing on
        # small textures), full user range at lvl0 where rotation is
        # resolvable. The shift range at fine levels must cover the
        # displacement induced by the level's rot range (joint valley).
        mag_rng = F32(mag_pct) / F32(100.0) * s
        rot_rng = F32(rot_deg) * s
        rot_cov = float(np.sin(np.radians(float(rot_rng)))) * 0.5
        sf = 0.0 if fixed_center else max(float(shift_frac[0]), rot_cov)
        rng = [F32(sf), F32(sf), mag_rng, rot_rng]
        level_grid = grid_half
        if rot_rng == 0 and mag_rng == 0:
            # With translation alone, a coarse step above one pixel can skip
            # the true basin entirely before quarter-pixel refinement starts.
            radius = F32(F32(sf) * F32(max(A.shape) // 2) * F32(2))
            level_grid = max(grid_half, math.ceil(float(radius)))
        f8, _r, rounds = grid_level_4d(A, B, base_lvl, rng, coef=1.0,
                                       grid_half=level_grid, divisor=divisor,
                                       max_rounds=max_rounds, verbose=verbose,
                                       tag=f"wide{lvl}", should_cancel=should_cancel)
        rounds_total += rounds
        base = f8.copy()
        # carry back to full-res coordinates
        base[0] = F32(f8[0] / s)
        base[1] = F32(f8[1] / s)
        base[2] = F32(f8[2] / s)
        base[3] = F32(f8[3] / s)
        if verbose:
            print(f"  [wide] lvl{lvl}: eff=({float(base[0]-base[2]):.2f},"
                  f"{float(base[1]-base[3]):.2f}) rot={float(base[4]):.3f} "
                  f"mag={float(base[5]):.4f}")
    _check_cancel(should_cancel)
    return base, rounds_total


class AlignmentEstimationError(ValueError):
    """The verified detector could not estimate this frame pair."""


def _source_context(frame):
    import dataclasses
    import json

    def plain(value):
        if dataclasses.is_dataclass(value):
            return plain(dataclasses.asdict(value))
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    context = {key: plain(getattr(frame, key, None)) for key in (
        'linear_metadata', 'scale', 'orientation', 'transforms', 'native_shape')}
    if not isinstance(frame, np.ndarray) and hasattr(frame, 'source'):
        context['source'] = _source_context(frame.source)
    return json.dumps(context, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _bilinear_points(image, u, v):
    """Pointwise sampler, bounded by the caller's row/candidate chunk."""
    ix, jx, fx, invalid_x = _axis_samples(u, image.shape[1])
    iy, jy, fy, invalid_y = _axis_samples(v, image.shape[0])
    gx, gy = np.uint32(256) - fx, np.uint32(256) - fy
    floating = image.dtype.kind == 'f'
    sample_dtype = np.float64 if floating else np.uint32
    top = image[iy, ix].astype(sample_dtype) * gx + image[iy, jx].astype(sample_dtype) * fx
    bottom = image[jy, ix].astype(sample_dtype) * gx + image[jy, jx].astype(sample_dtype) * fx
    weighted = top * gy + bottom * fy
    value = weighted / 65536. if floating else weighted >> 16
    invalid = invalid_x | invalid_y
    return np.where(invalid, np.uint32(0), value), invalid


_IDENTITY_F8 = (0., 0., 0., 0., 0., 1., 1., 1.)
ALIGNMENT_PLAN_VERSION = 2


@dataclass(frozen=True)
class AlignmentTransform:
    source_index: int
    f8: tuple
    pair_f8: tuple
    source: SourceFingerprint
    reference_index: int
    rounds: int = 0
    failure: str = ''

    def __post_init__(self):
        if (not isinstance(self.f8, tuple) or not isinstance(self.pair_f8, tuple)
                or len(self.f8) != 8 or len(self.pair_f8) != 8
                or not all(math.isfinite(v) for v in self.f8 + self.pair_f8)
                or self.f8[5] <= 0 or self.f8[6] <= 0 or self.rounds < 0
                or self.f8[7] != 1 or self.pair_f8[7] != 1
                or not isinstance(self.failure, str)):
            raise ValueError('Invalid immutable alignment transform')

    @property
    def is_identity(self):
        tx, ty, cx, cy, rot, mx, my, _ = self.f8
        return tx == cx and ty == cy and rot == 0 and mx == 1 and my == 1

    @property
    def inverse_matrix(self):
        fields = gpu_params(self.f8)
        a, b, c, d = (float(fields[key]) for key in ('P18', 'P0C', 'P08', 'P1C'))
        tx, ty, cx, cy = self.f8[:4]
        return np.array([[a, b, tx - a * cx - b * cy],
                         [-c, d, ty + c * cx - d * cy], [0., 0., 1.]])

    @property
    def valid_bounds(self):
        h, w = self.source.shape[:2]
        if self.is_identity:
            return 0, 0, w, h
        fields = gpu_params(self.f8)
        if self.f8[4] == 0:
            # The inverse map is separable and increasing in each axis. Find
            # its exact half-open pixel bounds using the warp's float32 math.
            def first_at_least(length, axis, threshold):
                low, high = 0, length
                while low < high:
                    middle = (low + high) // 2
                    x = np.float32(middle if axis == 0 else 0)
                    y = np.float32(middle if axis == 1 else 0)
                    coord = map_coords(fields, x, y)[axis].item()
                    if coord < threshold:
                        low = middle + 1
                    else:
                        high = middle
                return low

            left = first_at_least(w, 0, 0)
            right = first_at_least(w, 0, w)
            top = first_at_least(h, 1, 0)
            bottom = first_at_least(h, 1, h)
            return ((left, top, right, bottom) if left < right and top < bottom
                    else (0, 0, 0, 0))
        # An affine image of a rectangle is convex. A centered output
        # rectangle whose four corners are valid is therefore wholly valid.
        def valid(left, top, right, bottom):
            x = np.array([left, right - 1], np.float32)
            y = np.array([top, bottom - 1], np.float32)
            u, v = map_coords(fields, x[None, :], y[:, None])
            return bool(((u >= 0) & (u < w) & (v >= 0) & (v < h)).all())
        low, high = 0, (min(h, w) - 1) // 2
        while low < high:
            inset = (low + high) // 2
            left, top = min(inset, w // 2), min(inset, h // 2)
            right, bottom = w - left, h - top
            if left < right and top < bottom and valid(left, top, right, bottom):
                high = inset
            else:
                low = inset + 1
        left, top = min(low, w // 2), min(low, h // 2)
        return ((left, top, w - left, h - top)
                if left < w - left and top < h - top and valid(left, top, w - left, h - top)
                else (0, 0, 0, 0))


@dataclass(frozen=True)
class AlignmentPlan:
    transforms: tuple
    reference_index: int
    processing_order: tuple
    max_shift_ratio: float
    estimation_bytes: int
    failure_policy: str = 'fail'
    rotation: float = 0.
    magnification: float = 3.

    def __post_init__(self):
        count = len(self.transforms)
        if (not isinstance(self.transforms, tuple) or count == 0
                or not 0 <= self.reference_index < count
                or tuple(sorted(self.processing_order)) != tuple(range(count))
                or self.processing_order[0] != self.reference_index
                or any(t.source_index != i or not 0 <= t.reference_index < count
                       for i, t in enumerate(self.transforms))
                or not math.isfinite(self.max_shift_ratio) or not 0 <= self.max_shift_ratio <= 1
                or self.failure_policy not in ('fail', 'continue')
                or not math.isfinite(self.rotation) or not 0 <= self.rotation <= 180
                or not math.isfinite(self.magnification) or not 0 <= self.magnification < 100
                or self.estimation_bytes < 0):
            raise ValueError('Invalid alignment plan')

    @property
    def reference(self):
        return self.transforms[self.reference_index].source

    def validate_sources(self, sources, *, should_cancel=None):
        _validate_plan_sources(sources, self.transforms, _alignment_geometry, should_cancel)

    def to_dict(self):
        from dataclasses import asdict
        return {'version': ALIGNMENT_PLAN_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data):
        from types import SimpleNamespace
        if data['version'] != ALIGNMENT_PLAN_VERSION:
            raise ValueError('Unsupported alignment plan version')
        transforms = []
        for item in data['transforms']:
            source = item['source']
            shape, dtype = tuple(source['shape']), np.dtype(source['dtype'])
            _alignment_geometry(SimpleNamespace(shape=shape, dtype=dtype, path=source['path']))
            digest = source['sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid source digest')
            fingerprint = SourceFingerprint(shape, dtype.str, digest, source['path'],
                int(source['page']), tuple(source['file_stamp']), source.get('context', ''))
            transforms.append(AlignmentTransform(int(item['source_index']), tuple(item['f8']),
                tuple(item['pair_f8']), fingerprint, int(item['reference_index']),
                int(item['rounds']), item['failure']))
        return cls(tuple(transforms), int(data['reference_index']), tuple(data['processing_order']),
                   float(data['max_shift_ratio']), int(data['estimation_bytes']), data['failure_policy'],
                   float(data['rotation']), float(data['magnification']))


def alignment_resolution_bytes(frame):
    """Warp-only output and row/hash scratch; source decode is charged separately."""
    shape, dtype = _alignment_geometry(frame)
    channels = shape[2] if len(shape) == 3 else 1
    return math.prod(shape) * dtype.itemsize + shape[1] * (channels * 64 + 256)


def estimate_alignment_bytes(sources):
    """Anchor/previous gray, two pyramids, current decode and bounded scoring."""
    if not sources:
        raise ValueError('Alignment requires at least one source')
    h, w = _alignment_geometry(sources[0])[0][:2]
    for source in sources:
        if _alignment_geometry(source)[0][:2] != (h, w):
            raise ValueError('Alignment sources must have equal shapes in both spatial axes')
    return (h * w * 16 + max(_decode_bytes(frame) for frame in sources)
            + w * 1024 + 16 * 1024 * 1024)


def _warp_alignment(image, f8, should_cancel):
    tx, ty, cx, cy, rotation, mx, my, _ = f8
    _check_cancel(should_cancel)
    if tx == cx and ty == cy and rotation == 0 and mx == 1 and my == 1:
        return image
    fields = gpu_params(f8)
    h, w = image.shape[:2]
    output = np.empty_like(image)
    xs = np.arange(w, dtype=np.float32)
    for y in range(h):
        _check_cancel(should_cancel)
        u, v = map_coords(fields, xs, np.float32(y))
        u, v = u.reshape(-1), v.reshape(-1)
        if image.ndim == 2:
            output[y] = _bilinear_points(image, u, v)[0]
        else:
            for channel in range(image.shape[2]):
                output[y, :, channel] = _bilinear_points(image[:, :, channel], u, v)[0]
    _check_cancel(should_cancel)
    return output


def apply_alignment(image, transform, *, should_cancel=None):
    _check_cancel(should_cancel)
    _alignment_geometry(image)
    _validate_internal_float(image, should_cancel)
    if not isinstance(transform, AlignmentTransform):
        raise TypeError('transform must be an AlignmentTransform')
    transform.source.validate_pixels(image, should_cancel)
    if not transform.is_identity:
        resource_policy.require_memory(alignment_resolution_bytes(image))
    return _warp_alignment(image, transform.f8, should_cancel)


def alignment_valid_mask(transform, *, should_cancel=None):
    """Geometric validity, independent of black source pixels and interpolation."""
    h, w = transform.source.shape[:2]
    mask = np.empty((h, w), bool)
    fields = gpu_params(transform.f8)
    xs = np.arange(w, dtype=np.float32)
    for y in range(h):
        _check_cancel(should_cancel)
        u, v = map_coords(fields, xs, np.float32(y))
        mask[y] = ((u >= 0) & (u < w) & (v >= 0) & (v < h)).reshape(-1)
    _check_cancel(should_cancel)
    return mask


def estimate_alignment_plan(sources, *, max_shift_ratio=.05, reference_frame=1,
                            rotation=0., magnification=3., should_cancel=None,
                            progress_callback=None, failure_policy='fail'):
    """Sequential wide search; rotation in degrees, magnification in percent.

    Source one follows the previous aligned reference. Each estimate maps to
    the anchor and starts from the previous absolute transform, so the local
    search can follow cumulative focus breathing across a long stack. A later
    reference anchors two independent chains, one on either side.
    """
    _check_cancel(should_cancel)
    ratio = float(max_shift_ratio)
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('max_shift_ratio must be finite and in [0, 1]')
    rotation, magnification = float(rotation), float(magnification)
    if not math.isfinite(rotation) or not 0 <= rotation <= 180:
        raise ValueError('rotation must be finite and in [0, 180] degrees')
    if not math.isfinite(magnification) or not 0 <= magnification < 100:
        raise ValueError('magnification must be finite and in [0, 100) percent')
    if failure_policy not in ('fail', 'continue'):
        raise ValueError('Invalid alignment failure policy')
    needed = estimate_alignment_bytes(sources)
    if not isinstance(reference_frame, (int, np.integer)) or not 1 <= reference_frame <= len(sources):
        raise ValueError(f'OpenFocus reference frame {reference_frame} is outside the stack of {len(sources)} frames')
    h, w = sources[0].shape[:2]
    resource_policy.plan_execution(resource_policy.Workload(
        algorithm='openfocus_alignment', frames=len(sources), height=h, width=w,
        fixed_bytes=needed, per_worker_bytes=0, outer_tasks=1), user_threads=1)
    ref = reference_frame - 1
    order = (ref, *range(ref - 1, -1, -1), *range(ref + 1, len(sources)))
    transforms = [None] * len(sources)
    previous = anchor = None
    previous_index = ref
    for done, index in enumerate(order):
        _check_cancel(should_cancel)
        frame = sources[index]
        image = resolve_frame(frame)
        _check_cancel(should_cancel)
        if not isinstance(image, np.ndarray) or _alignment_geometry(image) != _alignment_geometry(frame):
            raise ValueError('Alignment source metadata does not match decoded pixels')
        if image.ndim == 3 and image.shape[2] == 4:
            from core.image_loader import _has_transparency, UnsupportedTransparency
            if _has_transparency(image):
                raise UnsupportedTransparency('Registration of transparent images is unsupported')
        path, page, file_stamp = _source_file_identity(frame)
        identity = SourceFingerprint(tuple(image.shape), image.dtype.str,
            _pixel_digest(image, should_cancel), path, page, file_stamp, _source_context(frame))
        identity.validate_file()
        gray = _alignment_gray16(image, should_cancel)
        del image
        if ref > 0 and index == ref + 1:
            previous, previous_index = anchor, ref
        pair = np.array(_IDENTITY_F8, np.float32)
        rounds, failure = 0, ''
        if index != ref and (ratio or rotation or magnification) and not np.array_equal(previous, gray):
            try:
                pair, rounds = align_pair_wide(previous, gray, h_pct=ratio * 100,
                    v_pct=ratio * 100, rot_deg=rotation, mag_pct=magnification,
                    should_cancel=should_cancel, initial_f8=transforms[previous_index].f8)
            except AlignmentEstimationError as error:
                if failure_policy == 'fail':
                    failure = AlignmentEstimationError(f'OpenFocus alignment failed at frame {index + 1}: {error}')
                    failure.frame_index = index
                    failure.completed_transforms = tuple(t for t in transforms if t is not None)
                    raise failure from error
                failure = str(error)
                pair = np.array(_IDENTITY_F8, np.float32)
        total = pair.copy()
        if (not failure and previous_index != ref
                and transforms[previous_index].failure):
            failure = 'reference alignment fell back'
        transform = AlignmentTransform(index, tuple(float(v) for v in total),
            tuple(float(v) for v in pair), identity, previous_index, rounds, failure)
        transforms[index] = transform
        previous = _warp_alignment(gray, total, should_cancel)
        if index == ref and ref > 0:
            anchor = previous
        del gray
        previous_index = index
        if progress_callback is not None:
            progress_callback(done + 1, len(sources))
    _check_cancel(should_cancel)
    return AlignmentPlan(tuple(transforms), ref, order, ratio, needed, failure_policy,
                         rotation, magnification)
