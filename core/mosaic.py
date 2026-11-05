"""Planar microscope mosaics with explicit placement and bounded rendering."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from core.frame_storage import create_frame, require_disk_space, storage_directory
from core.resource_policy import require_memory
from utils.image_utils import thumbnail_bgr


@dataclass(frozen=True)
class MosaicFrame:
    name: str
    image: np.ndarray
    preview: np.ndarray
    source_path: str = ''


@dataclass(frozen=True)
class MosaicAlignment:
    transforms: tuple
    connected: tuple
    matches: tuple


def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def make_frame(name, image, source_path='', preview_size=1400):
    """Prepare one thumbnail without converting an entire 16-bit image."""
    if image is None or image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Mosaic images must contain 8-bit or 16-bit pixels')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError('Mosaic images must be grayscale, BGR or BGRA')
    if min(image.shape[:2]) < 1:
        raise ValueError('Mosaic images cannot be empty')
    # thumbnail_bgr composites RGBA before resizing; resize first to keep
    # transparent, high-resolution inputs bounded as well.
    if image.ndim == 3 and image.shape[2] == 4:
        height, width = image.shape[:2]
        scale = min(1.0, preview_size / max(height, width))
        small = cv2.resize(image, (max(1, round(width * scale)),
                                   max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else image
        preview = thumbnail_bgr(small, preview_size)
    else:
        preview = thumbnail_bgr(image, preview_size)
    return MosaicFrame(str(name), image, preview, str(source_path))


def initial_transforms(frames):
    """Place images in input order with room to drag or auto-align them."""
    x = 0.0
    result = []
    for frame in frames:
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 2] = x
        result.append(matrix)
        x += frame.image.shape[1] * 0.8
    return result


def grid_transforms(frames, rows, cols, *, order='byrow',
                    direction='unidirectional', start_corner='topleft', overlap_pct=20):
    """Place capture-order frames in a declared grid without decoding pixels.

    Each column/row uses its largest actual tile dimension. Changing a grid
    declaration is separate from applying it, so callers retain manual edits.
    """
    if rows < 1 or cols < 1 or rows * cols != len(frames):
        raise ValueError('The grid must contain exactly one cell per image')
    if order not in ('byrow', 'bycolumn'):
        raise ValueError('Invalid capture order')
    if direction not in ('unidirectional', 'zigzag'):
        raise ValueError('Invalid capture direction')
    if start_corner not in ('topleft', 'topright', 'bottomleft', 'bottomright'):
        raise ValueError('Invalid start corner')
    if not math.isfinite(overlap_pct) or not 0 <= overlap_pct < 100:
        raise ValueError('Overlap must be between 0 and 100 percent')
    row_order = list(range(rows))
    col_order = list(range(cols))
    if start_corner.startswith('bottom'):
        row_order.reverse()
    if start_corner.endswith('right'):
        col_order.reverse()
    cells = []
    major, minor = (row_order, col_order) if order == 'byrow' else (col_order, row_order)
    for index, outer in enumerate(major):
        inner = minor[::-1] if direction == 'zigzag' and index % 2 else minor
        cells.extend((outer, value) if order == 'byrow' else (value, outer)
                     for value in inner)
    widths, heights = [0] * cols, [0] * rows
    for frame, (row, col) in zip(frames, cells):
        height, width = frame.image.shape[:2]
        widths[col] = max(widths[col], width)
        heights[row] = max(heights[row], height)
    spacing = 1 - overlap_pct / 100
    xs = np.cumsum([0] + widths[:-1]) * spacing
    ys = np.cumsum([0] + heights[:-1]) * spacing
    result = []
    for row, col in cells:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, 2] = (xs[col], ys[row])
        result.append(matrix)
    return result


def _features(frame, detector):
    gray = cv2.cvtColor(frame.preview, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 6:
        return None
    scale = np.array([frame.image.shape[1] / gray.shape[1],
                      frame.image.shape[0] / gray.shape[0]], dtype=np.float64)
    points = np.array([item.pt for item in keypoints], dtype=np.float64) * scale
    return points, descriptors, float(max(scale))


def _pair_match(first, second, first_features, second_features):
    if first_features is None or second_features is None:
        return None
    points_a, descriptors_a, scale_a = first_features
    points_b, descriptors_b, scale_b = second_features
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    neighbors = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    good = [pair[0] for pair in neighbors if len(pair) == 2
            and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 6:
        return None
    source = np.float32([points_b[item.queryIdx] for item in good])
    target = np.float32([points_a[item.trainIdx] for item in good])
    threshold = max(3.0, 3.0 * max(scale_a, scale_b))
    affine, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=threshold,
        maxIters=4000, confidence=0.995, refineIters=10)
    if affine is None or mask is None:
        return None
    selected = mask.ravel().astype(bool)
    count = int(selected.sum())
    if count < 6 or count / len(good) < 0.3:
        return None
    scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
    if not 0.5 <= scale <= 2.0:
        return None
    predicted = source[selected] @ affine[:, :2].T + affine[:, 2]
    residual = float(np.median(np.linalg.norm(predicted - target[selected], axis=1)))
    # A handful of points in one tiny patch is not enough to establish a tile.
    spread = np.ptp(source[selected], axis=0)
    if max(spread) < max(8.0, min(second.image.shape[:2]) * 0.02):
        return None
    transform = np.eye(3, dtype=np.float64)
    transform[:2] = affine
    return transform, count, residual


def auto_align(frames, transforms=None, should_cancel=None, progress=None):
    """Fit similarity transforms from shared features; leave disconnected tiles editable."""
    frames = list(frames)
    if not frames:
        return MosaicAlignment((), (), ())
    original = [np.array(matrix, dtype=np.float64, copy=True) for matrix in
                (transforms if transforms is not None else initial_transforms(frames))]
    if len(original) != len(frames):
        raise ValueError('Each mosaic image needs one placement')
    detector = cv2.SIFT_create(nfeatures=6000)
    features = []
    total = len(frames) + len(frames) * (len(frames) - 1) // 2
    done = 0
    if progress is not None:
        progress(done, total)
    for frame in frames:
        _cancel(should_cancel)
        features.append(_features(frame, detector))
        done += 1
        if progress is not None:
            progress(done, total)
    edges = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            _cancel(should_cancel)
            match = _pair_match(frames[i], frames[j], features[i], features[j])
            if match is not None:
                matrix, inliers, residual = match
                edges.append((i, j, matrix, inliers, residual))
            done += 1
            if progress is not None:
                progress(done, total)
    edges.sort(key=lambda item: (-item[3], item[4], item[0], item[1]))
    result = [None] * len(frames)
    connected = [False] * len(frames)
    for root in range(len(frames)):
        if result[root] is not None:
            continue
        result[root] = original[root]
        connected[root] = root == 0
        changed = True
        while changed:
            changed = False
            for i, j, matrix, _inliers, _residual in edges:
                if result[i] is not None and result[j] is None:
                    result[j] = result[i] @ matrix
                    connected[j] = connected[i]
                    changed = True
                elif result[j] is not None and result[i] is None:
                    result[i] = result[j] @ np.linalg.inv(matrix)
                    connected[i] = connected[j]
                    changed = True
    _cancel(should_cancel)
    return MosaicAlignment(tuple(result), tuple(connected),
                           tuple((i, j, inliers, residual)
                                 for i, j, _matrix, inliers, residual in edges))


def _bounds(frame, transform):
    height, width = frame.image.shape[:2]
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1],
                        [width, height, 1]], dtype=np.float64).T
    projected = transform @ corners
    if not np.all(np.isfinite(projected)) or not np.allclose(projected[2], 1):
        raise ValueError('Invalid mosaic placement')
    return (float(projected[0].min()), float(projected[1].min()),
            float(projected[0].max()), float(projected[1].max()))


def canvas_bounds(frames, transforms):
    if not frames or len(frames) != len(transforms):
        raise ValueError('Each mosaic image needs one placement')
    bounds = [_bounds(frame, matrix) for frame, matrix in zip(frames, transforms)]
    left = math.floor(min(item[0] for item in bounds))
    top = math.floor(min(item[1] for item in bounds))
    right = math.ceil(max(item[2] for item in bounds))
    bottom = math.ceil(max(item[3] for item in bounds))
    if right <= left or bottom <= top:
        raise ValueError('Mosaic canvas is empty')
    return left, top, right, bottom


def render_mosaic(frames, transforms, *, feather=64, tile_size=512,
                  should_cancel=None, progress=None):
    """Render a mosaic tile by tile into a temporary file-backed array."""
    frames = list(frames)
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in transforms]
    left, top, right, bottom = canvas_bounds(frames, matrices)
    width, height = right - left, bottom - top
    dtype = np.uint16 if any(frame.image.dtype == np.uint16 for frame in frames) else np.uint8
    has_alpha = any(frame.image.ndim == 3 and frame.image.shape[2] == 4 for frame in frames)
    channels = 4 if has_alpha else 3
    required_disk = width * height * channels * np.dtype(dtype).itemsize
    directory = storage_directory()
    require_disk_space(required_disk + 64 * 1024 ** 2, directory)
    tile_size = max(32, min(1024, int(tile_size)))
    require_memory(tile_size * tile_size * 160)
    boxes = [_bounds(frame, matrix) for frame, matrix in zip(frames, matrices)]
    inverses = [np.linalg.inv(matrix) for matrix in matrices]
    result = create_frame((height, width, channels), dtype, directory=directory)
    xs, ys = range(0, width, tile_size), range(0, height, tile_size)
    total = len(xs) * len(ys)
    done = 0
    if progress is not None:
        progress(0, total)
    for y in ys:
        for x in xs:
            _cancel(should_cancel)
            th, tw = min(tile_size, height - y), min(tile_size, width - x)
            yy, xx = np.mgrid[top + y:top + y + th, left + x:left + x + tw].astype(np.float32)
            acc = np.zeros((th, tw, 3), np.float32)
            weights = np.zeros((th, tw), np.float32)
            coverage = np.zeros((th, tw), np.float32) if has_alpha else None
            for frame, box, inverse in zip(frames, boxes, inverses):
                _cancel(should_cancel)
                if box[2] <= left + x or box[0] >= left + x + tw or box[3] <= top + y or box[1] >= top + y + th:
                    continue
                source = frame.image
                sh, sw = source.shape[:2]
                mx = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
                my = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]
                valid = (mx >= 0) & (my >= 0) & (mx < sw) & (my < sh)
                if not np.any(valid):
                    continue
                x0 = max(0, int(np.floor(mx[valid].min())) - 1)
                y0 = max(0, int(np.floor(my[valid].min())) - 1)
                x1 = min(sw, int(np.ceil(mx[valid].max())) + 2)
                y1 = min(sh, int(np.ceil(my[valid].max())) + 2)
                sample = cv2.remap(source[y0:y1, x0:x1],
                                   (mx - x0).astype(np.float32),
                                   (my - y0).astype(np.float32),
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if sample.ndim == 2:
                    sample = np.repeat(sample[:, :, None], 3, axis=2)
                if sample.shape[2] == 1:
                    sample = np.repeat(sample, 3, axis=2)
                maximum = float(np.iinfo(source.dtype).max)
                alpha = sample[:, :, 3].astype(np.float32) / maximum if sample.shape[2] == 4 else valid.astype(np.float32)
                alpha *= valid
                distance = np.minimum(np.minimum(mx + 0.5, sw - 0.5 - mx),
                                      np.minimum(my + 0.5, sh - 0.5 - my))
                # A valid remap can extend half a pixel beyond the final
                # pixel centre; it must still contribute at the image edge.
                weight = np.clip(np.maximum(distance, 0.5) / max(1, feather), 0, 1) * alpha
                colors = sample[:, :, :3].astype(np.float32)
                if dtype == np.uint16 and source.dtype == np.uint8:
                    colors *= 257.0
                acc += colors * weight[:, :, None]
                weights += weight
                if coverage is not None:
                    coverage += (1.0 - coverage) * alpha
            np.divide(acc, weights[:, :, None], out=acc, where=weights[:, :, None] > 0)
            result[y:y + th, x:x + tw, :3] = np.rint(np.clip(acc, 0, np.iinfo(dtype).max)).astype(dtype)
            if coverage is not None:
                result[y:y + th, x:x + tw, 3] = np.rint(coverage * np.iinfo(dtype).max).astype(dtype)
            done += 1
            if progress is not None:
                progress(done, total)
    _cancel(should_cancel)
    result.flush()
    return result, (left, top)
