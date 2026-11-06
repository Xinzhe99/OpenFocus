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


@dataclass(frozen=True)
class _ControlPoints:
    transform: np.ndarray
    source: np.ndarray
    target: np.ndarray
    tolerance: float


def _project(matrix, points):
    return points @ matrix[:2, :2].T + matrix[:2, 2]


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
    forward = [pair[0] for pair in neighbors if len(pair) == 2
               and pair[0].distance < 0.75 * pair[1].distance]
    if len(forward) < 12:
        return None
    # Only forward candidates need a reciprocal query. Searching every
    # descriptor again was the dominant cost on the real microscope sample.
    candidates = sorted({match.trainIdx for match in forward})
    reverse = matcher.knnMatch(descriptors_a[candidates], descriptors_b, k=2)
    reciprocal = {candidates[pair[0].queryIdx]: pair[0].trainIdx for pair in reverse
                  if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}
    good = [match for match in forward if reciprocal.get(match.trainIdx) == match.queryIdx]
    if len(good) < 12:
        return None
    source = np.float64([points_b[item.queryIdx] for item in good])
    target = np.float64([points_a[item.trainIdx] for item in good])
    threshold = max(2.0, 2.0 * max(scale_a, scale_b))
    affine, mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=threshold,
        maxIters=6000, confidence=0.999, refineIters=20)
    if affine is None or mask is None:
        return None
    selected = mask.ravel().astype(bool)
    if selected.sum() < 12 or selected.mean() < 0.3:
        return None
    scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
    if not 0.5 <= scale <= 2.0:
        return None
    source, target = source[selected], target[selected]
    # Require support across both axes: a few points along an edge cannot
    # determine magnification reliably outside that edge.
    for points, frame in ((source, second), (target, first)):
        if np.any(np.ptp(points, axis=0) < max(8.0, min(frame.image.shape[:2]) * 0.02)):
            return None
    # Keep the graph small even for large, highly textured stacks. Spatial
    # ordering avoids retaining only the strongest features in one patch.
    if len(source) > 512:
        order = np.lexsort((source[:, 0], source[:, 1]))
        keep = order[np.linspace(0, len(order) - 1, 512, dtype=int)]
        source, target = source[keep], target[keep]
    transform = np.eye(3, dtype=np.float64)
    transform[:2] = affine
    return _ControlPoints(transform, source, target, max(scale_a, scale_b))


def _optimize_component(frames, result, nodes, edges, should_cancel):
    """Joint robust control-point fit, with one exact anchor per component.

    Similarities retain isotropic magnification and rotation without the
    unconstrained perspective extrapolation of a homography fit to a thin
    microscope overlap. All graph edges participate, including loop closures.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsmr

    root = nodes[0]
    slots = {node: index * 4 for index, node in enumerate(nodes[1:])}
    unit = float(max(max(frames[node].image.shape[:2]) for node in nodes))
    rows, cols, values, targets, tolerances = [], [], [], [], []
    row = 0
    for i, j, controls in edges:
        if i not in slots and i != root:
            continue
        target, source = controls.target / unit, controls.source / unit
        fixed = np.zeros((len(source), 2), np.float64)
        for node, points, sign in ((i, target, 1), (j, source, -1)):
            if node == root:
                fixed += sign * (_project(result[root], points * unit) / unit)
                continue
            offset = slots[node]
            for axis in (0, 1):
                coefficients = (points[:, 0], -points[:, 1], np.ones(len(points))) if axis == 0 else (
                    points[:, 1], points[:, 0], np.ones(len(points)))
                for column, coefficient in zip((offset, offset + 1, offset + 2 + axis), coefficients):
                    rows.append(row + np.arange(len(points), dtype=np.int32) * 2 + axis)
                    cols.append(np.full(len(points), column, dtype=np.int32))
                    values.append(sign * coefficient)
        targets.append((-fixed).ravel())
        tolerances.append(np.full(len(source), max(1.0, controls.tolerance)))
        row += len(source) * 2
    if not row:
        return
    design = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(row, 4 * len(slots))).tocsr()
    target = np.concatenate(targets)
    tolerance = np.concatenate(tolerances)
    current = np.concatenate([np.array([result[node][0, 0], result[node][1, 0],
                                        result[node][0, 2] / unit, result[node][1, 2] / unit])
                              for node in nodes[1:]])
    for _iteration in range(12):
        _cancel(should_cancel)
        error = target - design @ current
        distance = np.linalg.norm(error.reshape(-1, 2), axis=1) * unit
        weight = np.repeat((1 + (distance / tolerance) ** 2) ** -0.25, 2)
        weighted = design.multiply(weight[:, None]).tocsr()
        delta = lsmr(weighted, error * weight, atol=1e-11, btol=1e-11,
                     maxiter=max(100, 8 * len(current)))[0]
        current += delta
        if np.max(np.abs(delta)) < 1e-9:
            break
    for node, offset in slots.items():
        a, b, x, y = current[offset:offset + 4]
        result[node] = np.array([[a, -b, x * unit], [b, a, y * unit], [0, 0, 1]])


def auto_align(frames, transforms=None, should_cancel=None, progress=None):
    """Register microscope texture and jointly fit scale/rotation/placement."""
    frames = list(frames)
    if not frames:
        return MosaicAlignment((), (), ())
    original = [np.array(matrix, dtype=np.float64, copy=True) for matrix in
                (transforms if transforms is not None else initial_transforms(frames))]
    if len(original) != len(frames):
        raise ValueError('Each mosaic image needs one placement')
    # Descriptors and capped control points are retained; SIFT's image
    # pyramid is temporary and bounded by the largest thumbnail, not TIFF size.
    pairs = len(frames) * (len(frames) - 1) // 2
    require_memory(len(frames) * 12000 * 520 + pairs * 512 * 512
                   + max(frame.preview.shape[0] * frame.preview.shape[1] for frame in frames) * 128)
    # The photographic SIFT default (.04) discards the faint surface texture
    # in microscope TIFFs, leaving outline points and disconnected images.
    detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008)
    features = []
    total = len(frames) + pairs + 1
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
                edges.append((i, j, match))
            done += 1
            if progress is not None:
                progress(done, total)
    edges.sort(key=lambda item: (-len(item[2].source), item[0], item[1]))
    result = [None] * len(frames)
    connected = [False] * len(frames)
    for root in range(len(frames)):
        if result[root] is not None:
            continue
        result[root] = original[root]
        nodes = [root]
        changed = True
        while changed:
            _cancel(should_cancel)
            changed = False
            for i, j, controls in edges:
                if result[i] is not None and result[j] is None:
                    result[j] = result[i] @ controls.transform
                    nodes.append(j)
                    changed = True
                elif result[j] is not None and result[i] is None:
                    result[i] = result[j] @ np.linalg.inv(controls.transform)
                    nodes.append(i)
                    changed = True
        for node in nodes:
            connected[node] = root == 0
        if len(nodes) > 1:
            _optimize_component(frames, result, nodes, edges, should_cancel)
    _cancel(should_cancel)
    if progress is not None:
        progress(total, total)
    matches = tuple((i, j, len(controls.source), float(np.median(np.linalg.norm(
        _project(result[j], controls.source) - _project(result[i], controls.target), axis=1))))
        for i, j, controls in edges)
    return MosaicAlignment(tuple(result), tuple(connected), matches)


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
            xx = np.arange(left + x, left + x + tw, dtype=np.float64)[None, :]
            yy = np.arange(top + y, top + y + th, dtype=np.float64)[:, None]
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
                colors *= weight[:, :, None]
                acc += colors
                weights += weight
                if coverage is not None:
                    coverage += (1.0 - coverage) * alpha
            np.divide(acc, weights[:, :, None], out=acc, where=weights[:, :, None] > 0)
            np.clip(acc, 0, np.iinfo(dtype).max, out=acc)
            np.rint(acc, out=acc)
            result[y:y + th, x:x + tw, :3] = acc
            if coverage is not None:
                result[y:y + th, x:x + tw, 3] = np.rint(coverage * np.iinfo(dtype).max).astype(dtype)
            done += 1
            if progress is not None:
                progress(done, total)
    _cancel(should_cancel)
    result.flush()
    return result, (left, top)
