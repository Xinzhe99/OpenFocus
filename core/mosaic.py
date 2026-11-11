"""Bounded microscope and camera mosaics with captured global render plans."""
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
import math

import cv2
import numpy as np

from core.frame_storage import (create_frame, require_disk_space, storage_directory,
                                frame_region)
from core.resource_policy import require_memory
from utils.image_utils import thumbnail_bgr
from core.mosaic_control import (MosaicControlPoint, optimize_control_points,
                                 measure_control_points)
from core.mosaic_projection import (CameraSpec, ProjectionSpec, PairObservation,
    undistort_points, seed_cameras, fit_cameras, camera_rays, rotation_matrix,
    project, source_coordinates, distort_points)
from core.mosaic_seam import (SeamSpec, LowSample, build_labels, label_weight,
                              _label_blocks, _LABEL_BLOCK_PIXELS)
from core.mosaic_blend import expanded_tile, blend_tiles
from core.mosaic_photo import (PhotometricSpec, PhotoObservation,
    solve_photometric, apply_photometric_tile, linear_to_srgb,
    resolve_icc_policy, convert_icc_tile)
from core.mosaic_hdr import HDRSpec, fit_response, merge_hdr_tile


@dataclass(frozen=True)
class MosaicFrame:
    name: str
    image: np.ndarray
    preview: np.ndarray
    source_path: str = ''
    mask: object = None
    icc_profile: bytes = b''
    exif: bytes = b''
    exposure_time: object = None
    bracket_id: object = None

    def __post_init__(self):
        if self.mask is not None:
            mask = np.asarray(self.mask)
            if (mask.shape != self.image.shape[:2]
                    or mask.dtype not in (np.dtype('bool'), np.dtype('uint8'),
                                          np.dtype('uint16'))):
                raise ValueError('Source mask must be bool, uint8 or uint16 at native size')
            view = mask.view()
            view.flags.writeable = False
            object.__setattr__(self, 'mask', view)
        object.__setattr__(self, 'icc_profile', bytes(self.icc_profile or b''))
        object.__setattr__(self, 'exif', bytes(self.exif or b''))
        if self.exposure_time is not None:
            value = float(self.exposure_time)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Exposure time must be positive seconds')
            object.__setattr__(self, 'exposure_time', value)
        if self.bracket_id is not None:
            hash(self.bracket_id)


@dataclass(frozen=True)
class MosaicAlignment:
    transforms: tuple
    connected: tuple
    matches: tuple
    control_points: tuple = ()
    qa: object = None
    cameras: tuple = ()


@dataclass(frozen=True)
class _ControlPoints:
    source: np.ndarray
    target: np.ndarray


def _project(matrix, points):
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled()


def make_frame(name, image, source_path='', preview_size=1400, *, mask=None,
               icc_profile=b'', exif=b'', exposure_time=None, bracket_id=None):
    """Prepare one thumbnail without converting an entire 16-bit image."""
    if image is None or image.dtype not in (np.uint8, np.uint16):
        raise ValueError('Mosaic images must contain 8-bit or 16-bit pixels')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError('Mosaic images must be grayscale, BGR or BGRA')
    if min(image.shape[:2]) < 1:
        raise ValueError('Mosaic images cannot be empty')
    with frame_region(image) as pixels:
        # thumbnail_bgr composites RGBA before resizing; resize first to keep
        # transparent, high-resolution inputs bounded as well.
        if pixels.ndim == 3 and pixels.shape[2] == 4:
            height, width = pixels.shape[:2]
            scale = min(1.0, preview_size / max(height, width))
            small = cv2.resize(pixels, (max(1, round(width * scale)),
                                       max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA) if scale < 1 else pixels
            preview = thumbnail_bgr(small, preview_size)
        else:
            preview = thumbnail_bgr(pixels, preview_size)
        preview = np.array(preview, copy=True)
    return MosaicFrame(str(name), image, preview, str(source_path), mask,
                       icc_profile, exif, exposure_time, bracket_id)


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
    valid = None
    for source in (frame.mask, frame.image[:, :, 3] if frame.image.ndim == 3
                   and frame.image.shape[2] == 4 else None):
        if source is None:
            continue
        # Sample first: OpenCV otherwise copies an entire strided alpha plane.
        sy = max(1, source.shape[0] // gray.shape[0])
        sx = max(1, source.shape[1] // gray.shape[1])
        small = (source[::sy, ::sx] > 0).astype(np.uint8)
        small = cv2.resize(small, (gray.shape[1], gray.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        valid = small if valid is None else valid & small
    keypoints, descriptors = detector.detectAndCompute(
        gray, None if valid is None else valid * 255)
    if descriptors is None or len(keypoints) < 6:
        return None
    scale = np.array([frame.image.shape[1] / gray.shape[1],
                      frame.image.shape[0] / gray.shape[0]], dtype=np.float64)
    points = np.array([item.pt for item in keypoints], dtype=np.float64) * scale
    return points, descriptors, float(max(scale))


def _pair_match(first, second, first_features, second_features, cameras=None):
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
    if cameras is None:
        affine, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=threshold,
            maxIters=6000, confidence=0.999, refineIters=20)
        if affine is None or mask is None:
            return None
        scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
        if not 0.5 <= scale <= 2.0:
            return None
    else:
        ax, ay, av = undistort_points(target[:, 0], target[:, 1], cameras[0])
        bx, by, bv = undistort_points(source[:, 0], source[:, 1], cameras[1])
        valid = av & bv
        source, target = source[valid], target[valid]
        if len(source) < 12:
            return None
        transform, mask = cv2.findHomography(
            np.column_stack((bx[valid], by[valid])),
            np.column_stack((ax[valid], ay[valid])), cv2.RANSAC, threshold,
            maxIters=6000, confidence=0.999)
        if transform is None or mask is None or not np.isfinite(transform).all():
            return None
    selected = mask.ravel().astype(bool)
    if selected.sum() < 12 or selected.mean() < 0.3:
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
    return _ControlPoints(source, target)


def observations_from_control_points(control_points, frame_count):
    """Collect enabled native CPs into supported, weighted camera pairs."""
    pairs = {}
    for point in control_points:
        if max(point.frame_a, point.frame_b) >= frame_count:
            raise ValueError('Control point references an absent source')
        if not point.enabled:
            continue
        i, j = sorted((point.frame_a, point.frame_b))
        a, b = ((point.point_a, point.point_b) if i == point.frame_a
                else (point.point_b, point.point_a))
        pairs.setdefault((i, j), []).append((a, b, point.weight))
    result = []
    for (i, j), rows in sorted(pairs.items()):
        if len(rows) < 4:
            continue
        a = np.asarray([row[0] for row in rows], np.float64)
        b = np.asarray([row[1] for row in rows], np.float64)
        if any(np.linalg.svd(p - p.mean(axis=0), compute_uv=False)[1]
               <= max(1e-12, np.linalg.svd(p - p.mean(axis=0),
                   compute_uv=False)[0] * 1e-4) for p in (a, b)):
            continue
        result.append(PairObservation(i, j, a, b,
                                      np.asarray([row[2] for row in rows])))
    return tuple(result)


def measure_camera_control_points(cameras, points, should_cancel=None):
    # Measure each pair in camera A's ideal pixel plane. Keep original row
    # order for the CP table; no synthetic image-overlap percentage is added.
    measured = []
    for point in points:
        _cancel(should_cancel)
        a, b = cameras[point.frame_a], cameras[point.frame_b]
        ax, ay, valid_a = undistort_points(*point.point_a, a,
                                          cancel=should_cancel)
        ray, valid_b = camera_rays(*point.point_b, b, cancel=should_cancel)
        local = ray @ rotation_matrix(a)
        if not bool(valid_a & valid_b) or local[2] <= 1e-9:
            raise ValueError('Control point does not project into its paired camera')
        bx = local[0] / local[2] * a.focal + a.center[0]
        by = local[1] / local[2] * a.focal + a.center[1]
        measured.append(replace(point, point_a=(float(ax), float(ay)),
                                point_b=(float(bx), float(by))))
    return measure_control_points([np.eye(3) for _ in cameras], measured,
                                  should_cancel=should_cancel)


def _fit_camera_points(cameras, points, optimize_fov, should_cancel):
    observations = observations_from_control_points(points, len(cameras))
    neighbors = [set() for _ in cameras]
    for observation in observations:
        neighbors[observation.i].add(observation.j)
        neighbors[observation.j].add(observation.i)
    fitted, seen = list(cameras), set()
    for root in range(len(cameras)):
        if root in seen:
            continue
        nodes = [root]
        seen.add(root)
        for node in nodes:
            for other in sorted(neighbors[node]):
                if other not in seen:
                    seen.add(other)
                    nodes.append(other)
        if len(nodes) == 1:
            continue
        slots = {node: index for index, node in enumerate(nodes)}
        local_points, homographies = [], []
        for observation in observations:
            if observation.i not in slots:
                continue
            i, j = slots[observation.i], slots[observation.j]
            local_points.append(PairObservation(i, j, observation.points_i,
                observation.points_j, observation.weights))
            ideal = []
            for index, values in ((observation.i, observation.points_i),
                                  (observation.j, observation.points_j)):
                xx, yy, valid = undistort_points(values[:, 0], values[:, 1],
                    cameras[index], cancel=should_cancel)
                if not valid.all():
                    raise ValueError('Control point is outside a usable lens branch')
                ideal.append(np.column_stack((xx, yy)))
            matrix, _inliers = cv2.findHomography(ideal[1], ideal[0],
                cv2.RANSAC, 2.0, maxIters=6000, confidence=0.999)
            if matrix is not None and np.isfinite(matrix).all():
                homographies.append((i, j, matrix))
        local_cameras = tuple(cameras[node] for node in nodes)
        # A partial seed graph may preserve valid current manual camera seeds.
        reached = {0}
        for _ in nodes:
            for i, j, _matrix in homographies:
                if i in reached or j in reached:
                    reached.update((i, j))
        if len(reached) == len(nodes):
            local_cameras = seed_cameras(local_cameras, homographies,
                                         cancel=should_cancel)
        fit = fit_cameras(local_cameras, local_points,
            optimize_fov=bool(optimize_fov), cancel=should_cancel)
        for node, camera in zip(nodes, fit.cameras):
            fitted[node] = camera
    return tuple(fitted), measure_camera_control_points(tuple(fitted), points, should_cancel)


def auto_align(frames, transforms=None, should_cancel=None, progress=None, *,
               control_points=(), cameras=(), optimize_fov=False,
               detect_features=True):
    """Detect CPs, or fit exactly the edited records without resurrecting them."""
    try:
        frames = tuple(frames)
        if not frames:
            return MosaicAlignment((), (), ())
        original = tuple(np.array(matrix, np.float64, copy=True) for matrix in
            (transforms if transforms is not None else initial_transforms(frames)))
        if len(original) != len(frames):
            raise ValueError('Each mosaic image needs one placement')
        cameras = tuple(cameras)
        if cameras and len(cameras) != len(frames):
            raise ValueError('Each panorama image needs one camera')
        points = [point for point in control_points
                  if not detect_features or point.source != 'auto']
        if detect_features:
            pairs = len(frames) * (len(frames) - 1) // 2
            require_memory(len(frames) * 12000 * 520 + pairs * 512 * 512
                + max(f.preview.shape[0] * f.preview.shape[1] for f in frames) * 128)
            detector = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008)
            features, total, done = [], len(frames) + pairs + 1, 0
            if progress is not None:
                progress(0, total)
            for frame in frames:
                _cancel(should_cancel)
                features.append(_features(frame, detector))
                done += 1
                if progress is not None:
                    progress(done, total)
            for i in range(len(frames)):
                for j in range(i + 1, len(frames)):
                    _cancel(should_cancel)
                    match = _pair_match(frames[i], frames[j], features[i], features[j],
                        (cameras[i], cameras[j]) if cameras else None)
                    if match is not None:
                        points.extend(MosaicControlPoint(i, j, tuple(a), tuple(b),
                            source='auto', weight=1.0)
                            for a, b in zip(match.target, match.source))
                    done += 1
                    if progress is not None:
                        progress(done, total)
        else:
            total = 1
            if progress is not None:
                progress(0, total)
        if cameras:
            fitted_cameras, qa = _fit_camera_points(cameras, points,
                                                   optimize_fov, should_cancel)
            result = tuple(np.frombuffer(matrix.tobytes(), np.float64).reshape(3, 3)
                           for matrix in original)
        else:
            fit = optimize_control_points(original, points,
                dimensions=tuple((f.image.shape[1], f.image.shape[0]) for f in frames),
                should_cancel=should_cancel)
            result, qa, fitted_cameras = fit.transforms, fit.qa, ()
        _cancel(should_cancel)
        if progress is not None:
            progress(total, total)
        matches = tuple((pair.frame_a, pair.frame_b, pair.count, pair.median)
                        for pair in qa.pairs)
        return MosaicAlignment(result, qa.connected, matches, tuple(points), qa,
                               fitted_cameras)
    except InterruptedError as exc:
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled() from exc


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




@dataclass(frozen=True)
class BlendSpec:
    mode: str = 'multiband'
    levels: int = 5

    def __post_init__(self):
        if self.mode not in ('multiband', 'feather', 'hard'):
            raise ValueError('Unknown mosaic blend mode')
        if not isinstance(self.levels, int) or not 1 <= self.levels <= 7:
            raise ValueError('Mosaic pyramid levels must be between one and seven')


@dataclass(frozen=True)
class OutputSpec:
    dtype: object = None
    icc_policy: str = 'auto'
    preserve_exif: bool = True
    scale: float = 1.0
    crop: tuple = (0.0, 0.0, 1.0, 1.0)
    alpha: str = 'auto'

    def __post_init__(self):
        if self.dtype is not None:
            dtype = np.dtype(self.dtype)
            if dtype not in (np.dtype('uint8'), np.dtype('uint16')):
                raise ValueError('Mosaic output must be uint8 or uint16')
            object.__setattr__(self, 'dtype', dtype)
        if self.icc_policy not in ('auto', 'srgb', 'keep_first', 'drop'):
            raise ValueError('Unknown mosaic ICC policy')
        if not math.isfinite(self.scale) or not 0.01 <= self.scale <= 10:
            raise ValueError('Mosaic output scale must lie between 0.01 and 10')
        crop = tuple(float(value) for value in self.crop)
        if (len(crop) != 4 or not all(math.isfinite(v) for v in crop)
                or not 0 <= crop[0] < crop[2] <= 1
                or not 0 <= crop[1] < crop[3] <= 1):
            raise ValueError('Mosaic crop must be a normalized nonempty rectangle')
        object.__setattr__(self, 'crop', crop)
        if self.alpha not in ('auto', 'on', 'off'):
            raise ValueError('Unknown mosaic transparency policy')


@dataclass(frozen=True)
class RenderSpec:
    seam: SeamSpec = field(default_factory=SeamSpec)
    blend: BlendSpec = field(default_factory=BlendSpec)
    photometric: PhotometricSpec = field(default_factory=PhotometricSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    tile_size: int = 512
    halo: object = None
    projection: object = None
    cameras: tuple = ()
    hdr: HDRSpec = field(default_factory=HDRSpec)
    bracket_size: int = 1
    bracket_groups: tuple = ()

    def __post_init__(self):
        if not isinstance(self.tile_size, int) or not 32 <= self.tile_size <= 1024:
            raise ValueError('Mosaic tile size must be between 32 and 1024')
        if self.halo is not None and (not isinstance(self.halo, int) or self.halo < 0):
            raise ValueError('Mosaic halo must be a nonnegative integer')
        if self.projection is not None and not isinstance(self.projection, ProjectionSpec):
            raise TypeError('Expected a supported ProjectionSpec')
        object.__setattr__(self, 'cameras', tuple(self.cameras))
        object.__setattr__(self, 'bracket_groups',
                           tuple(tuple(group) for group in self.bracket_groups))
        if self.projection is None and self.cameras:
            raise ValueError('Camera models require panorama projection mode')
        if not isinstance(self.bracket_size, int) or self.bracket_size < 1:
            raise ValueError('HDR bracket size must be positive')
        if self.hdr.mode != 'off':
            if self.photometric.linear:
                raise ValueError('HDR expects encoded input; disable linear-light blending')
            if self.hdr.mode == 'debevec' and self.photometric.mode != 'off':
                raise ValueError('Calibrated HDR requires exposure compensation to be off')
            if self.hdr.levels > 7:
                raise ValueError('HDR pyramid levels must not exceed seven')


@dataclass(frozen=True)
class MosaicPlan:
    frames: tuple
    transforms: tuple
    spec: RenderSpec
    origin: tuple
    size: tuple
    boxes: tuple
    labels: np.ndarray
    low_samples: tuple
    photometric: object
    icc: object
    dtype: np.dtype
    has_alpha: bool
    exif: bytes
    projection: object
    groups: tuple
    response: object
    directory: str
    control_points: tuple = ()
    qa: object = None

    @property
    def icc_profile(self):
        return self.icc.output_profile


def _immutable_transform(matrix):
    value = np.array(matrix, dtype=np.float64, copy=True)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.array_equal(value[2], (0, 0, 1))
            or abs(np.linalg.det(value[:2, :2])) < 1e-12):
        raise ValueError('Mosaic placements must be finite nonsingular affine matrices')
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(3, 3)


def _camera_box(camera, projection, cancel):
    count = 257
    xs = np.linspace(0, camera.width - 1, count)
    ys = np.linspace(0, camera.height - 1, count)
    x = np.concatenate((xs, xs, np.zeros(count), np.full(count, camera.width - 1)))
    y = np.concatenate((np.zeros(count), np.full(count, camera.height - 1), ys, ys))
    rays, visible = camera_rays(x, y, camera, cancel=cancel)
    if projection.kind == 'rectilinear' and np.any(rays[visible, 2] <= 1e-6):
        raise ValueError('Rectilinear panorama crosses its horizon; choose another projection')
    px, py, projected = project(rays, projection, cancel=cancel)
    valid = visible & projected
    if not np.any(valid):
        raise ValueError('A source camera lies outside the selected projection')
    bounds = [float(px[valid].min()), float(py[valid].min()),
              float(px[valid].max() + 1), float(py[valid].max() + 1)]
    if projection.kind in ('cylindrical', 'equirectangular'):
        rotation = rotation_matrix(camera)
        for sign in (-1, 1):
            local = np.array([0., float(sign), 0.]) @ rotation
            if local[2] <= 1e-12:
                continue
            ux = local[0] / local[2] * camera.focal + camera.center[0]
            uy = local[1] / local[2] * camera.focal + camera.center[1]
            sx, sy, usable = distort_points(ux, uy, camera, cancel=cancel)
            if not bool(usable) or not (0 <= sx <= camera.width-1 and 0 <= sy <= camera.height-1):
                continue
            if projection.kind == 'cylindrical':
                raise ValueError('Cylindrical panorama contains a pole; use equirectangular projection')
            bounds[0] = min(bounds[0], projection.center_x - math.pi * projection.focal)
            bounds[2] = max(bounds[2], projection.center_x + math.pi * projection.focal + 1)
            pole_y = projection.center_y + sign * math.pi / 2 * projection.focal
            if sign < 0:
                bounds[1] = min(bounds[1], pole_y)
            else:
                bounds[3] = max(bounds[3], pole_y + 1)
    return tuple(bounds)


def _geometry(frames, transforms, spec, cancel):
    if not frames or len(frames) != len(transforms):
        raise ValueError('Each mosaic source needs one placement')
    scaling = np.diag([spec.output.scale, spec.output.scale, 1.0])
    matrices = tuple(_immutable_transform(scaling @ np.asarray(matrix)) for matrix in transforms)
    projection = spec.projection
    if projection is None:
        boxes = tuple(_bounds(frame, matrix) for frame, matrix in zip(frames, matrices))
    else:
        if len(spec.cameras) != len(frames):
            raise ValueError('Provide one camera per panorama source')
        for camera, frame in zip(spec.cameras, frames):
            if not isinstance(camera, CameraSpec) or (camera.height, camera.width) != frame.image.shape[:2]:
                raise ValueError('Camera dimensions must match their source images')
        projection = replace(projection, focal=projection.focal * spec.output.scale,
                             center_x=projection.center_x * spec.output.scale,
                             center_y=projection.center_y * spec.output.scale)
        boxes = tuple(_camera_box(camera, projection, cancel) for camera in spec.cameras)
    left, top = math.floor(min(b[0] for b in boxes)), math.floor(min(b[1] for b in boxes))
    right, bottom = math.ceil(max(b[2] for b in boxes)), math.ceil(max(b[3] for b in boxes))
    width, height = right - left, bottom - top
    crop = spec.output.crop
    origin = (left + math.floor(crop[0] * width), top + math.floor(crop[1] * height))
    size = (left + math.ceil(crop[2] * width) - origin[0],
            top + math.ceil(crop[3] * height) - origin[1])
    if min(size) < 1:
        raise ValueError('Mosaic canvas is empty')
    return matrices, projection, boxes, origin, size


def _maps(frames, matrices, cameras, projection, index, xx, yy, cancel):
    if projection is None:
        inv = np.linalg.inv(matrices[index])
        mx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
        my = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
        sh, sw = frames[index].image.shape[:2]
        valid = (mx >= 0) & (my >= 0) & (mx < sw) & (my < sh)
    else:
        mx, my, valid = source_coordinates(xx, yy, cameras[index], projection, cancel=cancel)
    return mx, my, valid


def _masked_sample(source, mask, mx, my, valid, cancel):
    """Bilinear premultiplied sampling, bounded by the output tile size."""
    sh, sw = source.shape[:2]
    maximum = float(np.iinfo(source.dtype).max)
    # Match OpenCV's 1/32-pixel interpolation grid without promoting a native
    # source crop to float; minified tiles may cover an entire large source.
    fx = np.rint(np.where(valid, mx, 0) * 32) / 32
    fy = np.rint(np.where(valid, my, 0) * 32) / 32
    bx, by = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    fx, fy = fx - bx, fy - by
    colors = np.zeros((*valid.shape, 3), np.float32)
    opacity = np.zeros(valid.shape, np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            _cancel(cancel)
            iy, ix = np.clip(by + dy, 0, sh - 1), np.clip(bx + dx, 0, sw - 1)
            sample = source[iy, ix]
            if sample.ndim == 2:
                sample = np.repeat(sample[:, :, None], 3, axis=2)
            elif sample.shape[2] == 1:
                sample = np.repeat(sample, 3, axis=2)
            alpha = (sample[:, :, 3].astype(np.float32) / maximum
                     if sample.shape[2] == 4 else np.ones(valid.shape, np.float32))
            if mask is not None:
                masked = mask[iy, ix].astype(np.float32)
                if mask.dtype != np.bool_:
                    masked /= float(np.iinfo(mask.dtype).max)
                alpha *= masked
            coefficient = ((fx if dx else 1-fx) * (fy if dy else 1-fy)
                           * valid).astype(np.float32)
            alpha *= coefficient
            colors += sample[:, :, :3] * alpha[:, :, None]
            opacity += alpha
    np.divide(colors, opacity[:, :, None], out=colors,
              where=opacity[:, :, None] > 0)
    return colors, opacity


def _sample(frames, matrices, cameras, projection, icc, dtype, index,
            xx, yy, photo, cancel):
    """Sample native pixels; gray/BGR promotion follows bounded gathering."""
    _cancel(cancel)
    frame = frames[index]
    with frame_region(frame.image) as source, frame_region(frame.mask) as mask:
        sh, sw = source.shape[:2]
        mx, my, valid = _maps(frames, matrices, cameras, projection, index, xx, yy, cancel)
        shape = np.broadcast_shapes(np.shape(xx), np.shape(yy))
        maximum = float(np.iinfo(dtype).max)
        if not np.any(valid):
            return np.zeros((*shape, 3), np.float32), np.zeros(shape, np.float32), mx, my
        x0 = max(0, int(np.floor(mx[valid].min())) - 1)
        y0 = max(0, int(np.floor(my[valid].min())) - 1)
        x1 = min(sw, int(np.ceil(mx[valid].max())) + 2)
        y1 = min(sh, int(np.ceil(my[valid].max())) + 2)
        if (mask is not None or source.ndim == 3 and source.shape[2] == 4
                or max(x1-x0, y1-y0) >= 32767):
            colors, alpha = _masked_sample(source, mask, mx, my, valid, cancel)
            if icc.convert_frames[index]:
                colors = convert_icc_tile(np.rint(np.clip(colors, 0, 255)).astype(np.uint8),
                    index, icc, should_cancel=cancel).astype(np.float32)
        else:
            sample = cv2.remap(source[y0:y1, x0:x1],
                np.where(valid, mx - x0, -1).astype(np.float32),
                np.where(valid, my - y0, -1).astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if sample.ndim == 2 or sample.shape[2] == 1:
                sample = cv2.cvtColor(sample.reshape(shape), cv2.COLOR_GRAY2BGR)
            if icc.convert_frames[index]:
                sample = convert_icc_tile(sample, index, icc, should_cancel=cancel)
            colors = sample.astype(np.float32)
            alpha = valid.astype(np.float32)
        native_max = float(np.iinfo(source.dtype).max)
        if native_max != maximum:
            colors *= maximum / native_max
        if photo is not None and (photo.spec.mode != 'off' or photo.spec.linear
                                  or photo.spec.vignette_coefficients):
            ux = np.clip(np.where(valid, mx, 0) * (2.0 / max(1, sw - 1)) - 1, -1, 1)
            uy = np.clip(np.where(valid, my, 0) * (2.0 / max(1, sh - 1)) - 1, -1, 1)
            colors = apply_photometric_tile(colors / maximum, index, ux, uy, photo) * maximum
        colors[alpha <= 0] = 0
        return colors, alpha, mx, my


def _low_box(box, origin, size, scale):
    x0 = min(math.ceil(size[0] / scale), max(0, math.floor((box[0] - origin[0]) / scale)))
    y0 = min(math.ceil(size[1] / scale), max(0, math.floor((box[1] - origin[1]) / scale)))
    x1 = min(math.ceil(size[0] / scale), math.ceil((box[2] - origin[0]) / scale))
    y1 = min(math.ceil(size[1] / scale), math.ceil((box[3] - origin[1]) / scale))
    return x0, y0, max(x0, x1), max(y0, y1)


def _low_sources(frames, matrices, projection, boxes, origin, size, spec, icc,
                 dtype, directory, photo, cancel):
    p = spec.seam.precision
    maximum = float(np.iinfo(dtype).max)
    result = []
    for index, box in enumerate(boxes):
        _cancel(cancel)
        x0, y0, x1, y1 = _low_box(box, origin, size, p)
        shape = (y1 - y0, x1 - x0)
        if not all(shape):
            result.append(LowSample(index, np.zeros((*shape, 3), np.float32),
                                    np.zeros(shape, np.float32), (x0, y0)))
            continue
        colors = create_frame((*shape, 3), np.float32, directory=directory)
        opacity = create_frame(shape, np.float32, directory=directory)
        for y in range(y0, y1, 128):
            _cancel(cancel)
            yy = origin[1] + np.minimum((np.arange(y, min(y1, y + 128))[:, None]
                + .5) * p - .5, size[1] - 1)
            for x in range(x0, x1, 256):
                xx = origin[0] + np.minimum((np.arange(x, min(x1, x + 256))[None, :]
                    + .5) * p - .5, size[0] - 1)
                sample, alpha, _mx, _my = _sample(frames, matrices, spec.cameras,
                    projection, icc, dtype, index, xx, yy, photo, cancel)
                region = np.s_[y-y0:y-y0+sample.shape[0], x-x0:x-x0+sample.shape[1]]
                with frame_region(colors, region, write=True) as target:
                    target[:] = sample / maximum
                with frame_region(opacity, region, write=True) as target:
                    target[:] = alpha
        result.append(LowSample(index, colors, opacity, (x0, y0)))
    return tuple(result)


def _photo_observations(frames, matrices, projection, origin, spec, lows, cancel):
    used = 0
    for i, first in enumerate(lows):
        for j in range(i + 1, len(lows)):
            _cancel(cancel)
            second = lows[j]
            a, b = first.box, second.box
            x0, y0 = max(a[0], b[0]), max(a[1], b[1])
            x1, y1 = min(a[2], b[2]), min(a[3], b[3])
            area = max(0, x1 - x0) * max(0, y1 - y0)
            cap = min(spec.photometric.max_samples_per_observation,
                      spec.photometric.max_total_samples - used)
            if not area or cap <= 0:
                continue
            candidates = np.linspace(0, area - 1, min(area, cap * 4), dtype=np.int64)
            ys, xs = candidates // (x1 - x0) + y0, candidates % (x1 - x0) + x0
            al = first.alpha[ys-a[1], xs-a[0]]
            bl = second.alpha[ys-b[1], xs-b[0]]
            keep = np.flatnonzero((al > 0) & (bl > 0))[:cap]
            if not len(keep):
                continue
            xs, ys = xs[keep], ys[keep]
            xx = origin[0] + (xs + 0.5) * spec.seam.precision - 0.5
            yy = origin[1] + (ys + 0.5) * spec.seam.precision - 0.5
            uv = []
            for index in (i, j):
                mx, my, valid = _maps(frames, matrices, spec.cameras, projection, index, xx, yy, cancel)
                sh, sw = frames[index].image.shape[:2]
                uv.append(np.clip(np.column_stack((mx * (2 / max(1, sw-1)) - 1,
                                                    my * (2 / max(1, sh-1)) - 1)), -1, 1))
            used += len(keep)
            yield PhotoObservation(i, j, uv[0], uv[1],
                first.colors[ys-a[1], xs-a[0]], second.colors[ys-b[1], xs-b[0]],
                weights=np.minimum(al[keep], bl[keep]))


def _brackets(frames, spec):
    if spec.hdr.mode == 'off':
        return tuple((i,) for i in range(len(frames)))
    groups = spec.bracket_groups
    if not groups and any(frame.bracket_id is not None for frame in frames):
        metadata = {}
        for index, frame in enumerate(frames):
            if frame.bracket_id is None:
                raise ValueError('Every HDR source needs a bracket ID or explicit groups')
            metadata.setdefault(frame.bracket_id, []).append(index)
        groups = tuple(tuple(group) for group in metadata.values())
    if not groups:
        if spec.bracket_size < 2 or len(frames) % spec.bracket_size:
            raise ValueError('Declare complete HDR groups of at least two sources')
        groups = tuple(tuple(range(i, i + spec.bracket_size))
                       for i in range(0, len(frames), spec.bracket_size))
    indices = tuple(index for group in groups for index in group)
    if (any(len(group) < 2 for group in groups)
            or any(isinstance(index, (bool, np.bool_))
                   or not isinstance(index, (int, np.integer)) for index in indices)
            or sorted(indices) != list(range(len(frames)))):
        raise ValueError('HDR groups must include each source exactly once')
    return tuple(tuple(int(index) for index in group) for group in groups)


def _hdr_times(frames, spec):
    values = spec.hdr.exposure_times
    if not values and spec.hdr.mode == 'debevec':
        values = tuple(frame.exposure_time for frame in frames)
    if values and len(values) != len(frames):
        raise ValueError('Provide one exposure time per mosaic source')
    if spec.hdr.mode == 'debevec' and (not values or any(v is None or not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError('Calibrated HDR needs a positive exposure time for every source')
    return tuple(values)


def _group_box(group, boxes):
    return (min(boxes[i][0] for i in group), min(boxes[i][1] for i in group),
            max(boxes[i][2] for i in group), max(boxes[i][3] for i in group))


def _calibrate_response(lows, groups, times, cancel):
    for group in groups:
        _cancel(cancel)
        x0 = max(lows[i].box[0] for i in group)
        y0 = max(lows[i].box[1] for i in group)
        x1 = min(lows[i].box[2] for i in group)
        y1 = min(lows[i].box[3] for i in group)
        if x1 <= x0 or y1 <= y0:
            continue
        area = (x1-x0) * (y1-y0)
        positions = np.linspace(0, area-1, min(area, 4096), dtype=np.int64)
        xs, ys = positions % (x1-x0) + x0, positions // (x1-x0) + y0
        valid = np.ones(len(xs), bool)
        values = []
        for index in group:
            sample = lows[index]
            lx, ly = sample.origin
            with frame_region(sample.alpha) as alpha, frame_region(sample.colors) as colors:
                valid &= alpha[ys-ly, xs-lx] > 0
                values.append(colors[ys-ly, xs-lx])
        if not np.any(valid):
            continue
        matched = np.clip(np.stack(values)[:, valid], 0, 1)
        pick = np.linspace(0, matched.shape[1]-1, min(256, matched.shape[1]), dtype=int)
        matched = matched[:, pick]
        informative = (matched > 0) & (matched < 1)
        lo = np.min(np.where(informative, matched, np.inf), axis=0)
        hi = np.max(np.where(informative, matched, -np.inf), axis=0)
        if not np.all(np.any(hi - lo > 1e-6, axis=0)):
            continue
        return fit_response(matched, tuple(times[i] for i in group), should_cancel=cancel)
    raise ValueError('No visible HDR bracket has matched unclipped exposure variation')


def _hdr_low_groups(lows, groups, times, spec, response, directory, cancel,
                    canvas_shape):
    if spec.hdr.mode == 'off':
        return lows
    result = []
    low_levels = max(1, spec.hdr.levels - int(math.log2(spec.seam.precision)))
    grid = 1 << (low_levels - 1)
    for number, group in enumerate(groups):
        _cancel(cancel)
        x0 = min(lows[i].box[0] for i in group) // grid * grid
        y0 = min(lows[i].box[1] for i in group) // grid * grid
        x1 = min(canvas_shape[1], math.ceil(max(lows[i].box[2] for i in group) / grid) * grid)
        y1 = min(canvas_shape[0], math.ceil(max(lows[i].box[3] for i in group) / grid) * grid)
        shape = (y1-y0, x1-x0)
        if not all(shape):
            result.append(LowSample(number, np.empty((*shape, 3), np.float32),
                                    np.empty(shape, np.float32), (x0, y0)))
            continue
        require_memory(math.prod(shape) * len(group) * 64)
        with ExitStack() as regions:
            images, alpha = [], []
            for index in group:
                _cancel(cancel)
                sample = lows[index]
                colors = create_frame((*shape, 3), np.float32, directory=directory)
                opacity = create_frame(shape, np.float32, directory=directory)
                image = regions.enter_context(frame_region(colors, write=True))
                mask = regions.enter_context(frame_region(opacity, write=True))
                image.fill(0)
                mask.fill(0)
                sx, sy, ex, ey = sample.box
                with frame_region(sample.colors) as source:
                    image[sy-y0:ey-y0, sx-x0:ex-x0] = source
                with frame_region(sample.alpha) as source:
                    mask[sy-y0:ey-y0, sx-x0:ex-x0] = source
                np.clip(image, 0, 1, out=image)
                if not images:
                    merged_colors, merged_alpha = colors, opacity
                images.append(image)
                alpha.append(mask)
            hdr = replace(spec.hdr, levels=low_levels,
                          exposure_times=tuple(times[i] for i in group) if times else ())
            values, coverage = merge_hdr_tile(images, hdr, alpha=alpha, response=response,
                                               should_cancel=cancel)
            # Reuse the first bracket's mappings for the retained merged sample.
            images[0][:] = values
            alpha[0][:] = coverage
            del values, coverage
        result.append(LowSample(number, merged_colors, merged_alpha, (x0, y0)))
    return tuple(result)


def _planning_disk_bytes(low_boxes, low_shape, groups, spec, output_bytes):
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in low_boxes]
    original = sum(areas) * 16
    labels = math.prod(low_shape) * 2
    peak = original
    retained = 0
    if spec.hdr.mode != 'off':
        levels = max(1, spec.hdr.levels - int(math.log2(spec.seam.precision)))
        grid = 1 << (levels - 1)
        for group in groups:
            x0 = min(low_boxes[i][0] for i in group) // grid * grid
            y0 = min(low_boxes[i][1] for i in group) // grid * grid
            x1 = min(low_shape[1], math.ceil(max(low_boxes[i][2] for i in group) / grid) * grid)
            y1 = min(low_shape[0], math.ceil(max(low_boxes[i][3] for i in group) / grid) * grid)
            merged = (x1 - x0) * (y1 - y0) * 16
            peak = max(peak, original + retained + merged * len(group))
            retained += merged
    else:
        retained = original
    return max(peak, retained + labels, retained + labels + output_bytes)


def _labels_have_holes(labels, cancel):
    for key in _label_blocks(labels.shape):
        _cancel(cancel)
        with frame_region(labels, key) as block:
            if np.any(block == 0):
                return True
    return False


def plan_mosaic(frames, transforms, *, spec=RenderSpec(), alignment=None,
                should_cancel=None, progress=None):
    try:
        frames = tuple(frames)
        _cancel(should_cancel)
        if alignment is not None:
            if (not isinstance(alignment, MosaicAlignment)
                    or len(alignment.transforms) != len(frames)
                    or len(transforms) != len(frames)
                    or any(not np.allclose(a, b, rtol=0, atol=1e-8)
                           for a, b in zip(transforms, alignment.transforms))):
                raise ValueError('Alignment no longer matches source placements')
            transforms = alignment.transforms
            if spec.projection is not None:
                if len(alignment.cameras) != len(frames):
                    raise ValueError('Panorama alignment needs fitted cameras')
                spec = replace(spec, cameras=alignment.cameras)
        directory = storage_directory()
        matrices, projection, boxes, origin, size = _geometry(frames, transforms, spec, should_cancel)
        groups = _brackets(frames, spec)
        times = _hdr_times(frames, spec)
        if times != spec.hdr.exposure_times:
            spec = replace(spec, hdr=replace(spec.hdr, exposure_times=times))
        dtype = (spec.output.dtype or np.dtype('uint16' if spec.hdr.mode != 'off'
            or any(f.image.dtype == np.uint16 for f in frames) else 'uint8'))
        has_alpha = spec.output.alpha == 'on' or (spec.output.alpha == 'auto' and (
            projection is not None or any(f.mask is not None or (f.image.ndim == 3
                and f.image.shape[2] == 4) for f in frames)))
        p = spec.seam.precision
        low_shape = (math.ceil(size[1] / p), math.ceil(size[0] / p))
        low_boxes = tuple(_low_box(box, origin, size, p) for box in boxes)
        largest = max(((box[2] - box[0]) * (box[3] - box[1])
                       for box in low_boxes), default=0)
        output_bytes = size[0] * size[1] * 4 * dtype.itemsize
        require_disk_space(_planning_disk_bytes(low_boxes, low_shape, groups, spec, output_bytes)
                           + 64 * 1024**2, directory)
        require_memory(max(largest * 96, 128 * 256 * 128, _LABEL_BLOCK_PIXELS))
        icc = resolve_icc_policy([frame.icc_profile for frame in frames], spec.output.icc_policy)
        if any(icc.convert_frames[i] and frame.image.dtype != np.uint8 for i, frame in enumerate(frames)):
            raise ValueError('Mixed 16-bit ICC conversion is unavailable; explicitly choose keep_first or drop')
        if progress is not None:
            progress(0, 4)
        lows = _low_sources(frames, matrices, projection, boxes, origin, size, spec, icc,
                            dtype, directory, None, should_cancel)
        model = solve_photometric(_photo_observations(frames, matrices, projection, origin,
            spec, lows, should_cancel), len(frames), spec.photometric, should_cancel)
        if progress is not None:
            progress(1, 4)
        if model.spec.mode != 'off' or model.spec.linear or model.spec.vignette_coefficients:
            # Release uncorrected low pixels before building their corrected replacements.
            lows = ()
            lows = _low_sources(frames, matrices, projection, boxes, origin, size, spec, icc,
                                dtype, directory, model, should_cancel)
        response = _calibrate_response(lows, groups, times, should_cancel) if spec.hdr.mode == 'debevec' else None
        lows = _hdr_low_groups(lows, groups, times, spec, response, directory, should_cancel, low_shape)
        if progress is not None:
            progress(2, 4)
        labels = build_labels(lows, low_shape, spec.seam,
            allocate=lambda shape, kind: create_frame(shape, kind, directory=directory),
            should_cancel=should_cancel)
        if progress is not None:
            progress(3, 4)
        labels.flags.writeable = False
        if spec.output.alpha == 'auto' and _labels_have_holes(labels, should_cancel):
            has_alpha = True
            require_disk_space(size[0] * size[1] * 4 * dtype.itemsize + 64 * 1024**2, directory)
        if progress is not None:
            progress(4, 4)
        return MosaicPlan(frames, matrices, spec, origin, size, boxes, labels, lows, model,
            icc, dtype, has_alpha, frames[0].exif if spec.output.preserve_exif else b'',
            projection, groups, response, directory,
            alignment.control_points if alignment is not None else (),
            alignment.qa if alignment is not None else None)
    except InterruptedError as exc:
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled() from exc


def _tile_sources(plan, x0, y0, x1, y1, cancel):
    spec, origin = plan.spec, plan.origin
    shape = (y1-y0, x1-x0)
    xx = np.arange(origin[0]+x0, origin[0]+x1, dtype=np.float64)[None, :]
    yy = np.arange(origin[1]+y0, origin[1]+y1, dtype=np.float64)[:, None]
    maximum = float(np.iinfo(plan.dtype).max)
    for number, group in enumerate(plan.groups):
        box = _group_box(group, plan.boxes)
        if box[2] <= origin[0]+x0 or box[0] >= origin[0]+x1 or box[3] <= origin[1]+y0 or box[1] >= origin[1]+y1:
            continue
        _cancel(cancel)
        colors, opacities = [], []
        distance = None
        for index in group:
            sample, alpha, mx, my = _sample(plan.frames, plan.transforms, spec.cameras,
                plan.projection, plan.icc, plan.dtype, index, xx, yy, plan.photometric, cancel)
            colors.append(sample)
            opacities.append(alpha)
            sh, sw = plan.frames[index].image.shape[:2]
            local_distance = np.nan_to_num(np.minimum(
                np.minimum(mx + .5, sw - .5 - mx),
                np.minimum(my + .5, sh - .5 - my)), nan=0)
            local_distance = np.where(alpha > 0, local_distance, 0)
            distance = (local_distance if distance is None
                        else np.maximum(distance, local_distance))
        if len(group) == 1:
            image, alpha = colors[0], opacities[0]
        else:
            hdr = replace(spec.hdr, exposure_times=tuple(spec.hdr.exposure_times[i] for i in group)
                          if spec.hdr.exposure_times else ())
            image, alpha = merge_hdr_tile([np.clip(v / maximum, 0, 1) for v in colors], hdr,
                alpha=opacities, response=plan.response, should_cancel=cancel)
            image *= maximum
        if not np.any(alpha > 0):
            continue
        if spec.seam.mode == 'feather':
            weight = np.clip(np.maximum(distance, .5)
                / max(1, spec.seam.band), 0, 1).astype(np.float32)
        else:
            weight = label_weight(plan.labels, number, origin=origin,
                scale=spec.seam.precision, x=origin[0]+x0, y=origin[1]+y0,
                width=shape[1], height=shape[0],
                band=0 if spec.blend.mode == 'hard' else spec.seam.band)
            # A coarse cell can miss a thin valid footprint or an excluded owner.
            # Give uncovered label cells their actual available source, not black.
            xs = np.clip((np.arange(x0, x1) // spec.seam.precision), 0, plan.labels.shape[1]-1)
            ys = np.clip((np.arange(y0, y1) // spec.seam.precision), 0, plan.labels.shape[0]-1)
            weight = np.where(plan.labels[ys[:, None], xs[None, :]] == 0, 1.0, weight).astype(np.float32)
        yield image, alpha, weight


def render_mosaic(frames, transforms, *, plan=None, spec=None, feather=64,
                  tile_size=512, should_cancel=None, progress=None):
    """Render one captured global plan into a bounded, disk-backed canvas."""
    try:
        if plan is None:
            if spec is None:
                spec = RenderSpec(seam=SeamSpec(mode='feather', precision=8,
                    band=max(0, int(feather))), blend=BlendSpec(mode='feather', levels=1),
                    output=OutputSpec(alpha='on' if any(f.mask is not None or
                        f.image.ndim == 3 and f.image.shape[2] == 4 for f in frames)
                        else 'off'), tile_size=max(32, min(1024, int(tile_size))))
            plan = plan_mosaic(frames, transforms, spec=spec,
                               should_cancel=should_cancel)
        width, height = plan.size
        levels = max(plan.spec.blend.levels, plan.spec.hdr.levels if plan.spec.hdr.mode != 'off' else 1)
        band = plan.spec.seam.band
        if plan.spec.hdr.mode != 'off':
            band = max(band, 4 * (1 << (levels - 1)))
        tile = int(tile_size if spec is None else plan.spec.tile_size)
        if not 32 <= tile <= 1024:
            raise ValueError('Mosaic tile size must be between 32 and 1024')
        margin = max(4 * (1 << (levels-1)), 2 * band + 16)
        if plan.spec.halo is not None:
            margin = max(margin, plan.spec.halo)
        group_size = max(map(len, plan.groups))
        require_memory((tile + 2 * margin + (1 << levels)) ** 2 * (192 + group_size * 48))
        channels = 4 if plan.has_alpha else 3
        output = create_frame((height, width, channels), plan.dtype, directory=plan.directory)
        total = math.ceil(width/tile) * math.ceil(height/tile)
        done = 0
        maximum = float(np.iinfo(plan.dtype).max)
        if progress is not None:
            progress(0, total)
        for y in range(0, height, tile):
            for x in range(0, width, tile):
                _cancel(should_cancel)
                tw, th = min(tile, width-x), min(tile, height-y)
                (x0, y0, x1, y1), crop = expanded_tile(x, y, tw, th, width, height,
                    levels=levels, band=band, halo=plan.spec.halo)
                colors, alpha = blend_tiles(_tile_sources(plan, x0, y0, x1, y1, should_cancel),
                    shape=(y1-y0, x1-x0), mode=plan.spec.blend.mode,
                    levels=plan.spec.blend.levels, should_cancel=should_cancel)
                colors = colors[crop]
                if plan.photometric.spec.linear:
                    colors = linear_to_srgb(colors / maximum, preserve_alpha=False) * maximum
                np.clip(colors, 0, maximum, out=colors)
                np.rint(colors, out=colors)
                with frame_region(output, np.s_[y:y+th, x:x+tw], write=True) as target:
                    target[:, :, :3] = colors
                    if plan.has_alpha:
                        target[:, :, 3] = np.rint(np.clip(alpha[crop], 0, 1) * maximum)
                done += 1
                if progress is not None:
                    progress(done, total)
        _cancel(should_cancel)
        output.flush()
        return output, plan.origin
    except InterruptedError as exc:
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled() from exc


def preview_mosaic(plan, *, max_edge=1200, show_seams=False, show_numbers=False,
                   show_masks=False, selected_points=(), should_cancel=None):
    """Blend the captured corrected low groups, without allocating full output."""
    try:
        if not isinstance(max_edge, int) or max_edge < 1:
            raise ValueError('Preview size must be a positive integer')
        _cancel(should_cancel)
        step = max(1, math.ceil(max(plan.size) / max_edge))
        width, height = math.ceil(plan.size[0] / step), math.ceil(plan.size[1] / step)
        shape = (height, width)
        p = plan.spec.seam.precision
        xx = plan.origin[0] + (np.arange(width)[None, :] + .5) * step - .5
        yy = plan.origin[1] + (np.arange(height)[:, None] + .5) * step - .5
        require_memory(width * height * 192 + max(
            (sample.colors.nbytes * 3 for sample in plan.low_samples), default=0))

        def samples():
            for sample, group in zip(plan.low_samples, plan.groups):
                _cancel(should_cancel)
                if not sample.alpha.size:
                    continue
                mx = ((xx - plan.origin[0] + .5) / p - .5 - sample.origin[0])
                my = ((yy - plan.origin[1] + .5) / p - .5 - sample.origin[1])
                mapx = np.broadcast_to(mx, shape).astype(np.float32)
                mapy = np.broadcast_to(my, shape).astype(np.float32)
                with frame_region(sample.colors) as source, frame_region(sample.alpha) as opacity:
                    color = cv2.remap(source * opacity[:, :, None], mapx, mapy,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    alpha = cv2.remap(opacity, mapx, mapy, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
                covered = ((mx >= -.5) & (my >= -.5)
                           & (mx < sample.alpha.shape[1] - .5)
                           & (my < sample.alpha.shape[0] - .5))
                alpha *= covered
                if not np.any(alpha > 0):
                    continue
                np.divide(color, alpha[:, :, None], out=color,
                          where=alpha[:, :, None] > 0)
                if plan.spec.seam.mode == 'feather':
                    distance = np.zeros(shape, np.float32)
                    for index in group:
                        sx, sy, valid = _maps(plan.frames, plan.transforms,
                            plan.spec.cameras, plan.projection, index, xx, yy, should_cancel)
                        sh, sw = plan.frames[index].image.shape[:2]
                        local = np.minimum(np.minimum(sx + .5, sw - .5 - sx),
                                           np.minimum(sy + .5, sh - .5 - sy))
                        distance = np.maximum(distance, np.where(valid, local, 0))
                    weight = np.clip(np.maximum(distance, .5)
                                     / max(1, plan.spec.seam.band), 0, 1).astype(np.float32)
                else:
                    weight = label_weight(plan.labels, sample.index, origin=plan.origin,
                        scale=p, x=float(xx[0, 0]), y=float(yy[0, 0]), width=width,
                        height=height, band=0 if plan.spec.blend.mode == 'hard'
                        else plan.spec.seam.band, pixel_step=step)
                yield color, alpha, weight

        levels = max(1, plan.spec.blend.levels - int(math.log2(step)))
        colors, opacity = blend_tiles(samples(), shape=shape,
            mode=plan.spec.blend.mode, levels=levels, should_cancel=should_cancel)
        if plan.photometric.spec.linear:
            colors = linear_to_srgb(colors, preserve_alpha=False)
        preview = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
        xs = np.clip(np.floor((xx[0] - plan.origin[0] + .5) / p).astype(int),
                     0, plan.labels.shape[1] - 1)
        ys = np.clip(np.floor((yy[:, 0] - plan.origin[1] + .5) / p).astype(int),
                     0, plan.labels.shape[0] - 1)
        with frame_region(plan.labels) as labels:
            ids = labels[ys[:, None], xs[None, :]].copy()
        if show_seams:
            edges = np.zeros(shape, bool)
            edges[:, 1:] |= ids[:, 1:] != ids[:, :-1]
            edges[1:] |= ids[1:] != ids[:-1]
            preview[edges & (opacity > 0)] = (0, 255, 255)
        if show_masks:
            excluded = np.zeros(shape, bool)
            for index, frame in enumerate(plan.frames):
                if frame.mask is None and (frame.image.ndim != 3 or frame.image.shape[2] != 4):
                    continue
                _cancel(should_cancel)
                sx, sy, valid = _maps(plan.frames, plan.transforms, plan.spec.cameras,
                    plan.projection, index, xx, yy, should_cancel)
                sh, sw = frame.image.shape[:2]
                ix = np.clip(np.rint(np.where(valid, sx, 0)).astype(int), 0, sw - 1)
                iy = np.clip(np.rint(np.where(valid, sy, 0)).astype(int), 0, sh - 1)
                if frame.mask is not None:
                    with frame_region(frame.mask) as mask:
                        excluded |= valid & (mask[iy, ix] == 0)
                if frame.image.ndim == 3 and frame.image.shape[2] == 4:
                    with frame_region(frame.image) as image:
                        excluded |= valid & (image[iy, ix, 3] == 0)
            preview[excluded] = np.rint(preview[excluded] * .55 + np.array([0, 0, 255]) * .45)
            opacity[excluded] = np.maximum(opacity[excluded], .5)
        if show_numbers:
            for sample, group in zip(plan.low_samples, plan.groups):
                mask = ((ids == sample.index + 1) & (opacity > 0)).astype(np.uint8)
                moments = cv2.moments(mask, binaryImage=True)
                if moments['m00'] == 0:
                    continue
                position = (round(moments['m10'] / moments['m00']),
                            round(moments['m01'] / moments['m00']))
                text = ','.join(str(index + 1) for index in group)
                cv2.putText(preview, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(preview, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255, 255, 255), 1, cv2.LINE_AA)
        for point in selected_points:
            positions = []
            for index, value in ((point.frame_a, point.point_a),
                                  (point.frame_b, point.point_b)):
                if plan.projection is None:
                    px, py = _project(plan.transforms[index], np.asarray([value]))[0]
                else:
                    ray, valid = camera_rays(*value, plan.spec.cameras[index],
                                             cancel=should_cancel)
                    px, py, projected = project(ray, plan.projection, cancel=should_cancel)
                    if not bool(valid & projected):
                        continue
                position = (round((float(px) - plan.origin[0] + .5) / step - .5),
                            round((float(py) - plan.origin[1] + .5) / step - .5))
                positions.append(position)
                cv2.circle(preview, position, 5, (255, 0, 255), 2, cv2.LINE_AA)
            if len(positions) == 2:
                cv2.line(preview, positions[0], positions[1], (0, 255, 0), 1, cv2.LINE_AA)
        _cancel(should_cancel)
        if plan.has_alpha:
            return np.dstack((preview, np.rint(np.clip(opacity, 0, 1) * 255).astype(np.uint8)))
        return preview
    except InterruptedError as exc:
        from core.registration import RegistrationCancelled
        raise RegistrationCancelled() from exc
