"""Measure the integrated core.mosaic pipeline and independently read exported TIFFs.

Candidate for tools/; run only after the prepared production modules are integrated.
Generation writes deterministic scene-continuous pixels in row bands to the existing
frame_storage mappings. By default no input decoder is involved: these are generated
pixels, not encoded camera files. Run one case per process on an idle worker.

--encoded-inputs (default off) instead writes those same pixels to real lossless
uncompressed TIFF pages with the production imwrite_auto streaming writer, releases
the generated colour/alpha mappings (only the frame's mask mapping survives), and
reopens every page through core.image_loader.iter_decoded_sources with the admission
and disk-backed snapshot semantics of mosaic imports
(dialogs/mosaic.py::_decode_pages). Each encoded source then gets row-band pixel,
bit-depth and alpha round-trip invariants; encoded-source write and load timings,
page headers, native metadata and decoder provenance are reported separately.
Encoded mode covers synthetic lossless TIFF decoding only, not camera RAW or
JPEG/HEIF decoding. Masks stay generated because production masks are painted,
never decoded; a 4-channel source's alpha plane is a sidecar grayscale page because
the production TIFF loader rejects pages whose alpha is not fully opaque and drops
alpha from pages it does decode. Encoded mode holds one decoded colour page (plus
its alpha sidecar page) in RAM at a time — the real decoder's behaviour — on top of
the generated frame mappings.

Small smoke (one CPU, source and canvas below 1024, four frames):
  python tools/mosaic_engineering_benchmark.py --width 256 --height 192 \
    --rows 2 --cols 2 --channels 4 --dtype uint16 --tile-sizes 32 64 \
    --output evidence/mosaic-smoke.tif --report evidence/mosaic-smoke.json
Small encoded-input smoke (real TIFFs decoded through the production mosaic loader):
  python tools/mosaic_engineering_benchmark.py --encoded-inputs --width 256 \
    --height 192 --rows 2 --cols 2 --channels 4 --dtype uint16 \
    --tile-sizes 32 64 --output evidence/mosaic-encoded.tif \
    --report evidence/mosaic-encoded.json
Remote 4000-square, sixteen sources (explicitly opt in on remote CI):
  python tools/mosaic_engineering_benchmark.py --remote-size --width 4000 \
    --height 4000 --rows 4 --cols 4 --channels 4 --dtype uint16 \
    --tile-sizes 256 512 --rss-limit-mib 1024 --output evidence/mosaic-4000.tif \
    --report evidence/mosaic-4000.json
Remote 45 MP sources, four sources:
  python tools/mosaic_engineering_benchmark.py --remote-size --width 8192 \
    --height 5504 --rows 2 --cols 2 --channels 3 --dtype uint16 \
    --tile-sizes 256 512 --rss-limit-mib 1024 --extra-scenarios \
    --output evidence/mosaic-45mp.tif --report evidence/mosaic-45mp.json

Each command writes the main TIFF, a second tile-size TIFF, a small PNG preview,
and compact diagnostic TIFFs beside it. Paths must not already exist. --scratch-dir
selects a volume with space for source mappings, plan data and output mappings; with
--encoded-inputs it also holds one uncompressed TIFF copy of every source frame.
Peak RSS includes generation, planning, rendering, export and independent readback;
it is an observed process limit, not a promise of bounded OS mapping residency.
Failure produces JSON and exit 1; argument errors exit 2. No P1-P5 completion claim:
see acceptance_scope in the JSON for exactly what this engineering probe covers.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--cols', type=int, default=2)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--dtype', choices=('uint8', 'uint16'), default='uint16')
    parser.add_argument('--channels', type=int, choices=(1, 3, 4), default=4)
    parser.add_argument('--origin', type=float, nargs=2, default=(-17.375, 23.625),
                        metavar=('X', 'Y'))
    parser.add_argument('--tile-sizes', type=int, nargs=2, default=(32, 64))
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--seam-band', type=int, default=16)
    parser.add_argument('--seam-precision', type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument('--band-rows', type=int, default=32)
    parser.add_argument('--difference-tolerance', type=int, default=1,
                        help='Maximum native-code color AND alpha tile difference')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--rss-limit-mib', type=float,
                        help='Process peak RSS threshold; defaults to local 512 / remote 1024')
    parser.add_argument('--remote-size', action='store_true',
                        help='Explicit remote-only permission for >8 frames or >1024 canvas')
    parser.add_argument('--encoded-inputs', action='store_true',
                        help='Write generated sources to real lossless TIFFs and reopen them '
                             'through the production mosaic image loader; measures encoded TIFF '
                             'decoding only when requested (no RAW/JPEG claim)')
    parser.add_argument('--extra-scenarios', action='store_true',
                        help='Also export compact three-projection and two-HDR-mode cases')
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if min(args.width, args.height) < 64 or min(args.rows, args.cols) < 1:
        parser.error('source dimensions must be >=64 and grid dimensions positive')
    if not math.isfinite(args.overlap) or not 0.05 <= args.overlap <= 0.6:
        parser.error('--overlap must lie between 0.05 and 0.6')
    if not all(math.isfinite(v) for v in args.origin):
        parser.error('--origin must be finite')
    if (not any(v != 0 for v in args.origin)
            or not any(v != math.floor(v) for v in args.origin)):
        parser.error('this probe requires a fractional, nonzero --origin')
    if (len(set(args.tile_sizes)) != 2
            or any(not 32 <= n <= 1024 for n in args.tile_sizes)):
        parser.error('choose two distinct tile sizes between 32 and 1024')
    if not 1 <= args.levels <= 5 or not 0 <= args.seam_band <= 128:
        parser.error('--levels must be 1..5 and --seam-band 0..128')
    if not 1 <= args.band_rows <= 128 or args.difference_tolerance < 0 or args.threads < 1:
        parser.error('band rows must be 1..128, tolerance nonnegative, threads positive')
    args.step_x = max(1, round(args.width * (1 - args.overlap)))
    args.step_y = max(1, round(args.height * (1 - args.overlap)))
    canvas = (args.width + args.step_x * (args.cols - 1) + 1,
              args.height + args.step_y * (args.rows - 1) + 1)
    if not args.remote_size and (max(args.width, args.height, *canvas) > 1024
                                 or args.rows * args.cols > 8 or args.threads != 1):
        parser.error('local limits: one CPU, <=8 sources, source and canvas edges <=1024')
    if args.rss_limit_mib is None:
        args.rss_limit_mib = 1024 if args.remote_size else 512
    if not math.isfinite(args.rss_limit_mib) or args.rss_limit_mib <= 0:
        parser.error('--rss-limit-mib must be positive and finite')
    if not args.remote_size and args.rss_limit_mib > 512:
        parser.error('local --rss-limit-mib cannot exceed 512')
    if args.output.suffix.lower() not in ('.tif', '.tiff'):
        parser.error('--output must name a TIFF')
    if args.output.resolve() == args.report.resolve():
        parser.error('--output and --report must differ')
    return args


class Measurements:
    def __init__(self, report, memory):
        self.report = report
        self.memory = memory
        self.stop = threading.Event()
        self.peak = memory.process_rss_bytes()
        self.stage = 'startup'
        self.stage_peaks = {}
        self.monitor = threading.Thread(target=self.sample, daemon=True)

    def sample_once(self):
        rss = self.memory.process_rss_bytes()
        self.peak = max(self.peak, rss)
        self.stage_peaks[self.stage] = max(self.stage_peaks.get(self.stage, 0), rss)

    def sample(self):
        while not self.stop.wait(0.02):
            self.sample_once()

    @contextmanager
    def timed(self, name):
        previous, self.stage = self.stage, name
        start = time.perf_counter()
        self.sample_once()
        try:
            yield
        finally:
            self.sample_once()
            self.report['stages_s'][name] = time.perf_counter() - start
            self.stage = previous

    def finish(self):
        self.stop.set()
        self.monitor.join()
        self.sample_once()
        self.report['sampled_peak_rss_bytes'] = self.peak
        self.report['stage_peak_rss_bytes'] = self.stage_peaks
        self.report['rss_sampling_interval_s'] = 0.02
        # ru_maxrss / Windows peak_wset catch short spikes missed by sampling.
        kernel_peak = None
        try:
            if sys.platform == 'win32':
                import psutil
                kernel_peak = int(psutil.Process().memory_info().peak_wset)
            else:
                import resource
                value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                kernel_peak = int(value if sys.platform == 'darwin' else value * 1024)
        except (ImportError, AttributeError, OSError):
            pass
        self.report['os_peak_rss_bytes'] = kernel_peak
        self.report['peak_rss_bytes'] = max(self.peak, kernel_peak or 0)


def require(report, name, passed, **measurements):
    item = {'name': name, 'passed': bool(passed), **measurements}
    report['invariants'].append(item)
    if not passed:
        report['errors'].append(name)


def artifact(args, suffix, extension='.tif'):
    return args.output.with_name(args.output.stem + suffix + extension)


def reserve_path(path, used):
    path = path.resolve()
    if path in used or path.exists():
        raise FileExistsError(f'Artifact would overwrite an existing or reserved path: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    used.add(path)
    return path


def scene_band(np, width, y0, y1, offset_x, offset_y, channels, dtype):
    """Same world coordinates have identical pixels in every overlapping source."""
    xx = np.arange(width, dtype=np.int64)[None, :] + offset_x
    yy = np.arange(y0, y1, dtype=np.int64)[:, None] + offset_y
    maximum = np.iinfo(dtype).max
    shape = (y1-y0, width) if channels == 1 else (y1-y0, width, channels)
    band = np.empty(shape, dtype=dtype)
    for c in range(1 if channels == 1 else 3):
        # Smooth structure plus deterministic low-amplitude native-bit detail.
        value = (0.44 + .12 * np.sin(xx * .017 + c)
                 + .10 * np.cos(yy * .023 - c * .7)
                 + .04 * np.sin((xx + yy) * .031)) * maximum
        detail = (xx * 37 + yy * 53 + c * 71) % (127 if maximum > 255 else 3)
        plane = np.rint(value + detail).astype(dtype)
        if channels == 1:
            band[:] = plane
        else:
            band[..., c] = plane
    if channels == 4:
        band[..., 3] = maximum
    return band


def source_band(np, args, y0, y1, offset_x, offset_y):
    """Deterministic scene rows plus the harness's synthetic hole/alpha pattern."""
    maximum = np.iinfo(args.dtype).max
    band = scene_band(np, args.width, y0, y1, offset_x, offset_y, args.channels, args.dtype)
    xx = np.arange(args.width)[None, :] + offset_x
    yy = np.arange(y0, y1)[:, None] + offset_y
    hole = ((xx >= args.width * .10) & (xx < args.width * .35)
            & (yy >= args.height * .10) & (yy < args.height * .35))
    valid = np.where(hole, 0, 255).astype(np.uint8)
    if args.channels == 1:
        band[hole] = maximum
    else:
        band[hole, :3] = (maximum, 0, maximum)
    if args.channels == 4:
        alpha_hole = ((xx >= args.width * .50) & (xx < args.width * .70)
                      & (yy >= args.height * .10) & (yy < args.height * .35))
        partial = ((xx >= args.width * .40) & (xx < args.width * .65)
                   & (yy >= args.height * .55) & (yy < args.height * .80))
        band[alpha_hole, 3] = 0
        band[alpha_hole, :3] = (maximum, 0, maximum)
        band[partial, 3] = maximum // 2
    return band, valid


def generate(args, np, create_frame, make_frame):
    frames, transforms = [], []
    for row in range(args.rows):
        for col in range(args.cols):
            ox, oy = col * args.step_x, row * args.step_y
            shape = (args.height, args.width) + (() if args.channels == 1 else (args.channels,))
            pixels = create_frame(shape, args.dtype)
            mask = create_frame((args.height, args.width), np.uint8)
            # Write descriptors rather than touching all writable mapping pages.
            with open(pixels.filename, 'r+b') as target, open(mask.filename, 'r+b') as mask_file:
                for y in range(0, args.height, args.band_rows):
                    end = min(y + args.band_rows, args.height)
                    band, valid = source_band(np, args, y, end, ox, oy)
                    band.tofile(target)
                    valid.tofile(mask_file)
            frames.append(make_frame(f'r{row}c{col}', pixels, mask=mask, preview_size=256))
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2, 2] = (args.origin[0] + ox, args.origin[1] + oy)
            transforms.append(matrix)
    return frames, transforms


def tiff_headers(tifffile, path):
    """Describe one encoded source page from its header, without decoding pixels."""
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        tags = {}
        for name in ('ImageWidth', 'ImageLength', 'BitsPerSample', 'SamplesPerPixel',
                     'PhotometricInterpretation', 'ExtraSamples', 'Compression',
                     'RowsPerStrip', 'PlanarConfiguration', 'SampleFormat', 'Orientation'):
            tag = page.tags.get(name)
            value = tag.value if tag is not None else None
            if isinstance(value, tuple):
                value = [int(v) for v in value]
            elif value is not None:
                value = int(value)
            tags[name] = value
        return {'path': os.path.abspath(path), 'file_bytes': os.path.getsize(path),
                'pages': len(container.pages), 'bigtiff': bool(container.is_bigtiff),
                'shape': list(page.shape), 'dtype': str(page.dtype), 'tags': tags}


def encode_source(args, np, create_frame, imwrite_auto, root, name, offset_x, offset_y):
    """Write one synthetic source as real lossless TIFF pages.

    Colour is the page a mosaic import can admit: BGR, or grayscale for
    --channels 1. A 4-channel source's alpha plane is a separate grayscale
    page because the production TIFF loader rejects pages whose alpha is not
    fully opaque and drops alpha from the pages it does decode.
    """
    shape = ((args.height, args.width) if args.channels == 1
             else (args.height, args.width, 3))
    color = create_frame(shape, args.dtype)
    alpha = (create_frame((args.height, args.width), args.dtype)
             if args.channels == 4 else None)
    mask = create_frame((args.height, args.width), np.uint8)
    color_path = root / f'{name}-color.tif'
    alpha_path = root / f'{name}-alpha.tif' if alpha is not None else None
    invalid = 0
    try:
        # Descriptor writes, never assignment into the mappings: generated pages
        # must not become resident alongside the decoded ones.
        with ExitStack() as files:
            color_file = files.enter_context(open(color.filename, 'r+b'))
            mask_file = files.enter_context(open(mask.filename, 'r+b'))
            alpha_file = (files.enter_context(open(alpha.filename, 'r+b'))
                          if alpha is not None else None)
            for y in range(0, args.height, args.band_rows):
                end = min(y + args.band_rows, args.height)
                band, valid = source_band(np, args, y, end, offset_x, offset_y)
                np.ascontiguousarray(
                    band if args.channels == 1 else band[..., :3]).tofile(color_file)
                if alpha_file is not None:
                    np.ascontiguousarray(band[..., 3]).tofile(alpha_file)
                valid.tofile(mask_file)
                invalid += int(np.count_nonzero(valid == 0))
        if not imwrite_auto(str(color_path), color, exif=b'', stream_tiff=True):
            raise OSError(f'Encoded source TIFF write failed: {color_path}')
        if alpha is not None and not imwrite_auto(str(alpha_path), alpha, exif=b'',
                                                  stream_tiff=True):
            raise OSError(f'Encoded alpha TIFF write failed: {alpha_path}')
    finally:
        # Generation owners are released before the real decoder runs.
        del color, alpha
        gc.collect()
    return color_path, alpha_path, mask, invalid


def decode_encoded_source(path):
    """Decode one page exactly as a mosaic import does, admission included."""
    from core.image_loader import iter_decoded_sources
    from core.resource_policy import require_memory
    from core.source_frames import frame_resolution_bytes, scan_source
    headers = scan_source(os.fspath(path), header_only=True)
    admission = max((frame_resolution_bytes(frame) for frame in headers), default=0)
    require_memory(admission)
    meta = {
        'path': os.path.abspath(path),
        'decoder': 'core.image_loader.iter_decoded_sources',
        'admission_bytes': int(admission),
        'admission_headers': [
            {'path': os.path.abspath(frame.path), 'page': frame.page,
             'native_shape': list(frame.native_shape or frame.shape),
             'dtype': str(frame.dtype), 'grayscale': bool(frame.grayscale),
             'orientation': frame.orientation} for frame in headers],
        'scale_factor': 1.0, 'thread_count': 1, 'disk_backed': True,
    }
    decoder = iter_decoded_sources([os.fspath(path)], 1.0, thread_count=1,
                                   allow_transparent_png=True, disk_backed=True)
    try:
        item = next(decoder)
        if item.error is not None:
            raise item.error
        if len(item.pages) != 1:
            raise ValueError(f'Encoded source must decode to one page: {path}')
        page = item.pages[0]
        meta.update({'pages': len(item.pages),
                     'returned_storage': type(page).__name__,
                     'decoded_shape': list(page.shape),
                     'decoded_dtype': str(page.dtype)})
        return page, meta
    finally:
        decoder.close()


def rebuild_encoded_frame(args, np, create_frame, color_path, alpha_path,
                          offset_x, offset_y):
    """Rebuild the original frame layout from genuinely decoded pages.

    Each row band re-derives the deterministic scene instead of reading the
    released generation mapping, so the comparison proves the TIFF codec and
    the production decoder preserved pixels, bit depth and alpha without
    keeping a second whole stack. Reconstructed rows are written through a
    descriptor, so the frame mapping's pages stay out of RSS exactly like the
    default generated path.
    """
    maximum = np.iinfo(args.dtype).max
    stats = {'compared_samples': 0, 'max_abs_difference': 0, 'non_257_samples': 0,
             'above_8bit_samples': 0, 'odd_samples': 0,
             'alpha_zero_samples': 0, 'alpha_partial_samples': 0,
             'alpha_full_samples': 0, 'expected_alpha_zero_samples': 0,
             'expected_alpha_partial_samples': 0, 'expected_alpha_full_samples': 0,
             'grayscale_planes_identical': None}
    page, color_meta = decode_encoded_source(color_path)
    alpha, alpha_meta = (decode_encoded_source(alpha_path) if alpha_path is not None
                         else (None, None))
    shape = ((args.height, args.width) if args.channels == 1
             else (args.height, args.width, args.channels))
    image = create_frame(shape, args.dtype)
    try:
        with open(image.filename, 'r+b') as target:
            for y in range(0, args.height, args.band_rows):
                end = min(y + args.band_rows, args.height)
                expected, _valid = source_band(np, args, y, end, offset_x, offset_y)
                band = page[y:end]
                if band.ndim != 3 or band.shape[2] != 3:
                    raise ValueError(f'Encoded colour page decoded to {band.shape}')
                if band.dtype != np.dtype(args.dtype):
                    raise ValueError(f'Encoded colour page decoded as {band.dtype}')
                if args.channels == 1:
                    same = (np.array_equal(band[..., 0], band[..., 1])
                            and np.array_equal(band[..., 0], band[..., 2]))
                    stats['grayscale_planes_identical'] = (
                        same if stats['grayscale_planes_identical'] is None
                        else stats['grayscale_planes_identical'] and same)
                    colors = band[..., 0]
                    expected_colors = expected
                else:
                    colors = band[..., :3]
                    expected_colors = expected[..., :3]
                stats['compared_samples'] += int(colors.size)
                delta = np.abs(colors.astype(np.int32) - expected_colors.astype(np.int32))
                stats['max_abs_difference'] = max(stats['max_abs_difference'],
                                                  int(delta.max()))
                if args.dtype == 'uint16':
                    stats['non_257_samples'] += int(np.count_nonzero(colors % 257))
                    stats['above_8bit_samples'] += int(np.count_nonzero(colors > 255))
                    stats['odd_samples'] += int(np.count_nonzero(colors % 2))
                if args.channels != 4:
                    np.ascontiguousarray(colors).tofile(target)
                    continue
                if alpha is None:
                    raise ValueError('A 4-channel encoded source has no alpha page')
                plane = alpha[y:end]
                if (plane.ndim != 3 or not np.array_equal(plane[..., 0], plane[..., 1])
                        or not np.array_equal(plane[..., 0], plane[..., 2])):
                    raise ValueError(f'Encoded alpha page decoded to {plane.shape}')
                if plane.dtype != np.dtype(args.dtype):
                    raise ValueError(f'Encoded alpha page decoded as {plane.dtype}')
                plane = plane[..., 0]
                stats['compared_samples'] += int(plane.size)
                delta = np.abs(plane.astype(np.int32) - expected[..., 3].astype(np.int32))
                stats['max_abs_difference'] = max(stats['max_abs_difference'],
                                                  int(delta.max()))
                stats['alpha_zero_samples'] += int(np.count_nonzero(plane == 0))
                stats['alpha_partial_samples'] += int(
                    np.count_nonzero((plane > 0) & (plane < maximum)))
                stats['alpha_full_samples'] += int(np.count_nonzero(plane == maximum))
                expected_alpha = expected[..., 3]
                stats['expected_alpha_zero_samples'] += int(np.count_nonzero(expected_alpha == 0))
                stats['expected_alpha_partial_samples'] += int(
                    np.count_nonzero((expected_alpha > 0) & (expected_alpha < maximum)))
                stats['expected_alpha_full_samples'] += int(np.count_nonzero(expected_alpha == maximum))
                merged = np.empty((end - y, args.width, 4), args.dtype)
                merged[..., :3] = colors
                merged[..., 3] = plane
                merged.tofile(target)
    finally:
        page = None
        alpha = None
        gc.collect()
    return image, stats, color_meta, alpha_meta


def generate_encoded(args, report, meter, np, tifffile, create_frame, make_frame,
                     imwrite_auto, root):
    """Generate the same scene as encoded TIFFs and decode it back for real."""
    root.mkdir(parents=True, exist_ok=True)
    frames, transforms, sources = [], [], []
    for row in range(args.rows):
        for col in range(args.cols):
            name = f'r{row}c{col}'
            offset_x, offset_y = col * args.step_x, row * args.step_y
            write_stage, read_stage = f'encoded.write.{name}', f'encoded.load.{name}'
            with meter.timed(write_stage):
                color_path, alpha_path, mask, invalid = encode_source(
                    args, np, create_frame, imwrite_auto, root, name, offset_x, offset_y)
            with meter.timed(read_stage):
                image, stats, color_meta, alpha_meta = rebuild_encoded_frame(
                    args, np, create_frame, color_path, alpha_path, offset_x, offset_y)
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2, 2] = (args.origin[0] + offset_x, args.origin[1] + offset_y)
            frames.append(make_frame(name, image, str(color_path), mask=mask,
                                     preview_size=256))
            transforms.append(matrix)
            sources.append({
                'name': name,
                'color_path': os.path.abspath(color_path),
                'alpha_path': os.path.abspath(alpha_path) if alpha_path else None,
                'write_s': report['stages_s'][write_stage],
                'load_s': report['stages_s'][read_stage],
                'color_header': tiff_headers(tifffile, color_path),
                'alpha_header': tiff_headers(tifffile, alpha_path) if alpha_path else None,
                'decode': {'color': color_meta, 'alpha': alpha_meta},
                'native': {'shape': list(image.shape), 'dtype': str(image.dtype),
                           'channels': args.channels, 'alpha_page': bool(alpha_path),
                           'mask_invalid_pixels': invalid,
                           'mask_storage': 'generated frame mapping (production masks '
                                           'are painted, not decoded)',
                           'offset': [offset_x, offset_y],
                           'transform': matrix.tolist()},
                'roundtrip': stats,
            })
            expected_samples = args.height * args.width * args.channels
            require(report, f'encoded.{name}.pixel_roundtrip',
                    stats['compared_samples'] == expected_samples
                    and stats['max_abs_difference'] == 0,
                    compared_samples=stats['compared_samples'],
                    expected_samples=expected_samples,
                    max_abs_difference=stats['max_abs_difference'],
                    color_path=sources[-1]['color_path'])
            expected_bits = np.dtype(args.dtype).itemsize * 8
            color_tags = sources[-1]['color_header']['tags']
            color_bits = color_tags['BitsPerSample']
            color_samples = 1 if args.channels == 1 else 3
            header_ok = (
                sources[-1]['color_header']['pages'] == 1
                and color_tags['ImageWidth'] == args.width
                and color_tags['ImageLength'] == args.height
                and color_tags['SamplesPerPixel'] == color_samples
                and color_tags['PhotometricInterpretation'] == (1 if color_samples == 1 else 2)
                and color_tags['Compression'] == 1
                and (color_bits in (expected_bits, [expected_bits]) if color_samples == 1
                     else color_bits == [expected_bits] * color_samples))
            alpha_tags = sources[-1]['alpha_header']['tags'] if alpha_path else None
            if alpha_tags is not None:
                alpha_bits = alpha_tags['BitsPerSample']
                header_ok = header_ok and (
                    sources[-1]['alpha_header']['pages'] == 1
                    and alpha_tags['SamplesPerPixel'] == 1
                    and alpha_tags['PhotometricInterpretation'] == 1
                    and alpha_tags['Compression'] == 1
                    and alpha_bits in (expected_bits, [expected_bits]))
            require(report, f'encoded.{name}.source_tiff_header', header_ok,
                    color_tags=color_tags, alpha_tags=alpha_tags)
            if args.dtype == 'uint16':
                require(report, f'encoded.{name}.bit_depth_roundtrip',
                        stats['above_8bit_samples'] > 0 and stats['non_257_samples'] > 0
                        and stats['odd_samples'] > 0,
                        above_8bit_samples=stats['above_8bit_samples'],
                        non_257_samples=stats['non_257_samples'],
                        odd_samples=stats['odd_samples'])
            if args.channels == 1:
                require(report, f'encoded.{name}.grayscale_layout',
                        stats['grayscale_planes_identical'] is True,
                        grayscale_planes_identical=stats['grayscale_planes_identical'])
            if args.channels == 4:
                require(report, f'encoded.{name}.alpha_roundtrip',
                        all(stats[f'alpha_{kind}_samples'] == stats[f'expected_alpha_{kind}_samples']
                            for kind in ('zero', 'partial', 'full')),
                        actual={kind: stats[f'alpha_{kind}_samples']
                                for kind in ('zero', 'partial', 'full')},
                        expected={kind: stats[f'expected_alpha_{kind}_samples']
                                  for kind in ('zero', 'partial', 'full')})
            require(report, f'encoded.{name}.decoded_page',
                    color_meta['pages'] == 1
                    and color_meta['decoded_dtype'] == args.dtype
                    and color_meta['decoded_shape'][:2] == [args.height, args.width],
                    decoded_shape=color_meta['decoded_shape'],
                    decoded_dtype=color_meta['decoded_dtype'],
                    admitted_bytes=color_meta['admission_bytes'])
    if args.channels == 4:
        alpha_counts = {kind: sum(item['roundtrip'][f'alpha_{kind}_samples'] for item in sources)
                        for kind in ('zero', 'partial', 'full')}
        require(report, 'encoded.alpha_fixture_coverage',
                all(count > 0 for count in alpha_counts.values()), **alpha_counts)
    write_total = sum(item['write_s'] for item in sources)
    load_total = sum(item['load_s'] for item in sources)
    report['stages_s']['encoded.write'] = write_total
    report['stages_s']['encoded.load'] = load_total
    report['encoded_inputs'] = {
        'enabled': True,
        'completed': True,
        'scratch_directory': os.path.abspath(root),
        'writer': {'callable': 'utils.image_utils.imwrite_auto', 'stream_tiff': True,
                   'compression': 'uncompressed', 'exif': 'stripped',
                   'icc_profile': 'stripped',
                   'note': 'synthetic lossless TIFF pages; no camera RAW/JPEG encoder'},
        'decoder': {
            'callable': 'core.image_loader.iter_decoded_sources',
            'chain': 'iter_decoded_sources -> _decode_one_source -> decode_image -> '
                     'decode_tiff_page (cv2.imdecodemulti, IMREAD_UNCHANGED)',
            'admission': 'core.source_frames.scan_source(header_only=True) then '
                         'core.resource_policy.require_memory(frame_resolution_bytes)',
            'snapshot': 'disk_backed=True, one page per path, then one owned disk-backed '
                        'frame mapping per source (the same independent snapshot contract '
                        'as store_frame(prefer_ram=False)); admission matches '
                        'dialogs/mosaic.py::_decode_pages for mosaic imports',
            'provenance_sha256': {
                relative: hashlib.sha256(
                    (args.repo_root / relative).read_bytes()).hexdigest()
                for relative in ('core/image_loader.py', 'core/source_frames.py')},
            'alpha_and_mask': '4-channel alpha is a sidecar grayscale page because the '
                              'production TIFF loader rejects non-opaque alpha pages and '
                              'drops alpha from pages it decodes; masks stay generated '
                              'because production masks are painted, never decoded',
        },
        'write_s': write_total, 'load_s': load_total,
        'timing_note': 'write_s sums synthetic row-band generation, mask generation and the '
                       'streamed TIFF encode per source; load_s sums admission, production '
                       'decode, layout reconstruction and row-band verification per source; '
                       'the preview thumbnail is outside both timers',
        'source_file_bytes': sum(
            item['color_header']['file_bytes']
            + (item['alpha_header']['file_bytes'] if item['alpha_header'] else 0)
            for item in sources),
        'sources': sources,
    }
    require(report, 'encoded.sources',
            len(sources) == args.rows * args.cols
            and all(item['write_s'] > 0 and item['load_s'] > 0 for item in sources),
            sources=len(sources), write_s=write_total, load_s=load_total)
    return frames, transforms


def differences(np, first, second, band_rows):
    if first.shape != second.shape or first.dtype != second.dtype:
        return {'compatible': False, 'first_shape': list(first.shape),
                'second_shape': list(second.shape), 'first_dtype': str(first.dtype),
                'second_dtype': str(second.dtype)}
    maxima = np.zeros(first.shape[2], np.int64)
    sums = np.zeros(first.shape[2], np.float64)
    changed = np.zeros(first.shape[2], np.int64)
    for y in range(0, first.shape[0], band_rows):
        delta = np.abs(first[y:y+band_rows].astype(np.int32)
                       - second[y:y+band_rows].astype(np.int32))
        maxima = np.maximum(maxima, delta.max(axis=(0, 1)))
        sums += delta.sum(axis=(0, 1), dtype=np.float64)
        changed += np.count_nonzero(delta, axis=(0, 1))
    count = first.shape[0] * first.shape[1]
    return {'compatible': True, 'max_abs_by_channel': maxima.tolist(),
            'mean_abs_by_channel': (sums / count).tolist(),
            'changed_samples_by_channel': changed.tolist(), 'pixels': count}


def tiff_bands(np, page, band_rows):
    """Use tifffile's real strip decoder, never a full decoded image."""
    if page.is_tiled or page.planarconfig != 1 or page.rowsperstrip > 128:
        raise ValueError('Export must use contiguous strips of at most 128 rows')
    expected_y = 0
    strip_bytes = page.imagewidth * page.samplesperpixel * page.dtype.itemsize * page.rowsperstrip
    for decoded, position, _shape in page.segments(maxworkers=1, buffersize=strip_bytes):
        if (decoded is None or position[0] != 0 or position[1] != 0
                or position[2] != expected_y or position[3] != 0 or position[4] != 0):
            raise ValueError('Unexpected TIFF strip geometry or missing strip')
        strip = decoded[0]
        if strip.shape[1] != page.imagewidth:
            raise ValueError('TIFF strip width does not match its header')
        height = min(strip.shape[0], page.imagelength - expected_y)
        for start in range(0, height, band_rows):
            raw = strip[start:min(start+band_rows, height)]
            order = [2, 1, 0, 3] if raw.shape[2] == 4 else [2, 1, 0]
            yield expected_y + start, raw[..., order]
        expected_y += height
    if expected_y != page.imagelength:
        raise ValueError('Decoded TIFF strips do not span the declared image')


def read_export(np, tifffile, path, expected, band_rows):
    """TIFF reader is independent of production decoder/export implementation."""
    from core.frame_storage import frame_region
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        tags = {}
        for name in ('ImageWidth', 'ImageLength', 'BitsPerSample', 'SamplesPerPixel',
                     'PhotometricInterpretation', 'ExtraSamples', 'Compression',
                     'RowsPerStrip', 'PlanarConfiguration', 'SampleFormat'):
            tag = page.tags.get(name)
            value = tag.value if tag is not None else None
            if isinstance(value, tuple):
                value = [int(v) for v in value]
            elif value is not None:
                value = int(value)
            tags[name] = value
        profile = page.tags.get('InterColorProfile')
        tags['ICC_sha256'] = hashlib.sha256(profile.value).hexdigest() if profile else None
        pages, bigtiff = len(container.pages), bool(container.is_bigtiff)
        exact = True
        max_difference = lowbits = visible = 0
        alpha_zero = alpha_partial = alpha_full = poison = 0
        digest = hashlib.sha256()
        compatible = page.shape == expected.shape and page.dtype == expected.dtype
        if compatible:
            maximum = np.iinfo(expected.dtype).max
            for y, bgr in tiff_bands(np, page, band_rows):
                digest.update(bgr.tobytes())
                with frame_region(expected, (slice(y, y + bgr.shape[0]), slice(None))) as region:
                    delta = np.abs(bgr.astype(np.int32) - region.astype(np.int32))
                max_difference = max(max_difference, int(delta.max()))
                exact = exact and not bool(np.any(delta))
                alpha = bgr[..., 3] if bgr.shape[2] == 4 else np.full(bgr.shape[:2], maximum, expected.dtype)
                valid = alpha > 0
                visible += int(np.count_nonzero(valid))
                alpha_zero += int(np.count_nonzero(alpha == 0))
                alpha_partial += int(np.count_nonzero((alpha > 0) & (alpha < maximum)))
                alpha_full += int(np.count_nonzero(alpha == maximum))
                colors = bgr[..., :3]
                if expected.dtype == np.uint16:
                    lowbits += int(np.count_nonzero((colors % 257 != 0) & valid[..., None]))
                sentinel = ((colors[..., 0] > .90 * maximum)
                            & (colors[..., 1] < .10 * maximum)
                            & (colors[..., 2] > .90 * maximum) & valid)
                poison += int(np.count_nonzero(sentinel))
        return {'path': str(path), 'file_bytes': path.stat().st_size, 'tags': tags,
                'pages': pages, 'bigtiff': bigtiff, 'shape': list(page.shape),
                'dtype': str(page.dtype), 'compatible': compatible,
                'roundtrip_exact': compatible and exact,
                'roundtrip_max_abs': max_difference if compatible else None,
                'bgr_sha256': digest.hexdigest() if compatible else None,
                'visible_pixels': visible, 'alpha_zero_pixels': alpha_zero,
                'alpha_partial_pixels': alpha_partial, 'alpha_full_pixels': alpha_full,
                'visible_non257_color_samples': lowbits, 'visible_poison_pixels': poison}


def compare_exports(np, tifffile, first_path, second_path, band_rows):
    from itertools import zip_longest
    with tifffile.TiffFile(first_path) as first, tifffile.TiffFile(second_path) as second:
        a, b = first.pages[0], second.pages[0]
        if a.shape != b.shape or a.dtype != b.dtype:
            return {'compatible': False, 'first_shape': list(a.shape), 'second_shape': list(b.shape)}
        maxima = np.zeros(a.shape[2], np.int64)
        sums = np.zeros(a.shape[2], np.float64)
        changed = np.zeros(a.shape[2], np.int64)
        count = 0
        for left, right in zip_longest(tiff_bands(np, a, band_rows), tiff_bands(np, b, band_rows)):
            if left is None or right is None or left[0] != right[0] or left[1].shape != right[1].shape:
                raise ValueError('Tile comparison TIFF strip geometry differs')
            delta = np.abs(left[1].astype(np.int32) - right[1].astype(np.int32))
            maxima = np.maximum(maxima, delta.max(axis=(0, 1)))
            sums += delta.sum(axis=(0, 1), dtype=np.float64)
            changed += np.count_nonzero(delta, axis=(0, 1))
            count += delta.shape[0] * delta.shape[1]
        return {'compatible': True, 'max_abs_by_channel': maxima.tolist(),
                'mean_abs_by_channel': (sums/count).tolist(),
                'changed_samples_by_channel': changed.tolist(), 'pixels': count}


def export_case(args, report, meter, used, np, tifffile, imwrite_auto,
                name, pixels, plan, path):
    path = reserve_path(path, used)
    with meter.timed(name + '.export'):
        if not imwrite_auto(str(path), pixels, exif=plan.exif,
                            icc_profile=plan.icc_profile, stream_tiff=True):
            raise OSError(f'TIFF export failed: {path}')
    with meter.timed(name + '.readback'):
        evidence = read_export(np, tifffile, path, pixels, args.band_rows)
    report['exports'][name] = evidence
    require(report, name + '.roundtrip_exact', evidence['roundtrip_exact'],
            max_abs=evidence['roundtrip_max_abs'])
    expected_bits = np.dtype(pixels.dtype).itemsize * 8
    tags = evidence['tags']
    bits = tags['BitsPerSample']
    require(report, name + '.actual_tiff_tags',
            evidence['pages'] == 1 and tags['ImageWidth'] == pixels.shape[1]
            and tags['ImageLength'] == pixels.shape[0]
            and tags['SamplesPerPixel'] == pixels.shape[2]
            and tags['PhotometricInterpretation'] == 2
            and (bits == expected_bits or bits == [expected_bits] * pixels.shape[2])
            and (pixels.shape[2] != 4 or tags['ExtraSamples'] == [2]), tags=tags)
    return evidence


def compact_diagnostics(args, report, meter, used, np, tifffile, api):
    make_frame, plan_mosaic, render_mosaic, RenderSpec, BlendSpec, OutputSpec, SeamSpec, imwrite_auto = api
    from core.frame_storage import create_frame
    n = 160
    source = create_frame((n, n, 4), np.uint16)
    source[:] = scene_band(np, n, 0, n, 0, 0, 4, np.uint16)
    source[100:132, 20:52, 3] = 32767
    source[20:52, 100:132, 3] = 0
    mask = create_frame((n, n), np.uint8)
    mask[:] = 255
    mask[20:60, 20:60] = 0
    source[20:60, 20:60, :3] = (65535, 0, 65535)
    matrix = np.eye(3)
    matrix[:2, 2] = (-13, 19)
    spec = RenderSpec(seam=SeamSpec(mode='feather', precision=1, band=8),
                      blend=BlendSpec(mode='feather', levels=1),
                      output=OutputSpec(dtype='uint16', alpha='on'), tile_size=32)
    with meter.timed('diagnostic.native'):
        frame = make_frame('native-lowbits-mask-alpha', source, mask=mask, preview_size=160)
        plan = plan_mosaic([frame], [matrix], spec=spec)
        first, origin = render_mosaic([frame], [matrix], plan=plan, tile_size=32)
    evidence = export_case(args, report, meter, used, np, tifffile, imwrite_auto,
                           'diagnostic.native', first, plan, artifact(args, '-native'))
    # Check known native coordinates, independently of plan's own returned size.
    require(report, 'diagnostic.integer_geometry', tuple(origin) == (-13, 19)
            and first.shape == source.shape, origin=list(origin), shape=list(first.shape))
    require(report, 'diagnostic.native_lowbits_exact',
            np.array_equal(first[70:95, 70:95, :3], source[70:95, 70:95, :3]),
            exported_non257_samples=evidence['visible_non257_color_samples'])
    require(report, 'diagnostic.mask_hole_alpha', bool(np.all(first[25:55, 25:55, 3] == 0)))
    require(report, 'diagnostic.source_zero_alpha', bool(np.all(first[25:47, 105:127, 3] == 0)))
    require(report, 'diagnostic.partial_alpha', bool(np.all(first[105:127, 25:47, 3] == 32767)))
    # Separate captured plans: never mutate a source while its plan is alive.
    second_source = create_frame(source.shape, source.dtype)
    for y in range(0, n, args.band_rows):
        second_source[y:y+args.band_rows] = source[y:y+args.band_rows]
    second_source[20:60, 20:60, :3] = (0, 65535, 0)
    second_source[20:52, 100:132, :3] = (65535, 65535, 0)
    with meter.timed('diagnostic.excluded_colors'):
        other = make_frame('changed-only-excluded-colors', second_source, mask=mask, preview_size=160)
        other_plan = plan_mosaic([other], [matrix], spec=spec)
        second, _ = render_mosaic([other], [matrix], plan=other_plan, tile_size=64)
        # RGB at alpha=0 is unspecified. Covered output must be invariant.
        changed = maximum = 0
        for y in range(0, n, args.band_rows):
            valid = first[y:y+args.band_rows, :, 3] > 0
            delta = np.abs(first[y:y+args.band_rows, :, :3].astype(np.int32)
                           - second[y:y+args.band_rows, :, :3].astype(np.int32))
            selected = delta[valid]
            if selected.size:
                maximum = max(maximum, int(selected.max()))
                changed += int(np.count_nonzero(selected))
        require(report, 'diagnostic.excluded_colors_invariant', changed == 0,
                changed_color_samples=changed, max_abs=maximum)
        require(report, 'diagnostic.excluded_colors_alpha_invariant',
                np.array_equal(first[..., 3], second[..., 3]))
    export_case(args, report, meter, used, np, tifffile, imwrite_auto,
                'diagnostic.excluded_colors', second, other_plan, artifact(args, '-excluded'))


def extra_scenarios(args, report, meter, used, np, tifffile, api):
    make_frame, plan_mosaic, render_mosaic, RenderSpec, BlendSpec, OutputSpec, SeamSpec, imwrite_auto = api
    from core.frame_storage import create_frame
    from core.mosaic_projection import CameraSpec, ProjectionSpec, source_coordinates
    from core.mosaic_hdr import HDRSpec
    n = 160
    base = create_frame((n, n, 3), np.uint16)
    base[:] = scene_band(np, n, 0, n, 0, 0, 3, np.uint16)
    frame = make_frame('pinhole-texture', base, preview_size=n)
    matrix = np.eye(3)
    camera = CameraSpec(n, n, hfov=65, yaw=12, pitch=-4, roll=3)
    common = RenderSpec(seam=SeamSpec(mode='feather', precision=2, band=8),
                        blend=BlendSpec(mode='feather', levels=1),
                        output=OutputSpec(dtype='uint16', alpha='on'), tile_size=32)
    for kind in ('rectilinear', 'cylindrical', 'equirectangular'):
        name = 'projection.' + kind
        projection = ProjectionSpec(kind, camera.focal, -7.375, 13.625)
        # Independent zero-lens pinhole + Ry Rx Rz forward calculation.
        sx, sy = np.meshgrid(np.array([20., 80., 139.]), np.array([20., 80., 139.]))
        rays = np.stack(((sx-(n-1)/2)/camera.focal,
                         (sy-(n-1)/2)/camera.focal, np.ones(sx.shape)), axis=-1)
        yaw, pitch, roll = np.deg2rad([camera.yaw, camera.pitch, camera.roll])
        ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]])
        rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]])
        rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                       [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        rays = rays @ (ry @ rx @ rz).T
        x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
        if kind == 'rectilinear':
            px, py = x/z, y/z
        elif kind == 'cylindrical':
            px, py = np.arctan2(x, z), y/np.hypot(x, z)
        else:
            px, py = np.arctan2(x, z), np.arctan2(y, np.hypot(x, z))
        px = projection.center_x + projection.focal * px
        py = projection.center_y + projection.focal * py
        mx, my, visible = source_coordinates(px, py, camera, projection)
        residual = float(np.max(np.hypot(mx-sx, my-sy)))
        require(report, name + '.independent_pinhole_inverse',
                bool(np.all(visible)) and residual < .5, max_error_px=residual)
        with meter.timed(name + '.render'):
            plan = plan_mosaic([frame], [matrix],
                               spec=replace(common, projection=projection, cameras=(camera,)))
            image, _ = render_mosaic([frame], [matrix], plan=plan, tile_size=32)
        exported = export_case(args, report, meter, used, np, tifffile, imwrite_auto,
                               name, image, plan, artifact(args, '-' + kind))
        require(report, name + '.actual_coverage', exported['visible_pixels'] > n*n//4,
                visible_pixels=exported['visible_pixels'])
        del image, plan
    exposures = (1/16, 1/4, 1.)
    frames = []
    for index, seconds in enumerate(exposures):
        pixels = create_frame((n, n, 4), np.uint16)
        for y in range(0, n, args.band_rows):
            end = min(n, y+args.band_rows)
            # Known camera response z=sqrt(radiance*time), wide scene radiance.
            energy = np.geomspace(.02, 6., n)[None, :, None]
            color = np.array([.8, 1., 1.2])[None, None, :]
            band = np.sqrt(np.clip(energy * seconds * color, 0, 1))
            pixels[y:end, :, :3] = np.rint(band * 65535).astype(np.uint16)
            pixels[y:end, :, 3] = 65535
            pixels[y:end, :12, 3] = 0
            pixels[y:end, 12:24, 3] = 32767
        frames.append(make_frame(f'bracket-{index}', pixels, exposure_time=seconds, preview_size=n))
    matrices = [np.eye(3) for _ in frames]
    for mode in ('exposure_fusion', 'debevec'):
        results = []
        for variant in range(2):
            name = f'hdr.{mode}.{variant}'
            hdr = HDRSpec(mode=mode, exposure_times=exposures, levels=2,
                          exposure_target=.35 if variant == 0 else .7,
                          tone_key=.08 if variant == 0 else .32)
            with meter.timed(name + '.render'):
                plan = plan_mosaic(frames, matrices,
                                   spec=replace(common, hdr=hdr, bracket_size=3,
                                                bracket_groups=((0, 1, 2),)))
                image, _ = render_mosaic(frames, matrices, plan=plan, tile_size=32)
            exported = export_case(args, report, meter, used, np, tifffile, imwrite_auto,
                                   name, image, plan, artifact(args, '-' + name))
            require(report, name + '.lowbits', exported['visible_non257_color_samples'] > 0,
                    non257_samples=exported['visible_non257_color_samples'])
            require(report, name + '.opacity', bool(np.all(image[:, :8, 3] == 0))
                    and bool(np.all((image[:, 15:21, 3] > 32767)
                                    & (image[:, 15:21, 3] < 65535))))
            results.append(image)
            del plan
        delta = differences(np, results[0], results[1], args.band_rows)
        require(report, f'hdr.{mode}.parameter_affects_pixels',
                delta['compatible'] and max(delta['max_abs_by_channel'][:3]) > 1,
                difference=delta)
        del results


def execute(args, report, used):
    sys.path.insert(0, str(args.repo_root.resolve()))
    with tempfile.TemporaryDirectory(prefix='openfocus-mosaic-benchmark-',
                                     dir=args.scratch_dir) as scratch:
        os.environ['OPENFOCUS_DATA_DIR'] = scratch
        Path(scratch, 'settings.ini').write_text('[usage]\nenabled=false\n', encoding='utf-8')
        import cv2
        import numpy as np
        import tifffile
        from threadpoolctl import threadpool_limits
        from core.frame_storage import configure_storage_directory, create_frame
        from core.resource_policy import Workload, plan_execution, heavy_job
        from utils import memory
        meter = Measurements(report, memory)
        meter.monitor.start()
        previous_threads = cv2.getNumThreads()
        configure_storage_directory(scratch)
        try:
            # Missing integration is a real failure, never a substitute renderer.
            from core.mosaic import (make_frame, plan_mosaic, render_mosaic, preview_mosaic,
                                     RenderSpec, BlendSpec, OutputSpec)
            from core.mosaic_seam import SeamSpec
            from utils.image_utils import imwrite_auto
            api = (make_frame, plan_mosaic, render_mosaic, RenderSpec, BlendSpec,
                   OutputSpec, SeamSpec, imwrite_auto)
            report['versions'] = {'python': sys.version, 'numpy': np.__version__,
                                  'opencv': cv2.__version__, 'tifffile': tifffile.__version__}
            # Metadata hashes identify inputs to the run; they are not assertions.
            report['implementation_sha256'] = {}
            for relative in ('core/mosaic.py', 'core/mosaic_blend.py', 'core/mosaic_seam.py',
                             'core/mosaic_photo.py', 'core/mosaic_projection.py',
                             'core/mosaic_hdr.py', 'core/resource_policy.py', 'utils/image_utils.py'):
                path = args.repo_root / relative
                report['implementation_sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            band_bytes = args.band_rows * args.width * (args.channels * np.dtype(args.dtype).itemsize + 80)
            execution = plan_execution(Workload(frames=args.rows*args.cols,
                height=args.height, width=args.width, channels=args.channels,
                itemsize=np.dtype(args.dtype).itemsize, fixed_bytes=band_bytes,
                per_worker_bytes=0, outer_tasks=1, parallel_layer='native'),
                user_threads=args.threads)
            report['execution_policy'] = asdict(execution)
            report['rss_limit_bytes'] = int(args.rss_limit_mib * 1024**2)
            cv2.setNumThreads(execution.native_threads)
            report['opencv_native_threads'] = cv2.getNumThreads()
            with heavy_job(), threadpool_limits(limits=execution.native_threads):
                with meter.timed('generation'):
                    if args.encoded_inputs:
                        frames, transforms = generate_encoded(
                            args, report, meter, np, tifffile, create_frame, make_frame,
                            imwrite_auto, Path(scratch, 'encoded-sources'))
                    else:
                        frames, transforms = generate(args, np, create_frame, make_frame)
                        report['encoded_inputs'] = {'enabled': False, 'completed': True}
                report['source_storage_bytes'] = sum(f.image.nbytes + f.mask.nbytes for f in frames)
                report['source_shape'] = list(frames[0].image.shape)
                report['source_dtype'] = str(frames[0].image.dtype)
                spec = RenderSpec(seam=SeamSpec(mode='dp', precision=args.seam_precision,
                                              band=args.seam_band),
                                  blend=BlendSpec(mode='multiband', levels=args.levels),
                                  output=OutputSpec(dtype=args.dtype, alpha='on'),
                                  tile_size=args.tile_sizes[0])
                with meter.timed('planning'):
                    plan = plan_mosaic(frames, transforms, spec=spec)
                report['plan'] = {'origin': list(plan.origin), 'size': list(plan.size),
                                  'label_shape': list(plan.labels.shape),
                                  'groups': [list(group) for group in plan.groups],
                                  'fractional_source_origin': list(args.origin)}
                expected_origin = [math.floor(v) for v in args.origin]
                expected_size = [math.ceil(args.origin[0]+args.width+args.step_x*(args.cols-1))-expected_origin[0],
                                 math.ceil(args.origin[1]+args.height+args.step_y*(args.rows-1))-expected_origin[1]]
                require(report, 'plan.independent_bounds', list(plan.origin) == expected_origin
                        and list(plan.size) == expected_size,
                        expected_origin=expected_origin, expected_size=expected_size)
                label_digest = hashlib.sha256()
                for y in range(0, plan.labels.shape[0], args.band_rows):
                    label_digest.update(plan.labels[y:y+args.band_rows].tobytes())
                report['plan']['labels_sha256'] = label_digest.hexdigest()
                with meter.timed('preview'):
                    preview = preview_mosaic(plan, max_edge=384)
                    preview_path = reserve_path(artifact(args, '-preview', '.png'), used)
                    if not imwrite_auto(str(preview_path), preview, exif=b''):
                        raise OSError(f'Preview export failed: {preview_path}')
                require(report, 'preview.bounded_uint8', preview.dtype == np.uint8
                        and max(preview.shape[:2]) <= 384 and preview.shape[2] in (3, 4),
                        shape=list(preview.shape), path=str(preview_path))
                output_paths = []
                for index, tile in enumerate(args.tile_sizes):
                    name = f'tile{tile}'
                    with meter.timed(name + '.render'):
                        image, origin = render_mosaic(frames, transforms, plan=plan, tile_size=tile)
                    require(report, name + '.same_plan_geometry', tuple(origin) == tuple(plan.origin)
                            and image.shape[:2] == (plan.size[1], plan.size[0]))
                    exported = export_case(args, report, meter, used, np, tifffile, imwrite_auto,
                        name, image, plan, args.output if index == 0 else artifact(args, f'-tile{tile}'))
                    require(report, name + '.visible_source_colors', exported['visible_poison_pixels'] == 0,
                            excluded_magenta_visible_pixels=exported['visible_poison_pixels'])
                    require(report, name + '.alpha_present', exported['alpha_zero_pixels'] > 0
                            and exported['alpha_full_pixels'] > 0,
                            zero=exported['alpha_zero_pixels'], full=exported['alpha_full_pixels'],
                            partial=exported['alpha_partial_pixels'])
                    if args.dtype == 'uint16':
                        require(report, name + '.native_lowbits_present', exported['visible_non257_color_samples'] > 0,
                                non257_samples=exported['visible_non257_color_samples'])
                    output_paths.append(exported['path'])
                    del image
                    gc.collect()
                with meter.timed('tile_comparison'):
                    difference = compare_exports(np, tifffile, *output_paths, args.band_rows)
                report['tile_difference'] = difference
                require(report, 'tile_phase.color', difference['compatible']
                        and max(difference['max_abs_by_channel'][:3]) <= args.difference_tolerance,
                        tolerance=args.difference_tolerance, difference=difference)
                require(report, 'tile_phase.alpha', difference['compatible']
                        and difference['max_abs_by_channel'][3] <= args.difference_tolerance,
                        tolerance=args.difference_tolerance, difference=difference)
                del plan, frames, transforms
                gc.collect()
                compact_diagnostics(args, report, meter, used, np, tifffile, api)
                if args.extra_scenarios:
                    extra_scenarios(args, report, meter, used, np, tifffile, api)
                gc.collect()
        finally:
            cv2.setNumThreads(previous_threads)
            configure_storage_directory(None)
            meter.finish()
            require(report, 'process.peak_rss', report['peak_rss_bytes'] <= args.rss_limit_mib*1024**2,
                    peak_rss_bytes=report['peak_rss_bytes'], limit_mib=args.rss_limit_mib)


def main():
    args = arguments()
    report = {'options': {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
              'stages_s': {}, 'exports': {}, 'invariants': [], 'errors': [],
              'encoded_inputs': {'enabled': bool(args.encoded_inputs),
                                 'completed': False},
              'acceptance_scope': {
                  'measured': ['fractional/nonzero planar bounds', 'same captured plan at two tile sizes',
                               'banded color and alpha differences', 'actual TIFF tags and exact independent readback',
                               'native uint16 low-bit survival', 'mask/alpha excluded-color invariance in compact case',
                               'small production preview', 'elapsed stages and process peak RSS',
                               'encoded-source TIFF write and production image-loader decode with row-band '
                               'pixel, bit-depth and alpha round-trip invariants, write/load timings and '
                               'per-source header metadata, only when --encoded-inputs is requested'],
                  'optional': ['three camera projections with independent zero-lens pinhole inverse and actual exports',
                               'exposure fusion and calibrated Debevec exports, opacity and parameter effects'],
                  'not_established': ['all P1-P5 acceptance', 'Laplacian reference equivalence',
                      'seam quality and third-source label correctness', 'photometric gain/field/vignette accuracy',
                      'ICC/EXIF policy correctness', 'manual/automatic CP fitting and outlier rejection',
                      'lens-distortion inverse accuracy', 'absolute HDR radiance/dynamic-range accuracy',
                      'GUI interaction, locales, cancellation and visible desktop proof',
                      'camera RAW, JPEG, HEIF or PNG decoder behavior and platform regression; encoded-source '
                      'admission is measured only for synthetic uncompressed TIFF when --encoded-inputs is set'],
                  'encoded_inputs_requested': bool(args.encoded_inputs),
                  'memory_note': 'RSS includes mapped pages; production renderer and OS decide residency. '
                                 'A limit failure is retained, never hidden by changing the pipeline.'}}
    used = set()
    started = time.perf_counter()
    status = 1
    try:
        reserve_path(args.report, used)
        if args.output.resolve().exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        execute(args, report, used)
        status = 0 if not report['errors'] else 1
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        report['exception'] = traceback.format_exc()
    report['elapsed_s'] = time.perf_counter() - started
    report['passed'] = status == 0
    report['artifacts_reserved'] = sorted(str(path) for path in used)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    # Refuse to overwrite a previous report even on an early failure.
    try:
        with args.report.open('x', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
    except OSError as exc:
        print(f'Report write failed: {exc}', file=sys.stderr)
        status = 1
        report['passed'] = False
        report['errors'].append(f'Report write failed: {exc}')
        encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
