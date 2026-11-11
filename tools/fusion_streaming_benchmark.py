"""Reproducible fusion stage probe; run large cases serially on an idle machine.

Compare the same inputs/options in separate processes/checkouts, for example:
  python tools/fusion_streaming_benchmark.py --source-dir STACK --output serial.tif --threads 1
  python tools/fusion_streaming_benchmark.py --source-dir STACK --output parallel.tif --threads 8
Use --repo-root to execute this script against another checkout. Input generation
is explicit and excluded from load/fusion timings. The default loader retains
SourceFrame descriptors through public fuse; manifest/decode explicitly stage
pixels for comparison. No GUI or analytics starts. RSS fields are bytes and
stage timings are seconds; the process high-water includes fixture generation.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-dir', type=Path)
    source.add_argument('--generate-dir', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--method', choices=('openfocus_a', 'openfocus_b', 'openfocus_c'), default='openfocus_b')
    parser.add_argument('--entry', choices=('pipeline', 'adapter'), default='pipeline',
                        help='adapter matches direct _fuse_openfocus_b: no pipeline exposure/sort/base steps')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--alignment', choices=('none', 'openfocus', 'ecc', 'homography'), default='none')
    parser.add_argument('--load-mode', choices=('decode', 'manifest', 'loader'), default='loader')
    parser.add_argument('--radius', type=int, help='registered A/B radius; defaults to registry value')
    parser.add_argument('--smoothing', type=float, help='registered smoothing value')
    parser.add_argument('--exposure-match', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--base-frame', choices=('none', 'first', 'last'), default='none')
    parser.add_argument('--sort-mode', choices=('asc', 'desc', 'off'), default='asc')
    parser.add_argument('--tile-enabled', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tile-threshold', type=int, default=2048)
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--tile-overlap', type=int, default=256)
    parser.add_argument('--resource-preference', choices=('balanced', 'responsive', 'maximum'), default='balanced')
    parser.add_argument('--width', type=int, default=4000)
    parser.add_argument('--height', type=int, default=4000)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--channels', type=int, choices=(1, 3), default=1)
    parser.add_argument('--dtype', choices=('uint8', 'uint16'), default='uint16')
    parser.add_argument('--fixture', choices=('focus-halves', 'reported-4000'), default='focus-halves')
    parser.add_argument('--file-order', choices=('natural', 'lexical'))
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 7 if args.fixture == 'reported-4000' else 733
    if args.file_order is None:
        args.file_order = 'lexical' if args.fixture == 'reported-4000' else 'natural'
    if args.fixture == 'reported-4000' and (args.channels != 1 or args.dtype != 'uint16'):
        parser.error('reported-4000 requires --channels 1 --dtype uint16')
    if args.output.suffix.lower() not in ('.tif', '.tiff'):
        parser.error('--output must be a TIFF path')
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root))
    with tempfile.TemporaryDirectory(prefix='openfocus-fusion-benchmark-') as scratch:
        os.environ['OPENFOCUS_DATA_DIR'] = scratch
        Path(scratch, 'settings.ini').write_text('[usage]\nenabled=false\n', encoding='utf-8')
        import cv2
        import numpy as np
        from core.frame_storage import configure_storage_directory, store_frame
        from core.image_loader import admit_image_decode, decode_image
        from core.multi_focus_fusion import MultiFocusFusion, estimate_peak_bytes, memory_budget_bytes
        from fusion_methods import openfocus_ab
        from utils import memory
        from utils.image_utils import quantize_fuse_output, imwrite_auto
        from utils.sorting import natural_sort_key
        from core.algorithms import get_fusion, resolve_fusion_parameters
        get_fusion(args.method)
        if args.radius is not None and args.method == 'openfocus_c':
            parser.error('--radius is not supported by openfocus_c')
        overrides = {}
        if args.radius is not None:
            overrides['radius'] = args.radius
        if args.smoothing is not None:
            overrides['smoothing'] = args.smoothing
        try:
            parameters = resolve_fusion_parameters(args.method, overrides)
        except (ValueError, TypeError) as exc:
            parser.error(str(exc))
        configure_storage_directory(scratch)
        memory.configure_opencv_memory()

        def write_tiff(path, pixels):
            if pixels.ndim == 2 or pixels.shape[2] == 1:
                import tifffile
                plane = pixels if pixels.ndim == 2 else pixels[:, :, 0]
                tifffile.imwrite(str(path), plane, photometric='minisblack',
                                 rowsperstrip=128, metadata=None)
                return True
            return imwrite_auto(str(path), pixels, stream_tiff=True)

        report = {'options': vars(args).copy(), 'stages_s': {}, 'firstpasses': []}
        report['effective_fusion_parameters'] = parameters
        report['options'] = {key: str(value) if isinstance(value, Path) else value for key, value in report['options'].items()}
        report['commit'] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        report['python'] = sys.version
        report['opencv'] = cv2.__version__
        report['numpy'] = np.__version__
        report['opencv_native_threads'] = cv2.getNumThreads()
        implementation_paths = ('core/multi_focus_fusion.py', 'core/algorithm_adapters.py',
            'fusion_methods/openfocus_ab.py', 'fusion_methods/openfocus_a.py',
            'fusion_methods/openfocus_bc.py', 'fusion_methods/openfocus_depth.py',
            'fusion_methods/_depth_kernel.py')
        report['implementation_sha256'] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                                           for path in implementation_paths if (root / path).is_file()}
        if args.generate_dir:
            directory = args.generate_dir.resolve()
            directory.mkdir(parents=True, exist_ok=False)
            start = time.perf_counter()
            rng = np.random.default_rng(args.seed)
            if args.fixture == 'reported-4000':
                # Exact decoded pixels, naming, seed and default OpenCV TIFF
                # encoding from the supplied benchmark-4000x16-exact.py.
                # Stream generation instead of retaining N original arrays.
                base = rng.random((args.height // 20 + 1, args.width // 20 + 1)).astype(np.float32)
                base = cv2.resize(base, (args.width, args.height), interpolation=cv2.INTER_CUBIC) * 200 + 30
                for index in range(args.count):
                    image = cv2.GaussianBlur(base, (0, 0), 1.0 + index * 0.5)
                    image = np.clip(image + rng.random((args.height, args.width)).astype(np.float32) * 8,
                                    0, 255).astype(np.uint16)
                    ok, encoded_frame = cv2.imencode('.tif', image)
                    if not ok:
                        raise RuntimeError('Reported-fixture TIFF encode failed')
                    encoded_frame.tofile(str(directory / f'f{index}.tif'))
                del base, image, encoded_frame
                report['generation_retains_original_stack'] = False
            else:
                shape = (args.height, args.width) + ((3,) if args.channels == 3 else ())
                texture = rng.integers(0, 256, shape, dtype=np.uint8)
                blurred = cv2.GaussianBlur(texture, (0, 0), 3)
                columns = np.arange(args.width)[None, :]
                for index in range(args.count):
                    split = ((index + 1) * args.width) // (args.count + 1)
                    mask = columns < split
                    image = np.where(mask[:, :, None] if args.channels == 3 else mask, texture, blurred)
                    if args.dtype == 'uint16':
                        image = image.astype(np.uint16) * 256 + 5
                    if not write_tiff(directory / f'frame{index:03d}.tif', image):
                        raise RuntimeError('Synthetic TIFF write failed')
                del texture, blurred, image
            report['generation_s'] = time.perf_counter() - start
        else:
            directory = args.source_dir.resolve()
        extensions = ('.png', '.jpg', '.jpeg', '.jpe', '.jfif', '.webp', '.tif', '.tiff', '.bmp')
        order_key = (lambda path: path.name) if args.file_order == 'lexical' else (lambda path: natural_sort_key(path.name))
        paths = sorted((path for path in directory.iterdir() if path.suffix.lower() in extensions), key=order_key)
        if len(paths) < 2:
            raise ValueError('A benchmark requires at least two source images')
        if args.output.resolve() in paths:
            raise ValueError('Output must not replace a source image')
        def file_sha256(path):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        report['inputs'] = [{'path': str(path), 'bytes': path.stat().st_size,
                             'mtime_ns': path.stat().st_mtime_ns, 'sha256': file_sha256(path)}
                            for path in paths]
        stop = threading.Event()
        peak = [memory.process_rss_bytes()]

        def sample_rss():
            while not stop.wait(0.02):
                peak[0] = max(peak[0], memory.process_rss_bytes())

        monitor = threading.Thread(target=sample_rss, daemon=True)
        monitor.start()
        original_ordered = getattr(openfocus_ab, '_ordered_frame_results', None)
        if original_ordered is not None:
            def observed_ordered(count, produce, workers, should_cancel):
                lock = threading.Lock()
                state = {'workers': workers, 'frames': count, 'max_active': 0, 'producer_sum_s': 0.0}
                active = 0

                def observed_produce(index, cancelled):
                    nonlocal active
                    started = time.perf_counter()
                    with lock:
                        active += 1
                        state['max_active'] = max(state['max_active'], active)
                    try:
                        return produce(index, cancelled)
                    finally:
                        with lock:
                            active -= 1
                            state['producer_sum_s'] += time.perf_counter() - started

                started = time.perf_counter()
                iterator = original_ordered(count, observed_produce, workers, should_cancel)
                try:
                    for index, value in iterator:
                        yield index, value
                        del value
                finally:
                    iterator.close()
                    state['wall_s'] = time.perf_counter() - started
                    report['firstpasses'].append(state)
            openfocus_ab._ordered_frame_results = observed_ordered
        try:
            start = time.perf_counter()
            if args.load_mode == 'manifest':
                from core.source_frames import scan_source, materialize_frames
                sources = [frame for path in paths for frame in scan_source(str(path), header_only=True)]
                report['stages_s']['headers'] = time.perf_counter() - start
                start = time.perf_counter()
                frames = materialize_frames(sources)
            elif args.load_mode == 'loader':
                from core.image_loader import ImageStackLoader
                loader = ImageStackLoader()
                ok, message, frames, _names = loader.load_from_filepaths(
                    [str(path) for path in paths], thread_count=args.threads,
                    resource_preference=args.resource_preference)
                if not ok:
                    raise RuntimeError(message)
            else:
                frames = []
                for path in paths:
                    admit_image_decode(str(path))
                    image = decode_image(str(path), preserve_grayscale=True)
                    if image is None:
                        raise RuntimeError(f'Unable to decode {path}')
                    frames.append(store_frame(image))
                    del image
            report['stages_s']['load'] = time.perf_counter() - start
            report['source_shape'] = list(frames[0].shape)
            report['source_dtypes'] = sorted({str(frame.dtype) for frame in frames})
            is_16bit = any(np.dtype(frame.dtype) == np.dtype(np.uint16) for frame in frames)
            report['load_pixels_materialized'] = args.load_mode != 'loader'
            report['represented_stack_bytes'] = sum(frame.nbytes for frame in frames)
            report['decoded_stack_bytes'] = report['represented_stack_bytes'] if args.load_mode != 'loader' else None
            # No materialized processing stack exists on the descriptor path.
            report['processing_stack_bytes'] = None if args.load_mode == 'loader' else report['represented_stack_bytes']
            start = time.perf_counter()
            if args.alignment != 'none':
                from core.registration import ImageRegistration
                frames = ImageRegistration(method=args.alignment, defer_warps=args.load_mode == 'loader').process(
                    frames, thread_count=args.threads, resource_preference=args.resource_preference)
            report['stages_s']['alignment'] = time.perf_counter() - start
            fusion = MultiFocusFusion(args.method, use_gpu=False, exposure_match=args.exposure_match,
                base_frame=args.base_frame, sort_mode=args.sort_mode, fusion_parameters=parameters,
                tile_enabled=args.tile_enabled, tile_threshold=args.tile_threshold,
                tile_block_size=args.tile_size, tile_overlap=args.tile_overlap, resource_preference=args.resource_preference)
            from types import SimpleNamespace
            first = frames[0]
            # Also run against the serial baseline, whose estimate helper
            # expected ndarray.itemsize rather than SourceFrame.dtype.itemsize.
            # This metadata-only adapter does not alter either fusion path.
            header = SimpleNamespace(shape=first.shape, ndim=len(first.shape),
                dtype=first.dtype, itemsize=np.dtype(first.dtype).itemsize)
            report['serial_workspace_estimate'] = estimate_peak_bytes(args.method, [header] * len(frames))
            report['ram_ceiling_before_fusion'] = memory_budget_bytes()
            start = time.perf_counter()
            prepared = frames
            report['fusion_input_types'] = sorted({type(frame).__name__ for frame in prepared})
            report['stages_s']['source_binding'] = time.perf_counter() - start
            report['pipeline_exposure_sort_base_applied'] = args.entry == 'pipeline'
            start = time.perf_counter()
            if args.entry == 'pipeline':
                result = fusion.fuse(prepared, thread_count=args.threads)
            else:
                adapter = getattr(fusion, '_fuse_' + args.method)
                result = adapter(prepared, thread_count=args.threads, **parameters)
            report['stages_s']['fusion'] = time.perf_counter() - start
            report['fusion_outside_firstpass_s'] = report['stages_s']['fusion'] - sum(
                entry['wall_s'] for entry in report['firstpasses'])
            working_digest = hashlib.sha256()
            for row in range(0, result.shape[0], 128):
                working_digest.update(result[row:row + 128].tobytes())
            report['working_sha256'] = working_digest.hexdigest()
            report['working_dtype'] = str(result.dtype)
            report['working_shape'] = list(result.shape)
            start = time.perf_counter()
            output = quantize_fuse_output(result, is_16bit)
            report['stages_s']['quantization'] = time.perf_counter() - start
            digest = hashlib.sha256()
            for row in range(0, output.shape[0], 128):
                digest.update(output[row:row + 128].tobytes())
            report['output_sha256'] = digest.hexdigest()
            report['output_shape'] = list(output.shape)
            report['output_dtype'] = str(output.dtype)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            if not write_tiff(args.output, output):
                raise RuntimeError('Output write failed')
            report['stages_s']['write'] = time.perf_counter() - start
            import tifffile
            start = time.perf_counter()
            with tifffile.TiffFile(str(args.output)) as written:
                report['readback_shape'] = list(written.series[0].shape)
                report['readback_dtype'] = str(written.series[0].dtype)
                report['readback_page_count'] = len(written.pages)
            report['output_file_bytes'] = args.output.stat().st_size
            report['stages_s']['readback_metadata'] = time.perf_counter() - start
            if (report['readback_shape'] != report['output_shape']
                    or report['readback_dtype'] != report['output_dtype']):
                raise RuntimeError('Written TIFF metadata does not match fusion output')
            report['sampled_peak_rss'] = max(peak[0], memory.process_rss_bytes())
            report['process_high_water_rss'] = memory.peak_rss_bytes()
            report['timed_stage_total_s'] = sum(report['stages_s'].values())
            del output, result, prepared, frames, first
            if args.load_mode == 'manifest':
                del sources
            if args.load_mode == 'loader':
                del loader
            from core.source_frames import SOURCE_CACHE, PREVIEW_CACHE
            SOURCE_CACHE.clear()
            PREVIEW_CACHE.clear()
            gc.collect()
        finally:
            stop.set()
            monitor.join()
            if original_ordered is not None:
                openfocus_ab._ordered_frame_results = original_ordered
            configure_storage_directory(None)
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        print(encoded)
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(encoded + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
