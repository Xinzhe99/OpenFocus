"""Measure real source opening through the first usable display preview.

Generation is explicit, deterministic and outside the measured loader process.
Like fusion_streaming_benchmark.py, --generate-dir must name a new directory;
completed JPEG fixtures are retained there. Profiles and preview scratch files
are temporary. RSS fields are bytes and load_s is seconds.
Pipe EOF ends reception; a successful result also requires a zero child exit.
Time and memory limits request cancellation before forced termination.
"""
import argparse
import gc
import hashlib
from io import BytesIO
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]


def _generate(options, directory, connection):
    """Encode one textured RGB image, then write identical bytes to distinct paths."""
    try:
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[name] = '1'
        import cv2
        import numpy as np
        from PIL import Image

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        started = time.perf_counter()
        width, height, count = options['width'], options['height'], options['count']
        rng = np.random.default_rng(733)
        # Smooth coloured regions, lighting and fine grain exercise JPEG decoding
        # without keeping a stack of full-resolution generation arrays alive.
        coarse = rng.integers(35, 220, (max(2, height // 48), max(2, width // 48), 3),
                              dtype=np.uint8)
        pixels = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        del coarse
        columns = np.linspace(0, 2 * np.pi, width, dtype=np.float32)
        lighting = (18 * np.sin(columns) + 10 * np.cos(columns * 3)).astype(np.int16)
        for top in range(0, height, 32):
            block = pixels[top:top + 32].astype(np.int16)
            block += rng.integers(-12, 13, block.shape, dtype=np.int16)
            block += lighting[None, :, None]
            np.clip(block, 0, 255, out=block)
            pixels[top:top + 32] = block
        del block, lighting, columns
        with BytesIO() as encoded:
            with Image.fromarray(pixels) as image:
                image.save(encoded, format='JPEG', quality=92, subsampling=2,
                           progressive=False, optimize=False)
            del pixels
            view = encoded.getbuffer()
            digest = hashlib.sha256(view).hexdigest()
            encoded_bytes = len(view)
            paths = [directory / f'frame{index:06d}.jpg' for index in range(count)]
            for path in paths:
                with path.open('xb') as stream:
                    stream.write(view)
                if path.stat().st_size != encoded_bytes:
                    raise RuntimeError(f'Incomplete generated JPEG: {path}')
            del view
        # Read the encoded fixture back, including its authoritative header.
        with paths[0].open('rb') as stream:
            readback_digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                readback_digest.update(chunk)
            readback = readback_digest.hexdigest()
        if readback != digest:
            raise RuntimeError('Generated JPEG readback hash differs')
        with Image.open(paths[0]) as image:
            header = {'shape': [image.height, image.width, len(image.getbands())],
                      'mode': image.mode, 'format': image.format}
            image.verify()
        if header != {'shape': [height, width, 3], 'mode': 'RGB', 'format': 'JPEG'}:
            raise RuntimeError(f'Unexpected generated JPEG header: {header}')
        gc.collect()
        connection.send({'event': 'result', 'report': {
            'generation_s': time.perf_counter() - started,
            'fixture_seed': 733, 'fixture_sha256': digest,
            'fixture_bytes_per_frame': encoded_bytes, 'fixture_header': header,
            'generation_retains_original_stack': False,
        }})
    except Exception as exc:
        connection.send({'event': 'error', 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        connection.close()


def _load(options, directory, scratch, cancelled, connection):
    """Run the production loader in a fresh process, without generation's HWM."""
    stop = threading.Event()
    send_lock = threading.Lock()
    monitor = None
    loader = None
    frames = []
    preview = None
    started = None
    report = {}

    def send(message):
        with send_lock:
            connection.send(message)

    try:
        os.environ['OPENFOCUS_DATA_DIR'] = scratch
        # Bound native libraries before imports as well as OpenCV itself.
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[name] = '1'
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from core import image_loader, source_frames
        from core.frame_storage import configure_storage_directory
        from utils import memory
        from utils.image_utils import to_display_uint8

        memory.configure_opencv_memory()
        configure_storage_directory(scratch)
        counts = {'full_resolution_request_count': 0, 'decode_image_count': 0,
                  'preview_decode_count': 0}
        original_full = source_frames.SourceFrame.full_resolution
        original_decode = image_loader.decode_image
        original_preview = source_frames._pil_preview

        def observed_full(frame):
            counts['full_resolution_request_count'] += 1
            return original_full(frame)

        def observed_decode(*args, **kwargs):
            counts['decode_image_count'] += 1
            return original_decode(*args, **kwargs)

        def observed_preview(*args, **kwargs):
            counts['preview_decode_count'] += 1
            return original_preview(*args, **kwargs)

        source_frames.SourceFrame.full_resolution = observed_full
        image_loader.decode_image = observed_decode
        source_frames._pil_preview = observed_preview
        peak = [0]

        def sample():
            rss, high_water = memory.process_rss_bytes(), memory.peak_rss_bytes()
            if rss <= 0 or high_water <= 0:
                raise RuntimeError('Process RSS measurement is unavailable')
            peak[0] = max(peak[0], rss)
            send({'event': 'sample', 'sampled_peak_rss': peak[0],
                  'process_high_water_rss': high_water})

        def sample_loop():
            try:
                while not stop.wait(0.01):
                    sample()
            except Exception as exc:
                cancelled.set()
                send({'event': 'error', 'error': f'RSS monitor failed: {exc}'})

        sample()
        monitor = threading.Thread(target=sample_loop, name='source-open-rss', daemon=True)
        monitor.start()
        paths = [str(directory / f'frame{index:06d}.jpg') for index in range(options['count'])]
        started = time.perf_counter()
        send({'event': 'started', 'started': started})
        loader = image_loader.ImageStackLoader()
        loader.should_cancel = cancelled.is_set
        ok, message, frames, names = loader.load_from_filepaths(paths, thread_count=1)
        if not ok:
            raise RuntimeError(message)
        # Access the real display path and perform its normal depth conversion;
        # timing includes loading preview pixels from their backing storage.
        preview = to_display_uint8(source_frames.display_image(frames[0]))
        report['load_s'] = time.perf_counter() - started
        report.update({
            'frame_count': len(frames), 'source_shape': list(frames[0].shape),
            'source_native_shape': list(frames[0].native_shape),
            'source_dtype': str(frames[0].dtype), 'source_color_order': 'BGR',
            'source_paths': [frame.path for frame in frames], 'frame_names': names,
            'first_preview_shape': list(preview.shape), 'first_preview_dtype': str(preview.dtype),
            'first_preview_min': int(preview.min()), 'first_preview_max': int(preview.max()),
            'first_preview_sha256': hashlib.sha256(preview.tobytes()).hexdigest(),
            'source_cache_bytes': source_frames.SOURCE_CACHE.bytes,
            'preview_ready_count': sum(source_frames.preview_ready(frame) for frame in frames),
            **counts,
        })
        expected_shape = (options['height'], options['width'], 3)
        if len(frames) != options['count'] or loader.failed_count:
            raise RuntimeError(f'Incomplete load: {message}')
        if names != [Path(path).name for path in paths] or loader.image_paths != paths:
            raise RuntimeError('Loader changed the requested frame order')
        for frame, path in zip(frames, paths):
            if (not isinstance(frame, source_frames.SourceFrame) or frame.path != path
                    or frame.shape != expected_shape or frame.native_shape != expected_shape
                    or frame.dtype != np.dtype('uint8') or frame.page != 0 or frame.orientation != 1):
                raise RuntimeError(f'Source metadata differs from generated JPEG: {path}')
            frame.validate_source()
        expected_preview = (max(1, round(options['height'] * min(1, source_frames.PREVIEW_EDGE / max(expected_shape)))),
                            max(1, round(options['width'] * min(1, source_frames.PREVIEW_EDGE / max(expected_shape)))), 3)
        if (not frames[0].preview_is_available or preview.dtype != np.uint8
                or preview.ndim != 3 or preview.shape[2] != 3
                or max(preview.shape[:2]) > 1024
                or any(abs(a - b) > 1 for a, b in zip(preview.shape, expected_preview))):
            raise RuntimeError('First preview is unavailable or has unexpected dimensions/depth')
        if options['width'] * options['height'] > 1 and int(preview.max()) == int(preview.min()):
            raise RuntimeError('Generated textured input produced a uniform preview')
        if counts['full_resolution_request_count'] or counts['decode_image_count'] or source_frames.SOURCE_CACHE.bytes:
            raise RuntimeError('Opening sources unexpectedly resolved full-resolution frames')
        sample()
        report['load_complete'] = True
        send({'event': 'result', 'report': report})
    except Exception as exc:
        if started is not None:
            report.setdefault('load_s', time.perf_counter() - started)
        send({'event': 'error', 'error': f'{type(exc).__name__}: {exc}', 'report': report})
    finally:
        stop.set()
        if monitor is not None:
            monitor.join()
        if 'original_full' in locals():
            source_frames.SourceFrame.full_resolution = original_full
            image_loader.decode_image = original_decode
            source_frames._pil_preview = original_preview
            frames.clear()
            if loader is not None:
                loader.images.clear()
            preview = None
            source_frames.SOURCE_CACHE.clear()
            source_frames.PREVIEW_CACHE.clear()
            gc.collect()
        if 'configure_storage_directory' in locals():
            configure_storage_directory(None)
        connection.close()


def _run_child(context, target, arguments, cancelled, options, report):
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*arguments, writer))
    started = None
    abort_at = None
    result_received = False
    process.start()
    writer.close()
    try:
        while True:
            try:
                ready = reader.poll(0.02)
                message = reader.recv() if ready else None
            except (EOFError, BrokenPipeError):
                break
            except OSError as exc:
                if getattr(exc, 'winerror', None) == 109:
                    break
                raise
            if message is not None:
                event = message['event']
                if event == 'sample':
                    for field in ('sampled_peak_rss', 'process_high_water_rss'):
                        report[field] = max(report[field] or 0, message[field])
                elif event == 'started':
                    started = message['started']
                elif event in ('result', 'error'):
                    report.update(message.get('report', {}))
                    result_received = True
                    if event == 'error':
                        report['errors'].append(message['error'])
                if event == 'sample' and options['max_rss_mib'] is not None:
                    limit = options['max_rss_mib'] * 1024 ** 2
                    if max(report['sampled_peak_rss'], report['process_high_water_rss']) > limit and abort_at is None:
                        report['errors'].append('Requested --max-rss-mib exceeded')
                        abort_at = time.perf_counter()
                        cancelled.set()
            if (started is not None and options['max_load_seconds'] is not None
                    and time.perf_counter() - started > options['max_load_seconds']
                    and not result_received and abort_at is None):
                report['errors'].append('Requested --max-load-seconds exceeded')
                abort_at = time.perf_counter()
                cancelled.set()
            if not process.is_alive() and not ready:
                break
            if abort_at is not None and time.perf_counter() - abort_at >= 0.5:
                if not report.get('load_complete') and started is not None:
                    report['load_s'] = time.perf_counter() - started
                report['terminated_for_limit'] = process.is_alive()
                break
        process.join(timeout=0.5)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        reader.close()
        if process.exitcode and abort_at is None:
            report['errors'].append(f'{target.__name__} process exited with code {process.exitcode}')
        if not result_received and abort_at is None:
            report['errors'].append(f'{target.__name__} process produced no result')
        process.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-dir', type=Path, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--json-output', type=Path, required=True)
    parser.add_argument('--max-load-seconds', type=float)
    parser.add_argument('--max-rss-mib', type=float)
    args = parser.parse_args()
    for name in ('width', 'height', 'count'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    for name in ('max_load_seconds', 'max_rss_mib'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be finite and positive')
    directory = args.generate_dir.resolve()
    output = args.json_output.resolve()
    if output == directory or output.is_relative_to(directory):
        parser.error('--json-output must be outside --generate-dir')
    options = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    report = {'options': options, 'load_s': None, 'sampled_peak_rss': None,
              'process_high_water_rss': None, 'frame_count': None, 'source_shape': None,
              'first_preview_shape': None, 'load_complete': False, 'errors': [],
              'memory_scope': 'fresh loader process including imports; generation is a separate process',
              'rss_sample_interval_s': 0.01,
              'decode_count_scope': 'production decode_image, SourceFrame.full_resolution and _pil_preview calls',
              'python': sys.version}
    context = multiprocessing.get_context('spawn')
    cancelled = context.Event()
    generated = False
    created = False
    try:
        directory.mkdir(parents=True, exist_ok=False)
        created = True
        _run_child(context, _generate, (options, directory), cancelled, options, report)
        if not report['errors']:
            generated = True
            with tempfile.TemporaryDirectory(prefix='openfocus-source-open-') as scratch:
                Path(scratch, 'settings.ini').write_text('[usage]\nenabled=false\n', encoding='utf-8')
                _run_child(context, _load, (options, directory, scratch, cancelled), cancelled, options, report)
        if args.max_load_seconds is not None and report['load_s'] is not None and report['load_s'] > args.max_load_seconds:
            if 'Requested --max-load-seconds exceeded' not in report['errors']:
                report['errors'].append('Requested --max-load-seconds exceeded')
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        if created and not generated:
            shutil.rmtree(directory)
    report['passed'] = report['load_complete'] and not report['errors']
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
