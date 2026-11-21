"""Keep native video reads outside the GUI process so cancellation can stop them."""
import json
import multiprocessing
from pathlib import Path
import tempfile

import cv2
import numpy as np

from core import frame_storage, scratch_registry
from utils.logging_setup import get_logger


logger = get_logger(__name__)
_SPAWN = multiprocessing.get_context('spawn')


def _run_child(target, args, should_cancel):
    """Return the exit code, or None once a cancelled child has been reaped."""
    if should_cancel():
        return None
    process = _SPAWN.Process(target=target, args=args, name='OpenFocus video', daemon=True)
    try:
        process.start()
        while not should_cancel():
            process.join(.05)
            if process.exitcode is not None:
                return process.exitcode
        return None
    finally:
        # Only the child owns the native decoder. Cancellation does not depend
        # on read(), open(), or release() returning, and the parent owns cleanup.
        if process.pid is not None:
            if process.is_alive():
                process.kill()
            process.join()
        process.close()


def _decode_video_child(video_path, scale, pixels_path, result_path):
    from core.image_loader import ImageStackLoader

    layouts = []
    try:
        with pixels_path.open('wb', buffering=0) as stream:
            def append_frame(frame):
                frame_storage.require_disk_space(frame.nbytes, pixels_path.parent)
                frame.tofile(stream)
                layouts.append((frame.shape, frame.dtype.str))

            ok, message = ImageStackLoader()._decode_video(video_path, scale, append_frame)
    except Exception as exc:
        ok, message = False, str(exc)
    with result_path.open('w', encoding='utf-8') as stream:
        json.dump((ok, message, layouts if ok else []), stream)


def load_video(video_path, scale, should_cancel):
    """Return ordered video frames in storage owned by the stack on success."""
    directory = Path(tempfile.mkdtemp(prefix='openfocus-video-', dir=frame_storage.storage_directory()))
    pixels_path = directory / 'openfocus-video-frames.bin'
    result_path = directory / 'openfocus-video-result.json'
    images = []
    adopted = False
    try:
        for path in (directory, pixels_path, result_path):
            scratch_registry.register(path)
        # Wait for the child before reading the manifest or mapping pixels. A
        # blocked decoder cannot leave the supervising thread reading a pipe.
        code = _run_child(_decode_video_child, (video_path, scale, pixels_path, result_path),
                          should_cancel)
        if code is None or should_cancel():
            return False, 'CANCELLED', []
        if code != 0:
            return False, f'Video decoder exited unexpectedly (exit code {code})', []
        with result_path.open(encoding='utf-8') as stream:
            ok, message, layouts = json.load(stream)
        result_path.unlink()
        if not ok:
            return False, message, []

        offset = 0
        for shape, dtype in layouts:
            if should_cancel():
                return False, 'CANCELLED', []
            # ndarray.tofile writes C-order pixels regardless of input strides.
            strides = [0] * len(shape)
            frame_size = np.dtype(dtype).itemsize
            for axis in range(len(shape) - 1, -1, -1):
                strides[axis] = frame_size
                frame_size *= shape[axis]
            images.append(frame_storage.open_frame({
                'path': pixels_path, 'shape': shape, 'dtype': dtype, 'offset': offset,
                'strides': strides, 'cleanup_directory': directory,
            }, owned=True))
            offset += frame_size
        adopted = True
        return True, message, images
    finally:
        if not adopted:
            images.clear()
            try:
                pixels_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)
                directory.rmdir()
            except FileNotFoundError:
                pass  # The last mapped frame may have removed the directory.
            except OSError:
                logger.warning('Could not remove video import scratch', exc_info=True)


def _probe_video_child(video_path, dimensions):
    capture = cv2.VideoCapture(video_path)
    try:
        if capture.isOpened():
            dimensions[0] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            dimensions[1] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def probe_video_dimensions(video_path, should_cancel):
    """Inspect video geometry for presets while the scale dialog is open."""
    dimensions = _SPAWN.Array('i', 2, lock=False)
    code = _run_child(_probe_video_child, (video_path, dimensions), should_cancel)
    if code == 0 and not should_cancel() and all(value > 0 for value in dimensions):
        return tuple(dimensions)
    return None
