"""Integer pixel-domain boundary shared by the HF9-compatible methods."""
import cv2
import numpy as np

from core.frame_storage import create_transient_frame as create_frame


def frame_codes(source, img_resize, should_cancel):
    """Map u8 to v<<8, retain u16, and restore normalized internal floats."""
    from core.source_frames import resolve_frame
    from fusion_methods.openfocus_ab import _check_cancel

    _check_cancel(should_cancel)
    frame = resolve_frame(source)
    _check_cancel(should_cancel)
    if frame.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError(f"Unsupported fusion pixel type: {frame.dtype}")
    if img_resize is not None and frame.dtype.kind == 'f':
        # Interpolate the documented 0-255 samples before quantizing codes.
        # Quantizing first changes fractional area-resize contributions.
        frame = cv2.resize(np.asarray(frame, dtype=np.float32),
                           tuple(map(int, img_resize)), interpolation=cv2.INTER_AREA)
        _check_cancel(should_cancel)
        img_resize = None
    code = create_frame(frame.shape, np.int32)
    for start in range(0, frame.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(frame.shape[0], start + 128))
        if frame.dtype == np.uint8:
            code[rows] = frame[rows].astype(np.int32) << 8
        elif frame.dtype == np.uint16:
            code[rows] = frame[rows]
        else:
            # Float arrays here are the application's documented 0-255 domain,
            # not decoded float TIFFs (which must be rejected by the loader).
            code[rows] = np.rint(np.clip(frame[rows], 0, 255) * 257.0)
    if img_resize is not None:
        _check_cancel(should_cancel)
        resized = cv2.resize(np.asarray(code, dtype=np.float32),
                             tuple(map(int, img_resize)), interpolation=cv2.INTER_AREA)
        output = create_frame(resized.shape, np.int32)
        for start in range(0, output.shape[0], 128):
            _check_cancel(should_cancel)
            output[start:start + 128] = np.rint(resized[start:start + 128])
        return output
    return code


def gray_codes(code, should_cancel, *, u8=False):
    """Equal integer channel mean, in the original pixel code domain."""
    from fusion_methods.openfocus_ab import _check_cancel

    gray = create_frame(code.shape[:2], np.int32)
    for start in range(0, gray.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(gray.shape[0], start + 128))
        pixels = code[rows]
        if code.ndim == 2:
            gray[rows] = pixels
        elif code.shape[2] == 1:
            gray[rows] = pixels[:, :, 0]
        else:
            total = pixels[:, :, :3].sum(axis=2, dtype=np.int32)
            gray[rows] = ((total >> 8) // 3) << 8 if u8 else total // 3
    return gray


def frame_gray_codes(source, img_resize, should_cancel):
    """Prepare native gray without a full-size three-channel code copy."""
    from core.source_frames import resolve_frame
    from fusion_methods.openfocus_ab import _check_cancel

    _check_cancel(should_cancel)
    frame = resolve_frame(source)
    _check_cancel(should_cancel)
    if img_resize is None and frame.dtype in (np.uint8, np.uint16):
        from fusion_methods._hf9_kernels import gray_rows

        gray = create_frame(frame.shape[:2], np.uint16)
        scale = 256 if frame.dtype == np.uint8 else 1
        for start in range(0, gray.shape[0], 128):
            _check_cancel(should_cancel)
            gray_rows(frame, gray, start, min(gray.shape[0], start + 128), scale)
        return gray
    return gray_codes(frame_codes(frame, img_resize, should_cancel), should_cancel,
                      u8=frame.dtype == np.uint8)



def output_codes(code, frames, should_cancel, *, e37=False):
    """Return u8 or the common float32 domain consumed by quantize_fuse_output."""
    from fusion_methods.openfocus_ab import _check_cancel

    is_u8 = all(source.dtype == np.uint8 for source in frames)
    output = create_frame(code.shape, np.uint8 if is_u8 else np.float32)
    for start in range(0, code.shape[0], 128):
        _check_cancel(should_cancel)
        rows = slice(start, min(code.shape[0], start + 128))
        block = np.clip(code[rows], 0, 65535)
        if is_u8:
            values = block.astype(np.int32) >> 8
            if e37:
                if values.ndim == 3 and values.shape[2] == 4:
                    values[:, :, :3] = np.maximum(values[:, :, :3] - 1, 0)
                else:
                    values = np.maximum(values - 1, 0)
            output[rows] = values
        else:
            output[rows] = block.astype(np.float32) * (1.0 / 257.0)
    return output
