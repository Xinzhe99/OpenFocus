"""Shared relative/physical height fields and transactional OBJ file groups."""
import json
import os
import stat
import tempfile

import cv2
import numpy as np

from utils.image_utils import imwrite_auto, to_display_uint8


class MeshExportCancelled(Exception):
    """The file group was cancelled before committing."""


def physical_depth(index01, capture_indices, z0_um, step_um, direction=1):
    """Map participating-frame indices through their original capture positions."""
    captures = np.asarray(capture_indices, np.float64)
    if (captures.ndim != 1 or len(captures) < 2 or not np.isfinite(captures).all()
            or not np.isfinite(z0_um) or not np.isfinite(step_um) or step_um <= 0
            or direction not in (-1, 1)):
        raise ValueError("Valid capture indices and a positive finite Z step are required")
    positions = float(z0_um) + captures * float(step_um) * direction
    indices = np.asarray(index01, np.float32)
    depth = np.interp(indices * (len(captures) - 1), np.arange(len(captures)), positions)
    depth[~np.isfinite(indices)] = np.nan
    return depth.astype(np.float32), positions.tolist()


def build_height_field(index01: np.ndarray, max_side: int = 600,
                       smooth: int = 0) -> np.ndarray:
    """Sample the shared height field without quantizing or filling missing data."""
    h, w = index01.shape[:2]
    if min(h, w) < 2 or max_side < 2:
        raise ValueError("A surface needs at least a 2 by 2 domain")
    s = min(1.0, float(max_side) / max(h, w))
    hf = np.asarray(index01, np.float32)
    if s < 1.0:
        shape = (max(2, int(round(w * s))), max(2, int(round(h * s))))
        valid = np.isfinite(hf)
        weights = cv2.resize(valid.astype(np.float32), shape, interpolation=cv2.INTER_AREA)
        hf = cv2.resize(np.where(valid, hf, 0), shape, interpolation=cv2.INTER_AREA)
        hf = np.divide(hf, weights, out=np.full_like(hf, np.nan), where=weights > 0)
        hf[weights < 1.0 - 1e-6] = np.nan
    if smooth:
        from scipy.ndimage import median_filter
        mask = np.isfinite(hf)
        for _ in range(max(0, int(smooth))):
            hf = median_filter(hf, size=5, mode="nearest")
        hf[~mask] = np.nan
    return hf


def obj_paths(path):
    base, _ = os.path.splitext(os.fspath(path))
    return [base + ".obj", base + ".mtl", base + "_texture.png", base + ".json"]

def _identity(path):
    info = os.lstat(path)
    return (info.st_dev, info.st_ino, info.st_mode)



def _entry(path):
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if (not stat.S_ISREG(info.st_mode) or
            getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError(f'Cannot replace non-regular object: {path} (mode {stat.filemode(info.st_mode)})')
    return (info.st_dev, info.st_ino, info.st_mode)


def inspect_obj_targets(path):
    """Resolve the parent once and record the four authorized directory entries."""
    original_parent = os.path.abspath(os.path.dirname(os.fspath(path)) or '.')
    parent = os.path.realpath(original_parent)
    parent_info = os.stat(original_parent)
    files = obj_paths(os.path.join(parent, os.path.basename(os.fspath(path))))
    return files, tuple(_entry(name) for name in files), (original_parent, parent_info.st_dev, parent_info.st_ino)


def _check_authorization(files, entries, parent):
    info = os.stat(parent[0])
    if (info.st_dev, info.st_ino) != parent[1:]:
        raise ValueError(f'Output directory changed: {parent[0]}')
    resolved = os.stat(os.path.dirname(files[0]))
    if (resolved.st_dev, resolved.st_ino) != parent[1:]:
        raise ValueError(f'Output directory changed: {os.path.dirname(files[0])}')
    for name, expected in zip(files, entries):
        if _entry(name) != expected:
            raise ValueError(f'Output changed; confirm again: {name}')


def export_obj(path: str, index01: np.ndarray, texture: np.ndarray,
               relief: float = 0.15, invert: bool = False,
               max_side: int = 600, max_texture: int = 4096, *,
               height_field=None, pixel_size_um=None, metadata=None,
               overwrite=False, should_cancel=None, progress=None, authorization=None) -> dict:
    """Stage a complete file group, then commit it with rollback on failure.

    A supplied height_field is already sampled and processed, shared with the
    final preview. With pixel_size_um, its Z values are physical micrometres;
    relative relief/inversion must already be reflected in that field.
    """
    files, entries, parent = authorization or inspect_obj_targets(path)
    expected_files = obj_paths(os.path.join(os.path.realpath(parent[0]), os.path.basename(os.fspath(path))))
    if len(files) != 4 or len(entries) != 4 or files != expected_files:
        raise ValueError('Mesh output authorization does not match the selected path')
    _check_authorization(files, entries, parent)
    conflicts = [name for name, entry in zip(files, entries) if entry is not None]
    if conflicts and not overwrite:
        raise FileExistsError("\n".join(conflicts))

    def checkpoint(stage):
        if should_cancel is not None and should_cancel():
            raise MeshExportCancelled()
        if progress is not None:
            progress(stage)

    checkpoint("prepare")
    hf = (build_height_field(index01, max_side=max_side) if height_field is None
          else np.asarray(height_field, np.float32))
    if height_field is None and invert:
        hf = 1.0 - hf
    gh, gw = hf.shape
    if min(gh, gw) < 2:
        raise ValueError("A surface needs at least a 2 by 2 grid")
    H, W = index01.shape[:2]
    physical = pixel_size_um is not None
    if physical and (not np.isfinite(pixel_size_um) or pixel_size_um <= 0):
        raise ValueError("Physical OBJ needs a positive XY calibration")
    unit = float(pixel_size_um) if physical else 1.0
    xs = np.linspace(0, W - 1, gw, dtype=np.float64) * unit
    ys = np.linspace(0, H - 1, gh, dtype=np.float64) * unit
    gx, gy = np.meshgrid(xs, ys)
    z = hf if physical else hf * relief * max(W - 1, H - 1)
    valid = np.isfinite(hf)
    verts = np.stack([gx - xs[-1] / 2, ys[-1] / 2 - gy, z], axis=-1)[valid]
    uu, vv = np.meshgrid(np.linspace(0, 1, gw), np.linspace(1, 0, gh))
    uvs = np.stack([uu, vv], axis=-1)[valid]
    idx = np.zeros((gh, gw), np.int64)
    idx[valid] = np.arange(len(verts)) + 1
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # A cell touching missing data creates no triangles across its hole.
    keep = (a > 0) & (b > 0) & (c > 0) & (d > 0)
    tris = np.concatenate([np.stack([a[keep], c[keep], b[keep]], 1),
                           np.stack([b[keep], c[keep], d[keep]], 1)])
    if not len(tris):
        raise ValueError("No valid surface cells")
    tex = to_display_uint8(texture)
    if tex.ndim == 2:
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    th, tw = tex.shape[:2]
    if max(th, tw) > max_texture:
        scale = max_texture / float(max(th, tw))
        tex = cv2.resize(tex, (max(1, int(tw * scale)), max(1, int(th * scale))),
                         interpolation=cv2.INTER_AREA)
    meta = dict(metadata or {})
    meta.update(unit="um" if physical else "image_pixel", pixel_size_um=pixel_size_um,
                grid=[gw, gh], vertices=len(verts), faces=len(tris),
                relative_relief=None if physical else relief,
                origin="image centre; X right; Y up", missing="NaN; cells omitted")
    folder = os.path.dirname(os.path.abspath(files[0]))
    staging = tempfile.mkdtemp(prefix=".openfocus-mesh-", dir=folder)
    staged = [os.path.join(staging, os.path.basename(name)) for name in files]
    backups, committed, owned = {}, [], {}
    errors = []
    failure = None
    try:
        with open(staged[2], 'xb'):
            pass
        owned[staged[2]] = _entry(staged[2])
        checkpoint("texture")
        if not imwrite_auto(staged[2], tex):
            raise IOError(f"Could not write {files[2]}")
        owned[staged[2]] = _entry(staged[2])
        with open(staged[1], "x", encoding="utf-8") as stream:
            owned[staged[1]] = _entry(staged[1])
            stream.write("newmtl surface\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
                         f"map_Kd {os.path.basename(files[2])}\n")
        owned[staged[1]] = _entry(staged[1])
        checkpoint("model")
        with open(staged[0], "x", encoding="utf-8") as stream:
            owned[staged[0]] = _entry(staged[0])
            stream.write(f"# OpenFocus {gw}x{gh}; 1 unit = {meta['unit']}\n"
                         f"mtllib {os.path.basename(files[1])}\no surface\n")
            for start in range(0, len(verts), 16384):
                checkpoint("model")
                np.savetxt(stream, verts[start:start + 16384], fmt="v %.12g %.12g %.12g")
            np.savetxt(stream, uvs, fmt="vt %.9g %.9g")
            stream.write("usemtl surface\n")
            for start in range(0, len(tris), 16384):
                checkpoint("model")
                np.savetxt(stream, np.repeat(tris[start:start + 16384], 2, axis=1),
                           fmt="f %d/%d %d/%d %d/%d")
        owned[staged[0]] = _entry(staged[0])
        with open(staged[3], "x", encoding="utf-8") as stream:
            owned[staged[3]] = _entry(staged[3])
            json.dump(meta, stream, ensure_ascii=False, indent=2, allow_nan=False)
        owned[staged[3]] = _entry(staged[3])
        checkpoint("commit")
        _check_authorization(files, entries, parent)
        for target, expected in zip(files, entries):
            if expected is not None:
                backup = os.path.join(staging, "backup-" + os.path.basename(target))
                os.replace(target, backup)
                backups[target] = (backup, expected, None)
                moved_identity = _identity(backup)
                backups[target] = (backup, expected, moved_identity)
                if _entry(backup) != expected:
                    raise ValueError(f'Output changed during backup: {target}')
        for source, target in zip(staged, files):
            if _entry(target) is not None:
                raise ValueError(f'Output appeared during commit: {target}')
            identity = owned[source]
            if _entry(source) != identity:
                raise ValueError(f'Staged file changed: {source}')
            os.replace(source, target)
            committed.append((target, identity))
            del owned[source]
    except Exception as exc:
        failure = exc
        for target, identity in reversed(committed):
            try:
                if _entry(target) != identity:
                    raise ValueError('committed file changed')
                os.unlink(target)
            except Exception as rollback_error:
                errors.append(f'{target}: {rollback_error}')
        for target, (backup, identity, moved_identity) in backups.items():
            try:
                if moved_identity is None or _identity(backup) != moved_identity:
                    raise ValueError('backup identity changed')
                if os.path.lexists(target):
                    raise ValueError('restore destination occupied')
                os.replace(backup, target)
            except Exception as rollback_error:
                errors.append(f'{target} -> {backup}: {rollback_error}')
    else:
        for target, (backup, identity, moved_identity) in backups.items():
            try:
                if _entry(backup) != identity:
                    raise ValueError('backup changed')
                os.unlink(backup)
            except Exception as cleanup_error:
                errors.append(f'{target} -> {backup}: {cleanup_error}')
    finally:
        for name, identity in owned.items():
            try:
                if _entry(name) != identity:
                    raise ValueError('staged file changed')
                os.unlink(name)
            except Exception as cleanup_error:
                errors.append(f'{name}: {cleanup_error}')
        try:
            os.rmdir(staging)
        except OSError as cleanup_error:
            errors.append(f'Staging retained at {staging}: {cleanup_error}')
    if errors:
        raise OSError('\n'.join(([str(failure)] if failure else []) + errors)) from failure
    if failure is not None:
        raise failure
    return {"vertices": len(verts), "faces": len(tris), "files": files}
