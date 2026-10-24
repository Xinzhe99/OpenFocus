"""Shared relative/physical height fields and transactional OBJ file groups."""
import json
import os
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


def export_obj(path: str, index01: np.ndarray, texture: np.ndarray,
               relief: float = 0.15, invert: bool = False,
               max_side: int = 600, max_texture: int = 4096, *,
               height_field=None, pixel_size_um=None, metadata=None,
               overwrite=False, should_cancel=None, progress=None) -> dict:
    """Stage a complete file group, then commit it with rollback on failure.

    A supplied height_field is already sampled and processed, shared with the
    final preview. With pixel_size_um, its Z values are physical micrometres;
    relative relief/inversion must already be reflected in that field.
    """
    files = obj_paths(path)
    conflicts = [name for name in files if os.path.exists(name)]
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
    with tempfile.TemporaryDirectory(prefix=".openfocus-mesh-", dir=folder) as staging:
        staged = [os.path.join(staging, os.path.basename(name)) for name in files]
        checkpoint("texture")
        if not imwrite_auto(staged[2], tex):
            raise IOError(f"Could not write {files[2]}")
        with open(staged[1], "w", encoding="utf-8") as stream:
            stream.write("newmtl surface\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
                         f"map_Kd {os.path.basename(files[2])}\n")
        checkpoint("model")
        with open(staged[0], "w", encoding="utf-8") as stream:
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
        with open(staged[3], "w", encoding="utf-8") as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2, allow_nan=False)
        checkpoint("commit")
        backups, committed = {}, []
        try:
            for target in files:
                if os.path.exists(target):
                    if not overwrite:
                        raise FileExistsError(target)
                    backup = os.path.join(staging, "backup-" + os.path.basename(target))
                    os.replace(target, backup)
                    backups[target] = backup
            for source, target in zip(staged, files):
                os.replace(source, target)
                committed.append(target)
        except Exception:
            for target in committed:
                os.remove(target)
            for target, backup in backups.items():
                os.replace(backup, target)
            raise
    return {"vertices": len(verts), "faces": len(tris), "files": files}
