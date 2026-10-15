"""Depth map + fused image -> textured 3D surface (Wavefront OBJ).

A focus stack already encodes a height field: the depth map says, per
pixel, which focal plane was sharp. Exporting it as a mesh with the fused
image as texture gives a quick 3D view of the subject (Blender, MeshLab,
Windows 3D Viewer, any glTF/OBJ tool) — Helicon Focus sells this as its
"3D model" feature.

Output: <name>.obj + <name>.mtl + <name>_texture.png side by side.
The mesh is a regular grid (two triangles per cell) downsampled so the
longest side has at most `max_side` vertices; z is the depth index scaled
to `relief` × the image's longest side.
"""
import os

import cv2
import numpy as np

from utils.image_utils import imwrite_auto, to_display_uint8


def build_height_field(index01: np.ndarray, max_side: int = 600,
                       smooth: int = 2) -> np.ndarray:
    """Downsample + despeckle the [0, 1] depth index for meshing."""
    h, w = index01.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    hf = np.asarray(index01, np.float32)
    if s < 1.0:
        hf = cv2.resize(hf, (max(2, int(round(w * s))), max(2, int(round(h * s)))),
                        interpolation=cv2.INTER_AREA)
    for _ in range(max(0, int(smooth))):
        hf = cv2.medianBlur(hf, 5)
    return hf


def export_obj(path: str, index01: np.ndarray, texture: np.ndarray,
               relief: float = 0.15, invert: bool = False,
               max_side: int = 600, max_texture: int = 4096) -> dict:
    """Write OBJ + MTL + texture. Returns {'vertices', 'faces', 'files'}.

    `invert` flips which end of the stack is "near" (the stack order is a
    capture convention, not something the pixels can tell)."""
    base, _ = os.path.splitext(path)
    obj_path, mtl_path = base + ".obj", base + ".mtl"
    tex_path = base + "_texture.png"
    name = os.path.basename(base)

    hf = build_height_field(index01, max_side=max_side)
    if invert:
        hf = 1.0 - hf
    gh, gw = hf.shape
    H, W = index01.shape[:2]
    longest = float(max(W, H))

    # vertices in image units: x right, y up, z towards the viewer
    xs = np.linspace(0.0, W, gw, dtype=np.float32)
    ys = np.linspace(0.0, H, gh, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    verts = np.stack([gx - W / 2.0, H / 2.0 - gy, hf * relief * longest], axis=-1)
    uu, vv = np.meshgrid(np.linspace(0, 1, gw, dtype=np.float32),
                         np.linspace(1, 0, gh, dtype=np.float32))
    uvs = np.stack([uu, vv], axis=-1)

    idx = np.arange(gh * gw).reshape(gh, gw) + 1  # OBJ is 1-based
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])

    tex = to_display_uint8(texture)
    if tex.ndim == 2:
        tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    th, tw = tex.shape[:2]
    if max(th, tw) > max_texture:
        k = max_texture / float(max(th, tw))
        tex = cv2.resize(tex, (int(tw * k), int(th * k)), interpolation=cv2.INTER_AREA)
    if not imwrite_auto(tex_path, tex):
        raise IOError(f"could not write {tex_path}")

    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl surface\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\n"
                f"map_Kd {os.path.basename(tex_path)}\n")
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"# OpenFocus depth surface ({gw}x{gh} grid)\n"
                f"mtllib {os.path.basename(mtl_path)}\no {name}\n")
        np.savetxt(f, verts.reshape(-1, 3), fmt="v %.3f %.3f %.3f")
        np.savetxt(f, uvs.reshape(-1, 2), fmt="vt %.5f %.5f")
        f.write("usemtl surface\n")
        np.savetxt(f, np.repeat(tris, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")
    return {"vertices": gh * gw, "faces": len(tris),
            "files": [obj_path, mtl_path, tex_path]}
