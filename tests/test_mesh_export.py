"""Depth map -> textured OBJ export (utils/mesh_export.py)."""
import os

import numpy as np

from utils.mesh_export import export_obj


def test_obj_export_writes_consistent_mesh(tmp_path):
    h, w = 120, 200
    yy, xx = np.mgrid[0:h, 0:w]
    index01 = (xx / (w - 1)).astype(np.float32)          # ramp
    tex = np.full((h, w, 3), 40000, np.uint16)           # 16-bit texture
    out = export_obj(os.path.join(str(tmp_path), "模型.obj"), index01, tex,
                     relief=0.1, max_side=50)
    obj, mtl, png = out["files"]
    assert all(os.path.exists(p) for p in out["files"])
    lines = open(obj, encoding="utf-8").read().splitlines()
    v = [l for l in lines if l.startswith("v ")]
    vt = [l for l in lines if l.startswith("vt ")]
    f = [l for l in lines if l.startswith("f ")]
    assert len(v) == len(vt) == out["vertices"] == 50 * 30
    assert len(f) == out["faces"] == 2 * 49 * 29
    # every face index is a valid 1-based vertex
    ids = np.array([int(t.split("/")[0]) for l in f for t in l.split()[1:]])
    assert ids.min() == 1 and ids.max() == len(v)
    # relief: z spans 0 .. relief * longest side along the ramp
    z = np.array([float(l.split()[3]) for l in v])
    # area-downsampled endpoints average a few pixels of the ramp
    assert z.min() < 0.05 * 0.1 * w and z.max() > 0.95 * 0.1 * w
    assert "map_Kd 模型_texture.png" in open(mtl, encoding="utf-8").read()


def test_invert_flips_relief(tmp_path):
    index01 = np.zeros((20, 20), np.float32)
    out = export_obj(os.path.join(str(tmp_path), "m.obj"), index01,
                     np.zeros((20, 20, 3), np.uint8), relief=0.5, invert=True)
    z = [float(l.split()[3]) for l in open(out["files"][0]) if l.startswith("v ")]
    assert min(z) > 9.9  # 1.0 * 0.5 * 20
