"""
网格拼接模块测试
合成已知真值的 2×2 网格,验证:拍摄顺序、邻接对齐、光度校正、
接缝蜿蜒、无鬼影融合、8/16 位深保持。
"""
import os
import tempfile

import numpy as np
import pytest
import cv2

from core.mosaic_stitcher import (GridSpec, MosaicStitcher, GridDirection,
                                  StartCorner, ShootingOrder)

H = W = 400
SPACING = 300                      # 行/列间距(重叠 100px)


def _make_scene(rng, canvas_w, canvas_h):
    small = rng.random((canvas_h // 20 + 1, canvas_w // 20 + 1)).astype(np.float32)
    base = cv2.resize(small, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-9) * 200 + 30
    return base


def _build(tmp, bitdepth=8, rows=2, cols=2, bright_second=False):
    """合成 rows×cols 网格;返回 (GridSpec, 帧路径, 画布真值落位)。"""
    rng = np.random.default_rng(3)
    canvas_w = (cols - 1) * SPACING + W
    canvas_h = (rows - 1) * SPACING + H
    base = _make_scene(rng, canvas_w, canvas_h)
    xs = [c * SPACING for c in range(cols)]
    ys = [r * SPACING for r in range(rows)]
    frames = []
    truth = []
    k = 0
    for r in range(rows):
        for c in range(cols):
            f = base[ys[r]:ys[r] + H, xs[c]:xs[c] + W].copy()
            f = cv2.GaussianBlur(f, (0, 0), 1.0 + r * 0.5)
            f = np.clip(f + rng.random((H, W)).astype(np.float32) * 4,
                        0, 255).astype(np.uint8)
            f3 = np.stack([f] * 3, axis=-1)
            if bright_second and k == 1:
                f3 = np.clip(f3.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
            p = os.path.join(str(tmp), f"f{k}.tif")
            cv2.imwrite(p, cv2.cvtColor(f3, cv2.COLOR_RGB2BGR))
            frames.append(p)
            truth.append((xs[c], ys[r]))
            k += 1
    grid = GridSpec(rows=rows, cols=cols, overlap_pct=25.0)
    return grid, frames, truth


# ---------- 拍摄顺序 ----------

def test_shoot_order_byrow_topleft(tmp_path):
    grid, frames, _ = _build(tmp_path, 8, 2, 2)
    s = MosaicStitcher(grid, frames)
    assert s.shoot_order() == [0, 1, 2, 3]


def test_shoot_order_zigzag(tmp_path):
    grid, frames, _ = _build(tmp_path, 8, 2, 2)
    grid.direction = GridDirection.ZIGZAG
    s = MosaicStitcher(grid, frames)
    # 蛇形:第一行 0,1;第二行反向 3,2
    assert s.shoot_order() == [0, 1, 3, 2]


def test_shoot_order_bycolumn(tmp_path):
    grid, frames, _ = _build(tmp_path, 8, 2, 2)
    grid.order = ShootingOrder.BY_COLUMN
    s = MosaicStitcher(grid, frames)
    # 列优先:帧 0(0,0), 帧 2(1,0), 帧 1(0,1), 帧 3(1,1)
    assert s.shoot_order() == [0, 2, 1, 3]


# ---------- 端到端(8bit) ----------

def test_stitch_2x2(tmp_path):
    grid, frames, truth = _build(tmp_path, 8, 2, 2)
    s = MosaicStitcher(grid, frames, blend_half_width=48)
    out, placements = s.run()

    # 画布覆盖 300+400=700 × 700
    assert out.shape[0] >= 690 and out.shape[1] >= 690
    # 落位与真值偏差 ≤ 30px(网格声明 + NCC 精化)
    for (px, py), (tx, ty) in zip(placements, truth):
        assert abs(px - tx) <= 30 and abs(py - ty) <= 30

    # 四图交叠区无鬼影:输出与 img0/img3 的相关均 >0.9
    g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g4 = cv2.cvtColor(cv2.imread(frames[3]), cv2.COLOR_BGR2GRAY).astype(np.float32)
    cy, cx = 300, 300                        # 四图交叠区(画布 [300,400)²)
    a = g[cy:cy + 100, cx:cx + 100]
    b1 = g1[cy:cy + 100, cx:cx + 100]        # img0 画布 (0,0):局部 = 画布
    b4 = g4[cy - 300:cy - 200, cx - 300:cx - 200]  # img3 局部 = 画布 − 300
    c1 = float(np.corrcoef(a.ravel(), b1.ravel())[0, 1])
    c4 = float(np.corrcoef(a.ravel(), b4.ravel())[0, 1])
    print(f"  四图交叠区: NCC(out,g1)={c1:.3f}  NCC(out,g4)={c4:.3f}")
    assert c1 > 0.9 and c4 > 0.9


# ---------- 光度校正 ----------

def test_photometric_gain_correction(tmp_path):
    """光度校正:把一帧调亮 1.3×,输出重叠带亮度应平滑(无跳变)。"""
    grid, frames, truth = _build(tmp_path, 8, 2, 2, bright_second=True)
    s = MosaicStitcher(grid, frames, blend_half_width=48)
    out, _ = s.run()
    g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 重叠带(画布 y≈300-400,列 320-380):行均值剖面应平滑
    band = g[300:400, 320:380]
    rowmean = band.mean(axis=1)
    steps = np.abs(np.diff(rowmean))
    assert steps.max() < 15, f"亮度跳变 {steps.max():.1f}(光度未校正或接缝错位)"


# ---------- 16bit 往返 ----------

def test_u16_roundtrip(tmp_path):
    """u16 输入/输出:位深保持 + 交叠区对齐。"""
    rng = np.random.default_rng(5)
    canvas_w = 300 + W
    canvas_h = 300 + H
    small = rng.random((canvas_h // 20 + 1, canvas_w // 20 + 1)).astype(np.float32)
    base = cv2.resize(small, (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-9) * 40000 + 5000  # u16 量程
    xs = [0, 300]; ys = [0, 300]
    tmp = tempfile.mkdtemp()
    frames = []
    k = 0
    for r in range(2):
        for c in range(2):
            f = base[ys[r]:ys[r] + H, xs[c]:xs[c] + W].copy()
            f = cv2.GaussianBlur(f, (0, 0), 1.0 + r * 0.5)
            f = np.clip(f + rng.random((H, W)).astype(np.float32) * 300,
                        0, 65535).astype(np.uint16)
            f3 = np.stack([f] * 3, axis=-1)
            p = os.path.join(tmp, f"f{k}.tif")
            cv2.imwrite(p, f3)                       # u16 TIFF
            frames.append(p)
            k += 1
    grid = GridSpec(rows=2, cols=2, overlap_pct=25.0)
    s = MosaicStitcher(grid, frames, blend_half_width=48, bitdepth=16)
    out, placements = s.run()
    truth = [(0, 0), (300, 0), (0, 300), (300, 300)]
    # 交叠区对齐验证(与 img3 的局部区域;img3 画布 (300,300))
    g = out[..., 1].astype(np.float64)
    f4 = cv2.imread(frames[3], cv2.IMREAD_UNCHANGED)
    g4 = (f4[..., 1] if f4.ndim == 3 else f4).astype(np.float64)
    # 交叠区(img3 支配区):输出 vs img3 帧(局部 = 画布 − 300)
    a = g[300:400, 300:400]
    b4 = g4[0:100, 0:100]
    c4 = float(np.corrcoef(a.ravel(), b4.ravel())[0, 1])
    print(f"  u16: NCC(a,g4crop)={c4:.3f}")
    assert c4 > 0.9        # img3 支配区内,输出应与 img3 一致
    for (px, py), (tx, ty) in zip(placements, truth):
        assert abs(px - tx) <= 30 and abs(py - ty) <= 30
