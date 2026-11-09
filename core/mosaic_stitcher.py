"""
网格拼接模块(Grid Stitcher)
将多张已融合的平面图像(显微多区域拍摄)按网格声明对齐、光度校正、
min-cut 接缝分割、窄带多频段融合为一张无缝大图。


支持两种邻接方向:
    H(右邻):垂直条带(前图右缘 vs 当前图左缘),接缝 = 逐行的列路径
    V(下邻):水平条带(前图底缘 vs 当前图顶缘),接缝 = 逐列的行路径

内存模型:画布 u8(≈W·H·3 字节)+ 单带浮点缓冲,与帧数无关。
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from utils.image_utils import imwrite_auto
    _IMWRITE_AUTO = True
except ImportError:  # 允许在仓库外独立使用
    _IMWRITE_AUTO = False


class ShootingOrder(Enum):
    BY_ROW = "byrow"
    BY_COLUMN = "bycolumn"


class GridDirection(Enum):
    UNIDIRECTIONAL = "unidirectional"
    ZIGZAG = "zigzag"


class StartCorner(Enum):
    TOP_LEFT = "topleft"
    TOP_RIGHT = "topright"
    BOTTOM_LEFT = "bottomleft"
    BOTTOM_RIGHT = "bottomright"


@dataclass
class GridSpec:
    """网格声明:行列数 + 拍摄顺序 + 重叠率(%)。"""
    rows: int
    cols: int
    order: ShootingOrder = ShootingOrder.BY_ROW
    direction: GridDirection = GridDirection.UNIDIRECTIONAL
    start_corner: StartCorner = StartCorner.TOP_LEFT
    overlap_pct: float = 20.0          # 相邻重叠占图像宽/高的百分比


@dataclass
class AlignResult:
    """一次邻接对齐的结果。"""
    index_pair: Tuple[int, int]        # (prev_k, cur_k) 帧索引
    offset: Tuple[int, int]            # cur 在画布中的 (x, y)
    band_px: int                       # 重叠带宽或高(px)
    ncc: float                         # 带内相关系数
    orientation: str                   # "H"(右邻)或 "V"(下邻)


class MosaicStitcher:
    """网格拼接器:输入图像路径列表(网格顺序),输出无缝大图。"""

    def __init__(self, grid: GridSpec, images: List[str],
                 blend_half_width: int = 64,
                 pyramid_levels: int = 5,
                 ncc_scale: int = 4,
                 bitdepth: int = 8):
        if len(images) != grid.rows * grid.cols:
            raise ValueError(
                f"images ({len(images)}) != grid.rows*grid.cols "
                f"({grid.rows * grid.cols})")
        self.grid = grid
        self.paths = list(images)
        self.blend_half_width = blend_half_width
        self.pyramid_levels = pyramid_levels
        self.ncc_scale = max(1, int(ncc_scale))
        self.bitdepth = int(bitdepth)
        self.progress_cb: Optional[Callable[[int, int, str], None]] = None
        self.orient: List[str] = []

    def _progress(self, stage: int, total: int, msg: str) -> None:
        if self.progress_cb is None:
            return
        frac = int((stage + 0.5) / total * 100)
        try:
            self.progress_cb(frac, 100, f"mosaic: {msg}")
        except TypeError:
            self.progress_cb(frac, 100)

    # ---------- 公共入口 ----------

    def run(self, output_path: Optional[str] = None,
            progress_cb: Optional[Callable[[int, int, str], None]] = None
            ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """执行完整拼接。返回 (画布 u8/u16 RGB, 各图画布落位列表)。"""
        self.progress_cb = progress_cb
        total_stages = 4
        imgs = self._load_stage(0, total_stages)
        placements, offsets, aligns = self._align_stage(imgs, 1, total_stages)
        imgs = self._photometric_stage(imgs, offsets, aligns, 2, total_stages)
        canvas = self._render_stage(imgs, offsets, aligns, 3, total_stages)
        maxv = 65535 if self.bitdepth == 16 else 255
        out = np.clip(canvas, 0, maxv)
        out_dtype = np.uint16 if self.bitdepth == 16 else np.uint8
        out = out.astype(out_dtype)
        if output_path:
            self._save(output_path, out)
        return out, [(int(p[0]), int(p[1])) for p in placements]

    # ---------- 网格顺序与邻接 ----------

    def shoot_order(self) -> List[int]:
        """按网格声明返回拍摄/融合访问顺序(帧索引序列)。"""
        g = self.grid
        flip_r = g.start_corner in (StartCorner.BOTTOM_LEFT,
                                    StartCorner.BOTTOM_RIGHT)
        flip_c = g.start_corner in (StartCorner.TOP_RIGHT,
                                    StartCorner.BOTTOM_RIGHT)
        row_range = list(range(g.rows))
        col_range = list(range(g.cols))
        if flip_r:
            row_range.reverse()
        if flip_c:
            col_range.reverse()
        seq = []
        if g.order == ShootingOrder.BY_ROW:
            for ri, r in enumerate(row_range):
                cols = (col_range if (g.direction == GridDirection.UNIDIRECTIONAL
                                      or ri % 2 == 0)
                        else list(reversed(col_range)))
                for c in cols:
                    seq.append(r * g.cols + c)
        else:  # BY_COLUMN
            for ci, c in enumerate(col_range):
                rows = (row_range if (g.direction == GridDirection.UNIDIRECTIONAL
                                      or ci % 2 == 0)
                        else list(reversed(row_range)))
                for r in rows:
                    seq.append(r * g.cols + c)
        return seq

    def _grid_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.grid.rows)
                for c in range(self.grid.cols)]

    def grid_edges(self) -> List[Tuple[int, int, str]]:
        """网格邻接边:(prev_k, cur_k, orient);H=右邻,V=下邻。"""
        edges = []
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                k = r * self.grid.cols + c
                if c + 1 < self.grid.cols:
                    edges.append((k, k + 1, "H"))
                if r + 1 < self.grid.rows:
                    edges.append((k, k + self.grid.cols, "V"))
        return edges

    # ---------- 阶段 1:解码 ----------

    def _load_stage(self, stage, total):
        self._progress(stage, total, "decoding frames")
        imgs = []
        for p in self.paths:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise ValueError(f"cannot decode: {p}")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im.ndim == 2:                       # 灰度输入提升 3 通道
                im = np.stack([im] * 3, axis=-1)
            imgs.append(im.astype(np.float32))     # 原生标度(u8/u16)
        self._progress(stage, total, f"decoded {len(imgs)} frames")
        return imgs

    # ---------- 阶段 2/3/4:对齐(H/V 双方向) ----------

    def _align_stage(self, imgs, stage, total):
        g = self.grid
        order = self.shoot_order()
        cell_of = {k: cell for k, cell in zip(order, self._grid_cells())}
        placements = {0: (0, 0)}
        offsets = [(0, 0)]
        aligns: List[AlignResult] = []
        self.orient = ["V"] * len(imgs)
        edges = self.grid_edges()
        for pi, (prev_k, cur_k, orient) in enumerate(edges):
            prev = imgs[prev_k]
            cur = imgs[cur_k]
            pr, pc = cell_of[prev_k]
            cr, cc = cell_of[cur_k]
            prev_h, prev_w = prev.shape[:2]
            cur_h, cur_w = cur.shape[:2]
            if cr == pr:                                   # 右邻:垂直条带
                bw = max(64, int(cur_w * g.overlap_pct / 100))
                ncc, dx_rel, dy_rel = self._align_H(prev, cur, bw)
                x_new = placements[prev_k][0] + dx_rel
                y_new = placements[prev_k][1] + dy_rel
                band_px = bw
                self.orient[cur_k] = "H"
            else:                                          # 下邻:水平条带
                bh = max(64, int(cur_h * g.overlap_pct / 100))
                ncc, dx_rel, dy_rel = self._align_V(prev, cur, bh)
                x_new = placements[prev_k][0] + dx_rel
                y_new = placements[prev_k][1] + dy_rel
                band_px = bh
                self.orient[cur_k] = "V"
            aligns.append(AlignResult((prev_k, cur_k), (x_new, y_new),
                                      band_px, ncc,
                                      "H" if cr == pr else "V"))
            placements[cur_k] = (x_new, y_new)
            offsets.append((x_new, y_new))
            self._progress(stage, total,
                           f"aligned edge {pi + 1}/{len(edges)} [{orient}]"
                           f" NCC={ncc:.3f}")
        ordered = [placements[k] for k in range(len(imgs))]
        return ordered, offsets, aligns

    def _align_H(self, prev, cur, bw):
        """右邻:前图右缘条带(bw 列,全高)在当前图左缘区域(2bw)内匹配。
        返回 (ncc, dx_rel, dy_rel):
            dx_rel = cur_x − prev_x;dy_rel = cur_y − prev_y。"""
        gp = cv2.cvtColor(prev[:, prev.shape[1] - bw:],
                          cv2.COLOR_RGB2GRAY).astype(np.float32)
        gc = cv2.cvtColor(cur[:, :2 * bw],
                          cv2.COLOR_RGB2GRAY).astype(np.float32)
        gp /= max(float(gp.max()), 1.0)
        gc /= max(float(gc.max()), 1.0)
        gp4 = cv2.resize(gp, (max(1, gp.shape[1] // 4),
                              max(1, gp.shape[0] // 4)))
        gc4 = cv2.resize(gc, (max(1, gc.shape[1] // 4),
                              max(1, gc.shape[0] // 4)))
        ncc, mxl = self._ncc_search(gp4, gc4)
        dx_rel = prev.shape[1] - bw - mxl[0] * 4       # cur_x − prev_x
        dy_rel = mxl[1] * 4                            # cur_y − prev_y
        return ncc, dx_rel, dy_rel

    def _align_V(self, prev, cur, bh):
        """下邻:前图底缘条带(bh 行,全宽)在当前图顶缘区域(2bh)内匹配。
        返回 (ncc, dx_rel, dy_rel):
            dx_rel = cur_x − prev_x;dy_rel = cur_y − prev_y。"""
        gp = cv2.cvtColor(prev[prev.shape[0] - bh:, :],
                          cv2.COLOR_RGB2GRAY).astype(np.float32)
        gc = cv2.cvtColor(cur[:2 * bh, :],
                          cv2.COLOR_RGB2GRAY).astype(np.float32)
        gp /= max(float(gp.max()), 1.0)
        gc /= max(float(gc.max()), 1.0)
        gp4 = cv2.resize(gp, (max(1, gp.shape[1] // 4),
                              max(1, gp.shape[0] // 4)))
        gc4 = cv2.resize(gc, (max(1, gc.shape[1] // 4),
                              max(1, gc.shape[0] // 4)))
        ncc, mxl = self._ncc_search(gp4, gc4)
        dx_rel = mxl[0] * 4                            # cur_x − prev_x
        dy_rel = prev.shape[0] - bh - mxl[1] * 4       # cur_y − prev_y
        return ncc, dx_rel, dy_rel

    @staticmethod
    def _ncc_search(template, region):
        r = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, ncc, _, mxl = cv2.minMaxLoc(r)
        return float(ncc), mxl

    # ---------- 阶段 5:光度/色彩对齐(画布矩形交集,天然容错) ----------

    def _photometric_stage(self, imgs, offsets, aligns, stage, total):
        """重叠带逐通道最小二乘增益(亮度锚定)。帧尺寸不同时按画布交集裁剪。"""
        gains = [(1.0, 1.0, 1.0)]
        for k in range(1, len(imgs)):
            px, py = offsets[k - 1]
            cx, cy = offsets[k]
            prev = imgs[k - 1]
            cur = imgs[k]
            ph, pw = prev.shape[:2]
            ch, cw = cur.shape[:2]
            x_lo = max(px, cx); y_lo = max(py, cy)
            x_hi = min(px + pw, cx + cw); y_hi = min(py + ph, cy + ch)
            if x_hi - x_lo < 200 or y_hi - y_lo < 200:
                gains.append(gains[-1])
                continue
            aa = prev[y_lo - py:y_hi - py, x_lo - px:x_hi - px]
            bb = cur[y_lo - cy:y_hi - cy, x_lo - cx:x_hi - cx]
            g = []
            for c in range(3):
                av = aa[..., c].ravel(); bv = bb[..., c].ravel()
                gain = float((av * bv).sum() / max((bv * bv).sum(), 1e-6))
                g.append(np.clip(gain, 0.5, 2.0))
            m = sum(g) / 3
            g = [x / m for x in g]                 # 亮度锚定(只校相对色差)
            gains.append(tuple(g))
            maxv = 65535.0 if self.bitdepth == 16 else 255.0
            imgs[k] = (imgs[k] * np.array(g)[None, None, :]).clip(0, maxv).astype(np.float32)
            self._progress(stage, total,
                           f"photometric img{k} gains "
                           f"{tuple(round(x, 4) for x in g)}")
        return imgs

    # ---------- 阶段 6/7:接缝 DP(H/V 双向) ----------

    def seam_dp_v(self, a, b):
        """下邻(V):逐列 DP;返回每列的接缝行号(长度=带宽)。"""
        ga = a.mean(axis=2); gb = b.mean(axis=2)
        D = np.abs(ga - gb).T                        # (cols, rows)
        C, R = D.shape
        INF = 1e18
        cost = D[0].astype(np.float64).copy()
        back = np.zeros((C, R), dtype=np.int32)
        for c in range(1, C):
            prev = cost
            best = prev.copy()
            back[c] = 0
            up = np.concatenate([[INF], prev[:-1]])
            m = up < best
            best[m] = up[m]; back[c, m] = -1
            down = np.concatenate([prev[1:], [INF]])
            m = down < best
            best[m] = down[m]; back[c, m] = 1
            cost = best + D[c]
        seam = [int(np.argmin(cost))]
        for c in range(C - 1, 0, -1):
            r = int(min(max(seam[-1] + back[c, seam[-1]], 0), R - 1))
            seam.append(r)
        seam.reverse()
        return np.array(seam)

    def seam_dp_h(self, a, b):
        """右邻(H):逐行 DP;返回每行的接缝列号(长度=带高)。"""
        ga = a.mean(axis=2); gb = b.mean(axis=2)
        D = np.abs(ga - gb)                          # (rows, band_cols)
        R, C = D.shape
        INF = 1e18
        cost = D[0].astype(np.float64).copy()
        back = np.zeros((R, C), dtype=np.int32)
        for r in range(1, R):
            prev = cost
            best = prev.copy()
            back[r] = 0
            left = np.concatenate([[INF], prev[:-1]])
            m = left < best
            best[m] = left[m]; back[r, m] = -1
            right = np.concatenate([prev[1:], [INF]])
            m = right < best
            best[m] = right[m]; back[r, m] = 1
            cost = best + D[r]
        seam = [int(np.argmin(cost))]
        for r in range(R - 1, 0, -1):
            c = int(min(max(seam[-1] + back[r, seam[-1]], 0), C - 1))
            seam.append(c)
        seam.reverse()
        return np.array(seam)

    # ---------- 阶段 8:窄带多频段融合(H/V 双向) ----------

    def blend_band(self, a, b, seam, orientation):
        rows, cols = a.shape[:2]
        half = self.blend_half_width
        mask = np.zeros((rows, cols), dtype=np.float32)
        if orientation == "V":                     # 逐列行路径
            seam = np.clip(seam, 0, rows - 1)
            for c in range(cols):
                s = int(seam[c])
                hi_lo = max(0, s - half); hi_hi = min(rows, s + half)
                mask[:hi_lo, c] = 1.0
                if hi_hi > hi_lo:
                    mask[hi_lo:hi_hi, c] = np.linspace(
                        1, 0, hi_hi - hi_lo, dtype=np.float32)
        else:                                      # 逐行列路径
            seam = np.clip(seam, 0, cols - 1)
            for r in range(rows):
                s = int(seam[r])
                hi_lo = max(0, s - half); hi_hi = min(cols, s + half)
                mask[r, :hi_lo] = 1.0
                if hi_hi > hi_lo:
                    mask[r, hi_lo:hi_hi] = np.linspace(
                        1, 0, hi_hi - hi_lo, dtype=np.float32)
        la = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2BGR)
        lb = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_RGB2BGR)
        ga, gb = la.copy(), lb.copy()
        pyr = []
        for _ in range(self.pyramid_levels):
            sa = cv2.pyrDown(ga); sb = cv2.pyrDown(gb)
            ua = cv2.pyrUp(sa, dstsize=(ga.shape[1], ga.shape[0]))
            ub = cv2.pyrUp(sb, dstsize=(gb.shape[1], gb.shape[0]))
            pyr.append((ga - ua, gb - ub))
            ga, gb = sa, sb
        ms = cv2.resize(mask.astype(np.float32),
                        (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
        out = ga * ms[..., None] + gb * (1 - ms[..., None])
        for pa, pb in reversed(pyr):
            out = cv2.pyrUp(out, dstsize=(pa.shape[1], pa.shape[0]))
            mh = cv2.resize(mask, (pa.shape[1], pa.shape[0]),
                            interpolation=cv2.INTER_AREA)
            out = out + (pa - pb) * mh[..., None] + pb
        return np.clip(out[:, :, ::-1], 0, 65535 if self.bitdepth == 16 else 255)

    # ---------- 阶段 9:装配 ----------

    def _render_stage(self, imgs, offsets, aligns, stage, total):
        maxv = 65535.0 if self.bitdepth == 16 else 255.0
        canvas_h = max(p[1] + im.shape[0] for p, im in zip(offsets, imgs))
        canvas_w = max(p[0] + im.shape[1] for p, im in zip(offsets, imgs))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        first = imgs[0]
        x0, y0 = offsets[0]
        canvas[y0:y0 + first.shape[0], x0:x0 + first.shape[1]] = first
        self._progress(stage, total, "base image placed")
        for k in range(1, len(imgs)):
            y_top = offsets[k][1]; x_off = offsets[k][0]
            prev_h = imgs[k - 1].shape[0]
            band_rows = offsets[k - 1][1] + prev_h - y_top
            orient = self.orient[k]
            if band_rows <= 0:
                continue
            if orient == "V":                      # 下邻:水平条带
                a = imgs[k - 1][prev_h - band_rows:, :]
                b = imgs[k][:band_rows, :]
                seam = self.seam_dp_v(a, b)
                blended = self.blend_band(a, b, seam, "V")
                x_lo = max(0, x_off)
                w_ins = min(blended.shape[1], canvas_w - x_lo)
                h_ins = min(blended.shape[0], canvas_h - y_top)
                canvas[y_top:y_top + h_ins, x_lo:x_lo + w_ins] = \
                    blended[:h_ins, :w_ins].astype(np.float32)
                y2 = y_top + band_rows
                rem = min(imgs[k].shape[0] - band_rows, canvas_h - y2)
                w2 = min(imgs[k].shape[1], canvas_w - x_off)
                if rem > 0 and w2 > 0:
                    canvas[y2:y2 + rem, x_off:x_off + w2] = \
                        imgs[k][band_rows:band_rows + rem, :w2].astype(np.float32)
            else:                                  # 右邻:垂直条带
                prev = imgs[k - 1]
                bw = aligns[k - 1].band_px
                a = prev[:, prev.shape[1] - bw:]
                b = imgs[k][:, :bw]
                rows_n = min(a.shape[0], b.shape[0])
                a = a[:rows_n]; b = b[:rows_n]
                seam = self.seam_dp_h(a, b)
                blended = self.blend_band(a, b, seam, "H")
                x_lo = max(0, x_off)
                w_ins = min(blended.shape[1], canvas_w - x_lo)
                h_ins = min(blended.shape[0], canvas_h - y_top)
                canvas[y_top:y_top + h_ins, x_lo:x_lo + w_ins] = \
                    blended[:h_ins, :w_ins].astype(np.float32)
                x2 = x_off + bw
                rem_w = min(imgs[k].shape[1] - bw, canvas_w - x2)
                if rem_w > 0:
                    hh = min(imgs[k].shape[0], canvas_h - y_top)
                    canvas[y_top:y_top + hh, x2:x2 + rem_w] = \
                        imgs[k][:hh, bw:bw + rem_w].astype(np.float32)
            self._progress(stage, total, f"blended pair {k}")
        return canvas

    def _save(self, path, out):
        rgb = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if _IMWRITE_AUTO:
            imwrite_auto(path, rgb)
        else:
            cv2.imwrite(path, rgb)


def stitch_folder(folder: str, rows: int, cols: int,
                  output_path: str, **grid_kwargs) -> Tuple[np.ndarray, list]:
    """便捷入口:目录内图像(自然排序)按网格拼接。返回 (输出, 落位)。"""
    from utils.sorting import natural_sort_key  # 复用现有自然排序
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(exts)]
    files = sorted(files, key=natural_sort_key)
    grid = GridSpec(rows=rows, cols=cols, **grid_kwargs)
    stitcher = MosaicStitcher(grid, files[:rows * cols])
    return stitcher.run(output_path=output_path)
