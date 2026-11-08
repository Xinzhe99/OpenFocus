"""Streaming tile Laplacian blending; inputs are normalized, unassociated BGR.

Callers supply global-phase-aligned halo tiles. Each iterator item is
(colors, opacity, seam_weight). Only one source pyramid is retained at a time.
"""
import itertools
import math

import cv2
import numpy as np


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError('Mosaic blending cancelled')


def expanded_tile(x, y, width, height, canvas_width, canvas_height, *,
                  levels=5, band=64, halo=None):
    """Return expanded local bounds and valid crop, aligned to canvas phase zero."""
    if not 1 <= levels <= 7:
        raise ValueError('Pyramid levels must be between one and seven')
    grid = 1 << (levels - 1)
    required = max(4 * grid, 2 * band + 16)
    if halo is not None and halo < required:
        raise ValueError('Tile halo is smaller than the filter and seam support')
    margin = required if halo is None else int(halo)
    x0 = max(0, ((x - margin) // grid) * grid)
    y0 = max(0, ((y - margin) // grid) * grid)
    x1 = min(canvas_width, math.ceil((x + width + margin) / grid) * grid)
    y1 = min(canvas_height, math.ceil((y + height + margin) / grid) * grid)
    return (x0, y0, x1, y1), np.s_[y - y0:y - y0 + height, x - x0:x - x0 + width]


def _pyramids(colors, alpha, weight, levels, cancel):
    colors = np.asarray(colors, np.float32)
    alpha = np.asarray(alpha, np.float32)
    gauss, weights = [colors], [np.asarray(weight, np.float32) * alpha]
    coverage = alpha
    for _ in range(1, levels):
        _cancel(cancel)
        next_coverage = cv2.pyrDown(coverage)
        premultiplied = cv2.pyrDown(gauss[-1] * coverage[:, :, None])
        next_colors = np.zeros_like(premultiplied)
        np.divide(premultiplied, next_coverage[:, :, None], out=next_colors,
                  where=next_coverage[:, :, None] > 1e-12)
        gauss.append(next_colors)
        weights.append(cv2.pyrDown(weights[-1]))
        coverage = next_coverage
    for level in range(levels - 1):
        _cancel(cancel)
        size = (gauss[level].shape[1], gauss[level].shape[0])
        gauss[level] = gauss[level] - cv2.pyrUp(gauss[level + 1], dstsize=size)
    return gauss, weights


def blend_tiles(samples, *, shape, mode='multiband', levels=5, should_cancel=None):
    """Return (float32 BGR, opacity), preserving single-source pixel values.

    No source-image count factor appears in the live pyramid buffers. The
    caller admits this workspace and supplies only sources intersecting a tile.
    A masked source contributes neither pixel color nor output opacity.
    """
    if mode not in ('multiband', 'feather', 'hard'):
        raise ValueError('Unknown mosaic blend mode')
    if not 1 <= levels <= 7:
        raise ValueError('Pyramid levels must be between one and seven')
    iterator = iter(samples)
    first = next(iterator, None)
    _cancel(should_cancel)
    if first is None:
        return np.zeros((*shape, 3), np.float32), np.zeros(shape, np.float32)
    second = next(iterator, None)
    if second is None:
        colors, alpha, _weight = first
        result = np.array(colors, np.float32, copy=True)
        result[np.asarray(alpha) <= 0] = 0
        return result, np.array(alpha, np.float32, copy=True)
    coverage = np.zeros(shape, np.float32)
    fallback = np.zeros((*shape, 3), np.float32)
    fallback_weight = np.zeros(shape, np.float32)
    contributors = np.zeros(shape, np.uint16)
    reference = np.zeros((*shape, 3), np.float32)
    consistent = np.ones(shape, bool)
    numerator, denominator = [], []
    if mode in ('feather', 'hard'):
        result = np.zeros((*shape, 3), np.float32)
        total = np.zeros(shape, np.float32)
        best = np.full(shape, -1.0, np.float32) if mode == 'hard' else None
    for colors, alpha, weight in itertools.chain((first, second), iterator):
        _cancel(should_cancel)
        if colors.shape != (*shape, 3) or alpha.shape != shape or weight.shape != shape:
            raise ValueError('Mosaic tile colors, opacity and weights do not match')
        coverage += (1.0 - coverage) * alpha
        visible = alpha > 0
        previous = visible & (contributors > 0)
        consistent[previous] &= np.all(colors[previous] == reference[previous], axis=1)
        first_visible = visible & (contributors == 0)
        reference[first_visible] = colors[first_visible]
        fallback += colors * alpha[:, :, None]
        fallback_weight += alpha
        contributors += alpha > 0
        if mode == 'hard':
            score = weight * alpha
            selected = (alpha > 0) & (score > best)
            result[selected] = colors[selected]
            best[selected] = score[selected]
            continue
        if mode == 'feather':
            effective = weight * alpha
            result += colors * effective[:, :, None]
            total += effective
            continue
        laplacian, weights = _pyramids(colors, alpha, weight, levels, should_cancel)
        if not numerator:
            numerator = [np.zeros_like(layer) for layer in laplacian]
            denominator = [np.zeros_like(layer) for layer in weights]
        for level, (layer, wgt) in enumerate(zip(laplacian, weights)):
            numerator[level] += layer * wgt[:, :, None]
            denominator[level] += wgt
        del laplacian, weights
    missing = (total <= 1e-12 if mode == 'feather' else
               best <= 0 if mode == 'hard' else denominator[0] <= 1e-12)
    if mode == 'feather':
        np.divide(result, total[:, :, None], out=result, where=total[:, :, None] > 1e-12)
    elif mode == 'multiband':
        result = None
        for level in range(levels - 1, -1, -1):
            _cancel(should_cancel)
            layer = numerator[level]
            den = denominator[level][:, :, None]
            np.divide(layer, den, out=layer, where=den > 1e-12)
            layer[den[:, :, 0] <= 1e-12] = 0
            if result is None:
                result = layer
            else:
                result = cv2.pyrUp(result, dstsize=(layer.shape[1], layer.shape[0])) + layer
    repair = (fallback_weight > 0) & missing
    np.divide(fallback, fallback_weight[:, :, None], out=fallback,
              where=fallback_weight[:, :, None] > 0)
    result[repair] = fallback[repair]
    unchanged = (contributors > 0) & consistent
    result[unchanged] = reference[unchanged]
    result[coverage <= 0] = 0
    _cancel(should_cancel)
    return result, coverage
