#!/usr/bin/env python3
"""Generate MoonBit fixture data for the millow alignment test suite.

This script builds small test images, runs reference implementations that
match millow's exact algorithms (using numpy / skimage / Pillow where they
agree), and writes the input + expected-output byte arrays to
``fixtures.mbt`` as ``Array[Int]`` literals.

Usage:
    source /home/shun/venv310/bin/activate
    python generate_fixtures.py

The generated file is checked into version control; re-run only when millow's
algorithms change.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Rounding helpers — must match MoonBit's @math.round (half away from zero)
# ---------------------------------------------------------------------------

def moon_round(x: float | np.ndarray) -> np.ndarray:
    """Round half away from zero, matching MoonBit @math.round."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def round_byte(x):
    """MoonBit round_byte: round half away from zero, then clamp to [0, 255]."""
    r = moon_round(x)
    if isinstance(r, np.ndarray):
        return np.clip(r.astype(int), 0, 255)
    return int(np.clip(int(r), 0, 255))


def clampi(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def clampd(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Image representation: H x W x 4 uint8 RGBA, row-major (matches millow)
# ---------------------------------------------------------------------------

def rgba(h: int, w: int, r: int, g: int, b: int, a: int = 255) -> np.ndarray:
    """Solid color image."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    img[:, :, 3] = a
    return img


def gradient_4x4() -> np.ndarray:
    """4x4 gradient: R = x*60, G = y*50, B = 100, A = 255."""
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    for y in range(4):
        for x in range(4):
            img[y, x] = (x * 60, y * 50, 100, 255)
    return img


def dot_3x3() -> np.ndarray:
    """3x3 black image with a single white center pixel (opaque)."""
    img = np.zeros((3, 3, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # all opaque
    img[1, 1] = (255, 255, 255, 255)
    return img


def bimodal_1x4() -> np.ndarray:
    """1x4 bimodal image [0, 0, 255, 255] for Otsu testing (opaque)."""
    img = np.zeros((1, 4, 4), dtype=np.uint8)
    img[:, :, 3] = 255  # all opaque
    img[0, 2] = (255, 255, 255, 255)
    img[0, 3] = (255, 255, 255, 255)
    return img


def alpha_2x2() -> np.ndarray:
    """2x2 image with varying alpha for compositing tests."""
    img = np.zeros((2, 2, 4), dtype=np.uint8)
    img[0, 0] = (200, 100, 50, 0)
    img[0, 1] = (200, 100, 50, 128)
    img[1, 0] = (200, 100, 50, 255)
    img[1, 1] = (200, 100, 50, 64)
    return img


# ---------------------------------------------------------------------------
# Reference implementations — match millow's algorithms exactly
# ---------------------------------------------------------------------------

def luma(r, g, b):
    """BT.601 luma as integer: (R*77 + G*150 + B*29) >> 8."""
    return (r.astype(int) * 77 + g.astype(int) * 150 + b.astype(int) * 29) >> 8


def to_grayscale(img):
    out = np.zeros_like(img)
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    out[..., 0] = y
    out[..., 1] = y
    out[..., 2] = y
    out[..., 3] = img[..., 3]
    return out


def to_bgr(img):
    out = img.copy()
    out[..., 0] = img[..., 2]
    out[..., 2] = img[..., 0]
    return out


def invert(img):
    out = img.copy()
    out[..., 0] = np.clip(255 - img[..., 0].astype(int), 0, 255)
    out[..., 1] = np.clip(255 - img[..., 1].astype(int), 0, 255)
    out[..., 2] = np.clip(255 - img[..., 2].astype(int), 0, 255)
    return out


def tint(img, r, g, b):
    out = img.copy()
    out[..., 0] = np.clip(img[..., 0].astype(int) * r // 255, 0, 255)
    out[..., 1] = np.clip(img[..., 1].astype(int) * g // 255, 0, 255)
    out[..., 2] = np.clip(img[..., 2].astype(int) * b // 255, 0, 255)
    return out


def flatten_alpha(img, r, g, b):
    out = img.copy()
    a = img[..., 3].astype(int)
    inv = 255 - a
    out[..., 0] = np.clip((img[..., 0].astype(int) * a + r * inv) // 255, 0, 255)
    out[..., 1] = np.clip((img[..., 1].astype(int) * a + g * inv) // 255, 0, 255)
    out[..., 2] = np.clip((img[..., 2].astype(int) * a + b * inv) // 255, 0, 255)
    out[..., 3] = 255
    return out


def crop(img, y, x, h, w):
    return img[y:y + h, x:x + w].copy()


def flip_horizontal(img):
    return img[:, ::-1].copy()


def flip_vertical(img):
    return img[::-1, :].copy()


def rotate_90(img):
    """Clockwise 90 = np.rot90(k=-1)."""
    return np.rot90(img, k=-1).copy()


def rotate_180(img):
    return np.rot90(img, k=2).copy()


def rotate_270(img):
    """Clockwise 270 = np.rot90(k=1) (CCW 90)."""
    return np.rot90(img, k=1).copy()


def _reflect_index(i, n):
    if n == 1:
        return 0
    period = 2 * n
    m = i % period
    if m < 0:
        m += period
    if m >= n:
        return period - 1 - m
    return m


def _wrap_index(i, n):
    m = i % n
    if m < 0:
        m += n
    return m


def pad(img, top, right, bottom, left, mode):
    h, w = img.shape[:2]
    nh = h + top + bottom
    nw = w + left + right
    out = np.zeros((nh, nw, 4), dtype=np.uint8)
    for oy in range(nh):
        for ox in range(nw):
            sy = oy - top
            sx = ox - left
            if 0 <= sy < h and 0 <= sx < w:
                out[oy, ox] = img[sy, sx]
            else:
                if isinstance(mode, tuple) and mode[0] == "constant":
                    out[oy, ox] = mode[1]
                elif mode == "replicate":
                    cy = clampi(sy, 0, h - 1)
                    cx = clampi(sx, 0, w - 1)
                    out[oy, ox] = img[cy, cx]
                elif mode == "reflect":
                    out[oy, ox] = img[_reflect_index(sy, h), _reflect_index(sx, w)]
                elif mode == "wrap":
                    out[oy, ox] = img[_wrap_index(sy, h), _wrap_index(sx, w)]
    return out


def resize_nearest(img, dst_h, dst_w):
    h, w = img.shape[:2]
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    for y in range(dst_h):
        sy = clampi(y * h // dst_h, 0, h - 1)
        for x in range(dst_w):
            sx = clampi(x * w // dst_w, 0, w - 1)
            out[y, x] = img[sy, sx]
    return out


def resize_bilinear(img, dst_h, dst_w):
    h, w = img.shape[:2]
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    sh, sw = float(h), float(w)
    for y in range(dst_h):
        fy = clampd((y + 0.5) * sh / dst_h - 0.5, 0.0, sh - 1.0)
        y0 = int(math.floor(fy))
        y1 = clampi(y0 + 1, 0, h - 1)
        wy = fy - y0
        for x in range(dst_w):
            fx = clampd((x + 0.5) * sw / dst_w - 0.5, 0.0, sw - 1.0)
            x0 = int(math.floor(fx))
            x1 = clampi(x0 + 1, 0, w - 1)
            wx = fx - x0
            v00 = img[y0, x0].astype(float)
            v01 = img[y0, x1].astype(float)
            v10 = img[y1, x0].astype(float)
            v11 = img[y1, x1].astype(float)
            top = v00 * (1 - wx) + v01 * wx
            bot = v10 * (1 - wx) + v11 * wx
            out[y, x] = round_byte(top * (1 - wy) + bot * wy)
    return out


def adjust_brightness(img, delta):
    out = img.copy()
    for c in range(3):
        out[..., c] = np.clip(img[..., c].astype(int) + delta, 0, 255)
    return out


def adjust_contrast(img, factor):
    out = img.copy()
    for c in range(3):
        out[..., c] = round_byte((img[..., c].astype(float) - 128.0) * factor + 128.0)
    return out


def adjust_gamma(img, gamma):
    out = img.copy()
    for c in range(3):
        out[..., c] = round_byte(
            np.power(img[..., c].astype(float) / 255.0, gamma) * 255.0
        )
    return out


def normalize(img, mn, mx):
    out = img.copy()
    lo = 255
    hi = 0
    for c in range(3):
        v = img[..., c].astype(int)
        lo = min(lo, v.min())
        hi = max(hi, v.max())
    if hi <= lo:
        return out
    span = (mx - mn) / (hi - lo)
    for c in range(3):
        out[..., c] = np.clip(
            mn + round_byte((img[..., c].astype(float) - lo) * span).astype(int),
            0, 255
        )
    return out


def threshold(img, thresh):
    out = img.copy()
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    v = np.where(y >= thresh, 255, 0).astype(np.uint8)
    out[..., 0] = v
    out[..., 1] = v
    out[..., 2] = v
    return out


def threshold_inv(img, thresh):
    out = img.copy()
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    v = np.where(y >= thresh, 0, 255).astype(np.uint8)
    out[..., 0] = v
    out[..., 1] = v
    out[..., 2] = v
    return out


def histogram_luma(img):
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    h = [0] * 256
    for v in y.ravel():
        h[v] += 1
    return h


def histogram_color(img):
    hs = [[0] * 256 for _ in range(3)]
    for c in range(3):
        for v in img[..., c].ravel():
            hs[c][v] += 1
    return hs


def otsu_threshold(hist, total):
    sum_total = sum(i * hist[i] for i in range(256))
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    thr = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        diff = m_b - m_f
        v = w_b * w_f * diff * diff
        if v > max_var:
            max_var = v
            thr = t
    return thr


def threshold_otsu(img):
    total = img.shape[0] * img.shape[1]
    hist = histogram_luma(img)
    applied = clampi(otsu_threshold(hist, total) + 1, 0, 255)
    return applied, threshold(img, applied)


def equalize_histogram(img):
    hist = histogram_luma(img)
    total = img.shape[0] * img.shape[1]
    cdf_min = 0
    found = False
    cdf = [0] * 256
    acc = 0
    for i in range(256):
        acc += hist[i]
        cdf[i] = acc
        if not found and hist[i] != 0:
            cdf_min = acc
            found = True
    denom = total - cdf_min
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if denom <= 0:
            lut[i] = i
        else:
            lut[i] = round_byte((cdf[i] - cdf_min) * 255.0 / denom)
    out = img.copy()
    for c in range(3):
        out[..., c] = lut[img[..., c]]
    return out


def _at_clamped(img, y, x, c):
    h, w = img.shape[:2]
    cy = clampi(y, 0, h - 1)
    cx = clampi(x, 0, w - 1)
    return float(img[cy, cx, c])


def convolve(img, kernel, normalize=False):
    kh = len(kernel)
    kw = len(kernel[0])
    ay = kh // 2
    ax = kw // 2
    ksum = sum(kernel[ky][kx] for ky in range(kh) for kx in range(kw))
    norm = ksum if normalize and ksum != 0.0 else 1.0
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                acc = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        acc += kernel[ky][kx] * _at_clamped(img, y + ky - ay, x + kx - ax, c)
                out[y, x, c] = round_byte(acc / norm)
            out[y, x, 3] = img[y, x, 3]
    return out


def box_blur(img, radius):
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    w1 = w + 1
    for c in range(3):
        integ = np.zeros((h + 1) * w1, dtype=int)
        for y in range(h):
            rowsum = 0
            for x in range(w):
                rowsum += int(img[y, x, c])
                integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + rowsum
        for y in range(h):
            y0 = clampi(y - radius, 0, h - 1)
            y1 = clampi(y + radius, 0, h - 1)
            for x in range(w):
                x0 = clampi(x - radius, 0, w - 1)
                x1 = clampi(x + radius, 0, w - 1)
                area = (y1 - y0 + 1) * (x1 - x0 + 1)
                s = (integ[(y1 + 1) * w1 + (x1 + 1)]
                     - integ[y0 * w1 + (x1 + 1)]
                     - integ[(y1 + 1) * w1 + x0]
                     + integ[y0 * w1 + x0])
                out[y, x, c] = np.clip(s // area, 0, 255)
    for i in range(h * w):
        out[i // w, i % w, 3] = img[i // w, i % w, 3]
    return out


def _gaussian_kernel(ksize, sigma):
    c = ksize // 2
    s2 = 2.0 * sigma * sigma
    k = [math.exp(-((i - c) ** 2) / s2) for i in range(ksize)]
    s = sum(k)
    return [v / s for v in k]


def gaussian_blur(img, sigma):
    radius = int(math.ceil(sigma * 3.0))
    ks = clampi(radius * 2 + 1, 3, 99)
    return gaussian_blur_kernel(img, ks, sigma)


def gaussian_blur_kernel(img, ksize, sigma):
    k = _gaussian_kernel(ksize, sigma)
    c = ksize // 2
    h, w = img.shape[:2]
    # Horizontal pass
    tmp = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                acc = 0.0
                for i in range(ksize):
                    d = i - c
                    acc += k[i] * _at_clamped(img, y, x + d, ch)
                tmp[y, x, ch] = int(round_byte(acc))
            tmp[y, x, 3] = img[y, x, 3]
    # Vertical pass
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                acc = 0.0
                for i in range(ksize):
                    d = i - c
                    acc += k[i] * _at_clamped(tmp, y + d, x, ch)
                out[y, x, ch] = int(round_byte(acc))
            out[y, x, 3] = tmp[y, x, 3]
    return out


def _window_vals(img, y, x, r, c):
    h, w = img.shape[:2]
    vals = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cy = clampi(y + dy, 0, h - 1)
            cx = clampi(x + dx, 0, w - 1)
            vals.append(int(img[cy, cx, c]))
    return vals


def _window_map(img, size, reduce_fn):
    r = size // 2
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                out[y, x, c] = reduce_fn(_window_vals(img, y, x, r, c))
            out[y, x, 3] = img[y, x, 3]
    return out


def median_filter(img, size):
    return _window_map(img, size, lambda vals: sorted(vals)[len(vals) // 2])


def min_filter(img, size):
    return _window_map(img, size, min)


def max_filter(img, size):
    return _window_map(img, size, max)


def _luma_at(img, y, x):
    h, w = img.shape[:2]
    cy = clampi(y, 0, h - 1)
    cx = clampi(x, 0, w - 1)
    return float(luma(img[cy, cx, 0], img[cy, cx, 1], img[cy, cx, 2]))


def _grad(img, kernel):
    h, w = img.shape[:2]
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(3):
                for kx in range(3):
                    acc += kernel[ky][kx] * _luma_at(img, y + ky - 1, x + kx - 1)
            out[y][x] = acc
    return out


def _mag_image(gx, gy):
    h = len(gx)
    w = len(gx[0]) if h > 0 else 0
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            dx = gx[y][x]
            dy = gy[y][x]
            vb = int(round_byte(math.sqrt(dx * dx + dy * dy)))
            out[y, x] = (vb, vb, vb, 255)
    return out


def sobel_x(img):
    return _grad(img, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def sobel_y(img):
    return _grad(img, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def sobel(img):
    return _mag_image(sobel_x(img), sobel_y(img))


def prewitt(img):
    gx = _grad(img, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    gy = _grad(img, [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    return _mag_image(gx, gy)


def scharr(img):
    gx = _grad(img, [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
    gy = _grad(img, [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
    return _mag_image(gx, gy)


def laplacian(img):
    g = _grad(img, [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = g[y][x]
            vb = int(round_byte(abs(v)))
            out[y, x] = (vb, vb, vb, 255)
    return out


def _kernel_offsets(kernel):
    if kernel[0] == "cross":
        size = kernel[1]
        r = size // 2
        offs = []
        for d in range(-r, r + 1):
            offs.append((d, 0))
            if d != 0:
                offs.append((0, d))
        return offs
    elif kernel[0] == "square":
        size = kernel[1]
        r = size // 2
        offs = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                offs.append((dy, dx))
        return offs
    return []


def _morph(img, kernel, is_dilate):
    offs = _kernel_offsets(kernel)
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                m = 0 if is_dilate else 255
                for dy, dx in offs:
                    cy = clampi(y + dy, 0, h - 1)
                    cx = clampi(x + dx, 0, w - 1)
                    v = int(img[cy, cx, c])
                    if is_dilate:
                        if v > m:
                            m = v
                    else:
                        if v < m:
                            m = v
                out[y, x, c] = m
            out[y, x, 3] = img[y, x, 3]
    return out


def dilate(img, kernel):
    return _morph(img, kernel, True)


def erode(img, kernel):
    return _morph(img, kernel, False)


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def arr_to_moonbit(name: str, data) -> str:
    """Convert a flat array/list of ints to a MoonBit Array[Int] literal."""
    vals = list(data)
    lines = [f"///|", f"let {name} : Array[Int] = ["]
    for i in range(0, len(vals), 8):
        chunk = vals[i:i + 8]
        lines.append("  " + ", ".join(str(v) for v in chunk) + ",")
    lines.append("]")
    return "\n".join(lines)


def img_to_moonbit(name: str, img: np.ndarray) -> str:
    return arr_to_moonbit(name, img.ravel().tolist())


def int_arr_to_moonbit(name: str, data) -> str:
    return arr_to_moonbit(name, data)


def arr2_to_moonbit(name: str, data) -> str:
    """Convert a 2D list of floats to MoonBit Array[Array[Double]]."""
    lines = [f"///|", f"let {name} : Array[Array[Double]] = ["]
    for row in data:
        lines.append("  [" + ", ".join(repr(float(v)) for v in row) + "],")
    lines.append("]")
    return "\n".join(lines)


def generate():
    grad = gradient_4x4()
    dot = dot_3x3()
    bimodal = bimodal_1x4()
    alpha = alpha_2x2()

    sections = [
        "// Generated by generate_fixtures.py. DO NOT EDIT.",
        "// Reference inputs and expected outputs for the millow alignment",
        "// test suite. Each Array[Int] is a flat RGBA8 byte buffer (H*W*4).",
        "",
        "// ===== Test inputs =====",
        "",
        img_to_moonbit("in_grad_4x4", grad),
        img_to_moonbit("in_dot_3x3", dot),
        img_to_moonbit("in_bimodal_1x4", bimodal),
        img_to_moonbit("in_alpha_2x2", alpha),
        "",
        "// ===== Color =====",
        "",
        img_to_moonbit("exp_invert", invert(grad)),
        img_to_moonbit("exp_to_bgr", to_bgr(grad)),
        img_to_moonbit("exp_to_grayscale", to_grayscale(grad)),
        img_to_moonbit("exp_tint", tint(grad, 200, 100, 50)),
        img_to_moonbit("exp_flatten_alpha", flatten_alpha(alpha, 10, 20, 30)),
        "",
        "// ===== Geometry =====",
        "",
        img_to_moonbit("exp_crop", crop(grad, 0, 1, 2, 2)),
        img_to_moonbit("exp_flip_h", flip_horizontal(grad)),
        img_to_moonbit("exp_flip_v", flip_vertical(grad)),
        img_to_moonbit("exp_rotate_90", rotate_90(grad)),
        img_to_moonbit("exp_rotate_180", rotate_180(grad)),
        img_to_moonbit("exp_rotate_270", rotate_270(grad)),
        img_to_moonbit("exp_pad_const", pad(grad, 1, 1, 1, 1, ("constant", (7, 8, 9, 10)))),
        img_to_moonbit("exp_pad_replicate", pad(grad, 1, 1, 1, 1, "replicate")),
        img_to_moonbit("exp_pad_reflect", pad(grad, 1, 1, 1, 1, "reflect")),
        img_to_moonbit("exp_pad_wrap", pad(grad, 1, 1, 1, 1, "wrap")),
        "",
        "// ===== Resize =====",
        "",
        img_to_moonbit("exp_resize_nearest_8x8", resize_nearest(grad, 8, 8)),
        img_to_moonbit("exp_resize_bilinear_2x2", resize_bilinear(grad, 2, 2)),
        "",
        "// ===== Enhancement =====",
        "",
        img_to_moonbit("exp_brightness_50", adjust_brightness(grad, 50)),
        img_to_moonbit("exp_contrast_1_5", adjust_contrast(grad, 1.5)),
        img_to_moonbit("exp_gamma_2_0", adjust_gamma(grad, 2.0)),
        img_to_moonbit("exp_normalize", normalize(grad, 0, 255)),
        "",
        "// ===== Threshold =====",
        "",
        img_to_moonbit("exp_threshold_128", threshold(grad, 128)),
        img_to_moonbit("exp_threshold_inv_128", threshold_inv(grad, 128)),
    ]

    # Otsu returns (threshold, image)
    otsu_thr, otsu_img = threshold_otsu(bimodal)
    sections.append("")
    sections.append(f"///|")
    sections.append(f"let exp_otsu_threshold : Int = {otsu_thr}")
    sections.append("")
    sections.append(img_to_moonbit("exp_otsu_img", otsu_img))

    sections.extend([
        "",
        "// ===== Histogram =====",
        "",
        int_arr_to_moonbit("exp_histogram", histogram_luma(grad)),
        int_arr_to_moonbit("exp_hist_color_r", histogram_color(grad)[0]),
        int_arr_to_moonbit("exp_hist_color_g", histogram_color(grad)[1]),
        int_arr_to_moonbit("exp_hist_color_b", histogram_color(grad)[2]),
        img_to_moonbit("exp_equalize", equalize_histogram(grad)),
        "",
        "// ===== Filters =====",
        "",
        img_to_moonbit("exp_box_blur_1", box_blur(grad, 1)),
        img_to_moonbit("exp_gaussian_blur_1", gaussian_blur(grad, 1.0)),
        img_to_moonbit(
            "exp_convolve_identity",
            convolve(grad, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], False),
        ),
        img_to_moonbit("exp_median_3", median_filter(grad, 3)),
        img_to_moonbit("exp_min_filter_3", min_filter(grad, 3)),
        img_to_moonbit("exp_max_filter_3", max_filter(grad, 3)),
        "",
        "// ===== Edges (on dot_3x3) =====",
        "",
        img_to_moonbit("exp_sobel", sobel(dot)),
        img_to_moonbit("exp_prewitt", prewitt(dot)),
        img_to_moonbit("exp_scharr", scharr(dot)),
        img_to_moonbit("exp_laplacian", laplacian(dot)),
        arr2_to_moonbit("exp_sobel_x", sobel_x(dot)),
        arr2_to_moonbit("exp_sobel_y", sobel_y(dot)),
        "",
        "// ===== Morphology (on dot_3x3) =====",
        "",
        img_to_moonbit("exp_erode_cross", erode(dot, ("cross", 3))),
        img_to_moonbit("exp_dilate_cross", dilate(dot, ("cross", 3))),
        img_to_moonbit("exp_erode_square", erode(dot, ("square", 3))),
        img_to_moonbit("exp_dilate_square", dilate(dot, ("square", 3))),
        "",
    ])

    return "\n".join(sections)


if __name__ == "__main__":
    out = generate()
    out_path = Path(__file__).parent / "fixtures_test.mbt"
    out_path.write_text(out, encoding="utf-8")
    print(f"Generated {out_path} ({len(out)} bytes)")
