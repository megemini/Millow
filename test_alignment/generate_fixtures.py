#!/usr/bin/env python3
"""Generate MoonBit fixture data for the millow alignment test suite.

This script builds small test images, runs reference implementations using
numpy to match millow's exact implementation, and writes the input + expected-output
byte arrays to ``fixtures_test.mbt`` as ``Array[Int]`` literals.

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
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import skimage
from skimage import filters, exposure, morphology, feature, measure
from skimage.util import img_as_ubyte, img_as_float


def moon_round(x):
    """Round half away from zero, matching MoonBit @math.round."""
    if isinstance(x, np.ndarray):
        return np.sign(x) * np.floor(np.abs(x) + 0.5)
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def round_byte(x):
    """MoonBit round_byte: round half away from zero, then clamp to [0, 255]."""
    r = moon_round(x)
    if isinstance(r, np.ndarray):
        return np.clip(r.astype(int), 0, 255)
    return int(max(0, min(255, int(r))))


def clampi(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


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
    img[:, :, 3] = 255
    img[1, 1] = (255, 255, 255, 255)
    return img


def bimodal_1x4() -> np.ndarray:
    """1x4 bimodal image [0, 0, 255, 255] for Otsu testing (opaque)."""
    img = np.zeros((1, 4, 4), dtype=np.uint8)
    img[:, :, 3] = 255
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


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy RGBA array to PIL Image."""
    return Image.fromarray(img, mode='RGBA')


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy RGBA array."""
    return np.array(pil_img.convert('RGBA'))


# ---------------------------------------------------------------------------
# Color operations
# ---------------------------------------------------------------------------

def luma(r, g, b):
    """BT.601 luma as integer: (R*77 + G*150 + B*29) >> 8."""
    return (r.astype(int) * 77 + g.astype(int) * 150 + b.astype(int) * 29) >> 8


def to_grayscale(img):
    """Custom BT.601 luma."""
    out = np.zeros_like(img)
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    out[..., 0] = y
    out[..., 1] = y
    out[..., 2] = y
    out[..., 3] = img[..., 3]
    return out


def to_bgr(img):
    """Swap R and B channels."""
    out = img.copy()
    out[..., 0] = img[..., 2]
    out[..., 2] = img[..., 0]
    return out


def invert(img):
    """Use Pillow ImageOps.invert."""
    pil = numpy_to_pil(img)
    inverted = ImageOps.invert(pil.convert('RGB'))
    result = pil_to_numpy(inverted)
    result[..., 3] = img[..., 3]
    return result


def tint(img, r, g, b):
    """Custom: each channel scaled by color / 255."""
    out = img.copy()
    out[..., 0] = np.clip(img[..., 0].astype(int) * r // 255, 0, 255)
    out[..., 1] = np.clip(img[..., 1].astype(int) * g // 255, 0, 255)
    out[..., 2] = np.clip(img[..., 2].astype(int) * b // 255, 0, 255)
    return out


def flatten_alpha(img, r, g, b):
    """Use Pillow's alpha composite with solid background."""
    pil = numpy_to_pil(img)
    bg = Image.new('RGBA', pil.size, (r, g, b, 255))
    result = Image.alpha_composite(bg, pil).convert('RGB')
    out = pil_to_numpy(result)
    out[..., 3] = 255
    return out


# ---------------------------------------------------------------------------
# Geometry operations - Pillow
# ---------------------------------------------------------------------------

def crop(img, y, x, h, w):
    """Use Pillow Image.crop."""
    pil = numpy_to_pil(img)
    cropped = pil.crop((x, y, x + w, y + h))
    return pil_to_numpy(cropped)


def flip_horizontal(img):
    """Use Pillow Image.transpose(FLIP_LEFT_RIGHT)."""
    pil = numpy_to_pil(img)
    flipped = pil.transpose(Image.FLIP_LEFT_RIGHT)
    return pil_to_numpy(flipped)


def flip_vertical(img):
    """Use Pillow Image.transpose(FLIP_TOP_BOTTOM)."""
    pil = numpy_to_pil(img)
    flipped = pil.transpose(Image.FLIP_TOP_BOTTOM)
    return pil_to_numpy(flipped)


def rotate_90(img):
    """Use Pillow Image.rotate(-90, expand=True)."""
    pil = numpy_to_pil(img)
    rotated = pil.rotate(-90, expand=True)
    return pil_to_numpy(rotated)


def rotate_180(img):
    """Use Pillow Image.rotate(180)."""
    pil = numpy_to_pil(img)
    rotated = pil.rotate(180)
    return pil_to_numpy(rotated)


def rotate_270(img):
    """Use Pillow Image.rotate(90, expand=True)."""
    pil = numpy_to_pil(img)
    rotated = pil.rotate(90, expand=True)
    return pil_to_numpy(rotated)


def pad(img, top, right, bottom, left, mode):
    """Custom pad implementation."""
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
                    period = 2 * h
                    ry = sy % period
                    if ry < 0:
                        ry += period
                    if ry >= h:
                        ry = period - 1 - ry
                    period = 2 * w
                    rx = sx % period
                    if rx < 0:
                        rx += period
                    if rx >= w:
                        rx = period - 1 - rx
                    out[oy, ox] = img[ry, rx]
                elif mode == "wrap":
                    ry = sy % h
                    if ry < 0:
                        ry += h
                    rx = sx % w
                    if rx < 0:
                        rx += w
                    out[oy, ox] = img[ry, rx]
    return out


# ---------------------------------------------------------------------------
# Resize - numpy (matching millow)
# ---------------------------------------------------------------------------

def resize_nearest(img, dst_h, dst_w):
    """Use Pillow Image.resize(resample=Image.NEAREST)."""
    pil = numpy_to_pil(img)
    resized = pil.resize((dst_w, dst_h), resample=Image.NEAREST)
    return pil_to_numpy(resized)


def resize_bilinear(img, dst_h, dst_w):
    """Custom bilinear resize (matching millow)."""
    h, w = img.shape[:2]
    if h == 0 or w == 0 or dst_h == 0 or dst_w == 0:
        return np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    sh = float(h)
    sw = float(w)
    for y in range(dst_h):
        fy = max(0.0, min(sh - 1.0, (float(y) + 0.5) * sh / float(dst_h) - 0.5))
        y0 = int(math.floor(fy))
        y1 = clampi(y0 + 1, 0, h - 1)
        wy = fy - float(y0)
        for x in range(dst_w):
            fx = max(0.0, min(sw - 1.0, (float(x) + 0.5) * sw / float(dst_w) - 0.5))
            x0 = int(math.floor(fx))
            x1 = clampi(x0 + 1, 0, w - 1)
            wx = fx - float(x0)
            for c in range(4):
                v00 = float(img[y0, x0, c])
                v01 = float(img[y0, x1, c])
                v10 = float(img[y1, x0, c])
                v11 = float(img[y1, x1, c])
                top = v00 * (1.0 - wx) + v01 * wx
                bot = v10 * (1.0 - wx) + v11 * wx
                out[y, x, c] = round_byte(top * (1.0 - wy) + bot * wy)
    return out


# ---------------------------------------------------------------------------
# Enhancement - Pillow
# ---------------------------------------------------------------------------

def adjust_brightness(img, factor):
    """Use Pillow ImageEnhance.Brightness."""
    pil = numpy_to_pil(img)
    enhancer = ImageEnhance.Brightness(pil)
    result = enhancer.enhance(factor)
    return pil_to_numpy(result)


def adjust_contrast(img, factor):
    """Use Pillow ImageEnhance.Contrast."""
    pil = numpy_to_pil(img)
    enhancer = ImageEnhance.Contrast(pil)
    result = enhancer.enhance(factor)
    return pil_to_numpy(result)


def adjust_gamma(img, gamma):
    """Use Pillow Image.point for gamma correction."""
    pil = numpy_to_pil(img)
    gamma_img = pil.point(lambda x: round_byte(((x / 255.0) ** gamma) * 255.0))
    return pil_to_numpy(gamma_img)


def normalize(img, mn, mx):
    """Use skimage.exposure.rescale_intensity."""
    rgb = img[..., :3].astype(float)
    out = exposure.rescale_intensity(rgb, in_range='image', out_range=(mn, mx))
    result = np.zeros_like(img)
    result[..., :3] = out.astype(np.uint8)
    result[..., 3] = img[..., 3]
    return result


# ---------------------------------------------------------------------------
# Threshold - custom
# ---------------------------------------------------------------------------

def threshold(img, thresh):
    """Custom threshold using BT.601 luma."""
    out = img.copy()
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    v = np.where(y >= thresh, 255, 0).astype(np.uint8)
    out[..., 0] = v
    out[..., 1] = v
    out[..., 2] = v
    return out


def threshold_inv(img, thresh):
    """Custom inverse threshold using BT.601 luma."""
    out = img.copy()
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    v = np.where(y >= thresh, 0, 255).astype(np.uint8)
    out[..., 0] = v
    out[..., 1] = v
    out[..., 2] = v
    return out


def histogram_luma(img):
    """Custom luma histogram."""
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    h = [0] * 256
    for v in y.ravel():
        h[v] += 1
    return h


def histogram_color(img):
    """Per-channel histograms."""
    hs = [[0] * 256 for _ in range(3)]
    for c in range(3):
        for v in img[..., c].ravel():
            hs[c][v] += 1
    return hs


def otsu_threshold(hist, total):
    """Otsu's thresholding."""
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
    """Otsu threshold using skimage.filters.threshold_otsu."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(float) / 255.0
    thr = int(round_byte(filters.threshold_otsu(gray) * 255))
    applied = clampi(thr + 1, 0, 255)
    return applied, threshold(img, applied)


def equalize_histogram(img):
    """Histogram equalization (matching millow)."""
    h, w = img.shape[:2]
    total = h * w
    out = img.copy()
    for c in range(3):
        hist = [0] * 256
        for i in range(total):
            y = i // w
            x = i % w
            v = int(img[y, x, c])
            hist[v] += 1
        cdf = [0] * 256
        acc = 0
        for i in range(256):
            acc += hist[i]
            cdf[i] = acc
        cdf_min_idx = 0
        while cdf_min_idx < 256 and hist[cdf_min_idx] == 0:
            cdf_min_idx += 1
        cdf_min = cdf[cdf_min_idx] if cdf_min_idx < 256 else 0
        denom = acc - cdf_min
        lut = [0] * 256
        for i in range(256):
            if denom <= 0:
                lut[i] = i
            else:
                lut[i] = round_byte((cdf[i] - cdf_min) * 255.0 / float(denom))
        for i in range(total):
            y = i // w
            x = i % w
            out[y, x, c] = lut[int(img[y, x, c])]
    return out


# ---------------------------------------------------------------------------
# Filters - custom (matching millow)
# ---------------------------------------------------------------------------

def _at_clamped(img, y, x, c):
    h, w = img.shape[:2]
    cy = clampi(y, 0, h - 1)
    cx = clampi(x, 0, w - 1)
    return float(img[cy, cx, c])


def convolve(img, kernel, normalize=False):
    """Custom convolution."""
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
    """Custom box blur (matching millow)."""
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
    """Custom gaussian blur (3σ truncation)."""
    radius = int(math.ceil(sigma * 3.0))
    ks = clampi(radius * 2 + 1, 3, 99)
    k = _gaussian_kernel(ks, sigma)
    c = ks // 2
    h, w = img.shape[:2]
    tmp = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                acc = 0.0
                for i in range(ks):
                    d = i - c
                    acc += k[i] * _at_clamped(img, y, x + d, ch)
                tmp[y, x, ch] = int(round_byte(acc))
            tmp[y, x, 3] = img[y, x, 3]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                acc = 0.0
                for i in range(ks):
                    d = i - c
                    acc += k[i] * _at_clamped(tmp, y + d, x, ch)
                out[y, x, ch] = int(round_byte(acc))
            out[y, x, 3] = tmp[y, x, 3]
    return out


def median_filter(img, size):
    """Custom median filter."""
    r = size // 2
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                vals = []
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        cy = clampi(y + dy, 0, h - 1)
                        cx = clampi(x + dx, 0, w - 1)
                        vals.append(int(img[cy, cx, c]))
                out[y, x, c] = sorted(vals)[len(vals) // 2]
            out[y, x, 3] = img[y, x, 3]
    return out


def min_filter(img, size):
    """Custom min filter."""
    r = size // 2
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                vals = []
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        cy = clampi(y + dy, 0, h - 1)
                        cx = clampi(x + dx, 0, w - 1)
                        vals.append(int(img[cy, cx, c]))
                out[y, x, c] = min(vals)
            out[y, x, 3] = img[y, x, 3]
    return out


def max_filter(img, size):
    """Custom max filter."""
    r = size // 2
    h, w = img.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                vals = []
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        cy = clampi(y + dy, 0, h - 1)
                        cx = clampi(x + dx, 0, w - 1)
                        vals.append(int(img[cy, cx, c]))
                out[y, x, c] = max(vals)
            out[y, x, 3] = img[y, x, 3]
    return out


# ---------------------------------------------------------------------------
# Edge detection - numpy (matching millow)
# ---------------------------------------------------------------------------

def _reflect_coord(v, max_v):
    """Reflect coordinates (mirror with edge duplication)."""
    if max_v == 0:
        return 0
    size = max_v + 1
    twice = size * 2
    r = v
    if r < 0:
        r = -r - 1
    r = r % twice
    if r > max_v:
        r = twice - 1 - r
    return r


def _luma_at_reflect(img, y, x):
    """Get luma at coordinates with reflect border mode."""
    h, w = img.shape[:2]
    ry = _reflect_coord(y, h - 1)
    rx = _reflect_coord(x, w - 1)
    return float(luma(img[ry, rx, 0], img[ry, rx, 1], img[ry, rx, 2]))


def _grad(img, k):
    """Apply a 3x3 kernel to the luminance field."""
    h, w = img.shape[:2]
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(3):
                for kx in range(3):
                    acc += k[ky][kx] * _luma_at_reflect(img, y + ky - 1, x + kx - 1)
            out[y][x] = acc
    return out


def _mag_image(gx, gy):
    """Combine two gradient maps into a magnitude image."""
    h, w = len(gx), len(gx[0])
    out = np.zeros((h, w, 4), dtype=np.uint8)
    max_val = 0.0
    for y in range(h):
        for x in range(w):
            dx = gx[y][x]
            dy = gy[y][x]
            mag = math.sqrt(dx * dx + dy * dy)
            if mag > max_val:
                max_val = mag
    scale = 255.0 / max_val if max_val > 0 else 1.0
    for y in range(h):
        for x in range(w):
            dx = gx[y][x]
            dy = gy[y][x]
            vb = round_byte(math.sqrt(dx * dx + dy * dy) * scale)
            out[y, x, :3] = vb
            out[y, x, 3] = 255
    return out


def sobel(img):
    """Sobel edge detection (matching millow)."""
    smoothed = gaussian_blur(img, 1.0)
    gx = _grad(smoothed, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    gy = _grad(smoothed, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return _mag_image(gx, gy)


def sobel_x(img):
    """Sobel horizontal gradient (matching millow)."""
    smoothed = gaussian_blur(img, 1.0)
    gx = _grad(smoothed, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    return gx


def sobel_y(img):
    """Sobel vertical gradient (matching millow)."""
    smoothed = gaussian_blur(img, 1.0)
    gy = _grad(smoothed, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return gy


def prewitt(img):
    """Prewitt edge detection (matching millow)."""
    smoothed = gaussian_blur(img, 1.0)
    gx = _grad(smoothed, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    gy = _grad(smoothed, [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    return _mag_image(gx, gy)


def scharr(img):
    """Scharr edge detection (matching millow)."""
    smoothed = gaussian_blur(img, 1.0)
    gx = _grad(smoothed, [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
    gy = _grad(smoothed, [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
    return _mag_image(gx, gy)


def laplacian(img):
    """Laplacian edge detection (matching millow)."""
    smoothed = gaussian_blur(img, 1.0)
    h, w = img.shape[:2]
    g = _grad(smoothed, [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    out = np.zeros((h, w, 4), dtype=np.uint8)
    max_val = 0.0
    for y in range(h):
        for x in range(w):
            v = abs(g[y][x])
            if v > max_val:
                max_val = v
    scale = 255.0 / max_val if max_val > 0 else 1.0
    for y in range(h):
        for x in range(w):
            v = abs(g[y][x]) * scale
            vb = round_byte(v)
            out[y, x, :3] = vb
            out[y, x, 3] = 255
    return out


# ---------------------------------------------------------------------------
# Morphology - skimage
# ---------------------------------------------------------------------------

def _make_structuring_element(kernel):
    """Create structuring element for skimage morphology."""
    if kernel[0] == "cross":
        size = kernel[1]
        selem = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        selem[center, :] = 1
        selem[:, center] = 1
        return selem
    elif kernel[0] == "square":
        size = kernel[1]
        return np.ones((size, size), dtype=np.uint8)
    return np.ones((3, 3), dtype=np.uint8)


def dilate(img, kernel):
    """Dilation using skimage.morphology.dilation."""
    selem = _make_structuring_element(kernel)
    out = np.zeros_like(img)
    for c in range(3):
        out[..., c] = skimage.morphology.dilation(img[..., c], selem)
    out[..., 3] = img[..., 3]
    return out


def erode(img, kernel):
    """Erosion using skimage.morphology.erosion."""
    selem = _make_structuring_element(kernel)
    out = np.zeros_like(img)
    for c in range(3):
        out[..., c] = skimage.morphology.erosion(img[..., c], selem)
    out[..., 3] = img[..., 3]
    return out


# ---------------------------------------------------------------------------
# Bilateral filter - numpy (matching millow)
# ---------------------------------------------------------------------------

def bilateral_filter(img, d, sigma_color, sigma_space):
    """Bilateral filter (matching millow)."""
    h, w = img.shape[:2]
    r_f = max(5, 2 * int(math.ceil(3.0 * sigma_space)) + 1) / 2.0 if d <= 0 else d / 2.0
    r = int(r_f)
    out = np.zeros_like(img)
    sc_norm = sigma_color / 255.0
    sc2 = 2.0 * sc_norm * sc_norm
    ss2 = 2.0 * sigma_space * sigma_space
    for y in range(h):
        for x in range(w):
            center_r = float(img[y, x, 0]) / 255.0
            center_g = float(img[y, x, 1]) / 255.0
            center_b = float(img[y, x, 2]) / 255.0
            w_sum_r = 0.0
            w_sum_g = 0.0
            w_sum_b = 0.0
            w_total = 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cy = _reflect_coord(y + dy, h - 1)
                    cx = _reflect_coord(x + dx, w - 1)
                    cr = float(img[cy, cx, 0]) / 255.0
                    cg = float(img[cy, cx, 1]) / 255.0
                    cb = float(img[cy, cx, 2]) / 255.0
                    dist2 = float(dy * dy + dx * dx)
                    dr = cr - center_r
                    dg = cg - center_g
                    db = cb - center_b
                    color_dist2 = dr * dr + dg * dg + db * db
                    weight = math.exp(-(dist2 / ss2 + color_dist2 / sc2))
                    w_total += weight
                    w_sum_r += cr * weight
                    w_sum_g += cg * weight
                    w_sum_b += cb * weight
            if w_total > 0.0:
                out[y, x, 0] = round_byte(w_sum_r / w_total * 255.0)
                out[y, x, 1] = round_byte(w_sum_g / w_total * 255.0)
                out[y, x, 2] = round_byte(w_sum_b / w_total * 255.0)
            out[y, x, 3] = img[y, x, 3]
    return out


# ---------------------------------------------------------------------------
# Affine transform - numpy (matching millow)
# ---------------------------------------------------------------------------

def _sample_bilinear_constant(img, fy, fx):
    """Bilinear sample with constant border (0,0,0,0)."""
    h, w = img.shape[:2]
    y0 = int(math.floor(fy))
    x0 = int(math.floor(fx))
    y1 = y0 + 1
    x1 = x0 + 1
    dy = fy - y0
    dx = fx - x0
    
    def get_p(y, x):
        if 0 <= y < h and 0 <= x < w:
            return img[y, x]
        return np.array([0, 0, 0, 0], dtype=np.uint8)
    
    p00 = get_p(y0, x0)
    p01 = get_p(y0, x1)
    p10 = get_p(y1, x0)
    p11 = get_p(y1, x1)
    
    out = np.zeros(4, dtype=np.uint8)
    for c in range(3):
        v00 = float(p00[c])
        v01 = float(p01[c])
        v10 = float(p10[c])
        v11 = float(p11[c])
        top = v00 * (1.0 - dx) + v01 * dx
        bot = v10 * (1.0 - dx) + v11 * dx
        out[c] = round_byte(top * (1.0 - dy) + bot * dy)
    out[3] = p00[3]
    return out


def rotate_any(img, angle_deg, interp):
    """Rotate (matching millow - clockwise)."""
    rad = -angle_deg * math.pi / 180.0
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    h, w = img.shape[:2]
    ch = h / 2.0
    cw = w / 2.0
    corners = [(-cw, -ch), (cw - 1.0, -ch), (-cw, ch - 1.0), (cw - 1.0, ch - 1.0)]
    x_min = x_max = y_min = y_max = 0.0
    for i, (x, y) in enumerate(corners):
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        if i == 0:
            x_min = x_max = rx
            y_min = y_max = ry
        else:
            x_min = min(x_min, rx)
            x_max = max(x_max, rx)
            y_min = min(y_min, ry)
            y_max = max(y_max, ry)
    dst_w = max(1, int(math.ceil(x_max - x_min + 1.0)))
    dst_h = max(1, int(math.ceil(y_max - y_min + 1.0)))
    dst_ch = dst_h / 2.0
    dst_cw = dst_w / 2.0
    matrix = [
        cos_a, sin_a, cw - dst_cw * cos_a - dst_ch * sin_a,
        -sin_a, cos_a, ch + dst_cw * sin_a - dst_ch * cos_a,
    ]
    return _affine_transform(img, matrix, dst_h, dst_w)


def _affine_transform(img, matrix, dst_h, dst_w):
    """General affine transform (matching millow)."""
    a, b, c, d, e, f = matrix
    det = a * e - b * d
    h, w = img.shape[:2]
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    for y in range(dst_h):
        for x in range(dst_w):
            xp = float(x)
            yp = float(y)
            sx = (e * xp - b * yp + b * f - e * c) / det
            sy = (-d * xp + a * yp + d * c - a * f) / det
            out[y, x] = _sample_bilinear_constant(img, sy, sx)
    return out


def translate(img, dy, dx):
    """Translate (matching millow)."""
    h, w = img.shape[:2]
    matrix = [1.0, 0.0, -dx, 0.0, 1.0, -dy]
    return _affine_transform(img, matrix, h, w)


def affine_transform(img, matrix, dst_h, dst_w):
    """General affine transform (matching millow)."""
    return _affine_transform(img, matrix, dst_h, dst_w)


# ---------------------------------------------------------------------------
# Feature detection - numpy (matching millow)
# ---------------------------------------------------------------------------

def corner_harris(img, block_size, ksize, k):
    """Corner detection (matching millow)."""
    h, w = img.shape[:2]
    gray = to_grayscale(img)
    smoothed = gaussian_blur(gray, 1.0)
    gx = _grad(smoothed, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    gy = _grad(smoothed, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    r = block_size // 2
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            sxx = syy = sxy = 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cy = clampi(y + dy, 0, h - 1)
                    cx = clampi(x + dx, 0, w - 1)
                    sxx += gx[cy][cx] * gx[cy][cx]
                    syy += gy[cy][cx] * gy[cy][cx]
                    sxy += gx[cy][cx] * gy[cy][cx]
            det = sxx * syy - sxy * sxy
            trace = sxx + syy
            out[y][x] = det - k * trace * trace
    return out


def corner_shi_tomasi(img, block_size, ksize):
    """Corner detection (matching millow)."""
    h, w = img.shape[:2]
    gray = to_grayscale(img)
    smoothed = gaussian_blur(gray, 1.0)
    gx = _grad(smoothed, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    gy = _grad(smoothed, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    r = block_size // 2
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            sxx = syy = sxy = 0.0
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    cy = clampi(y + dy, 0, h - 1)
                    cx = clampi(x + dx, 0, w - 1)
                    sxx += gx[cy][cx] * gx[cy][cx]
                    syy += gy[cy][cx] * gy[cy][cx]
                    sxy += gx[cy][cx] * gy[cy][cx]
            trace = sxx + syy
            det = sxx * syy - sxy * sxy
            disc = math.sqrt(max(0.0, trace * trace - 4.0 * det))
            l1 = (trace + disc) / 2.0
            l2 = (trace - disc) / 2.0
            out[y][x] = min(l1, l2)
    return out


def hog(img, cell_size, block_size, nbins):
    """HOG feature descriptor (matching millow)."""
    h, w = img.shape[:2]
    gray = to_grayscale(img)
    gray_luma = luma(gray[..., 0], gray[..., 1], gray[..., 2]).astype(float)
    gx = np.zeros((h, w), dtype=float)
    gy = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            r = gray_luma[y, min(x + 1, w - 1)]
            le = gray_luma[y, max(x - 1, 0)]
            gx[y, x] = r - le
            t = gray_luma[min(y + 1, h - 1), x]
            b = gray_luma[max(y - 1, 0), x]
            gy[y, x] = t - b
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    cell_hists = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            hist = [0.0] * nbins
            for dy in range(cell_size):
                for dx in range(cell_size):
                    y = cy * cell_size + dy
                    x = cx * cell_size + dx
                    mag = math.sqrt(gx[y, x] ** 2 + gy[y, x] ** 2)
                    angle = math.atan2(gy[y, x], gx[y, x]) * 180.0 / math.pi
                    a = angle + 180.0 if angle < 0.0 else angle
                    a = a % 180.0
                    bin_idx = clampi(int(a / 180.0 * nbins), 0, nbins - 1)
                    hist[bin_idx] += mag
            cell_hists.append(hist)
    result = []
    for by in range(n_cells_y - block_size + 1):
        for bx in range(n_cells_x - block_size + 1):
            block = []
            norm = 0.0
            for dy in range(block_size):
                for dx in range(block_size):
                    hist = cell_hists[(by + dy) * n_cells_x + (bx + dx)]
                    for b in range(nbins):
                        block.append(hist[b])
                        norm += hist[b] ** 2
            norm = math.sqrt(norm + 0.0001)
            for v in block:
                result.append(v / norm)
    return result


def lbp(img, radius, n_points):
    """Local Binary Pattern (matching millow)."""
    h, w = img.shape[:2]
    out = []
    for y in range(h):
        for x in range(w):
            center = _luma_at_reflect(img, y, x)
            code = 0
            for p in range(n_points):
                angle = 2.0 * math.pi * p / n_points
                py = y + radius * math.sin(angle)
                px = x + radius * math.cos(angle)
                y0 = int(math.floor(py))
                x0 = int(math.floor(px))
                y1 = y0 + 1
                x1 = x0 + 1
                dy = py - y0
                dx = px - x0
                v00 = _luma_at_reflect(img, y0, x0)
                v01 = _luma_at_reflect(img, y0, x1)
                v10 = _luma_at_reflect(img, y1, x0)
                v11 = _luma_at_reflect(img, y1, x1)
                neighbor = v00 * (1.0 - dy) * (1.0 - dx) + v01 * (1.0 - dy) * dx + v10 * dy * (1.0 - dx) + v11 * dy * dx
                if neighbor >= center:
                    code += 1 << p
            out.append(code)
    return out


def lbp_histogram(img, radius, n_points):
    """LBP histogram (matching millow)."""
    codes = lbp(img, radius, n_points)
    n_bins = 1 << n_points
    hist = [0] * n_bins
    for code in codes:
        hist[code] += 1
    return hist


# ---------------------------------------------------------------------------
# Measurement - skimage
# ---------------------------------------------------------------------------

def connected_components(img):
    """Connected components using skimage.measure.label."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2])
    binary = (gray > 0).astype(np.int32)
    labeled, num = measure.label(binary, return_num=True, connectivity=1)
    h, w = labeled.shape
    return [[int(labeled[y, x]) for x in range(w)] for y in range(h)], num


def find_contours(img):
    """Find contours using skimage.measure.find_contours."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2])
    binary = (gray > 0).astype(np.float64)
    contours = measure.find_contours(binary, level=0.5)
    return [[(int(y), int(x)) for y, x in contour] for contour in contours]


def moments(img):
    """Image moments using skimage.measure.moments."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(float) / 255.0
    m = measure.moments(gray, order=3)
    return [
        float(m[0, 0]),
        float(m[1, 0]),
        float(m[0, 1]),
        float(m[2, 0]),
        float(m[1, 1]),
        float(m[0, 2]),
        float(m[3, 0]),
        float(m[2, 1]),
        float(m[1, 2]),
        float(m[0, 3]),
    ]


def hu_moments(img):
    """Hu moments using skimage.measure.moments_hu."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(float) / 255.0
    hu = measure.moments_hu(measure.moments(gray, order=3))
    return [float(v) for v in hu]


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


def int_arr2_to_moonbit(name: str, data) -> str:
    """Convert a 2D list of ints to MoonBit Array[Array[Int]]."""
    lines = [f"///|", f"let {name} : Array[Array[Int]] = ["]
    for row in data:
        lines.append("  [" + ", ".join(str(v) for v in row) + "],")
    lines.append("]")
    return "\n".join(lines)


def float_arr_to_moonbit(name: str, data) -> str:
    """Convert a list of floats to MoonBit Array[Double]."""
    lines = [f"///|", f"let {name} : Array[Double] = ["]
    for i in range(0, len(data), 8):
        chunk = data[i:i + 8]
        lines.append("  " + ", ".join(repr(float(v)) for v in chunk) + ",")
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
        img_to_moonbit("exp_brightness_1_5", adjust_brightness(grad, 1.5)),
        img_to_moonbit("exp_contrast_1_5", adjust_contrast(grad, 1.5)),
        img_to_moonbit("exp_gamma_2_0", adjust_gamma(grad, 2.0)),
        img_to_moonbit("exp_normalize", normalize(grad, 0, 255)),
        "",
        "// ===== Threshold =====",
        "",
        img_to_moonbit("exp_threshold_128", threshold(grad, 128)),
        img_to_moonbit("exp_threshold_inv_128", threshold_inv(grad, 128)),
    ]

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
        "// ===== Edges =====",
        "",
        img_to_moonbit("exp_sobel", sobel(dot)),
        img_to_moonbit("exp_prewitt", prewitt(dot)),
        img_to_moonbit("exp_scharr", scharr(dot)),
        img_to_moonbit("exp_laplacian", laplacian(dot)),
        arr2_to_moonbit("exp_sobel_x", sobel_x(dot)),
        arr2_to_moonbit("exp_sobel_y", sobel_y(dot)),
        "",
        "// ===== Morphology =====",
        "",
        img_to_moonbit("exp_erode_cross", erode(dot, ("cross", 3))),
        img_to_moonbit("exp_dilate_cross", dilate(dot, ("cross", 3))),
        img_to_moonbit("exp_erode_square", erode(dot, ("square", 3))),
        img_to_moonbit("exp_dilate_square", dilate(dot, ("square", 3))),
        "",
        "// ===== Bilateral filter =====",
        "",
        img_to_moonbit("exp_bilateral_5_10_10", bilateral_filter(grad, 5, 10.0, 10.0)),
        "",
        "// ===== Affine transform =====",
        "",
        img_to_moonbit("exp_rotate_any_45", rotate_any(grad, 45.0, "bilinear")),
        img_to_moonbit("exp_translate_1_1", translate(grad, 1.0, 1.0)),
        "",
        "// ===== Feature detection =====",
        "",
        arr2_to_moonbit("exp_corner_harris", corner_harris(grad, 2, 3, 0.04)),
        arr2_to_moonbit("exp_corner_shi_tomasi", corner_shi_tomasi(grad, 2, 3)),
        float_arr_to_moonbit("exp_hog", hog(grad, 2, 1, 9)),
        int_arr_to_moonbit("exp_lbp", lbp(grad, 1, 8)),
        int_arr_to_moonbit("exp_lbp_hist", lbp_histogram(grad, 1, 8)),
        "",
        "// ===== Measurement =====",
        "",
        int_arr2_to_moonbit("exp_connected_components", connected_components(dot)[0]),
        float_arr_to_moonbit("exp_moments", moments(grad)),
        float_arr_to_moonbit("exp_hu_moments", hu_moments(grad)),
        "",
    ])

    return "\n".join(sections)


if __name__ == "__main__":
    out = generate()
    out_path = Path(__file__).parent / "fixtures_test.mbt"
    out_path.write_text(out, encoding="utf-8")
    print(f"Generated {out_path} ({len(out)} bytes)")