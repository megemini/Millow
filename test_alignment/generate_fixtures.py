#!/usr/bin/env python3
"""Generate MoonBit fixture data for the millow alignment test suite.

This script builds small test images, runs reference implementations using
numpy to match millow's exact implementation, and writes the input + expected-output
byte arrays to ``fixtures_test.mbt`` as ``Array[Int]`` literals.

Usage:
    source $HOME/venv310/bin/activate
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
    """Use Pillow Image.resize(resample=Image.BILINEAR)."""
    pil = numpy_to_pil(img)
    resized = pil.resize((dst_w, dst_h), resample=Image.BILINEAR)
    return pil_to_numpy(resized)


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
    """Histogram equalization using skimage."""
    from skimage import exposure
    out = img.copy()
    for c in range(3):
        channel = img[..., c]
        equalized = exposure.equalize_hist(channel)
        out[..., c] = (equalized * 255).astype(np.uint8)
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


def _gaussian_blur_2d(arr, sigma):
    """Gaussian blur on 2D double array with zero border (matching millow)."""
    h, w = arr.shape
    radius = int(math.ceil(sigma * 3.0))
    ks = clampi(radius * 2 + 1, 3, 99)
    k = _gaussian_kernel(ks, sigma)
    c = ks // 2
    # Horizontal pass
    temp = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for i in range(ks):
                dx = i - c
                cx = x + dx
                v = arr[y, cx] if cx >= 0 and cx < w else 0.0
                acc += k[i] * v
            temp[y, x] = acc
    # Vertical pass
    out = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for i in range(ks):
                dy = i - c
                cy = y + dy
                v = temp[cy, x] if cy >= 0 and cy < h else 0.0
                acc += k[i] * v
            out[y, x] = acc
    return out


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
    gx = _grad(img, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    gy = _grad(img, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return _mag_image(gx, gy)


def sobel_x(img):
    """Sobel horizontal gradient (matching millow)."""
    gx = _grad(img, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    return gx


def sobel_y(img):
    """Sobel vertical gradient (matching millow)."""
    gy = _grad(img, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return gy


def prewitt(img):
    """Prewitt edge detection (matching millow)."""
    gx = _grad(img, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    gy = _grad(img, [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    return _mag_image(gx, gy)


def scharr(img):
    """Scharr edge detection (matching millow)."""
    gx = _grad(img, [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
    gy = _grad(img, [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
    return _mag_image(gx, gy)


def laplacian(img):
    """Laplacian edge detection (matching millow)."""
    h, w = img.shape[:2]
    g = _grad(img, [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
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
    win_size = max(5, 2 * int(math.ceil(3.0 * sigma_space)) + 1) if d <= 0 else d
    r = win_size // 2
    out = np.zeros_like(img)
    sc2 = 2.0 * sigma_color * sigma_color
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
    """Bilinear sample with constant border (0,0,0,255)."""
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
        return np.array([0, 0, 0, 255], dtype=np.uint8)
    
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
    """Rotate (clockwise, matching millow) using a numpy reference.

    Mirrors millow's `rotate_any` exactly: rotate by `angle_deg` clockwise
    around the image center, expand the canvas to fit the rotated image, and
    center the result so the source center maps to the canvas center. Border
    samples use opaque black `(0, 0, 0, 255)`.
    """
    h, w = img.shape[:2]
    rad = -math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    cx = w / 2.0
    cy = h / 2.0
    # matrix2: rotate around (cx, cy) — forward map.
    t2 = -cx * cos_a - cy * sin_a + cx
    t5 = cx * sin_a - cy * cos_a + cy
    matrix2 = [cos_a, sin_a, t2, -sin_a, cos_a, t5]
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    xs, ys = [], []
    for x, y in corners:
        xs.append(cos_a * x + sin_a * y + t2)
        ys.append(-sin_a * x + cos_a * y + t5)
    dst_w = int(math.ceil(max(xs) - min(xs)))
    dst_h = int(math.ceil(max(ys) - min(ys)))
    # Center the rotated image in the expanded canvas.
    dx_expand = (dst_w - w) / 2.0
    dy_expand = (dst_h - h) / 2.0
    matrix3 = [
        matrix2[0], matrix2[1], matrix2[2] + dx_expand,
        matrix2[3], matrix2[4], matrix2[5] + dy_expand,
    ]
    if interp == "nearest":
        return _affine_transform_nearest(img, matrix3, dst_h, dst_w)
    return _affine_transform(img, matrix3, dst_h, dst_w)


def _affine_transform_nearest(img, matrix, dst_h, dst_w):
    """Affine transform with nearest-neighbor sampling (matching millow)."""
    a, b, c, d, e, f = matrix
    det = a * e - b * d
    h, w = img.shape[:2]
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    for y in range(dst_h):
        for x in range(dst_w):
            sx = (e * x - b * y + b * f - e * c) / det
            sy = (-d * x + a * y + d * c - a * f) / det
            if sx < 0.0 or sy < 0.0 or sx >= w or sy >= h:
                out[y, x] = np.array([0, 0, 0, 255], dtype=np.uint8)
            else:
                si = max(0, min(h - 1, int(sy)))
                sj = max(0, min(w - 1, int(sx)))
                out[y, x] = img[si, sj]
    return out


def _affine_transform(img, matrix, dst_h, dst_w):
    """Affine transform with bilinear sampling (matching millow)."""
    a, b, c, d, e, f = matrix
    det = a * e - b * d
    h, w = img.shape[:2]
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    for y in range(dst_h):
        for x in range(dst_w):
            sx = (e * x - b * y + b * f - e * c) / det
            sy = (-d * x + a * y + d * c - a * f) / det
            y0 = int(math.floor(sy))
            x0 = int(math.floor(sx))
            y1 = y0 + 1
            x1 = x0 + 1
            dy = sy - y0
            dx = sx - x0
            def gp(yy, xx):
                if 0 <= yy < h and 0 <= xx < w:
                    return img[yy, xx]
                return np.array([0, 0, 0, 255], dtype=np.uint8)
            p00 = gp(y0, x0); p01 = gp(y0, x1)
            p10 = gp(y1, x0); p11 = gp(y1, x1)
            for ch in range(3):
                v = (p00[ch] * (1 - dy) * (1 - dx) + p01[ch] * (1 - dy) * dx +
                     p10[ch] * dy * (1 - dx) + p11[ch] * dy * dx)
                out[y, x, ch] = round_byte(v)
            out[y, x, 3] = p00[3]
    return out


def translate(img, dy, dx, interp="bilinear"):
    """Translate (matching millow) using the affine transform.

    millow's forward map is `dst = src + (-dx, -dy)`. Output keeps the source
    dimensions; uncovered pixels become opaque black.
    """
    matrix = [1.0, 0.0, -dx, 0.0, 1.0, -dy]
    if interp == "nearest":
        return _affine_transform_nearest(img, matrix, img.shape[0], img.shape[1])
    return _affine_transform(img, matrix, img.shape[0], img.shape[1])


def affine_transform(img, matrix, dst_h, dst_w):
    """General affine transform (matching millow)."""
    return _affine_transform(img, matrix, dst_h, dst_w)


# ---------------------------------------------------------------------------
# Feature detection - numpy (matching millow)
# ---------------------------------------------------------------------------

def corner_harris(img, k):
    """Corner detection matching millow's implementation."""
    gray = to_grayscale(img)
    h, w = img.shape[:2]
    luma_arr = luma(gray[..., 0], gray[..., 1], gray[..., 2]).astype(float)

    def luma_at(y, x):
        if y < 0 or y >= h or x < 0 or x >= w:
            return 0.0
        return luma_arr[y, x]

    gx = np.zeros((h, w), dtype=float)
    gy = np.zeros((h, w), dtype=float)
    sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    for y in range(h):
        for x in range(w):
            ax = 0.0
            ay = 0.0
            for ky in range(3):
                for kx in range(3):
                    v = luma_at(y + ky - 1, x + kx - 1)
                    ax += sobel_x[ky][kx] * v
                    ay += sobel_y[ky][kx] * v
            gx[y, x] = ax
            gy[y, x] = ay

    gx2 = gx * gx
    gy2 = gy * gy
    gxy = gx * gy

    sxx = _gaussian_blur_2d(gx2, 1.0)
    syy = _gaussian_blur_2d(gy2, 1.0)
    sxy = _gaussian_blur_2d(gxy, 1.0)

    out = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            det = sxx[y, x] * syy[y, x] - sxy[y, x] * sxy[y, x]
            trace = sxx[y, x] + syy[y, x]
            out[y, x] = det - k * trace * trace
    return out.tolist()


def corner_shi_tomasi(img):
    """Corner detection matching millow's implementation."""
    gray = to_grayscale(img)
    h, w = img.shape[:2]
    luma_arr = luma(gray[..., 0], gray[..., 1], gray[..., 2]).astype(float)

    def luma_at(y, x):
        if y < 0 or y >= h or x < 0 or x >= w:
            return 0.0
        return luma_arr[y, x]

    gx = np.zeros((h, w), dtype=float)
    gy = np.zeros((h, w), dtype=float)
    sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    for y in range(h):
        for x in range(w):
            ax = 0.0
            ay = 0.0
            for ky in range(3):
                for kx in range(3):
                    v = luma_at(y + ky - 1, x + kx - 1)
                    ax += sobel_x[ky][kx] * v
                    ay += sobel_y[ky][kx] * v
            gx[y, x] = ax
            gy[y, x] = ay

    gx2 = gx * gx
    gy2 = gy * gy
    gxy = gx * gy

    sxx = _gaussian_blur_2d(gx2, 1.0)
    syy = _gaussian_blur_2d(gy2, 1.0)
    sxy = _gaussian_blur_2d(gxy, 1.0)

    out = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            trace = sxx[y, x] + syy[y, x]
            det = sxx[y, x] * syy[y, x] - sxy[y, x] * sxy[y, x]
            disc = max(trace * trace - 4.0 * det, 0.0)
            import math
            disc = math.sqrt(disc)
            l1 = (trace + disc) / 2.0
            l2 = (trace - disc) / 2.0
            out[y, x] = min(l1, l2)
    return out.tolist()


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
    """Image moments matching millow's implementation (raw luma 0-255)."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(float)
    h, w = gray.shape
    m = [0.0] * 10
    for y in range(h):
        for x in range(w):
            v = gray[y, x]
            m[0] += v
            m[1] += y * v
            m[2] += x * v
            m[3] += y * y * v
            m[4] += y * x * v
            m[5] += x * x * v
            m[6] += y * y * y * v
            m[7] += y * y * x * v
            m[8] += y * x * x * v
            m[9] += x * x * x * v
    return [float(v) for v in m]


def hu_moments(img):
    """Hu moments matching millow's implementation (raw luma 0-255)."""
    m = moments(img)
    m00 = m[0]
    if m00 == 0.0:
        return [0.0] * 7
    yb = m[1] / m00
    xb = m[2] / m00
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(float)
    h, w = gray.shape
    mu20 = mu11 = mu02 = mu30 = mu21 = mu12 = mu03 = 0.0
    for y in range(h):
        for x in range(w):
            v = gray[y, x]
            dy = y - yb
            dx = x - xb
            mu20 += dy * dy * v
            mu11 += dy * dx * v
            mu02 += dx * dx * v
            mu30 += dy * dy * dy * v
            mu21 += dy * dy * dx * v
            mu12 += dy * dx * dx * v
            mu03 += dx * dx * dx * v
    n20 = mu20 / (m00 * m00)
    n11 = mu11 / (m00 * m00)
    n02 = mu02 / (m00 * m00)
    n30 = mu30 / (m00 * m00 * m00 ** 0.5)
    n21 = mu21 / (m00 * m00 * m00 ** 0.5)
    n12 = mu12 / (m00 * m00 * m00 ** 0.5)
    n03 = mu03 / (m00 * m00 * m00 ** 0.5)
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4.0 * n11 * n11
    h3 = (n30 - 3.0 * n12) ** 2 + (3.0 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = (n30 - 3.0 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2) + \
         (3.0 * n21 - n03) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + \
         4.0 * n11 * (n30 + n12) * (n21 + n03)
    h7 = (3.0 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3.0 * (n21 + n03) ** 2) - \
         (n30 - 3.0 * n12) * (n21 + n03) * (3.0 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    return [float(v) for v in [h1, h2, h3, h4, h5, h6, h7]]


# ---------------------------------------------------------------------------
# OCR preprocessing
# ---------------------------------------------------------------------------

def project_horizontal(img):
    """Sum of luma per row, length h."""
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    return np.sum(y, axis=1).tolist()


def project_vertical(img):
    """Sum of luma per column, length w."""
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    return np.sum(y, axis=0).tolist()


def project_horizontal_binary(img):
    """Count of non-zero (foreground) pixels per row, length h."""
    fg = (img[..., 0] != 0) | (img[..., 1] != 0) | (img[..., 2] != 0)
    return np.count_nonzero(fg, axis=1).tolist()


def project_vertical_binary(img):
    """Count of non-zero (foreground) pixels per column, length w."""
    fg = (img[..., 0] != 0) | (img[..., 1] != 0) | (img[..., 2] != 0)
    return np.count_nonzero(fg, axis=0).tolist()


def auto_crop(img, threshold=240):
    """Crop to bounding box of pixels with luma < threshold."""
    y = luma(img[..., 0], img[..., 1], img[..., 2])
    mask = y < threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return img[y_min:y_max + 1, x_min:x_max + 1].copy()


def remove_small_objects(img, min_area):
    """Remove connected components with area < min_area (8-connectivity)."""
    fg = (img[..., 0] != 0) | (img[..., 1] != 0) | (img[..., 2] != 0)
    out = np.zeros_like(img)
    out[..., 3] = img[..., 3]
    labeled = skimage.measure.label(fg, connectivity=2)
    props = skimage.measure.regionprops(labeled)
    for prop in props:
        if prop.area >= min_area:
            for r, c in prop.coords:
                out[r, c, 0] = 255
                out[r, c, 1] = 255
                out[r, c, 2] = 255
    return out


def remove_small_holes(img, max_area):
    """Fill holes (dark regions) with area <= max_area."""
    fg = (img[..., 0] != 0) | (img[..., 1] != 0) | (img[..., 2] != 0)
    filled = morphology.remove_small_holes(fg, area_threshold=max_area + 1)
    out = np.zeros_like(img)
    out[..., 3] = img[..., 3]
    out[filled, 0] = 255
    out[filled, 1] = 255
    out[filled, 2] = 255
    return out


def threshold_niblack(img, window_size, k):
    """Niblack local thresholding: T = mean + k * std."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    h, w = gray.shape
    r = window_size // 2
    out = np.zeros_like(img)
    out[..., 3] = img[..., 3]
    # Build integral images
    w1 = w + 1
    integ = np.zeros((h + 1, w1))
    integ2 = np.zeros((h + 1, w1))
    for y in range(h):
        row_sum = 0.0
        row_sum2 = 0.0
        for x in range(w):
            v = gray[y, x]
            row_sum += v
            row_sum2 += v * v
            integ[y + 1, x + 1] = integ[y, x + 1] + row_sum
            integ2[y + 1, x + 1] = integ2[y, x + 1] + row_sum2
    for y in range(h):
        y0 = max(0, y - r)
        y1 = min(h - 1, y + r)
        for x in range(w):
            x0 = max(0, x - r)
            x1 = min(w - 1, x + r)
            count = (y1 - y0 + 1) * (x1 - x0 + 1)
            cnt = float(count)
            s = integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]
            s2 = integ2[y1 + 1, x1 + 1] - integ2[y0, x1 + 1] - integ2[y1 + 1, x0] + integ2[y0, x0]
            mean = s / cnt
            variance = s2 / cnt - mean * mean
            std = math.sqrt(variance) if variance > 0 else 0.0
            t = mean + k * std
            vb = 255 if gray[y, x] >= t else 0
            out[y, x, 0] = vb
            out[y, x, 1] = vb
            out[y, x, 2] = vb
    return out


# ---------------------------------------------------------------------------
# Integral image, guided filter, line removal, perspective, model input
# ---------------------------------------------------------------------------

def integral_image_ref(img):
    """Integral image (summed-area table) from luma, (h+1)x(w+1)."""
    y = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    h, w = y.shape
    integ = np.zeros((h + 1, w + 1))
    integ[1:, 1:] = y.cumsum(axis=0).cumsum(axis=1)
    return integ.flatten().tolist()


def ssim_ref(a, b):
    """SSIM with uniform 11x11 window (matching millow's implementation)."""
    ya = luma(a[..., 0], a[..., 1], a[..., 2]).astype(np.float64)
    yb = luma(b[..., 0], b[..., 1], b[..., 2]).astype(np.float64)
    return float(skimage.metrics.structural_similarity(
        ya, yb, win_size=11, gaussian_weights=False, use_sample_covariance=False, data_range=255.0
    ))


def guided_filter_ref(guide, img, radius, epsilon):
    """Guided filter using box mean via integral images."""
    Ig = luma(guide[..., 0], guide[..., 1], guide[..., 2]).astype(np.float64)
    Ip = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    h, w = Ig.shape

    def box_mean(arr, r):
        integ = np.zeros((h + 1, w + 1))
        integ[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
        y_idx = np.arange(h)
        x_idx = np.arange(w)
        y0 = np.clip(y_idx - r, 0, h - 1)
        y1 = np.clip(y_idx + r, 0, h - 1)
        x0 = np.clip(x_idx - r, 0, w - 1)
        x1 = np.clip(x_idx + r, 0, w - 1)
        Y0, X0 = np.meshgrid(y0, x0, indexing='ij')
        Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
        n = (Y1 - Y0 + 1) * (X1 - X0 + 1)
        s = integ[Y1 + 1, X1 + 1] - integ[Y0, X1 + 1] - integ[Y1 + 1, X0] + integ[Y0, X0]
        return s / n

    mean_I = box_mean(Ig, radius)
    mean_p = box_mean(Ip, radius)
    mean_Ip = box_mean(Ig * Ip, radius)
    mean_II = box_mean(Ig * Ig, radius)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I
    mean_a = box_mean(a, radius)
    mean_b = box_mean(b, radius)
    q = mean_a * Ig + mean_b
    out = np.zeros_like(guide)
    out[..., 0] = np.clip(q, 0, 255).astype(np.uint8)
    out[..., 1] = out[..., 0]
    out[..., 2] = out[..., 0]
    out[..., 3] = 255
    return out


def remove_lines_ref(img, direction, max_thickness):
    """Remove horizontal/vertical lines via morphological opening + subtraction.
    Matches millow: operates per-RGB-channel (grayscale min/max morphology),
    then subtracts the opened image from the original."""
    h, w = img.shape[:2]
    thickness = max(max_thickness, 1)
    result = img.copy()
    if direction in ('horizontal', 'both'):
        kw = max(w // 2, 3)
        kernel = np.ones((thickness, kw), dtype=bool)
        opened = _morph_open_rgb(img, kernel)
        for c in range(3):
            result[..., c] = np.clip(result[..., c].astype(int) - opened[..., c].astype(int), 0, 255).astype(np.uint8)
    if direction in ('vertical', 'both'):
        kh = max(h // 2, 3)
        kernel = np.ones((kh, thickness), dtype=bool)
        opened = _morph_open_rgb(img, kernel)
        for c in range(3):
            result[..., c] = np.clip(result[..., c].astype(int) - opened[..., c].astype(int), 0, 255).astype(np.uint8)
    return result


def _morph_open_rgb(img, kernel):
    """Grayscale morphological opening (erode then dilate) on each RGB channel
    with replicate border, matching millow's morph_open."""
    from scipy import ndimage
    h, w = img.shape[:2]
    kh, kw = kernel.shape
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[..., c].astype(int)
        # Replicate border padding.
        padded = np.pad(ch, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='edge')
        eroded = np.zeros((h, w), dtype=int)
        for y in range(h):
            for x in range(w):
                window = padded[y:y + kh, x:x + kw]
                eroded[y, x] = window[kernel].min()
        # Dilate the eroded result.
        padded_e = np.pad(eroded, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode='edge')
        dilated = np.zeros((h, w), dtype=int)
        for y in range(h):
            for x in range(w):
                window = padded_e[y:y + kh, x:x + kw]
                dilated[y, x] = window[kernel].max()
        out[..., c] = np.clip(dilated, 0, 255).astype(np.uint8)
    out[..., 3] = img[..., 3]
    return out


def resize_to_model_input_ref(img, model_h, model_w, interp='bilinear', pad_color=(255, 255, 255, 255)):
    """Resize preserving aspect ratio, then center-pad to exact size."""
    h, w = img.shape[:2]
    scale = min(model_h / h, model_w / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    if interp == 'nearest':
        resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.NEAREST))
    else:
        resized = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    out = np.full((model_h, model_w, 4), 0, dtype=np.uint8)
    out[..., 0] = pad_color[0]
    out[..., 1] = pad_color[1]
    out[..., 2] = pad_color[2]
    out[..., 3] = pad_color[3]
    top = (model_h - new_h) // 2
    left = (model_w - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def perspective_transform_ref(img, matrix_3x3, dst_h, dst_w, interp='bilinear'):
    """Perspective transform using skimage."""
    m = np.array(matrix_3x3).reshape(3, 3)
    # skimage warp uses inverse map: output → input
    inv = np.linalg.inv(m)

    def inverse_map(coords):
        pts = np.column_stack([coords[:, 1], coords[:, 0], np.ones(len(coords))])
        src = (inv @ pts.T).T
        w = src[:, 2]
        src = src[:, :2] / w[:, None]
        return np.column_stack([src[:, 1], src[:, 0]])  # (sy, sx)

    from skimage.transform import warp
    out = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    out[..., 3] = 255
    for c in range(3):
        # skimage.warp preserves the input range for float inputs, so we pass
        # the channel as float64 in [0, 255] and clip the warped result directly.
        warped = warp(img[..., c].astype(np.float64), inverse_map, output_shape=(dst_h, dst_w),
                       order=1 if interp == 'bilinear' else 0, mode='constant', cval=0)
        out[..., c] = np.clip(np.round(warped), 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Test images for OCR preprocessing
# ---------------------------------------------------------------------------

def bordered_6x6():
    """6x6 image with white border and dark content in rows 2-3, cols 2-3."""
    img = np.full((6, 6, 4), 255, dtype=np.uint8)
    img[..., 3] = 255
    img[2:4, 2:4, 0] = 0
    img[2:4, 2:4, 1] = 0
    img[2:4, 2:4, 2] = 0
    return img


def dots_5x5():
    """5x5 binary image: one 3x3 block and two isolated single-pixel dots."""
    img = np.zeros((5, 5, 4), dtype=np.uint8)
    img[..., 3] = 255
    # 3x3 block at top-left
    img[0:3, 0:3, 0] = 255
    img[0:3, 0:3, 1] = 255
    img[0:3, 0:3, 2] = 255
    # Single pixel at (1, 4)
    img[1, 4] = (255, 255, 255, 255)
    # Single pixel at (4, 4)
    img[4, 4] = (255, 255, 255, 255)
    return img


def holed_5x5():
    """5x5 binary image: a 5x5 white block with a 1x1 black hole at center."""
    img = np.full((5, 5, 4), 255, dtype=np.uint8)
    img[..., 3] = 255
    img[2, 2, 0] = 0
    img[2, 2, 1] = 0
    img[2, 2, 2] = 0
    return img


def lines_6x6():
    """6x6 binary image with a horizontal line (row 1) and a vertical line (col 4),
    plus a 2x2 block at bottom-left."""
    img = np.zeros((6, 6, 4), dtype=np.uint8)
    img[..., 3] = 255
    # Horizontal line at row 1
    img[1, :, 0] = 255
    img[1, :, 1] = 255
    img[1, :, 2] = 255
    # Vertical line at col 4
    img[:, 4, 0] = 255
    img[:, 4, 1] = 255
    img[:, 4, 2] = 255
    # 2x2 block at (4,0)-(5,1)
    img[4:6, 0:2, 0] = 255
    img[4:6, 0:2, 1] = 255
    img[4:6, 0:2, 2] = 255
    return img


def arr_to_moonbit(name: str, data) -> str:
    """Convert a flat array/list of ints to a MoonBit Array[Int] literal."""
    vals = list(data)
    lines = [f"///|", f"let {name} : Array[Int] = ["]
    for i in range(0, len(vals), 8):
        chunk = vals[i:i + 8]
        lines.append("  " + ", ".join(str(v) for v in chunk) + ",")
    lines.append("]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# New algorithm reference implementations
# ---------------------------------------------------------------------------

def threshold_bernsen_ref(img, window_size, contrast_limit):
    """Bernsen local threshold: T = (min+max)/2, low-contrast -> background."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(int)
    h, w = gray.shape
    r = window_size // 2
    out = np.zeros_like(img)
    for y in range(h):
        y0, y1 = clampi(y - r, 0, h - 1), clampi(y + r, 0, h - 1)
        for x in range(w):
            x0, x1 = clampi(x - r, 0, w - 1), clampi(x + r, 0, w - 1)
            window = gray[y0:y1 + 1, x0:x1 + 1]
            vmin, vmax = int(window.min()), int(window.max())
            pl = int(gray[y, x])
            if vmax - vmin < contrast_limit:
                vb = 255
            elif pl >= (vmin + vmax) // 2:
                vb = 255
            else:
                vb = 0
            out[y, x, 0] = vb
            out[y, x, 1] = vb
            out[y, x, 2] = vb
            out[y, x, 3] = img[y, x, 3]
    return out


def threshold_wolf_ref(img, window_size, k):
    """Wolf normalized Niblack: T = mean + k*(std/std_max)*(mean-min_global)."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    h, w = gray.shape
    r = window_size // 2
    # Integral images for mean/variance.
    w1 = w + 1
    integ = np.zeros((h + 1, w1))
    integ2 = np.zeros((h + 1, w1))
    for y in range(h):
        rs = 0.0
        rs2 = 0.0
        for x in range(w):
            v = gray[y, x]
            rs += v
            rs2 += v * v
            integ[y + 1, x + 1] = integ[y, x + 1] + rs
            integ2[y + 1, x + 1] = integ2[y, x + 1] + rs2
    min_global = float(gray.min())
    stds = np.zeros((h, w))
    std_max = 0.0
    for y in range(h):
        y0, y1 = clampi(y - r, 0, h - 1), clampi(y + r, 0, h - 1)
        for x in range(w):
            x0, x1 = clampi(x - r, 0, w - 1), clampi(x + r, 0, w - 1)
            cnt = (y1 - y0 + 1) * (x1 - x0 + 1)
            s = integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]
            s2 = integ2[y1 + 1, x1 + 1] - integ2[y0, x1 + 1] - integ2[y1 + 1, x0] + integ2[y0, x0]
            mean = s / cnt
            var = s2 / cnt - mean * mean
            std = math.sqrt(var) if var > 0 else 0.0
            stds[y, x] = std
            if std > std_max:
                std_max = std
    out = np.zeros_like(img)
    for y in range(h):
        y0, y1 = clampi(y - r, 0, h - 1), clampi(y + r, 0, h - 1)
        for x in range(w):
            x0, x1 = clampi(x - r, 0, w - 1), clampi(x + r, 0, w - 1)
            cnt = (y1 - y0 + 1) * (x1 - x0 + 1)
            s = integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]
            mean = s / cnt
            std = stds[y, x]
            norm = std / std_max if std_max > 0 else 0.0
            t = mean + k * norm * (mean - min_global)
            vb = 255 if float(gray[y, x]) >= t else 0
            out[y, x, 0] = vb
            out[y, x, 1] = vb
            out[y, x, 2] = vb
            out[y, x, 3] = img[y, x, 3]
    return out


def adaptive_threshold_mean_ref(img, block_size, c):
    """OpenCV ADAPTIVE_THRESH_MEAN_C: T = mean(window) - C, pixel > T -> 255."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    h, w = gray.shape
    r = block_size // 2
    w1 = w + 1
    integ = np.zeros((h + 1, w1))
    for y in range(h):
        rs = 0.0
        for x in range(w):
            rs += gray[y, x]
            integ[y + 1, x + 1] = integ[y, x + 1] + rs
    out = np.zeros_like(img)
    for y in range(h):
        y0, y1 = clampi(y - r, 0, h - 1), clampi(y + r, 0, h - 1)
        for x in range(w):
            x0, x1 = clampi(x - r, 0, w - 1), clampi(x + r, 0, w - 1)
            cnt = (y1 - y0 + 1) * (x1 - x0 + 1)
            s = integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]
            t = s / cnt - c
            vb = 255 if float(gray[y, x]) > t else 0
            out[y, x, 0] = vb
            out[y, x, 1] = vb
            out[y, x, 2] = vb
            out[y, x, 3] = img[y, x, 3]
    return out


def adaptive_threshold_gaussian_ref(img, block_size, c):
    """OpenCV ADAPTIVE_THRESH_GAUSSIAN_C: T = gauss_weighted_mean(window) - C."""
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    h, w = gray.shape
    r = block_size // 2
    sigma = 0.3 * (r - 1) + 0.8
    kernel = np.zeros((block_size, block_size))
    ksum = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            v = math.exp(-((dy * dy + dx * dx)) / (2.0 * sigma * sigma))
            kernel[dy + r, dx + r] = v
            ksum += v
    kernel /= ksum
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for dy in range(-r, r + 1):
                sy = clampi(y + dy, 0, h - 1)
                for dx in range(-r, r + 1):
                    sx = clampi(x + dx, 0, w - 1)
                    acc += gray[sy, sx] * kernel[dy + r, dx + r]
            t = acc - c
            vb = 255 if float(gray[y, x]) > t else 0
            out[y, x, 0] = vb
            out[y, x, 1] = vb
            out[y, x, 2] = vb
            out[y, x, 3] = img[y, x, 3]
    return out


def wiener_filter_ref(img, window_size, noise_var):
    """scipy.signal.wiener-style local Wiener on luma stats, applied per RGB."""
    h, w = img.shape[:2]
    r = window_size // 2
    gray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    w1 = w + 1
    integ = np.zeros((h + 1, w1))
    integ2 = np.zeros((h + 1, w1))
    for y in range(h):
        rs = 0.0
        rs2 = 0.0
        for x in range(w):
            v = gray[y, x]
            rs += v
            rs2 += v * v
            integ[y + 1, x + 1] = integ[y, x + 1] + rs
            integ2[y + 1, x + 1] = integ2[y, x + 1] + rs2
    if noise_var < 0:
        total = 0.0
        for y in range(h):
            y0, y1 = clampi(y - r, 0, h - 1), clampi(y + r, 0, h - 1)
            for x in range(w):
                x0, x1 = clampi(x - r, 0, w - 1), clampi(x + r, 0, w - 1)
                cnt = (y1 - y0 + 1) * (x1 - x0 + 1)
                s = integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]
                s2 = integ2[y1 + 1, x1 + 1] - integ2[y0, x1 + 1] - integ2[y1 + 1, x0] + integ2[y0, x0]
                mean = s / cnt
                var = s2 / cnt - mean * mean
                total += var
        nvar = total / (h * w)
    else:
        nvar = noise_var
    out = np.zeros_like(img)
    for y in range(h):
        y0, y1 = clampi(y - r, 0, h - 1), clampi(y + r, 0, h - 1)
        for x in range(w):
            x0, x1 = clampi(x - r, 0, w - 1), clampi(x + r, 0, w - 1)
            cnt = (y1 - y0 + 1) * (x1 - x0 + 1)
            s = integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0]
            s2 = integ2[y1 + 1, x1 + 1] - integ2[y0, x1 + 1] - integ2[y1 + 1, x0] + integ2[y0, x0]
            mean = s / cnt
            var = s2 / cnt - mean * mean
            for ch in range(3):
                pv = float(img[y, x, ch])
                res = mean + (var - nvar) / var * (pv - mean) if var > nvar else mean
                out[y, x, ch] = round_byte(res)
            out[y, x, 3] = img[y, x, 3]
    return out


def dog_ref(img, sigma1, sigma2):
    """Difference of Gaussians on luma, mapped to [0,255] by +128."""
    def gblur(im, sigma):
        # Match millow gaussian_blur: ksize = clamp(ceil(sigma*3)*2+1, 3, 99).
        ks = max(3, min(99, (math.ceil(sigma * 3)) * 2 + 1))
        # Use Pillow's GaussianBlur which is close but not exact; instead
        # implement separable Gaussian with replicate border.
        c = ks // 2
        weights = np.zeros(ks)
        s = 0.0
        for i in range(ks):
            d = i - c
            v = math.exp(-(d * d) / (2.0 * sigma * sigma))
            weights[i] = v
            s += v
        weights /= s
        src = luma(im[..., 0], im[..., 1], im[..., 2]).astype(np.float64)
        h, w = src.shape
        # Horizontal pass.
        tmp = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                acc = 0.0
                for i in range(ks):
                    sx = clampi(x + i - c, 0, w - 1)
                    acc += src[y, sx] * weights[i]
                tmp[y, x] = acc
        # Vertical pass.
        out = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                acc = 0.0
                for i in range(ks):
                    sy = clampi(y + i - c, 0, h - 1)
                    acc += tmp[sy, x] * weights[i]
                out[y, x] = acc
        return out
    l1 = gblur(img, sigma1)
    l2 = gblur(img, sigma2)
    diff = (l1 - l2 + 128).astype(int)
    out = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            vb = int(max(0, min(255, int(diff[y, x]))))
            out[y, x, 0] = vb
            out[y, x, 1] = vb
            out[y, x, 2] = vb
            out[y, x, 3] = img[y, x, 3]
    return out


def distance_transform_ref(img, foreground_value=128):
    """Two-pass 3-4 chamfer distance, rescaled by 1/3."""
    h, w = img.shape[:2]
    fg = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            l = (int(img[y, x, 0]) * 77 + int(img[y, x, 1]) * 150 + int(img[y, x, 2]) * 29) >> 8
            fg[y, x] = l >= foreground_value
    large = float(h + w) * 3.0
    dist = np.where(fg, large, 0.0)
    # Forward pass.
    for y in range(h):
        for x in range(w):
            if not fg[y, x]:
                continue
            d = dist[y, x]
            if y > 0:
                up = dist[y - 1, x] + 3.0
                if up < d:
                    d = up
                if x > 0:
                    ul = dist[y - 1, x - 1] + 4.0
                    if ul < d:
                        d = ul
                if x < w - 1:
                    ur = dist[y - 1, x + 1] + 4.0
                    if ur < d:
                        d = ur
            if x > 0:
                left = dist[y, x - 1] + 3.0
                if left < d:
                    d = left
            dist[y, x] = d
    # Backward pass.
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if not fg[y, x]:
                continue
            d = dist[y, x]
            if y < h - 1:
                dn = dist[y + 1, x] + 3.0
                if dn < d:
                    d = dn
                if x < w - 1:
                    dr = dist[y + 1, x + 1] + 4.0
                    if dr < d:
                        d = dr
                if x > 0:
                    dl = dist[y + 1, x - 1] + 4.0
                    if dl < d:
                        d = dl
            if x < w - 1:
                right = dist[y, x + 1] + 3.0
                if right < d:
                    d = right
            dist[y, x] = d
    result = [[float(dist[y, x]) / 3.0 for x in range(w)] for y in range(h)]
    return result


def kmeans_quantize_ref(img, k, max_iter=20, seed=0):
    """K-means color quantization with deterministic seeded init (matches millow)."""
    h, w = img.shape[:2]
    n = h * w
    if n == 0 or k <= 0:
        return img.copy()
    pixels = img[..., :3].astype(np.float64).reshape(n, 3)
    if k == 1:
        mean = pixels.mean(axis=0)
        out = np.zeros_like(img)
        for i in range(n):
            y, x = i // w, i % w
            mr = round_byte(mean[0])
            mg = round_byte(mean[1])
            mb = round_byte(mean[2])
            out[y, x, 0] = mr
            out[y, x, 1] = mg
            out[y, x, 2] = mb
            out[y, x, 3] = img[y, x, 3]
        return out
    # Match millow's init: even spacing + LCG jitter.
    stride_n = max(1, n // k) if n >= k else 1
    centroids = np.zeros((k, 3))
    state = seed
    for i in range(k):
        # Match MoonBit's 32-bit signed integer overflow behavior.
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        if state >= 0x80000000:
            state -= 0x100000000
        base = i * stride_n
        offset = (state if state >= 0 else -state) % stride_n
        idx = base + offset
        if idx >= n:
            idx = n - 1
        centroids[i] = pixels[idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        changed = False
        # Assignment.
        for i in range(n):
            best, best_d = 0, -1.0
            for c in range(k):
                d = float(((pixels[i] - centroids[c]) ** 2).sum())
                if best_d < 0 or d < best_d:
                    best_d, best = d, c
            if labels[i] != best:
                labels[i] = best
                changed = True
        # Update.
        sums = np.zeros((k, 3))
        cnts = np.zeros(k, dtype=int)
        for i in range(n):
            c = labels[i]
            sums[c] += pixels[i]
            cnts[c] += 1
        for c in range(k):
            if cnts[c] > 0:
                centroids[c] = sums[c] / cnts[c]
        if not changed:
            break
    out = np.zeros_like(img)
    for i in range(n):
        y, x = i // w, i % w
        c = labels[i]
        out[y, x, 0] = round_byte(centroids[c, 0])
        out[y, x, 1] = round_byte(centroids[c, 1])
        out[y, x, 2] = round_byte(centroids[c, 2])
        out[y, x, 3] = img[y, x, 3]
    return out


def template_match_ref(img, template):
    """Normalized cross-correlation on luma. Returns 2D list of doubles."""
    ih, iw = img.shape[:2]
    th, tw = template.shape[:2]
    if th > ih or tw > iw or th == 0 or tw == 0:
        return []
    igray = luma(img[..., 0], img[..., 1], img[..., 2]).astype(np.float64)
    tgray = luma(template[..., 0], template[..., 1], template[..., 2]).astype(np.float64)
    tmean = float(tgray.mean())
    tnorm_sq = float(((tgray - tmean) ** 2).sum())
    out_h, out_w = ih - th + 1, iw - tw + 1
    result = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    if tnorm_sq == 0:
        return result
    for oy in range(out_h):
        for ox in range(out_w):
            patch = igray[oy:oy + th, ox:ox + tw]
            imean = float(patch.mean())
            cross = float(((patch - imean) * (tgray - tmean)).sum())
            inorm_sq = float(((patch - imean) ** 2).sum())
            denom = math.sqrt(inorm_sq * tnorm_sq)
            result[oy][ox] = cross / denom if denom > 0 else 0.0
    return result





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
    bordered = bordered_6x6()
    dots = dots_5x5()
    holed = holed_5x5()
    grad12_input = np.zeros((12, 12, 4), dtype=np.uint8)
    for y in range(12):
        for x in range(12):
            grad12_input[y, x] = ((x * 21) % 256, (y * 21) % 256, ((x + y) * 17) % 256, 255)

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
        img_to_moonbit("in_bordered_6x6", bordered),
        img_to_moonbit("in_dots_5x5", dots),
        img_to_moonbit("in_holed_5x5", holed),
        img_to_moonbit("in_lines_6x6", lines_6x6()),
        img_to_moonbit("in_grad_12x12", grad12_input),
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
        img_to_moonbit("exp_auto_crop_bordered", auto_crop(bordered, 240)),
        img_to_moonbit("exp_resize_model_input_8x8", resize_to_model_input_ref(grad, 8, 8)),
        img_to_moonbit("exp_perspective_identity", perspective_transform_ref(grad, [1, 0, 0, 0, 1, 0, 0, 0, 1], 4, 4)),
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
    sections.append("")
    sections.append(img_to_moonbit("exp_threshold_niblack_3", threshold_niblack(grad, 3, -0.2)))

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
        img_to_moonbit("exp_remove_small_objects_4", remove_small_objects(dots, 4)),
        img_to_moonbit("exp_remove_small_holes_1", remove_small_holes(holed, 1)),
        img_to_moonbit("exp_remove_lines_h", remove_lines_ref(lines_6x6(), 'horizontal', 1)),
        img_to_moonbit("exp_remove_lines_both", remove_lines_ref(lines_6x6(), 'both', 1)),
        "",
        "// ===== Bilateral filter =====",
        "",
        img_to_moonbit("exp_bilateral_5_01_10", bilateral_filter(grad, 5, 0.1, 10.0)),
        img_to_moonbit("exp_guided_filter_2_50", guided_filter_ref(grad, grad, 2, 50.0)),
        "",
        "// ===== Affine transform =====",
        "",
        img_to_moonbit("exp_rotate_any_45", rotate_any(grad, 45.0, "bilinear")),
        img_to_moonbit("exp_translate_1_1", translate(grad, 1.0, 1.0, "bilinear")),
        "",
        "// ===== Feature detection =====",
        "",
        arr2_to_moonbit("exp_corner_harris", corner_harris(grad, 0.04)),
        arr2_to_moonbit("exp_corner_shi_tomasi", corner_shi_tomasi(grad)),
        float_arr_to_moonbit("exp_hog", hog(grad, 2, 1, 9)),
        int_arr_to_moonbit("exp_lbp", lbp(grad, 1, 8)),
        int_arr_to_moonbit("exp_lbp_hist", lbp_histogram(grad, 1, 8)),
        "",
        "// ===== Measurement =====",
        "",
        int_arr2_to_moonbit("exp_connected_components", connected_components(dot)[0]),
        float_arr_to_moonbit("exp_moments", moments(grad)),
        float_arr_to_moonbit("exp_hu_moments", hu_moments(grad)),
        int_arr_to_moonbit("exp_proj_h_grad", project_horizontal(grad)),
        int_arr_to_moonbit("exp_proj_v_grad", project_vertical(grad)),
        int_arr_to_moonbit("exp_proj_h_binary_dot", project_horizontal_binary(dot)),
        int_arr_to_moonbit("exp_proj_v_binary_dot", project_vertical_binary(dot)),
        "",
    ])

    # Integral image and SSIM (float values, generated separately).
    sections.append("")
    sections.append(float_arr_to_moonbit("exp_integral_image_grad", integral_image_ref(grad)))
    sections.append("")
    # SSIM needs >= 11x11 images, so use the 12x12 gradient.
    blurred12 = gaussian_blur(grad12_input, 1.0)
    sections.append("///|")
    sections.append(f"let exp_ssim_grad_blur : Double = {ssim_ref(grad12_input, blurred12)}")
    sections.append("")

    # ===== New algorithm fixtures =====
    # Template for template_match: a 2x2 piece extracted from grad (top-left).
    tpl_2x2 = grad[0:2, 0:2].copy()
    sections.append("// ===== New algorithms =====")
    sections.append("")
    sections.append(img_to_moonbit("in_tpl_2x2", tpl_2x2))
    sections.append(img_to_moonbit("exp_bernsen_3_15", threshold_bernsen_ref(grad, 3, 15)))
    sections.append(img_to_moonbit("exp_wolf_3_neg02", threshold_wolf_ref(grad, 3, -0.2)))
    sections.append(img_to_moonbit("exp_adapt_mean_3_5", adaptive_threshold_mean_ref(grad, 3, 5.0)))
    sections.append(img_to_moonbit("exp_adapt_gauss_3_5", adaptive_threshold_gaussian_ref(grad, 3, 5.0)))
    sections.append(img_to_moonbit("exp_wiener_3_neg1", wiener_filter_ref(grad, 3, -1.0)))
    sections.append(img_to_moonbit("exp_dog_05_10", dog_ref(grad, 0.5, 1.0)))
    sections.append(img_to_moonbit("exp_kmeans_2", kmeans_quantize_ref(grad, 2, 20, 0)))
    sections.append(arr2_to_moonbit("exp_distance_dot", distance_transform_ref(dot)))
    sections.append(arr2_to_moonbit("exp_template_match_grad", template_match_ref(grad, tpl_2x2)))
    sections.append("")

    return "\n".join(sections)


if __name__ == "__main__":
    out = generate()
    out_path = Path(__file__).parent / "fixtures_test.mbt"
    out_path.write_text(out, encoding="utf-8")
    print(f"Generated {out_path} ({len(out)} bytes)")