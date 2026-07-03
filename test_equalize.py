#!/usr/bin/env python3
"""Debug equalize_histogram alignment."""

import numpy as np
from skimage import exposure
import math

def round_byte(x):
    """Round half away from zero, then clamp to [0, 255]."""
    r = int(math.copysign(math.floor(abs(x) + 0.5), x))
    return max(0, min(255, r))

# Create 4x4 gradient: R = x*60, G = y*50, B = 100, A = 255
img = np.zeros((4, 4, 4), dtype=np.uint8)
for y in range(4):
    for x in range(4):
        img[y, x] = (x * 60, y * 50, 100, 255)

print("Input image (4x4):")
for y in range(4):
    for x in range(4):
        print(f"  [{y},{x}] = {list(img[y,x])}")

# skimage equalize_hist
result_sk = img.copy()
for c in range(3):
    channel = img[..., c]
    equalized = exposure.equalize_hist(channel)
    result_sk[..., c] = (equalized * 255).astype(np.uint8)

print("\nskimage equalize_hist:")
for y in range(4):
    for x in range(4):
        print(f"  [{y},{x}] = {list(result_sk[y,x])}")

# millow-style equalization
def millow_equalize(img):
    h, w = img.shape[:2]
    total = h * w
    out = img.copy()
    for c in range(3):
        hist = [0] * 256
        for i in range(total):
            v = int(img[i // w, i % w, c])
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
            y_idx = i // w
            x_idx = i % w
            out[y_idx, x_idx, c] = lut[int(img[y_idx, x_idx, c])]
    return out

result_millow = millow_equalize(img)
print("\nmillow-style equalize:")
for y in range(4):
    for x in range(4):
        print(f"  [{y},{x}] = {list(result_millow[y,x])}")

print("\nDifference (skimage - millow):")
for y in range(4):
    for x in range(4):
        diff = result_sk[y,x].astype(int) - result_millow[y,x].astype(int)
        print(f"  [{y},{x}] = {list(diff)}")
