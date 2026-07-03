#!/usr/bin/env python3
"""Debug resize_bilinear alignment."""

import numpy as np
from PIL import Image
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

# Pillow resize
pil = Image.fromarray(img, mode='RGBA')
resized = pil.resize((2, 2), resample=Image.BILINEAR)
result = np.array(resized)
print("\nPillow resize 4x4 -> 2x2:")
for y in range(2):
    for x in range(2):
        print(f"  [{y},{x}] = {list(result[y,x])}")

# Simulate millow's separable bilinear
def bilinear_filter(x):
    ax = abs(x)
    return 1.0 - ax if ax < 1.0 else 0.0

def precompute_weights(src_size, dst_size):
    scale = src_size / dst_size
    support = max(1.0, scale)
    inv_fs = 1.0 / support
    weights_list = []
    for i in range(dst_size):
        center = (i + 0.5) * scale
        xmin = max(0, int(math.floor(center - support + 0.5)))
        xmax = min(src_size, int(math.ceil(center + support + 0.5)))
        n = xmax - xmin
        weights = []
        ww = 0.0
        for j in range(n):
            w = bilinear_filter(((xmin + j) - center + 0.5) * inv_fs)
            weights.append(w)
            ww += w
        if ww > 0.0:
            weights = [w / ww for w in weights]
        weights_list.append((weights, xmin, xmax))
    return weights_list

# Horizontal pass
x_weights = precompute_weights(4, 2)
print("\nX weights:")
for dx, (wts, xmin, xmax) in enumerate(x_weights):
    print(f"  dx={dx}: xmin={xmin}, xmax={xmax}, weights={wts}")

temp = np.zeros((4, 2, 4), dtype=float)
for y in range(4):
    for dx in range(2):
        wts, xmin, xmax = x_weights[dx]
        for c in range(4):
            acc = 0.0
            for j in range(xmax - xmin):
                sx = xmin + j
                acc += wts[j] * img[y, sx, c]
            temp[y, dx, c] = acc

print("\nAfter horizontal pass (4x2):")
for y in range(4):
    for x in range(2):
        print(f"  [{y},{x}] = {list(temp[y,x])}")

# Vertical pass
y_weights = precompute_weights(4, 2)
print("\nY weights:")
for dy, (wts, ymin, ymax) in enumerate(y_weights):
    print(f"  dy={dy}: ymin={ymin}, ymax={ymax}, weights={wts}")

final = np.zeros((2, 2, 4), dtype=np.uint8)
for dy in range(2):
    wts, ymin, ymax = y_weights[dy]
    for x in range(2):
        for c in range(4):
            acc = 0.0
            for j in range(ymax - ymin):
                sy = ymin + j
                acc += wts[j] * temp[sy, x, c]
            final[dy, x, c] = round_byte(acc)

print("\nMillow simulation resize 4x4 -> 2x2:")
for y in range(2):
    for x in range(2):
        print(f"  [{y},{x}] = {list(final[y,x])}")

print("\nDifference (Pillow - Millow):")
for y in range(2):
    for x in range(2):
        diff = result[y,x].astype(int) - final[y,x].astype(int)
        print(f"  [{y},{x}] = {list(diff)}")
