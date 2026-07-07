///|
/// Binarize by luminance: pixels with luma >= `thresh` become white.
///
/// Each pixel's BT.601 luma is compared against `thresh`; foreground pixels
/// are set to white `(255, 255, 255)` and background pixels to black. The
/// alpha channel is preserved.
pub fn threshold(img : Image, thresh : Byte) -> Image {
  let t = thresh.to_int()
  map_rgba(img, fn(r, g, b, a) {
    let vb : Byte = if luma(r, g, b) >= t { 255 } else { 0 }
    (vb, vb, vb, a)
  })
}

///|
/// Inverse binarization: pixels with luma >= `thresh` become black.
///
/// The inverse of `threshold`: foreground pixels (luma >= `thresh`) are set
/// to black and background pixels to white. The alpha channel is preserved.
pub fn threshold_inv(img : Image, thresh : Byte) -> Image {
  let t = thresh.to_int()
  map_rgba(img, fn(r, g, b, a) {
    let vb : Byte = if luma(r, g, b) >= t { 0 } else { 255 }
    (vb, vb, vb, a)
  })
}

///|
/// Compute an Otsu threshold from a 256-bin histogram.
fn otsu(hist : Array[Int], total : Int) -> Int {
  let mut sum = 0.0
  for i = 0; i < 256; i = i + 1 {
    sum = sum + i.to_double() * hist[i].to_double()
  }
  let mut sum_b = 0.0
  let mut w_b = 0
  let mut max_var = -1.0
  let mut thr = 0
  for t = 0; t < 256; t = t + 1 {
    w_b = w_b + hist[t]
    if w_b == 0 {
      continue
    }
    let w_f = total - w_b
    if w_f == 0 {
      break
    }
    sum_b = sum_b + t.to_double() * hist[t].to_double()
    let m_b = sum_b / w_b.to_double()
    let m_f = (sum - sum_b) / w_f.to_double()
    let diff = m_b - m_f
    let v = w_b.to_double() * w_f.to_double() * diff * diff
    if v > max_var {
      max_var = v
      thr = t
    }
  }
  thr
}

///|
/// Otsu adaptive binarization. Returns the chosen threshold and the result.
///
/// Selects the threshold that maximizes the inter-class variance of the
/// luminance histogram and applies `threshold`. The returned `Byte` is the
/// applied cut (one above the background class) so values `>=` it are
/// foreground.
pub fn threshold_otsu(img : Image) -> (Byte, Image) {
  let total = img.h * img.w
  if total == 0 {
    return (0, img.clone())
  }
  // `otsu` returns the last index of the background class; `threshold` treats
  // values `>= thresh` as foreground, so the applied cut is `t + 1`.
  let applied = clampi(otsu(histogram(img), total) + 1, 0, 255).to_byte()
  (applied, threshold(img, applied))
}

///|
/// Sauvola local thresholding. Computes a threshold per pixel based on the
/// local mean and standard deviation within a `window_size × window_size`
/// window. Suitable for images with uneven illumination.
///
/// The per-pixel threshold is `mean * (1 + k * (std / 128 - 1))`. Border
/// coordinates are clamped (replicate). `window_size` should be odd. Returns a
/// binary image; alpha is preserved.
///
/// Uses integral images for O(1) window statistics (O(n) total instead of
/// O(n * window_size²)).
pub fn threshold_sauvola(img : Image, window_size : Int, k : Double) -> Image {
  let r = window_size / 2
  let gray = to_grayscale(img)
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  // Build integral images: I[y][x] = sum of gray[0..y-1, 0..x-1]
  // I2[y][x] = sum of gray²[0..y-1, 0..x-1]
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  let integ2 = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let mut row_sum2 = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let v = gray.data[gray_row + x * 4].to_double()
      row_sum = row_sum + v
      row_sum2 = row_sum2 + v * v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
      integ2[(y + 1) * w1 + (x + 1)] = integ2[y * w1 + (x + 1)] + row_sum2
    }
  }
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let count = (y1 - y0 + 1) * (x1 - x0 + 1)
      let cnt = count.to_double()
      let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
        integ[y0 * w1 + (x1 + 1)] -
        integ[(y1 + 1) * w1 + x0] +
        integ[y0 * w1 + x0]
      let s2 = integ2[(y1 + 1) * w1 + (x1 + 1)] -
        integ2[y0 * w1 + (x1 + 1)] -
        integ2[(y1 + 1) * w1 + x0] +
        integ2[y0 * w1 + x0]
      let mean = s / cnt
      let variance = s2 / cnt - mean * mean
      let std = if variance > 0.0 { variance.sqrt() } else { 0.0 }
      let t = mean * (1.0 - k * (1.0 - std / 128.0))
      let pixel_luma = gray.data[out_base + x * 4].to_double()
      let vb : Byte = if pixel_luma >= t { 255 } else { 0 }
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Niblack local thresholding. Computes a threshold per pixel based on the
/// local mean and standard deviation within a `window_size × window_size`
/// window: `T = mean + k * std`.
///
/// Suitable for images with uneven illumination where text is small. Typically
/// `k = -0.2` for dark text on light background. `window_size` should be odd.
/// Returns a binary image; alpha is preserved.
///
/// Uses integral images for O(1) window statistics (O(n) total), sharing the
/// same infrastructure as `threshold_sauvola`.
pub fn threshold_niblack(img : Image, window_size : Int, k : Double) -> Image {
  let r = window_size / 2
  let gray = to_grayscale(img)
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  let integ2 = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let mut row_sum2 = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let v = gray.data[gray_row + x * 4].to_double()
      row_sum = row_sum + v
      row_sum2 = row_sum2 + v * v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
      integ2[(y + 1) * w1 + (x + 1)] = integ2[y * w1 + (x + 1)] + row_sum2
    }
  }
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let count = (y1 - y0 + 1) * (x1 - x0 + 1)
      let cnt = count.to_double()
      let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
        integ[y0 * w1 + (x1 + 1)] -
        integ[(y1 + 1) * w1 + x0] +
        integ[y0 * w1 + x0]
      let s2 = integ2[(y1 + 1) * w1 + (x1 + 1)] -
        integ2[y0 * w1 + (x1 + 1)] -
        integ2[(y1 + 1) * w1 + x0] +
        integ2[y0 * w1 + x0]
      let mean = s / cnt
      let variance = s2 / cnt - mean * mean
      let std = if variance > 0.0 { variance.sqrt() } else { 0.0 }
      let t = mean + k * std
      let pixel_luma = gray.data[out_base + x * 4].to_double()
      let vb : Byte = if pixel_luma >= t { 255 } else { 0 }
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Bernsen local thresholding. Computes a threshold per pixel as the midpoint
/// of the local minimum and maximum within a `window_size × window_size`
/// window: `T = (min + max) / 2`.
///
/// Low-contrast windows (where `max - min < contrast_limit`) are classified as
/// background and forced to white. `window_size` should be odd. Returns a binary
/// image; alpha is preserved.
///
/// Suitable for images with uneven illumination where both text and background
/// have locally bimodal distributions. Brute-force O(n * window_size²).
pub fn threshold_bernsen(
  img : Image,
  window_size : Int,
  contrast_limit : Int,
) -> Image {
  let r = window_size / 2
  let gray = to_grayscale(img)
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let mut vmin = 255
      let mut vmax = 0
      for wy = y0; wy <= y1; wy = wy + 1 {
        let row_base = wy * stride
        for wx = x0; wx <= x1; wx = wx + 1 {
          let v = gray.data[row_base + wx * 4].to_int()
          if v < vmin {
            vmin = v
          }
          if v > vmax {
            vmax = v
          }
        }
      }
      let pixel_luma = gray.data[out_base + x * 4].to_int()
      let vb : Byte = if vmax - vmin < contrast_limit {
        255
      } else if pixel_luma >= (vmin + vmax) / 2 {
        255
      } else {
        0
      }
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Wolf local thresholding. A normalized variant of Niblack that rescales the
/// local standard deviation by the global range, making the threshold more
/// robust to varying contrast across the image.
///
/// Per-pixel threshold:
/// `T = mean + k * (std / std_max) * (mean - min_global)`
///
/// where `min_global` is the global minimum luma and `std_max` is the maximum
/// local standard deviation over all windows. `window_size` should be odd.
/// Returns a binary image; alpha is preserved.
///
/// Uses integral images for O(1) window statistics (O(n) total).
pub fn threshold_wolf(img : Image, window_size : Int, k : Double) -> Image {
  let r = window_size / 2
  let gray = to_grayscale(img)
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  let integ2 = Array::make((h + 1) * w1, 0.0)
  let mut min_global = 255.0
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let mut row_sum2 = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let v = gray.data[gray_row + x * 4].to_double()
      if v < min_global {
        min_global = v
      }
      row_sum = row_sum + v
      row_sum2 = row_sum2 + v * v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
      integ2[(y + 1) * w1 + (x + 1)] = integ2[y * w1 + (x + 1)] + row_sum2
    }
  }
  let mut std_max = 0.0
  let stds = Array::make(h * w, 0.0)
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let count = (y1 - y0 + 1) * (x1 - x0 + 1)
      let cnt = count.to_double()
      let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
        integ[y0 * w1 + (x1 + 1)] -
        integ[(y1 + 1) * w1 + x0] +
        integ[y0 * w1 + x0]
      let s2 = integ2[(y1 + 1) * w1 + (x1 + 1)] -
        integ2[y0 * w1 + (x1 + 1)] -
        integ2[(y1 + 1) * w1 + x0] +
        integ2[y0 * w1 + x0]
      let mean = s / cnt
      let variance = s2 / cnt - mean * mean
      let std = if variance > 0.0 { variance.sqrt() } else { 0.0 }
      stds[y * w + x] = std
      if std > std_max {
        std_max = std
      }
    }
  }
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let count = (y1 - y0 + 1) * (x1 - x0 + 1)
      let cnt = count.to_double()
      let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
        integ[y0 * w1 + (x1 + 1)] -
        integ[(y1 + 1) * w1 + x0] +
        integ[y0 * w1 + x0]
      let mean = s / cnt
      let std = stds[y * w + x]
      let norm = if std_max > 0.0 { std / std_max } else { 0.0 }
      let t = mean + k * norm * (mean - min_global)
      let pixel_luma = gray.data[out_base + x * 4].to_double()
      let vb : Byte = if pixel_luma >= t { 255 } else { 0 }
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Adaptive thresholding using the local mean within a `block_size × block_size`
/// window. The per-pixel threshold is `T = mean(window) - C`.
///
/// Pixels with luma `> T` become white (foreground); otherwise black. This
/// matches OpenCV's `ADAPTIVE_THRESH_MEAN_C` with `THRESH_BINARY`. Set `C` to a
/// small positive value (e.g. 5) to bias toward background. `block_size`
/// should be odd.
///
/// Uses integral images for O(1) window mean (O(n) total).
pub fn adaptive_threshold_mean(
  img : Image,
  block_size : Int,
  c : Double,
) -> Image {
  let r = block_size / 2
  let gray = to_grayscale(img)
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let v = gray.data[gray_row + x * 4].to_double()
      row_sum = row_sum + v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
    }
  }
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let count = (y1 - y0 + 1) * (x1 - x0 + 1)
      let cnt = count.to_double()
      let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
        integ[y0 * w1 + (x1 + 1)] -
        integ[(y1 + 1) * w1 + x0] +
        integ[y0 * w1 + x0]
      let t = s / cnt - c
      let pixel_luma = gray.data[out_base + x * 4].to_double()
      let vb : Byte = if pixel_luma > t { 255 } else { 0 }
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}

///|
/// Adaptive thresholding using a Gaussian-weighted local mean within a
/// `block_size × block_size` window. The per-pixel threshold is
/// `T = gaussian_weighted_mean(window) - C`.
///
/// Pixels with luma `> T` become white (foreground); otherwise black. This
/// matches OpenCV's `ADAPTIVE_THRESH_GAUSSIAN_C` with `THRESH_BINARY`. The
/// Gaussian sigma defaults to `0.3 * (block_size/2 - 1) + 0.8` (OpenCV's
/// `getGaussianKernel` default). `block_size` should be odd.
pub fn adaptive_threshold_gaussian(
  img : Image,
  block_size : Int,
  c : Double,
) -> Image {
  let r = block_size / 2
  let gray = to_grayscale(img)
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  let sigma = 0.3 * (r.to_double() - 1.0) + 0.8
  let kernel = Array::make(block_size * block_size, 0.0)
  let mut ksum = 0.0
  for dy = -r; dy <= r; dy = dy + 1 {
    for dx = -r; dx <= r; dx = dx + 1 {
      let v = @math.exp(
        -(dy * dy + dx * dx).to_double() / (2.0 * sigma * sigma),
      )
      kernel[(dy + r) * block_size + (dx + r)] = v
      ksum = ksum + v
    }
  }
  for i = 0; i < block_size * block_size; i = i + 1 {
    kernel[i] = kernel[i] / ksum
  }
  for y = 0; y < h; y = y + 1 {
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let mut acc = 0.0
      for dy = -r; dy <= r; dy = dy + 1 {
        let sy = clampi(y + dy, 0, h - 1)
        for dx = -r; dx <= r; dx = dx + 1 {
          let sx = clampi(x + dx, 0, w - 1)
          let kw = kernel[(dy + r) * block_size + (dx + r)]
          acc = acc + gray.data[sy * stride + sx * 4].to_double() * kw
        }
      }
      let t = acc - c
      let pixel_luma = gray.data[out_base + x * 4].to_double()
      let vb : Byte = if pixel_luma > t { 255 } else { 0 }
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}
