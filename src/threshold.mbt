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
