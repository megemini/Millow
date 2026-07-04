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
pub fn threshold_sauvola(img : Image, window_size : Int, k : Double) -> Image {
  let r = window_size / 2
  let gray = to_grayscale(img)
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let mut sum = 0.0
      let mut sum2 = 0.0
      let mut count = 0
      for dy = -r; dy <= r; dy = dy + 1 {
        for dx = -r; dx <= r; dx = dx + 1 {
          let cy = clampi(y + dy, 0, img.h - 1)
          let cx = clampi(x + dx, 0, img.w - 1)
          let v = gray.data[gray.offset(cy, cx)].to_double()
          sum = sum + v
          sum2 = sum2 + v * v
          count = count + 1
        }
      }
      let mean = sum / count.to_double()
      let variance = sum2 / count.to_double() - mean * mean
      let std = if variance > 0.0 { variance.sqrt() } else { 0.0 }
      let t = mean * (1.0 - k * (1.0 - std / 128.0))
      let pixel_luma = gray.data[gray.offset(y, x)].to_int()
      let vb : Byte = if pixel_luma.to_double() >= t { 255 } else { 0 }
      let o = out.offset(y, x)
      out.data[o] = vb
      out.data[o + 1] = vb
      out.data[o + 2] = vb
      out.data[o + 3] = img.data[img.offset(y, x) + 3]
    }
  }
  out
}
