///|
/// BT.601 luminance of an RGB triple as an integer in `[0, 255]`.
fn luma(r : Byte, g : Byte, b : Byte) -> Int {
  (r.to_int() * 77 + g.to_int() * 150 + b.to_int() * 29) >> 8
}

///|
/// 256-bin luminance histogram.
pub fn histogram(img : Image) -> Array[Int] {
  let h = Array::make(256, 0)
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let v = luma(img.data[o], img.data[o + 1], img.data[o + 2])
    h[v] = h[v] + 1
  }
  h
}

///|
/// Per-channel histograms for R, G, B (each 256 bins).
pub fn histogram_color(img : Image) -> Array[Array[Int]] {
  let hr = Array::make(256, 0)
  let hg = Array::make(256, 0)
  let hb = Array::make(256, 0)
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let r = img.data[o].to_int()
    let g = img.data[o + 1].to_int()
    let b = img.data[o + 2].to_int()
    hr[r] = hr[r] + 1
    hg[g] = hg[g] + 1
    hb[b] = hb[b] + 1
  }
  [hr, hg, hb]
}

///|
/// Histogram equalization based on the luminance CDF, applied to RGB.
pub fn equalize_histogram(img : Image) -> Image {
  let hist = histogram(img)
  let total = img.h * img.w
  if total == 0 {
    return img.clone()
  }
  // Find first non-zero bin for CDF normalization.
  let mut cdf_min = 0
  let mut found = false
  let cdf = Array::make(256, 0)
  let mut acc = 0
  for i = 0; i < 256; i = i + 1 {
    acc = acc + hist[i]
    cdf[i] = acc
    if !found && hist[i] != 0 {
      cdf_min = acc
      found = true
    }
  }
  let denom = total - cdf_min
  let lut = Array::makei(256, fn(i) {
    if denom <= 0 {
      i.to_byte()
    } else {
      round_byte((cdf[i] - cdf_min).to_double() * 255.0 / denom.to_double())
    }
  })
  apply_rgb_lut(img, lut)
}
