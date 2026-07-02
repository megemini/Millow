///|
/// Binarize by luminance: pixels with luma >= `thresh` become white.
pub fn threshold(img : Image, thresh : Byte) -> Image {
  let t = thresh.to_int()
  map_rgba(img, fn(r, g, b, a) {
    let vb : Byte = if luma(r, g, b) >= t { 255 } else { 0 }
    (vb, vb, vb, a)
  })
}

///|
/// Inverse binarization: pixels with luma >= `thresh` become black.
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
