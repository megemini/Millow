///|
/// Append the ASCII bytes of `s` to `out`.
fn push_ascii(out : Array[Byte], s : String) -> Unit {
  for c in s.iter() {
    out.push(c.to_int().to_byte())
  }
}

///|
/// Encode as a binary PPM (P6) byte stream.
pub fn to_ppm(img : Image) -> Array[Byte] {
  let out : Array[Byte] = []
  push_ascii(out, "P6\n\{img.w} \{img.h}\n255\n")
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    out.push(img.data[o])
    out.push(img.data[o + 1])
    out.push(img.data[o + 2])
  }
  out
}

///|
/// Encode as a binary PGM (P5) grayscale byte stream.
pub fn to_pgm(img : Image) -> Array[Byte] {
  let out : Array[Byte] = []
  push_ascii(out, "P5\n\{img.w} \{img.h}\n255\n")
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    out.push(luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_byte())
  }
  out
}

///|
/// RGB-only byte buffer (alpha dropped), length `h * w * 3`.
pub fn to_display_bytes(img : Image) -> Array[Byte] {
  let n = img.h * img.w
  let out = Array::make(n * 3, (0 : Byte))
  for i = 0; i < n; i = i + 1 {
    out[i * 3] = img.data[i * 4]
    out[i * 3 + 1] = img.data[i * 4 + 1]
    out[i * 3 + 2] = img.data[i * 4 + 2]
  }
  out
}

///|
/// Human-readable summary of the image.
pub fn info(img : Image) -> String {
  "Image \{img.w}x\{img.h}, \{img.len()} bytes"
}

///|
/// Pixel-exact equality (same dimensions and bytes).
pub fn equals(a : Image, b : Image) -> Bool {
  if a.h != b.h || a.w != b.w {
    return false
  }
  for i = 0; i < a.data.length(); i = i + 1 {
    if a.data[i] != b.data[i] {
      return false
    }
  }
  true
}

///|
/// Mean squared error over all RGBA channels; `-1.0` on size mismatch.
pub fn mse(a : Image, b : Image) -> Double {
  if a.h != b.h || a.w != b.w || a.is_empty() {
    return -1.0
  }
  let mut acc = 0.0
  let len = a.data.length()
  for i = 0; i < len; i = i + 1 {
    let d = a.data[i].to_double() - b.data[i].to_double()
    acc = acc + d * d
  }
  acc / len.to_double()
}

///|
/// Peak signal-to-noise ratio in dB. Returns a large value for identical input.
pub fn psnr(a : Image, b : Image) -> Double {
  let m = mse(a, b)
  if m <= 0.0 {
    return 99.0
  }
  10.0 * @math.log10(255.0 * 255.0 / m)
}

///|
/// Downscale to fit within `max_h x max_w` (nearest-neighbor).
pub fn thumbnail(img : Image, max_h : Int, max_w : Int) -> Image {
  resize_to_fit(img, max_h, max_w, Nearest)
}

///|
/// Structural Similarity Index (SSIM) between two images.
///
/// Uses local windowed statistics (Wang et al. 2004) with an 11×11 uniform
/// window. Integral images provide O(1) per-window computation, giving O(n)
/// total complexity. Returns a value in `[-1, 1]` where 1 means identical.
pub fn ssim(a : Image, b : Image) -> Double {
  if a.h != b.h || a.w != b.w || a.is_empty() {
    return -1.0
  }
  let h = a.h
  let w = a.w
  let c1 = 0.01 * 255.0 * (0.01 * 255.0)
  let c2 = 0.03 * 255.0 * (0.03 * 255.0)
  let r = 5 // (11-1)/2
  let w1 = w + 1
  let stride = w * 4
  // Build integral images directly from pixel data.
  // integ_a: sum of luma_a, integ_b: sum of luma_b
  // integ_a2: sum of luma_a^2, integ_b2: sum of luma_b^2
  // integ_ab: sum of luma_a * luma_b
  let integ_a = Array::make((h + 1) * w1, 0.0)
  let integ_b = Array::make((h + 1) * w1, 0.0)
  let integ_a2 = Array::make((h + 1) * w1, 0.0)
  let integ_b2 = Array::make((h + 1) * w1, 0.0)
  let integ_ab = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut ra = 0.0
    let mut rb = 0.0
    let mut ra2 = 0.0
    let mut rb2 = 0.0
    let mut rab = 0.0
    let row = y * stride
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      let va = luma(a.data[o], a.data[o + 1], a.data[o + 2]).to_double()
      let vb = luma(b.data[o], b.data[o + 1], b.data[o + 2]).to_double()
      ra = ra + va
      rb = rb + vb
      ra2 = ra2 + va * va
      rb2 = rb2 + vb * vb
      rab = rab + va * vb
      let idx = (y + 1) * w1 + (x + 1)
      integ_a[idx] = integ_a[y * w1 + (x + 1)] + ra
      integ_b[idx] = integ_b[y * w1 + (x + 1)] + rb
      integ_a2[idx] = integ_a2[y * w1 + (x + 1)] + ra2
      integ_b2[idx] = integ_b2[y * w1 + (x + 1)] + rb2
      integ_ab[idx] = integ_ab[y * w1 + (x + 1)] + rab
    }
  }
  // Compute per-window SSIM and average.
  let mut total = 0.0
  let mut count = 0
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let n = ((y1 - y0 + 1) * (x1 - x0 + 1)).to_double()
      let sa = query_integral(integ_a, w1, y0, x0, y1, x1)
      let sb = query_integral(integ_b, w1, y0, x0, y1, x1)
      let sa2 = query_integral(integ_a2, w1, y0, x0, y1, x1)
      let sb2 = query_integral(integ_b2, w1, y0, x0, y1, x1)
      let sab = query_integral(integ_ab, w1, y0, x0, y1, x1)
      let mean_a = sa / n
      let mean_b = sb / n
      let var_a = sa2 / n - mean_a * mean_a
      let var_b = sb2 / n - mean_b * mean_b
      let cov_ab = sab / n - mean_a * mean_b
      let num = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2)
      let den = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
      total = total + num / den
      count = count + 1
    }
  }
  if count == 0 {
    return -1.0
  }
  total / count.to_double()
}
