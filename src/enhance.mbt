///|
/// Apply a 256-entry lookup table to the RGB channels; alpha is preserved.
fn apply_rgb_lut(img : Image, lut : Array[Byte]) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    (lut[r.to_int()], lut[g.to_int()], lut[b.to_int()], a)
  })
}

///|
/// Brightness adjustment using a multiplicative factor.
/// `factor = 1.0` returns the original image, `factor = 0.0` returns a black image.
pub fn adjust_brightness(img : Image, factor : Double) -> Image {
  apply_rgb_lut(
    img,
    Array::makei(256, fn(i) { round_byte(i.to_double() * factor) }),
  )
}

///|
/// Contrast adjustment based on image mean.
/// `factor = 1.0` returns the original image, `factor = 0.0` returns a solid gray image.
pub fn adjust_contrast(img : Image, factor : Double) -> Image {
  let mut sum = 0
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    sum = sum + luma(img.data[o], img.data[o + 1], img.data[o + 2])
  }
  let mean = sum.to_double() / n.to_double()
  apply_rgb_lut(
    img,
    Array::makei(256, fn(i) {
      round_byte(mean * (1.0 - factor) + i.to_double() * factor)
    }),
  )
}

///|
/// Gamma correction. `gamma > 1` darkens, `gamma < 1` brightens.
pub fn adjust_gamma(img : Image, gamma : Double) -> Image {
  apply_rgb_lut(
    img,
    Array::makei(256, fn(i) {
      round_byte(@math.pow(i.to_double() / 255.0, gamma) * 255.0)
    }),
  )
}

///|
/// Linearly rescale the RGB value range into `[min, max]`.
pub fn normalize(img : Image, min : Byte, max : Byte) -> Image {
  let mut lo = 255
  let mut hi = 0
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    for c = 0; c < 3; c = c + 1 {
      let v = img.data[o + c].to_int()
      if v < lo {
        lo = v
      }
      if v > hi {
        hi = v
      }
    }
  }
  if hi <= lo {
    return img.clone()
  }
  let mn = min.to_int()
  let mx = max.to_int()
  let span = (mx - mn).to_double() / (hi - lo).to_double()
  apply_rgb_lut(
    img,
    Array::makei(256, fn(i) {
      clamp_byte(mn + round_byte((i - lo).to_double() * span).to_int())
    }),
  )
}

///|
/// Automatic contrast stretch, ignoring `cutoff` percent of pixels per end.
pub fn auto_contrast(img : Image, cutoff : Int) -> Image {
  let hist = histogram(img)
  let total = img.h * img.w
  if total == 0 {
    return img.clone()
  }
  let cut = total * clampi(cutoff, 0, 49) / 100
  let mut lo = 0
  let mut acc = 0
  for i = 0; i < 256; i = i + 1 {
    acc = acc + hist[i]
    if acc > cut {
      lo = i
      break
    }
  }
  let mut hi = 255
  acc = 0
  for i = 255; i >= 0; i = i - 1 {
    acc = acc + hist[i]
    if acc > cut {
      hi = i
      break
    }
  }
  if hi <= lo {
    return img.clone()
  }
  let span = 255.0 / (hi - lo).to_double()
  apply_rgb_lut(
    img,
    Array::makei(256, fn(i) { round_byte((i - lo).to_double() * span) }),
  )
}

///|
/// Standardize the image to zero mean and unit standard deviation.
/// Returns a 2D array of `Float` values (one per pixel, based on luma).
pub fn standardize(img : Image) -> Array[Array[Double]] {
  let n = img.h * img.w
  let mut sum = 0.0
  let mut sum2 = 0.0
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let v = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
    sum = sum + v
    sum2 = sum2 + v * v
  }
  let mean = sum / n.to_double()
  let variance = sum2 / n.to_double() - mean * mean
  let std = if variance > 0.0 { variance.sqrt() } else { 1.0 }
  let out = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o = img.offset(y, x)
      let v = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
      out[y][x] = (v - mean) / std
    }
  }
  out
}
