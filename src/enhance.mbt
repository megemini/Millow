///|
/// Apply a 256-entry lookup table to the RGB channels; alpha is preserved.
fn apply_rgb_lut(img : Image, lut : Array[Byte]) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    (lut[r.to_int()], lut[g.to_int()], lut[b.to_int()], a)
  })
}

///|
/// Additive brightness adjustment (per-channel, clamped).
pub fn adjust_brightness(img : Image, delta : Int) -> Image {
  apply_rgb_lut(img, Array::makei(256, fn(i) { clamp_byte(i + delta) }))
}

///|
/// Contrast adjustment about mid-gray 128. `factor > 1` increases contrast.
pub fn adjust_contrast(img : Image, factor : Double) -> Image {
  apply_rgb_lut(
    img,
    Array::makei(256, fn(i) {
      round_byte((i.to_double() - 128.0) * factor + 128.0)
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
