///|
/// Apply a per-pixel RGBA transform, producing a new same-size image.
fn map_rgba(
  img : Image,
  f : (Byte, Byte, Byte, Byte) -> (Byte, Byte, Byte, Byte),
) -> Image {
  let data = Array::make(img.data.length(), (0 : Byte))
  let n = img.h * img.w
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let (r, g, b, a) = f(
      img.data[o],
      img.data[o + 1],
      img.data[o + 2],
      img.data[o + 3],
    )
    data[o] = r
    data[o + 1] = g
    data[o + 2] = b
    data[o + 3] = a
  }
  { data, h: img.h, w: img.w }
}

///|
/// ITU-R BT.601 grayscale: `(R*77 + G*150 + B*29) >> 8`, alpha preserved.
pub fn to_grayscale(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    let y = (r.to_int() * 77 + g.to_int() * 150 + b.to_int() * 29) >> 8
    let yb = clamp_byte(y)
    (yb, yb, yb, a)
  })
}

///|
/// Grayscale with custom linear weights, alpha preserved.
///
/// - `r_w`, `g_w`, `b_w`: per-channel weights (typically summing to 1.0).
pub fn to_grayscale_weighted(
  img : Image,
  r_w : Double,
  g_w : Double,
  b_w : Double,
) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    let y = r_w * r.to_double() + g_w * g.to_double() + b_w * b.to_double()
    let yb = round_byte(y)
    (yb, yb, yb, a)
  })
}

///|
/// Swap the red and blue channels.
pub fn to_bgr(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, a) { (b, g, r, a) })
}

///|
/// Invert the RGB channels (`255 - value`), alpha preserved.
pub fn invert(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, a) {
    (
      clamp_byte(255 - r.to_int()),
      clamp_byte(255 - g.to_int()),
      clamp_byte(255 - b.to_int()),
      a,
    )
  })
}

///|
/// Multiplicative color tint: each channel scaled by `color / 255`.
///
/// - `r`, `g`, `b`: tint color; `(255, 255, 255)` leaves the image unchanged.
pub fn tint(img : Image, r : Byte, g : Byte, b : Byte) -> Image {
  map_rgba(img, fn(pr, pg, pb, a) {
    (
      clamp_byte(pr.to_int() * r.to_int() / 255),
      clamp_byte(pg.to_int() * g.to_int() / 255),
      clamp_byte(pb.to_int() * b.to_int() / 255),
      a,
    )
  })
}

///|
/// Composite the image over a solid background color, dropping transparency.
///
/// - `r`, `g`, `b`: the solid background color; output alpha is forced to 255.
pub fn flatten_alpha(img : Image, r : Byte, g : Byte, b : Byte) -> Image {
  map_rgba(img, fn(pr, pg, pb, a) {
    let alpha = a.to_double() / 255.0
    (
      round_byte(pr.to_double() * alpha + r.to_double() * (1.0 - alpha)),
      round_byte(pg.to_double() * alpha + g.to_double() * (1.0 - alpha)),
      round_byte(pb.to_double() * alpha + b.to_double() * (1.0 - alpha)),
      255,
    )
  })
}

///|
/// Alpha "over" compositing of `src` onto `dst`; result matches `dst` size.
/// Source pixels outside its bounds are treated as fully transparent.
pub fn composite_over(src : Image, dst : Image) -> Image {
  let out = Image::new(dst.h, dst.w)
  for y = 0; y < dst.h; y = y + 1 {
    for x = 0; x < dst.w; x = x + 1 {
      let doff = dst.offset(y, x)
      let da = dst.data[doff + 3].to_double() / 255.0
      let (sr, sg, sb, sa) = match src.pixel_at(y, x) {
        Some(p) => p
        None => ((0 : Byte), (0 : Byte), (0 : Byte), (0 : Byte))
      }
      let saf = sa.to_double() / 255.0
      let oa = saf + da * (1.0 - saf)
      let ooff = out.offset(y, x)
      if oa <= 0.0 {
        out.data[ooff + 3] = 0
      } else {
        let mix = fn(sc : Byte, dc : Byte) -> Byte {
          let v = (sc.to_double() * saf + dc.to_double() * da * (1.0 - saf)) /
            oa
          round_byte(v)
        }
        out.data[ooff] = mix(sr, dst.data[doff])
        out.data[ooff + 1] = mix(sg, dst.data[doff + 1])
        out.data[ooff + 2] = mix(sb, dst.data[doff + 2])
        out.data[ooff + 3] = round_byte(oa * 255.0)
      }
    }
  }
  out
}

///|
/// Convert a grayscale image (R==G==B) to RGB by forcing alpha to 255.
pub fn grayscale_to_rgb(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, _a) { (r, g, b, 255) })
}

///|
/// Convert a grayscale image to RGBA with fully opaque alpha.
pub fn grayscale_to_rgba(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, _a) { (r, g, b, 255) })
}

///|
/// Drop the alpha channel, returning an RGB-only image (A=255 forced).
pub fn to_rgb(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, _a) { (r, g, b, 255) })
}

///|
/// Ensure the image has an alpha channel (A=255 if missing).
/// Since all millow images are RGBA8, this is a no-op that forces A=255.
pub fn to_rgba(img : Image) -> Image {
  map_rgba(img, fn(r, g, b, _a) { (r, g, b, 255) })
}

///|
/// RGB → HSV. Returns three `h × w` arrays: H in `[0, 360)`, S in `[0, 1]`,
/// V in `[0, 1]`.
pub fn to_hsv(
  img : Image,
) -> (Array[Array[Double]], Array[Array[Double]], Array[Array[Double]]) {
  let hh = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let ss = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let vv = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o = img.offset(y, x)
      let ri = img.data[o].to_double() / 255.0
      let gi = img.data[o + 1].to_double() / 255.0
      let bi = img.data[o + 2].to_double() / 255.0
      let mx = ri.max(gi).max(bi)
      let mn = ri.min(gi).min(bi)
      let delta = mx - mn
      let v = mx
      let s = if mx <= 0.0 { 0.0 } else { delta / mx }
      let h = if delta <= 0.0 {
        0.0
      } else if mx == ri {
        60.0 * ((gi - bi) / delta % 6.0)
      } else if mx == gi {
        60.0 * ((bi - ri) / delta + 2.0)
      } else {
        60.0 * ((ri - gi) / delta + 4.0)
      }
      let h_clamped = if h < 0.0 { h + 360.0 } else { h }
      hh[y][x] = h_clamped
      ss[y][x] = s
      vv[y][x] = v
    }
  }
  (hh, ss, vv)
}

///|
/// HSV → RGBA8. H in `[0, 360)`, S in `[0, 1]`, V in `[0, 1]`.
///
/// - `h`, `s`, `v`: equally-shaped `rows × cols` arrays; output alpha is 255.
pub fn from_hsv(
  h : Array[Array[Double]],
  s : Array[Array[Double]],
  v : Array[Array[Double]],
) -> Image {
  let rows = h.length()
  let cols = if rows == 0 { 0 } else { h[0].length() }
  let img = Image::new(rows, cols)
  for y = 0; y < rows; y = y + 1 {
    for x = 0; x < cols; x = x + 1 {
      let hi = h[y][x]
      let si = s[y][x]
      let vi = v[y][x]
      let c = vi * si
      let hp = hi / 60.0
      let x_val = c * (1.0 - (hp % 2.0 - 1.0).abs())
      let (r1, g1, b1) = if hp < 1.0 {
        (c, x_val, 0.0)
      } else if hp < 2.0 {
        (x_val, c, 0.0)
      } else if hp < 3.0 {
        (0.0, c, x_val)
      } else if hp < 4.0 {
        (0.0, x_val, c)
      } else if hp < 5.0 {
        (x_val, 0.0, c)
      } else {
        (c, 0.0, x_val)
      }
      let m = vi - c
      let o = img.offset(y, x)
      img.data[o] = round_byte((r1 + m) * 255.0)
      img.data[o + 1] = round_byte((g1 + m) * 255.0)
      img.data[o + 2] = round_byte((b1 + m) * 255.0)
      img.data[o + 3] = 255
    }
  }
  img
}

///|
/// RGB → YCbCr (ITU-R BT.601). Returns three `h × w` arrays `(Y, Cb, Cr)`.
/// `Y` is in `[0, 255]`; `Cb` and `Cr` are centred at 128.
pub fn to_ycbcr(
  img : Image,
) -> (Array[Array[Double]], Array[Array[Double]], Array[Array[Double]]) {
  let yy = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let cb = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let cr = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o = img.offset(y, x)
      let ri = img.data[o].to_double()
      let gi = img.data[o + 1].to_double()
      let bi = img.data[o + 2].to_double()
      yy[y][x] = 0.299 * ri + 0.587 * gi + 0.114 * bi
      cb[y][x] = -0.168736 * ri - 0.331264 * gi + 0.5 * bi + 128.0
      cr[y][x] = 0.5 * ri - 0.418688 * gi - 0.081312 * bi + 128.0
    }
  }
  (yy, cb, cr)
}

///|
/// YCbCr → RGBA8. Expects `Y` in `[0, 255]`, `Cb`/`Cr` centred at 128.
/// Output alpha is 255.
pub fn from_ycbcr(
  y_arr : Array[Array[Double]],
  cb_arr : Array[Array[Double]],
  cr_arr : Array[Array[Double]],
) -> Image {
  let rows = y_arr.length()
  let cols = if rows == 0 { 0 } else { y_arr[0].length() }
  let img = Image::new(rows, cols)
  for y = 0; y < rows; y = y + 1 {
    for x = 0; x < cols; x = x + 1 {
      let yi = y_arr[y][x]
      let cbi = cb_arr[y][x] - 128.0
      let cri = cr_arr[y][x] - 128.0
      let o = img.offset(y, x)
      img.data[o] = round_byte(yi + 1.402 * cri)
      img.data[o + 1] = round_byte(yi - 0.344136 * cbi - 0.714136 * cri)
      img.data[o + 2] = round_byte(yi + 1.772 * cbi)
      img.data[o + 3] = 255
    }
  }
  img
}

///|
/// Composite a solid color onto `img` using `mask` as the alpha source.
/// Where `mask` is bright, the color is applied more strongly.
///
/// - `mask`: its red channel (normalized to `[0, 1]`) is used as the blend weight.
/// - `r`, `g`, `b`: the color to composite.
pub fn composite_color(
  img : Image,
  mask : Image,
  r : Byte,
  g : Byte,
  b : Byte,
) -> Image {
  let out = img.clone()
  for y = 0; y < out.h; y = y + 1 {
    for x = 0; x < out.w; x = x + 1 {
      let ma = match mask.pixel_at(y, x) {
        Some((mr, _mg, _mb, _ma)) => mr.to_double() / 255.0
        None => 0.0
      }
      if ma <= 0.0 {
        continue
      }
      let o = out.offset(y, x)
      let inv = 1.0 - ma
      out.data[o] = round_byte(
        out.data[o].to_double() * inv + r.to_double() * ma,
      )
      out.data[o + 1] = round_byte(
        out.data[o + 1].to_double() * inv + g.to_double() * ma,
      )
      out.data[o + 2] = round_byte(
        out.data[o + 2].to_double() * inv + b.to_double() * ma,
      )
    }
  }
  out
}
