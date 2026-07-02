///|
/// Nearest-neighbor resampling.
fn resize_nearest(img : Image, dst_h : Int, dst_w : Int) -> Image {
  let out = Image::new(dst_h, dst_w)
  if img.is_empty() || dst_h == 0 || dst_w == 0 {
    return out
  }
  for y = 0; y < dst_h; y = y + 1 {
    let sy = clampi(y * img.h / dst_h, 0, img.h - 1)
    for x = 0; x < dst_w; x = x + 1 {
      let sx = clampi(x * img.w / dst_w, 0, img.w - 1)
      copy_px(img, img.offset(sy, sx), out, out.offset(y, x))
    }
  }
  out
}

///|
/// Bilinear resampling.
fn resize_bilinear(img : Image, dst_h : Int, dst_w : Int) -> Image {
  let out = Image::new(dst_h, dst_w)
  if img.is_empty() || dst_h == 0 || dst_w == 0 {
    return out
  }
  let sh = img.h.to_double()
  let sw = img.w.to_double()
  for y = 0; y < dst_h; y = y + 1 {
    let fy = clampd(
      (y.to_double() + 0.5) * sh / dst_h.to_double() - 0.5,
      0.0,
      sh - 1.0,
    )
    let y0 = @math.floor(fy).to_int()
    let y1 = clampi(y0 + 1, 0, img.h - 1)
    let wy = fy - y0.to_double()
    for x = 0; x < dst_w; x = x + 1 {
      let fx = clampd(
        (x.to_double() + 0.5) * sw / dst_w.to_double() - 0.5,
        0.0,
        sw - 1.0,
      )
      let x0 = @math.floor(fx).to_int()
      let x1 = clampi(x0 + 1, 0, img.w - 1)
      let wx = fx - x0.to_double()
      let oo = out.offset(y, x)
      for c = 0; c < 4; c = c + 1 {
        let v00 = img.data[img.offset(y0, x0) + c].to_double()
        let v01 = img.data[img.offset(y0, x1) + c].to_double()
        let v10 = img.data[img.offset(y1, x0) + c].to_double()
        let v11 = img.data[img.offset(y1, x1) + c].to_double()
        let top = v00 * (1.0 - wx) + v01 * wx
        let bot = v10 * (1.0 - wx) + v11 * wx
        out.data[oo + c] = round_byte(top * (1.0 - wy) + bot * wy)
      }
    }
  }
  out
}

///|
/// Cubic kernel for bicubic interpolation (a = -0.5).
fn cubic_w(t : Double) -> Double {
  let a = -0.5
  let at = if t < 0.0 { -t } else { t }
  if at <= 1.0 {
    (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0
  } else if at < 2.0 {
    a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a
  } else {
    0.0
  }
}

///|
/// Bicubic resampling using a 4x4 neighborhood with the standard cubic kernel.
fn resize_bicubic(img : Image, dst_h : Int, dst_w : Int) -> Image {
  let out = Image::new(dst_h, dst_w)
  if img.is_empty() || dst_h == 0 || dst_w == 0 {
    return out
  }
  let sh = img.h.to_double()
  let sw = img.w.to_double()
  for y = 0; y < dst_h; y = y + 1 {
    let fy = clampd(
      (y.to_double() + 0.5) * sh / dst_h.to_double() - 0.5,
      0.0,
      sh - 1.0,
    )
    let y0 = @math.floor(fy).to_int()
    let wy = fy - y0.to_double()
    for x = 0; x < dst_w; x = x + 1 {
      let fx = clampd(
        (x.to_double() + 0.5) * sw / dst_w.to_double() - 0.5,
        0.0,
        sw - 1.0,
      )
      let x0 = @math.floor(fx).to_int()
      let wx = fx - x0.to_double()
      let oo = out.offset(y, x)
      for c = 0; c < 4; c = c + 1 {
        let mut acc = 0.0
        for j = -1; j <= 2; j = j + 1 {
          let wyi = cubic_w(wy - j.to_double())
          let sy = clampi(y0 + j, 0, img.h - 1)
          for i = -1; i <= 2; i = i + 1 {
            let wxi = cubic_w(wx - i.to_double())
            let sx = clampi(x0 + i, 0, img.w - 1)
            acc = acc + wyi * wxi * img.data[img.offset(sy, sx) + c].to_double()
          }
        }
        out.data[oo + c] = round_byte(acc)
      }
    }
  }
  out
}

///|
/// Resize to an explicit size using the given interpolation.
pub fn resize(img : Image, dst_h : Int, dst_w : Int, interp : Interp) -> Image {
  match interp {
    Nearest => resize_nearest(img, dst_h, dst_w)
    Bilinear => resize_bilinear(img, dst_h, dst_w)
    Bicubic => resize_bicubic(img, dst_h, dst_w)
  }
}

///|
/// Scale both dimensions by a factor.
pub fn rescale(img : Image, scale : Double, interp : Interp) -> Image {
  let dh = @math.round(img.h.to_double() * scale).to_int()
  let dw = @math.round(img.w.to_double() * scale).to_int()
  resize(img, dh, dw, interp)
}

///|
/// Scale to fit inside `max_h x max_w`, preserving aspect ratio.
pub fn resize_to_fit(
  img : Image,
  max_h : Int,
  max_w : Int,
  interp : Interp,
) -> Image {
  if img.is_empty() {
    return img.clone()
  }
  let sy = max_h.to_double() / img.h.to_double()
  let sx = max_w.to_double() / img.w.to_double()
  let s = if sy < sx { sy } else { sx }
  rescale(img, s, interp)
}

///|
/// Scale to cover `min_h x min_w`, preserving aspect ratio, cropping overflow.
pub fn resize_to_cover(
  img : Image,
  min_h : Int,
  min_w : Int,
  interp : Interp,
) -> Image raise ImageError {
  if img.is_empty() {
    return img.clone()
  }
  let sy = min_h.to_double() / img.h.to_double()
  let sx = min_w.to_double() / img.w.to_double()
  let s = if sy > sx { sy } else { sx }
  let scaled = rescale(img, s, interp)
  let oy = clampi((scaled.h - min_h) / 2, 0, scaled.h)
  let ox = clampi((scaled.w - min_w) / 2, 0, scaled.w)
  crop(scaled, oy, ox, clampi(min_h, 0, scaled.h), clampi(min_w, 0, scaled.w))
}
