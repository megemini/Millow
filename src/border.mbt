///|
/// Border handling mode for convolution and interpolation operations.
pub enum BorderMode {
  Replicate
  Reflect
  Wrap
  Constant(Byte, Byte, Byte, Byte)
}

///|
/// Clamp coordinates to valid range (replicate border).
fn clamp_coord(v : Int, max_v : Int) -> Int {
  if v < 0 {
    0
  } else if v > max_v {
    max_v
  } else {
    v
  }
}

///|
/// Reflect coordinates (mirror with edge duplication).
fn reflect_coord(v : Int, max_v : Int) -> Int {
  if max_v == 0 {
    0
  } else {
    let size = max_v + 1
    let twice = size * 2
    let mut r = v
    if r < 0 {
      r = -r - 1
    }
    r = r % twice
    if r > max_v {
      twice - 1 - r
    } else {
      r
    }
  }
}

///|
/// Wrap coordinates (tile).
fn wrap_coord(v : Int, max_v : Int) -> Int {
  if max_v == 0 {
    0
  } else {
    let size = max_v + 1
    let mut r = v % size
    if r < 0 {
      r = r + size
    }
    r
  }
}

///|
/// Get pixel at coordinates with specified border mode.
pub fn get_pixel(
  img : Image,
  y : Int,
  x : Int,
  mode : BorderMode,
) -> (Byte, Byte, Byte, Byte) {
  match mode {
    Replicate => {
      let cy = clamp_coord(y, img.h - 1)
      let cx = clamp_coord(x, img.w - 1)
      let o = img.offset(cy, cx)
      (img.data[o], img.data[o + 1], img.data[o + 2], img.data[o + 3])
    }
    Reflect => {
      let cy = reflect_coord(y, img.h - 1)
      let cx = reflect_coord(x, img.w - 1)
      let o = img.offset(cy, cx)
      (img.data[o], img.data[o + 1], img.data[o + 2], img.data[o + 3])
    }
    Wrap => {
      let cy = wrap_coord(y, img.h - 1)
      let cx = wrap_coord(x, img.w - 1)
      let o = img.offset(cy, cx)
      (img.data[o], img.data[o + 1], img.data[o + 2], img.data[o + 3])
    }
    Constant(r, g, b, a) =>
      if y >= 0 && y < img.h && x >= 0 && x < img.w {
        let o = img.offset(y, x)
        (img.data[o], img.data[o + 1], img.data[o + 2], img.data[o + 3])
      } else {
        (r, g, b, a)
      }
  }
}

///|
/// Get luma at coordinates with specified border mode.
pub fn luma_at_mode(img : Image, y : Int, x : Int, mode : BorderMode) -> Double {
  let (r, g, b, _) = get_pixel(img, y, x, mode)
  ((r.to_int() * 77 + g.to_int() * 150 + b.to_int() * 29) >> 8).to_double()
}

///|
/// Get luma at coordinates with replicate border mode.
pub fn luma_at(img : Image, y : Int, x : Int) -> Double {
  luma_at_mode(img, y, x, Replicate)
}

///|
/// Bilinear interpolation of luma at fractional coordinates with specified border mode.
pub fn bilinear_luma(
  img : Image,
  fy : Double,
  fx : Double,
  mode : BorderMode,
) -> Double {
  let y0 = fy.floor().to_int()
  let x0 = fx.floor().to_int()
  let y1 = y0 + 1
  let x1 = x0 + 1
  let dy = fy - y0.to_double()
  let dx = fx - x0.to_double()
  let v00 = luma_at_mode(img, y0, x0, mode)
  let v01 = luma_at_mode(img, y0, x1, mode)
  let v10 = luma_at_mode(img, y1, x0, mode)
  let v11 = luma_at_mode(img, y1, x1, mode)
  v00 * (1.0 - dy) * (1.0 - dx) +
  v01 * (1.0 - dy) * dx +
  v10 * dy * (1.0 - dx) +
  v11 * dy * dx
}
