///|
/// Reflect an index into `[0, n)` with edge duplication (BORDER_REFLECT).
fn reflect_index(i : Int, n : Int) -> Int {
  if n == 1 {
    return 0
  }
  let period = 2 * n
  let mut m = i % period
  if m < 0 {
    m = m + period
  }
  if m >= n {
    period - 1 - m
  } else {
    m
  }
}

///|
/// Wrap an index into `[0, n)`.
fn wrap_index(i : Int, n : Int) -> Int {
  let m = i % n
  if m < 0 {
    m + n
  } else {
    m
  }
}

///|
/// Pad the image by adding margins around it.
///
/// - `top`, `right`, `bottom`, `left`: margin sizes in pixels.
/// - `mode`: `PadMode` controlling how the new border pixels are filled:
///   - `Constant(r, g, b, a)` — fill with the given color.
///   - `Replicate` — copy the nearest edge pixel.
///   - `Reflect` — mirror with edge duplication.
///   - `Wrap` — tile the source.
///
/// The output size is `(img.h + top + bottom, img.w + left + right)`.
pub fn pad(
  img : Image,
  top : Int,
  right : Int,
  bottom : Int,
  left : Int,
  mode : PadMode,
) -> Image {
  let nh = img.h + top + bottom
  let nw = img.w + left + right
  let out = Image::new(nh, nw)
  for oy = 0; oy < nh; oy = oy + 1 {
    for ox = 0; ox < nw; ox = ox + 1 {
      let sy = oy - top
      let sx = ox - left
      let oo = out.offset(oy, ox)
      if sy >= 0 && sy < img.h && sx >= 0 && sx < img.w {
        copy_px(img, img.offset(sy, sx), out, oo)
      } else {
        match mode {
          Constant(cr, cg, cb, ca) => {
            out.data[oo] = cr
            out.data[oo + 1] = cg
            out.data[oo + 2] = cb
            out.data[oo + 3] = ca
          }
          Replicate =>
            copy_px(
              img,
              img.offset(clampi(sy, 0, img.h - 1), clampi(sx, 0, img.w - 1)),
              out,
              oo,
            )
          Reflect =>
            copy_px(
              img,
              img.offset(reflect_index(sy, img.h), reflect_index(sx, img.w)),
              out,
              oo,
            )
          Wrap =>
            copy_px(
              img,
              img.offset(wrap_index(sy, img.h), wrap_index(sx, img.w)),
              out,
              oo,
            )
        }
      }
    }
  }
  out
}

///|
/// Center-pad the image to at least `h x w` using the chosen border mode.
///
/// - `h`, `w`: target minimum dimensions.
/// - `mode`: `PadMode` for the new border pixels.
///
/// Padding is split evenly (left/top-biased on odd sizes). If the image
/// already meets or exceeds a dimension, no padding is added on that axis.
pub fn pad_to_size(img : Image, h : Int, w : Int, mode : PadMode) -> Image {
  let dh = clampi(h - img.h, 0, h)
  let dw = clampi(w - img.w, 0, w)
  let top = dh / 2
  let left = dw / 2
  pad(img, top, dw - left, dh - top, left, mode)
}
