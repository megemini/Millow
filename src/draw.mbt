///|
/// Alpha-composite a color onto pixel `(y, x)` in place (no-op if OOB).
fn blend_at(
  img : Image,
  y : Int,
  x : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
) -> Unit {
  if y < 0 || x < 0 || y >= img.h || x >= img.w {
    return
  }
  let o = img.offset(y, x)
  let af = a.to_double() / 255.0
  let inv = 1.0 - af
  img.data[o] = round_byte(img.data[o].to_double() * inv + r.to_double() * af)
  img.data[o + 1] = round_byte(
    img.data[o + 1].to_double() * inv + g.to_double() * af,
  )
  img.data[o + 2] = round_byte(
    img.data[o + 2].to_double() * inv + b.to_double() * af,
  )
  let ea = img.data[o + 3].to_double()
  img.data[o + 3] = round_byte(a.to_double() + ea * inv)
}

///|
/// Stamp a filled `width x width` block centered at `(y, x)`.
fn stamp(
  img : Image,
  y : Int,
  x : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  width : Int,
) -> Unit {
  let hw = if width < 1 { 0 } else { (width - 1) / 2 }
  for dy = -hw; dy <= hw; dy = dy + 1 {
    for dx = -hw; dx <= hw; dx = dx + 1 {
      blend_at(img, y + dy, x + dx, r, g, b, a)
    }
  }
}

///|
/// Draw a single pixel (alpha blended). Returns a new image.
///
/// Sets the pixel at `(y, x)` to `(r, g, b)` alpha-composited with the
/// existing colour using source-over blending. Out-of-bounds coordinates are
/// ignored. The input image is not modified.
pub fn draw_pixel(
  img : Image,
  y : Int,
  x : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
) -> Image {
  let out = img.clone()
  blend_at(out, y, x, r, g, b, a)
  out
}

///|
/// Draw a line with the Bresenham algorithm and the given stroke width.
///
/// Connects `(y0, x0)` to `(y1, x1)` inclusive, stamping a `width x width`
/// block at each rasterized pixel using source-over alpha blending. The input
/// image is not modified.
pub fn draw_line(
  img : Image,
  y0 : Int,
  x0 : Int,
  y1 : Int,
  x1 : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  width : Int,
) -> Image {
  let out = img.clone()
  let dx = (x1 - x0).abs()
  let dy = (y1 - y0).abs()
  let sx = if x0 < x1 { 1 } else { -1 }
  let sy = if y0 < y1 { 1 } else { -1 }
  let mut err = dx - dy
  let mut cx = x0
  let mut cy = y0
  for ;; {
    stamp(out, cy, cx, r, g, b, a, width)
    if cx == x1 && cy == y1 {
      break
    }
    let e2 = 2 * err
    if e2 > -dy {
      err = err - dy
      cx = cx + sx
    }
    if e2 < dx {
      err = err + dx
      cy = cy + sy
    }
  }
  out
}
