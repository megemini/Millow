///|
/// Draw a rectangle. When `fill` is false, `width` sets the stroke thickness.
pub fn draw_rect(
  img : Image,
  y : Int,
  x : Int,
  h : Int,
  w : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  fill : Bool,
  width : Int,
) -> Image {
  let out = img.clone()
  if fill {
    for yy = y; yy < y + h; yy = yy + 1 {
      for xx = x; xx < x + w; xx = xx + 1 {
        blend_at(out, yy, xx, r, g, b, a)
      }
    }
  } else {
    for xx = x; xx < x + w; xx = xx + 1 {
      stamp(out, y, xx, r, g, b, a, width)
      stamp(out, y + h - 1, xx, r, g, b, a, width)
    }
    for yy = y; yy < y + h; yy = yy + 1 {
      stamp(out, yy, x, r, g, b, a, width)
      stamp(out, yy, x + w - 1, r, g, b, a, width)
    }
  }
  out
}

///|
/// Stamp the eight symmetric points of a circle octant.
fn circle_points(
  img : Image,
  cy : Int,
  cx : Int,
  yy : Int,
  xx : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  width : Int,
) -> Unit {
  stamp(img, cy + yy, cx + xx, r, g, b, a, width)
  stamp(img, cy + yy, cx - xx, r, g, b, a, width)
  stamp(img, cy - yy, cx + xx, r, g, b, a, width)
  stamp(img, cy - yy, cx - xx, r, g, b, a, width)
  stamp(img, cy + xx, cx + yy, r, g, b, a, width)
  stamp(img, cy + xx, cx - yy, r, g, b, a, width)
  stamp(img, cy - xx, cx + yy, r, g, b, a, width)
  stamp(img, cy - xx, cx - yy, r, g, b, a, width)
}

///|
/// Draw a circle. When `fill` is false, `width` sets the stroke thickness.
pub fn draw_circle(
  img : Image,
  cy : Int,
  cx : Int,
  radius : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  fill : Bool,
  width : Int,
) -> Image {
  let out = img.clone()
  if fill {
    let r2 = (radius * radius).to_double()
    for dy = -radius; dy <= radius; dy = dy + 1 {
      let span = (r2 - (dy * dy).to_double()).sqrt().to_int()
      for dx = -span; dx <= span; dx = dx + 1 {
        blend_at(out, cy + dy, cx + dx, r, g, b, a)
      }
    }
  } else {
    let mut xx = radius
    let mut yy = 0
    let mut err = 0
    while xx >= yy {
      circle_points(out, cy, cx, yy, xx, r, g, b, a, width)
      yy = yy + 1
      err = err + 1 + 2 * yy
      if 2 * (err - xx) + 1 > 0 {
        xx = xx - 1
        err = err + 1 - 2 * xx
      }
    }
  }
  out
}
