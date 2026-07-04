///|
/// Draw a rectangle. When `fill` is false, `width` sets the stroke thickness.
///
/// The rectangle spans rows `[y, y + h)` and columns `[x, x + w)`. When
/// `fill` is true the interior is filled; otherwise a stroked outline of
/// thickness `width` is drawn. Uses source-over alpha blending; the input
/// image is not modified.
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
///
/// Centred at `(cy, cx)` with the given `radius`. Filled circles scan each
/// row over the implicit `x² + y² <= r²` disc; outlines use the midpoint
/// circle algorithm. Uses source-over alpha blending; the input image is not
/// modified.
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

///|
/// Draw an ellipse. When `fill` is false, `width` sets the stroke thickness.
///
/// Centred at `(cy, cx)` with semi-axes `ry` (rows) and `rx` (columns).
/// Outlines use the midpoint ellipse algorithm; filled ellipses scan each
/// row over the implicit disc. Uses source-over alpha blending; the input
/// image is not modified.
pub fn draw_ellipse(
  img : Image,
  cy : Int,
  cx : Int,
  ry : Int,
  rx : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  fill : Bool,
  width : Int,
) -> Image {
  let out = img.clone()
  if fill {
    for dy = -ry; dy <= ry; dy = dy + 1 {
      let rx2 = (rx * rx).to_double()
      let ry2 = (ry * ry).to_double()
      let span = (rx2 * (1.0 - (dy * dy).to_double() / ry2)).sqrt().to_int()
      for dx = -span; dx <= span; dx = dx + 1 {
        blend_at(out, cy + dy, cx + dx, r, g, b, a)
      }
    }
  } else {
    // Midpoint ellipse algorithm
    let mut x = 0
    let mut y = ry
    let rx2 = (rx * rx).to_double()
    let ry2 = (ry * ry).to_double()
    let mut d1 = ry2 - rx2 * ry.to_double() + 0.25 * rx2
    let mut dx = 2.0 * ry2 * x.to_double()
    let mut dy = 2.0 * rx2 * y.to_double()
    // Region 1
    while dx < dy {
      stamp(out, cy + y, cx + x, r, g, b, a, width)
      stamp(out, cy + y, cx - x, r, g, b, a, width)
      stamp(out, cy - y, cx + x, r, g, b, a, width)
      stamp(out, cy - y, cx - x, r, g, b, a, width)
      if d1 < 0.0 {
        x = x + 1
        dx = dx + 2.0 * ry2
        d1 = d1 + dx + ry2
      } else {
        x = x + 1
        y = y - 1
        dx = dx + 2.0 * ry2
        dy = dy - 2.0 * rx2
        d1 = d1 + dx - dy + ry2
      }
    }
    // Region 2
    let mut d2 = ry2 * (x.to_double() + 0.5) * (x.to_double() + 0.5) +
      rx2 * (y.to_double() - 1.0) * (y.to_double() - 1.0) -
      rx2 * ry2
    while y >= 0 {
      stamp(out, cy + y, cx + x, r, g, b, a, width)
      stamp(out, cy + y, cx - x, r, g, b, a, width)
      stamp(out, cy - y, cx + x, r, g, b, a, width)
      stamp(out, cy - y, cx - x, r, g, b, a, width)
      if d2 > 0.0 {
        y = y - 1
        dy = dy - 2.0 * rx2
        d2 = d2 + rx2 - dy
      } else {
        y = y - 1
        x = x + 1
        dx = dx + 2.0 * ry2
        dy = dy - 2.0 * rx2
        d2 = d2 + dx - dy + rx2
      }
    }
  }
  out
}

///|
/// Draw a polygon. When `fill` is false, `width` sets the stroke thickness.
///
/// `points` is an array of `(y, x)` vertices connected in order, closing the
/// path from the last back to the first. Filled polygons use a scanline fill;
/// outlines call `draw_line` for each edge. Returns the input image unchanged
/// when fewer than two vertices are supplied. The input image is not modified.
pub fn draw_polygon(
  img : Image,
  points : Array[(Int, Int)],
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  fill : Bool,
  width : Int,
) -> Image {
  let mut out = img.clone()
  if points.length() < 2 {
    return out
  }
  if fill {
    // Scanline fill
    let mut min_y = points[0].0
    let mut max_y = points[0].0
    for i = 1; i < points.length(); i = i + 1 {
      min_y = min_y.min(points[i].0)
      max_y = max_y.max(points[i].0)
    }
    for y = min_y; y <= max_y; y = y + 1 {
      let intersections : Array[Int] = []
      let n = points.length()
      for i = 0; i < n; i = i + 1 {
        let (y0, x0) = points[i]
        let (y1, x1) = points[(i + 1) % n]
        if (y0 <= y && y1 > y) || (y1 <= y && y0 > y) {
          let x_int = x0.to_double() +
            (y - y0).to_double() * (x1 - x0).to_double() / (y1 - y0).to_double()
          intersections.push(x_int.to_int())
        }
      }
      // Sort intersections
      let arr = intersections
      for i = 0; i < arr.length(); i = i + 1 {
        for j = i + 1; j < arr.length(); j = j + 1 {
          if arr[j] < arr[i] {
            let tmp = arr[i]
            arr[i] = arr[j]
            arr[j] = tmp
          }
        }
      }
      // Fill between pairs
      for i = 0; i + 1 < arr.length(); i = i + 2 {
        for x = arr[i]; x <= arr[i + 1]; x = x + 1 {
          blend_at(out, y, x, r, g, b, a)
        }
      }
    }
  } else {
    // Draw outline
    for i = 0; i < points.length(); i = i + 1 {
      let (y0, x0) = points[i]
      let (y1, x1) = points[(i + 1) % points.length()]
      out = draw_line(out, y0, x0, y1, x1, r, g, b, a, width)
    }
  }
  out
}

///|
/// Flood fill: replace the colour at `(y, x)` and all connected pixels of
/// similar colour (within `tolerance`) with the new colour.
///
/// 4-connected BFS from the seed pixel; a pixel is replaced when each of its
/// RGBA channels is within `tolerance` of the seed colour. Out-of-bounds seeds
/// and no-op seeds (fill colour equals seed colour) return a clone of the
/// input. The input image is not modified.
pub fn flood_fill(
  img : Image,
  y : Int,
  x : Int,
  r : Byte,
  g : Byte,
  b : Byte,
  a : Byte,
  tolerance : Byte,
) -> Image {
  let out = img.clone()
  if y < 0 || x < 0 || y >= out.h || x >= out.w {
    return out
  }
  let o0 = out.offset(y, x)
  let sr = out.data[o0]
  let sg = out.data[o0 + 1]
  let sb = out.data[o0 + 2]
  let sa = out.data[o0 + 3]
  // If target colour is the same as fill colour, nothing to do
  if sr == r && sg == g && sb == b && sa == a {
    return out
  }
  let tol = tolerance.to_int()
  let is_match = fn(img_ : Image, yy : Int, xx : Int) -> Bool {
    if yy < 0 || xx < 0 || yy >= img_.h || xx >= img_.w {
      return false
    }
    let o = img_.offset(yy, xx)
    (img_.data[o].to_int() - sr.to_int()).abs() <= tol &&
    (img_.data[o + 1].to_int() - sg.to_int()).abs() <= tol &&
    (img_.data[o + 2].to_int() - sb.to_int()).abs() <= tol &&
    (img_.data[o + 3].to_int() - sa.to_int()).abs() <= tol
  }
  // BFS flood fill
  let stack : Array[(Int, Int)] = [(y, x)]
  while stack.length() > 0 {
    let (cy, cx) = stack.unsafe_pop()
    if !is_match(out, cy, cx) {
      continue
    }
    let o = out.offset(cy, cx)
    out.data[o] = r
    out.data[o + 1] = g
    out.data[o + 2] = b
    out.data[o + 3] = a
    stack.push((cy - 1, cx))
    stack.push((cy + 1, cx))
    stack.push((cy, cx - 1))
    stack.push((cy, cx + 1))
  }
  out
}
