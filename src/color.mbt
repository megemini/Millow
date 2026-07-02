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
pub fn flatten_alpha(img : Image, r : Byte, g : Byte, b : Byte) -> Image {
  map_rgba(img, fn(pr, pg, pb, a) {
    let ai = a.to_int()
    let inv = 255 - ai
    (
      clamp_byte((pr.to_int() * ai + r.to_int() * inv) / 255),
      clamp_byte((pg.to_int() * ai + g.to_int() * inv) / 255),
      clamp_byte((pb.to_int() * ai + b.to_int() * inv) / 255),
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
