///|
/// Box blur using per-channel integral images (O(1) per pixel).
/// Border windows are shrunk to the valid area.
pub fn box_blur(img : Image, radius : Int) -> Image {
  if radius <= 0 || img.is_empty() {
    return img.clone()
  }
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let w1 = w + 1
  for c = 0; c < 3; c = c + 1 {
    let integ = Array::make((h + 1) * w1, 0)
    for y = 0; y < h; y = y + 1 {
      let mut rowsum = 0
      for x = 0; x < w; x = x + 1 {
        rowsum = rowsum + img.data[img.offset(y, x) + c].to_int()
        integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + rowsum
      }
    }
    for y = 0; y < h; y = y + 1 {
      let y0 = clampi(y - radius, 0, h - 1)
      let y1 = clampi(y + radius, 0, h - 1)
      for x = 0; x < w; x = x + 1 {
        let x0 = clampi(x - radius, 0, w - 1)
        let x1 = clampi(x + radius, 0, w - 1)
        let area = (y1 - y0 + 1) * (x1 - x0 + 1)
        let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
          integ[y0 * w1 + (x1 + 1)] -
          integ[(y1 + 1) * w1 + x0] +
          integ[y0 * w1 + x0]
        out.data[out.offset(y, x) + c] = clamp_byte(s / area)
      }
    }
  }
  let n = h * w
  for i = 0; i < n; i = i + 1 {
    out.data[i * 4 + 3] = img.data[i * 4 + 3]
  }
  out
}
