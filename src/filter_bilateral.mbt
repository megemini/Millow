///|
/// Bilateral filter: edge-preserving denoising. Uses reflect boundary mode.
///
/// - `d`: neighbourhood diameter; if `<= 0`, computed as
///   `max(5, 2 * ceil(3 * sigma_space) + 1)`.
/// - `sigma_color`: tonal weight, in `[0, 1]` (normalized RGB distance).
/// - `sigma_space`: spatial weight.
pub fn bilateral_filter(
  img : Image,
  d : Int,
  sigma_color : Double,
  sigma_space : Double,
) -> Image {
  let win_size = if d <= 0 {
    (2 * (3.0 * sigma_space).ceil().to_int() + 1).max(5)
  } else {
    d
  }
  let r = win_size / 2
  let out = Image::new(img.h, img.w)
  // sigma_color is in [0, 1] range (matching skimage)
  let sc2 = 2.0 * sigma_color * sigma_color
  let ss2 = 2.0 * sigma_space * sigma_space
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let o0 = img.offset(y, x)
      let mut w_sum_r = 0.0
      let mut w_sum_g = 0.0
      let mut w_sum_b = 0.0
      let mut w_total = 0.0
      let center_r = img.data[o0].to_double() / 255.0
      let center_g = img.data[o0 + 1].to_double() / 255.0
      let center_b = img.data[o0 + 2].to_double() / 255.0
      for dy = -r; dy <= r; dy = dy + 1 {
        for dx = -r; dx <= r; dx = dx + 1 {
          let cy = reflect_coord(y + dy, img.h - 1)
          let cx = reflect_coord(x + dx, img.w - 1)
          let co = img.offset(cy, cx)
          let cr = img.data[co].to_double() / 255.0
          let cg = img.data[co + 1].to_double() / 255.0
          let cb = img.data[co + 2].to_double() / 255.0
          let dist2 = (dy * dy + dx * dx).to_double()
          let dr = cr - center_r
          let dg = cg - center_g
          let db = cb - center_b
          let color_dist2 = dr * dr + dg * dg + db * db
          let weight = @math.exp(-(dist2 / ss2 + color_dist2 / sc2))
          w_total = w_total + weight
          w_sum_r = w_sum_r + cr * weight
          w_sum_g = w_sum_g + cg * weight
          w_sum_b = w_sum_b + cb * weight
        }
      }
      let oo = out.offset(y, x)
      if w_total > 0.0 {
        out.data[oo] = round_byte(w_sum_r / w_total * 255.0)
        out.data[oo + 1] = round_byte(w_sum_g / w_total * 255.0)
        out.data[oo + 2] = round_byte(w_sum_b / w_total * 255.0)
      }
      out.data[oo + 3] = img.data[o0 + 3]
    }
  }
  out
}
