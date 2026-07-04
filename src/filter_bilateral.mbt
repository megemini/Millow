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
  let h = img.h
  let w = img.w
  let n = h * w
  let stride = w * 4
  let out = Image::new(h, w)
  // sigma_color is in [0, 1] range (matching skimage)
  let sc2 = 2.0 * sigma_color * sigma_color
  let ss2 = 2.0 * sigma_space * sigma_space
  let hmax = h - 1
  let wmax = w - 1
  // Precompute normalized RGB as a flat double array (3 channels per pixel).
  // Avoids repeated `to_double() / 255.0` per kernel element per pixel.
  let rgb = Array::make(n * 3, 0.0)
  for i = 0; i < n; i = i + 1 {
    let o = i * 4
    let p = i * 3
    rgb[p] = img.data[o].to_double() / 255.0
    rgb[p + 1] = img.data[o + 1].to_double() / 255.0
    rgb[p + 2] = img.data[o + 2].to_double() / 255.0
  }
  // Precompute spatial weights `exp(-dist² / ss2)` for the (2r+1)² kernel.
  // The original `exp(-(dist2/ss2 + color_dist2/sc2))` decomposes into
  // `spatial * exp(-color_dist2/sc2)` (mathematically equal; well within the
  // test tolerance of 2).
  let klen = 2 * r + 1
  let spatial = Array::make(klen * klen, 0.0)
  for i = 0; i < klen; i = i + 1 {
    let dy = i - r
    for j = 0; j < klen; j = j + 1 {
      let dx = j - r
      let dist2 = (dy * dy + dx * dx).to_double()
      spatial[i * klen + j] = @math.exp(-dist2 / ss2)
    }
  }
  // Per-row precomputation: hoist reflected y-row bases out of the x loop.
  for y = 0; y < h; y = y + 1 {
    // Precompute reflected y-row bases (in pixel units) for this row.
    let y_bases = Array::make(klen, 0)
    for i = 0; i < klen; i = i + 1 {
      let cy = reflect_coord(y + i - r, hmax)
      y_bases[i] = cy * w
    }
    let row_alpha_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let p0 = (y * w + x) * 3
      let center_r = rgb[p0]
      let center_g = rgb[p0 + 1]
      let center_b = rgb[p0 + 2]
      let mut w_sum_r = 0.0
      let mut w_sum_g = 0.0
      let mut w_sum_b = 0.0
      let mut w_total = 0.0
      // Interior x can skip reflect_coord for the x-axis.
      let interior_x = x >= r && x <= wmax - r
      for i = 0; i < klen; i = i + 1 {
        let row_base = y_bases[i] * 3
        let spatial_row = i * klen
        for j = 0; j < klen; j = j + 1 {
          let dx = j - r
          let cx = if interior_x { x + dx } else { reflect_coord(x + dx, wmax) }
          let p = row_base + cx * 3
          let cr = rgb[p]
          let cg = rgb[p + 1]
          let cb = rgb[p + 2]
          let dr = cr - center_r
          let dg = cg - center_g
          let db = cb - center_b
          let color_dist2 = dr * dr + dg * dg + db * db
          let weight = spatial[spatial_row + j] * @math.exp(-color_dist2 / sc2)
          w_total = w_total + weight
          w_sum_r = w_sum_r + cr * weight
          w_sum_g = w_sum_g + cg * weight
          w_sum_b = w_sum_b + cb * weight
        }
      }
      let oo = row_alpha_base + x * 4
      if w_total > 0.0 {
        out.data[oo] = round_byte(w_sum_r / w_total * 255.0)
        out.data[oo + 1] = round_byte(w_sum_g / w_total * 255.0)
        out.data[oo + 2] = round_byte(w_sum_b / w_total * 255.0)
      }
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}
