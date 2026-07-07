///|
/// Local Wiener filter — a per-pixel adaptive minimum-mean-square-error
/// estimator over a `window_size × window_size` neighborhood.
///
/// For each channel, the output is:
/// ```
/// if local_var > noise_var:
///   mean + (local_var - noise_var) / local_var * (pixel - mean)
/// else:
///   mean
/// ```
///
/// - `noise_var`: estimated noise variance. If `noise_var < 0`, it defaults to
///   the mean of all local window variances (matching scipy's
///   `scipy.signal.wiener` default).
///
/// Edge-preserving denoiser: in low-variance regions the output collapses to
/// the local mean (heavy smoothing); in high-variance regions near edges the
/// original pixel is largely kept. `window_size` should be odd. Border
/// coordinates are clamped (replicate); alpha is preserved.
///
/// Operates per-RGB-channel on luma-space statistics shared across channels.
pub fn wiener_filter(
  img : Image,
  window_size : Int,
  noise_var : Double,
) -> Image {
  let r = window_size / 2
  let h = img.h
  let w = img.w
  let out = Image::new(h, w)
  let stride = w * 4
  if h == 0 || w == 0 {
    return out
  }
  // Compute luma and its integral images for local mean / variance.
  let gray = to_grayscale(img)
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  let integ2 = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let mut row_sum2 = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let v = gray.data[gray_row + x * 4].to_double()
      row_sum = row_sum + v
      row_sum2 = row_sum2 + v * v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
      integ2[(y + 1) * w1 + (x + 1)] = integ2[y * w1 + (x + 1)] + row_sum2
    }
  }
  // Determine the noise variance: use the provided value, or estimate as the
  // mean of local variances (scipy default).
  let nvar = if noise_var >= 0.0 {
    noise_var
  } else {
    let mut total = 0.0
    for y = 0; y < h; y = y + 1 {
      let y0 = clampi(y - r, 0, h - 1)
      let y1 = clampi(y + r, 0, h - 1)
      for x = 0; x < w; x = x + 1 {
        let x0 = clampi(x - r, 0, w - 1)
        let x1 = clampi(x + r, 0, w - 1)
        let count = (y1 - y0 + 1) * (x1 - x0 + 1)
        let cnt = count.to_double()
        let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
          integ[y0 * w1 + (x1 + 1)] -
          integ[(y1 + 1) * w1 + x0] +
          integ[y0 * w1 + x0]
        let s2 = integ2[(y1 + 1) * w1 + (x1 + 1)] -
          integ2[y0 * w1 + (x1 + 1)] -
          integ2[(y1 + 1) * w1 + x0] +
          integ2[y0 * w1 + x0]
        let mean = s / cnt
        let variance = s2 / cnt - mean * mean
        total = total + variance
      }
    }
    total / (h * w).to_double()
  }
  // Apply the Wiener filter per channel, sharing luma-space local statistics.
  for y = 0; y < h; y = y + 1 {
    let y0 = clampi(y - r, 0, h - 1)
    let y1 = clampi(y + r, 0, h - 1)
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let x0 = clampi(x - r, 0, w - 1)
      let x1 = clampi(x + r, 0, w - 1)
      let count = (y1 - y0 + 1) * (x1 - x0 + 1)
      let cnt = count.to_double()
      let s = integ[(y1 + 1) * w1 + (x1 + 1)] -
        integ[y0 * w1 + (x1 + 1)] -
        integ[(y1 + 1) * w1 + x0] +
        integ[y0 * w1 + x0]
      let s2 = integ2[(y1 + 1) * w1 + (x1 + 1)] -
        integ2[y0 * w1 + (x1 + 1)] -
        integ2[(y1 + 1) * w1 + x0] +
        integ2[y0 * w1 + x0]
      let mean = s / cnt
      let variance = s2 / cnt - mean * mean
      let oo = out_base + x * 4
      for c = 0; c < 3; c = c + 1 {
        let pv = img.data[oo + c].to_double()
        let res = if variance > nvar {
          mean + (variance - nvar) / variance * (pv - mean)
        } else {
          mean
        }
        out.data[oo + c] = round_byte(res)
      }
      out.data[oo + 3] = img.data[oo + 3]
    }
  }
  out
}
