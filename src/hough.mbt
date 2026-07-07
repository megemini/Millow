///|
/// A detected line in Hough space: `(rho, theta)` in polar coordinates, plus
/// the vote count.
priv struct HoughLine {
  rho : Double
  theta : Double
  votes : Int
}

///|
/// Hough line transform.
///
/// Detects straight lines in a binary edge image by accumulating votes in the
/// `(rho, theta)` parameter space. For each foreground pixel (luma > 0), all
/// lines passing through it vote in the accumulator.
///
/// - `threshold`: minimum vote count for a line to be returned.
/// - `theta_resolution`: number of theta bins in `[0, π)` (default 180).
/// - `rho_resolution`: bin width in pixels for the rho axis (default 1).
///
/// Returns an array of `(rho, theta, votes)` tuples, sorted by votes
/// descending. `rho` is the perpendicular distance from the origin (top-left
/// corner) to the line; `theta` is the angle in radians. The line equation is
/// `rho = x * cos(theta) + y * sin(theta)`.
///
/// The input is treated as a binary edge image (e.g. the output of `sobel` or
/// `canny`); any non-zero luma pixel votes.
pub fn hough_lines(
  img : Image,
  threshold : Int,
  theta_resolution? : Int = 180,
  rho_resolution? : Double = 1.0,
) -> Array[(Double, Double, Int)] {
  let h = img.h
  let w = img.w
  let stride = w * 4
  if h == 0 || w == 0 {
    return []
  }
  // Rho range: [-diag, +diag], diag = sqrt(h² + w²).
  let diag = (h * h + w * w).to_double().sqrt().floor().to_int() + 1
  let n_rho = ((2 * diag + 1).to_double() / rho_resolution).ceil().to_int() + 1
  let n_theta = theta_resolution
  // Accumulator: accumulator[rho_idx][theta_idx].
  let acc = Array::make(n_rho * n_theta, 0)
  let rho_offset = (n_rho - 1) / 2
  let pi = @math.PI
  // Precompute cos/sin for each theta bin.
  let cos_t = Array::make(n_theta, 0.0)
  let sin_t = Array::make(n_theta, 0.0)
  for t = 0; t < n_theta; t = t + 1 {
    let theta = t.to_double() * pi / n_theta.to_double()
    cos_t[t] = @math.cos(theta)
    sin_t[t] = @math.sin(theta)
  }
  // Vote: for each foreground pixel, for each theta, compute rho and increment.
  for y = 0; y < h; y = y + 1 {
    let row_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let o = row_base + x * 4
      let l = luma(img.data[o], img.data[o + 1], img.data[o + 2])
      if l > 0 {
        let xd = x.to_double()
        let yd = y.to_double()
        for t = 0; t < n_theta; t = t + 1 {
          let rho = xd * cos_t[t] + yd * sin_t[t]
          let rho_idx = (rho / rho_resolution).round().to_int() + rho_offset
          if rho_idx >= 0 && rho_idx < n_rho {
            acc[rho_idx * n_theta + t] = acc[rho_idx * n_theta + t] + 1
          }
        }
      }
    }
  }
  // Collect peaks above threshold.
  let lines : Array[HoughLine] = []
  for r = 0; r < n_rho; r = r + 1 {
    for t = 0; t < n_theta; t = t + 1 {
      let v = acc[r * n_theta + t]
      if v >= threshold {
        lines.push({
          rho: (r - rho_offset).to_double() * rho_resolution,
          theta: t.to_double() * pi / n_theta.to_double(),
          votes: v,
        })
      }
    }
  }
  // Sort by votes descending.
  lines.sort_by(fn(a, b) { b.votes - a.votes })
  let result = Array::make(lines.length(), (0.0, 0.0, 0))
  for i = 0; i < lines.length(); i = i + 1 {
    let l = lines[i]
    result[i] = (l.rho, l.theta, l.votes)
  }
  result
}
