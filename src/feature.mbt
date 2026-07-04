///|
/// Compute the Gaussian-weighted structure tensor `(sxx, syy, sxy)` of the
/// image's luma gradients. Shared by `corner_harris` and `corner_shi_tomasi`.
///
/// Uses Sobel kernels with `Constant(0,0,0,0)` border (out-of-bounds = luma 0)
/// and fuses the three Gaussian smoothing passes into a single pair of
/// separable 1D passes via `gaussian_blur_double3`.
fn structure_tensor(
  img : Image,
  sigma : Double,
) -> (Array[Array[Double]], Array[Array[Double]], Array[Array[Double]]) {
  let gray = to_grayscale(img)
  let gx = grad_double(
    gray,
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    Constant(0, 0, 0, 0),
  )
  let gy = grad_double(
    gray,
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    Constant(0, 0, 0, 0),
  )
  let h = img.h
  let w = img.w
  let gx2 = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  let gy2 = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  let gxy = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    let gx_row = gx[y]
    let gy_row = gy[y]
    let gx2_row = gx2[y]
    let gy2_row = gy2[y]
    let gxy_row = gxy[y]
    for x = 0; x < w; x = x + 1 {
      let gxi = gx_row[x]
      let gyi = gy_row[x]
      gx2_row[x] = gxi * gxi
      gy2_row[x] = gyi * gyi
      gxy_row[x] = gxi * gyi
    }
  }
  gaussian_blur_double3(gx2, gy2, gxy, sigma)
}

///|
/// Harris corner response map.
///
/// Computes the Gaussian-weighted structure tensor of the image's luma
/// gradients and returns the Harris response `det(M) - k * trace(M)²` per
/// pixel.
///
/// - `k`: Harris sensitivity parameter, typically in the range 0.04–0.06.
///
/// Returns an `h × w` array of response values; large positive values
/// indicate corners.
pub fn corner_harris(img : Image, k : Double) -> Array[Array[Double]] {
  let (sxx, syy, sxy) = structure_tensor(img, 1.0)
  let h = img.h
  let w = img.w
  let out = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    let sxx_row = sxx[y]
    let syy_row = syy[y]
    let sxy_row = sxy[y]
    let out_row = out[y]
    for x = 0; x < w; x = x + 1 {
      let sxx_v = sxx_row[x]
      let syy_v = syy_row[x]
      let sxy_v = sxy_row[x]
      let det = sxx_v * syy_v - sxy_v * sxy_v
      let trace = sxx_v + syy_v
      out_row[x] = det - k * trace * trace
    }
  }
  out
}

///|
/// Shi-Tomasi corner response map.
///
/// Computes the Gaussian-weighted structure tensor of the image's luma
/// gradients and returns the smaller eigenvalue of the tensor per pixel.
///
/// Returns an `h × w` array of response values; large values indicate
/// corner-like regions.
pub fn corner_shi_tomasi(img : Image) -> Array[Array[Double]] {
  let (sxx, syy, sxy) = structure_tensor(img, 1.0)
  let h = img.h
  let w = img.w
  let out = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    let sxx_row = sxx[y]
    let syy_row = syy[y]
    let sxy_row = sxy[y]
    let out_row = out[y]
    for x = 0; x < w; x = x + 1 {
      let sxx_v = sxx_row[x]
      let syy_v = syy_row[x]
      let sxy_v = sxy_row[x]
      let trace = sxx_v + syy_v
      let det = sxx_v * syy_v - sxy_v * sxy_v
      let disc = (trace * trace - 4.0 * det).max(0.0).sqrt()
      let l1 = (trace + disc) / 2.0
      let l2 = (trace - disc) / 2.0
      out_row[x] = l1.min(l2)
    }
  }
  out
}

///|
/// Histogram of Oriented Gradients (HOG) feature descriptor.
///
/// Computes gradients on the luma field, accumulates unsigned gradient
/// orientations (0–180°) into per-cell histograms of `nbins` bins, then
/// normalizes each overlapping `block_size × block_size` block of cells
/// with an L2 norm.
///
/// - `cell_size`: side length (in pixels) of each cell.
/// - `block_size`: side length (in cells) of each normalization block.
/// - `nbins`: number of orientation bins per cell.
///
/// Returns a flat array of normalized histogram values.
pub fn hog(
  img : Image,
  cell_size : Int,
  block_size : Int,
  nbins : Int,
) -> Array[Double] {
  let h = img.h
  let w = img.w
  // Build a flat luma array directly. Since `to_grayscale` produces an image
  // whose R==G==B==luma, we read the R channel of the grayscale image to
  // avoid recomputing luma per gradient sample.
  let gray = to_grayscale(img)
  let luma_flat = Array::make(h * w, 0.0)
  let stride = w * 4
  for y = 0; y < h; y = y + 1 {
    let row_base = y * stride
    let out_base = y * w
    for x = 0; x < w; x = x + 1 {
      luma_flat[out_base + x] = gray.data[row_base + x * 4].to_double()
    }
  }
  // Compute gradients with inlined replicate border (clamp indices).
  let gx = Array::make(h * w, 0.0)
  let gy = Array::make(h * w, 0.0)
  let hmax = h - 1
  let wmax = w - 1
  for y = 0; y < h; y = y + 1 {
    let ybase = y * w
    let yp = (if y < hmax { y + 1 } else { hmax }) * w
    let ym = (if y > 0 { y - 1 } else { 0 }) * w
    for x = 0; x < w; x = x + 1 {
      let xp = if x < wmax { x + 1 } else { wmax }
      let xm = if x > 0 { x - 1 } else { 0 }
      gx[ybase + x] = luma_flat[ybase + xp] - luma_flat[ybase + xm]
      gy[ybase + x] = luma_flat[yp + x] - luma_flat[ym + x]
    }
  }
  // Cell histograms
  let n_cells_y = h / cell_size
  let n_cells_x = w / cell_size
  let cell_hists = Array::makei(n_cells_y * n_cells_x, fn(_i) {
    Array::make(nbins, 0.0)
  })
  let nbins_d = nbins.to_double()
  for cy = 0; cy < n_cells_y; cy = cy + 1 {
    for cx = 0; cx < n_cells_x; cx = cx + 1 {
      let hist = Array::make(nbins, 0.0)
      for dy = 0; dy < cell_size; dy = dy + 1 {
        let y = cy * cell_size + dy
        let ybase = y * w
        for dx = 0; dx < cell_size; dx = dx + 1 {
          let x = cx * cell_size + dx
          let idx = ybase + x
          let gxi = gx[idx]
          let gyi = gy[idx]
          let mag = (gxi * gxi + gyi * gyi).sqrt()
          let angle = @math.atan2(gyi, gxi) * 180.0 / @math.PI
          let mut a = if angle < 0.0 { angle + 180.0 } else { angle }
          a = a % 180.0
          let bin = clampi((a / 180.0 * nbins_d).to_int(), 0, nbins - 1)
          hist[bin] = hist[bin] + mag
        }
      }
      cell_hists[cy * n_cells_x + cx] = hist
    }
  }
  // Block normalization
  let result : Array[Double] = []
  for by = 0; by + block_size <= n_cells_y; by = by + 1 {
    for bx = 0; bx + block_size <= n_cells_x; bx = bx + 1 {
      let block : Array[Double] = []
      let mut norm = 0.0
      for dy = 0; dy < block_size; dy = dy + 1 {
        for dx = 0; dx < block_size; dx = dx + 1 {
          let hist = cell_hists[(by + dy) * n_cells_x + (bx + dx)]
          for b = 0; b < nbins; b = b + 1 {
            block.push(hist[b])
            norm = norm + hist[b] * hist[b]
          }
        }
      }
      norm = (norm + 0.0001).sqrt()
      for i = 0; i < block.length(); i = i + 1 {
        result.push(block[i] / norm)
      }
    }
  }
  result
}

///|
/// Local Binary Pattern (LBP) codes for each pixel.
///
/// For every pixel, samples `n_points` neighbors on a circle of the given
/// `radius` (using bilinear interpolation) and builds a bitmask of which
/// neighbors are greater than or equal to the center value.
///
/// - `radius`: circle radius in pixels.
/// - `n_points`: number of sampled neighbors (also the bitmask width).
///
/// Returns a flat `Array[Int]` of length `h * w`, row-major.
pub fn lbp(img : Image, radius : Int, n_points : Int) -> Array[Int] {
  let n = img.h * img.w
  let out = Array::make(n, 0)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let center = luma_at_mode(img, y, x, Reflect)
      let mut code = 0
      for p = 0; p < n_points; p = p + 1 {
        let angle = 2.0 * @math.PI * p.to_double() / n_points.to_double()
        let py = y.to_double() + radius.to_double() * @math.sin(angle)
        let px = x.to_double() + radius.to_double() * @math.cos(angle)
        let neighbor = bilinear_luma(img, py, px, Reflect)
        if neighbor >= center {
          code = code + (1 << p)
        }
      }
      out[y * img.w + x] = code
    }
  }
  out
}

///|
/// Histogram of Local Binary Pattern codes over the whole image.
///
/// - `radius`: circle radius in pixels, forwarded to `lbp`.
/// - `n_points`: number of sampled neighbors, forwarded to `lbp`.
///
/// Returns an `Array[Int]` of length `2 ^ n_points` containing per-bin
/// counts.
pub fn lbp_histogram(img : Image, radius : Int, n_points : Int) -> Array[Int] {
  let codes = lbp(img, radius, n_points)
  let n_bins = 1 << n_points
  let hist = Array::make(n_bins, 0)
  for i = 0; i < codes.length(); i = i + 1 {
    hist[codes[i]] = hist[codes[i]] + 1
  }
  hist
}
