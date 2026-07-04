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
  let gx2 = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let gy2 = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let gxy = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      gx2[y][x] = gx[y][x] * gx[y][x]
      gy2[y][x] = gy[y][x] * gy[y][x]
      gxy[y][x] = gx[y][x] * gy[y][x]
    }
  }
  let sxx = gaussian_blur_double(gx2, 1.0)
  let syy = gaussian_blur_double(gy2, 1.0)
  let sxy = gaussian_blur_double(gxy, 1.0)
  let out = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let det = sxx[y][x] * syy[y][x] - sxy[y][x] * sxy[y][x]
      let trace = sxx[y][x] + syy[y][x]
      out[y][x] = det - k * trace * trace
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
  let gx2 = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let gy2 = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let gxy = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      gx2[y][x] = gx[y][x] * gx[y][x]
      gy2[y][x] = gy[y][x] * gy[y][x]
      gxy[y][x] = gx[y][x] * gy[y][x]
    }
  }
  let sxx = gaussian_blur_double(gx2, 1.0)
  let syy = gaussian_blur_double(gy2, 1.0)
  let sxy = gaussian_blur_double(gxy, 1.0)
  let out = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let trace = sxx[y][x] + syy[y][x]
      let det = sxx[y][x] * syy[y][x] - sxy[y][x] * sxy[y][x]
      let disc = (trace * trace - 4.0 * det).max(0.0).sqrt()
      let l1 = (trace + disc) / 2.0
      let l2 = (trace - disc) / 2.0
      out[y][x] = l1.min(l2)
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
  let gray = to_grayscale(img)
  // Compute gradients
  let gx = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  let gy = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      gx[y][x] = luma_at(gray, y, x + 1) - luma_at(gray, y, x - 1)
      gy[y][x] = luma_at(gray, y + 1, x) - luma_at(gray, y - 1, x)
    }
  }
  // Cell histograms
  let n_cells_y = img.h / cell_size
  let n_cells_x = img.w / cell_size
  let cell_hists = Array::makei(n_cells_y * n_cells_x, fn(_i) {
    Array::make(nbins, 0.0)
  })
  for cy = 0; cy < n_cells_y; cy = cy + 1 {
    for cx = 0; cx < n_cells_x; cx = cx + 1 {
      let hist = Array::make(nbins, 0.0)
      for dy = 0; dy < cell_size; dy = dy + 1 {
        for dx = 0; dx < cell_size; dx = dx + 1 {
          let y = cy * cell_size + dy
          let x = cx * cell_size + dx
          let mag = (gx[y][x] * gx[y][x] + gy[y][x] * gy[y][x]).sqrt()
          let angle = @math.atan2(gy[y][x], gx[y][x]) * 180.0 / @math.PI
          let mut a = if angle < 0.0 { angle + 180.0 } else { angle }
          a = a % 180.0
          let bin = clampi(
            (a / 180.0 * nbins.to_double()).to_int(),
            0,
            nbins - 1,
          )
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
