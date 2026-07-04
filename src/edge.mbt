///|
/// Compute luma from a flat byte array at byte offset `o`.
fn luma_at_idx(data : Array[Byte], o : Int) -> Double {
  ((
    data[o].to_int() * 77 +
    data[o + 1].to_int() * 150 +
    data[o + 2].to_int() * 29
  ) >>
  8).to_double()
}

///|
/// Apply a 3x3 kernel to the luminance field, returning the response map.
/// Inlined border handling for Reflect mode (used by all edge detectors).
fn grad(
  img : Image,
  k : Array[Array[Double]],
  mode : BorderMode,
) -> Array[Array[Float]] {
  let h = img.h
  let w = img.w
  let stride = w * 4
  let out = Array::makei(h, fn(_i) { Array::make(w, (0.0 : Float)) })
  let k00 = k[0][0]
  let k01 = k[0][1]
  let k02 = k[0][2]
  let k10 = k[1][0]
  let k11 = k[1][1]
  let k12 = k[1][2]
  let k20 = k[2][0]
  let k21 = k[2][1]
  let k22 = k[2][2]
  let hmax = h - 1
  let wmax = w - 1
  match mode {
    Reflect =>
      for y = 0; y < h; y = y + 1 {
        let ym = reflect_coord(y - 1, hmax)
        let yp = reflect_coord(y + 1, hmax)
        let rym = ym * stride
        let ryc = y * stride
        let ryp = yp * stride
        let out_row = out[y]
        for x = 0; x < w; x = x + 1 {
          let xm = reflect_coord(x - 1, wmax)
          let xp = reflect_coord(x + 1, wmax)
          let bxm = xm * 4
          let bxc = x * 4
          let bxp = xp * 4
          let v00 = luma_at_idx(img.data, rym + bxm)
          let v01 = luma_at_idx(img.data, rym + bxc)
          let v02 = luma_at_idx(img.data, rym + bxp)
          let v10 = luma_at_idx(img.data, ryc + bxm)
          let v11 = luma_at_idx(img.data, ryc + bxc)
          let v12 = luma_at_idx(img.data, ryc + bxp)
          let v20 = luma_at_idx(img.data, ryp + bxm)
          let v21 = luma_at_idx(img.data, ryp + bxc)
          let v22 = luma_at_idx(img.data, ryp + bxp)
          let acc = k00 * v00 +
            k01 * v01 +
            k02 * v02 +
            k10 * v10 +
            k11 * v11 +
            k12 * v12 +
            k20 * v20 +
            k21 * v21 +
            k22 * v22
          out_row[x] = Float::from_double(acc)
        }
      }
    _ =>
      for y = 0; y < h; y = y + 1 {
        let out_row = out[y]
        for x = 0; x < w; x = x + 1 {
          let mut acc = 0.0
          for ky = 0; ky < 3; ky = ky + 1 {
            for kx = 0; kx < 3; kx = kx + 1 {
              acc = acc +
                k[ky][kx] * luma_at_mode(img, y + ky - 1, x + kx - 1, mode)
            }
          }
          out_row[x] = Float::from_double(acc)
        }
      }
  }
  out
}

///|
/// Apply a 3x3 kernel to a grayscale image's double values.
/// Inlined for Constant(0,0,0,0) mode (used by corner detectors).
fn grad_double(
  img : Image,
  k : Array[Array[Double]],
  mode : BorderMode,
) -> Array[Array[Double]] {
  let h = img.h
  let w = img.w
  let stride = w * 4
  let out = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  let k00 = k[0][0]
  let k01 = k[0][1]
  let k02 = k[0][2]
  let k10 = k[1][0]
  let k11 = k[1][1]
  let k12 = k[1][2]
  let k20 = k[2][0]
  let k21 = k[2][1]
  let k22 = k[2][2]
  let hmax = h - 1
  let wmax = w - 1
  match mode {
    Constant(0, 0, 0, _) =>
      // Out-of-bounds = luma 0, so skip those terms
      for y = 0; y < h; y = y + 1 {
        let out_row = out[y]
        for x = 0; x < w; x = x + 1 {
          let mut acc = 0.0
          // Top row
          if y > 0 {
            let ry = (y - 1) * stride
            if x > 0 {
              acc = acc + k00 * luma_at_idx(img.data, ry + (x - 1) * 4)
            }
            acc = acc + k01 * luma_at_idx(img.data, ry + x * 4)
            if x < wmax {
              acc = acc + k02 * luma_at_idx(img.data, ry + (x + 1) * 4)
            }
          }
          // Middle row
          let ry = y * stride
          if x > 0 {
            acc = acc + k10 * luma_at_idx(img.data, ry + (x - 1) * 4)
          }
          acc = acc + k11 * luma_at_idx(img.data, ry + x * 4)
          if x < wmax {
            acc = acc + k12 * luma_at_idx(img.data, ry + (x + 1) * 4)
          }
          // Bottom row
          if y < hmax {
            let ry = (y + 1) * stride
            if x > 0 {
              acc = acc + k20 * luma_at_idx(img.data, ry + (x - 1) * 4)
            }
            acc = acc + k21 * luma_at_idx(img.data, ry + x * 4)
            if x < wmax {
              acc = acc + k22 * luma_at_idx(img.data, ry + (x + 1) * 4)
            }
          }
          out_row[x] = acc
        }
      }
    _ =>
      for y = 0; y < h; y = y + 1 {
        let out_row = out[y]
        for x = 0; x < w; x = x + 1 {
          let mut acc = 0.0
          for ky = 0; ky < 3; ky = ky + 1 {
            for kx = 0; kx < 3; kx = kx + 1 {
              acc = acc +
                k[ky][kx] * luma_at_mode(img, y + ky - 1, x + kx - 1, mode)
            }
          }
          out_row[x] = acc
        }
      }
  }
  out
}

///|
/// Combine two gradient maps into a magnitude image (grayscale, opaque).
/// Caches magnitudes to avoid computing sqrt twice per pixel.
fn mag_image(gx : Array[Array[Float]], gy : Array[Array[Float]]) -> Image {
  let h = gx.length()
  let w = if h == 0 { 0 } else { gx[0].length() }
  let out = Image::new(h, w)
  let stride = w * 4
  // Compute all magnitudes in one pass, tracking max
  let mags = Array::make(h * w, 0.0)
  let mut max_val = 0.0
  for y = 0; y < h; y = y + 1 {
    let gx_row = gx[y]
    let gy_row = gy[y]
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      let dx = gx_row[x].to_double()
      let dy = gy_row[x].to_double()
      let mag = (dx * dx + dy * dy).sqrt()
      mags[row_base + x] = mag
      if mag > max_val {
        max_val = mag
      }
    }
  }
  let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 }
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let vb = round_byte(mags[row_base + x] * scale)
      let oo = out_base + x * 4
      out.data[oo] = vb
      out.data[oo + 1] = vb
      out.data[oo + 2] = vb
      out.data[oo + 3] = 255
    }
  }
  out
}

///|
/// Compute the Sobel horizontal (x-direction) gradient response.
///
/// Convolves the image's luma field with the 3×3 Sobel-x kernel using
/// `Reflect` border handling.
///
/// Returns an `h × w` array of per-pixel gradient values as `Float`.
pub fn sobel_x(img : Image) -> Array[Array[Float]] {
  grad(img, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], Reflect)
}

///|
/// Compute the Sobel vertical (y-direction) gradient response.
///
/// Convolves the image's luma field with the 3×3 Sobel-y kernel using
/// `Reflect` border handling.
///
/// Returns an `h × w` array of per-pixel gradient values as `Float`.
pub fn sobel_y(img : Image) -> Array[Array[Float]] {
  grad(img, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], Reflect)
}

///|
/// Sobel edge magnitude as a grayscale image.
///
/// Combines the x and y Sobel gradients into `sqrt(gx² + gy²)` per pixel,
/// then scales the result to fill the 0–255 range. The alpha channel is set
/// to 255.
pub fn sobel(img : Image) -> Image {
  mag_image(sobel_x(img), sobel_y(img))
}

///|
/// Prewitt edge magnitude as a grayscale image.
///
/// Combines the x and y Prewitt gradients into `sqrt(gx² + gy²)` per pixel,
/// then scales the result to fill the 0–255 range. The alpha channel is set
/// to 255.
pub fn prewitt(img : Image) -> Image {
  let gx = grad(
    img,
    [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
    Reflect,
  )
  let gy = grad(
    img,
    [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    Reflect,
  )
  mag_image(gx, gy)
}

///|
/// Scharr edge magnitude as a grayscale image.
///
/// Combines the x and y Scharr gradients into `sqrt(gx² + gy²)` per pixel,
/// then scales the result to fill the 0–255 range. The Scharr kernel is
/// more rotationally symmetric than Sobel, giving more accurate gradients.
/// The alpha channel is set to 255.
pub fn scharr(img : Image) -> Image {
  let gx = grad(
    img,
    [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
    Reflect,
  )
  let gy = grad(
    img,
    [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
    Reflect,
  )
  mag_image(gx, gy)
}

///|
/// Laplacian edge response as a grayscale image.
///
/// Applies the 3×3 Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]` to the
/// image's luma field, takes the absolute value of the response, and scales
/// it to fill the 0–255 range. The alpha channel is set to 255.
pub fn laplacian(img : Image) -> Image {
  let g = grad(
    img,
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    Reflect,
  )
  let out = Image::new(img.h, img.w)
  let mut max_val = 0.0
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let v = g[y][x].to_double().abs()
      if v > max_val {
        max_val = v
      }
    }
  }
  let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 }
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let v = g[y][x].to_double().abs() * scale
      let vb = round_byte(v)
      let o = out.offset(y, x)
      out.data[o] = vb
      out.data[o + 1] = vb
      out.data[o + 2] = vb
      out.data[o + 3] = 255
    }
  }
  out
}

///|
/// Canny edge detector producing a binary edge image.
///
/// The pipeline smooths the image with a Gaussian blur, computes Sobel
/// gradients, applies non-maximum suppression along the gradient direction,
/// and uses dual-threshold hysteresis to trace the final edges.
///
/// - `low`: lower hysteresis threshold (0–255). Responses below this are
///   discarded.
/// - `high`: upper hysteresis threshold (0–255). Responses at or above this
///   become strong edges; responses between `low` and `high` are kept only
///   if connected to a strong edge.
///
/// Returns a grayscale image where edge pixels are 255 and all others are 0.
pub fn canny(img : Image, low : Double, high : Double) -> Image {
  let smoothed = gaussian_blur(img, 1.4)
  let gx = grad(
    smoothed,
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
    Reflect,
  )
  let gy = grad(
    smoothed,
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
    Reflect,
  )
  let h = img.h
  let w = img.w
  let n = h * w
  // Flat arrays for better cache locality
  let mag = Array::make(n, 0.0)
  let dir = Array::make(n, 0.0)
  for y = 0; y < h; y = y + 1 {
    let gx_row = gx[y]
    let gy_row = gy[y]
    let row_base = y * w
    for x = 0; x < w; x = x + 1 {
      let dx = gx_row[x].to_double()
      let dy = gy_row[x].to_double()
      mag[row_base + x] = (dx * dx + dy * dy).sqrt()
      dir[row_base + x] = @math.atan2(dy, dx) * 180.0 / @math.PI
    }
  }
  // Non-maximum suppression (flat array)
  let nms = Array::make(n, 0.0)
  for y = 1; y < h - 1; y = y + 1 {
    let row_base = y * w
    for x = 1; x < w - 1; x = x + 1 {
      let idx = row_base + x
      let mut angle = dir[idx]
      if angle < 0.0 {
        angle = angle + 180.0
      }
      let (n1, n2) = if angle < 22.5 || angle >= 157.5 {
        (mag[idx + 1], mag[idx - 1])
      } else if angle < 67.5 {
        (mag[idx + w + 1], mag[idx - w - 1])
      } else if angle < 112.5 {
        (mag[idx + w], mag[idx - w])
      } else {
        (mag[idx + w - 1], mag[idx - w + 1])
      }
      nms[idx] = if mag[idx] >= n1 && mag[idx] >= n2 { mag[idx] } else { 0.0 }
    }
  }
  // Dual threshold + stack-based hysteresis (O(n) instead of O(n*iterations))
  let strong = 255.0
  let weak = 75.0
  let edges = Array::make(n, 0.0)
  let stack : Array[Int] = []
  for i = 0; i < n; i = i + 1 {
    if nms[i] >= high {
      edges[i] = strong
      stack.push(i)
    } else if nms[i] >= low {
      edges[i] = weak
    }
  }
  // Flood fill from strong edges: promote connected weak pixels
  let mut si = 0
  while si < stack.length() {
    let idx = stack[si]
    si = si + 1
    let y = idx / w
    let x = idx - y * w
    for dy = -1; dy <= 1; dy = dy + 1 {
      let ny = y + dy
      if ny < 0 || ny >= h {
        continue
      }
      for dx = -1; dx <= 1; dx = dx + 1 {
        if dy == 0 && dx == 0 {
          continue
        }
        let nx = x + dx
        if nx < 0 || nx >= w {
          continue
        }
        let nidx = ny * w + nx
        if edges[nidx] == weak {
          edges[nidx] = strong
          stack.push(nidx)
        }
      }
    }
  }
  let out = Image::new(h, w)
  let stride = w * 4
  for y = 0; y < h; y = y + 1 {
    let row_base = y * w
    let out_base = y * stride
    for x = 0; x < w; x = x + 1 {
      let v : Byte = if edges[row_base + x] == strong { 255 } else { 0 }
      let oo = out_base + x * 4
      out.data[oo] = v
      out.data[oo + 1] = v
      out.data[oo + 2] = v
      out.data[oo + 3] = 255
    }
  }
  out
}
