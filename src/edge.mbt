///|
/// Apply a 3x3 kernel to the luminance field, returning the response map.
fn grad(
  img : Image,
  k : Array[Array[Double]],
  mode : BorderMode,
) -> Array[Array[Float]] {
  let out = Array::makei(img.h, fn(_i) { Array::make(img.w, (0.0 : Float)) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let mut acc = 0.0
      for ky = 0; ky < 3; ky = ky + 1 {
        for kx = 0; kx < 3; kx = kx + 1 {
          acc = acc +
            k[ky][kx] * luma_at_mode(img, y + ky - 1, x + kx - 1, mode)
        }
      }
      out[y][x] = Float::from_double(acc)
    }
  }
  out
}

///|
/// Apply a 3x3 kernel to a grayscale image's double values.
fn grad_double(
  img : Image,
  k : Array[Array[Double]],
  mode : BorderMode,
) -> Array[Array[Double]] {
  let out = Array::makei(img.h, fn(_i) { Array::make(img.w, 0.0) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let mut acc = 0.0
      for ky = 0; ky < 3; ky = ky + 1 {
        for kx = 0; kx < 3; kx = kx + 1 {
          acc = acc +
            k[ky][kx] * luma_at_mode(img, y + ky - 1, x + kx - 1, mode)
        }
      }
      out[y][x] = acc
    }
  }
  out
}

///|
/// Combine two gradient maps into a magnitude image (grayscale, opaque).
fn mag_image(gx : Array[Array[Float]], gy : Array[Array[Float]]) -> Image {
  let h = gx.length()
  let w = if h == 0 { 0 } else { gx[0].length() }
  let out = Image::new(h, w)
  let mut max_val = 0.0
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let dx = gx[y][x].to_double()
      let dy = gy[y][x].to_double()
      let mag = (dx * dx + dy * dy).sqrt()
      if mag > max_val {
        max_val = mag
      }
    }
  }
  let scale = if max_val > 0.0 { 255.0 / max_val } else { 1.0 }
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let dx = gx[y][x].to_double()
      let dy = gy[y][x].to_double()
      let vb = round_byte((dx * dx + dy * dy).sqrt() * scale)
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
  let mag = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  let dir = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let dx = gx[y][x].to_double()
      let dy = gy[y][x].to_double()
      mag[y][x] = (dx * dx + dy * dy).sqrt()
      dir[y][x] = @math.atan2(dy, dx) * 180.0 / @math.PI
    }
  }
  let nms = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 1; y < h - 1; y = y + 1 {
    for x = 1; x < w - 1; x = x + 1 {
      let mut angle = dir[y][x]
      if angle < 0.0 {
        angle = angle + 180.0
      }
      let (n1, n2) = if angle < 22.5 || angle >= 157.5 {
        (mag[y][x + 1], mag[y][x - 1])
      } else if angle < 67.5 {
        (mag[y - 1][x + 1], mag[y + 1][x - 1])
      } else if angle < 112.5 {
        (mag[y - 1][x], mag[y + 1][x])
      } else {
        (mag[y - 1][x - 1], mag[y + 1][x + 1])
      }
      nms[y][x] = if mag[y][x] >= n1 && mag[y][x] >= n2 {
        mag[y][x]
      } else {
        0.0
      }
    }
  }
  let strong = 255.0
  let weak = 75.0
  let edges = Array::makei(h, fn(_i) { Array::make(w, 0.0) })
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      if nms[y][x] >= high {
        edges[y][x] = strong
      } else if nms[y][x] >= low {
        edges[y][x] = weak
      }
    }
  }
  let mut changed = true
  while changed {
    changed = false
    for y = 1; y < h - 1; y = y + 1 {
      for x = 1; x < w - 1; x = x + 1 {
        if edges[y][x] == weak {
          let has_strong = edges[y - 1][x - 1] == strong ||
            edges[y - 1][x] == strong ||
            edges[y - 1][x + 1] == strong ||
            edges[y][x - 1] == strong ||
            edges[y][x + 1] == strong ||
            edges[y + 1][x - 1] == strong ||
            edges[y + 1][x] == strong ||
            edges[y + 1][x + 1] == strong
          if has_strong {
            edges[y][x] = strong
            changed = true
          }
        }
      }
    }
  }
  let out = Image::new(h, w)
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let v : Byte = if edges[y][x] == strong { 255 } else { 0 }
      let o = out.offset(y, x)
      out.data[o] = v
      out.data[o + 1] = v
      out.data[o + 2] = v
      out.data[o + 3] = 255
    }
  }
  out
}
