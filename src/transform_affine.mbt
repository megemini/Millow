///|
/// Bilinear sample at fractional coordinates `(fy, fx)` with specified border mode.
fn sample_bilinear_mode(
  img : Image,
  fy : Double,
  fx : Double,
  mode : BorderMode,
) -> (Byte, Byte, Byte, Byte) {
  let y0 = fy.floor().to_int()
  let x0 = fx.floor().to_int()
  let y1 = y0 + 1
  let x1 = x0 + 1
  let dy = fy - y0.to_double()
  let dx = fx - x0.to_double()
  let (r00, g00, b00, a00) = get_pixel(img, y0, x0, mode)
  let (r01, g01, b01, _a01) = get_pixel(img, y0, x1, mode)
  let (r10, g10, b10, _a10) = get_pixel(img, y1, x0, mode)
  let (r11, g11, b11, _a11) = get_pixel(img, y1, x1, mode)
  let mix = fn(v00 : Byte, v01 : Byte, v10 : Byte, v11 : Byte) -> Byte {
    round_byte(
      v00.to_double() * (1.0 - dy) * (1.0 - dx) +
      v01.to_double() * (1.0 - dy) * dx +
      v10.to_double() * dy * (1.0 - dx) +
      v11.to_double() * dy * dx,
    )
  }
  (
    mix(r00, r01, r10, r11),
    mix(g00, g01, g10, g11),
    mix(b00, b01, b10, b11),
    a00,
  )
}

///|
/// Apply a general affine transform. `matrix` is `[a, b, c, d, e, f]`:
///   `x' = a*x + b*y + c`
///   `y' = d*x + e*y + f`
/// The output size is `(dst_h, dst_w)`. For each output pixel, the inverse
/// mapping is computed and the source is sampled with the given interpolation.
fn affine(
  img : Image,
  matrix : Array[Double],
  dst_h : Int,
  dst_w : Int,
  interp : Interp,
  mode : BorderMode,
) -> Image {
  let a = matrix[0]
  let b = matrix[1]
  let c = matrix[2]
  let d = matrix[3]
  let e = matrix[4]
  let f = matrix[5]
  let det = a * e - b * d
  let out = Image::new(dst_h, dst_w)
  for y = 0; y < dst_h; y = y + 1 {
    for x = 0; x < dst_w; x = x + 1 {
      let xp = x.to_double()
      let yp = y.to_double()
      let sx = (e * xp - b * yp + b * f - e * c) / det
      let sy = (-d * xp + a * yp + d * c - a * f) / det
      let o = out.offset(y, x)
      let (r, g, bb, a_val) = match interp {
        Nearest => get_pixel(img, sy.to_int(), sx.to_int(), mode)
        _ => sample_bilinear_mode(img, sy, sx, mode)
      }
      out.data[o] = r
      out.data[o + 1] = g
      out.data[o + 2] = bb
      out.data[o + 3] = a_val
    }
  }
  out
}

///|
/// Rotate the image by an arbitrary angle in degrees, clockwise.
///
/// - `angle`: rotation angle in degrees (positive = clockwise).
/// - `interp`: interpolation method (`Nearest` or `Bilinear`).
///
/// The output canvas expands to fit the entire rotated image; areas outside
/// the source are filled as opaque black `(0, 0, 0, 255)`.
pub fn rotate_any(img : Image, angle : Double, interp : Interp) -> Image {
  let rad = -angle * @math.PI / 180.0
  let cos_a = @math.cos(rad)
  let sin_a = @math.sin(rad)
  let w = img.w.to_double()
  let h = img.h.to_double()
  let cx = w / 2.0
  let cy = h / 2.0
  let matrix = [cos_a, sin_a, 0.0, -sin_a, cos_a, 0.0]
  let transform_pt = fn(
    x : Double,
    y : Double,
    m : Array[Double],
  ) -> (Double, Double) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
  }
  let (t2, t5) = transform_pt(-cx, -cy, matrix)
  let matrix2 = [matrix[0], matrix[1], t2 + cx, matrix[3], matrix[4], t5 + cy]
  let corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
  let mut x_min = 0.0
  let mut x_max = 0.0
  let mut y_min = 0.0
  let mut y_max = 0.0
  for i = 0; i < 4; i = i + 1 {
    let (x, y) = corners[i]
    let (tx, ty) = transform_pt(x, y, matrix2)
    if i == 0 {
      x_min = tx
      x_max = tx
      y_min = ty
      y_max = ty
    } else {
      x_min = x_min.min(tx)
      x_max = x_max.max(tx)
      y_min = y_min.min(ty)
      y_max = y_max.max(ty)
    }
  }
  let dst_w = (x_max - x_min).ceil().to_int().max(1)
  let dst_h = (y_max - y_min).ceil().to_int().max(1)
  // Center the rotated image in the expanded canvas: the rotation center
  // (w/2, h/2) should map to the canvas center (dst_w/2, dst_h/2).
  let dx_expand = (dst_w.to_double() - w) / 2.0
  let dy_expand = (dst_h.to_double() - h) / 2.0
  let matrix3 = [
    matrix2[0],
    matrix2[1],
    matrix2[2] + dx_expand,
    matrix2[3],
    matrix2[4],
    matrix2[5] + dy_expand,
  ]
  affine(img, matrix3, dst_h, dst_w, interp, Constant(0, 0, 0, 255))
}

///|
/// Apply a general affine transform to the image.
///
/// `matrix` is `[a, b, c, d, e, f]` defining the forward mapping:
///
/// - `x' = a*x + b*y + c`
/// - `y' = d*x + e*y + f`
///
/// - `dst_h`, `dst_w`: output image dimensions.
/// - `interp`: interpolation method (`Nearest` or `Bilinear`).
/// - `mode`: border handling for samples outside the source
///   (default `Constant(0, 0, 0, 255)`).
///
/// Each output pixel is sampled from the inverse-mapped source location.
pub fn affine_transform(
  img : Image,
  matrix : Array[Double],
  dst_h : Int,
  dst_w : Int,
  interp : Interp,
  mode? : BorderMode = Constant(0, 0, 0, 255),
) -> Image {
  affine(img, matrix, dst_h, dst_w, interp, mode)
}

///|
/// Invert a 3×3 matrix (flat row-major `Array[Double]` of length 9).
fn invert_3x3(m : Array[Double]) -> Array[Double] {
  let det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  let inv_det = 1.0 / det
  [
    (m[4] * m[8] - m[5] * m[7]) * inv_det,
    (m[2] * m[7] - m[1] * m[8]) * inv_det,
    (m[1] * m[5] - m[2] * m[4]) * inv_det,
    (m[5] * m[6] - m[3] * m[8]) * inv_det,
    (m[0] * m[8] - m[2] * m[6]) * inv_det,
    (m[2] * m[3] - m[0] * m[5]) * inv_det,
    (m[3] * m[7] - m[4] * m[6]) * inv_det,
    (m[1] * m[6] - m[0] * m[7]) * inv_det,
    (m[0] * m[4] - m[1] * m[3]) * inv_det,
  ]
}

///|
/// Warp an image using a 3×3 homography matrix (perspective transform).
///
/// `matrix_3x3` is a flat `Array[Double]` of length 9:
/// ```
/// [a, b, c]     x' = (a*x + b*y + c) / (g*x + h*y + 1)
/// [d, e, f]     y' = (d*x + e*y + f) / (g*x + h*y + 1)
/// [g, h, 1]
/// ```
///
/// For each output pixel, the inverse homography is applied to find the
/// source coordinates, then sampled with the given interpolation.
///
/// - `dst_h`, `dst_w`: output image dimensions.
/// - `interp`: interpolation method.
/// - `mode`: border handling (default `Constant(0, 0, 0, 255)`).
pub fn perspective_transform(
  img : Image,
  matrix_3x3 : Array[Double],
  dst_h : Int,
  dst_w : Int,
  interp : Interp,
  mode? : BorderMode = Constant(0, 0, 0, 255),
) -> Image {
  let inv = invert_3x3(matrix_3x3)
  let out = Image::new(dst_h, dst_w)
  for y = 0; y < dst_h; y = y + 1 {
    for x = 0; x < dst_w; x = x + 1 {
      let xd = x.to_double()
      let yd = y.to_double()
      let w = inv[6] * xd + inv[7] * yd + inv[8]
      if w.abs() < 1.0e-10 {
        let o = out.offset(y, x)
        out.data[o + 3] = 255
        continue
      }
      let sx = (inv[0] * xd + inv[1] * yd + inv[2]) / w
      let sy = (inv[3] * xd + inv[4] * yd + inv[5]) / w
      let o = out.offset(y, x)
      let (r, g, bb, a_val) = match interp {
        Nearest => get_pixel(img, sy.to_int(), sx.to_int(), mode)
        _ => sample_bilinear_mode(img, sy, sx, mode)
      }
      out.data[o] = r
      out.data[o + 1] = g
      out.data[o + 2] = bb
      out.data[o + 3] = a_val
    }
  }
  out
}

///|
/// Convenience: warp from 4 source points to 4 destination points.
///
/// Solves the 8-parameter homography mapping `src_pts` → `dst_pts` and applies
/// `perspective_transform`. Both point arrays must have exactly 4 elements,
/// each a `(y, x)` tuple. Output dimensions default to the bounding box of
/// `dst_pts`.
pub fn perspective_4pt(
  img : Image,
  src_pts : Array[(Double, Double)],
  dst_pts : Array[(Double, Double)],
  interp : Interp,
  mode? : BorderMode = Constant(0, 0, 0, 255),
) -> Image {
  let h = solve_homography_4pt(src_pts, dst_pts)
  // Output dimensions: bounding box of dst_pts.
  let mut y_max = 0.0
  let mut x_max = 0.0
  for i = 0; i < 4; i = i + 1 {
    let (py, px) = dst_pts[i]
    if py > y_max {
      y_max = py
    }
    if px > x_max {
      x_max = px
    }
  }
  let dst_h = y_max.ceil().to_int().max(1)
  let dst_w = x_max.ceil().to_int().max(1)
  perspective_transform(img, h, dst_h, dst_w, interp, mode~)
}

///|
/// Solve homography from 4 correspondences using direct linear method.
/// Returns 3×3 matrix flat as `Array[Double]` length 9.
fn solve_homography_4pt(
  src : Array[(Double, Double)],
  dst : Array[(Double, Double)],
) -> Array[Double] {
  // Build 8×8 linear system Ah = b.
  // Each correspondence (x, y) → (xp, yp) contributes 2 rows:
  // [x, y, 1, 0, 0, 0, -xp*x, -xp*y] h = xp
  // [0, 0, 0, x, y, 1, -yp*x, -yp*y] h = yp
  // Note: src/dst are (y, x) tuples, so x=src_x, y=src_y.
  let a = Array::make(8, Array::make(8, 0.0))
  let b = Array::make(8, 0.0)
  for i = 0; i < 4; i = i + 1 {
    let (sy, sx) = src[i]
    let (dy, dx) = dst[i]
    let row0 = i * 2
    let row1 = i * 2 + 1
    a[row0][0] = sx
    a[row0][1] = sy
    a[row0][2] = 1.0
    a[row0][6] = -dx * sx
    a[row0][7] = -dx * sy
    b[row0] = dx
    a[row1][3] = sx
    a[row1][4] = sy
    a[row1][5] = 1.0
    a[row1][6] = -dy * sx
    a[row1][7] = -dy * sy
    b[row1] = dy
  }
  let h = gauss_elimination(a, b)
  [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]
}

///|
/// Gaussian elimination with partial pivoting for an n×n system.
fn gauss_elimination(
  a : Array[Array[Double]],
  b : Array[Double],
) -> Array[Double] {
  let n = b.length()
  // Forward elimination with partial pivoting.
  for k = 0; k < n - 1; k = k + 1 {
    // Find pivot.
    let mut max_row = k
    let mut max_val = a[k][k].abs()
    for i = k + 1; i < n; i = i + 1 {
      let v = a[i][k].abs()
      if v > max_val {
        max_val = v
        max_row = i
      }
    }
    // Swap rows.
    if max_row != k {
      let tmp = a[k]
      a[k] = a[max_row]
      a[max_row] = tmp
      let tmpb = b[k]
      b[k] = b[max_row]
      b[max_row] = tmpb
    }
    // Eliminate.
    for i = k + 1; i < n; i = i + 1 {
      let factor = a[i][k] / a[k][k]
      for j = k; j < n; j = j + 1 {
        a[i][j] = a[i][j] - factor * a[k][j]
      }
      b[i] = b[i] - factor * b[k]
    }
  }
  // Back substitution.
  let x = Array::make(n, 0.0)
  for i = n - 1; i >= 0; i = i - 1 {
    let mut sum = b[i]
    for j = i + 1; j < n; j = j + 1 {
      sum = sum - a[i][j] * x[j]
    }
    x[i] = sum / a[i][i]
  }
  x
}

///|
/// Translate the image by `(dy, dx)` pixels.
///
/// - `dy`: vertical shift in pixels (positive = downward).
/// - `dx`: horizontal shift in pixels (positive = rightward).
/// - `interp`: interpolation method (`Nearest` or `Bilinear`).
///
/// The output keeps the source dimensions; areas uncovered by the source
/// become opaque black `(0, 0, 0, 255)`.
pub fn translate(
  img : Image,
  dy : Double,
  dx : Double,
  interp : Interp,
) -> Image {
  let matrix = [1.0, 0.0, -dx, 0.0, 1.0, -dy]
  affine(img, matrix, img.h, img.w, interp, Constant(0, 0, 0, 255))
}

///|
/// Shear the image.
///
/// - `sh_x`: horizontal shear factor (x shifts by `sh_x * y`).
/// - `sh_y`: vertical shear factor (y shifts by `sh_y * x`).
/// - `interp`: interpolation method (`Nearest` or `Bilinear`).
/// - `mode`: border handling for samples outside the source
///   (default `Constant(0, 0, 0, 255)`).
///
/// The output canvas expands to hold the sheared image.
pub fn shear(
  img : Image,
  sh_x : Double,
  sh_y : Double,
  interp : Interp,
  mode? : BorderMode = Constant(0, 0, 0, 255),
) -> Image {
  let det = 1.0 - sh_x * sh_y
  let matrix = [1.0 / det, -sh_x / det, 0.0, -sh_y / det, 1.0 / det, 0.0]
  let h = img.h.to_double()
  let w = img.w.to_double()
  let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
  let mut x_min = 0.0
  let mut x_max = 0.0
  let mut y_min = 0.0
  let mut y_max = 0.0
  for i = 0; i < 4; i = i + 1 {
    let (x, y) = corners[i]
    let rx = x + sh_x * y
    let ry = y + sh_y * x
    if i == 0 {
      x_min = rx
      x_max = rx
      y_min = ry
      y_max = ry
    } else {
      x_min = x_min.min(rx)
      x_max = x_max.max(rx)
      y_min = y_min.min(ry)
      y_max = y_max.max(ry)
    }
  }
  let dst_w = (x_max - x_min).ceil().to_int().max(1)
  let dst_h = (y_max - y_min).ceil().to_int().max(1)
  affine(img, matrix, dst_h, dst_w, interp, mode)
}
