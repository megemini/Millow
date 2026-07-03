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
        Nearest =>
          if sx < 0.0 ||
            sy < 0.0 ||
            sx >= img.w.to_double() ||
            sy >= img.h.to_double() {
            match mode {
              Constant(cr, cg, cb, ca) => (cr, cg, cb, ca)
              _ => img.pixel_at_unchecked(0)
            }
          } else {
            let si = clampi(sy.to_int(), 0, img.h - 1)
            let sj = clampi(sx.to_int(), 0, img.w - 1)
            img.pixel_at_unchecked(si * img.w + sj)
          }
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
/// Rotate by an arbitrary angle (degrees, clockwise, matching Pillow).
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
  let (t2_exp, t5_exp) = transform_pt(
    -(dst_w.to_double() - w) / 2.0,
    -(dst_h.to_double() - h) / 2.0,
    matrix2,
  )
  let matrix3 = [matrix2[0], matrix2[1], t2_exp, matrix2[3], matrix2[4], t5_exp]
  affine(img, matrix3, dst_h, dst_w, interp, Constant(0, 0, 0, 255))
}

///|
/// General affine transform. `matrix` is `[a, b, c, d, e, f]`:
///   `x' = a*x + b*y + c`
///   `y' = d*x + e*y + f`
/// `mode` specifies the border handling (default is Constant(0,0,0,255)).
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
/// Translate the image by `(dy, dx)` pixels (matching Pillow).
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
/// Shear the image. `sh_x` shears along x for each y; `sh_y` shears along y
/// for each x.
/// `mode` specifies the border handling (default is Constant(0,0,0,255)).
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
