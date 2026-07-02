///|
/// Luminance of the clamped pixel at `(y, x)` as a double.
fn luma_at(img : Image, y : Int, x : Int) -> Double {
  let cy = clampi(y, 0, img.h - 1)
  let cx = clampi(x, 0, img.w - 1)
  let o = img.offset(cy, cx)
  luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
}

///|
/// Apply a 3x3 kernel to the luminance field, returning the response map.
fn grad(img : Image, k : Array[Array[Double]]) -> Array[Array[Float]] {
  let out = Array::makei(img.h, fn(_i) { Array::make(img.w, (0.0 : Float)) })
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let mut acc = 0.0
      for ky = 0; ky < 3; ky = ky + 1 {
        for kx = 0; kx < 3; kx = kx + 1 {
          acc = acc + k[ky][kx] * luma_at(img, y + ky - 1, x + kx - 1)
        }
      }
      out[y][x] = Float::from_double(acc)
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
  for y = 0; y < h; y = y + 1 {
    for x = 0; x < w; x = x + 1 {
      let dx = gx[y][x].to_double()
      let dy = gy[y][x].to_double()
      let vb = round_byte((dx * dx + dy * dy).sqrt())
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
/// Sobel horizontal gradient (per-pixel response).
pub fn sobel_x(img : Image) -> Array[Array[Float]] {
  grad(img, [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
}

///|
/// Sobel vertical gradient (per-pixel response).
pub fn sobel_y(img : Image) -> Array[Array[Float]] {
  grad(img, [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
}

///|
/// Sobel edge magnitude (grayscale image).
pub fn sobel(img : Image) -> Image {
  mag_image(sobel_x(img), sobel_y(img))
}

///|
/// Prewitt edge magnitude (grayscale image).
pub fn prewitt(img : Image) -> Image {
  let gx = grad(img, [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
  let gy = grad(img, [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
  mag_image(gx, gy)
}

///|
/// Scharr edge magnitude (grayscale image).
pub fn scharr(img : Image) -> Image {
  let gx = grad(img, [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
  let gy = grad(img, [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
  mag_image(gx, gy)
}

///|
/// Laplacian edge response (absolute value, grayscale image).
pub fn laplacian(img : Image) -> Image {
  let g = grad(img, [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
  let out = Image::new(img.h, img.w)
  for y = 0; y < img.h; y = y + 1 {
    for x = 0; x < img.w; x = x + 1 {
      let v = g[y][x].to_double()
      let vb = round_byte(if v < 0.0 { -v } else { v })
      let o = out.offset(y, x)
      out.data[o] = vb
      out.data[o + 1] = vb
      out.data[o + 2] = vb
      out.data[o + 3] = 255
    }
  }
  out
}
