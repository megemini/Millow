///|
/// Integral image (summed-area table) from an image, operating on luma values.
///
/// Returns a flat `Array[Double]` of size `(h+1) * (w+1)` where
/// `integ[(y+1)*(w+1) + (x+1)]` = sum of luma values in the rectangle
/// `[0..y, 0..x]`. The first row and column are zeros.
///
/// Uses `Double` to avoid overflow on large images.
pub fn integral_image(img : Image) -> Array[Double] {
  let h = img.h
  let w = img.w
  let stride = w * 4
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let o = gray_row + x * 4
      let v = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
      row_sum = row_sum + v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
    }
  }
  integ
}

///|
/// Integral image from a raw byte array (e.g. a luma channel).
///
/// `data` is a flat `Array[Byte]` of length `w * h`, row-major.
/// Returns a flat `Array[Double]` of size `(h+1) * (w+1)`.
pub fn integral_array(data : Array[Byte], w : Int, h : Int) -> Array[Double] {
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    for x = 0; x < w; x = x + 1 {
      row_sum = row_sum + data[y * w + x].to_double()
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
    }
  }
  integ
}

///|
/// Square integral image (sum of squared luma values) from an image.
///
/// Returns a flat `Array[Double]` of size `(h+1) * (w+1)`.
/// Used for computing local variance via `query_integral`.
pub fn integral_image_sq(img : Image) -> Array[Double] {
  let h = img.h
  let w = img.w
  let stride = w * 4
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    let gray_row = y * stride
    for x = 0; x < w; x = x + 1 {
      let o = gray_row + x * 4
      let v = luma(img.data[o], img.data[o + 1], img.data[o + 2]).to_double()
      let v2 = v * v
      row_sum = row_sum + v2
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
    }
  }
  integ
}

///|
/// Square integral image from a raw byte array.
///
/// `data` is a flat `Array[Byte]` of length `w * h`, row-major.
/// Returns a flat `Array[Double]` of size `(h+1) * (w+1)`.
pub fn integral_array_sq(data : Array[Byte], w : Int, h : Int) -> Array[Double] {
  let w1 = w + 1
  let integ = Array::make((h + 1) * w1, 0.0)
  for y = 0; y < h; y = y + 1 {
    let mut row_sum = 0.0
    for x = 0; x < w; x = x + 1 {
      let v = data[y * w + x].to_double()
      row_sum = row_sum + v * v
      integ[(y + 1) * w1 + (x + 1)] = integ[y * w1 + (x + 1)] + row_sum
    }
  }
  integ
}

///|
/// Query a rectangular sum from an integral image.
///
/// - `integ_w`: stride of the integral image (= image width + 1).
/// - `y0`, `x0`: top-left corner (inclusive, clamped to image bounds).
/// - `y1`, `x1`: bottom-right corner (inclusive, clamped to image bounds).
///
/// Returns the sum of all values in the rectangle `[y0..y1, x0..x1]`.
pub fn query_integral(
  integ : Array[Double],
  integ_w : Int,
  y0 : Int,
  x0 : Int,
  y1 : Int,
  x1 : Int,
) -> Double {
  integ[(y1 + 1) * integ_w + (x1 + 1)] -
  integ[y0 * integ_w + (x1 + 1)] -
  integ[(y1 + 1) * integ_w + x0] +
  integ[y0 * integ_w + x0]
}
