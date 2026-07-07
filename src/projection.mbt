///|
/// Horizontal projection: sum of luma per row.
///
/// Returns an `Array[Int]` of length `h` where `out[y]` is the sum of BT.601
/// luma values across row `y`. Useful for skew detection and text-line
/// analysis on grayscale or color images.
pub fn project_horizontal(img : Image) -> Array[Int] {
  let h = img.h
  let w = img.w
  let out = Array::make(h, 0)
  for y = 0; y < h; y = y + 1 {
    let mut sum = 0
    let row = y * w * 4
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      sum = sum + luma(img.data[o], img.data[o + 1], img.data[o + 2])
    }
    out[y] = sum
  }
  out
}

///|
/// Vertical projection: sum of luma per column.
///
/// Returns an `Array[Int]` of length `w` where `out[x]` is the sum of BT.601
/// luma values down column `x`. Useful for character segmentation and column
/// detection.
pub fn project_vertical(img : Image) -> Array[Int] {
  let h = img.h
  let w = img.w
  let out = Array::make(w, 0)
  for y = 0; y < h; y = y + 1 {
    let row = y * w * 4
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      out[x] = out[x] + luma(img.data[o], img.data[o + 1], img.data[o + 2])
    }
  }
  out
}

///|
/// Binary horizontal projection: count of non-zero (foreground) pixels per row.
///
/// A pixel is foreground when any RGB channel is non-zero. Returns an
/// `Array[Int]` of length `h`. Designed for binary images; for grayscale use
/// `project_horizontal` instead.
pub fn project_horizontal_binary(img : Image) -> Array[Int] {
  let h = img.h
  let w = img.w
  let out = Array::make(h, 0)
  for y = 0; y < h; y = y + 1 {
    let mut count = 0
    let row = y * w * 4
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      if img.data[o] != 0 || img.data[o + 1] != 0 || img.data[o + 2] != 0 {
        count = count + 1
      }
    }
    out[y] = count
  }
  out
}

///|
/// Binary vertical projection: count of non-zero (foreground) pixels per column.
///
/// A pixel is foreground when any RGB channel is non-zero. Returns an
/// `Array[Int]` of length `w`. Designed for binary images; for grayscale use
/// `project_vertical` instead.
pub fn project_vertical_binary(img : Image) -> Array[Int] {
  let h = img.h
  let w = img.w
  let out = Array::make(w, 0)
  for y = 0; y < h; y = y + 1 {
    let row = y * w * 4
    for x = 0; x < w; x = x + 1 {
      let o = row + x * 4
      if img.data[o] != 0 || img.data[o + 1] != 0 || img.data[o + 2] != 0 {
        out[x] = out[x] + 1
      }
    }
  }
  out
}
